#  ********************************************************************************
#    _____  ____ _
#   / _ \ \/ / _` |  Framework for control
#  |  __/>  < (_| |  and measurement of
#   \___/_/\_\__,_|  superconducting qubits
#
#  Copyright 2019-2026 IQM Finland Oy
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ********************************************************************************
"""Unified execution and storage runtime for the EXA core.

This module provides the :class:`ExaRuntime`, a process-wide singleton that acts
as the central access point for quantum job execution and data persistence.
By utilizing the standardized :class:`ExecutorInterface` and :class:`StorageInterface`,
it decouples high-level experiment logic from specific backend implementations
(e.g., IQM Server, local simulators, or mock hardware).

The runtime is designed to be initialized once during the application startup
using a concrete provider helper, after which the global executor and storage
instances can be accessed throughout the process.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import functools
import importlib.metadata
import logging
import platform
from threading import Lock
from typing import Any, cast

from iqm.iqm_server_client.iqm_server_client import IQMServerClient
from iqm.iqm_server_client.iqm_server_executor import IQMServerExecutor
from iqm.iqm_server_client.iqm_server_storage import IQMServerStorage

from exa.common.errors import ConflictError, ValidationError
from exa.common.helpers import load_yaml
import exa.common.logger
from exa.core import environment
from exa.core.util.serialization import fqn_to_object, fully_qual_name
from iqm.cpc.core.config import ExperimentConfiguration
from iqm.station_control.client.station_control_client import StationControlClient
from iqm.station_control.client.station_control_executor import StationControlExecutor
from iqm.station_control.client.station_control_storage import StationControlStorage
from iqm.station_control.interface.executor_interface import ExecutorInterface
from iqm.station_control.interface.serializable import Serializable
from iqm.station_control.interface.storage_interface import StorageInterface

_CTX_LOCK = Lock()
"""Thread-safety lock protecting concurrent mutations to the global runtime stack."""

_CTX_STACK: list[ExaRuntime] = []
"""Thread-local global stack tracking active ExaRuntime frames for nesting scopes."""

logger = logging.getLogger(__name__)


_SOURCE_FROM_SC = environment._SOURCE_FROM_SC


def _close_executor(executor: ExecutorInterface) -> None:
    """Best-effort close of an outgoing executor; logs and swallows any error.

    An outgoing executor's cleanup must never prevent installing the next runtime.
    """
    try:
        executor.close()
    except Exception:
        logger.exception("Error while closing outgoing executor %s.", type(executor).__name__)


def _close_storage(storage: StorageInterface) -> None:
    """Best-effort close of an outgoing storage; logs and swallows any error.

    An outgoing storage's cleanup must never prevent installing the next runtime.
    """
    try:
        storage.close()
    except Exception:
        logger.exception("Error while closing outgoing storage %s.", type(storage).__name__)


class ExaRuntime(Serializable):
    """Process-wide singleton providing quantum execution and storage interfaces."""

    def __init__(
        self,
        *,
        executor: ExecutorInterface,
        storage: StorageInterface,
        experiment_configuration: ExperimentConfiguration | None = None,
    ) -> None:
        self._executor = executor
        self._storage = storage
        self._experiment_configuration = experiment_configuration or ExperimentConfiguration({})
        self._software_version_set_id: int | None = None

    def serialize(self) -> dict[str, Any]:
        return {
            "executor": self._executor.serialize(),
            "storage": self._storage.serialize(),
            "experiment_configuration": self._experiment_configuration.serialize(),
            "executor_fqn": fully_qual_name(self._executor),
            "storage_fqn": fully_qual_name(self._storage),
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> ExaRuntime:
        executor: ExecutorInterface = fqn_to_object(data["executor_fqn"]).deserialize(data["executor"])
        storage: StorageInterface = fqn_to_object(data["storage_fqn"]).deserialize(data["storage"])
        experiment_configuration = ExperimentConfiguration.deserialize(data["experiment_configuration"])

        runtime = cls(
            executor=executor,
            storage=storage,
            experiment_configuration=experiment_configuration,
        )

        experiment_configuration.add_conventional_controller_mapping(executor)
        experiment_configuration.validate_controllers(executor)
        return runtime

    @staticmethod
    def get() -> ExaRuntime:
        """Return the current active runtime instance."""
        return ExaRuntime._resolve()

    @staticmethod
    def executor() -> ExecutorInterface:
        """Standardized interface for job execution from the active runtime."""
        return ExaRuntime._resolve()._executor

    @staticmethod
    def storage() -> StorageInterface:
        """Standardized interface for data persistence from the active runtime."""
        return ExaRuntime._resolve()._storage

    @staticmethod
    def experiment_configuration() -> ExperimentConfiguration:
        """Access the experiment configuration instance for this runtime."""
        return ExaRuntime._resolve()._experiment_configuration

    @staticmethod
    def get_or_create_software_version_set() -> int:
        """Get the database ID of the package version information of the current Python runtime."""
        ctx = ExaRuntime._resolve()

        with _CTX_LOCK:
            if ctx._software_version_set_id is None:
                versions = {dist.name: dist.version for dist in importlib.metadata.distributions()}
                versions["python"] = platform.python_version()
                if None in versions:
                    del versions[None]  # type: ignore[arg-type]

                ctx._software_version_set_id = ctx._executor.get_or_create_software_version_set(versions)

                logger.debug(
                    "Software version set ID %s registered, content: %s", ctx._software_version_set_id, versions
                )

            return ctx._software_version_set_id

    @staticmethod
    def _resolve() -> ExaRuntime:
        """Internal helper to find the innermost active runtime."""
        with _CTX_LOCK:
            if _CTX_STACK:
                return _CTX_STACK[-1]

        raise ConflictError("ExaRuntime has not been initialized.")

    @staticmethod
    def set(ctx: ExaRuntime) -> None:
        """Set the root-level runtime (index 0).

        If a root-level runtime is already set, its executor and storage are closed (see
        :meth:`ExecutorInterface.close` and :meth:`StorageInterface.close`) before being replaced,
        so they can release any resources they own (e.g. in-process background threads, pooled HTTP
        connections) before the new runtime takes over.
        """
        with _CTX_LOCK:
            if len(_CTX_STACK) > 1:
                raise ConflictError("Cannot set the root-level runtime while a scope is active.")

            if not _CTX_STACK:
                _CTX_STACK.append(ctx)
            else:
                _close_executor(_CTX_STACK[0]._executor)
                _close_storage(_CTX_STACK[0]._storage)
                _CTX_STACK[0] = ctx

    @staticmethod
    @contextmanager
    def scoped(ctx: ExaRuntime) -> Iterator[ExaRuntime]:
        """Context manager to temporarily override the runtime."""
        with _CTX_LOCK:
            _CTX_STACK.append(ctx)
        try:
            yield ctx
        finally:
            with _CTX_LOCK:
                # We always pop the top, but ensure we don't pop the root unless explicitly cleared
                if len(_CTX_STACK) > 1:
                    _CTX_STACK.pop()

    @staticmethod
    def clear() -> None:
        """Forcefully clear the entire runtime stack, including active scopes.

        Closes (see :meth:`ExecutorInterface.close` and :meth:`StorageInterface.close`) every
        executor and storage in the stack before discarding it, so any resources they own
        (e.g. in-process background threads, pooled HTTP connections) are released instead of leaked.
        """
        with _CTX_LOCK:
            for ctx in _CTX_STACK:
                _close_executor(ctx._executor)
                _close_storage(ctx._storage)
            _CTX_STACK.clear()

    @staticmethod
    def is_set() -> bool:
        """Return True if a runtime is available."""
        with _CTX_LOCK:
            return bool(_CTX_STACK)


def init_iqm_server_runtime(
    iqm_server_url: str,
    *,
    experiment_configuration_path: str | None = _SOURCE_FROM_SC,
    username: str = "",
    quantum_computer: str | None = None,
    token: str | None = None,
    tokens_file: str | None = None,
    client_signature: str | None = None,
) -> None:
    """Initialize the global ExaRuntime using IQM Server backend and Experiment Configuration.

    Args:
        iqm_server_url: URL for accessing the IQM Server. Has to start with http or https.
        experiment_configuration_path: If not given, will fetch the recommended configuration from the server.
            If None, an empty configuration is returned, which is useful only for loading old data.
            If `path` is a string, it must be a relative path to a configuration YAML file,
            which should have a top-level 'experiment' key.
        username: Overrides ``user`` in the configuration.
            If ``user`` is not set in the loaded configuration, giving the username here is mandatory.
        quantum_computer: ID or alias of the quantum computer to connect to, if the IQM Server
            instance controls more than one.
        token: Long-lived authentication token in plain text format.
            If ``token`` is given no other user authentication parameters should be given.
        tokens_file: Path to a tokens file used for authentication.
            If ``tokens_file`` is given no other user authentication parameters should be given.
        client_signature: String that the client adds to User-Agent header of requests
            it sends to the server.

    """

    def new_client() -> IQMServerClient:
        return IQMServerClient(
            iqm_server_url=iqm_server_url,
            quantum_computer=quantum_computer,
            token=token,
            tokens_file=tokens_file,
            client_signature=client_signature,
        )

    # The executor and the storage each get their own client, so that each owns an independent HTTP session
    # (a requests.Session is not thread-safe) and can be closed independently.
    executor = IQMServerExecutor(client=new_client())
    storage = IQMServerStorage(client=new_client())
    experiment_configuration = init_experiment_configuration(
        executor=executor, path=experiment_configuration_path, username=username
    )

    runtime = ExaRuntime(executor=executor, storage=storage, experiment_configuration=experiment_configuration)
    ExaRuntime.set(runtime)

    ExaRuntime.get_or_create_software_version_set()


def init_station_control_runtime(
    station_control_url: str,
    *,
    experiment_configuration_path: str | None = _SOURCE_FROM_SC,
    username: str = "",
    token: str | None = None,
    tokens_file: str | None = None,
    client_signature: str | None = None,
) -> None:
    """Initialize the global ExaRuntime using Station Control backend and Experiment Configuration.

    Args:
        station_control_url: URL for accessing the Station Control backend. Has to start with http or https.
        experiment_configuration_path: If not given, will fetch the recommended configuration from the server.
            If None, an empty configuration is returned, which is useful only for loading old data.
            If `path` is a string, it must be a relative path to a configuration YAML file,
            which should have a top-level 'experiment' key.
        username: Overrides ``user`` in the configuration.
            If ``user`` is not set in the loaded configuration, giving the username here is mandatory.
        token: Long-lived authentication token in plain text format.
            If ``token`` is given no other user authentication parameters should be given.
        tokens_file: Path to a tokens file used for authentication.
            If ``tokens_file`` is given no other user authentication parameters should be given.
        client_signature: String that the client adds to User-Agent header of requests
            it sends to the server.

    """

    def new_client() -> StationControlClient:
        return StationControlClient(
            station_control_url=station_control_url,
            token=token,
            tokens_file=tokens_file,
            client_signature=client_signature,
        )

    # The executor and the storage each get their own client, so that each owns an independent HTTP session
    # and can be closed independently. Keeps close() on either independent and safe to call without affecting
    # the other.
    executor = StationControlExecutor(client=new_client())
    storage = StationControlStorage(client=new_client())
    experiment_configuration = init_experiment_configuration(
        executor=executor, path=experiment_configuration_path, username=username
    )

    runtime = ExaRuntime(executor=executor, storage=storage, experiment_configuration=experiment_configuration)
    ExaRuntime.set(runtime)

    ExaRuntime.get_or_create_software_version_set()


def init_experiment_configuration(
    executor: ExecutorInterface | None, path: str | None = _SOURCE_FROM_SC, username: str = ""
) -> ExperimentConfiguration:
    """Initialize experiment configuration from a selected source.

    Args:
        executor: Interface used to fetch configuration from the server and validate controllers.
        path: If ``'__FROM STATION CONTROL__'``, fetches configuration from the server.
            If a string, loads from the local YAML path. If None, uses empty config.
        username: Overrides the 'user' field in the configuration. Mandatory if not in config.

    Returns:
        The fully initialized and validated ExperimentConfiguration.

    """
    if path == _SOURCE_FROM_SC:
        if executor is None:
            raise ValidationError(
                f"Executor interface is mandatory when fetching configuration from '{_SOURCE_FROM_SC}'."
            )
        experiment_configuration = ExperimentConfiguration(executor.get_experiment_configuration())
        log_text = "EXA configuration loaded from server."
    elif path:
        experiment_configuration = ExperimentConfiguration(load_yaml(path).get("experiment", {}))
        log_text = f"EXA configuration file loaded from {path}."
    else:
        log_text = "Initialized with empty EXA configuration."
        experiment_configuration = ExperimentConfiguration({})

    if username:
        experiment_configuration.user = username

    if not experiment_configuration.user:
        raise ValidationError("Username is not set. Please provide it in config or as an argument.")

    if experiment_configuration.dut_label == "":
        if executor is None:
            raise ValidationError("Executor interface is mandatory when 'dut_label' is not provided in configuration.")
        dut_label = executor.get_dut_label()
        experiment_configuration.dut_label = dut_label
        logger.debug(
            "Experiment 'dut_label' not provided in the experiment configuration, "
            "using the DUT provided by the executor. "
        )
    if executor is not None:
        experiment_configuration.add_conventional_controller_mapping(executor)
        experiment_configuration.validate_controllers(executor)

    logger.info("%s Username: %s", log_text, experiment_configuration.user)
    return experiment_configuration


_LOGGER_STATE = {"initialized": False}


DEFAULT_LOGGERS = {
    "loggers": {
        "iqm": None,
        "exa": None,
        "qcodes": None,
        "gbc": None,
        "calibration_graphs": None,
        "iqm.station_control.engine.util.sweep": "warning",  # info level is meant for server.
    },
    "verbose": False,
    "default_level": "INFO",
}


def init_loggers(path: str = "", force: bool = False) -> None:
    """Initialize logger format and handlers.

    Args:
        path: Path to a YAML configuration file.
        force: If True, re-initialize even if loggers were already set up.

    """
    if _LOGGER_STATE["initialized"] and not force:
        logger.debug("Loggers already initialized. Skipping.")
        return

    loggers_conf = DEFAULT_LOGGERS
    source_info = "default configuration"

    if path:
        try:
            loggers_conf = load_yaml(path)
            source_info = f"file: {path}"
        except (FileNotFoundError, ValueError) as e:
            logger.warning("Could not load logger config from %s (%s). Falling back to defaults.", path, e)

    # Standardize internal calls
    loggers_dict = cast(dict[str, str | None] | None, loggers_conf.get("loggers", DEFAULT_LOGGERS["loggers"]))
    verbose_val: bool = bool(loggers_conf.get("verbose", DEFAULT_LOGGERS["verbose"]))
    default_level_val: str = str(loggers_conf.get("default_level", DEFAULT_LOGGERS["default_level"]))

    exa.common.logger.init_loggers(
        loggers=loggers_dict,
        verbose=verbose_val,
        default_level=default_level_val,
    )

    _LOGGER_STATE["initialized"] = True
    logger.info("Loggers initialized using %s.", source_info)


def reset() -> None:
    """Reset the global ExaRuntime state.

    This clears the internal runtime stack, removing both the root runtime
    and any active scopes. This is primarily used for unit testing to
    ensure a clean slate between test cases.
    """
    ExaRuntime.clear()
    _LOGGER_STATE["initialized"] = False


def init_from_env(username: str = "", env_path: str | None = None) -> None:
    """Initialize the loggers and ExaRuntime from environment variables.

    Loggers will be initialized with default configuration if `environment.LOGGERS_CONF_PATH` is empty.

    Args:
        username: Passed to init_station_control_runtime to override or set the user.
        env_path: Path to a .env file to load explicitly. If provided, this file has the highest priority,
            overriding the current process environment and any other .env files.
            If ``env_path`` is not given, values are loaded from the current process environment and any
            .env files in current working directory. In case of conflict, existing environment variables
            are given priority (following the default logic of :func:`dotenv.load_dotenv`)

    Raises:
        FileNotFoundError: If env_path is provided but the file does not exist.


    """
    config = environment.load_environment(env_path=env_path)
    init_loggers(config.loggers_conf_path)

    station_control_url = config.station_control_service_url

    if station_control_url:
        init_station_control_runtime(
            station_control_url=config.station_control_service_url,
            experiment_configuration_path=config.configuration_path,
            username=username,
            token=config.iqm_token,
        )
    else:
        logger.warning("Environment variable 'STATION_CONTROL_SERVICE_URL' not found. ExaRuntime is not initialized.")


@functools.cache
def get_current_chip_label() -> str:
    """Returns the current chip label from the experiment configuration."""
    return ExaRuntime.experiment_configuration().dut_label
