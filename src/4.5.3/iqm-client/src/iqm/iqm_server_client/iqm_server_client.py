# Copyright 2025 IQM
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Client implementation for IQM Server REST API."""

from __future__ import annotations

import abc
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from functools import cache
from importlib.metadata import distributions, version
import json
import logging
import os
import platform
from time import sleep
from typing import Any, Literal, TypeAlias, TypeVar
from urllib.parse import urlparse
from uuid import UUID
import warnings

from iqm.data_definitions.station_control.v2.run_definition_pb2 import RunDefinition as RunDefinitionProto
from iqm.iqm_server_client.models import (
    CalibrationSet,
    JobData,
    JobStatus,
    ListQuantumComputersResponse,
    QualityMetricSet,
    Source,
    TimelineEntry,
)
from iqm.models.channel_properties import ChannelProperties
from opentelemetry import propagate, trace
from pydantic import BaseModel, TypeAdapter
import requests

from exa.common.data.setting_node import SettingNode
from exa.common.errors.station_control_errors import StationControlError, map_from_status_code_to_error
from iqm.station_control.client.authentication import ClientConfigurationError, TokenManager
from iqm.station_control.client.list_models import (
    DutList,
    ListModel,
    StaticQuantumArchitectureList,
)
from iqm.station_control.client.serializers import (
    deserialize_run_definition,
    deserialize_sweep_results,
    serialize_run_definition,
    serialize_sweep_job_request,
)
from iqm.station_control.client.serializers.channel_property_serializer import unpack_channel_properties
from iqm.station_control.client.serializers.setting_node_serializer import deserialize_setting_node
from iqm.station_control.client.utils import get_progress_bar_callback
from iqm.station_control.interface.models import (
    CircuitMeasurementCountsBatch,
    CircuitMeasurementResultsBatch,
    DutData,
    DynamicQuantumArchitecture,
    ProgressCallback,
    RunDefinition,
    RunRequest,
    StaticQuantumArchitecture,
    StrUUID,
    SweepDefinition,
    SweepResults,
)
from iqm.station_control.interface.pydantic_base import PydanticBase

logger = logging.getLogger(__name__)

TypePydanticBase = TypeVar("TypePydanticBase", bound=PydanticBase)
CircuitMeasurementResultsBatchAdapter = TypeAdapter(CircuitMeasurementResultsBatch)
CircuitCountsBatchAdapter = TypeAdapter(CircuitMeasurementCountsBatch)

StrUUIDOrDefault: TypeAlias = str | UUID | Literal["default"]

REQUESTS_TIMEOUT = float(os.environ.get("IQM_CLIENT_REQUESTS_TIMEOUT", "120"))

_POLLING_INTERVAL: float = float(os.environ.get("IQM_CLIENT_SECONDS_BETWEEN_CALLS", "1.0"))
"""IQM Server polling interval (in seconds)."""

DEFAULT_TIMEOUT_SECONDS: float = 10800.0  # 3 hours
"""Default timeout (in seconds) for waiting a job to finish."""


class _IQMServerClient:
    """Client implementation for IQM Server REST API.

    .. warning::

       ``_IQMServerClient`` is an unstable, private API.
       It may change without notice. Do not use it outside this package.

    Args:
        iqm_server_url: Remote IQM Server URL to connect to.
        quantum_computer: ID or alias of the quantum computer to connect to, if the IQM Server
            instance controls more than one.
        token: Long-lived IQM token in plain text format.
        tokens_file: Path to a tokens file used for authentication.
        client_signature: String that is added to the User-Agent header of requests
            sent to the server.
        enable_opentelemetry: Iff True, enable Jaeger/OpenTelemetry tracing.
        timeout: Timeout for the request in seconds.

    """

    @staticmethod
    def normalize_url(iqm_server_url: str, quantum_computer: str | None) -> tuple[str, str | None, bool]:
        """Validate the connection details, provide some backwards compatibility."""
        # Security measure: mitigate UTF-8 read order control character
        # exploits by allowing only ASCII urls
        if not iqm_server_url.isascii():
            raise ClientConfigurationError(f"Non-ASCII characters in URL: {iqm_server_url}")
        try:
            url = urlparse(iqm_server_url)
        except Exception as e:
            raise ClientConfigurationError(f"Invalid URL: {iqm_server_url}") from e

        if url.scheme not in {"http", "https"}:
            raise ClientConfigurationError(
                f"The URL schema has to be http or https. Incorrect schema in URL: {iqm_server_url}"
            )

        hostname = url.hostname or ""
        path_segments = (url.path or "").split("/")

        # Compatibility: to maintain the API compatibility, Resonance URLs included the quantum computer
        # alias as a path prefix and served the Cocos/SC compatible API per QC under that prefix.
        # Now that this new IQM client implementation does not have such compatibility requirements,
        # the preferred way is to explicitly set the connected quantum computer alias as an initialization
        # parameter, e.g. IQMClient(root_url, quantum_computer="garnet") and also target jobs for
        # timeslot explicitly when submitting the job e.g. client.submit_circuits(..., use_timeslot=True).
        # However, we want to maintain some sort of compatibility by allowing such URLs for now and
        # warn the user about the "old-style" usage. Once this version of client is released, the Resonance
        # documentation can be updated to instruct the "new-style" client initialization, and eventually
        # we can also drop this compatibility code.
        quantum_computer_from_path = path_segments[-1].removesuffix(":timeslot")
        if quantum_computer is not None and quantum_computer_from_path:
            raise ClientConfigurationError(
                "The IQM Server URL must not contain quantum computer name when initializing client with "
                + "explicit quantum computer name. To fix this error, use server base url."
            )
        if quantum_computer_from_path:
            quantum_computer = quantum_computer_from_path
            warnings.warn(
                "The given IQM Server URL is in a deprecated format, see the client initialization instructions "
                + "and correct URL format from the server web dashboard."
            )

        # Same for timeslots: the timeslot / FIFO queue selection had to be embedded into URL, whereas in this
        # new implementation, the explicit timeslot usage is preferred upon the actual job submission. Fixing
        # compatibility but giving a warning about the deprecated usage.
        use_timeslot_default = path_segments[-1].endswith(":timeslot")
        if use_timeslot_default:
            warnings.warn(
                "Quantum computer timeslot URL is deprecated. Jobs can be submitted to timeslots by using "
                + "`use_timeslot=True` parameter per job. See the server web dashboard or "
                + "https://docs.iqm.tech/iqm-client/ for more detailed instructions."
            )

        # Same here; the "cocos" subdomain was used to handle the backwards compatibility so we can just drop
        # it now and give a warning. This only resonance specific compatibility change so we can be precise
        # in the warning message.
        if hostname.startswith("cocos."):
            hostname = hostname.removeprefix("cocos.")
            warnings.warn(
                "Resonance CoCoS API is deprecated. Use https://resonance.iqm.tech. See the Resonance "
                + "documentation or https://docs.iqm.tech/iqm-client/ for more detailed instructions."
            )

        # Use hostname without "cocos" subdomain and quantum computer name
        port_suffix = f":{url.port}" if url.port else ""
        netloc = "/".join([hostname] + path_segments[:-1]).rstrip("/")
        base_url = f"{url.scheme}://{netloc}{port_suffix}"

        return base_url, quantum_computer, use_timeslot_default

    def __init__(
        self,
        iqm_server_url: str | None = None,
        *,
        quantum_computer: str | None = None,
        token: str | None = None,
        tokens_file: str | None = None,
        client_signature: str | None = None,
        enable_opentelemetry: bool = False,
        timeout: float = REQUESTS_TIMEOUT,
    ):
        server_url_param_name = "iqm_server_url"
        quantum_computer_param_name = "quantum_computer"
        init_parameters = {server_url_param_name: iqm_server_url, quantum_computer_param_name: quantum_computer}
        env_variables = {server_url_param_name: "IQM_SERVER_URL", quantum_computer_param_name: "IQM_QUANTUM_COMPUTER"}
        # given args take precedence over env variables
        for param_name, env_var in env_variables.items():
            if (env_var_value := os.environ.get(env_var)) is not None and init_parameters.get(param_name) is None:
                init_parameters[param_name] = env_var_value
        iqm_server_url = init_parameters[server_url_param_name]
        if iqm_server_url is None:
            raise ValueError("IQM Server URL must be provided.")
        quantum_computer = init_parameters[quantum_computer_param_name]

        root_url, quantum_computer, use_timeslot_default = _IQMServerClient.normalize_url(
            iqm_server_url, quantum_computer
        )
        self.root_url = root_url
        # authentication
        tm = TokenManager(token, tokens_file)
        self._token_manager = tm
        self._auth_header_callback = tm.get_auth_header_callback()

        self._signature = self._create_signature(client_signature)

        self._enable_opentelemetry = enable_opentelemetry
        self._timeout = timeout
        self._quantum_computer = self._resolve_quantum_computer(quantum_computer)
        self._use_timeslot = use_timeslot_default

    @property
    def api_version(self) -> str:
        """API version of the IQM Server API this client is using."""
        return "v1"

    @property
    def quantum_computer(self) -> str:
        """Human-readable alias of the quantum computer this client connects to."""
        return self._quantum_computer

    def get_health(self) -> dict[str, Any]:
        """Get the status of the IQM Server."""
        response = self._send_request(requests.get, f"quantum-computers/{self._quantum_computer}/health")
        return response.json()

    def _debug_info(self) -> dict[str, Any]:
        """Information about the client and server versions, and the platform.

        .. note:: The output of this method is for internal use only, and may change without notice.
        """

        def mask_env_var(name: str) -> int | None:
            """Mask the sensitive information in an env var, returning just its length."""
            temp = os.environ.get(name)
            return None if temp is None else len(temp)

        # locally installed packages
        local_dist_pkgs = distributions()

        def pkg_filter(name: str) -> bool:
            """Filter out the packages we are interested in."""
            return name.startswith("iqm-") or name in {"cirq", "qiskit", "qiskit-aer", "qrisp"}

        info = {
            "platform.platform": platform.platform(),
            "platform.version": platform.version(),
            "platform.python_version": platform.python_version(),
            "root_url": self.root_url,
            "quantum_computer": self._quantum_computer,
            "use_timeslot": self._use_timeslot,
            "len(IQM_TOKEN)": mask_env_var("IQM_TOKEN"),
            "len(IQM_TOKENS_FILE)": mask_env_var("IQM_TOKENS_FILE"),
            "token_provider": type(self._token_manager._token_provider),
            "auth_header_callback": self._token_manager._auth_header_callback,
            "local packages": {dist.name: dist.version for dist in local_dist_pkgs if pkg_filter(dist.name)},
        }
        try:
            info["about"] = self.get_about()
            about_station = self.get_about_station()
            info["about_station"] = {
                "version": about_station["version"],
                "software_versions": {
                    key: value for key, value in about_station["software_versions"].items() if key.startswith("iqm-")
                },
            }
        except StationControlError as exc:
            info["server error"] = str(exc)
        return info

    @cache
    def get_about(self) -> dict[str, Any]:
        """Get information about IQM Server."""
        response = self._send_request(requests.get, "about", use_api=False)
        return response.json()

    @cache
    def get_about_station(self) -> dict[str, Any]:
        """Get information about Station Control."""
        response = self._send_request(requests.get, f"quantum-computers/{self._quantum_computer}/artifacts/about")
        return response.json()

    def get_settings(self) -> SettingNode:
        """Default settings tree of the quantum computer, as defined in the configuration files."""
        return self._get_cached_settings().model_copy()

    @cache
    def _get_cached_settings(self) -> SettingNode:
        """Get and cache the quantum computer default settings."""
        headers = {"Accept": "application/protobuf"}
        response = self._send_request(
            requests.get, f"quantum-computers/{self._quantum_computer}/artifacts/settings", headers=headers
        )
        return deserialize_setting_node(response.content)

    @cache
    def get_chip_design_records(self) -> list[dict[str, Any]]:
        """Get the chip design records of the quantum computer."""
        response = self._send_request(
            requests.get, f"quantum-computers/{self._quantum_computer}/artifacts/chip-design-records"
        )
        return response.json()

    @cache
    def get_channel_properties(self) -> dict[str, ChannelProperties]:
        """Get the channel properties from the quantum computer.

        Channel properties contain information about the hardware limitations, e.g. the sample rate,
        granularity and supported instructions for the various control channels.

        Returns:
            Mapping from channel name to AWGProperties or ReadoutProperties.

        """
        headers = {"Accept": "application/protobuf"}
        response = self._send_request(
            requests.get, f"quantum-computers/{self._quantum_computer}/artifacts/channel-properties", headers=headers
        )
        decoded_dict = unpack_channel_properties(response.content)
        return decoded_dict

    def get_duts(self) -> list[DutData]:
        """Get the DUT(s) of the quantum computer."""
        response = self._send_request(requests.get, f"quantum-computers/{self._quantum_computer}/artifacts/duts")
        return self._deserialize_response(response, DutList)

    def get_static_quantum_architectures(self) -> list[StaticQuantumArchitecture]:
        """Get the static quantum architecture(s) of the quantum computer."""
        response = self._send_request(
            requests.get, f"quantum-computers/{self._quantum_computer}/artifacts/static-quantum-architectures"
        )
        return self._deserialize_response(response, StaticQuantumArchitectureList)

    def get_calibration_set(self, calibration_set_id: StrUUIDOrDefault) -> CalibrationSet:
        """Get a calibration set from the database."""
        response = self._send_request(requests.get, f"calibration-sets/{self._quantum_computer}/{calibration_set_id}")
        return self._deserialize_response(response, CalibrationSet)

    def get_dynamic_quantum_architecture(self, calibration_set_id: StrUUIDOrDefault) -> DynamicQuantumArchitecture:
        """Get the dynamic quantum architecture for the given calibration set ID.

        Returns:
            Dynamic quantum architecture of the quantum computer for the given calibration set ID.

        """
        response = self._send_request(
            requests.get, f"calibration-sets/{self._quantum_computer}/{calibration_set_id}/dynamic-quantum-architecture"
        )
        return self._deserialize_response(response, DynamicQuantumArchitecture)

    def get_calibration_set_quality_metric_set(self, calibration_set_id: StrUUIDOrDefault) -> QualityMetricSet:
        """Get the latest quality metric set for the given calibration set ID."""
        response = self._send_request(
            requests.get,
            f"calibration-sets/{self._quantum_computer}/{calibration_set_id}/metrics",
        )
        return self._deserialize_response(response, QualityMetricSet)

    def submit_sweep(
        self,
        sweep_definition: SweepDefinition,
        *,
        use_timeslot: bool = False,
    ) -> JobData:
        """Submit an N-dimensional sweep for execution.

        Args:
            sweep_definition: The content of the sweep to be created.
            use_timeslot: If ``True`` submit the job to the timeslot queue, otherwise
                submit it to the shared FIFO queue.

        Returns:
            Upon successful submission: sweep job data, including the job ID that can be used to track it.

        """
        data = serialize_sweep_job_request(sweep_definition, queue_name="sweeps")
        return self._submit_job(job_type="sweep", protobuf_data=data, use_timeslot=use_timeslot)

    def submit_run(
        self,
        run_definition: RunDefinition,
        *,
        use_timeslot: bool = False,
    ) -> JobData:
        """Submit an experiment run for execution.

        Args:
            run_definition: The content of the run to be created.
            use_timeslot: If ``True`` submit the job to the timeslot queue, otherwise
                submit it to the shared FIFO queue.

        Returns:
            Upon successful submission: run job data, including the job ID that can be used to track it.

        """
        # We use purposefully "serialize_run_definition" here instead of "serialize_run_job_request"
        # Enveloping run inside "SweepTaskRequestProto" isn't necessary when using the Core API.
        # "submit_sweep" works still with the old way, new Core API works differently than the Station Control API.
        data = serialize_run_definition(run_definition).SerializeToString()
        return self._submit_job(job_type="run", protobuf_data=data, use_timeslot=use_timeslot)

    def submit_circuits(self, run_request: RunRequest, *, use_timeslot: bool = False) -> JobData:
        """Submit a batch of quantum circuits for execution.

        Args:
            run_request: Circuit execution request.
            use_timeslot: If ``True`` submit the job to the timeslot queue, otherwise
                submit it to the shared FIFO queue.

        Returns:
            Upon successful submission: circuit job data, including the job ID that can be used to track it.

        """
        data = self._serialize_model(run_request)
        return self._submit_job(job_type="circuit", json_data=data, use_timeslot=use_timeslot)

    def get_job(self, job_id: StrUUID) -> JobData:
        """Get the current status and metadata of the job."""
        response = self._send_request(requests.get, f"jobs/{job_id}")
        return self._deserialize_response(response, JobData)

    def cancel_job(self, job_id: StrUUID) -> None:
        """Cancel a job.

        A canceled job will remain in the server database, but it will not be executed.
        If the job is currently being executed, it is interrupted.
        If the job was already executed (or failed), it will remain in its current terminal state.

        Args:
            job_id: The ID of the job to cancel.

        """
        self._send_request(requests.post, f"jobs/{job_id}/cancel")

    def delete_job(self, job_id: StrUUID) -> None:
        """Delete a job with the given ID.

        Works like :meth:`cancel_job`, but also removes the job from the IQM Server database.
        """
        self._send_request(requests.delete, f"jobs/{job_id}")

    def get_submit_run_payload(self, job_id: StrUUID) -> RunDefinition:
        """Get the job payload, i.e. the contents of the run definition sent to IQM Server."""
        response = self._send_request(requests.get, f"jobs/{job_id}/payload")
        run_definition_proto = RunDefinitionProto()
        run_definition_proto.ParseFromString(response.content)
        return deserialize_run_definition(run_definition_proto)

    def get_submit_circuits_payload(self, job_id: StrUUID) -> RunRequest:
        """Get the job payload, i.e. the contents of the run request sent to IQM Server."""
        response = self._send_request(requests.get, f"jobs/{job_id}/payload")
        return self._deserialize_response(response, RunRequest)

    def get_job_artifact_sweep_results(self, job_id: StrUUID) -> SweepResults:
        """Get N-dimensional sweep results from the database for the given sweep job."""
        response = self._send_request(requests.get, f"jobs/{job_id}/artifacts/sweep_results")
        return deserialize_sweep_results(response.content)

    def get_job_artifact_measurements(self, job_id: StrUUID) -> CircuitMeasurementResultsBatch:
        """Get the "measurements" artifact of the given circuit job."""
        response = self._send_request(requests.get, f"jobs/{job_id}/artifacts/measurements")
        return CircuitMeasurementResultsBatchAdapter.validate_json(response.text)

    def get_job_artifact_measurement_counts(self, job_id: StrUUID) -> CircuitMeasurementCountsBatch:
        """Get the "measurement_counts" artifact of the given circuit job."""
        response = self._send_request(requests.get, f"jobs/{job_id}/artifacts/measurement_counts")
        return CircuitCountsBatchAdapter.validate_json(response.text)

    def _submit_job(
        self,
        *,
        job_type: str,
        json_data: str | None = None,
        protobuf_data: bytes | None = None,
        use_timeslot: bool = False,
    ) -> JobData:
        """Submit a job for execution."""
        params = _serialize_query_params({"use_timeslot": use_timeslot or self._use_timeslot})
        response = self._send_request(
            requests.post,
            f"jobs/{self._quantum_computer}/{job_type}",
            params=params,
            json_data=json_data,
            protobuf_data=protobuf_data,
        )
        return self._deserialize_response(response, JobData)

    def _wait_job_completion(
        self,
        job_id: StrUUID,
        *,
        progress_callback: ProgressCallback | None,
        timeout_secs: float = 0.0,
    ) -> JobData:
        """Wait for the completion of a job.

        Polls the server, updating ``progress_callback``, until the job is in a terminal state.
        Will stop the polling upon receiving a KeyboardInterrupt (Ctrl-C).
        Does not cancel the job.

        Args:
            job_id: The ID of the job to wait for.
            progress_callback: If not None, used to report the job progress to the caller while waiting.
                Called with the relevant progress indicator info after each poll.
            timeout_secs: If nonzero, stop polling after this many seconds and return the non-terminal status.

        Returns:
            Last seen job data.

        """
        logger.info("Waiting for job %s to finish...", job_id)
        progress_callback = progress_callback or (lambda status: None)
        # TODO How should the progress meter work? all the progress items should be in one list.
        max_seen_queue_position = 0
        start_time = datetime.now()

        while True:
            job_data = self.get_job(job_id)

            if job_data.queue_position is not None:
                # Job is still in the iqm-server queue
                position = job_data.queue_position
                max_seen_queue_position = max(max_seen_queue_position, position)
                progress_callback([("Progress in queue", max_seen_queue_position - position, max_seen_queue_position)])
            elif (execution := job_data.execution) is not None:
                # Convert the progress info into the old format and report it using the callback
                statuses = [(label, v.value, v.max_value) for label, v in execution.progress.items()]
                progress_callback(statuses)

            # Check for completion before enforcing the timeout to avoid false negatives
            if job_data.status in JobStatus.terminal_statuses():
                return job_data

            # Enforce the absolute timeout if one was requested
            if timeout_secs and (datetime.now() - start_time).total_seconds() >= timeout_secs:
                logger.warning(
                    f"Job {job_id} reached the timeout of {timeout_secs}s. "
                    "Stopping polling and returning the current non-terminal status."
                )
                return job_data

            sleep(_POLLING_INTERVAL)

    @staticmethod
    def _serialize_model(model: BaseModel) -> str:
        """Serialize a Pydantic model into a JSON string.

        All Pydantic models should be serialized using this method, to keep the client behavior uniform.

        Args:
            model: Pydantic model to JSON-serialize.

        Returns:
            Corresponding JSON string, may contain arbitrary Unicode characters.

        """
        # Strings in model can contain non-latin-1 characters. Unlike json.dumps which encodes non-latin-1 chars
        # using the \uXXXX syntax, BaseModel.model_dump_json() keeps them in the produced JSON str.
        return model.model_dump_json()

    @classmethod
    def _create_signature(cls, client_signature: str | None) -> str:
        """Prepare the User-Agent header sent to the server."""
        signature = f"{platform.platform(terse=True)}"
        signature += f", python {platform.python_version()}"
        dist_pkg_name = "iqm-client"
        signature += f", {cls.__name__} {dist_pkg_name} {version(dist_pkg_name)}"
        if client_signature:
            signature += f", {client_signature}"
        return signature

    def _resolve_quantum_computer(self, user_defined_quantum_computer: str | None) -> str:
        """Human-readable alias of the quantum computer this client connects to."""
        response = self._send_request(requests.get, "quantum-computers")
        quantum_computers = self._deserialize_response(response, ListQuantumComputersResponse).quantum_computers
        aliases = ", ".join((qc.alias for qc in quantum_computers))
        if user_defined_quantum_computer is None:
            if len(quantum_computers) == 1:
                return quantum_computers[0].alias
            raise ClientConfigurationError(f"Quantum computer not selected. Available quantum computers are: {aliases}")

        qc = next((qc for qc in quantum_computers if qc.alias == user_defined_quantum_computer), None)
        if qc is None:
            raise ClientConfigurationError(
                f'Quantum computer "{user_defined_quantum_computer}" does not exist. '
                + f"Available quantum computers are: {aliases}"
            )
        return qc.alias

    def _send_request(
        self,
        http_method: Callable[..., requests.Response],
        url_path: str,
        *,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        json_data: str | None = None,
        protobuf_data: bytes | None = None,
        use_api: bool = True,
    ) -> requests.Response:
        """Send an HTTP request.

        Parameters ``json_str`` and ``octets`` are mutually exclusive.
        The first non-None argument (in this order) will be used to construct the body of the request.

        Args:
            http_method: HTTP method to use for the request, any of requests.[post|get|put|head|delete|patch|options].
            url_path: URL for the request.
            headers: Additional HTTP headers for the request. Some may be overridden.
            params: HTTP query parameters to store in the query string of the request URL.
            json_data: JSON string to store in the body, may contain arbitrary Unicode characters.
            protobuf_data: Pre-serialized protobuf binary data to store in the body.
            use_api: Iff False, append ``url_path`` to the root URL, otherwise insert the API path
                between them.

        Returns:
            Response to the request.

        Raises:
            StationControlError: Request was not successful.

        """
        # Will raise an error if respectively an error response code is returned.
        # http_method should be any of requests.[post|get|put|head|delete|patch|options]

        request_kwargs = self._prepare_request_kwargs(
            headers=headers or {},
            params=params or {},
            json_data=json_data,
            protobuf_data=protobuf_data,
            timeout=self._timeout,
        )
        api_path = f"api/{self.api_version}/" if use_api else ""
        url = f"{self.root_url}/{api_path}{url_path}"
        response = http_method(url, **request_kwargs)
        if not response.ok:
            try:
                response_json = response.json()
                error_message = response_json.get("message") or response_json["detail"]
            except (json.JSONDecodeError, KeyError):
                error_message = response.text

            error_class = map_from_status_code_to_error(response.status_code)
            raise error_class(error_message)
        return response

    def _prepare_request_kwargs(
        self,
        *,
        headers: dict[str, str],
        params: dict[str, Any],
        json_data: str | None = None,
        protobuf_data: bytes | None = None,
        timeout: float,
    ) -> dict[str, Any]:
        """Return the keyword arguments for a :mod:`requests` HTTP method."""
        # Add default headers
        _headers = self._default_headers()

        # "json_str" and "protobuf_data" are mutually exclusive
        data: bytes | None = None
        if json_data is not None:
            # Must be able to handle JSON strings with arbitrary Unicode characters,
            # so we use an explicit encoding into bytes,
            # and set the headers so the recipient can decode the request body correctly.
            data = json_data.encode("utf-8")
            _headers["Content-Type"] = "application/json; charset=UTF-8"
        elif protobuf_data is not None:
            data = protobuf_data
            _headers["Content-Type"] = "application/protobuf"

        if "Accept" in headers:
            _headers["Accept"] = headers["Accept"]

        if self._enable_opentelemetry:
            parent_span_context = trace.set_span_in_context(trace.get_current_span())
            propagate.inject(carrier=headers, context=parent_span_context)

        kwargs = {
            "headers": _headers,
            "params": params,
            "data": data,
            "timeout": timeout,
        }
        return _remove_empty_values(kwargs)

    def _default_headers(self) -> dict[str, str]:
        """Return the default headers for an HTTP request to IQM Server."""
        headers = {
            "User-Agent": self._signature,
            "Accept": "application/json",
        }
        # If auth header callback exists, use it to add the header
        if self._auth_header_callback:
            headers["Authorization"] = self._auth_header_callback()
        return headers

    @staticmethod
    def _deserialize_response(
        response: requests.Response,
        model_class: type[TypePydanticBase | ListModel],
    ) -> TypePydanticBase:
        """Deserialize data using a Pydantic model."""
        # Use "model_validate_json(response.text)" instead of "model_validate(response.json())".
        # This validates the provided data as a JSON string or bytes object.
        # If your incoming data is a JSON payload, this is generally considered faster.
        model = model_class.model_validate_json(response.text)
        if isinstance(model, ListModel):
            return model.root
        return model


def _remove_empty_values(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the given dict without values that are None or {}."""
    return {key: value for key, value in kwargs.items() if value not in [None, {}]}


def _serialize_query_params(params: dict[str, Any]) -> dict[str, Any]:
    return {key: _serialize_query_param(value) for key, value in params.items() if value not in [None, {}]}


def _serialize_query_param(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


@dataclass
class IQMServerClientJob(abc.ABC):
    """ABC for classes representing an IQMServerClient job."""

    data: JobData
    """Light job-related data.

    Experimental, should be considered private for now.
    """

    @property
    def job_id(self) -> UUID:
        """Unique ID of the job."""
        return self.data.id

    @property
    def status(self) -> JobStatus:
        """Last queried status of the job.

        Note that this is not necessarily the same as the current status of the job,
        unless the status is terminal.

        To get the current status, use :meth:`update`.
        """
        return self.data.status

    @property
    @abc.abstractmethod
    def _iqm_server_client(self) -> _IQMServerClient:
        """A way to reach a client instance."""
        raise NotImplementedError

    @property
    def _errors(self) -> str:
        """All errors formatted as a string."""
        return "\n".join(f"  {str(error)}" for error in self.data.errors)

    def update(self) -> JobStatus:
        """Update the job data by querying the server.

        Modifies ``self``.

        Returns:
            Current status of the job.

        """
        # TODO we somewhat unnecessarily call get_job() for COMPLETED, FAILED and CANCELLED jobs here,
        # since those states are terminal.
        job_data = self._iqm_server_client.get_job(self.job_id)
        self.data = job_data
        return self.status

    def cancel(self) -> None:
        """Cancel the job.

        See :meth:`_IQMServerClient.cancel_job`.

        """
        self._iqm_server_client.cancel_job(self.job_id)

    def wait_for_completion(
        self,
        *,
        timeout_secs: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> JobStatus:
        """Poll the server updating the job status, until the job reaches a terminal state, or until we hit a timeout.

        The terminal states are "completed", "failed", and "cancelled".


        Will stop the polling (but does not cancel the job) upon receiving a KeyboardInterrupt (Ctrl-C).
        If you want to cancel the job, call :meth:`cancel`.

        Modifies ``self``.

        Args:
            timeout_secs: If nonzero, stop polling after this many seconds and return the non-terminal status.

        Returns:
            Last seen job status.

        Raises:
            KeyboardInterrupt: Received Ctrl-C while waiting for the job to finish.

        """
        try:
            job_data = self._iqm_server_client._wait_job_completion(
                self.job_id,
                progress_callback=get_progress_bar_callback(),
                timeout_secs=timeout_secs,
            )
        except KeyboardInterrupt:
            # user pressed Ctrl-C
            job_data = self._iqm_server_client.get_job(self.job_id)

        self.data = job_data

        if self.data.messages:
            logger.debug("Job messages:\n%s", "\n".join(f"  {msg.source}: {msg.message}" for msg in self.data.messages))

        if self.status == JobStatus.FAILED:
            logger.error(
                "Job failed! Error(s):\n%s",
                self._errors,
            )
        elif self.status == JobStatus.CANCELLED:
            logger.error("Job was cancelled!")

        return self.status

    def find_timeline_entry(
        self,
        status: str,
        source: Source | None = None,
    ) -> TimelineEntry | None:
        """Search the timeline for an entry matching the given criteria.

        Args:
            status: Status of the searched timeline entry.
            source: Source of the searched timeline entry. If None, accepts any source.

        Returns:
            The first matching entry or ``None`` if the job timeline does not have any matching entries.

        """
        for entry in self.data.timeline:
            if entry.status == status and (entry.source == source or source is None):
                return entry
        return None
