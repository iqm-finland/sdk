# Copyright 2021-2024 IQM client developers
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
"""Client for connecting to the IQM quantum computer server interface."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from http import HTTPStatus
from importlib.metadata import version
import json
import os
import platform
import time
from typing import Any, TypeVar
from uuid import UUID
import warnings

from iqm.iqm_client.api import APIConfig, APIEndpoint
from iqm.iqm_client.authentication import TokenManager
from iqm.iqm_client.errors import (
    APITimeoutError,
    CircuitExecutionError,
    CircuitValidationError,
    ClientAuthenticationError,
    ClientConfigurationError,
    EndpointRequestError,
    JobAbortionError,
)
from iqm.iqm_client.models import (
    CalibrationSet,
    CircuitBatch,
    CircuitCompilationOptions,
    ClientLibrary,
    ClientLibraryDict,
    DynamicQuantumArchitecture,
    QualityMetricSet,
    QuantumArchitectureSpecification,
    RunCounts,
    RunRequest,
    RunResult,
    RunStatus,
    StaticQuantumArchitecture,
    Status,
    serialize_qubit_mapping,
    validate_circuit,
)
from iqm.iqm_client.validation import validate_circuit_instructions, validate_qubit_mapping
from iqm.models.channel_properties import AWGProperties
from packaging.version import parse
from pydantic import BaseModel, ValidationError
import requests
from requests import HTTPError

from iqm.station_control.client.iqm_server.iqm_server_client import IqmServerClient
from iqm.station_control.client.utils import init_station_control
from iqm.station_control.interface.models import ObservationLite
from iqm.station_control.interface.station_control import StationControlInterface

T_BaseModel = TypeVar("T_BaseModel", bound=BaseModel)

REQUESTS_TIMEOUT = float(os.environ.get("IQM_CLIENT_REQUESTS_TIMEOUT", 120.0))
DEFAULT_TIMEOUT_SECONDS = 900
SECONDS_BETWEEN_CALLS = float(os.environ.get("IQM_CLIENT_SECONDS_BETWEEN_CALLS", 1.0))


class IQMClient:
    """Provides access to IQM quantum computers.

    Args:
        url: Endpoint for accessing the server. Has to start with http or https.
        client_signature: String that IQMClient adds to User-Agent header of requests
            it sends to the server. The signature is appended to IQMClient's own version
            information and is intended to carry additional version information,
            for example the version information of the caller.
        token: Long-lived authentication token in plain text format. Used by IQM Resonance.
            If ``token`` is given no other user authentication parameters should be given.
        tokens_file: Path to a tokens file used for authentication.
            If ``tokens_file`` is given no other user authentication parameters should be given.
        auth_server_url: Base URL of the authentication server.
            If ``auth_server_url`` is given also ``username`` and ``password`` must be given.
        username: Username to log in to authentication server.
        password: Password to log in to authentication server.

    Alternatively, the user authentication related keyword arguments can also be given in
    environment variables :envvar:`IQM_TOKEN`, :envvar:`IQM_TOKENS_FILE`, :envvar:`IQM_AUTH_SERVER`,
    :envvar:`IQM_AUTH_USERNAME` and :envvar:`IQM_AUTH_PASSWORD`. All parameters must be given either
    as keyword arguments or as environment variables. Same combination restrictions apply
    for values given as environment variables as for keyword arguments.

    """

    def __init__(
        self,
        url: str,
        *,
        client_signature: str | None = None,
        token: str | None = None,
        tokens_file: str | None = None,
        auth_server_url: str | None = None,
        username: str | None = None,
        password: str | None = None,
    ):
        if not url.startswith(("http:", "https:")):
            raise ClientConfigurationError(f"The URL schema has to be http or https. Incorrect schema in URL: {url}")
        self._token_manager = TokenManager(
            token,
            tokens_file,
            auth_server_url,
            username,
            password,
        )
        version_string = "iqm-client"
        self._signature = f"{platform.platform(terse=True)}"
        self._signature += f", python {platform.python_version()}"
        self._signature += f", iqm-client {version(version_string)}"
        if client_signature:
            self._signature += f", {client_signature}"
        self._architecture: QuantumArchitectureSpecification | None = None
        self._static_architecture: StaticQuantumArchitecture | None = None
        self._dynamic_architectures: dict[UUID, DynamicQuantumArchitecture] = {}

        self._station_control: StationControlInterface = init_station_control(
            root_url=url,
            get_token_callback=self._token_manager.get_bearer_token,  # type:ignore[arg-type]
            client_signature=client_signature,
        )
        self._api = APIConfig(url)
        if (version_incompatibility_msg := self._check_versions()) is not None:
            warnings.warn(version_incompatibility_msg)

    def __del__(self):
        try:
            # Try our best to close the auth session, doesn't matter if it fails
            self.close_auth_session()
        except Exception:
            pass

    def get_about(self) -> dict:
        """Return information about the IQM client."""
        return self._station_control.get_about()

    def get_health(self) -> dict:
        """Return the status of the station control service."""
        return self._station_control.get_health()

    def submit_circuits(
        self,
        circuits: CircuitBatch,
        *,
        qubit_mapping: dict[str, str] | None = None,
        custom_settings: dict[str, Any] | None = None,
        calibration_set_id: UUID | None = None,
        shots: int = 1,
        options: CircuitCompilationOptions | None = None,
    ) -> UUID:
        """Submit a batch of quantum circuits for execution on a quantum computer.

        Args:
            circuits: Circuits to be executed.
            qubit_mapping: Mapping of logical qubit names to physical qubit names.
                Can be set to ``None`` if all ``circuits`` already use physical qubit names.
                Note that the ``qubit_mapping`` is used for all ``circuits``.
            custom_settings: Custom settings to override default settings and calibration data.
                Note: This field should always be ``None`` in normal use.
            calibration_set_id: ID of the calibration set to use, or ``None`` to use the current default calibration.
            shots: Number of times ``circuits`` are executed. Must be greater than zero.
            options: Various discrete options for compiling quantum circuits to instruction schedules.

        Returns:
            ID for the created job. This ID is needed to query the job status and the execution results.

        """
        run_request = self.create_run_request(
            circuits=circuits,
            qubit_mapping=qubit_mapping,
            custom_settings=custom_settings,
            calibration_set_id=calibration_set_id,
            shots=shots,
            options=options,
        )
        job_id = self.submit_run_request(run_request)
        return job_id

    def create_run_request(
        self,
        circuits: CircuitBatch,
        *,
        qubit_mapping: dict[str, str] | None = None,
        custom_settings: dict[str, Any] | None = None,
        calibration_set_id: UUID | None = None,
        shots: int = 1,
        options: CircuitCompilationOptions | None = None,
    ) -> RunRequest:
        """Create a run request for executing circuits without sending it to the server.

        This is called in :meth:`submit_circuits` and does not need to be called separately in normal usage.

        Can be used to inspect the run request that would be submitted by :meth:`submit_circuits`, without actually
        submitting it for execution.

        Args:
            circuits: Circuits to be executed.
            qubit_mapping: Mapping of logical qubit names to physical qubit names.
                Can be set to ``None`` if all ``circuits`` already use physical qubit names.
                Note that the ``qubit_mapping`` is used for all ``circuits``.
            custom_settings: Custom settings to override default settings and calibration data.
                Note: This field should always be ``None`` in normal use.
            calibration_set_id: ID of the calibration set to use, or ``None`` to use the current default calibration.
            shots: Number of times ``circuits`` are executed. Must be greater than zero.
            options: Various discrete options for compiling quantum circuits to instruction schedules.

        Returns:
            RunRequest that would be submitted by equivalent call to :meth:`submit_circuits`.

        """
        if shots < 1:
            raise ValueError("Number of shots must be greater than zero.")
        if options is None:
            options = CircuitCompilationOptions()

        for i, circuit in enumerate(circuits):
            try:
                # validate the circuit against the static information in iqm.iqm_client.models._SUPPORTED_OPERATIONS
                validate_circuit(circuit)
            except ValueError as e:
                raise CircuitValidationError(f"The circuit at index {i} failed the validation").with_traceback(
                    e.__traceback__
                )

        dynamic_quantum_architecture = self.get_dynamic_quantum_architecture(calibration_set_id)

        validate_qubit_mapping(dynamic_quantum_architecture, circuits, qubit_mapping)
        # validate the circuit against the calibration-dependent dynamic quantum architecture
        validate_circuit_instructions(
            dynamic_quantum_architecture,
            circuits,
            qubit_mapping,
            validate_moves=options.move_gate_validation,
            must_close_sandwiches=False,
        )

        serialized_qubit_mapping = serialize_qubit_mapping(qubit_mapping) if qubit_mapping else None

        return RunRequest(
            qubit_mapping=serialized_qubit_mapping,
            circuits=circuits,
            custom_settings=custom_settings,
            calibration_set_id=calibration_set_id,
            shots=shots,
            max_circuit_duration_over_t2=options.max_circuit_duration_over_t2,
            heralding_mode=options.heralding_mode,
            move_validation_mode=options.move_gate_validation,
            move_gate_frame_tracking_mode=options.move_gate_frame_tracking,
            active_reset_cycles=options.active_reset_cycles,
            dd_mode=options.dd_mode,
            dd_strategy=options.dd_strategy,
        )

    def submit_run_request(self, run_request: RunRequest) -> UUID:
        """Submit a run request for execution on a quantum computer.

        This is called in :meth:`submit_circuits` and does not need to be called separately in normal usage.

        Args:
            run_request: Run request to be submitted for execution.

        Returns:
            ID for the created job. This ID is needed to query the job status and the execution results.

        """
        headers = {
            "Expect": "100-Continue",
            **self._default_headers(),
        }
        try:
            # Check if someone is trying to profile us with OpenTelemetry
            from opentelemetry import propagate

            propagate.inject(headers)
        except ImportError as _:
            # No OpenTelemetry, no problem
            pass

        if os.environ.get("IQM_CLIENT_DEBUG") == "1":
            print(f"\nIQM CLIENT DEBUGGING ENABLED\nSUBMITTING RUN REQUEST:\n{run_request}\n")

        # Use UTF-8 encoding for the JSON payload
        result = requests.post(
            # TODO SW-1434: Use station control client
            self._api.url(APIEndpoint.SUBMIT_JOB),
            data=run_request.model_dump_json(exclude_none=True).encode("utf-8"),
            headers=headers | {"Content-Type": "application/json; charset=UTF-8"},
            timeout=REQUESTS_TIMEOUT,
        )

        self._check_not_found_error(result)

        if result.status_code == 401:
            raise ClientAuthenticationError(f"Authentication failed: {result.text}")

        if 400 <= result.status_code < 500:
            raise ClientConfigurationError(f"Client configuration error: {result.text}")

        result.raise_for_status()

        try:
            job_id = UUID(result.json()["id"])
            return job_id
        except (json.decoder.JSONDecodeError, KeyError) as e:
            raise CircuitExecutionError(f"Invalid response: {result.text}, {e}") from e

    def get_run(self, job_id: UUID, *, timeout_secs: float = REQUESTS_TIMEOUT) -> RunResult:
        """Query the status and results of a submitted job.

        Args:
            job_id: ID of the job to query.
            timeout_secs: Network request timeout (seconds).

        Returns:
            Result of the job (can be pending).

        Raises:
            CircuitExecutionError: IQM server specific exceptions
            HTTPException: HTTP exceptions

        """
        status_response = self._get_request(
            APIEndpoint.GET_JOB_STATUS,
            (str(job_id),),
            timeout=timeout_secs,
        )
        status = status_response.json()
        if Status(status["status"]) not in Status.terminal_statuses():
            return RunResult.from_dict({"status": status["status"], "metadata": {}})

        result = self._get_request(APIEndpoint.GET_JOB_RESULT, (str(job_id),), timeout=timeout_secs, allow_errors=True)

        error_log_response = self._get_request(
            APIEndpoint.GET_JOB_ERROR_LOG, (str(job_id),), timeout=timeout_secs, allow_errors=True
        )
        if error_log_response.status_code == 200:
            error_log = error_log_response.json()
            if isinstance(error_log, dict) and "user_error_message" in error_log:
                error_message = error_log["user_error_message"]
            else:
                # backwards compatibility for older error_log format
                # TODO: remove when not needed anymore
                error_message = error_log_response.text
        else:
            error_message = None

        if result.status_code == 404:
            run_result = RunResult.from_dict({"status": status["status"], "message": error_message, "metadata": {}})
        else:
            result.raise_for_status()

            measurements = result.json()
            request_parameters = self._get_request(
                APIEndpoint.GET_JOB_REQUEST_PARAMETERS, (str(job_id),), timeout=timeout_secs, allow_errors=True
            ).json()
            calibration_set_id = self._get_request(
                APIEndpoint.GET_JOB_CALIBRATION_SET_ID, (str(job_id),), timeout=timeout_secs, allow_errors=True
            ).json()
            circuits_batch = self._get_request(
                APIEndpoint.GET_JOB_CIRCUITS_BATCH, (str(job_id),), timeout=timeout_secs, allow_errors=True
            ).json()
            timeline = self._get_request(
                APIEndpoint.GET_JOB_TIMELINE, (str(job_id),), timeout=timeout_secs, allow_errors=True
            ).json()

            run_result = RunResult.from_dict(
                {
                    "measurements": measurements,
                    "status": status["status"],
                    "message": error_message,
                    "metadata": {
                        "calibration_set_id": calibration_set_id,
                        "circuits_batch": circuits_batch,
                        "parameters": {
                            "shots": request_parameters["shots"],
                            "max_circuit_duration_over_t2": request_parameters.get(
                                "max_circuit_duration_over_t2", None
                            ),
                            "heralding_mode": request_parameters["heralding_mode"],
                            "move_validation_mode": request_parameters["move_validation_mode"],
                            "move_gate_frame_tracking_mode": request_parameters["move_gate_frame_tracking_mode"],
                        },
                        "timestamps": {datapoint["status"]: datapoint["timestamp"] for datapoint in timeline},
                    },
                    "warnings": status.get("warnings", []),
                }
            )

        if run_result.warnings:
            for warning in run_result.warnings:
                warnings.warn(warning)
        if run_result.status == Status.FAILED:
            raise CircuitExecutionError(run_result.message)
        return run_result

    def get_run_status(self, job_id: UUID, *, timeout_secs: float = REQUESTS_TIMEOUT) -> RunStatus:
        """Query the status of a submitted job.

        Args:
            job_id: ID of the job to query.
            timeout_secs: Network request timeout (seconds).

        Returns:
            Job status.

        Raises:
            CircuitExecutionError: IQM server specific exceptions
            HTTPException: HTTP exceptions

        """
        response = self._get_request(
            APIEndpoint.GET_JOB_STATUS,
            (str(job_id),),
            timeout=timeout_secs,
        )
        run_status = self._deserialize_response(response, RunStatus)

        if run_status.warnings:
            for warning in run_status.warnings:
                warnings.warn(warning)
        return run_status

    def wait_for_compilation(self, job_id: UUID, timeout_secs: float = DEFAULT_TIMEOUT_SECONDS) -> RunResult:
        """Poll results until a job is either compiled, pending execution, ready, failed, aborted, or timed out.

        Args:
            job_id: ID of the job to wait for.
            timeout_secs: How long to wait for a response before raising an APITimeoutError (seconds).

        Returns:
            Job result.

        Raises:
            APITimeoutError: time exceeded the set timeout

        """
        start_time = datetime.now()
        while (datetime.now() - start_time).total_seconds() < timeout_secs:
            status = self.get_run_status(job_id).status
            if status in Status.terminal_statuses() | {Status.PENDING_EXECUTION, Status.COMPILATION_ENDED}:
                return self.get_run(job_id)
            time.sleep(SECONDS_BETWEEN_CALLS)
        raise APITimeoutError(f"The job {job_id} compilation didn't finish in {timeout_secs} seconds.")

    def wait_for_results(self, job_id: UUID, timeout_secs: float = DEFAULT_TIMEOUT_SECONDS) -> RunResult:
        """Poll results until a job is either ready, failed, aborted, or timed out.

           Note that jobs handling on the server side is async and if we try to request the results
           right after submitting the job (which is usually the case)
           we will find the job is still pending at least for the first query.

        Args:
            job_id: ID of the job to wait for.
            timeout_secs: How long to wait for a response before raising an APITimeoutError (seconds).

        Returns:
            Job result.

        Raises:
            APITimeoutError: time exceeded the set timeout

        """
        start_time = datetime.now()
        while (datetime.now() - start_time).total_seconds() < timeout_secs:
            status = self.get_run_status(job_id).status
            if status in Status.terminal_statuses():
                return self.get_run(job_id)
            time.sleep(SECONDS_BETWEEN_CALLS)
        raise APITimeoutError(f"The job {job_id} didn't finish in {timeout_secs} seconds.")

    def abort_job(self, job_id: UUID, *, timeout_secs: float = REQUESTS_TIMEOUT) -> None:
        """Abort a job that was submitted for execution.

        Args:
            job_id: ID of the job to be aborted.
            timeout_secs: Network request timeout (seconds).

        Raises:
            JobAbortionError: aborting the job failed

        """
        result = requests.post(
            self._api.url(APIEndpoint.ABORT_JOB, str(job_id)),
            headers=self._default_headers(),
            timeout=timeout_secs,
        )
        if result.status_code not in (200, 204):  # CoCos returns 200, station-control 204.
            raise JobAbortionError(result.text)

    def get_static_quantum_architecture(self) -> StaticQuantumArchitecture:
        """Retrieve the static quantum architecture (SQA) from the server.

        Caches the result and returns it on later invocations.

        Returns:
            Static quantum architecture of the server.

        Raises:
            EndpointRequestError: did not understand the endpoint response
            ClientAuthenticationError: no valid authentication provided
            HTTPException: HTTP exceptions

        """
        if self._static_architecture:
            return self._static_architecture

        dut_label = self._get_dut_label()
        static_quantum_architecture = self._station_control.get_static_quantum_architecture(dut_label)
        self._static_architecture = StaticQuantumArchitecture(**static_quantum_architecture.model_dump())
        return self._static_architecture

    def get_quality_metric_set(self, calibration_set_id: UUID | None = None) -> QualityMetricSet:
        """Retrieve the latest quality metric set for the given calibration set from the server.

        Args:
            calibration_set_id: ID of the calibration set for which the quality metrics are returned.
                If ``None``, the current default calibration set is used.

        Returns:
            Requested quality metric set.

        Raises:
            EndpointRequestError: did not understand the endpoint response
            ClientAuthenticationError: no valid authentication provided
            HTTPException: HTTP exceptions

        """
        if isinstance(self._station_control, IqmServerClient):
            raise ValueError("'get_quality_metric_set' method is not supported for IqmServerClient.")

        if not calibration_set_id:
            quality_metrics = self._station_control.get_default_calibration_set_quality_metrics()
        else:
            quality_metrics = self._station_control.get_calibration_set_quality_metrics(calibration_set_id)

        calibration_set = quality_metrics.calibration_set

        return QualityMetricSet(
            **{  # type:ignore[arg-type]
                "calibration_set_id": calibration_set.observation_set_id,
                "calibration_set_dut_label": calibration_set.dut_label,
                "calibration_set_number_of_observations": len(calibration_set.observation_ids),
                "calibration_set_created_timestamp": str(calibration_set.created_timestamp.isoformat()),
                "calibration_set_end_timestamp": None
                if calibration_set.end_timestamp is None
                else str(calibration_set.end_timestamp.isoformat()),
                "calibration_set_is_invalid": calibration_set.invalid,
                "quality_metric_set_id": quality_metrics.observation_set_id,
                "quality_metric_set_dut_label": quality_metrics.dut_label,
                "quality_metric_set_created_timestamp": str(quality_metrics.created_timestamp.isoformat()),
                "quality_metric_set_end_timestamp": None
                if quality_metrics.end_timestamp is None
                else str(quality_metrics.end_timestamp.isoformat()),
                "quality_metric_set_is_invalid": quality_metrics.invalid,
                "metrics": {
                    observation.dut_field: {
                        "value": str(observation.value),
                        "unit": observation.unit,
                        "uncertainty": str(observation.uncertainty),
                        # created_timestamp is the interesting one, since observations are effectively immutable
                        "timestamp": str(observation.created_timestamp.isoformat()),
                    }
                    for observation in quality_metrics.observations
                    if not observation.invalid
                },
            }
        )

    def get_calibration_set(self, calibration_set_id: UUID | None = None) -> CalibrationSet:
        """Retrieve the given calibration set from the server.

        Args:
            calibration_set_id: ID of the calibration set to retrieve.
                If ``None``, the current default calibration set is retrieved.

        Returns:
            Requested calibration set.

        Raises:
            EndpointRequestError: did not understand the endpoint response
            ClientAuthenticationError: no valid authentication provided
            HTTPException: HTTP exceptions

        """

        def _observation_lite_to_json(obs: ObservationLite) -> dict[str, Any]:
            """Convert ObservationLite to JSON serializable dictionary."""
            json_dict = obs.model_dump()
            json_dict["created_timestamp"] = obs.created_timestamp.isoformat(timespec="microseconds")
            json_dict["modified_timestamp"] = obs.modified_timestamp.isoformat(timespec="microseconds")
            return json_dict

        if isinstance(self._station_control, IqmServerClient):
            raise ValueError("'get_calibration_set' method is not supported for IqmServerClient.")

        if not calibration_set_id:
            calibration_set = self._station_control.get_default_calibration_set()
        else:
            calibration_set = self._station_control.get_observation_set(calibration_set_id)

        observations = self._station_control.get_observation_set_observations(calibration_set.observation_set_id)

        return CalibrationSet(
            calibration_set_id=calibration_set.observation_set_id,
            calibration_set_dut_label=calibration_set.dut_label,  # type:ignore[arg-type]
            calibration_set_created_timestamp=str(calibration_set.created_timestamp.isoformat()),
            calibration_set_end_timestamp=""
            if calibration_set.end_timestamp is None
            else str(calibration_set.end_timestamp.isoformat()),
            calibration_set_is_invalid=calibration_set.invalid,
            observations={
                observation.dut_field: _observation_lite_to_json(observation) for observation in observations
            },
        )

    def get_dynamic_quantum_architecture(self, calibration_set_id: UUID | None = None) -> DynamicQuantumArchitecture:
        """Retrieve the dynamic quantum architecture (DQA) for the given calibration set from the server.

        Caches the result and returns the same result on later invocations, unless ``calibration_set_id`` is ``None``.
        If ``calibration_set_id`` is ``None``, always retrieves the result from the server because the default
        calibration set may have changed.

        Args:
            calibration_set_id: ID of the calibration set for which the DQA is retrieved.
                If ``None``, use current default calibration set on the server.

        Returns:
            Dynamic quantum architecture corresponding to the given calibration set.

        Raises:
            EndpointRequestError: did not understand the endpoint response
            ClientAuthenticationError: no valid authentication provided
            HTTPException: HTTP exceptions

        """
        if calibration_set_id in self._dynamic_architectures:
            return self._dynamic_architectures[calibration_set_id]

        if not calibration_set_id:
            if isinstance(self._station_control, IqmServerClient):
                dut_label = self._get_dut_label()
                calibration_set_id = self._station_control.get_latest_calibration_set_id(dut_label)
            else:
                calibration_set_id = self._station_control.get_default_calibration_set().observation_set_id
        data = self._station_control.get_dynamic_quantum_architecture(calibration_set_id)
        dynamic_quantum_architecture = DynamicQuantumArchitecture(**data.model_dump())

        # Cache architecture so that later invocations do not need to query it again
        self._dynamic_architectures[dynamic_quantum_architecture.calibration_set_id] = dynamic_quantum_architecture
        return dynamic_quantum_architecture

    def get_feedback_groups(self) -> tuple[frozenset[str], ...]:
        """Retrieve groups of qubits that can receive real-time feedback signals from each other.

        Real-time feedback enables conditional gates such as `cc_prx`.
        Some hardware configurations support routing real-time feedback only between certain qubits.

        Returns:
            Feedback groups. Within a group, any qubit can receive real-time feedback from any other qubit in
                the same group. A qubit can belong to multiple groups.
                If there is only one group, there are no restrictions regarding feedback routing.

        Raises:
            EndpointRequestError: did not understand the endpoint response
            ClientAuthenticationError: no valid authentication provided
            HTTPException: HTTP exceptions

        """
        channel_properties = self._station_control.get_channel_properties()

        all_qubits = self.get_static_quantum_architecture().qubits
        groups: dict[str, set[str]] = {}
        # All qubits that can read from the same source belong to the same group.
        # A qubit may belong to multiple groups.
        for channel_name, properties in channel_properties.items():
            # Relying on naming convention because we don't have proper mapping available:
            qubit = channel_name.split("__")[0]
            if qubit not in all_qubits:
                continue
            if isinstance(properties, AWGProperties):
                for source in properties.fast_feedback_sources:
                    groups.setdefault(source, set()).add(qubit)
        # Merge identical groups
        unique_groups: set[frozenset[str]] = {frozenset(group) for group in groups.values()}
        # Sort by group size
        return tuple(sorted(unique_groups, key=len, reverse=True))

    def get_run_counts(self, job_id: UUID, *, timeout_secs: float = REQUESTS_TIMEOUT) -> RunCounts:
        """Query the counts of an executed job.

        Args:
            job_id: ID of the job to query.
            timeout_secs: Network request timeout (seconds).

        Returns:
            Measurement results of the job in histogram representation.

        Raises:
            EndpointRequestError: did not understand the endpoint response
            ClientAuthenticationError: no valid authentication provided
            HTTPException: HTTP exceptions

        """
        response = self._get_request(
            APIEndpoint.GET_JOB_COUNTS,
            (str(job_id),),
            timeout=timeout_secs,
        )
        return self._deserialize_response(response, RunCounts)

    def get_supported_client_libraries(self, timeout_secs: float = REQUESTS_TIMEOUT) -> dict[str, ClientLibrary] | None:
        """Retrieve information about supported client libraries from the server.

        Args:
            timeout_secs: Network request timeout (seconds).

        Returns:
            Mapping from library identifiers to their metadata.

        Raises:
            EndpointRequestError: did not understand the endpoint response
            ClientAuthenticationError: no valid authentication provided
            HTTPException: HTTP exceptions

        """
        # TODO: Remove "client-libraries" usage after using versioned URLs in station control
        #  Version incompatibility shouldn't be a problem after that anymore,
        #  so we can delete this "client-libraries" implementation and usage.
        response = requests.get(
            #  "/info/client-libraries" is implemented by Nginx so it won't work on locally running service.
            #  We will simply give warning in that case, so that IQMClient can be initialized also locally.
            #  "/station" is set by Nginx, so we will drop it to get the correct root for "/info/client-libraries".
            self._api.station_control_url.replace("/station", "") + "/info/client-libraries",
            headers=self._default_headers(),
            timeout=timeout_secs,
        )
        if response.status_code == HTTPStatus.NOT_FOUND:
            return None
        try:
            return ClientLibraryDict.validate_json(response.text)
        except ValidationError as e:
            raise EndpointRequestError(f"Invalid response: {response.text}, {e!r}") from e

    def close_auth_session(self) -> bool:
        """Terminate session with authentication server if there is one.

        Returns:
            True iff session was successfully closed.

        Raises:
            ClientAuthenticationError: logout failed
            ClientAuthenticationError: asked to close externally managed authentication session

        """
        return self._token_manager.close()

    def _default_headers(self) -> dict[str, str]:
        """Default headers for HTTP requests to the IQM server."""
        headers = {"User-Agent": self._signature}
        if bearer_token := self._token_manager.get_bearer_token():
            headers["Authorization"] = bearer_token
        return headers

    @staticmethod
    def _check_authentication_errors(result: requests.Response) -> None:
        """Raise ClientAuthenticationError with appropriate message if the authentication failed for some reason."""
        # for not strictly authenticated endpoints,
        # we need to handle 302 redirects to the auth server login page
        if result.history and any(
            response.status_code == 302 for response in result.history
        ):  # pragma: no cover (generators are broken in coverage)
            raise ClientAuthenticationError("Authentication is required.")
        if result.status_code == 401:
            raise ClientAuthenticationError(f"Authentication failed: {result.text}")

    def _check_not_found_error(self, response: requests.Response) -> None:
        """Raise HTTPError with appropriate message if ``response.status_code == 404``."""
        if response.status_code == 404:
            version_message = ""
            if (version_incompatibility_msg := self._check_versions()) is not None:
                version_message = (
                    f" This may be caused by the server version not supporting this endpoint. "
                    f"{version_incompatibility_msg}"
                )
            raise HTTPError(f"{response.url} not found.{version_message}", response=response)

    def _check_versions(self) -> str | None:
        """Check the client version against compatible client versions reported by server.

        Returns:
            Message about client incompatibility with the server if the versions are incompatible or if the
            compatibility could not be confirmed, ``None`` if they are compatible.

        """
        try:
            libraries = self.get_supported_client_libraries()
            if not libraries:
                return "Got 'Not found' response from server. Couldn't check version compatibility."
            compatible_iqm_client = libraries.get(
                "iqm-client",
                libraries.get("iqm_client"),
            )
            if compatible_iqm_client is None:
                return "Could not verify IQM Client compatibility with the server. You might encounter issues."
            min_version = parse(compatible_iqm_client.min)
            max_version = parse(compatible_iqm_client.max)
            client_version = parse(version("iqm-client"))
            if client_version < min_version or client_version >= max_version:
                return (
                    f"Your IQM Client version {client_version} was built for a different version of IQM Server. "
                    f"You might encounter issues. For the best experience, consider using a version "
                    f"of IQM Client that satisfies {min_version} <= iqm-client < {max_version}."
                )
            return None
        except Exception as e:
            # we don't want the version check to prevent usage of IQMClient in any situation
            check_error = e
        return f"Could not verify IQM Client compatibility with the server. You might encounter issues. {check_error}"

    @lru_cache(maxsize=1)
    def _get_dut_label(self) -> str:
        duts = self._station_control.get_duts()
        if len(duts) != 1:
            raise RuntimeError(f"Expected exactly 1 DUT, but got {len(duts)}.")
        return duts[0].label

    def _get_request(
        self,
        api_endpoint: APIEndpoint,
        endpoint_args: tuple[str, ...] = (),
        *,
        timeout: float,
        headers: dict | None = None,
        allow_errors: bool = False,
    ) -> requests.Response:
        """Make an HTTP GET request to an IQM server endpoint.

        Contains all the boilerplate code for making a simple GET request.

        Args:
            api_endpoint: API endpoint to GET.
            endpoint_args: Arguments for the endpoint.
            timeout: HTTP request timeout (in seconds).

        Returns:
            HTTP response to the request.

        Raises:
            ClientAuthenticationError: No valid authentication provided.
            HTTPError: Various HTTP exceptions.

        """
        url = self._api.url(api_endpoint, *endpoint_args)
        response = requests.get(
            url,
            headers=headers or self._default_headers(),
            timeout=timeout,
        )
        if not allow_errors:
            self._check_not_found_error(response)
            self._check_authentication_errors(response)
            response.raise_for_status()
        return response

    @staticmethod
    def _deserialize_response(
        response: requests.Response,
        model_class: type[T_BaseModel],
    ) -> T_BaseModel:
        """Deserialize a HTTP endpoint response.

        Args:
            response: HTTP response data.
            model_class: Pydantic model to deserialize the data into.

        Returns:
            Deserialized endpoint response.

        Raises:
            EndpointRequestError: Did not understand the endpoint response.

        """
        try:
            model = model_class.model_validate(response.json())
            # TODO this would be faster but MockJsonResponse.text in our unit tests cannot handle UUID
            # model = model_class.model_validate_json(response.text)
        except json.decoder.JSONDecodeError as e:
            raise EndpointRequestError(f"Invalid response: {response.text}, {e!r}") from e
        return model
