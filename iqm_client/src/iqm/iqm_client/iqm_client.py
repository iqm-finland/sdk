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

from dataclasses import dataclass
from functools import cache, lru_cache
import logging
import os
from typing import Any
from uuid import UUID
import warnings

from iqm.iqm_client.errors import (
    CircuitValidationError,
)
from iqm.iqm_client.models import CircuitCompilationOptions, CircuitJobParameters, validate_circuit
from iqm.iqm_client.validation import validate_circuit_instructions, validate_qubit_mapping
from iqm.iqm_server_client.iqm_server_client import (
    DEFAULT_TIMEOUT_SECONDS,  # noqa: F401
    IQMServerClientJob,
    StrUUIDOrDefault,
    _IQMServerClient,
)
from iqm.iqm_server_client.models import (
    CalibrationSet,
    JobStatus,
    QualityMetricSet,
)
from iqm.models.channel_properties import AWGProperties

from iqm.station_control.client.qon import ObservationFinder
from iqm.station_control.interface.models import (
    CircuitBatch,
    CircuitMeasurementCountsBatch,
    CircuitMeasurementResults,  # noqa: F401
    CircuitMeasurementResultsBatch,
    DynamicQuantumArchitecture,
    QIRCode,
    QubitMapping,
    RunRequest,
    StaticQuantumArchitecture,
)
from iqm.station_control.interface.models.circuit import _Circuit

logger = logging.getLogger(__name__)


class IQMClient:
    """Provides access to IQM quantum computers, enabling quantum circuit execution with
    the selected quantum computer.

    Args:
        iqm_server_url: URL for accessing the IQM Server. Has to start with http or https.
        quantum_computer: ID or alias of the quantum computer to connect to, if the IQM Server
            instance controls more than one.
        token: Long-lived authentication token in plain text format.
            If ``token`` is given no other user authentication parameters should be given.
        tokens_file: Path to a tokens file used for authentication.
            If ``tokens_file`` is given no other user authentication parameters should be given.
        client_signature: String that IQMClient adds to User-Agent header of requests
            it sends to the server. The signature is appended to IQMClient's own version
            information and is intended to carry additional version information,
            for example the version information of the caller.

    Alternatively, the user authentication related keyword arguments can also be given in
    environment variables :envvar:`IQM_TOKEN`, :envvar:`IQM_TOKENS_FILE`.

    All parameters must be given either as keyword arguments or as environment variables.
    Same combination restrictions apply for values given as environment variables as for
    keyword arguments.

    """

    def __init__(
        self,
        iqm_server_url: str,
        *,
        quantum_computer: str | None = None,
        token: str | None = None,
        tokens_file: str | None = None,
        client_signature: str | None = None,
    ):
        self._iqm_server_client = _IQMServerClient(
            iqm_server_url=iqm_server_url,
            token=token,
            tokens_file=tokens_file,
            client_signature=client_signature,
            quantum_computer=quantum_computer,
        )
        self._dynamic_quantum_architectures: dict[UUID, DynamicQuantumArchitecture] = {}

    def get_health(self) -> dict[str, Any]:
        """Status of the quantum computer."""
        return self._iqm_server_client.get_health()

    def get_about(self) -> dict[str, Any]:
        """Information about the quantum computer."""
        return self._iqm_server_client.get_about()

    def submit_circuits(
        self,
        circuits: CircuitBatch,
        *,
        qubit_mapping: QubitMapping | None = None,
        calibration_set_id: UUID | None = None,
        shots: int = 1,
        options: CircuitCompilationOptions | None = None,
        use_timeslot: bool = False,
    ) -> CircuitJob:
        """Submit a batch of quantum circuits for execution on a quantum computer.

        Args:
            circuits: Circuits to be executed.
            qubit_mapping: Mapping of logical qubit names to physical qubit names.
                Can be set to ``None`` if all ``circuits`` already use physical qubit names.
                Note that the ``qubit_mapping`` is used for all ``circuits``.
            calibration_set_id: ID of the calibration set to use, or ``None`` to use the current default calibration.
            shots: Number of times ``circuits`` are executed. Must be greater than zero.
            options: Various discrete options for compiling quantum circuits to instruction schedules.
            use_timeslot: Submits the job to the timeslot queue if set to ``True``. If set to ``False``,
                the job is submitted to the normal on-demand queue.

        Returns:
            Job object, containing the ID for the created job.
            This ID is needed to query the job status and the execution results.
            Alternatively you can use the methods of the job object.

        """
        run_request = self.create_run_request(
            circuits=circuits,
            qubit_mapping=qubit_mapping,
            calibration_set_id=calibration_set_id,
            shots=shots,
            options=options,
        )
        job = self.submit_run_request(run_request, use_timeslot=use_timeslot)
        return job

    def create_run_request(
        self,
        circuits: CircuitBatch,
        *,
        qubit_mapping: QubitMapping | None = None,
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
                if isinstance(circuit, _Circuit):
                    raise CircuitValidationError(f"Circuit {i}: obsolete circuit type.")
                if isinstance(circuit, QIRCode):
                    # do not validate
                    continue
                # validate the circuit against the static information in iqm.iqm_client.models._SUPPORTED_OPERATIONS
                validate_circuit(circuit)
            except ValueError as e:
                raise CircuitValidationError(f"The circuit at index {i} failed the validation: {e}").with_traceback(
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

        return RunRequest(
            qubit_mapping=qubit_mapping,
            circuits=circuits,
            calibration_set_id=calibration_set_id,
            shots=shots,
            max_circuit_duration_over_t2=options.max_circuit_duration_over_t2,
            heralding_mode=options.heralding_mode,
            move_gate_validation=options.move_gate_validation,
            move_gate_frame_tracking=options.move_gate_frame_tracking,
            active_reset_cycles=options.active_reset_cycles,
            dd_mode=options.dd_mode,
            dd_strategy=options.dd_strategy,
        )

    def submit_run_request(self, run_request: RunRequest, use_timeslot: bool = False) -> CircuitJob:
        """Submit a run request for execution on a quantum computer.

        This is called in :meth:`submit_circuits` and does not need to be called separately in normal usage.

        Args:
            run_request: Run request to be submitted for execution.
            use_timeslot: Submits the job to the timeslot queue if set to ``True``. If set to ``False``,
                the job is submitted to the normal on-demand queue.

        Returns:
            Job object, containing the ID for the created job.
            This ID is needed to query the job status and the execution results.
            Alternatively you can use the methods of the job object.

        """
        if os.environ.get("IQM_CLIENT_DEBUG") == "1":
            print(f"\nIQM CLIENT DEBUGGING ENABLED\nSUBMITTING RUN REQUEST:\n{run_request}\n")

        job_data = self._iqm_server_client.submit_circuits(run_request, use_timeslot=use_timeslot)
        return CircuitJob(_iqm_client=self, data=job_data)

    def get_job(self, job_id: UUID) -> CircuitJob:
        """Query the status and results of a submitted job.

        Args:
            job_id: ID of the job to query.

        Returns:
            Status of the job, can be used to query the results if the job has finished.

        """
        job_data = self._iqm_server_client.get_job(job_id)
        for message in job_data.messages:
            warnings.warn(str(message))
        for error in job_data.errors:
            warnings.warn(str(error))

        return CircuitJob(_iqm_client=self, data=job_data)

    def cancel_job(self, job_id: UUID) -> None:
        """Cancel a job that was submitted for execution.

        Args:
            job_id: ID of the job to be canceled.

        """
        self._iqm_server_client.cancel_job(job_id)

    def delete_job(self, job_id: UUID) -> None:
        self._iqm_server_client.delete_job(job_id)

    @cache
    def get_static_quantum_architecture(self) -> StaticQuantumArchitecture:
        """Retrieve the static quantum architecture (SQA) from the server.

        Caches the result and returns it on later invocations.

        Returns:
            Static quantum architecture of the server.

        Raises:
            ClientAuthenticationError: no valid authentication provided

        """
        self._get_dut_label()  # Called just to make sure that there will be only one DUT available
        static_quantum_architectures = self._iqm_server_client.get_static_quantum_architectures()
        return static_quantum_architectures[0]

    def get_quality_metric_set(self, calibration_set_id: UUID | None = None) -> QualityMetricSet:
        """Retrieve the latest quality metric set for the given calibration set from the server.

        Args:
            calibration_set_id: ID of the calibration set for which the quality metrics are returned.
                If ``None``, the current default calibration set is used.

        Returns:
            Requested quality metric set.

        Raises:
            ClientAuthenticationError: no valid authentication provided

        """
        _calibration_set_id: StrUUIDOrDefault = calibration_set_id if calibration_set_id is not None else "default"
        quality_metrics = self._iqm_server_client.get_calibration_set_quality_metric_set(_calibration_set_id)
        return QualityMetricSet(**quality_metrics.model_dump())

    def get_calibration_set(self, calibration_set_id: UUID | None = None) -> CalibrationSet:
        """Retrieve the given calibration set from the server.

        Args:
            calibration_set_id: ID of the calibration set to retrieve.
                If ``None``, the current default calibration set is retrieved.

        Returns:
            Requested calibration set.

        Raises:
            ClientAuthenticationError: no valid authentication provided

        """
        _calibration_set_id: StrUUIDOrDefault = calibration_set_id if calibration_set_id is not None else "default"
        calibration_set = self._iqm_server_client.get_calibration_set(_calibration_set_id)
        return calibration_set

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
            ClientAuthenticationError: no valid authentication provided

        """
        if calibration_set_id in self._dynamic_quantum_architectures:
            return self._dynamic_quantum_architectures[calibration_set_id]

        _calibration_set_id: StrUUIDOrDefault = calibration_set_id if calibration_set_id is not None else "default"
        dynamic_quantum_architecture = self._iqm_server_client.get_dynamic_quantum_architecture(_calibration_set_id)

        self._dynamic_quantum_architectures[dynamic_quantum_architecture.calibration_set_id] = (
            dynamic_quantum_architecture
        )
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
            ClientAuthenticationError: no valid authentication provided

        """
        channel_properties = self._iqm_server_client.get_channel_properties()

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

    def get_job_measurement_counts(self, job_id: UUID) -> CircuitMeasurementCountsBatch:
        """Query the measurement counts of an executed job.

        Args:
            job_id: ID of the job to query.

        Returns:
            For each circuit in the batch, measurement results in histogram representation.

        Raises:
            ClientAuthenticationError: no valid authentication provided

        """
        return self._iqm_server_client.get_job_artifact_measurement_counts(job_id)

    def get_job_measurements(self, job_id: UUID) -> CircuitMeasurementResultsBatch:
        """Query the measurement results of an executed job.

        Args:
            job_id: ID of the job to query.

        Returns:
            For each circuit in the batch, the measurement results.

        """
        return self._iqm_server_client.get_job_artifact_measurements(job_id)

    def _get_submit_circuits_payload(self, job_id: UUID) -> RunRequest:
        """Get the original payload that created the given circuit job.

        Args:
            job_id: ID of the job to query.

        Returns:
            Payload of the original circuit job.

        """
        return self._iqm_server_client.get_submit_circuits_payload(job_id)

    @lru_cache(maxsize=1)
    def _get_dut_label(self) -> str:
        """Get the singular dut_label of the quantum computer, or raise an error."""
        duts = self._iqm_server_client.get_duts()
        if len(duts) != 1:
            raise RuntimeError(f"Expected exactly 1 DUT, but got {len(duts)}.")
        return duts[0].label

    def get_calibration_quality_metrics(self, calibration_set_id: UUID | None = None) -> ObservationFinder:
        """Retrieve the given calibration set and related quality metrics from the server.

        .. warning::

           This method is an experimental interface to the quality metrics and calibration data.
           The API may change considerably in the next versions *with no backwards compatibility*,
           including the API of the ObservationFinder class.

        Args:
            calibration_set_id: ID of the calibration set to retrieve.
                If ``None``, the current default calibration set is retrieved.

        Returns:
            Requested calibration set and related quality metrics in a searchable structure.

        Raises:
            ClientAuthenticationError: no valid authentication provided

        """
        logger.warning(
            "IQMClient.get_calibration_quality_metrics is an experimental method, and the API will likely change "
            "in the future with no backwards compatibility."
        )
        return self._get_calibration_quality_metrics(calibration_set_id)

    def _get_calibration_quality_metrics(self, calibration_set_id: UUID | None = None) -> ObservationFinder:
        """See :meth:`get_calibration_quality_metrics`."""
        _calibration_set_id: StrUUIDOrDefault = calibration_set_id if calibration_set_id is not None else "default"
        calibration_set = self._iqm_server_client.get_calibration_set(_calibration_set_id)
        quality_metrics = self._iqm_server_client.get_calibration_set_quality_metric_set(_calibration_set_id)
        return ObservationFinder(calibration_set.observations + quality_metrics.observations)


@dataclass
class CircuitJob(IQMServerClientJob):
    """Status and results of a quantum circuit execution job.

    If the job succeeded, :meth:`result` returns the output of the batch of circuits.
    """

    _iqm_client: IQMClient
    """Client instance used to create the job."""

    _result: CircuitMeasurementResultsBatch | None = None
    """If the job has finished successfully, the measurement results for the circuit(s).
    Populated by :meth:`result`
    """

    _circuits: CircuitBatch | None = None
    """Circuits batch submitted for execution. Populated by :meth:`payload`."""

    _parameters: CircuitJobParameters | None = None
    """Job parameters sent in the execution request. Populated by :meth:`payload`."""

    @property
    def _iqm_server_client(self) -> _IQMServerClient:
        return self._iqm_client._iqm_server_client

    def result(self) -> CircuitMeasurementResultsBatch | None:
        """Get (and cache) the job result, if the job has completed.

        Returns:
            Circuit measurement results for a completed job, or None if the results are not (yet?) available.

        """
        # TODO should we name this method something more specific like "measurements"?
        if not self._result:
            self.update()
            # if successful, get the results
            if self.status != JobStatus.COMPLETED:
                return None

            self._result = self._iqm_client.get_job_measurements(self.job_id)
            # TODO refactor RunRequest
            # Consider replacing CircuitCompilationOptions with CircuitJobParameters

            for message in self.data.messages:
                warnings.warn(f"{message.source}: {message.message}")

        return self._result

    def payload(self) -> tuple[CircuitBatch, CircuitJobParameters]:
        """Get the circuit job payload.

        Returns:
            Circuits sent for execution, circuit execution options used.

        """
        if not self._circuits or not self._parameters:
            run_request = self._iqm_client._get_submit_circuits_payload(self.job_id)
            # TODO: Remove "exclude" when deprecated computed fields are removed from the model
            # `move_validation_mode` and `move_gate_frame_tracking_mode` are deprecated since 2025-10-17
            run_request_dict = run_request.model_dump(exclude={"move_validation_mode", "move_gate_frame_tracking_mode"})
            self._circuits = run_request_dict.pop("circuits")
            self._parameters = CircuitJobParameters(**run_request_dict)

        return self._circuits, self._parameters
