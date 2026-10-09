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

from collections.abc import Sequence
from copy import deepcopy
from functools import cache, cached_property
import logging
import os
import pprint
from typing import TYPE_CHECKING, Any
from uuid import UUID
import warnings

from iqm.iqm_client.models import CircuitCompilationOptions, validate_circuit
from iqm.iqm_client.validation import validate_circuit_instructions, validate_qubit_mapping
from iqm.iqm_server_client.iqm_server_client import IQMServerClient
from iqm.iqm_server_client.iqm_server_executor import IQMServerExecutor
from iqm.iqm_server_client.iqm_server_storage import IQMServerStorage
from iqm.models.channel_properties import AWGProperties

from exa.common.errors.iqm_error import CircuitValidationError, InternalSystemError, NotFoundError
from exa.common.qcm_data.chip_topology import ChipTopology
from iqm.cpc.compiler.compiler import Compiler
from iqm.cpc.compiler.post_process import (
    _STANDARD_CIRCUIT_POST_PROCESSING_STAGES,
    _STANDARD_POST_PROCESSING_STAGES,
)
from iqm.cpc.compiler.standard_stages import (
    _STANDARD_CIRCUIT_STAGES,
    _STANDARD_FINAL_STAGES,
    _STANDARD_PULSE_STAGES,
)
from iqm.cpc.core.observation.observation_handler import ObservationHandler
from iqm.cpc.core.observation.observation_loading_rules import LatestFromStash, _ObservationSetStash
from iqm.station_control.client.job_trackers import CircuitJobTracker, JobTracker, RunJobTracker
from iqm.station_control.client.qon import ObservationFinder
from iqm.station_control.interface.models import (
    CircuitBatch,
    DynamicQuantumArchitecture,
    QIRCode,
    QubitMapping,
    StaticQuantumArchitecture,
)
from iqm.station_control.interface.models.circuit import (
    CircuitJobDefinition,
    CircuitMeasurementCounts,
    CircuitMeasurementResults,
    _Circuit,
)
from iqm.station_control.interface.models.observation_set import CalibrationSet, QualityMetricSet
from iqm.station_control.interface.models.type_aliases import StrUUIDOrDefault

if TYPE_CHECKING:
    from iqm.cpc.core.observation.observation_loading_rules import RuleType
    from iqm.pulse.quantum_ops import QuantumOp
    from iqm.station_control.interface.models.run import RunDefinition

logger = logging.getLogger(__name__)


class IQMClient:
    """Provides access to IQM quantum computers, enabling quantum circuit and pulse schedule execution.

    IQMClient represents a connection to a quantum computer (real or simulated), and can be used to
    execute two kinds of jobs:

    * batches of quantum circuits (circuit-level jobs), and
    * playlists of control pulse schedules (pulse-level jobs).

    Args:
        executor: IQMServerExecutor instance mapping to backend services.
        storage: IQMServerStorage instance mapping to backend data repositories.

    """

    # TODO (Marko): Review IQMClient INTEGRATION_GUIDE.rst and update for the latest changes

    def __init__(
        self,
        *,
        executor: IQMServerExecutor,
        storage: IQMServerStorage,
    ):
        self._executor = executor
        self._storage = storage
        self._dynamic_quantum_architectures: dict[UUID, DynamicQuantumArchitecture] = {}
        self._software_version_set_id = 0  # TODO

    @staticmethod
    def from_url(
        iqm_server_url: str | None = None,
        *,
        quantum_computer: str | None = None,
        token: str | None = None,
        tokens_file: str | None = None,
        client_signature: str | None = None,
    ) -> IQMClient:
        """Utility constructor using :class:`.IQMServerClient`.

        Args:
            iqm_server_url: URL for accessing the IQM Server. Has to start with http or https.
            quantum_computer: ID or alias of the quantum computer to connect to, if the IQM Server
                instance controls more than one. If not given, uses the default one on the server.
            token: Long-lived authentication token in plain text format.
                If ``token`` is given no other user authentication parameters should be given.
            tokens_file: Path to a tokens file used for authentication.
                If ``tokens_file`` is given no other user authentication parameters should be given.
            client_signature: String that IQMClient adds to User-Agent header of requests
                it sends to the server. The signature is appended to IQMClient's own version
                information and is intended to carry additional version information,
                for example the version information of the caller.

        Alternatively, the arguments can also be given in environment variables
        :envvar:`IQM_SERVER_URL`, :envvar:`IQM_QUANTUM_COMPUTER`, :envvar:`IQM_TOKEN`, :envvar:`IQM_TOKENS_FILE`.
        Same combination restrictions apply for values given as environment variables as for the arguments.

        """
        client = IQMServerClient(
            iqm_server_url=iqm_server_url,
            quantum_computer=quantum_computer,
            token=token,
            tokens_file=tokens_file,
            client_signature=client_signature,
        )
        return IQMClient(
            executor=IQMServerExecutor(client=client),
            storage=IQMServerStorage(client=client),
        )

    def get_health(self) -> dict[str, Any]:
        """Health status of the quantum computer."""
        return self._executor.client.get_health()

    def get_about(self) -> dict[str, Any]:
        """Information about the quantum computer."""
        return self._executor.client.get_about()

    def _debug_info(self) -> None:
        """Print information about the client and server versions, and the Python environment.

        Please include the output of this method in your bug reports.
        """
        pprint.pp(self._executor.client._debug_info())

    def submit_circuits(
        self,
        circuits: CircuitBatch,
        *,
        qubit_mapping: QubitMapping | None = None,
        calibration_set_id: UUID | None = None,
        shots: int = 1,
        options: CircuitCompilationOptions | None = None,
        use_timeslot: bool = False,
    ) -> CircuitJobTracker:
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
            You can query the job status and the execution results using the methods of this object.

        """
        circuit_job_definition = self.create_circuit_job_definition(
            circuits=circuits,
            qubit_mapping=qubit_mapping,
            calibration_set_id=calibration_set_id,
            shots=shots,
            options=options,
        )
        circuit_job_tracker = self.submit_circuit_job(circuit_job_definition, use_timeslot=use_timeslot)
        return circuit_job_tracker

    def create_circuit_job_definition(
        self,
        circuits: CircuitBatch,
        *,
        qubit_mapping: QubitMapping | None = None,
        calibration_set_id: UUID | None = None,
        shots: int = 1,
        options: CircuitCompilationOptions | None = None,
    ) -> CircuitJobDefinition:
        """Create a circuit execution job definition without sending it to the server.

        This is called in :meth:`submit_circuits` and does not need to be called separately in normal usage.

        Can be used to inspect the circuit job definition that would be submitted by :meth:`submit_circuit_job`,
        without actually submitting it for execution.

        Args:
            circuits: Circuits to be executed.
            qubit_mapping: Mapping of logical qubit names to physical qubit names.
                Can be set to ``None`` if all ``circuits`` already use physical qubit names.
                Note that the ``qubit_mapping`` is used for all ``circuits``.
            calibration_set_id: ID of the calibration set to use, or ``None`` to use the current default calibration.
            shots: Number of times ``circuits`` are executed. Must be greater than zero.
            options: Various discrete options for compiling quantum circuits to instruction schedules.

        Returns:
            CircuitJobDefinition that would be submitted by equivalent call to :meth:`submit_circuits`.

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

        return CircuitJobDefinition(
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

    def submit_circuit_job(
        self,
        circuit_job_definition: CircuitJobDefinition,
        *,
        use_timeslot: bool = False,
    ) -> CircuitJobTracker:
        """Submit a circuit job to execute a circuit batch to a quantum computer.

        This is called in :meth:`submit_circuits` and does not need to be called separately in normal usage.

        Args:
            circuit_job_definition: Circuit job definition to be submitted for execution.
            use_timeslot: Submits the job to the timeslot queue if set to ``True``. If set to ``False``,
                the job is submitted to the normal on-demand queue.

        Returns:
            CircuitJobTracker object, can be used to query the results.

        """
        if os.environ.get("IQM_CLIENT_DEBUG") == "1":
            print(f"\nIQM CLIENT DEBUGGING ENABLED\nSUBMITTING CIRCUIT JOB DEFINITION:\n{circuit_job_definition}\n")

        return self._executor.submit_job(circuit_job_definition, use_timeslot=use_timeslot)

    def get_job(self, job_id: UUID) -> JobTracker:
        """Build a new job tracker object for a submitted job.

        Can be used e.g. when you wish to query the results of a job submitted in a different Python session.

        Args:
            job_id: ID of the job to query.

        Returns:
            JobTracker object, can be used to query the results.

        """
        job_data = self._executor.get_job(job_id)
        for message in job_data.messages:
            warnings.warn(str(message))
        for error in job_data.errors:
            warnings.warn(str(error))

        if job_data.type == "circuit":
            return CircuitJobTracker(job_data=job_data, _executor=self._executor)
        elif job_data.type == "run":
            return RunJobTracker(job_data=job_data, _executor=self._executor)
        else:
            raise InternalSystemError(f"Unknown job type: {job_data.type}")

    def cancel_job(self, job_id: UUID) -> None:
        """Cancel a job that has been submitted for execution.

        A canceled job will remain in the server database, but it will not be executed.
        If the job is currently being executed, it is interrupted.
        If the job was already executed (or failed), it will remain in its current terminal state.

        Args:
            job_id: ID of the job to be canceled.

        """
        self._executor.cancel_job(job_id)

    def delete_job(self, job_id: UUID) -> None:
        """Delete a job that has been submitted for execution.

        Works like :meth:`cancel_job`, but also removes the job from the IQM Server database.

        .. warning:: There is no way to recover a deleted job.

        Args:
            job_id: ID of the job to be deleted.

        """
        self._executor.delete_job(job_id)

    @cache
    def get_static_quantum_architecture(self) -> StaticQuantumArchitecture:
        """Retrieve the static quantum architecture (SQA) from the server.

        Caches the result and returns it on later invocations.

        Returns:
            Static quantum architecture of the executor.

        """
        return self._executor.get_static_quantum_architecture()

    def get_quality_metric_set(self, calibration_set_id: UUID | None = None) -> QualityMetricSet:
        """Retrieve the latest quality metric set for the given calibration set from the server.

        Args:
            calibration_set_id: ID of the calibration set for which the quality metrics are returned.
                If ``None``, the current default calibration set is used.

        Returns:
            Requested quality metric set.

        """
        _calibration_set_id: StrUUIDOrDefault = calibration_set_id if calibration_set_id is not None else "default"
        return self._storage.get_calibration_set_quality_metric_set(_calibration_set_id)

    def get_calibration_set(self, calibration_set_id: UUID | None = None) -> CalibrationSet:
        """Retrieve the given calibration set from the server.

        Args:
            calibration_set_id: ID of the calibration set to retrieve.
                If ``None``, the current default calibration set is retrieved.

        Returns:
            Requested calibration set.

        """
        _calibration_set_id: StrUUIDOrDefault = calibration_set_id if calibration_set_id is not None else "default"
        return self._storage.get_calibration_set(_calibration_set_id)

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

        """
        if calibration_set_id in self._dynamic_quantum_architectures:
            return self._dynamic_quantum_architectures[calibration_set_id]

        _calibration_set_id: StrUUIDOrDefault = calibration_set_id if calibration_set_id is not None else "default"
        dynamic_quantum_architecture = self._storage.get_dynamic_quantum_architecture(_calibration_set_id)

        self._dynamic_quantum_architectures[dynamic_quantum_architecture.calibration_set_id] = (
            dynamic_quantum_architecture
        )
        return dynamic_quantum_architecture

    @cache
    def get_feedback_groups(self) -> tuple[frozenset[str], ...]:
        """Retrieve groups of qubits that can receive real-time feedback signals from each other.

        Real-time feedback enables conditional gates such as `cc_prx`.
        Some hardware configurations support routing real-time feedback only between certain qubits.

        Returns:
            Feedback groups. Within a group, any qubit can receive real-time feedback from any other qubit in
                the same group. A qubit can belong to multiple groups.
                If there is only one group, there are no restrictions regarding feedback routing.

        """
        channel_properties = self._executor.get_channel_properties()

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

    def get_job_measurement_counts(self, job_id: UUID) -> list[CircuitMeasurementCounts]:
        """Query the measurement counts of an executed job.

        Args:
            job_id: ID of the job to query.

        Returns:
            For each circuit in the batch, measurement results in histogram representation.

        """
        return self._executor.get_artifact_measurement_counts(job_id)

    def get_job_measurements(self, job_id: UUID) -> list[CircuitMeasurementResults]:
        """Query the measurement results of an executed job.

        Args:
            job_id: ID of the job to query.

        Returns:
            For each circuit in the batch, the measurement results.

        """
        return self._executor.get_artifact_measurements(job_id)

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

        """
        logger.warning(
            "IQMClient.get_calibration_quality_metrics is an experimental method, and the API will likely change "
            "in the future with no backwards compatibility."
        )
        return self._get_calibration_quality_metrics(calibration_set_id)

    def _get_calibration_quality_metrics(self, calibration_set_id: UUID | None = None) -> ObservationFinder:
        """See :meth:`get_calibration_quality_metrics`."""
        _calibration_set_id: StrUUIDOrDefault = calibration_set_id if calibration_set_id is not None else "default"
        calibration_set = self._storage.get_calibration_set(_calibration_set_id)
        quality_metrics = self._storage.get_calibration_set_quality_metric_set(_calibration_set_id)
        return ObservationFinder(calibration_set.observations + quality_metrics.observations)

    @cached_property
    def quantum_computer_name(self) -> str:
        """Name of the quantum computer this client is connected to."""
        quantum_computer = self._executor.client.quantum_computer
        if quantum_computer == "default":
            quantum_computers = self._executor.client._get_quantum_computers()
            quantum_computer = next(qc for qc in quantum_computers if qc.alias == quantum_computer).display_name
        return quantum_computer

    def get_standard_compiler(
        self,
        loading_rules: Sequence[RuleType] | None = None,
        *,
        exa_style_pp: bool = True,
        controller_mapping: dict[str, dict[str, str]] | None = None,
        gate_definitions: dict[str, QuantumOp] | None = None,
    ) -> Compiler:
        """Create a compiler instance for the connected quantum computer.

        .. note::

           If you only intend to execute circuit-level jobs, you do not need to use this method.

        The Compiler enables pulse-level access to the quantum computer. It is used for compiling

        * quantum circuits to :class:`TimeBoxes <~iqm.pulse.TimeBox>`
        * TimeBoxes to abstract hardware instruction :class:`Schedules <~iqm.pulse.Schedule>`
        * Schedules to a :class:`Playlists <~iqm.pulse.Playlist>`, which can be submitted for
          execution using :meth:`submit_playlist`. TODO explain RunDefinition?

        Args:
            loading_rules: Observation loading rules. If ``None``, will use the current default calibration set.
            exa_style_pp: Whether to do EXA-style dataset post-processing by default.
            controller_mapping: Dictionary that maps physical QPU component names to their device controller names.
                The dictionary is of the form: ``{<component_name>: {<operation_name>: <controller name>}}``,
                where operation is one of the following: "drive", "readout", "flux"
                (not all components have all operations supported).
                If None, use the default controller mapping for the connected quantum computer.
            gate_definitions: Names of quantum operations mapped to their definitions, see :class:`.QuantumOp`.
                If None, use the default quantum operations from :mod:`iqm.pulse`.

        Returns:
            The compiler object.

        """
        # Data needed for the compiler.
        settings = self._executor.get_settings()
        pp_stages = (
            deepcopy(_STANDARD_POST_PROCESSING_STAGES)
            if exa_style_pp
            else deepcopy(_STANDARD_CIRCUIT_POST_PROCESSING_STAGES)
        )
        load_rules = loading_rules if loading_rules is not None else [LatestFromStash(self.get_calibration_set_stash())]
        return Compiler(
            dut_label=self.get_chip_label(),
            chip_topology=self.get_chip_topology(),
            software_version_set_id=self._software_version_set_id,
            station_control_settings=settings,
            observation_handler=ObservationHandler(
                [],
                dut_label=self.get_chip_label(),
                default_storage=self._storage,
                load_rules=load_rules,
            ),
            component_mapping=None,
            controller_mapping=controller_mapping,
            gate_definitions=gate_definitions,
            circuit_stages=deepcopy(_STANDARD_CIRCUIT_STAGES),
            pulse_stages=deepcopy(_STANDARD_PULSE_STAGES),
            final_stages=deepcopy(_STANDARD_FINAL_STAGES),
            pp_stages=pp_stages,
        )

    def submit_playlist(
        self,
        run_definition: RunDefinition,
        *,
        context: dict[str, Any],
        use_timeslot: bool = False,
    ) -> RunJobTracker:
        """Submit a run definition for execution."""
        job_tracker = self._executor.submit_job(run_definition, use_timeslot=use_timeslot)
        # TODO (Marko): Consider how to pass the context for the job tracker,
        #  we shouldn't edit private fields like this.
        job_tracker._context = context
        return job_tracker

    @cache
    def get_chip_label(self) -> str:
        """QPU label of the current quantum computer."""
        try:
            return self._executor.get_dut_label()
        except Exception as e:
            raise NotFoundError("Could not fetch chip design record") from e

    @cache
    def get_chip_topology(self) -> ChipTopology:
        """Chip topology of the current quantum computer.

        Describes the QPU components and their connectivity.
        Similar to :meth:`get_static_quantum_architecture`, but provides more detailed information.
        """
        try:
            chip_design_record = self._executor.get_chip_design_record()
        except Exception as e:
            raise NotFoundError("Could not fetch chip design record") from e
        return ChipTopology.from_chip_design_record(chip_design_record)

    def get_calibration_set_stash(self, calibration_set_id: StrUUIDOrDefault = "default") -> _ObservationSetStash:
        """Contents of a calibration set in a stash adapter."""
        try:
            calibration_set_observations = self._storage.get_calibration_set(calibration_set_id).observations
        except NotFoundError:
            if calibration_set_id == "default":
                logger.warning("No default calibration set available. Will initialize an empty stash.")
            else:
                warn = f"Calibration set with id={calibration_set_id} not found. Will initialize an empty stash."
                logger.warning(warn)
            calibration_set_observations = []
        return _ObservationSetStash(
            {observation.dut_field: observation for observation in calibration_set_observations}
        )
