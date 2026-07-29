# Copyright 2024-2025 IQM
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

"""Pulse level access library for IQM's circuit-to-pulse compiler and Station Control API."""

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from typing import Any
import uuid

from iqm.iqm_server_client.iqm_server_client import IQMServerClientJob, StrUUIDOrDefault, _IQMServerClient
from iqm.iqm_server_client.models import JobData, JobStatus
import requests

from exa.common.errors.station_control_errors import NotFoundError
from exa.common.logger import init_loggers
from exa.common.qcm_data.chip_topology import ChipTopology
from iqm.cpc.compiler.compiler import (
    Compiler,
)
from iqm.cpc.compiler.post_process import (
    _STANDARD_CIRCUIT_POST_PROCESSING_STAGES,
    _STANDARD_POST_PROCESSING_STAGES,
    PullaData,
    construct_circuit_execution_results,
)
from iqm.cpc.compiler.standard_stages import (
    _STANDARD_CIRCUIT_STAGES,
    _STANDARD_FINAL_STAGES,
    _STANDARD_PULSE_STAGES,
)
from iqm.cpc.core.observation.observation_loading_rules import LatestFromStash, RuleType
from iqm.pulla.interface import (
    CHADRetrievalException,
    ChipLabelRetrievalException,
)
from iqm.pulse.quantum_ops import QuantumOp
from iqm.station_control.interface.models import JobExecutorStatus, ObservationLite, RunData, RunDefinition, SweepData

#    ██████  ██    ██ ██      ██       █████
#    ██   ██ ██    ██ ██      ██      ██   ██
#    ██████  ██    ██ ██      ██      ███████
#    ██      ██    ██ ██      ██      ██   ██
#    ██       ██████  ███████ ███████ ██   ██

logger = logging.getLogger(__name__)


@dataclass
class PullaStash:
    """Minimal stash that can be used with :class:`.LatestFromStash`."""

    observations: dict[str, ObservationLite]
    """Observations mapped to their dut fields."""

    def get_latest_observation(self, observation_name: str, **kwargs) -> ObservationLite | None:
        """Get the observation value if it exists, otherwise ``None``."""
        return self.observations.get(observation_name)


class Pulla:
    """Client providing pulse level access to IQM's quantum computers.

    Conceptually, represents a connection to a remote quantum computer.
    Can create a circuit-to-pulse compiler instance ready to be used with the connected quantum computer.

    Args:
        iqm_server_url: URL for accessing the IQM Server. Has to start with http or https.
        quantum_computer: ID or alias of the quantum computer to connect to, if the IQM Server
            instance controls more than one. If not given, uses the default one on the server.
        token: Long-lived authentication token in plain text format.
            If ``token`` is given no other user authentication parameters should be given.
        tokens_file: Path to a tokens file used for authentication.
            If ``tokens_file`` is given no other user authentication parameters should be given.
        client_signature: String that the client adds to User-Agent header of requests
            it sends to the server. The signature is appended to Pulla's own version
            information and is intended to carry additional version information,
            for example the version information of the caller.

    Alternatively, the arguments can also be given in environment variables
    :envvar:`IQM_SERVER_URL`, :envvar:`IQM_QUANTUM_COMPUTER`, :envvar:`IQM_TOKEN`, :envvar:`IQM_TOKENS_FILE`.
    Same combination restrictions apply for values given as environment variables as for the arguments.

    """

    def __init__(
        self,
        iqm_server_url: str | None = None,
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

        # Data needed for the compiler.
        self._station_control_settings = self._iqm_server_client.get_settings()
        self._chip_topology: ChipTopology = self.get_chip_topology()
        self._chip_label: str = self.get_chip_label()
        self._software_version_set_id = 0

    def get_standard_compiler(
        self,
        loading_rules: list[RuleType] | None = None,
        *,
        exa_style_pp: bool = True,
        controller_mapping: dict[str, dict[str, str]] | None = None,
        gate_definitions: dict[str, QuantumOp] | None = None,
    ) -> Compiler:
        """Create a compiler instance for the connected quantum computer.

        Args:
            loading_rules: Observation loading rules. If ``None``, will use the current default calibration set.
            exa_style_pp: Whether to do EXA-style dataset post-processing by default.
            controller_mapping: Dictionary that maps physical QPU component names to their device controller names.
                The dictionary is of the form: ``{<component_name>: {<operation_name>: <controller name>}}``,
                where operation is one of the following: "drive", "readout", "flux"
                (not all components have all operations supported).
                If None, use the default controller mapping for the connected quantum computer.
            gate_definitions: Names of quantum operations mapped to their definitions, see :class:`.QuantumOp`.
                If None, use the default quantum operations.

        Returns:
            The compiler object.

        """
        pp_stages = (
            deepcopy(_STANDARD_POST_PROCESSING_STAGES)
            if exa_style_pp
            else deepcopy(_STANDARD_CIRCUIT_POST_PROCESSING_STAGES)
        )
        loading_rules = loading_rules if loading_rules is not None else [LatestFromStash(self.get_calibration_stash())]
        return Compiler(
            dut_label=self.get_chip_label(),
            loading_rules=loading_rules,  # type:ignore[arg-type]
            chip_topology=self._chip_topology,
            software_version_set_id=self._software_version_set_id,
            station_control_settings=self._station_control_settings.copy(),
            component_mapping=None,
            controller_mapping=controller_mapping,
            gate_definitions=gate_definitions,
            circuit_stages=deepcopy(_STANDARD_CIRCUIT_STAGES),
            pulse_stages=deepcopy(_STANDARD_PULSE_STAGES),
            final_stages=deepcopy(_STANDARD_FINAL_STAGES),
            pp_stages=pp_stages,
        )

    def get_chip_label(self) -> str:
        """QPU label of the current quantum computer."""
        try:
            duts = self._iqm_server_client.get_duts()
        except requests.RequestException as e:
            raise ChipLabelRetrievalException(f"Failed to retrieve the chip label: {e}") from e

        if len(duts) != 1:
            raise ChipLabelRetrievalException(f"Expected exactly one chip label, but got {len(duts)}")
        return duts[0].label

    def get_chip_topology(self) -> ChipTopology:
        """Chip topology of the current quantum computer."""
        try:
            # TODO: Assume there is only one DUT for now, implement proper support for multi-DUT or delete the feature
            record = self._iqm_server_client.get_chip_design_records()[0]
        except Exception as e:
            raise CHADRetrievalException("Could not fetch chip design record") from e
        return ChipTopology.from_chip_design_record(record)

    def submit_playlist(
        self,
        run_definition: RunDefinition,
        *,
        context: dict[str, Any],
        use_timeslot: bool = False,
    ) -> "PullaJob":
        """Submit a run definition for execution."""
        run_definition.run_id = uuid.uuid4()
        run_definition.sweep_definition.sweep_id = uuid.uuid4()
        job_data = self._iqm_server_client.submit_run(run_definition, use_timeslot=use_timeslot)

        return PullaJob(
            data=job_data,
            _pulla=self,
            _context=deepcopy(context),
        )

    def get_job(self, job_id: uuid.UUID) -> JobData:
        """Get the current status and metadata of the job.

        Args:
            job_id: ID of the job to query.

        Returns:
            JobData object.

        """
        # TODO: This interface will change when Pulla is merged to IQMClient
        return self._iqm_server_client.get_job(job_id)

    def get_calibration_stash(self, calibration_set_id: StrUUIDOrDefault = "default") -> PullaStash:
        """Contents of a calibration set as a stash object."""
        try:
            calibration_set_observations = self._iqm_server_client.get_calibration_set(calibration_set_id).observations
        except NotFoundError:
            if calibration_set_id == "default":
                logger.warning("No default calibration set available. Will initialize an empty PullaStash.")
            else:
                warn = f"Calibration set with id={calibration_set_id} not found. Will initialize an empty PullaStash."
                logger.warning(warn)
            calibration_set_observations = []
        return PullaStash({observation.dut_field: observation for observation in calibration_set_observations})


@dataclass
class PullaJob(IQMServerClientJob):
    """Status and results of a pulse-level job."""

    _pulla: Pulla
    """Client instance used to create the job."""

    _context: dict[str, Any]
    """Final context object of the compiler run used to produce the sweep, contains information needed
    for processing the results."""

    _result: PullaData | None = None
    """Raw Pulla data."""

    @property
    def _iqm_server_client(self) -> _IQMServerClient:
        return self._pulla._iqm_server_client

    def result(self, compiler: Compiler | None = None) -> Any | None:
        """Get (and cache) the job result, if the job has completed.

        Args:
            compiler: Compiler for custom post-processing of the job results. If None, return
                normal circuit execution results.

        Returns:
            Results for the job, or None if the results are not available.

        """
        if not self._result:
            self.update()
            # TODO: What to do with the partial data?
            if self.status != JobStatus.COMPLETED:
                return None

            sweep_results = self._iqm_server_client.get_job_artifact_sweep_results(self.job_id)
            #  TODO: The current IQM Server interface can respond only with the RunDefinition object,
            #   which was the payload for the request. Refactor everything to use RunDefinition,
            #   or change IQM Server to respond with RunData.
            run_definition = self._iqm_server_client.get_submit_run_payload(self.job_id)
            run_data = RunData(
                run_id=run_definition.run_id,
                username=run_definition.username,
                experiment_name=run_definition.experiment_name,
                experiment_label=run_definition.experiment_label,
                options=run_definition.options,
                additional_run_properties=run_definition.additional_run_properties,
                software_version_set_id=run_definition.software_version_set_id,
                hard_sweeps=run_definition.hard_sweeps,
                components=run_definition.components,
                default_data_parameters=run_definition.default_data_parameters,
                default_sweep_parameters=run_definition.default_sweep_parameters,
                sweep_data=SweepData(
                    sweep_id=run_definition.sweep_definition.sweep_id,
                    dut_label=run_definition.sweep_definition.dut_label,
                    settings=run_definition.sweep_definition.settings,
                    sweeps=run_definition.sweep_definition.sweeps,
                    return_parameters=run_definition.sweep_definition.return_parameters,
                    # TODO: Set timestamps to datetime.now for now, we aren't using them in this context
                    #  We should either use RunDefinition, or make IQM Server to respond with RunData (TODO above).
                    created_timestamp=datetime.now(timezone.utc),
                    modified_timestamp=datetime.now(timezone.utc),
                    begin_timestamp=datetime.now(timezone.utc),
                    end_timestamp=datetime.now(timezone.utc),
                    job_status=JobExecutorStatus.READY,
                ),
                # TODO: Set timestamps to datetime.now for now, we aren't using them in this context
                #  We should either use RunDefinition, or make IQM Server to respond with RunData (TODO above).
                created_timestamp=datetime.now(timezone.utc),
                modified_timestamp=datetime.now(timezone.utc),
                begin_timestamp=datetime.now(timezone.utc),
                end_timestamp=datetime.now(timezone.utc),
            )
            self._result = PullaData(sweep_results=sweep_results, run_data=run_data)

        if not compiler:
            result = construct_circuit_execution_results(self._result, self._context)
        else:
            result, _ = compiler.post_process(self._result, self._context)  # type:ignore[arg-type, assignment]
        return result


init_loggers({"iqm": "INFO"})
