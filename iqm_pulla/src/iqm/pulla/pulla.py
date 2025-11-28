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
"""Pulse level access to IQM quantum computers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from importlib.metadata import version
import logging
from typing import Any
import uuid

from iqm.iqm_client.iqm_client import IQMServerClientJob
from iqm.iqm_server_client.iqm_server_client import _IQMServerClient
from iqm.iqm_server_client.models import JobStatus
import requests

from exa.common.data.setting_node import SettingNode
from exa.common.logger import init_loggers
from exa.common.qcm_data.chip_topology import ChipTopology
from iqm.cpc.compiler.compiler import (
    STANDARD_CIRCUIT_EXECUTION_OPTIONS,
    STANDARD_CIRCUIT_EXECUTION_OPTIONS_DICT,
    Compiler,
)
from iqm.cpc.compiler.standard_stages import get_standard_stages
from iqm.cpc.interface.compiler import CircuitExecutionOptions
from iqm.pulla.calibration import CalibrationDataProvider
from iqm.pulla.interface import (
    CalibrationSetValues,
    CHADRetrievalException,
    ChipLabelRetrievalException,
    SettingsRetrievalException,
)
from iqm.pulla.utils import extract_readout_controller_result_names, map_sweep_results_to_logical_qubits
from iqm.pulse.playlist.channel import ChannelProperties, get_channel_properties_from_station_settings
from iqm.pulse.playlist.playlist import Playlist
from iqm.station_control.interface.models import CircuitMeasurementResultsBatch, SweepDefinition

#    ██████  ██    ██ ██      ██       █████
#    ██   ██ ██    ██ ██      ██      ██   ██
#    ██████  ██    ██ ██      ██      ███████
#    ██      ██    ██ ██      ██      ██   ██
#    ██       ██████  ███████ ███████ ██   ██

logger = logging.getLogger(__name__)
init_loggers({"iqm": "INFO"})


class Pulla:
    """Pulse level access to IQM quantum computers.

    Each instance of this class represents a connection to a remote quantum computer.
    Can create a :class:`~iqm.cpc.compiler.compiler.Compiler` instance ready to be used with
    the connected quantum computer.

    Args:
        iqm_server_url: URL for accessing the server. Has to start with http or https.
        quantum_computer: ID or alias of the quantum computer to connect to, if the IQM Server
            instance controls more than one.
        token: Long-lived authentication token in plain text format.
            If ``token`` is given no other user authentication parameters should be given.
        tokens_file: Path to a tokens file used for authentication.
            If ``tokens_file`` is given no other user authentication parameters should be given.
        client_signature: String that Pulla adds to User-Agent header of requests
            it sends to the server. The signature is appended to IQMServerClient's own version
            information and is intended to carry additional version information,
            for example the version information of the caller.

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
        if not client_signature:
            client_signature = f"iqm-pulla {version('iqm-pulla')}"
        try:
            self._iqm_server_client = _IQMServerClient(
                iqm_server_url=iqm_server_url,
                token=token,
                tokens_file=tokens_file,
                client_signature=client_signature,
                quantum_computer=quantum_computer,
            )
        except Exception as e:
            logger.error("Failed to initialize IQM Server client: %s", e)
            raise ValueError("Failed to initialize IQM Server client") from e
        # Separate wrapper on top of IQM Server client to simplify calibration data fetching.
        self._calibration_data_provider = CalibrationDataProvider(self._iqm_server_client)

        # Data needed for the compiler.
        self._station_control_settings: SettingNode | None = None
        self._chip_topology: ChipTopology = self.get_chip_topology()
        self._chip_label: str = self.get_chip_label()

        self._channel_properties: dict[str, ChannelProperties]
        self._component_channels: dict[str, dict[str, str]]
        self._channel_properties, self._component_channels = self.get_channel_properties()

    def get_standard_compiler(
        self,
        calibration_set_values: CalibrationSetValues | None = None,
        circuit_execution_options: CircuitExecutionOptions | dict | None = None,
    ) -> Compiler:
        """Returns a new instance of the compiler with the default calibration set and standard stages.

        Args:
            calibration_set_values: Calibration set to use. If None, the current calibration set will be used.
            circuit_execution_options: circuit execution options to use for the compiler. If a CircuitExecutionOptions
                object is provided, the compiler use it as is. If a dict is provided, the default values will be
                overridden for the present keys in that dict. If left ``None``, the default options will be used.

        Returns:
            The compiler object.

        """
        if circuit_execution_options is None:
            circuit_execution_options = STANDARD_CIRCUIT_EXECUTION_OPTIONS
        elif isinstance(circuit_execution_options, dict):
            circuit_execution_options = CircuitExecutionOptions(
                **STANDARD_CIRCUIT_EXECUTION_OPTIONS_DICT | circuit_execution_options  # type: ignore
            )
        return Compiler(
            calibration_set_values=calibration_set_values or self.fetch_default_calibration_set()[0],
            chip_topology=self._chip_topology,
            channel_properties=self._channel_properties,
            component_channels=self._component_channels,
            component_mapping=None,
            stages=get_standard_stages(),
            options=circuit_execution_options,
        )

    def fetch_default_calibration_set(self) -> tuple[CalibrationSetValues, uuid.UUID]:
        """Fetch the default calibration set from the server, in a minimal format.

        Returns:
            Calibration set contents, calibration set ID.

        """
        default_calibration_set, default_calibration_set_id = (
            self._calibration_data_provider.get_default_calibration_set()
        )
        return default_calibration_set, default_calibration_set_id

    def fetch_calibration_set_values_by_id(self, calibration_set_id: uuid.UUID) -> CalibrationSetValues:
        """Fetch a specific calibration set from the server.

        All calibration sets are cached in-memory, so if the calibration set with the given
        id has already been fetched, it will be returned immediately.

        Args:
            calibration_set_id: ID of the calibration set to fetch.

        Returns:
            Calibration set contents.

        """
        calibration_set = self._calibration_data_provider.get_calibration_set_values(calibration_set_id)
        return calibration_set

    def get_chip_label(self) -> str:
        """QPU label of the quantum computer we are connected to."""
        try:
            duts = self._iqm_server_client.get_duts()
        except requests.RequestException as e:
            raise ChipLabelRetrievalException(f"Failed to retrieve the chip label: {e}") from e

        if len(duts) != 1:
            raise ChipLabelRetrievalException(f"Expected exactly one chip label, but got {len(duts)}")
        return duts[0].label

    def get_chip_topology(self) -> ChipTopology:
        """QPU topology of the quantum computer we are connected to."""
        self.get_chip_label()  # Called just to make sure that there will be only one DUT available
        try:
            chip_design_record = self._iqm_server_client.get_chip_design_records()[0]
        except Exception as e:
            raise CHADRetrievalException("Could not fetch chip design record") from e
        return ChipTopology.from_chip_design_record(chip_design_record)

    def _get_station_control_settings(self) -> SettingNode:
        """Station Control default settings tree."""
        if self._station_control_settings is None:
            # request the station settings, cache the results
            try:
                self._station_control_settings = self._iqm_server_client.get_settings()
            except Exception as e:
                raise SettingsRetrievalException("Could not fetch station settings") from e
        return self._station_control_settings

    def get_channel_properties(
        self,
    ) -> tuple[dict[str, ChannelProperties], dict[str, dict[str, str]]]:
        """Control channel properties from Station Control controller settings.

        Returns:
            channel_properties: Mapping from channel names to  their properties.
            component_to_channel: Mapping from chip component names to functions to channel names.
                For example, `'QB1' -> 'drive' -> 'QB1__drive.awg'`

        """
        return get_channel_properties_from_station_settings(
            self._get_station_control_settings(), self.get_chip_topology()
        )

    def submit_playlist(
        self,
        playlist: Playlist,
        settings: SettingNode,
        *,
        context: dict[str, Any],
        use_timeslot: bool = False,
    ) -> SweepJob:
        """Submit a Playlist of instruction schedules for execution on the remote quantum computer.

        Args:
            playlist: Schedules to execute.
            settings: Station settings to be used for the execution.
            context: Context object of the compiler run that produced ``playlist``, containing the readout mappings.
                Required for postprocessing the results.
            use_timeslot: Submits the job to the timeslot queue if set to ``True``. If set to ``False``,
                the job is submitted to the normal on-demand queue.

        Returns:
            Created job object, used to query the job status and the execution results.

        """
        readout_components = []
        for _, channel in self._component_channels.items():
            for k, v in channel.items():
                if k == "readout":
                    readout_components.append(v)

        sweep = SweepDefinition(
            sweep_id=uuid.uuid4(),
            playlist=playlist,
            return_parameters=list(extract_readout_controller_result_names(context["readout_mappings"])),
            settings=settings,
            dut_label=self.get_chip_label(),
            sweeps=[],
        )
        job_data = self._iqm_server_client.submit_sweep(sweep, use_timeslot=use_timeslot)
        logger.info("Submitted a job with ID: %s", job_data.id)

        # Initialize the job object, which can be then used to query
        return SweepJob(
            data=job_data,
            _pulla=self,
            _context=deepcopy(context),
        )


@dataclass
class SweepJob(IQMServerClientJob):
    """Status and results of a Pulla sweep job.

    Created by :meth:`Pulla.submit_playlist`.
    """

    _pulla: Pulla
    """Client instance used to create the job."""

    _context: dict[str, Any]
    """Final context object of the compiler run used to produce the sweep, contains information needed
    for processing the results."""

    _result: CircuitMeasurementResultsBatch | None = None
    """Sweep results converted to the circuit measurement results expected by the client."""

    @property
    def _iqm_server_client(self) -> _IQMServerClient:
        return self._pulla._iqm_server_client

    def result(self) -> CircuitMeasurementResultsBatch | None:
        """Get (and cache) the job result, if the job has completed.

        Returns:
            Circuit measurement results for the job, or None if the results are not available.

        """
        if not self._result:
            self.update()
            # if successful, get the results (TODO what about possible partial data?)
            if self.status != JobStatus.COMPLETED:
                return None

            sweep_results = self._iqm_server_client.get_job_artifact_sweep_results(self.job_id)
            self._result = map_sweep_results_to_logical_qubits(
                sweep_results,
                self._context["readout_mappings"],
                self._context["options"].heralding_mode,
            )
        return self._result
