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
"""Pydantic models used by the API."""

from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeAlias

from exa.common.data.setting_node import SettingNode
from iqm.pulse import Circuit
from iqm.pulse.builder import Locus
from iqm.pulse.playlist.playlist import Playlist
from iqm.station_control.interface.models import (
    DDMode,
    DDStrategy,
    HeraldingMode,
    MoveGateFrameTrackingMode,
    MoveGateValidationMode,
)

CircuitBatch: TypeAlias = list[Circuit]
"""Type that represents a list of quantum circuits to be executed together in a single batch."""


class MeasurementMode(StrEnum):
    """Measurement mode for circuit execution.

    Determines which QPU components are measured by Station Control in the final measurement.
    Measurement results which are not required by the circuits to be executed are discarded.
    """

    CIRCUIT = "circuit"
    """In each circuit separately, measure only the components that have final measurement
    operations on them."""
    ALL = "all"
    """Measure all the components on the QPU that have measurement data in the calset.
    This is typically how measurement is calibrated."""


class CircuitBoundaryMode(StrEnum):
    """Circuit boundary mode for circuit compilation."""

    NEIGHBOUR = "neighbour"
    """
    Circuit boundary consists of those QPU elements (qubits and couplers) that
    are adjacent to the qubits and couplers used by the circuit, but do not belong to them.
    Specifically,

    * Boundary qubits are connected to a circuit qubit by any coupler, but are not circuit qubits themselves.
    * Boundary couplers are connected to at least one circuit qubit, but are not used in the circuit themselves.
    """
    ALL = "all"
    """Circuit boundary consists of all the QPU elements that are not used in the circuit."""


@dataclass(frozen=True)
class CircuitExecutionOptions:
    """Various discrete options for quantum circuit execution."""

    measurement_mode: MeasurementMode
    heralding_mode: HeraldingMode
    dd_mode: DDMode
    dd_strategy: DDStrategy | None
    circuit_boundary_mode: CircuitBoundaryMode
    move_gate_validation: MoveGateValidationMode
    move_gate_frame_tracking: MoveGateFrameTrackingMode
    active_reset_cycles: int | None
    convert_terminal_measurements: bool
    """Iff True, convert terminal measurements to a non-QND, high-fidelity measurement."""


ReadoutMapping: TypeAlias = dict[str, tuple[str, ...]]
"""Type for matching measurement keys from the quantum circuit with acquisition labels in Station Control.

In quantum circuits, measurements are identified by measurement keys.
Measurements in Station Control are identified by acquisition labels specific to a readout controller.
This type is a dictionary mapping measurement keys to lists of acquisition labels --- each acquisition
label should hold the readout of a single qubit at a single point in the circuit, and the order in
the list corresponds to the order of qubits in the measurement instruction. E.g. if one has
measurement instruction with ``key='mk'`` and ``qubits=[QB2, QB1]``, then the corresponding entry in
this dict would be ``'mk': ('QB2__mk', 'QB1__mk')``

The values of the ReadoutMapping are used to determine which measurement results Station Control
should return to Cocos.
"""

ReadoutMappingBatch: TypeAlias = tuple[ReadoutMapping, ...]
"""Type that represents tuple of readout mappings, one per each circuit in a circuit batch."""


@dataclass
class CircuitMetrics:
    """Metrics describing a circuit and its compilation result."""

    components: frozenset[str]
    """Locus components used in the circuit."""
    component_pairs_with_gates: frozenset[tuple[str, str]]
    """Pairs of locus components which have two-component gates between them in the circuit."""
    gate_loci: dict[str, dict[str, Counter[Locus]]] = field(default_factory=dict)
    """Mapping from operation name to mapping from implementation name to a counter of loci of
    that operation in the circuit."""
    schedule_duration: float = 0.0
    """Duration of the instruction schedule created for the circuit, in seconds."""
    min_execution_time: float = 0.0
    """Lower bound on the actual execution time: shots * (instruction schedule duration + reset), in seconds."""


@dataclass
class CircuitCompilationResult:
    """Compiled circuit and associated settings returned by CPC to Cocos."""

    playlist: Playlist
    """sequence of instruction schedules corresponding to the batch of circuits to be executed"""

    readout_mappings: ReadoutMappingBatch
    """For each circuit in the batch, mapping from measurement keys to the names of readout
    controller result parameters that will hold the measurement results. If heralding is enabled, qubits
    which are not measured in the circuit itself but are heralded appear under the reserved key "__herald."""

    settings: SettingNode
    """Station Control settings node containing the settings for circuit execution"""

    circuit_metrics: tuple[CircuitMetrics, ...]
    """metrics describing the circuit and its compilation result for each circuit in the batch"""
