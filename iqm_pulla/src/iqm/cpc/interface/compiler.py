# Copyright 2024 IQM
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
from typing import Final, TypeAlias

from exa.common.data.setting_node import SettingNode
from iqm.pulse.builder import CircuitOperation, Locus
from iqm.pulse.playlist.playlist import Playlist


@dataclass
class Circuit:
    """Quantum circuit to be executed."""

    name: str
    """name of the circuit"""
    instructions: tuple[CircuitOperation, ...]
    """operations comprising the circuit"""


CircuitBatch: TypeAlias = list[Circuit]
"""Type that represents a list of quantum circuits to be executed together in a single batch."""


class MeasurementMode(StrEnum):
    """Measurement mode for circuit execution.

    Determines which QPU components are measured by Station Control in the final measurement.
    Measurement results which are not required by the circuits to be executed are discarded.
    """

    CIRCUIT: Final[str] = "circuit"
    """In each circuit separately, measure only the components that have final measurement
    operations on them."""
    ALL: Final[str] = "all"
    """Measure all the components on the QPU that have measurement data in the calset.
    This is typically how measurement is calibrated."""


class HeraldingMode(StrEnum):
    """Heralding mode for circuit execution."""

    NONE: Final[str] = "none"
    """Do not do any heralding."""
    ZEROS: Final[str] = "zeros"
    """Perform a heralding measurement on all the components used in each circuit (if they have
    measurement data available in the calset), only retain shots where all the components
    are measured to be in the zero state."""


class DDMode(StrEnum):
    """Dynamical Decoupling (DD) mode for circuit execution."""

    DISABLED: Final[str] = "disabled"
    """Do not apply dynamical decoupling."""
    ENABLED: Final[str] = "enabled"
    """Apply dynamical decoupling."""


PRXSequence: TypeAlias = list[tuple[float, float]]
"""A sequence of PRX gates. A generic PRX gate is defined by rotation angle and phase angle, Theta and Phi
respectively."""


@dataclass
class DDStrategy:
    """Describes a particular dynamical decoupling strategy.

    The current standard DD stategy can be found in :attr:`.STANDARD_DD_STRATEGY`,
    but users can use this class to provide their own dynamical decoupling strategies.

    See :cite:`Ezzell_2022` for information on DD sequences.
    """

    merge_contiguous_waits: bool = True
    """Merge contiguous ``Wait`` instructions into one if they are separated only by ``Block`` instructions."""

    target_qubits: frozenset[str] | None = None
    """Qubits on which dynamical decoupling should be applied. If ``None``, all qubits are targeted."""

    skip_leading_wait: bool = True
    """Skip processing leading ``Wait`` instructions."""

    skip_trailing_wait: bool = True
    """Skip processing trailing ``Wait`` instructions."""

    gate_sequences: list[tuple[int, str | PRXSequence, str]] = field(default_factory=list)
    """Available decoupling gate sequences to chose from in this strategy.

    Each sequence is defined by a tuple of ``(ratio, gate pattern, align)``:

        * ratio: Minimal duration for the sequence (in PRX gate durations).

        * gate pattern: Gate pattern can be defined in two ways. It can be a string containing "X" and "Y" characters,
          encoding a PRX gate sequence. For example, "YXYX" corresponds to the
          XY4 sequence, "XYXYYXYX" to the EDD sequence, etc. If more flexibility is needed, a gate pattern can be
          defined as a sequence of PRX gate argument tuples (that contain a rotation angle and a phase angle). For
          example, sequence "YX" could be written as ``[(math.pi, math.pi / 2), (math.pi, 0)]``.

        * align: Controls the alignment of the sequence within the time window it is inserted in. Supported values:

          - "asap": Corresponds to a ASAP-aligned sequence with no waiting time before the first pulse.
          - "center": Corresponds to a symmetric sequence.
          - "alap": Corresponds to a ALAP-aligned sequence.

    The Dynamical Decoupling algorithm uses the best fitting gate sequence by first sorting them
    by ``ratio`` in descending order. Then the longest fitting pattern is determined by comparing ``ratio``
    with the duration of the time window divided by the PRX gate duration.
    """


class CircuitBoundaryMode(StrEnum):
    """Circuit boundary mode for circuit compilation."""

    NEIGHBOUR: Final[str] = "neighbour"
    """
    Circuit boundary consists of those QPU elements (qubits and couplers) that
    are adjacent to the qubits and couplers used by the circuit, but do not belong to them.
    Specifically,

    * Boundary qubits are connected to a circuit qubit by any coupler, but are not circuit qubits themselves.
    * Boundary couplers are connected to at least one circuit qubit, but are not used in the circuit themselves.
    """
    ALL: Final[str] = "all"
    """Circuit boundary consists of all the QPU elements that are not used in the circuit."""


class MoveGateValidationMode(StrEnum):
    """MOVE gate validation mode for circuit compilation."""

    STRICT: Final[str] = "strict"
    """Perform standard MOVE gate validation: MOVE(qubit, resonator) gates must only
    appear in sandwiches (pairs). Inside a sandwich there must be no gates acting on the
    MOVE qubit, and no other MOVE gates acting on the resonator."""
    ALLOW_PRX: Final[str] = "allow_prx"
    """Allow PRX gates on the MOVE qubit inside MOVE sandwiches during validation."""
    NONE: Final[str] = "none"
    """Do not perform any MOVE gate validation."""


class MoveGateFrameTrackingMode(StrEnum):
    """MOVE gate frame tracking mode for circuit compilation."""

    FULL: Final[str] = "full"
    """Perform complete MOVE gate frame tracking, applying both the explicit z rotations
    on the resonator and the dynamic phase correction due to qubit-resonator detuning to
    the qubit at the end of a MOVE sandwich."""
    NO_DETUNING_CORRECTION: Final[str] = "no_detuning_correction"
    """Do not apply the detuning correction at the end of a MOVE sandwich."""
    NONE: Final[str] = "none"
    """Do not perform any MOVE gate frame tracking."""


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
