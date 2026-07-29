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
"""Data models used in circuit execution."""

from collections import Counter
from dataclasses import dataclass, field
from typing import Generic, TypeAlias, TypeVar

from exa.common.data.parameter import SettingValue
from exa.common.data.setting_node import SettingNode
from iqm.pulse.builder import Locus
from iqm.pulse.playlist.playlist import Playlist

# TODO: SC circuit execution imports this stuff so we cannot yet remove it, this should be cleaned up when we eventually

ReadoutMapping: TypeAlias = dict[str, tuple[str, ...]]
"""Type for matching measurement keys from a quantum circuit with acquisition labels in Station Control.

In quantum circuits, measurements are identified by measurement keys.
Measurements in Station Control are identified by acquisition labels. This type is a dictionary mapping
measurement keys to lists of acquisition labels --- each acquisition label should hold the readout of a
single qubit at a single point in the circuit, and the order in the list corresponds to the order of qubits
in the measurement instruction. E.g. if one has measurement instruction with ``key='mk'`` and
``qubits=[QB2, QB1]``, then the corresponding entry in this dict would be ``'mk': ('QB2__mk', 'QB1__mk')``

The values of the ReadoutMapping are used to determine which measurement results Station Control
should return.
"""

ReadoutMappingBatch: TypeAlias = tuple[ReadoutMapping, ...]
"""Type that represents readout mappings for a batch of circuits."""


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


T_settings = TypeVar("T_settings", SettingNode, dict[str, SettingValue])


@dataclass
class CircuitCompilationResult(Generic[T_settings]):
    """Compiled circuit and associated settings returned by CPC to Station Control."""

    playlist: Playlist
    """sequence of instruction schedules corresponding to the batch of circuits to be executed"""

    readout_mappings: ReadoutMappingBatch
    """For each circuit in the batch, mapping from measurement keys to the names of readout
    controller result parameters that will hold the measurement results. If heralding is enabled, qubits
    which are not measured in the circuit itself but are heralded appear under the reserved key "__herald."""

    settings: T_settings
    """Station Control settings for circuit execution.
    Either a settings tree or a mapping of setting names to their values.
    If a mapping, must only contain non-None, non-readonly controller settings."""

    circuit_metrics: tuple[CircuitMetrics, ...]
    """metrics describing the circuit and its compilation result for each circuit in the batch"""
