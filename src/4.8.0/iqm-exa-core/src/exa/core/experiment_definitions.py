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
"""Type definitions and options for :class:`.Experiment`."""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Any, Literal, TypeAlias

from exa.common.sweep import Sweeps
from exa.core.analyze_result import AnalyzeResult
from exa.core.experiment_options import ExperimentOptions  # noqa: F401
from iqm.cpc.core.config import (  # noqa: F401
    ComponentGrouping,
    ComponentGroupingMode,
    Components,
)
from iqm.cpc.core.parallel_utils import (
    FullTGF,
    SubTGF,
)

CircuitGenerationFunction: TypeAlias = SubTGF | FullTGF
SweepGenerationFunction: TypeAlias = Callable[..., Sweeps]
AnalyzeFunction: TypeAlias = Callable[..., AnalyzeResult | None]
ChipProperties: TypeAlias = Callable[
    [
        Literal["qubit", "tunable_coupler", "computational_resonator", "probe_line"],
        Literal["design", "fabrication"],
    ],
    dict[str, dict[str, Any]],
]
"""Type for the QPU properties callback."""


class ReadoutMode(Enum):
    """Allowed values for :attr:`.ExperimentOptions.readout_mode`."""

    COMPLEX = "complex"
    """Complex integrated results, averaging done in the hardware."""
    THRESHOLD = "threshold"
    """Threshold discriminated integrated results (i.e. 0 for the ground state and 1 for the excited state)."""
    SINGLE_SHOT = "single_shot"
    """Complex integrated results, individual shot results returned, averaging done in the software."""
    SINGLE_SHOT_THRESHOLD = "single_shot_threshold"
    """Threshold discriminated integrated results (i.e. 0 for the ground state and 1 for the excited state), individual
    shot results returned, averaging done in the software."""
    COUNTER = "counter"
    """Multi-qubit fock-state counts (histograms), used e.g. when multi-qubit correlations are needed."""


class QPUType(Enum):
    """Type of the QPU, crystal or star."""

    CRYSTAL = "crystal"
    STAR = "star"
    CONSTELLATION = "constellation"


class ParallelGroup(Enum):
    """QPU component unit of parallelization, for :attr:`.Experiment.parallel_group`.

    See :meth:`.Experiment.get_default_circuit_generation_function`.
    """

    NONE = "none"
    """Given component grouping is passed to the CGF as is."""
    SINGLES = "singles"
    """Parallel unit is a single QPU component."""
    PROBE_LINES = "probe_lines"
    """Like SINGLES, but for probe lines specifically."""
    CONNECTED_PAIRS = "connected_pairs"
    """Parallel unit is a pair of components connected by a coupler."""
    QUBIT_RESONATOR_PAIRS = "qubit_resonator_pairs"
    """Parallel unit is a (qubit, resonator) pair connected by a coupler."""
    STAR_TRIPLETS = "star_triplets"
    """Parallel unit is (qubit_1, resonator, qubit_2), both qubits coupled to the resonator."""
    COLORED_GROUPS = "colored_groups"
    """Parallel units and their groups are given by the user, they are only automatically merged into color groups."""
    NNN_PAIRS = "nnn_pairs"
    """Parallel unit is a pair of components connected by a coupler - computational component - coupler chain."""


class ReadoutProgrammingMode(Enum):
    """Allowed values for :attr:`.ExperimentOptions.readout_programming_mode`."""

    CLASSIC = "classic"
    """Automatically adds a 'measure' gate to all qubits at the end of each segment in the playlist."""
    HERALDING = "heralding"
    """Automatically adds a 'measure' gate at the beginning and end of each segment in the playlist."""
    RESET_AT_END = "reset_at_end"
    """The terminal measurement is done with a feedback key, and a conditional prx is applied after. NOTE: this mode
    overrides :attr:`.ExperimentOptions.active_reset_cycles` (forcing one conditional reset at the end of the segment)
    and requires a station which supports fast feedback."""
    CUSTOM = "custom"
    """Nothing is added automatically. All the readouts need to be explicitly added in the circuit function itself."""
