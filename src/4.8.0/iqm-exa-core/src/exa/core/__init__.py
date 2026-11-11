# Copyright 2019-2026 IQM Finland Oy
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
"""Core package for experiment and calibration procedures.

The top-level module reimports some definitions from IQM's other library packages.
The reimported definitions are most commonly needed when scripting and building Experiments.

.. code-block:: python

    import exa.core as ec
    def my_circuit_func(components: ec.ComponentGrouping, builder: ec.ScheduleBuilder) -> list[ec.TimeBox]:
        return [ec.TimeBox([...])]

"""

__all__ = [
    "AnalyzeResult",
    "ChipTopology",
    "CircuitOperationList",
    "CollectionType",
    "ComponentGrouping",
    "ComponentGroupingMode",
    "DataType",
    "ErrorExperiment",
    "Experiment",
    "ExperimentOptions",
    "FromExperimentConfiguration",
    "GetCalibrationSet",
    "GetDefaultCalibrationSet",
    "GetLatestCharacterizationSet",
    "GetLatestForTags",
    "GetObservationSet",
    "GetQualityMetricSet",
    "LatestFromStash",
    "NdSweep",
    "ParallelGroup",
    "Parameter",
    "PlotDefinition",
    "QPUType",
    "RangeValidationRule",
    "ReadoutMode",
    "ReadoutProgrammingMode",
    "RunResult",
    "ScheduleBuilder",
    "SchedulingStrategy",
    "Setting",
    "SettingNode",
    "Sweep",
    "Sweeps",
    "TimeBox",
    "analyze_singles",
    "annotate",
    "apply_along_coordinate",
    "environment",
    "generic_plot",
    "generic_plot_definition",
    "get_default_implementation_name",
    "get_experiment",
    "get_gate_node_for_locus",
    "get_loaded_stash_with",
    "load_sequence_result",
    "set_global_observation_stash",
    "wrap_mpl_plot",
]

from importlib.metadata import PackageNotFoundError, version

from exa.common.data import CollectionType, DataType, Parameter, Setting, SettingNode, Sweep
from exa.common.qcm_data import ChipTopology
from exa.common.sweep import NdSweep, Sweeps
from exa.core.error_experiment import ErrorExperiment
from exa.core.experiment_options import ExperimentOptions
from exa.core.observation import set_global_default_observation_loading_rules
from iqm.cpc.core.config import ComponentGrouping, ComponentGroupingMode
from iqm.cpc.core.dataset import annotate, apply_along_coordinate
from iqm.cpc.core.observation.observation_loading_rules import (
    LatestFromStash,
)
from iqm.cpc.core.run_result import RunResult
from iqm.pulse.builder import ScheduleBuilder, SchedulingStrategy
from iqm.pulse.circuit_operations import CircuitOperationList
from iqm.pulse.timebox import TimeBox

from . import environment
from .analyze_result import AnalyzeResult
from .analyze_result.plot_definition import PlotDefinition, generic_plot, generic_plot_definition, wrap_mpl_plot
from .experiment.experiment import Experiment, get_experiment
from .experiment.utils.analysis_utils import analyze_singles
from .experiment.utils.setting_utils import (
    get_default_implementation_name,
    get_gate_node_for_locus,
)
from .experiment_definitions import (
    ParallelGroup,
    QPUType,
    ReadoutMode,
    ReadoutProgrammingMode,
)
from .observation.observation_loading_rules import (
    FromExperimentConfiguration,
    GetCalibrationSet,
    GetDefaultCalibrationSet,
    GetLatestCharacterizationSet,
    GetLatestForTags,
    GetObservationSet,
    GetQualityMetricSet,
)
from .observation.observation_stash import (
    get_loaded_stash_from_rules,
    get_loaded_stash_with,
    set_global_observation_stash,
)
from .observation.observation_validation_rules import RangeValidationRule
from .sequence import load_sequence_result

# Change here if project is renamed and does not equal the package name
DIST_NAME = "iqm-exa-core"

try:
    __version__ = version(DIST_NAME)
except PackageNotFoundError:
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError
