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
"""Utils for Experiment."""

__all__ = [
    "CIRCUIT_SWEEP_PARAMETER",
    "COUNTER_INDEX",
    "COUNTER_RESULT",
    "GLOBAL_LOCUS_STRING_REPRESENTATION",
    "RAGGED_DATA_INDEX",
    "READOUT_SLOT_SWEEP_PARAMETER",
    "REPETITIONS",
    "TIME_TRACE_INDEX",
    "ConstantDelay",
    "DoubleMOVE",
    "MOVE_CZ_MOVE_Single",
    "MOVE_NCZ_MOVE_Composite",
    "add_mitigated_results_tensor_product_assignment",
    "add_move_detuning_to_implementations",
    "analyze_singles",
    "color_grouped_analysis",
    "cz_n_quantum_op",
    "filter_components_with_drive",
    "flatten_subscribed_components",
    "gates_data_from",
    "generate_color_grouped_circuit",
    "get_1qb_default_sweeps",
    "get_2qb_gate_groups",
    "get_color_grouped_sweeps",
    "get_coupler_from_mapping",
    "get_default_channel_properties",
    "get_default_implementation_name",
    "get_flat_run_components_by_type",
    "get_gate_node_for_locus",
    "get_inverse_assignment_matrix",
    "get_run_components_for_one_colour_group",
    "get_sweep_parameters",
    "group_is_subscribed",
    "id_quantum_op",
    "limit_components_by_settings",
    "multi_kron",
    "process_components_to_pairs",
    "process_groups_to_colors",
    "process_probe_lines",
    "process_singular_run_components",
    "update_options_for_default_sweeps",
    "validate_options",
    "plot_with_circuit_index",
    "wrap_2qb_plot_for_circuit_dimension",
]

from iqm.cpc.compiler._utils.settings import (
    GLOBAL_LOCUS_STRING_REPRESENTATION,
    gates_data_from,
    limit_components_by_settings,
)
from iqm.cpc.compiler._utils.stages import (
    CIRCUIT_SWEEP_PARAMETER,
    COUNTER_INDEX,
    RAGGED_DATA_INDEX,
    READOUT_SLOT_SWEEP_PARAMETER,
    REPETITIONS,
    TIME_TRACE_INDEX,
    flatten_subscribed_components,
)
from iqm.cpc.core.parallel_utils import (
    generate_full_tgf as generate_color_grouped_circuit,  # backwards compatibility
)

from .analysis_utils import (
    add_mitigated_results_tensor_product_assignment,
    analyze_singles,
    get_inverse_assignment_matrix,
    multi_kron,
)
from .circuit_utils import get_default_channel_properties
from .gates_utils import (
    ConstantDelay,
    DoubleMOVE,
    MOVE_CZ_MOVE_Single,
    MOVE_NCZ_MOVE_Composite,
    cz_n_quantum_op,
    id_quantum_op,
)
from .parallel_utils import (
    color_grouped_analysis,
    get_2qb_gate_groups,
    get_color_grouped_sweeps,
    get_coupler_from_mapping,
    get_run_components_for_one_colour_group,
    group_is_subscribed,
    plot_with_circuit_index,
    process_components_to_pairs,
    process_groups_to_colors,
    process_probe_lines,
    process_singular_run_components,
    wrap_2qb_plot_for_circuit_dimension,
)
from .setting_utils import (
    COUNTER_RESULT,
    add_move_detuning_to_implementations,
    filter_components_with_drive,
    get_default_implementation_name,
    get_flat_run_components_by_type,
    get_gate_node_for_locus,
    validate_options,
)
from .sweep_utils import (
    get_1qb_default_sweeps,
    get_sweep_parameters,
    update_options_for_default_sweeps,
)
