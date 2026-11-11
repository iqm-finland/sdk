# Copyright 2022-2026 IQM
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

"""Clifford data regression (CDR): fit measured against exact values of Clifford training circuits."""

from iqm.error_reduction_tools.cdr.cdr_clifford_utils import (
    append_stim_sequence,
    build_stim_single_qubit_clifford_table,
    get_clifford_r_library,
    is_logically_single_qubit_clifford_r,
    is_signed_permutation_matrix,
    normalize_angle_2pi,
    r_gate_so3_matrix,
    signed_permutation_key_from_so3,
    so3_h_matrix,
    so3_s_matrix,
    stim_sequence_for_r_gate,
)
from iqm.error_reduction_tools.cdr.cdr_observables import (
    diagonal_observable_to_terms,
    get_observable_num_qubits,
    pauli_label_to_z_qubits,
    to_pauli_terms,
)
from iqm.error_reduction_tools.cdr.cdr_plots import plot_cdr_fit, plot_cdr_summary
from iqm.error_reduction_tools.cdr.cdr_processors import (
    cdr_fit_and_predict,
    r_squared,
)
from iqm.error_reduction_tools.cdr.cdr_technique import CDR, CDRDiagnostics, CDREstimation, TrainingVariant
from iqm.error_reduction_tools.cdr.cdr_training_circuits import (
    build_clbit_to_stim_qubit_map,
    exact_expectation_from_stim,
    is_instruction_allowed_in_training_circuit,
    stim_circuit_without_measurements,
    training_circuit_to_stim,
    validate_training_circuit_is_clifford,
)
from iqm.error_reduction_tools.cdr.cdr_training_set import (
    TrainingSet,
    build_clifford_circuit_from_assignment,
    exact_expectation_vector,
    farthest_point_selection,
    generate_clifford_training_set,
    load_training_set,
    non_clifford_slot_indices,
    random_assignment,
    save_training_set,
    training_set_spread,
)

__all__ = [
    "CDR",
    "CDRDiagnostics",
    "CDREstimation",
    "TrainingVariant",
    "plot_cdr_fit",
    "plot_cdr_summary",
    "TrainingSet",
    "append_stim_sequence",
    "build_clbit_to_stim_qubit_map",
    "build_clifford_circuit_from_assignment",
    "build_stim_single_qubit_clifford_table",
    "cdr_fit_and_predict",
    "diagonal_observable_to_terms",
    "exact_expectation_from_stim",
    "exact_expectation_vector",
    "farthest_point_selection",
    "generate_clifford_training_set",
    "get_clifford_r_library",
    "get_observable_num_qubits",
    "is_instruction_allowed_in_training_circuit",
    "is_logically_single_qubit_clifford_r",
    "is_signed_permutation_matrix",
    "load_training_set",
    "non_clifford_slot_indices",
    "normalize_angle_2pi",
    "pauli_label_to_z_qubits",
    "r_gate_so3_matrix",
    "r_squared",
    "random_assignment",
    "save_training_set",
    "signed_permutation_key_from_so3",
    "so3_h_matrix",
    "so3_s_matrix",
    "stim_circuit_without_measurements",
    "stim_sequence_for_r_gate",
    "to_pauli_terms",
    "training_circuit_to_stim",
    "training_set_spread",
    "validate_training_circuit_is_clifford",
]
