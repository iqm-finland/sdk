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
"""Generic interface to implement classification of complex/voltage readout data to state."""

from enum import IntEnum

import numpy as np
import xarray as xr

from exa.common.data import Parameter, SettingNode
from exa.core.experiment.utils import get_gate_node_for_locus
from exa.core.post_processing.excited_state_probability_computation import (
    EXCITED_STATE_PROBABILITY,
    F_STATE_PROBABILITY,
    GROUND_STATE_PROBABILITY,
    compute_three_state_probability,
)
from exa.core.run_result import RunResult
from exa.core.util.dataset import annotate

from .parameters import (
    get_corrected_state_label,
    get_uncorrected_state_label,
)


def get_assigment_inverse(input_data: xr.Dataset, component: str) -> np.ndarray:
    """Compute the inverse assignment error matrix for a component from labeled calibration data.

    The innput data is expected to be in of the form {'0': data, '1': data, '2': data}.

    Args:
        input_data: The labeled calibration dataset.
        component: The component to compute the inverse assignment error matrix for.

    Returns:
        The inverse of the assignment error matrix.

    """
    assigment_error_matrix = np.array(
        [
            [
                input_data.sel(pi_pulse_index=pulse_label)[get_uncorrected_state_label(component, state_label)]
                for state_label in [0, 1, 2]
            ]
            for pulse_label in [0, 1, 2]
        ]
    )

    assigment_error_matrix = np.transpose(assigment_error_matrix)
    assigment_error_matrix = np.linalg.pinv(assigment_error_matrix)

    return assigment_error_matrix


def apply_assignment_inverse(input_data: xr.Dataset, component: str, inverse_matrix: np.ndarray) -> None:
    """Apply assignment inverse to correct for mixing.

    Updates the provided data with the corrected state probabilities.

    Args:
        input_data: The dataset with the uncorrected state probabilities.
        component: The component to correct the state probabilities for.
        inverse_matrix: The inverse assignment error matrix.

    """
    g_data = input_data[get_uncorrected_state_label(component, 0)]
    e_data = input_data[get_uncorrected_state_label(component, 1)]
    f_data = input_data[get_uncorrected_state_label(component, 2)]

    fixed = np.tensordot(inverse_matrix, (np.array([g_data, e_data, f_data])), axes=([-1], [0]))

    input_data.update(
        {
            get_corrected_state_label(component, 0): (g_data.dims, fixed[0]),
            get_corrected_state_label(component, 1): (e_data.dims, fixed[1]),
            get_corrected_state_label(component, 2): (f_data.dims, fixed[2]),
        }
    )


def add_classifier_to_settings(
    experiment_settings: SettingNode,
    components: list[str],
    classifier,  # noqa: ANN001
) -> SettingNode:
    """Get settings of an experiment with the classifier settings added, the values are loaded from the observations.

    Args:
        experiment_settings: The setting tree of the experiment.
        components: The components to add classifier settings for.
        classifier: The classifier subclass to get the settings from.

    Returns:
        The experiment settings containing an additional classifer node.

    """
    experiment_settings.classifier = classifier().get_settings(components)
    return experiment_settings


def get_state_vector(input_data: xr.Dataset, group: tuple[str, ...]) -> np.ndarray:
    """Calculate the statevector over n qubits.

    Args:
        input_data: Dataset with the state data.
        group: sequence of n qubit names

    Returns:
        An array representing average counts of all the combinations of states of all the qutrits in the list, from
            00...0 to 22...2, indexed by natural numbers from 0 to 3^n - 1.

    """
    total = input_data[f"{group[0]}_state"] * 0
    for i, qubit in enumerate(list(group)[::-1]):
        total += 3**i * input_data[f"{qubit}_state"]
    return np.array([(total == j).mean("repetitions").values for j in range(3 ** len(group))])


def package_mitigated_result(corrected: np.ndarray, input_data: xr.Dataset, group: tuple[str, ...]) -> xr.DataArray:
    """Make a single DataArray with the mitigated n-qubit array of counts.

    Args:
        corrected: numpy array processed by the classifier.
        input_data: Dataset to modify.
        group: Sequence of n qubits.

    Returns:
        Corrected array packaged as DataArray with correct names and parameters.

    """
    prefix = "__".join(list(group))
    state = Parameter(f"{prefix}_state_index", f"{prefix} state index")
    result = Parameter(f"{prefix}_qutrit_group_result", f"{prefix} Qutrit group result")
    state_array = state.build_data_array(np.arange(3 ** len(group)))
    collapsed = input_data[f"{group[0]}_state"].isel(repetitions=0, drop=True)
    corrected_array = result.build_data_array(
        corrected, [state.name] + list(collapsed.dims), {**collapsed.coords.variables.mapping, state.name: state_array}
    )
    return corrected_array


class MultiStateClassifierOptions(IntEnum):
    """Option specifying which decomposition for single-qubit Clifford gates is used."""

    DistanceClassifier = 1
    LDAClassifier = 2
    SVMClassifier = 3


def compute_three_state_probability_from_complex_data(run_result: RunResult, qubit: str) -> list[xr.DataArray]:
    """Post processor to calculate and add three state probabilities to the dataset.

    Use readout average response phase settings and observations to compute the probabilities of three states,
    using the barycentric coordinates method from raw complex readout.

    The processed data will have the shape ``(len(hard_sweeps[0]), ..., len(hard_sweeps[-1]), s0, s1, ...)``,
    where s0, s1 etc. denote the dimensions of the soft sweeps.

    See :func:`compute_three_state_probability` for more details about the computation.

    Args:
        run_result: The run result containing the full dataset and run data.
        qubit: Qubit whose readout to classify.

    Returns:
        The ``RunResult`` object with :attr:`GROUND_STATE_PROBABILITY`, :attr:`EXCITED_STATE_PROBABILITY`
            and :attr:`F_STATE_PROBABILITY` added to the dataset.

    """
    dataset = run_result.dataset
    settings = run_result.run_definition.sweep_definition.settings
    if "measure_gate_name" in run_result.run_definition.options:
        measure_gate_name = run_result.run_definition.options["measure_gate_name"]
    elif "used_measure_operations" in run_result.run_definition.options:
        measure_gate_name = run_result.run_definition.options["used_measure_operations"]["terminal"]
    else:
        measure_gate_name = "measure"
    qubit_readout_data = dataset[f"{qubit}_readout"]
    props_phase = settings.get_gate_properties_for_locus(measure_gate_name, [qubit]).average_response_phase.value
    measure_phase = get_gate_node_for_locus(qubit, settings, measure_gate_name).phase.value
    classifier_params = settings.classifier[f"{qubit}_parameters"]
    state_parameters = [GROUND_STATE_PROBABILITY, EXCITED_STATE_PROBABILITY, F_STATE_PROBABILITY]
    probabilities = compute_three_state_probability(
        data=qubit_readout_data,
        average_response_phase=measure_phase - props_phase,
        average_response_g=classifier_params.state_0_mean.value * np.exp(1j * measure_phase),
        average_response_e=classifier_params.state_1_mean.value * np.exp(1j * measure_phase),
        average_response_f=classifier_params.state_2_mean.value * np.exp(1j * measure_phase),
        state_probability_parameters=state_parameters,
    )
    dataset.update(
        {
            (anno_prob := annotate(prob, param, qubit)).name: anno_prob
            for prob, param in zip(probabilities, state_parameters)
        }
    )
    return probabilities


def multi_kron(*args: np.ndarray) -> np.ndarray:
    """Cascade Kronecker product of arguments in order.

    Args:
        args: NumPy arrays of arbitrary sizes representing matrices, whose elements can be
            multiplied and added together.

    Returns:
        Kronecker product of matrices in order they appear as arguments.

    """
    args_length = len(args)
    if args_length == 1:
        return args[0]
    if args_length == 2:
        return np.kron(*args)
    idx = args_length // 2
    return np.kron(multi_kron(*args[:idx]), multi_kron(*args[idx:]))
