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
"""Helper methods for computing state probabilities from readout voltage or multi-qubit counter results."""

from collections.abc import Hashable
from typing import Any

import numpy as np
import xarray as xr

from exa.common.data import DataType, Parameter
from exa.core import get_gate_node_for_locus
from exa.core.run_result import RunResult
from exa.core.util.dataset import annotate

MULTI_QUBIT_BASIS = Parameter("multi_qubit_basis", "Multi qubit basis states encoded as binary numbers")
"""Multi qubit basis states encoded as integers via the binary representation."""

GROUND_STATE_PROBABILITY = Parameter("ground_state_probability", "Ground state probability", "", DataType.FLOAT)
"""Ground state probability of the target qubit. Probabilities of each of the three states, based on normalization using
calibration pulses."""

EXCITED_STATE_PROBABILITY = Parameter("excited_state_probability", "Excited state probability", "", DataType.FLOAT)
"""Excited state probability of the target qubit. Probability of the first excited state, based on normalization using
calibration pulses."""

HERALDED_EXCITED_STATE_PROBABILITY = Parameter(
    "heralded_excited_state_probability", "Excited state probability", "", DataType.FLOAT
)
"""Heralded excited state probability of the target qubit. Probability of the first excited state,
based on normalization using calibration pulses and postselected based on a heralding pulse."""

F_STATE_PROBABILITY = Parameter("f_state_probability", "F state probability", "", DataType.FLOAT)
"""State probability of the target qubit. Probabilities of each of the three states, based on normalization using
calibration pulses."""

AVERAGE_EXCITED_STATE_PROBABILITY = Parameter(
    "average_excited_state_probability", "Average excited state probability", "", DataType.FLOAT
)
"""Excited state probability of the target qubit averaged over the first dimension, i.e., innermost sweep.
For standard and interleaved rb experiments, this corresponds to the excited state probability averaged across
different Clifford sequences, whereas for the purity rb experiment this corresponds to the excited state probability
averaged across an artificial extra dimension."""

GG_STATE_PROBABILITY = Parameter("gg_state_probability", "GG state probability", "", DataType.FLOAT)
"""Probability of both (target) qubits being in the ground state."""

AVERAGE_GG_STATE_PROBABILITY = Parameter(
    "average_gg_state_probability", "Average GG state probability", "", DataType.FLOAT
)
"""GG state probability of the (target) qubits averaged over the first dimension, i.e., innermost sweep."""


def add_excited_state_probabilities(run_result: RunResult) -> RunResult:
    """Post processor to calculate and add excited state probability to the dataset.

    * Use readout average response phase settings and observations to compute the probability of the excited state.

    * The default data parameter is used for in the computing the excited state probability.

    The processed data will have the shape ``(len(hard_sweeps[0]), ..., len(hard_sweeps[-1]), s0, s1, ...)``,
    where s0, s1 etc. denote the dimensions of the soft sweeps. Note that len(hard_sweeps[0]) corresponds to
    the dimension of the default sweep parameter.

    See :func:`compute_excitation_probability` for more details about the computation.

    Args:
        run_result: The run result containing the full dataset and run data.

    Returns:
        The ``RunResult`` object with :attr:`EXCITED_STATE_PROBABILITY` added to the dataset.

    """
    dataset = run_result.dataset
    settings = run_result.run_definition.sweep_definition.settings
    data_parameter_name = run_result.run_definition.default_data_parameters[0]
    qubit = run_result.run_definition.components[0]
    pulse_phase = get_gate_node_for_locus(qubit, settings, "measure").phase.value
    gate_properties = settings.get_gate_properties_for_locus("measure", qubit)
    probabilities = compute_excitation_probability(
        data=dataset[data_parameter_name],
        average_response_phase=pulse_phase - gate_properties.average_response_phase.value,
        average_response_g=gate_properties.average_response_g.value,
        average_response_e=gate_properties.average_response_e.value,
        excited_state_probability_parameter=EXCITED_STATE_PROBABILITY,
    )
    dataset.update({probabilities.name: probabilities})

    return run_result


def add_gg_state_probabilities(run_result: RunResult) -> RunResult:
    """Post processor to read the GG state probabilities for raw complex readout data of an RB experiment using the
    multi-qubit counter results.

    Args:
        run_result: The run result containing the full dataset and run data.
            Assuming the dataset has the parameter :attr:`MULTI_QUBIT_BASIS`.

    Returns:
        The ``RunResult`` object with :attr:`GG_STATE_PROBABILITY` added to the dataset.

    """
    dataset = run_result.dataset
    data_parameter_name = run_result.run_definition.default_data_parameters[0]

    gg_probabilities = dataset[data_parameter_name].sel({MULTI_QUBIT_BASIS.name: 0}).data

    coords_to_values_mapping = dataset[data_parameter_name].coords.variables.mapping
    dims_gg = list(dataset[data_parameter_name].dims[1:])
    coords_to_values_mapping_gg = {dim: coords_to_values_mapping[dim] for dim in dims_gg}
    gg_state_probabilities_parameter = GG_STATE_PROBABILITY.build_data_array(
        gg_probabilities, dims_gg, coords_to_values_mapping_gg
    )
    dataset.update({gg_state_probabilities_parameter.name: gg_state_probabilities_parameter})

    return run_result


def add_single_qubit_excited_state_probabilities(run_result: RunResult) -> RunResult:
    """Post processor to compute and add the single-qubit excited state probabilities from the multi qubit counter
    results to the dataset.

    Args:
        run_result: The run result containing the full dataset and run data.
            Assuming the raw data in the dataset has :attr:`MULTI_QUBIT_BASIS`.
            Also, the default data parameter should be the counter's result parameter.

    Returns:
        The ``RunResult`` object with single-qubit excited state probabilities added to the dataset.

    Returns:
        - Dataset containing the single-qubit excited state probabilities.

    """
    dataset = run_result.dataset
    data_parameter_name = run_result.run_definition.default_data_parameters[0]
    if len(run_result.run_definition.components) == 1:
        # This case is used when we characterise qubit flux tails and therefore the qubit and coupler component are the
        # same physical element on the chip. In this case, run_result.run_definition.components has only one entry,
        # which is a qubit.
        qubits = run_result.run_definition.components
    else:
        # This case removes the coupler from the list of components, which is necessary for ncphase_base and flux_pulse
        # distortion_long when the flux component (=coupler) is different from the qubit probe component (=qubit).
        qubits = run_result.run_definition.components[:-1]

    for qubit in qubits:
        excited_probability = compute_single_qubit_probability_from_counter_results(
            dataset[data_parameter_name], MULTI_QUBIT_BASIS.name, qubits[::-1], qubit
        )
        new_data = EXCITED_STATE_PROBABILITY.build_data_array(
            excited_probability.data,
            excited_probability.dims,  # type: ignore[arg-type]
            excited_probability.coords.variables.mapping,  # type: ignore[arg-type]
        )
        dataset.update({qubit + "_" + EXCITED_STATE_PROBABILITY.name: new_data})

    return run_result


def add_average_excited_state_probabilities(run_result: RunResult) -> RunResult:
    """Post processor to add average excited state probabilities using already calculated excited state probabilities.

    See more details in :func:`compute_average_state_probabilities`.

    Args:
        run_result: The run result containing the full dataset and run data
            The dataset should already have the parameter :attr:`EXCITED_STATE_PROBABILITY` added.

    Returns:
        The ``RunResult`` object with :attr:`AVERAGE_EXCITED_STATE_PROBABILITY` added to the dataset.

    """
    dataset = run_result.dataset
    data_parameter_name = run_result.run_definition.default_data_parameters[0]
    coords_to_values_mapping = dataset[data_parameter_name].coords.variables.mapping

    average_excited_state_probabilities = compute_average_state_probabilities(
        dataset[EXCITED_STATE_PROBABILITY.name], AVERAGE_EXCITED_STATE_PROBABILITY, coords_to_values_mapping
    )
    dataset.update({average_excited_state_probabilities.name: average_excited_state_probabilities})

    return run_result


def add_average_gg_state_probabilities(run_result: RunResult) -> RunResult:
    """Post processor to add average GG state probabilities using already calculated GG state probabilities.

    The shape of the added data is ``(len(hard_sweeps[0]), ..., len(hard_sweeps[-1]), s0, s1, ...)``, where s0, s1 etc
    denote the dimensions of the soft sweeps.

    See more details in :func:`compute_average_state_probabilities`.

    Args:
        run_result: The run result containing the full dataset and run data.
            The dataset should already have the parameter :attr:`GG_STATE_PROBABILITY` added.

    Returns:
        The ``RunResult`` object with :attr:`AVERAGE_GG_STATE_PROBABILITY` added to the dataset.

    """
    dataset = run_result.dataset
    data_parameter_name = run_result.run_definition.default_data_parameters[0]
    coords_to_values_mapping = dataset[data_parameter_name].coords.variables.mapping

    average_excited_state_probabilities = compute_average_state_probabilities(
        dataset[GG_STATE_PROBABILITY.name], AVERAGE_GG_STATE_PROBABILITY, coords_to_values_mapping
    )
    dataset.update({average_excited_state_probabilities.name: average_excited_state_probabilities})

    return run_result


def compute_excitation_probability(
    data: xr.DataArray,
    average_response_phase: float,
    average_response_g: float,
    average_response_e: float,
    excited_state_probability_parameter: Parameter,
) -> xr.DataArray:
    """Computes excited state probabilities from complex readout voltages stored in an N-dimensional dataset.

    Note:
        It is not guaranteed that the resulting values in data array are restricted to the interval 0...1.

    Args:
        data: (`N+M`)-dimensional measurement data corresponding usually to ``'readout_i.result'``.
            Can have dimensions that represent `N` soft sweeps and `M` hard sweeps.
        average_response_phase: Inverted phase of average response signal in radians, to rotate signal to real axis.
        average_response_g: Average response voltage of ground state projected to principal axis.
        average_response_e: Average response voltage of excited state projected to principal axis.
        excited_state_probability_parameter: Parameter for the excited state probability to be computed.

    Returns:
        Data array of real numbers containing the computed excited state probabilities.
        The data array has the same shape as `data`, and it is built using `excited_state_probability_parameter`.

    """
    if average_response_g == average_response_e:
        raise ValueError(
            "'average_response_g' and 'average_response_e' are identical, not possible because of division by zero"
        )
    angle = np.exp(-1j * average_response_phase)
    # Project values to real axis
    # After projection, the values are scaled such that the distance between the calibration points equals 1.
    # Transforms readout results from complex numbers to real numbers between 0 and 1.
    values = (np.real(data.values * angle) - average_response_g) / (average_response_e - average_response_g)

    return excited_state_probability_parameter.build_data_array(values, list(data.dims), data.coords.variables.mapping)


def compute_excitation_probability_threshold(
    data_array: xr.DataArray,
    single_shot_01_error: float,
    single_shot_10_error: float,
    excited_state_probability_parameter: Parameter,
) -> xr.DataArray:
    """Computes estimated excited state probabilities from averaged thresholded readout results.

    Note:
        It is not guaranteed that the estimated probabilities are restricted to the interval [0, 1].

    Args:
        data_array: Averaged thresholded measurement data.
        single_shot_01_error: Probability of measuring 1 if the 0 state was prepared.
        single_shot_10_error: Probability of measuring 0 if the 1 state was prepared.
        excited_state_probability_parameter: Parameter for the excited state probability to be computed.

    Returns:
        Estimated excited state probabilities.
        The array has the same shape as ``data_array``, and it is built using ``excited_state_probability_parameter``.

    """
    values = (data_array.values - single_shot_01_error) / (1 - single_shot_10_error - single_shot_01_error)
    return excited_state_probability_parameter.build_data_array(
        values, list(data_array.dims), data_array.coords.variables.mapping
    )


def compute_three_state_probability(
    data: xr.DataArray,
    average_response_phase: float,
    average_response_g: float,
    average_response_e: float,
    average_response_f: complex,
    state_probability_parameters: list[Parameter],
) -> list[xr.DataArray]:
    """Computes excited state probabilities from complex readout voltages stored in an N-dimensional dataset.

    Note:
        It is not guaranteed that the resulting values in data array are restricted to the interval 0...1.

    Args:
        data: (`N+M`)-dimensional measurement data corresponding usually to ``'readout_i.result'``.
            Can have dimensions that represent `N` soft sweeps and `M` hard sweeps.
        average_response_phase: Inverted phase of average response signal in radians, to rotate signal to real axis.
        average_response_g: Average response voltage of ground state projected to principal axis.
        average_response_e: Average response voltage of excited state projected to principal axis.
        average_response_f: Average response voltage of second excited state, complex.
        state_probability_parameters: Parameters for the state probability to be computed.

    Returns:
        Data array of real numbers containing the computed excited state probabilities.
        The data array has the same shape as `data`, and it is built using `excited_state_probability_parameter`.

    """
    if average_response_g == average_response_e:
        raise ValueError(
            "'average_response_g' and 'average_response_e' are identical, not possible because of division by zero"
        )
    angle = np.exp(-1j * average_response_phase)
    rotated_data = data.values * angle
    gef_triangle = np.array([average_response_g, average_response_e, average_response_f])
    gef_area = np.imag(np.linalg.det(np.array([gef_triangle, np.conjugate(gef_triangle), np.ones_like(gef_triangle)])))

    lambda_g_triangle = np.array(
        [rotated_data, np.ones_like(rotated_data) * average_response_e, np.ones_like(rotated_data) * average_response_f]
    )
    lambda_e_triangle = np.array(
        [np.ones_like(rotated_data) * average_response_g, rotated_data, np.ones_like(rotated_data) * average_response_f]
    )
    lambda_f_triangle = np.array(
        [np.ones_like(rotated_data) * average_response_g, np.ones_like(rotated_data) * average_response_e, rotated_data]
    )

    lambdas = []
    triangles = [lambda_g_triangle, lambda_e_triangle, lambda_f_triangle]
    for triangle, param in zip(triangles, state_probability_parameters):
        lambda_array = (
            np.imag(np.linalg.det(np.array([triangle, np.conjugate(triangle), np.ones_like(triangle)]).T).T) / gef_area
        )
        lambdas.append(annotate(data.real * 0.0 + lambda_array, param))

    return lambdas


def compute_single_qubit_probability_from_counter_results(
    data: xr.DataArray,
    qubit_basis_coordinate: str,
    qubits_list: list[str],
    qubit: str,
    get_excited_state: bool = True,
) -> xr.DataArray:
    """Compute single qubit state probabilities from the multi qubit counter results.

    For N qubits, the multi qubit states are written as binary numbers (of length N): `00...0, 00...1, ..., 11...0`.
    The counter returns the probabilities `P(00...0), P(00...1), ... P(11...1)`. The single qubit probability that
    a specific qubit is in the excited state can be obtained as the sum of all the probabilities where that qubit
    is in the excited state. For three qubits, e.g., the probability that the first qubit is in excited state is
    given as `P(E_1) = P(001) + P(011) + P(101) + P(111)`, and similarly the probability for the second qubit is
    `P(E_2) = P(010) + P(011) + P(110) + P(111)`.

    Args:
        data: Data containing the multi qubit counter results.
        qubit_basis_coordinate: The name of the multi qubit basis coordinate.
        qubits_list: The full list of qubit names for which the counter was used. The order of the list defines the
            order of the multi qubit states such that the first qubit in the list corresponds to the smallest digit
            in the binary representation of the multi qubit states.
        qubit: The name of the qubit for which to return the probability.
        get_excited_state: If set `False`, will return the ground state probability. Otherwise returns the exicted
            state.

    Returns:
        - Data array containing the given single qubit probability.

    """
    multi_qubit_states = range(2 ** len(qubits_list))
    qubit_state_mask = 2 ** qubits_list.index(qubit)
    if get_excited_state:
        states_to_sum = [state for state in multi_qubit_states if state & qubit_state_mask]
    else:
        states_to_sum = [state for state in multi_qubit_states if not state & qubit_state_mask]
    return data.sel({qubit_basis_coordinate: states_to_sum}).sum(dim=qubit_basis_coordinate)


def compute_average_state_probabilities(
    state_probabilities: xr.DataArray,
    average_state_probability_parameter: Parameter,
    coords_to_values_mapping: dict[Hashable, Any],
    dimension_for_averaging: str | None = None,
) -> xr.DataArray:
    """Calculate average state probabilities for raw complex readout data of an RB experiment using already calculated
    state probabilities.

    Excited state probability should be added to the dataset before executing this function.

    The procedure goes roughly as follows.
    Average over a dimension in the data. Often, this corresponds to averaging across different Clifford
    sequences. In purity RB this corresponds to taking an (additional) average of the selected state probabilities
    (``probability_parameter``) obtained in a state tomography measurement for a given final rotation.

    Args:
        state_probabilities: The data for which average state probabilities are computed.
        average_state_probability_parameter: The parameter that is used to build the returned data array.
        coords_to_values_mapping: Coordinate names to coordinate values mapping.
        dimension_for_averaging: Name of the dimension to take the average over. If not provided, will average
            over the first dimension.

    Returns:
        An instance of :class:`xarray.DataArray` built with parameter ``average_state_probability_parameter``.

    """
    # Coordinate variable values corresponding to the remaining dimensions
    dims = state_probabilities.dims
    avg_index = dims.index(dimension_for_averaging) if dimension_for_averaging else 0
    remaining_coords_mapping = {d: coords_to_values_mapping[d] for d in dims if d != dims[avg_index]}

    # Average over the innermost dimension
    average_state_probabilities = np.mean(state_probabilities.values, axis=avg_index)
    return average_state_probability_parameter.build_data_array(
        average_state_probabilities, list(remaining_coords_mapping.keys()), remaining_coords_mapping
    )
