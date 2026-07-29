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
"""Collection of utilities for post processing."""

from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np
import xarray as xr

from exa.common.data.parameter import Parameter
from iqm.cpc.core.dataset import annotate
from iqm.station_control.client.station_control import SettingNode


def _get_gate_and_impl_name(prefix: str, label_to_impl: dict[str, str | None] | None) -> tuple[str | None, str | None]:
    """Get the measure QuantumOp and implementation names.

    If ``prefix`` not found in ``label_to_impl``, both are returned as ``None``. Case of ``label_to_impl == None`` is
    for backwards compatibility.
    """
    if len(prefix.split("_")) == 1:  # The default label case
        prefix = f"{prefix}__readout.result"
    if label_to_impl is None:
        return "measure", None
    op_impl = label_to_impl.get(prefix)
    if op_impl is None:
        return "measure", None
    op, impl = op_impl.split(".", maxsplit=1)
    return op, impl


def _map_acquisitions(settings: SettingNode, prefixes: list[str], readout_label_to_impl: Any) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for prefix in prefixes:
        op, impl = _get_gate_and_impl_name(prefix, readout_label_to_impl)
        impl = impl or "constant"
        component = prefix.split("_")[0]
        mapping[prefix] = settings[f"gates.{op}.{impl}.{component}.acquisition_type"].value  # type: ignore[assignment]
    return mapping


class AverageResponse(NamedTuple):
    """Average response."""

    phase: float
    """Inverted phase of average response signal in radians, to rotate signal to real axis."""
    g: float
    """Average response voltage of ground state projected to principal axis."""
    e: float
    """Average response voltage of excited state projected to principal axis."""


def compute_excitation_probability(
    data: xr.DataArray,
    average_response: AverageResponse,
    excited_state_probability: Parameter,
) -> xr.DataArray:
    """Computes excited state probabilities from complex readout voltages stored in an N-dimensional dataset.

    Note:
        It is not guaranteed that the resulting values in data array are restricted to the interval 0...1.

    Args:
        data: (`N+M`)-dimensional measurement data corresponding usually to ``'readout_i.result'``.
            Can have dimensions that represent `N` soft sweeps and `M` hard sweeps.
        average_response: The average response values of phase, ground state, and excited state.
        excited_state_probability: Parameter for the excited state probability to be computed.

    Returns:
        Data array of real numbers containing the computed excited state probabilities.
        The data array has the same shape as `data`, and it is built using `excited_state_probability`.

    """
    if average_response.g == average_response.e:
        raise ValueError(
            "'average_response.g' and 'average_response.e' are identical, not possible because of division by zero."
        )
    complex_amplitude = np.exp(-1j * average_response.phase)
    # Project values to real axis
    # After projection, the values are scaled such that the distance between the calibration points equals 1.
    # Transforms readout results from complex numbers to real numbers between 0 and 1.
    values = (np.real(data.values * complex_amplitude) - average_response.g) / (average_response.e - average_response.g)

    return excited_state_probability.build_data_array(values, list(data.dims), data.coords.variables.mapping)


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


def principal_component_analysis(
    complex_amplitudes: np.ndarray, poq_param_name: str = "phase_optimal_quad"
) -> tuple[dict[str, float | np.ndarray], np.ndarray]:
    """Use Principal component analysis to compute contrast and the angle of the first principal component
    (Heinsoo2019).

    Can be replaced with sklearn.decomposition.PCA in the future
    (sklearn library version is approx. 2x faster and more versatile).

    Args:
        complex_amplitudes: array of complex numbers to perform principal component analysis on
        poq_param_name: optional custom parameter name to be used for the phase of the first principal component.
            By default, will be ``Parameter("phase_optimal_quad")``.

    Returns:
        - dictionary with parameter name as key and phase optimal quadrature value. This format is needed so it can
            be added to a DataArray
        - resulting real contrast after applying PCA and rotating complex_amplitudes

    """
    scatter = np.stack((np.real(complex_amplitudes), np.imag(complex_amplitudes)), axis=0)
    cov = np.cov(scatter)
    _, eivec = np.linalg.eigh(cov)
    contrast = np.sum(eivec[:, 1, None] * scatter, axis=0)
    phase_optimal_quad = np.angle(eivec[0, 1] + 1j * eivec[1, 1])

    if (contrast < 0).all():
        contrast *= -1
        phase_optimal_quad = (phase_optimal_quad % (2 * np.pi)) - np.pi

    return {poq_param_name: phase_optimal_quad}, contrast


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


def convert_single_shot_to_counter(
    single_shot_arrays: Sequence[xr.DataArray],
    index_parameter: Parameter,
    result_parameter: Parameter,
    averaging_parameter_name: str,
) -> xr.DataArray:
    """Convert multiple arrays of single-component data into a single array of counter-like data. Uses big endian
    convention (same as everywhere in exa, but opposite to eg. qiskit).

    Args:
        single_shot_arrays: List of DataArrays, which should have the same dimensions, including the averaging
            dimension.
        index_parameter: Parameter to index the measured counts.
        result_parameter: Parameter to annotate the results.
        averaging_parameter_name: Dimension of the arrays along which to average, typically "repetitions".

    Returns:
        A single DataArray containing the results.

    """
    full_state = single_shot_arrays[0] * 0
    for i, array in enumerate(single_shot_arrays[::-1]):
        full_state += 2**i * array
    indices = np.unique(full_state.values)  # only use counts which where measured
    index_array = index_parameter.build_data_array(indices.astype(int))
    results = []
    for idx in indices:
        logical = full_state == idx
        counter_array = logical.mean(averaging_parameter_name)  # calculate the average counts along the chosen dim
        results.append(counter_array)
    counter_result = annotate(xr.concat(results, index_array), result_parameter)
    return counter_result
