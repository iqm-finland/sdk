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
"""Post-processing on single-shot threshold results to convert them to counter-like data."""

from collections.abc import Sequence

import numpy as np
import xarray

from exa.common.data import Parameter
from exa.core.util.dataset import annotate


def convert_single_shot_to_counter(
    single_shot_arrays: Sequence[xarray.DataArray],
    index_parameter: Parameter,
    result_parameter: Parameter,
    averaging_parameter_name: str,
) -> xarray.DataArray:
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
    counter_result = annotate(xarray.concat(results, index_array), result_parameter)
    return counter_result
