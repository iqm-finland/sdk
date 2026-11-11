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
"""Helper methods for computing qubit readout contrast and optimal quadrature rotation."""

from collections.abc import Callable

import numpy as np
import xarray

from exa.common.data import DataType, Parameter
from exa.core.run_result import RunResult
from exa.core.util.dataset import add_many_data_arrays, apply_along_coordinate
from exa.core.util.pca import principal_component_analysis

CONTRAST = Parameter("contrast", "Qubit readout contrast", "", DataType.FLOAT)
PHASE_OPTIMAL_QUAD = Parameter("phase_optimal_quad", "Phase of the first principal component", "", DataType.FLOAT)


def add_contrast(default_sweep_parameters_index: int = 0) -> Callable[[RunResult], RunResult]:
    """Wrapper to create a post processor which adds the qubit readout contrast and the angle in the complex plane that
    maximizes the contrast.

    Convert the values of the first default_data_parameter into readout contrast values.

    Args:
        default_sweep_parameters_index: The index of the experiment's default sweep parameters. The corresponding
            parameter tells the axis along which to perform the principal component analysis in
            :func:`compute_contrast`. By default, the first default sweep parameter is used.

    Returns:
        Post processor to add the contrast data. See :func:`compute_contrast` for details about parameter names.

    """

    def _add_contrast(run_result: RunResult) -> RunResult:
        """The post processor created by :func:`add_contrast`.

        Args:
            run_result: The run result containing the full dataset and run data.

        Returns:
            The processed ``RunResult`` object.

        """
        dataset = run_result.dataset
        data_parameter = run_result.run_definition.default_data_parameters[0]
        primary_dim = run_result.run_definition.default_sweep_parameters[default_sweep_parameters_index]

        contrast, phase_optimal_quad = compute_contrast(dataset[data_parameter], primary_dim)
        add_many_data_arrays(dataset, (contrast, phase_optimal_quad))

        return run_result

    return _add_contrast


def compute_contrast(
    data: xarray.DataArray,
    primary_dim: str,
    contrast_param: Parameter = CONTRAST,
    poq_param: Parameter = PHASE_OPTIMAL_QUAD,
) -> tuple[xarray.DataArray, xarray.DataArray]:
    """Compute qubit readout contrast using principal component analysis (Heinsoo2019), and the angle in the complex
    plane that maximizes the contrast.

    Args:
        data: DataArray containing complex readout results. May be N-dimensional. One of the coordinates should
            have a name given by ``primary_dim``.
        primary_dim: The name of the axis in the ``data`` :class:`xr.DataArray`,
            along which to perform principal component analysis.
        contrast_param: optional custom parameter for the contrast.
        poq_param: optional custom parameter for the phase of the optimal quadrature component.
            By default, will be ``Parameter("phase_optimal_quad")``.

    Returns:
        - The calculated contrast, in same shape as ``data``.
        - The angle of maximal contrast, in same shape as ``data`` with one dimension removed

    """
    contrast_array = apply_along_coordinate(
        data_array=data,
        coord=primary_dim,
        func=lambda x, y: principal_component_analysis(y, poq_param_name=poq_param.name),
        result_parameter=contrast_param,
    )[0]

    coords_for_phase = data.coords.variables.mapping.copy()
    del coords_for_phase[primary_dim]

    phase_optimal_quad = [fp[poq_param.name] for fp in contrast_array.fit_parameters.values()]

    phase_optimal_quad_array = poq_param.build_data_array(
        np.reshape(np.ravel(phase_optimal_quad), data.shape[1:]), list(data.dims[1:]), coords_for_phase
    )
    return contrast_array, phase_optimal_quad_array
