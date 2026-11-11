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
"""Utilities related to xarray dataset handling."""

from iqm.cpc.core.dataset import (
    FitResults,  # noqa: F401
    add_many_data_arrays,  # noqa: F401
    annotate,  # noqa: F401
    apply_along_coordinate,  # noqa: F401
    compute_1d_fit,  # noqa: F401
    compute_2d_fit,  # noqa: F401
    convert_2d_model,  # noqa: F401
    generate_fit_parameter,  # noqa: F401
    multifit_along_coordinates,  # noqa: F401
    split_along_dimension,  # noqa: F401
    stack_along_dimension,  # noqa: F401
    turn_variable_to_coordinate,  # noqa: F401
)
