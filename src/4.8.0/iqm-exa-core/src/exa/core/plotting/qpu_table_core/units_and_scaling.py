#  *****************************************************************************
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
#  *****************************************************************************
"""Generates units and scales for given collection of values and unit from Observation."""

import numpy as np

from exa.core.util.formatting import format_unit_string, get_prefix, get_scaling


def create_unit_string(all_values_array: list | np.ndarray, unit: str = "", enclosure: str = "(") -> tuple[float, str]:
    """Creates a string that contains the units for that coordinates in the data_array.

    Args:
        all_values_array: Array-like with all the data and the coordinate information.
        unit: SI base unit.
        enclosure: enclosing bracket

    Returns:
        Scaling of the data, SI unit with a prefix.

    """
    scaling = 1.0
    all_values_array = np.asarray(all_values_array)
    if unit:
        unit_prefix = get_prefix(all_values_array)
        scaling = get_scaling(all_values_array)
        unit_string = format_unit_string(unit_prefix, unit, enclosure=enclosure)
    else:
        unit_string = ""
    return scaling, unit_string
