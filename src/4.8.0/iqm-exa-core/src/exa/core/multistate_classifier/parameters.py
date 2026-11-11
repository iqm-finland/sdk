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
"""Common naming for variable labels."""

STATE_TO_LABEL = {0: "ground", 1: "excited", 2: "f"}


def get_uncorrected_state_label(component: str, state: int) -> str:
    """Get data_variable label for uncorrected state.

    Args:
        component: The component the label belongs to.
        state: The state index.

    Returns:
        The data variable label for the uncorrected state probability.

    """
    return f"{component}_uncorrected_{STATE_TO_LABEL[state]}_state_probability"


def get_corrected_state_label(component: str, state: int) -> str:
    """Get data_variable label for corrected state.

    Args:
        component: The component the label belongs to.
        state: The state index.

    Returns:
        The data variable label for the assignment-corrected state probability.

    """
    return f"{component}_{STATE_TO_LABEL[state]}_state_probability"


def get_scaled_average_state_label(component: str, state: int) -> str:
    """Get data_variable label for scaled average state.

    Args:
        component: The component the label belongs to.
        state: The state index.

    Returns:
        The data variable label for the scaled average state probability.

    """
    return f"{component}_scaled_avg_{STATE_TO_LABEL[state]}_state_probability"
