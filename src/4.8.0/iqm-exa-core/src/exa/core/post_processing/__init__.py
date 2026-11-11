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
"""Modular post processor functions that can be used in constructing a dataset from unprocessed run data.

The post processor functions need to implement the callable signature defined in
:func:`~exa.core.run_result.construct_run_result`.
"""

__all__ = [
    "AVERAGE_EXCITED_STATE_PROBABILITY",
    "AVERAGE_GG_STATE_PROBABILITY",
    "CONTRAST",
    "EXCITED_STATE_PROBABILITY",
    "GG_STATE_PROBABILITY",
    "MULTI_QUBIT_BASIS",
    "PHASE_OPTIMAL_QUAD",
    "add_average_excited_state_probabilities",
    "add_average_gg_state_probabilities",
    "add_contrast",
    "add_excited_state_probabilities",
    "add_gg_state_probabilities",
    "add_single_qubit_excited_state_probabilities",
    "transpose_data",
]

from collections.abc import Callable
from dataclasses import replace
from typing import Any

from exa.core.run_result import RunResult

from .contrast_computation import CONTRAST, PHASE_OPTIMAL_QUAD, add_contrast
from .excited_state_probability_computation import (
    AVERAGE_EXCITED_STATE_PROBABILITY,
    AVERAGE_GG_STATE_PROBABILITY,
    EXCITED_STATE_PROBABILITY,
    GG_STATE_PROBABILITY,
    MULTI_QUBIT_BASIS,
    add_average_excited_state_probabilities,
    add_average_gg_state_probabilities,
    add_excited_state_probabilities,
    add_gg_state_probabilities,
    add_single_qubit_excited_state_probabilities,
)


def transpose_data(run_result: RunResult) -> RunResult:
    """Post processor for transposing the data of the first default data parameter of ``RunResult``.

    Args:
        run_result: The run result containing the full dataset and run definition.

    Returns:
        The processed ``RunResult`` object.

    """
    dataset = run_result.dataset
    data_parameter = run_result.run_definition.default_data_parameters[0]
    dataset[data_parameter] = dataset[data_parameter].transpose()
    return run_result
