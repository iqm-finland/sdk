# Copyright 2019-2026 IQM Finland Oy
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
"""Utility functions for three state experiments."""

from typing import Any

from exa.common.data import SettingNode
from exa.common.errors import ValidationError

from .experiment_options import ExperimentOptions
from .multistate_classifier import CLASSIFIERS, compute_three_state_probability_from_complex_data
from .post_processing.excited_state_probability_computation import F_STATE_PROBABILITY
from .run_result import RunResult


def get_measure_gate_name(
    options: dict[str, Any] | None | ExperimentOptions,
    gate_option_name: str = "measure_gate_name",
    measure_added_by_mqe: str = "terminal",
) -> str:
    """Get the measure gate name from options or options dict.

    If the experiment calibrates/benchmarks the default terminal/heralding/reset-at-end measurement, the gate name is
    read from  ``options["used_measure_operations"][measure_added_by_mqe]`` and if the measure operation is explicitly
    written into the circuit, it is read from ``options[gate_option_name]``. Backwards compatibility is achieved by
    defaulting to ``"measure"``.

    Args:
         options: The options dict or object.
         gate_option_name: The gate option name, default ``"measure_gate_name"`` (controls the measure gate used in
             CUSTOM readout programming mode).
         measure_added_by_mqe: The default measure operation added by MQE, default ``"terminal"``.

    Returns:
        The measure gate (quantum operation) name.

    """
    if not options:
        return "measure"
    if gate_option_name in options:
        return options[gate_option_name]  # type: ignore[index]
    if "used_measure_operations" in options:
        return options["used_measure_operations"][measure_added_by_mqe]  # type: ignore[index]
    return "measure"


def resolve_target_name_for_f_state(run_result: RunResult, settings: SettingNode, qubit: str) -> str:
    """Resolve the name of the variable in the dataset carrying the most information about the f-state.

    If the experiment was run in single_shot mode and the classifier exists, classifies the data into three states.
    If the classifier does not exist, or the experiment was run in complex mode, tries to use the average f state
    response.
    If that does not exist, tries to use contrast.
    If the experiment was run in any other mode, raises an error.

    Args:
        run_result: A RunResult instance.
        settings: A SettingNode instance.
        qubit: A qubit for which to resolve the f-state.

    Returns:
        The name of the f-state variable.

    """
    readout_mode = run_result.run_definition.options["readout_mode"]
    measure_gate = get_measure_gate_name(run_result.run_definition.options)
    if readout_mode == "single_shot" and hasattr(settings, "classifier"):
        classifier = CLASSIFIERS[run_result.run_definition.options["classifier"]]()
        classifier.apply_to_data(
            run_result.dataset, settings, {qubit: f"{qubit}_readout_single_shot"}, measure_gate=measure_gate
        )
        target_name = f"{qubit}_{F_STATE_PROBABILITY.name}"

    elif readout_mode in ("complex", "single_shot") and hasattr(settings, "classifier"):
        probabilities = compute_three_state_probability_from_complex_data(run_result, qubit)

        target_name = str(probabilities[2].name)
    elif readout_mode in ("complex", "single_shot"):
        target_name = f"{qubit}_contrast"
    else:
        raise ValidationError(
            f"The experiment was performed in {readout_mode} readout mode."
            f" Classification of the second excited state not possible."
        )

    return target_name
