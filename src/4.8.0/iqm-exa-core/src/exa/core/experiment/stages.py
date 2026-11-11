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
"""Compiler stages for Experiments."""

from dataclasses import replace

import numpy as np

from exa.common.data import SettingNode
from exa.core.experiment_definitions import ComponentGrouping
from iqm.cpc.compiler.standard_stages import (
    ONLY_PROBE_KEY,
)
from iqm.cpc.core.run_result import RunResult
from iqm.pulse.builder import ScheduleBuilder
from iqm.pulse.gates.measure import multiplex
from iqm.pulse.timebox import TimeBox


def _measure_timebox(
    builder: ScheduleBuilder,
    settings: SettingNode,
    op_name: str,
    impl_name: str | None,
    components_to_subscribe: list[str],
    components_to_probe: list[str],
    feedback_key: str,
) -> TimeBox:
    """Get measurement timebox.

    Args:
        builder: The schedule builder.
        settings: The settings tree.
        op_name: The measure operation name.
        impl_name: The measure implementation name. ``None`` means "take the default implementation for each locus)
        components_to_subscribe: The components to measure (and gather the measurement data from) in the
            terminal measurement.
        components_to_probe: The components to probe with the measurement pulse without gathering data. If
            a component exists both in this list and in ``additionally_subscribed_components`` the measurement data will
            be gathered from it.
        feedback_key: Reset feedback key for the terminal measurement.

    Returns:
        The measurement timebox.

    """

    def _default_has_calibration(c: str, impl_name: str | None) -> bool:
        """True iff the (given or default) implementation of the measurement operation
        has a settings node for locus ``(c,)``.
        """
        if not impl_name:
            impl_name = builder.op_table[op_name].get_default_implementation_for_locus((c,))
        return c in settings.gates[op_name][impl_name].subtrees

    components_to_subscribe = [c for c in components_to_subscribe if _default_has_calibration(c, impl_name)]
    naked_measure_boxes = builder.get_implementation(op_name, tuple(components_to_subscribe), impl_name=impl_name)(
        feedback_key=feedback_key
    )
    components_to_probe = [
        c
        for c in components_to_probe
        if _default_has_calibration(c, impl_name) and c not in naked_measure_boxes.measure_components  # type: ignore[union-attr]
    ]
    if components_to_probe:
        just_probed = builder.get_implementation(op_name, tuple(components_to_probe), impl_name=impl_name)(
            key=ONLY_PROBE_KEY, feedback_key=feedback_key
        )
        naked_measure_boxes = multiplex([naked_measure_boxes, just_probed])  # type: ignore[list-item]
    measure_timebox = TimeBox.composite(naked_measure_boxes)
    if feedback_key:
        return measure_timebox + [
            builder.cc_prx((q,))(np.pi, 0.0, q, feedback_key=feedback_key)  # type: ignore[misc]
            for q in components_to_subscribe + components_to_probe
        ]
    return measure_timebox


def add_terminal_measurement(  # noqa: PLR0913
    timeboxes: list[TimeBox],
    builder: ScheduleBuilder,
    settings: SettingNode,
    components: ComponentGrouping,
    additionally_subscribed_components: list[str],
    additionally_probed_components: list[str],
    measure_operation: str = "measure_fidelity",
    measure_implementation: str | None = None,
    reset_feedback_key: str = "",
    skip_terminal_measurement: bool = False,
) -> list[TimeBox]:
    """Add a terminal measurement to the circuits.

    A barrier is added in between the circuit and the terminal measure gate to make sure that the terminal measure gate
    does not overlap with the circuit.

    Args:
        timeboxes: The circuits (represented as TimeBoxes).
        builder: The schedule builder.
        settings: The settings tree.
        components: The active run components.
        additionally_subscribed_components: The components to measure (and gather the measurement data from) in the
            terminal measurement.
        additionally_probed_components: The components to probe with the measurement pulse without gathering data. If
            a component exists both in this list and in ``additionally_subscribed_components`` the measurement data will
            be gathered from it.
        measure_operation: The name of the measure operation for the terminal measurement.
        measure_implementation: The name of the measure implementation for the terminal measurement.
        reset_feedback_key: Reset feedback key for the terminal measurement.
        skip_terminal_measurement: Whether to skip adding the terminal measurement.

    Returns:
        The new timeboxes with the terminal measurement added.

    """
    if skip_terminal_measurement:
        return timeboxes
    if "subscribed_groups" in settings.stages.subtrees and settings.stages.subscribed_groups.settings:
        additionally_subscribed_components = []
        for group_setting in settings.stages.subscribed_groups.settings.values():
            additionally_subscribed_components.extend(group_setting.value)
        additionally_subscribed_components = list(set(additionally_subscribed_components))
    if not additionally_subscribed_components and not additionally_probed_components:
        return timeboxes
    components_to_subscribe = additionally_subscribed_components
    components_to_probe = [c for c in additionally_probed_components if c not in components_to_subscribe]
    new_timeboxes = []
    measure_timebox = _measure_timebox(
        builder,
        settings,
        measure_operation,
        measure_implementation,
        components_to_subscribe,
        components_to_probe,
        reset_feedback_key,
    )

    barrier_components = {*components.flatten(), *additionally_subscribed_components, *additionally_probed_components}
    barrier = builder.get_implementation("barrier", list(barrier_components))()
    new_timeboxes.extend([timebox | barrier | measure_timebox for timebox in timeboxes])
    return new_timeboxes


def post_process_target_variables(run: RunResult) -> RunResult:
    """Set the default data variables in run."""
    # FIXME: this target_data_variables -> default_data_parameters juggling is pretty stupid. Figure out if
    # it is actually needed
    additional_run_properties = run.run_definition.additional_run_properties
    target_data_parameters = additional_run_properties["target_data_parameters"]  # type: ignore[index]
    if target_data_parameters:
        run.run_definition = replace(run.run_definition, default_data_parameters=target_data_parameters)
    else:
        excited_state_parameters = [
            f"{c}_excited_state_probability" for c in additional_run_properties["readout_components"]
        ]

        run.run_definition = replace(run.run_definition, default_data_parameters=excited_state_parameters)
    return run
