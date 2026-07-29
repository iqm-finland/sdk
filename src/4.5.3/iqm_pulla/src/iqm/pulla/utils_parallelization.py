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
"""Utility functions for IQM Pulla."""

from collections.abc import Callable
from inspect import signature

from iqm.cpc.compiler._utils.stages import CircuitGenerationFunction
from iqm.cpc.core.config import ComponentGrouping
from iqm.pulse.builder import ScheduleBuilder
from iqm.pulse.timebox import SchedulingStrategy, TimeBox


def parallelize_timeboxes(group_circuit_function: Callable[..., TimeBox]) -> CircuitGenerationFunction:
    """Generate a full circuit function from a group-wise circuit function.

    The group-wise function is applied to every parallel group within every parallelly runnable partition of the
    QPU (color) such that the subcircuits of each parallel group are scheduled with ALAP logic.

    Args:
        group_circuit_function: The group-wise sub circuit function that acts on a single parallelly runnable group.
            The function should have the args `group` (of type ``tuple[str, ...]``), containing the names of the
            components in the parallel group, and the schedule builder, `builder` (of type :class:`.ScheduleBuilder`).
            In addition, it can have any number of additional arguments, which will become circuit parameters.

    Returns:
        The full circuit generation function.

    """

    def circuit_parallel_group(
        components: ComponentGrouping, builder: ScheduleBuilder, *args, **kwargs
    ) -> list[TimeBox]:
        color_timeboxes: list[TimeBox] = []
        for color_group in components:
            pair_timeboxes = [
                group_circuit_function(
                    group,
                    builder,
                    *args,
                    **kwargs,
                )
                for group in color_group
            ]
            color_timebox = TimeBox.composite(pair_timeboxes, scheduling=SchedulingStrategy.ALAP)
            color_timeboxes.append(color_timebox)
        return color_timeboxes

    sub_sig = signature(group_circuit_function)
    # find how many generic args sub-circuit has (the last generic arg is always "builder")
    num_subcircuit_generic_args = next(i + 1 for i, k in enumerate(sub_sig.parameters.keys()) if k == "builder")
    full_function = circuit_parallel_group
    full_sig = signature(full_function)
    # parse the full function signature from the generic first 2 args of the wrapper
    # i.e. components, builder and the specific arguments of the subfunction.
    new_sig = full_sig.replace(
        parameters=tuple(full_sig.parameters.values())[:2]
        + tuple(sub_sig.parameters.values())[num_subcircuit_generic_args:]
    )
    full_function.__signature__ = new_sig  # type: ignore[attr-defined]
    return full_function
