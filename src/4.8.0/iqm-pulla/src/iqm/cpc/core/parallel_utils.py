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
"""Configuring :class:`.Experiment` instances."""

from __future__ import annotations

from inspect import signature
from typing import TYPE_CHECKING, Any, Protocol

from iqm.pulse.timebox import SchedulingStrategy, TimeBox

if TYPE_CHECKING:
    from iqm.cpc.core.config import ComponentGrouping
    from iqm.pulse.builder import ScheduleBuilder


class SubTGF(Protocol):
    """Generate a TimeBox for a single parallelizable unit of QPU components."""

    def __call__(self, group: Any, builder: ScheduleBuilder, *args: Any, **kwargs: Any) -> TimeBox:
        """Generate the TimeBox.

        Args:
            group: Names of the QPU components in the parallelizable unit for which to build a TimeBox.
            builder: Tools for building the TimeBox.
            *args: Any number of additional positional parameters.
            **kwargs: Any number of additional keyword parameters,
                which will become circuit parameters for the Experiment.

        Returns:
            Timebox implementing a circuit or schedule to execute on ``group``.

        """


class FullTGF(Protocol):
    """Generate TimeBoxes for an arbitrary grouping of QPU components."""

    def __call__(self, components: Any, builder: ScheduleBuilder, *args: Any, **kwargs: Any) -> list[TimeBox]:
        """Generate the TimeBoxes.

        Args:
            components: QPU components for which to build TimeBoxes.
            builder: Tools for building the TimeBoxes.
            *args: Any number of additional positional parameters.
            **kwargs: Any number of additional keyword parameters,
                which will become circuit parameters for the Experiment.

        Returns:
            Timeboxes implementing circuits or schedules to execute on ``components``.

        """


def generate_full_tgf(group_circuit_function: SubTGF, sub_sig: Any = None) -> FullTGF:
    """Generate a full circuit generation function from a group-wise circuit generation function.

    In the returned function, ``group_circuit_function`` is applied to every parallel group (circuit
    locus) within every parallelly runnable partition of the QPU (e.g. a color group). The resulting
    TimeBoxes are then combined into a single composite TimeBox, one per partition,
    such that the TimeBox for each circuit locus is scheduled with ALAP logic.

    Args:
        group_circuit_function: The group-wise sub circuit function that acts on a single parallelly runnable group.
        sub_sig: Optionally override the signature of the group_circuit_function with this signature
            (used for backwards compatibility features).

    Returns:
        The full circuit generation function.

    """

    def circuit_parallel_group(
        components: ComponentGrouping, builder: ScheduleBuilder, *args, **kwargs
    ) -> list[TimeBox]:
        color_timeboxes: list[TimeBox] = []
        for color_group in components:
            group_timeboxes = [
                group_circuit_function(
                    group,
                    builder,
                    *args,
                    **kwargs,
                )
                for group in color_group
            ]
            color_timebox = TimeBox.composite(group_timeboxes, scheduling=SchedulingStrategy.ALAP)
            color_timeboxes.append(color_timebox)
        return color_timeboxes

    sub_sig = sub_sig or signature(group_circuit_function)
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
