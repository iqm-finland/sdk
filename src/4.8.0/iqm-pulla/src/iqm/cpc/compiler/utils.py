# Copyright 2024-2026 IQM
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
"""Utility functions for the compiler passes."""

from __future__ import annotations

from collections import namedtuple
from collections.abc import Iterable
import logging
from typing import TYPE_CHECKING

from iqm.pulse.playlist.schedule import Schedule, Segment

if TYPE_CHECKING:
    from iqm.pulse.playlist.instructions import Instruction


logger = logging.getLogger(__name__)

InstructionLocation = namedtuple("InstructionLocation", ["channel_name", "index", "duration"])
"""Return type for :func:`locate_instructions`."""


def locate_instructions(
    schedule: Schedule,
    instruction_type: type[Instruction],
    min_duration: int = 0,
    *,
    channels: Iterable[str] | None = None,
) -> list[InstructionLocation]:
    """Locate specific instructions in a schedule.

    Args:
        schedule: The schedule to search.
        instruction_type: The type of the instruction to search for.
        min_duration: The minimum duration of the instruction to search for (in samples).
        channels: Names of channels in ``schedule`` to search. Iff None, search all the channels.

    Returns:
        For each located instruction, a namedtuple containing the channel name, instruction index, and duration.

    """
    if channels is None:
        channels = schedule.channels()

    result = [
        InstructionLocation(channel_name=channel, index=index, duration=inst.duration)
        for channel in channels
        for index, inst in enumerate(schedule[channel]._instructions)
        if isinstance(inst, instruction_type)
        if inst.duration >= min_duration
    ]
    return result


def replace_instruction_in_place(
    schedule: Schedule,
    channel_name: str,
    index: int,
    replacement: Iterable[Instruction],
) -> Schedule:
    """Replace an instruction in a schedule with one or more instructions.

    Args:
        schedule: The schedule to modify.
        channel_name: The name of the channel containing the instruction to replace.
        index: The index of the instruction to replace.
        replacement: Instructions to replace the original instruction with.

    Returns:
        The modified schedule.

    """
    if channel_name not in schedule:
        raise ValueError(f"No channel named {channel_name} in schedule")
    segment = schedule[channel_name]
    if 0 <= index < len(segment):
        replacement_segment = Segment(replacement)
        if segment[index].duration != replacement_segment.duration:
            raise ValueError(
                f"""Replacement duration does not match original segment duration.
Original:    {segment[index].duration}
Replacement: {replacement_segment.duration}"""
            )
        # create a new segment with old values until index + replacement_segment contents + old values after index
        new_segment = Segment([])
        new_segment.extend(segment[0:index])
        new_segment.extend(replacement_segment._instructions)
        new_segment.extend(segment[index + 1 :])
    else:
        raise ValueError(f"Index {index} is not in the segment.")

    schedule[channel_name] = new_segment
    return schedule
