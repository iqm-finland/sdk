# Copyright 2024 IQM
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
"""Instruction schedules for controlling the instruments."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, KeysView, Mapping
from dataclasses import dataclass

from iqm.pulse.playlist.channel import ChannelProperties
from iqm.pulse.playlist.instructions import (
    Block,
    ConditionalInstruction,
    FluxPulse,
    Instruction,
    IQPulse,
    ReadoutTrigger,
    RealPulse,
    VirtualRZ,
    Wait,
)

TOLERANCE: float = 1e-12
"""TODO: remove when COMP-1281 is done."""
TOLERANCE_DECIMALS = 12


@dataclass(frozen=True)
class Nothing(Instruction):
    """Used to extend a control channel in time, without blocking it, during scheduling.

    Can overlap with any other Instruction.
    Converted to a :class:`.Wait` instruction at the end of scheduling.
    """


class Segment:
    """Sequence of instructions, for a single channel.

    Basically a list[Instruction] that keeps track of the time duration of its contents.

    Args:
        instructions: contents of the segment
        duration: If None, compute the sum of the durations of ``instructions``.
            Otherwise, the time duration (in samples) of the segment, which must match
            the aforementioned sum if the Segment is still to be modified.

    """

    def __init__(self, instructions: Iterable[Instruction], *, duration: int | None = None):
        self._duration: int = 0
        """total duration of the segment, in samples"""
        self._instructions: list[Instruction] = []
        """contents"""

        if duration is None:
            # compute the total duration
            self.extend(instructions)
        else:
            # use the given duration, for performance
            self._instructions.extend(instructions)
            self._duration = duration

    def __iter__(self) -> Iterator[Instruction]:
        return iter(self._instructions)

    def __len__(self) -> int:
        return len(self._instructions)

    def __getitem__(self, key):  # noqa: ANN001
        return self._instructions[key]

    def __reversed__(self):
        return reversed(self._instructions)

    def copy(self) -> Segment:
        """Make an independent copy of the segment.

        Mutating the original must not affect the copy, or vice versa.
        Instructions are immutable, so they need not be copied.

        Returns:
            copy of the segment

        """
        return Segment(self._instructions, duration=self.duration)

    def append(self, instruction: Instruction) -> None:
        """Append an instruction to the end of the segment."""
        self._instructions.append(instruction)
        self._duration += instruction.duration

    def prepend(self, instruction: Instruction) -> None:
        """Prepend an instruction to the beginning of the segment."""
        self._instructions.insert(0, instruction)
        self._duration += instruction.duration

    def extend(self, instructions: Iterable[Instruction]) -> None:
        """Append all instructions from an iterable to the end of the segment."""
        for i in instructions:
            self.append(i)

    def pop(self, idx: int = -1) -> Instruction:
        """Remove and return the instruction at the given index of the segment."""
        instruction = self._instructions.pop(idx)
        self._duration -= instruction.duration
        return instruction

    @property
    def duration(self) -> int:
        """Sum of the durations of the instructions in the segment, in samples."""
        return self._duration


class Schedule:
    """Set of synchronously executed AWG/readout programs that start on a single trigger.

    Consists of a number of *channels*, each containing a :class:`.Segment` of
    :class:`.Instruction` s for a specific controller.  For each channel, maps the channel name to
    its Segment of Instructions.

    Mutable. To make an independent copy, use :meth:`copy`.

    Args:
        contents: mapping from channel name to a list of Instructions for that channel
        duration: Optional precomputed duration (in samples). In cases where the duration is known and performance
            is critical, the duration can be given in the constructor, allowing one to skip computing it.

    """

    def __init__(
        self,
        contents: Mapping[str, Iterable[Instruction]] | None = None,
        *,
        duration: int | None = None,
    ):
        if contents is None:
            contents = {}
        self._contents = {channel: Segment(instructions) for channel, instructions in contents.items()}
        self._duration = duration

    def __getitem__(self, key: str) -> Segment:
        return self._contents[key]

    def __setitem__(self, key: str, value: Segment):
        self._contents[key] = value

    def __iter__(self):
        return iter(self._contents)

    def __len__(self) -> int:
        return len(self._contents)

    @property
    def duration(self) -> int:
        """The maximum duration of the Schedule's channels, in samples.

        Computed only when needed and cached for performance.
        """
        if self._duration is None:
            self._duration = max((segment.duration for segment in self._contents.values()), default=0)
        return self._duration

    def duration_in_seconds(self, channel_properties: dict[str, ChannelProperties]) -> float:
        """Schedule duration in seconds, taking into account the sample rates of the channels.

        Args:
            channel_properties: channel properties.

        Returns:
            schedule duration (in seconds)

        """
        return max(
            (channel_properties[channel].duration_to_seconds(segment.duration) for channel, segment in self.items()),
            default=0.0,
        )

    def pprint(self, time_unit: int = 16) -> str:
        """Fixed-width character graphics representation of the Schedule.

        Assumes the :attr:`Instruction.duration` s are in samples.

        Args:
            time_unit: unit of time represented by a single symbol (in samples)

        """
        if (length := self.duration / time_unit) > 1000:
            return f"Schedule too long to print ({int(length)} symbols)."

        symbols = {
            Wait: ("|", " "),
            Block: ("B", "."),
            IQPulse: ("!", "="),
            RealPulse: ("R", "-"),
            FluxPulse: ("F", "-"),
            VirtualRZ: ("Z", "~"),
            ConditionalInstruction: ("C", "*"),
            ReadoutTrigger: ("R", "="),
        }
        default_symbols = ("?", " ")  # everything else
        s = ""
        key_width = max(map(len, self), default=0) + 2
        for channel_name, segment in self._contents.items():
            s += f"{channel_name:{key_width}}{len(segment):3}:"
            t: float = 0
            printed_time: float = 0
            for instruction in segment:
                chars = symbols.get(type(instruction), default_symbols)
                t += instruction.duration
                # one symbol per time unit
                n = round((t - printed_time) / time_unit)
                printed_time += n * time_unit
                # the start of a instruction is denoted using a different symbol
                if n > 0:
                    s += chars[0]
                s += chars[1] * (n - 1)
            s += "|\n"
        return s

    def items(self):  # noqa: ANN201
        """Iterator over the schedule channel names and segments."""
        return self._contents.items()

    def channels(self) -> KeysView:
        """The channels occupied in ``self``."""
        return self._contents.keys()

    def copy(self) -> Schedule:
        """Make an independent copy of the schedule.

        Mutating the original must not affect the copy, or vice versa.
        Instructions are immutable, so they need not be copied.

        Returns:
            copy of the schedule

        """
        new = Schedule(duration=self.duration)
        new._contents = {channel: segment.copy() for channel, segment in self._contents.items()}
        return new

    def add_channels(self, channel_names: Iterable[str]) -> None:
        """Add new empty channels to the schedule.

        If a given channel (identified by its controller name) already exist in the schedule,
        it is unchanged.

        Modifies ``self``.

        Args:
            channel_names: names of the controllers for which empty channels are added

        """
        for channel in channel_names:
            if channel not in self._contents:
                self._contents[channel] = Segment([])

    def append(self, channel: str, instruction: Instruction) -> None:
        """Append a single Instruction to a specific channel in the Schedule.

        Args:
            channel: name of the channel to append the instruction to
            instruction: instruction to append

        """
        self._contents[channel].append(instruction)
        self._duration = None  # duration is no longer known and must be computed again

    def extend(self, channel: str, instructions: Iterable[Instruction]) -> None:
        """Append given Instruction to a specific channel in the Schedule.

        Args:
            channel: name of the channel to append the instructions to
            instructions: instructions to append

        """
        self._contents[channel].extend(instructions)
        self._duration = None  # duration is no longer known and must be computed again

    def front_pad(self, to_duration: int) -> Schedule:
        """Modifies the schedule in place by front-padding it with :class:`.Wait` instructions.

        NOTE: this method cannot be used when there are variable sampling rates present in the schedule. In that
        case, use the method ``front_pad_in_seconds``.

        Args:
            to_duration: duration of the resulting schedule, in samples
        Returns:
            ``self``, with the padding

        """
        duration = self.duration
        wait: int = to_duration - duration
        if wait < 0:
            raise ValueError(f"Target duration {to_duration} is shorter than the current schedule duration {duration}")
        if wait != 0:
            self._duration = wait + duration
            self._contents = {
                channel: Segment([Wait(duration=wait)] + seg._instructions, duration=wait + seg.duration)
                for channel, seg in self._contents.items()
            }
        return self

    def front_pad_in_seconds(self, to_duration: float, channel_properties: dict[str, ChannelProperties]):  # noqa: ANN201
        """Modifies the schedule in place by front-padding it with :class:`.Wait` instructions.

        The new duration is given in seconds, and this method works also with variable sample rates.

        Args:
            channel_properties: channel properties.

        Returns:
            ``self``, with the padding

        """
        duration = self.duration_in_seconds(channel_properties)
        wait_time_in_seconds = to_duration - duration
        if wait_time_in_seconds < 0:
            raise ValueError(f"Target duration {to_duration} is shorter than the current schedule duration {duration}")
        if wait_time_in_seconds > TOLERANCE:
            contents = {}
            max_sample_duration = 0
            for channel, seg in self._contents.items():
                channel_props = channel_properties[channel]
                wait_time_rounded = channel_props.round_duration_to_granularity(
                    round(wait_time_in_seconds, TOLERANCE_DECIMALS), round_up=False
                )
                # We allow less than min samples, because we can't simply make it more without causing sync problems.
                # Anyway, it will very likely be merged with the next Wait.
                wait_time_in_samples = channel_props.duration_to_int_samples(wait_time_rounded, check_min_samples=False)
                sample_duration = wait_time_in_samples + seg.duration
                contents[channel] = Segment(
                    [Wait(duration=wait_time_in_samples)] + seg._instructions, duration=sample_duration
                )
                max_sample_duration = max(max_sample_duration, sample_duration)
            self._contents = contents
            self._duration = max_sample_duration
        return self

    def pad_to_hard_box(self) -> None:
        """Pad channels in ``self`` to the maximum channel length found within with ``Wait`` instructions.

        The ``Wait``s are appended to the end of the segments. NOTE: this method cannot be used when there are variable
        sampling rates present in the schedule. In that case, use the method ``pad_to_hard_box_in_seconds``.
        """
        max_duration = self.duration
        for _, segment in self.items():
            segment_duration = segment.duration
            if segment_duration < max_duration:
                segment.append(Wait(max_duration - segment_duration))

    def pad_to_hard_box_in_seconds(self, channel_properties: dict[str, ChannelProperties]) -> None:
        """Pad channels in ``self`` to the maximum channel length (seconds) found within with Wait instructions.

        The Waits are appended to the end of the segments. The segment durations are compared in seconds, so this
        method works in the case of variable sampling rates as well. The padding is added to a channel only if the
        difference between the channel's duration and the maximum duration is larger than the smallest possible
        instruction duration for that channel.

        Args:
            channel_properties: channel properties (containing the sampling rates and granularities).

        """
        max_duration = self.duration_in_seconds(channel_properties)
        for channel, segment in self.items():
            channel_props = channel_properties[channel]
            segment_duration = channel_props.duration_to_seconds(segment.duration)
            if segment_duration + TOLERANCE < max_duration:
                # round to the nearest, not up
                wait_duration_in_seconds = channel_props.round_duration_to_granularity(max_duration - segment_duration)
                wait_duration_in_samples = channel_props.duration_to_int_samples(
                    wait_duration_in_seconds, check_min_samples=False
                )
                if wait_duration_in_samples >= channel_props.instruction_duration_min:
                    segment.append(Wait(wait_duration_in_samples))

    def reverse(self) -> Schedule:
        """Copy of the schedule with the order of the instructions in each channel reversed.

        NOTE: this method cannot be used when there are variable sampling rates present in the schedule.

        To preserve synchronization of the channels, the channels are first rear-padded
        with :class:`.Nothing` instructions.
        """
        reverse_contents = {}
        T = self.duration
        for channel, seg in self._contents.items():
            delta = T - seg.duration
            reverse_seg = Segment([], duration=T)
            if delta > 0:
                # TODO pylint produces a false positive below
                reverse_seg._instructions.append(Nothing(delta))
            reverse_seg._instructions.extend(reversed(seg))
            reverse_contents[channel] = reverse_seg
        new = Schedule()
        new._contents = reverse_contents
        return new

    def reverse_hard_box(self) -> Schedule:
        """Copy of the schedule with the order of the instructions in each channel reversed.

        No additional time-synchronisation logic is implemented, so this method will break the synchronisation
        if ``self`` is not a schedule with matching durations in all segments.
        """
        reverse_copy = self.copy()
        for channel, seg in self._contents.items():
            reverse_seg = Segment([], duration=seg.duration)
            reverse_seg._instructions.extend(reversed(seg))
            reverse_copy[channel] = reverse_seg
        return reverse_copy

    def cleanup(self) -> Schedule:
        """Cleans up the schedule by removing things that do not affect the execution.

        Removes

        * empty channels, and
        * channels that only have idling instructions.

        Modifies ``self``.
        """
        idling = (Wait, Block)
        unnecessary = [
            channel_name
            for channel_name, segment in self._contents.items()
            if all(isinstance(instruction, idling) for instruction in segment)
        ]
        for channel_name in unnecessary:
            del self._contents[channel_name]
        if unnecessary:
            self._duration = None  # duration is no longer known and must be computed again
        return self

    def validate(self, path: tuple[str, ...] = ()) -> None:
        """Validate the contents of the schedule."""
        for channel_name, segment in self._contents.items():
            for inst in segment:
                try:
                    inst.validate()
                except ValueError as ex:
                    print(str(path + (channel_name,)), str(ex))

    def has_content_in(self, channel_names: Iterable[str]) -> bool:
        """Returns ``True`` if ``self`` has content in any of the given channels, otherwise ``False``."""
        for key, segment in self.items():
            if key in channel_names and segment.duration > 0:
                return True
        return False
