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
"""Tools for composing instruction schedules in time.

Under the idealized, noiseless, crosstalk-less computational model, the scheduling algorithms
should preserve the effect of the schedules on the computational subspace, i.e.
executing :class:`.Schedule` ``A`` immediately followed by ``B`` should be equivalent to
executing the composed schedule ``A+B``.

The composing is always done so that in ``A+B`` all the channels of ``B`` start
their execution simultaneously, and remain in sync. :class:`.Nothing` instructions can be added
as spacers between the channels of ``A`` and ``B`` as necessary to make this happen.

Typically the scheduling algorithms also try to minimize the total duration of the composed schedule.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from iqm.pulse.playlist.channel import ChannelProperties
from iqm.pulse.playlist.instructions import Block, Instruction, Wait
from iqm.pulse.playlist.schedule import TOLERANCE, TOLERANCE_DECIMALS, Nothing, Schedule

# TODO: remove the above skips once COMP-1281 is done and the TETRIS mode is fixed.


NONSOLID = (Block, Nothing)
"""Instructions that can be converted to :class:`.Wait` after scheduling."""

BLOCKING = (Block,)  # TODO VirtualRZ, but we would need to keep track of the phase angles...
"""Nonsolid Instructions that nevertheless block."""


@dataclass
class SegmentPointer:
    """Tool for working with Segments.

    Serves as a time pointer inside the Segment.
    """

    source: list[Instruction]
    """contents of the Segment"""
    idx: int
    """index of the current instruction"""
    TOL: float
    """time durations equal or smaller than this tolerance are considered zero (in seconds)"""
    frac: float = 0.0  # assumed in ~[0, T)
    """time, in seconds, after the start of the current instruction"""

    def get(self) -> Instruction:
        """Current instruction."""
        return self.source[self.idx]

    @property
    def remainder(self) -> float:
        """Remaining duration of the current instruction (in seconds)."""
        return self.get().duration - self.frac

    def next(self) -> bool:
        """Move to the beginning of the next instruction."""
        self.idx += 1
        self.frac = 0.0
        return self.idx >= len(self.source)  # did we run out?

    def cut_tail(self) -> None:
        """Cut the source of the pointer at the current index.

        Truncate :attr:`source` at :attr:`idx`, make ``self`` point to
        the cut tail part only. :attr:`frac` is not changed.

        Note: modifies :attr:`source`.
        """
        tail = self.source[self.idx :]  # copy of the tail
        del self.source[self.idx :]  # remove tail from source, should be fast since it's the tail of a list?
        self.idx = 0
        self.source = tail

    def tail(self) -> Sequence[Instruction]:
        """Instructions from the current index onwards."""
        return self.source[self.idx :]

    def rewind(self, duration: float) -> None:
        """Move the pointer back by ``duration`` seconds."""
        while duration > self.TOL:
            delta = min(duration, self.frac)
            duration -= delta
            self.frac -= delta
            if self.frac < self.TOL:  # move generously, or we'll never get anywhere
                self.idx -= 1
                T = self.get().duration
                self.frac += T
        # finally, normalize the position
        if self.frac + self.TOL > self.get().duration:
            self.idx += 1
            self.frac = 0.0
        if self.idx < 0:
            raise IndexError("rewinded too far")

    def fastforward(self, duration: float) -> bool:
        """Move the pointer forward by ``duration`` seconds."""
        T = self.get().duration
        if duration <= self.TOL:
            if T < self.TOL:  # the current instruction is 0 duration, so we forward past that.
                self.idx += 1
            if self.idx >= len(self.source):  # ran out
                return True
        while duration > self.TOL:
            inst = self.get()
            if not isinstance(inst, NONSOLID):
                raise ValueError(f"trying to fastforward over solid things! {inst}")
            delta = min(duration, T - self.frac)
            duration -= delta
            self.frac += delta
            if self.frac + self.TOL > T:  # move generously
                self.idx += 1
                if self.idx >= len(self.source):
                    return True  # ran out
                self.frac -= T
                T = self.get().duration
        return False


def extend_schedule(
    A: Schedule,
    B: Schedule,
    channels: dict[str, ChannelProperties],
    *,
    TOL: float = TOLERANCE,
) -> None:
    """Extend a Schedule with another Schedule.

    Extends ``A`` with ``B``, modifying both. The extension can add new channels to ``A``.

    If ``B`` has a ragged left side, i.e. some of its channels begin with :class:`.Nothing` instructions,
    this algorithm will not always produce an optimally short schedule.

    Args:
        A: schedule to be extended
        B: schedule to extend ``A`` with
        channels: properties of the control channels
        TOL: time durations equal or smaller than this are considered zero (in seconds)

    """
    # TODO consequent VirtualRZ instructions could be merged
    # extend_schedule(A, A) causes issues if we do not make a copy
    B = B.copy()

    # add missing channels
    A.add_channels(tuple(B))

    start_times = {}  # earliest possible start times for each channel of B
    extra_block = {}  # blocking time after B_start_time for each channel
    for ch, segment_B in B.items():
        segment_A = A[ch]
        T = segment_A.duration
        if segment_A and isinstance(segment_A[-1], Block):
            # remove the block, record its duration
            block_A = segment_A.pop().duration
        else:
            block_A = 0

        if segment_B and isinstance(segment_B[0], Block):
            # remove the block, record its duration
            block_B = segment_B.pop(0).duration
            extra_block[ch] = block_B
            if not segment_B:
                # there is nothing else in segment_B
                block_B = float("inf")
        else:
            block_B = 0

        # Blocks can overlap (we assume there are no consequent Blocks)
        start_times[ch] = T - min(block_A, block_B)

    B_start_time = max(start_times.values(), default=0)

    # add the instructions from B to A, with an appropriate Block between them on each channel
    for ch, segment_B in B.items():
        segment_A = A[ch]
        duration = B_start_time - segment_A.duration + extra_block.get(ch, 0.0)
        if duration > TOL:
            # make sure the channel can handle the wait
            _ = channels[ch].duration_to_int_samples(duration, message=f"{ch}: Block duration")
            # TODO if a wait is shorter than the shortest the instruments can handle,
            # maybe make all the segments wait longer?
            segment_A.append(Block(duration))

        # add the instructions in B
        segment_A.extend(segment_B)


def extend_schedule_new(  # noqa: PLR0915
    A: Schedule,
    B: Schedule,
    channels: dict[str, ChannelProperties],
    *,
    TOL: float = TOLERANCE,
) -> None:
    """Extend a Schedule with another Schedule.

    Extends ``A`` with ``B``, modifying ``A``. The extension can add new channels to ``A``.

    Can also handle cases where ``B`` has a ragged left side, i.e. some of its channels begin
    with :class:`.Nothing` instructions.

    Args:
        A: schedule to be extended
        B: schedule to extend ``A`` with
        channels: properties of the control channels
        TOL: time durations equal or smaller than this are considered zero (in seconds)

    """
    if not B:
        return  # shortcut for empty schedules
    # add missing channels
    A.add_channels(tuple(B))

    def find_nonsolid_depth(instructions: Iterable[Instruction]) -> tuple[float, float]:
        """Returns the free and blocking nonsolid depths of the given Instruction sequence."""
        free = 0.0
        blocking = 0.0
        block_seen = False
        for inst in instructions:
            if block_seen:
                if isinstance(inst, NONSOLID):
                    # even Nothing becomes blocking depth after the first blocking instruction is encountered
                    blocking += inst.duration
                else:
                    break
            elif isinstance(inst, Nothing):
                free += inst.duration
            elif isinstance(inst, BLOCKING):
                block_seen = True
                blocking += inst.duration
            else:
                break  # all other Instructions are solid
        else:
            # ran out of instructions, infinite depth
            if block_seen:
                blocking = float("inf")
            else:
                free = float("inf")
        return free, blocking

    def merge_overlap(iA: Instruction | None, iB: Instruction | None, duration: float) -> Instruction:
        """Instruction resulting from the overlap of two nonsolid Instructions.

        Nonsolid instructions are all Wait-like, and can thus be chopped up into multiple
        pieces without it changing anything. This function must never be called on solid instructions.
        """
        # make sure the channel can handle the duration
        if duration > TOL:
            _ = channels[ch].duration_to_int_samples(duration, message=f"{ch}: {iA} + {iB}: overlap duration")

        if isinstance(iA, Nothing) and isinstance(iB, Nothing):
            return Nothing(duration=duration)  # type: ignore[arg-type]
        return Block(duration=duration)  # type: ignore[arg-type]

    def find_start_time() -> float:
        """Find the earliest possible start time for schedule B."""
        start_times = {}
        for ch, segment_B in B.items():
            segment_A = A[ch]  # corresponding segment in A
            free_A, blocking_A = find_nonsolid_depth(reversed(segment_A))
            free_B, blocking_B = find_nonsolid_depth(segment_B)
            start_times[ch] = segment_A.duration - free_A - free_B - min(blocking_A, blocking_B)
        # time, relative to the start of A, where B will start
        return max(0.0, *start_times.values())

    def is_solid(inst: Instruction) -> bool:
        return not isinstance(inst, NONSOLID)

    B_start = find_start_time()
    # add the instructions from B to A
    for ch, segment_B in B.items():
        segment_A = A[ch]
        pointer_B = SegmentPointer(segment_B._instructions, 0, TOL)
        # distance between end of A and start of B
        distance = B_start - segment_A.duration
        # This is how long the merged segment_A will be.
        # We bypass the duration count system in segment_A for efficiency.
        segment_A._duration = max(B_start + segment_B.duration, segment_A.duration)

        if distance >= -TOL:
            if distance > TOL:
                # gap, add a Nothing between A and B
                segment_A._instructions.append(Nothing(duration=distance))  # type: ignore[arg-type]
            # add B
            segment_A._instructions.extend(pointer_B.tail())
        else:
            # there is overlap between A and B
            pointer_A = SegmentPointer(segment_A._instructions, len(segment_A), TOL)
            # find the instruction in A on which the overlap with B starts
            pointer_A.rewind(-distance)
            # pointer_A only needs to know about the overlapping tail part, which should be short in comparison to
            # the whole segment_A. segment_A is truncated to the non-overlapping part.
            pointer_A.cut_tail()

            # add partially non-overlapping item of A
            A_ran_out = False
            B_ran_out = False
            if pointer_A.frac > TOL:
                iA = pointer_A.get()
                if is_solid(iA):
                    segment_A._instructions.append(iA)
                    B_ran_out = pointer_B.fastforward(pointer_A.remainder)
                    A_ran_out = pointer_A.next()
                else:
                    segment_A._instructions.append(merge_overlap(iA, None, pointer_A.frac))

            # add overlapping items from A and B until either is exhausted
            while not A_ran_out and not B_ran_out:
                iA = pointer_A.get()
                iB = pointer_B.get()
                if is_solid(iA):
                    # solid instructions are never split, frac is always 0, and can never overlap
                    segment_A._instructions.append(iA)
                    B_ran_out = pointer_B.fastforward(pointer_A.remainder)
                    A_ran_out = pointer_A.next()
                elif is_solid(iB):
                    segment_A._instructions.append(iB)
                    A_ran_out = pointer_A.fastforward(pointer_B.remainder)
                    B_ran_out = pointer_B.next()
                else:
                    step = min(pointer_A.remainder, pointer_B.remainder)
                    segment_A._instructions.append(merge_overlap(iA, iB, step))
                    A_ran_out = pointer_A.fastforward(step)
                    B_ran_out = pointer_B.fastforward(step)

            # A, B or both ran out
            if not B_ran_out:
                # some B remains, add partially non-overlapping item of B, if any
                if pointer_B.frac > TOL:
                    iB = pointer_B.get()
                    # iB must be nonsolid, since solid instructions are not split
                    segment_A._instructions.append(merge_overlap(None, iB, pointer_B.remainder))
                    pointer_B.next()
                # add rest of B, if any
                segment_A._instructions.extend(pointer_B.tail())
            elif not A_ran_out:
                # some A remains, add partially non-overlapping item of A, if any
                if pointer_A.frac > TOL:
                    iA = pointer_A.get()
                    # iA must be nonsolid, since solid instructions are not split
                    segment_A._instructions.append(merge_overlap(iA, None, pointer_A.remainder))
                    pointer_A.next()
                # add rest of A, if any
                segment_A._instructions.extend(pointer_A.tail())


def extend_hard_boundary(
    schedule: Schedule,
    child_schedule: Schedule,
    child_components: set[str],
    neighborhood_components: set[str],
    component_durations: dict[str, int],
    is_alap: bool,
) -> None:
    """Merge two Schedules together such that the timebox boundary is respected.

    This scheduling algorithm treats the Schedules as hard, rectangular boxes where any ragged edges
    will be padded with Waits, and the boxes are not allowed to overlap.

    The algorithm is as follows:

    1. When adding ``child_schedule`` to ``schedule``, the longest channel in ``schedule`` that overlaps with the
    channels present in child determines the earliest possible starting time for the ``child_schedule``, and all other
    channels in ``schedule`` are padded with ``Wait`` to the aforementioned max length.

    2. An occupied channel in ``schedule`` will always occupy all channels of the corresponding component (qubit,
    coupler, ...). This is handled by keeping track of occupied durations for each component (no unnecessary padding
    is added to channels which do not have an actual physical pulse).

    3. After the schedules are combined, all the common channels of ``schedule`` and ``child_schedule`` are blocked
    up to their common maximum length.

    This algorithm should not be used with variable sampling rates in the schedule channels. In that case, use
    :func:`extend_hard_boundary_in_seconds` instead.

    Args:
        schedule: Schedule that should be extended with ``child_schedule``. Modified in place.
        child_schedule: Child schedule to be added.
        child_components: Components (qubits, couplers, computational_resonators) that have at least
            one channel in ``child_schedule``.
        neighborhood_components: QPU components neighboring the ``child_components`` that should
            additionally be blocked in the scheduling.
        component_durations: Blocked durations for each component used by ``schedule``.
            These act as the earliest starting points for a new segment added to any of the channels
            of the component, but will also block the component even if it has no occupied channels
            in the schedule yet or ever (e.g. a computational resonator).
            Modified in place.
        is_alap: Whether the scheduling strategy is ALAP (As Late As Possible).

    """
    # add the child's channels to the schedule if they aren't there yet
    child_channels = child_schedule.channels()
    schedule.add_channels(child_channels)

    # the child schedule can start once all the components it uses are free
    child_start: int = max((T for c, T in component_durations.items() if c in child_components), default=0)
    child_duration: int = 0
    for channel in child_channels:
        # pad the corresponding parent segment with Wait
        segment = schedule[channel]
        if segment.duration < child_start:
            segment.append(Wait(child_start - segment.duration))
        # add the child segment's instructions to parent segment
        child_segment = child_schedule[channel]
        child_instructions = reversed(child_segment) if is_alap else child_segment
        segment._instructions.extend(child_instructions)
        segment._duration += child_segment.duration
        # compute the child schedule duration
        child_duration = max(child_segment.duration, child_duration)

    # update the blocked durations
    for component in neighborhood_components:
        component_durations[component] = child_start + child_duration
    # the schedule's total duration is not known anymore after adding child
    schedule._duration = None


def extend_hard_boundary_in_seconds(
    schedule: Schedule,
    child_schedule: Schedule,
    child_components: set[str],
    neighborhood_components: set[str],
    component_durations: dict[str, float],
    is_alap: bool,
    channel_properties: dict[str, ChannelProperties],
) -> None:
    """The same as ``extend_hard_boundary``, but the scheduling is done in seconds.

    Used when the probe channel sampling rate differs from the other channels' rate.
    The incoming schedules measure Instruction durations in samples, but ``component_durations``
    is in seconds.

    Args:
        schedule: Schedule that should be extended with ``child_schedule``. Modified in place.
        child_schedule: Child schedule to be added.
        child_components: Components (qubits, couplers, computational_resonators) that have at least
            one channel in ``child_schedule``.
        neighborhood_components: Components neighboring the ``child_components`` that should
            additionally be blocked in the scheduling.
        component_durations: Blocked durations for each component in ``schedule``.
            These act as the earliest starting points for new segment added to any of the channels
            of a given component, but will also block the component even if it has no occupied channels
            in the schedule yet or ever (e.g. a computational resonator).
            The durations are in seconds. Modified in place.
        is_alap: Whether the scheduling strategy is ALAP (As Late As Possible).
        channel_properties: Mapping from channel name to its properties (e.g. the sample rates
            and granularities).

    """
    # add the child's channels to the schedule if they aren't there yet
    child_channels = child_schedule.channels()
    schedule.add_channels(child_channels)

    # the child schedule can start once all the components it uses are free
    child_start = max((T for c, T in component_durations.items() if c in child_components), default=0.0)
    child_duration: float = 0.0
    for channel in child_channels:
        # pad the corresponding parent segment with Wait
        segment = schedule[channel]
        channel_props = channel_properties[channel]
        segment_duration_in_seconds = channel_props.duration_to_seconds(segment.duration)
        extra_offset = 0.0
        # TOLERANCE accounts for floating point errors etc.
        if segment_duration_in_seconds + TOLERANCE < child_start:
            # wait duration that needs to be added such that the TOLERANCE is accounted for
            wait_duration_in_seconds = round(child_start - segment_duration_in_seconds, TOLERANCE_DECIMALS)
            # round the wait duration up to the closest value that can be handled by the channel
            wait_duration_rounded = channel_props.round_duration_to_granularity(
                wait_duration_in_seconds, round_up=True, force_min_duration=True
            )
            wait_duration_in_samples = channel_props.duration_to_int_samples(wait_duration_rounded)
            segment.append(Wait(wait_duration_in_samples))
            # extra time offset that was added due to the variable sampling rate
            extra_offset = wait_duration_rounded - wait_duration_in_seconds

        # add the child segment's instructions to parent segment
        child_segment = child_schedule[channel]
        child_instructions = reversed(child_segment) if is_alap else child_segment
        segment._instructions.extend(child_instructions)
        segment._duration += child_segment.duration

        # the increase in duration added to channel includes the extra_offset caused by variable sampling rates
        child_segment_duration_in_seconds = channel_props.duration_to_seconds(child_segment.duration) + extra_offset
        child_duration = max(child_segment_duration_in_seconds, child_duration)

    for component in neighborhood_components:
        component_durations[component] = child_start + child_duration
    schedule._duration = None  # the schedule's total duration is not known anymore after adding child
