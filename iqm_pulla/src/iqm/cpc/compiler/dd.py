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
"""Dynamical decoupling utilities."""

from collections.abc import Iterable, Set
import logging
import math

from iqm.cpc.interface.compiler import DDStrategy, PRXSequence
from iqm.pulla.utils import InstructionLocation, locate_instructions, replace_instruction_in_place
from iqm.pulse.builder import ScheduleBuilder
from iqm.pulse.gate_implementation import GateImplementation
from iqm.pulse.playlist.instructions import Instruction, Wait
from iqm.pulse.playlist.schedule import Schedule, Segment
from iqm.pulse.scheduler import Block
from iqm.pulse.timebox import TimeBox

cpc_logger = logging.getLogger("cpc")

_PRX_INSTANCES = {"X": (math.pi, 0), "Y": (math.pi, math.pi / 2)}
"""A mapping of shorthand PRX instance names to their arguments for defining gate sequences in
dynamical decoupling strategies. For example, if user submits gate pattern consisting of "X" and "Y"
gates, e.g. "YXYX", it will get translated internally into a sequence of PRX args using this mapping."""


STANDARD_DD_STRATEGY = DDStrategy(gate_sequences=[(9, "XYXYYXYX", "asap"), (5, "YXYX", "asap"), (2, "XX", "center")])
"""The default DD strategy uses the following gate sequences:

* Simple symmetric CPMG sequence for short idling times.
* Asymmetric (left-aligned) universal XY4 sequence for medium idling times.
* Asymmetric (left-aligned) universal EDD sequence for longer idling times.
"""


def _gate_pattern_to_prx_sequence(pattern: str) -> PRXSequence:
    """Translate a gate pattern string (with gates noted as e.g. "X" and "Y") to a sequence of PRX args."""
    submitted_gates = set(pattern)
    supported_gates = set(_PRX_INSTANCES.keys())
    unsupported_gates = submitted_gates - supported_gates

    if unsupported_gates:
        raise ValueError(
            f"There are unsupported gate(s) {unsupported_gates} present in the submitted sequence {submitted_gates}"
        )
    return [_PRX_INSTANCES[gate] for gate in pattern]


def _merge_wait_instructions_in_segment(segment: Segment) -> Segment:
    """Merge contiguous ``Wait`` instructions within a segment.

    Build a new segment with merged instructions.

    Args:
        segment: Sequence of instructions for a single channel.

    Returns:
        Copy of ``segment`` with contiguous ``Wait`` instructions merged.

    """
    new_instructions: list[Instruction] = []
    accumulating_waits = False
    wait_duration: int = 0
    for instr in segment:
        if isinstance(instr, Wait):
            accumulating_waits = True
            wait_duration += instr.duration
        elif isinstance(instr, Block) and instr.duration == 0 and accumulating_waits:
            # Zero duration Blocks that immediately follow Waits are also eliminated, so that
            # barriers have no effect on DD.
            pass
        else:
            if accumulating_waits:
                new_instructions.append(Wait(duration=wait_duration))
            new_instructions.append(instr)
            accumulating_waits = False
            wait_duration = 0

    # Take care of a potential Wait instructions at the end of the segment
    if wait_duration > 0:
        new_instructions.append(Wait(duration=wait_duration))

    # segment duration should be the same, this way we don't have to recalculate it
    new_segment = Segment(new_instructions, duration=segment.duration)
    return new_segment


def merge_wait_instructions_in_schedule(builder: ScheduleBuilder, schedule: Schedule) -> Schedule:
    """Merge adjacent ``Wait`` instructions in the drive channels of the given schedule.

    Makes a deep copy ``schedule`` and iterates through its drive channels to merge adjacent
    ``Wait`` instructions into a single ``Wait`` instruction.

    Also merges ``Wait`` instructions if they are separated by ``Block`` instructions.

    Args:
        builder: Provides channel information.
        schedule: Schedule to process.

    Returns:
        Deep copy of ``schedule`` with ``Wait`` instructions merged.

    """
    copied_schedule = schedule.copy()

    for channel_name, segment in copied_schedule.items():
        if _is_drive_channel(builder, channel_name):
            new_segment = _merge_wait_instructions_in_segment(segment)
            copied_schedule[channel_name] = new_segment

    return copied_schedule


def _channel_to_component(builder: ScheduleBuilder, channel_name: str) -> str:
    """QPU component with which the given control channel is associated.

    Args:
        builder: Provides component to channel mapping.
        channel_name: Name of the control channel to look up.

    Returns:
        Name of the QPU component that owns ``channel_name``.

    Raises:
        ValueError: ``channel_name`` is not associated with any QPU component.

    """
    for component, channels in builder.component_channels.items():
        if channel_name in channels.values():
            return component

    raise ValueError(f"Channel {channel_name} is not associated with any QPU component.")


def _is_drive_channel(builder: ScheduleBuilder, channel_name: str) -> bool:
    """True iff the given channel is a non-virtual drive channel.

    Args:
        builder: Provides component to channel mapping.
        channel_name: Name of the control channel to look up.

    Returns:
        True iff ``channel_name`` refers to a non-virtual drive channel.

    """
    channel = builder.channels.get(channel_name)
    return channel is not None and channel.is_iq and not channel.is_virtual


def _find_wait_instructions_on_channels(
    schedule: Schedule,
    channels: Iterable[str],
    min_duration: int,
    skip_leading_wait: bool = True,
    skip_trailing_wait: bool = True,
) -> list[InstructionLocation]:
    """Information on ``Wait`` instructions in the given schedule that meet the given criteria.

    The Wait must

    * be on one of ``channels``,
    * have a duration >= the given minimum,
    * not be the first or last Wait on the channel, if the corresponding flag is set.

    Args:
        schedule: Schedule to search.
        channels: Names of channels in ``schedule`` to search.
        min_duration: Minimum duration for the Wait instructions (in samples).
        skip_leading_wait: Iff True, the first ``Wait`` instruction on each channel should be skipped.
        skip_trailing_wait: Iff True, the last ``Wait`` instruction on each channel should be skipped.

    Returns:
        Information on ``Wait`` instructions in ``schedule`` that match the given criteria.

    """
    all_waits = locate_instructions(schedule, Wait, min_duration=min_duration, channels=channels)
    filtered_waits = []

    for wait in all_waits:
        # Check if leading and/or trailing instruction should be excluded
        if skip_leading_wait and wait.index == 0:
            continue
        last_index = len(schedule[wait.channel_name]) - 1
        if skip_trailing_wait and wait.index == last_index:
            continue
        filtered_waits.append(wait)

    return filtered_waits


def _near_split(number: int, parts: int, step: int = 1) -> list[int]:
    """Partition an integer into the given number of nonnegative parts.

    The resulting parts must be as equal as possible, but nonzero parts also must be >= ``step``.

    Args:
        number: A number to split.
        parts: An amount of parts to split a given number into.
        step: Minimum size for nonzero parts.

    Returns:
        List of parts that sums up to ``number``.

    Example:
        >>> near_split(10, 3)
        [4, 3, 3]

        >>> near_split(10, 4)
        [3, 3, 2, 2]

        >>> near_split(10, 4, 4)
        [5, 5, 0, 0]

        >>> near_split(14, 10, 4)
        [5, 5, 4, 0, 0, 0, 0, 0, 0, 0]

    """
    if parts == 0:
        return []

    if number == 0:
        return [0] * parts

    if number < step:
        raise ValueError(f"It is not possible to split {number}, because it is smaller than the split step {step}.")

    quotient, remainder = divmod(number, parts)
    if quotient >= step:
        return [quotient + 1] * remainder + [quotient] * (parts - remainder)

    num_non_zero, remainder = divmod(number, step)
    splitted_reminder = _near_split(remainder, num_non_zero) + [0] * (parts - num_non_zero)
    non_zero_contribution = [step] * num_non_zero + [0] * (parts - num_non_zero)
    return [s + n for s, n in zip(splitted_reminder, non_zero_contribution)]


def _create_waiting_slots(
    duration_slots: int, num_pulses: int, min_duration_slots: int, alignment: str
) -> tuple[list[int], list[int]]:
    """Given a time duration, partition it into wait times to be inserted between DD pulses.

    The DD sequence is constructed by interleaving the waits like this:

    ``Wait(even[0]), pulse[0], Wait(odd[0]), pulse[1], Wait(even[1]), ...,
    Wait(odd[N-1]), pulse[2*N - 1], Wait(even[N])``,
    where ``even`` and ``odd`` are the two lists of waiting times returned.

    Splitting the time duration into two lists helps to achieve near-uniform spacing of the pulses.

    Args:
        duration_slots: Time duration (in slots).
        num_pulses: Number of pulses in the DD gate sequence. Even.
        min_duration_slots: Minimal time duration for instructions (in slots).
        alignment: A string literal with a value of either "asap", "alap", or "center". It aligns the dynamical
            decoupling sequence left, or right, or keeps it centered. The default value "center" is associated with a
            symmetric dynamical decoupling sequence with respect to the center of the original waiting time.

    Returns:
        ``duration_slots`` partitioned into ``even`` and ``odd`` lists of wait times (in slots)

    """
    SEQUENCE_ALIGNMENT = {
        "asap": -1,
        "center": 0,
        "alap": 1,
    }
    offset = SEQUENCE_ALIGNMENT.get(alignment.lower())
    if offset is None:
        raise ValueError(f"Unsupported single-qubit gate sequence alignment requested: {alignment}")

    # Divide the residual time in two
    even_duration_slots = round(duration_slots / 2)
    odd_duration_slots = duration_slots - even_duration_slots

    # Simple division in the number of pulses pairs
    odd = _near_split(odd_duration_slots, num_pulses // 2, min_duration_slots)

    # Slightly more complicated approach, to be able to shift the sequence (controlled by offset)
    even = _near_split(even_duration_slots, num_pulses // 2, min_duration_slots)
    first_even_bin = round(even[0] / 2 * (1 + offset))
    last_even_bin = even[0] - first_even_bin
    even[0] = first_even_bin
    even.append(last_even_bin)

    return even, odd


def _create_dd_sequence(
    dd_segments: list[Segment],
    wait_duration: int,
    granularity: int,
    min_duration: int,
    alignment: str = "center",
) -> list[Instruction]:
    """Dynamical decoupling sequence to replace a single ``Wait`` instruction in a drive channel.

    Args:
        dd_segments: Drive channel segments that implement the DD sequence gates. Has to have an even number
            of segments/gates.
        wait_duration: Duration of the ``Wait`` instruction (in samples) to be replaced with the DD sequence.
        granularity: Granularity of the drive channel (in samples), i.e. the duration of all instructions must be
            multiples of this number.
        min_duration: Minimal duration for instructions (in samples).
        alignment: A string literal with a value of either "asap", "alap", or "center". It shifts the dynamical
            decoupling sequence left, or right, or keeps it centered. The default value "center" produces a
            symmetric dynamical decoupling sequence with respect to the center of the original ``Wait`` instruction.

    Returns:
        Dynamical decoupling sequence.

    """
    if len(dd_segments) % 2 != 0:
        raise ValueError("The number of DD gates in the sequence has to be even.")

    seg_durations = [seg.duration for seg in dd_segments]

    if len(set(seg_durations)) != 1:
        raise ValueError(
            f"The durations of all the gates in the provided DD gate sequence must be equal (now {seg_durations})."
        )

    total_gate_duration = sum(seg_durations)
    residual_waiting_time: int = wait_duration - total_gate_duration
    if residual_waiting_time < 0:
        raise ValueError(
            f"The waiting time ({wait_duration} samples) cannot accomodate the {len(dd_segments)} DD gates required "
            f"(duration {total_gate_duration})."
        )

    residual_waiting_slots, remainder = divmod(residual_waiting_time, granularity)
    if remainder != 0:
        raise ValueError("The (remaining) waiting time is not a multiple of the granularity!")

    even_waits, odd_waits = _create_waiting_slots(
        duration_slots=residual_waiting_slots,
        num_pulses=len(dd_segments),
        min_duration_slots=min_duration // granularity,  # NOTE: assumes min_duration is divisible by granularity!
        alignment=alignment,
    )

    # Creation of corresponding instructions
    instructions: list[Instruction] = [Wait(duration=granularity * even_waits[0])]
    for seg_even, wait_odd, seg_odd, wait_even in zip(dd_segments[::2], odd_waits, dd_segments[1::2], even_waits[1:]):
        instructions += (
            list(seg_even)
            + [Wait(duration=granularity * wait_odd)]
            + list(seg_odd)
            + [Wait(duration=granularity * wait_even)]
        )

    cpc_logger.debug("Original waiting time: %d", wait_duration)
    cpc_logger.debug("Dynamical decoupling sequence with %d pairs", len(seg_durations) / 2)
    cpc_logger.debug(
        "Durations (each slot is %d samples): %s",
        granularity,
        str([inst.duration // granularity for inst in instructions]),
    )
    return instructions


def _find_dd_channels(
    builder: ScheduleBuilder,
    schedule: Schedule,
    target_components: Set[str] | None,
) -> dict[str, GateImplementation]:
    """Find channels in the given schedule on which DD should be applied.

    Finds all non-virtual drive channels in ``schedule`` whose QPU component has a PRX implementation
    available in the current calset.

    If ``target_components`` is given, further restrict the returned channels to the channels
    of the given components, and raise an error if a component has no DD channel available.

    Args:
        builder: Schedule builder used to build ``schedule``, containing channel information.
        schedule: Schedule to analyze.
        target_components: QPU components on whose drive channels to apply DD.
            If None, apply DD on all QPU components.

    Returns:
        DD channels mapped to the PRX implementation that is used to implement DD on that channel.

    Raises:
        ValueError: Something impossible was requested.

    """
    # TODO there should be a better way to get the channel info and component
    # Find the channels in the schedule on which DD should be applied
    dd_channels: dict[str, GateImplementation] = {}

    if target_components is None:
        for channel_name in schedule.channels():
            if not _is_drive_channel(builder, channel_name):
                continue

            # Get the highest-priority PRX gate implementation available for the channel component
            component_name = _channel_to_component(builder, channel_name)
            try:
                prx = builder.get_implementation("prx", (component_name,), use_priority_order=True)
            except ValueError:
                pass

            dd_channels[channel_name] = prx
    else:
        # We need more complicated error handling if target_components are given.
        all_components = builder.chip_topology.qubits | builder.chip_topology.computational_resonators
        diff = target_components - all_components
        if diff:
            raise ValueError(f"Unknown target components {list(diff)}.")

        errors = []
        schedule_channels = frozenset(schedule.channels())
        for component_name in target_components:
            try:
                channel_name = builder.get_drive_channel(component_name)
            except KeyError:
                errors.append(f"Target component {component_name} has no drive channel available.")
                continue

            if not _is_drive_channel(builder, channel_name):
                errors.append(
                    f"Drive channel {channel_name} of component {component_name} is a virtual channel"
                    " that does not support DD."
                )
                continue

            if channel_name not in schedule_channels:
                continue

            # Get the highest-priority PRX gate implementation available for the channel component
            try:
                prx = builder.get_implementation("prx", (component_name,), use_priority_order=True)
            except ValueError:
                errors.append(f"Drive channel {channel_name} of component {component_name} has no PRX available.")

            dd_channels[channel_name] = prx

        if errors:
            raise ValueError("\n".join(errors))

    return dd_channels


def insert_dd_sequences(
    builder: ScheduleBuilder,
    schedule: Schedule,
    strategy: DDStrategy,
) -> None:
    """Insert dynamical decoupling sequences into the given schedule.

    .. note:: Modifies ``schedule`` in-place.

    .. note::

       Assumes that the PRX implementation used only applies non-wait instructions on a single drive channel,
       and that the PRX duration does not depend on its arguments.

    Args:
        builder: Schedule builder used to build ``schedule``, containing channel information.
        schedule: Schedule to modify.
        strategy: Dynamical decoupling strategy to use.

    """

    def _get_channel_segment(builder: ScheduleBuilder, timebox: TimeBox, drive_channel: str) -> Segment:
        """Extract the ``drive_channel`` segment of the schedule generated from the given timebox."""
        schedule = builder.timebox_to_schedule(timebox, neighborhood=0)
        return schedule[drive_channel]

    # Sort sequences by ratio in descending order
    dd_sequences_sorted = sorted(strategy.gate_sequences, key=lambda seq: seq[0], reverse=True)

    # Check if "shorthand" notation with "X" and "Y" gate aliases is used to define SQG sequences and if so, translate
    # it to definitions using generic gates
    dd_sequences: list[tuple[int, PRXSequence, str]] = [
        (
            ratio,
            _gate_pattern_to_prx_sequence(pattern) if isinstance(pattern, str) else pattern,
            alignment,
        )
        for ratio, pattern, alignment in dd_sequences_sorted
    ]

    dd_channels = _find_dd_channels(builder, schedule, strategy.target_qubits)

    # Find Wait instructions on dd_channels with a sufficient duration etc.
    dd_windows = _find_wait_instructions_on_channels(
        schedule,
        channels=dd_channels.keys(),
        min_duration=0,
        skip_leading_wait=strategy.skip_leading_wait,
        skip_trailing_wait=strategy.skip_trailing_wait,
    )

    # Start from the end of the circuit, otherwise the indexes of the original waits change because of in-place
    # insertions
    for wait in reversed(dd_windows):
        channel_name = wait.channel_name
        prx = dd_channels[channel_name]
        wait_duration = wait.duration

        # NOTE: Duration is always calculated using "X"
        prx_duration = _get_channel_segment(builder, prx.rx(math.pi), channel_name).duration
        calculated_ratio = wait_duration / prx_duration

        cpc_logger.debug(
            "Processing Wait on channel %s, index %d with duration %d",
            channel_name,
            wait.index,
            wait_duration,
        )

        for ratio, prx_args, alignment in dd_sequences:
            # use the longest DD sequence that fits into the wait window
            if calculated_ratio >= ratio:
                # each PRX gate in the sequence must be implementable using a single drive channel segment
                dd_segments = []
                for theta, phi in prx_args:
                    seg = _get_channel_segment(builder, prx(theta, phi), channel_name)
                    dd_segments.append(seg)

                cpc_logger.debug(
                    "Applying dynamical decoupling sequence %s with alignment %d",
                    prx_args,
                    alignment,
                )

                dd_instructions = _create_dd_sequence(
                    dd_segments=dd_segments,
                    wait_duration=wait_duration,
                    granularity=builder.channels[channel_name].instruction_duration_granularity,
                    min_duration=builder.channels[channel_name].instruction_duration_min,
                    alignment=alignment,
                )

                replace_instruction_in_place(schedule, channel_name, wait.index, dd_instructions)
                break
