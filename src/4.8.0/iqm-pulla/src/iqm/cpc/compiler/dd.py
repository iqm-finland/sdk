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
"""Dynamical decoupling utilities.

We support three DD workflows, all accessible via the :func:`apply_dd_strategy` compilation pass
(and hence controllable via ``settings``):

* **Standard DD** — one :class:`.DDStrategy` is applied to all eligible idle
  windows on all qubits.
* **Qubit-specific DD** — *different* DD sequences for each qubit, defined using the
  ``qubit_specific_dd`` setting.
* **Idling-specific DD** — individual DD sequences per **idle window** described using
  :class:`IdleWindowKey`, defined using the
  ``idling_specific_dd`` setting. See the inspection helpers
  :func:`identify_idle_windows_for_settings` and :func:`display_idle_windows`.

Idling-specific DD can be applied in three different ways:

**Known map** — provide an :type:`IdleDDMap`:

.. code-block:: python

    settings = compiler.get_settings(circuits=circuits)
    settings.stages.dynamical_decoupling.apply_dd_strategy.dd_is_disabled = False
    settings.stages.dynamical_decoupling.apply_dd_strategy.idling_specific_dd = idle_dd_map
    compiler.compile(circuits=circuits, settings=settings)

**Inspect, then decide (for manual decisions)** — run the compilation stages up to the DD stage,
display the idle windows, set ``idling_specific_dd`` to the :type:`IdleDDMap` you want, and then run
the remaining stages:

.. code-block:: python

    settings = compiler.get_settings(circuits=circuits)
    settings.stages.dynamical_decoupling.apply_dd_strategy.dd_is_disabled = False
    stages = compiler.circuit_stages + compiler.pulse_stages
    dd_idx = [stage.name for stage in stages].index("dynamical_decoupling")
    context = compiler.compiler_context(components=None, settings=settings)

    schedules, context = compiler.run_stages(circuits, context, stages=stages[:dd_idx])
    display_idle_windows(identify_idle_windows_for_settings(schedules, context))

    settings.stages.dynamical_decoupling.apply_dd_strategy.idling_specific_dd = idle_dd_map
    schedules, context = compiler.run_stages(schedules, context, stages=stages[dd_idx:])
    job_def, context = compiler.finalize(schedules, context)

**Callable (for automated decisions)** — assign a callable ``idling_specific_dd`` of the type
:type:`IdleDDMapBuilder`. In the DD pass it is called on all the eligible idling windows in the pre-DD
schedules, and the :type:`IdleDDMap` it returns is applied. Everything happens inside a single
``compiler.compile()`` call.
"""

from __future__ import annotations

from collections.abc import Iterable, Set
from dataclasses import dataclass
import logging
import math
from typing import Any, Final, NamedTuple, Protocol, TypeAlias, cast

from exa.common.data import SettingNode
from iqm.cpc.compiler._utils.stages import setting_node_to_dict
from iqm.cpc.compiler.errors import ClientError
from iqm.cpc.compiler.utils import (
    InstructionLocation,
    locate_instructions,
    replace_instruction_in_place,
)
from iqm.pulse.builder import ScheduleBuilder
from iqm.pulse.gates.move import MoveMarker
from iqm.pulse.gates.prx import PrxGateImplementation
from iqm.pulse.playlist.instructions import Instruction, Wait
from iqm.pulse.playlist.schedule import Schedule, Segment
from iqm.pulse.scheduler import Block
from iqm.pulse.timebox import TimeBox
from iqm.station_control.interface.models.circuit import DDGateSequence, DDStrategy, PRXSequence

cpc_logger = logging.getLogger("cpc")

_PRX_INSTANCES = {"X": (math.pi, 0), "Y": (math.pi, math.pi / 2)}
"""A mapping of shorthand PRX instance names to their arguments for defining gate sequences in
dynamical decoupling strategies. For example, if user submits gate pattern consisting of "X" and "Y"
gates, e.g. "YXYX", it will get translated internally into a sequence of PRX args using this mapping."""


STANDARD_DD_STRATEGY: Final[DDStrategy] = DDStrategy(
    gate_sequences=[
        DDGateSequence(9, "XYXYYXYX", "asap"),
        DDGateSequence(5, "YXYX", "asap"),
        DDGateSequence(2, "XX", "center"),
    ]
)
"""The default DD strategy uses the following gate sequences:

* Simple symmetric CPMG sequence for short idling times.
* Asymmetric (left-aligned) universal XY4 sequence for medium idling times.
* Asymmetric (left-aligned) universal XYXYYXYX sequence for longer idling times.
"""

_STANDARD_DD_GATE_SEQUENCES_RATIO: list[int] = [seq.ratio for seq in STANDARD_DD_STRATEGY.gate_sequences]
"""Ratios of :attr:`STANDARD_DD_STRATEGY`, used to initialize the DD stage settings."""

_STANDARD_DD_GATE_SEQUENCES_GATE_PATTERN_XY: list[str] = [
    cast("str", seq.gate_pattern) for seq in STANDARD_DD_STRATEGY.gate_sequences
]
"""Gate patterns of :attr:`STANDARD_DD_STRATEGY`, used to initialize the DD stage settings."""

_STANDARD_DD_GATE_SEQUENCES_ALIGN: list[str] = [seq.alignment for seq in STANDARD_DD_STRATEGY.gate_sequences]
"""Alignments of :attr:`STANDARD_DD_STRATEGY`, used to initialize the DD stage settings."""


class DDError(ClientError):
    """Something was wrong in user input for dynamical decoupling."""


def apply_dd_strategy(  # noqa: PLR0913
    schedules: list[Schedule],
    builder: ScheduleBuilder,
    context: dict[str, Any],
    dd_is_disabled: bool = True,
    qubit_specific_dd: dict[str, list[DDGateSequence] | None] | None = None,
    idling_specific_dd: IdleDDMap | IdleDDMapBuilder | None = None,
    DDStrategy_merge_contiguous_waits: bool = True,
    DDStrategy_target_qubits: list[str] | None = None,
    DDStrategy_skip_leading_wait: bool = True,
    DDStrategy_skip_trailing_wait: bool = True,
    DDStrategy_gate_sequences_ratio: list[int] = _STANDARD_DD_GATE_SEQUENCES_RATIO,
    DDStrategy_gate_sequences_gate_pattern_xy: list[str] = _STANDARD_DD_GATE_SEQUENCES_GATE_PATTERN_XY,
    DDStrategy_gate_sequences_align: list[str] = _STANDARD_DD_GATE_SEQUENCES_ALIGN,
) -> list[Schedule]:
    """Insert dynamical decoupling sequences into the schedules iff ``dd_is_disabled`` is False.

    By default DD is controlled by a :class:`.DDStrategy`, which can be provided in ``context["DDStrategy"]``,
    or as individual fields in the settings tree (the ``DDStrategy_*`` parameters).
    Alternatively you can override this with qubit-specific DD strategies, or idling window -specific DD strategies.
    The precedence of the parameters to this function (highest first) is

    1. ``idling_specific_dd`` — per-idle-window DD. May be either

       * an :type:`IdleDDMap` mapping :class:`IdleWindowKey` keys to :class:`DDGateSequence` lists
         (the longest sequence that fits in the idle window is applied) or to ``None`` (no DD for this window).
       * a **callable** :class:`IdleDDMapBuilder`.
         Invoked inside the pass after waits are merged and idle windows
         enumerated, but before any DD pulse is applied.

       Idle windows not in the map use the default DD sequence defined in ``DDStrategy``, if any.

       To inspect the idle windows before choosing an
       ``idling_specific_dd`` map, see the workflow in :mod:`iqm.cpc.compiler.dd`.

    2. ``qubit_specific_dd`` — per-qubit DD.
       Maps qubit name to the DD sequence applied on that qubit, or ``None`` (no DD for this qubit).
       Qubits not in the map use the default DD strategy, if any.

    3. ``context["DDStrategy"]``, describing the global default DD strategy.

    4. :class:`.DDStrategy` built from the individual ``DDStrategy_*`` parameters. These are initialized
       to :attr:`STANDARD_DD_STRATEGY` in the settings tree, so the standard strategy is the default
       unless you change them.

    .. note::

       ``qubit_specific_dd`` and ``idling_specific_dd`` are compile-only settings: their values
       cannot be serialized (e.g. callables and dicts), so this pass removes them
       from ``context["settings"]`` when it finishes.

    Args:
        schedules: Schedules representing the compiled circuits.
        builder: The ScheduleBuilder.
        context: Compiler context.
        dd_is_disabled: Set to ``False`` to enable dynamical decoupling.
        qubit_specific_dd: Per-qubit DD map. Maps physical qubit name to the DD sequences applied to idle windows
            on that qubit, or ``None`` to mean no DD sequence is applied on that qubit. Qubits not in the map
            use the default DD strategy.
        idling_specific_dd: Per-idle-window DD. Either an :type:`IdleDDMap` (dict from
            :class:`IdleWindowKey` to a ``gate_sequences`` list or ``None``),
            or an :class:`IdleDDMapBuilder` callable
            ``fn(idle_windows, schedules, builder) -> IdleDDMap | None`` (useful for
            automated decisions — see the module docstring of :mod:`iqm.cpc.compiler.dd`).
        DDStrategy_merge_contiguous_waits: Whether to merge contiguous waits (see :class:`.DDStrategy`).
        DDStrategy_target_qubits: The qubits to which DD is applied (``None`` means apply to every applicable qubit).
        DDStrategy_skip_leading_wait: Whether to skip leading waits (see :class:`.DDStrategy`).
        DDStrategy_skip_trailing_wait: Whether to skip trailing waits (see :class:`.DDStrategy`).
        DDStrategy_gate_sequences_ratio: Minimal durations for the Wait to be replaced with the DD sequence
            (in PRX gate durations) in the DD sequence. Defaults to the ratios of ``STANDARD_DD_STRATEGY``.
        DDStrategy_gate_sequences_gate_pattern_xy: XY Gate patterns in the DD sequence. If you want to provide custom
            PRX angles instead of XY patterns, you must provide the DDStrategy in the Compiler context.
            Defaults to the gate patterns of ``STANDARD_DD_STRATEGY``.
        DDStrategy_gate_sequences_align: Alignments in the DD sequence ("asap", "alap" or "center").
            Defaults to the alignments of ``STANDARD_DD_STRATEGY``.

            These three parameters are parallel lists defining the default DD strategy, and must have
            the same length. Setting any of them to an empty list means there is no default DD strategy.

    Returns:
        The schedules where applicable Waits are replaced with DD sequences.

    """
    try:
        if dd_is_disabled:  # Skip the pass and move on
            return schedules

        # build the default strategy
        if "DDStrategy" in context:
            strategy: DDStrategy = context["DDStrategy"]
        else:
            if not (
                DDStrategy_gate_sequences_ratio
                and DDStrategy_gate_sequences_gate_pattern_xy
                and DDStrategy_gate_sequences_align
            ):
                # an empty gate sequence component means there is no default DD strategy
                gate_sequences: list[DDGateSequence] = []
            else:
                gate_sequences = [
                    DDGateSequence(r, g, a)
                    for r, g, a in zip(
                        DDStrategy_gate_sequences_ratio,
                        DDStrategy_gate_sequences_gate_pattern_xy,
                        DDStrategy_gate_sequences_align,
                        strict=True,
                    )
                ]

            strategy = DDStrategy(
                merge_contiguous_waits=DDStrategy_merge_contiguous_waits,
                target_qubits=None if DDStrategy_target_qubits is None else frozenset(DDStrategy_target_qubits),
                skip_leading_wait=DDStrategy_skip_leading_wait,
                skip_trailing_wait=DDStrategy_skip_trailing_wait,
                gate_sequences=gate_sequences,
            )

        if idling_specific_dd:
            # use idling-specific DD, with the default stategy as a fallback
            working_schedules, idle_windows = _prepare_idling_specific_dd_schedules(
                builder,
                schedules,
                merge_contiguous_waits=strategy.merge_contiguous_waits,
                skip_leading_wait=strategy.skip_leading_wait,
                skip_trailing_wait=strategy.skip_trailing_wait,
            )
            idle_map = _resolve_idle_dd_map(idling_specific_dd, idle_windows, working_schedules, builder)
            return _apply_idling_specific_dd_map(
                builder,
                working_schedules,
                idle_map,
                default_gate_sequences=strategy.gate_sequences,
                skip_leading_wait=strategy.skip_leading_wait,
                skip_trailing_wait=strategy.skip_trailing_wait,
            )

        new_schedules = []
        for schedule in schedules:
            if strategy.merge_contiguous_waits:
                new_schedule = merge_wait_instructions_in_schedule(builder, schedule)
            else:
                new_schedule = schedule.copy()
            new_schedule.cleanup()  # remove idling-only channels, where we want no DD taking place

            if qubit_specific_dd:
                # use qubit-specific DD strategies
                _insert_qubit_specific_dd_sequences(
                    builder,
                    new_schedule,
                    qubit_specific_dd,
                    skip_leading_wait=strategy.skip_leading_wait,
                    skip_trailing_wait=strategy.skip_trailing_wait,
                    default_gate_sequences=strategy.gate_sequences,
                )
            else:
                # use basic DD strategy
                insert_dd_sequences(builder, new_schedule, strategy)

            new_schedules.append(new_schedule)
        return new_schedules
    finally:
        # Compile-only DD settings (dicts/callables) are consumed here; strip them from the
        # settings tree so downstream finalization can serialize the settings into a RunDefinition.
        _strip_compile_only_dd_settings(context["settings"])


_COMPILE_ONLY_DD_SETTINGS: tuple[str, ...] = (
    "qubit_specific_dd",
    "idling_specific_dd",
)
"""Names of ``apply_dd_strategy`` settings whose values (dicts, callables) are
consumed at compile time and cannot be serialized into a ``RunDefinition``."""


def _strip_compile_only_dd_settings(node: SettingNode) -> None:
    """Null the compile-only DD fields on ``settings`` in place so it can be serialized."""
    for fragment in ("stages", "dynamical_decoupling", "apply_dd_strategy"):
        next_node = node.subtrees.get(fragment)
        if next_node is None:
            return
        node = next_node
    for setting_name in _COMPILE_ONLY_DD_SETTINGS:
        if setting_name in node.settings:
            node[setting_name] = None


def _gate_pattern_to_prx_sequence(pattern: str) -> PRXSequence:
    """Translate a gate pattern string (with gates noted as e.g. "X" and "Y") to a sequence of PRX args."""
    submitted_gates = set(pattern)
    supported_gates = set(_PRX_INSTANCES.keys())
    unsupported_gates = submitted_gates - supported_gates

    if unsupported_gates:
        raise DDError(
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
        # MoveMarker is a Wait subclass but marks the boundary of a MOVE sandwich;
        # collapsing it into an adjacent Wait would destroy the sandwich boundary and
        # cause the DD pass (which uses _get_move_sandwiched_indices to detect them)
        # to place pulses inside the sandwich. Flush any accumulated Wait first, then
        # keep the MoveMarker as its own instruction.
        if isinstance(instr, MoveMarker):
            if accumulating_waits:
                new_instructions.append(Wait(duration=wait_duration))
                accumulating_waits = False
                wait_duration = 0
            new_instructions.append(instr)
        elif isinstance(instr, Wait):
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

    raise DDError(f"Channel {channel_name} is not associated with any QPU component.")


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


def _get_move_sandwiched_indices(instructions: Iterable[Instruction]) -> frozenset[int]:
    """Return the indices of instructions that fall strictly inside a MOVE sandwich.

    A MOVE sandwich on a qubit drive channel is delimited by a pair of :class:`.MoveMarker`
    instructions (opening and closing). Any instructions at indices strictly between those
    markers must not have DD applied, because the qubit state has been moved to a resonator
    and applying single-qubit gates there is physically incorrect. In addition, because
    the qubit and the resonator operate at different frequencies, single-qubit gates
    applied inside the sandwich acquire extra phase errors (depending on the state of the
    MOVE qubit) and would no longer add up to the identity.

    The boundary ``MoveMarker`` indices themselves are *not* returned here. Callers that
    consume the result of :func:`locate_instructions` for ``Wait`` filter ``MoveMarker``
    out separately (via an explicit ``isinstance`` check), which keeps the two concerns
    — "this is a boundary annotation" and "this index is inside a sandwich" — cleanly
    separated.

    If the schedule contains no :class:`.MoveMarker` instructions (e.g. an IQM Crystal QPU),
    this function returns an empty set and has no effect on DD.

    Args:
        instructions: Instructions for a single channel.

    Returns:
        Instruction indices lying strictly inside any MOVE sandwich.

    """
    sandwiched: set[int] = set()
    open_idx: int | None = None
    for idx, instr in enumerate(instructions):
        if isinstance(instr, MoveMarker):
            if open_idx is None:
                open_idx = idx
            else:
                sandwiched.update(range(open_idx + 1, idx))
                open_idx = None
    return frozenset(sandwiched)


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
    * not be the first or last Wait on the channel, if the corresponding flag is set,
    * not be inside a MOVE sandwich (between a pair of :class:`.MoveMarker` instructions).

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

    # Cache sandwiched indices per channel to avoid recomputing for every Wait on the same channel.
    sandwiched_cache: dict[str, frozenset[int]] = {}

    for wait in all_waits:
        # MoveMarker is a Wait subclass but is a boundary annotation, not an idle window.
        if isinstance(schedule[wait.channel_name]._instructions[wait.index], MoveMarker):
            continue
        # Check if leading and/or trailing instruction should be excluded
        if skip_leading_wait and wait.index == 1:
            continue
        if skip_trailing_wait and wait.index == len(schedule[wait.channel_name]) - 3:  # third last index
            continue
        if skip_trailing_wait and wait.index == len(schedule[wait.channel_name]) - 1:  # last index
            continue
        # Skip Waits that fall inside a MOVE sandwich: the qubit state is in a resonator there,
        # so applying DD pulses would be physically incorrect.
        if wait.channel_name not in sandwiched_cache:
            sandwiched_cache[wait.channel_name] = _get_move_sandwiched_indices(
                schedule[wait.channel_name]._instructions
            )
        if wait.index in sandwiched_cache[wait.channel_name]:
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
        raise DDError(f"It is not possible to split {number}, because it is smaller than the split step {step}.")

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
        raise DDError(f"Unsupported single-qubit gate sequence alignment requested: {alignment}")

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
        raise DDError("The number of DD gates in the sequence has to be even.")

    seg_durations = [seg.duration for seg in dd_segments]

    if len(set(seg_durations)) != 1:
        raise DDError(
            f"The durations of all the gates in the provided DD gate sequence must be equal (now {seg_durations})."
        )

    total_gate_duration = sum(seg_durations)
    residual_waiting_time: int = wait_duration - total_gate_duration
    if residual_waiting_time < 0:
        raise DDError(
            f"The waiting time ({wait_duration} samples) cannot accomodate the {len(dd_segments)} DD gates required "
            f"(duration {total_gate_duration})."
        )

    residual_waiting_slots, remainder = divmod(residual_waiting_time, granularity)
    if remainder != 0:
        raise DDError("The (remaining) waiting time is not a multiple of the granularity!")

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
) -> dict[str, PrxGateImplementation]:
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
    dd_channels: dict[str, PrxGateImplementation] = {}

    if target_components is None:
        for channel_name in schedule.channels():
            if not _is_drive_channel(builder, channel_name):
                continue

            # Get the highest-priority PRX gate implementation available for the channel component.
            # A component without a usable PRX (e.g. no calibration data for it) cannot be dynamically
            # decoupled, so its channel is skipped.
            component_name = _channel_to_component(builder, channel_name)
            try:
                prx = builder.get_implementation("prx", (component_name,))
            except ValueError as error:
                cpc_logger.debug("No DD on channel %s: %s", channel_name, error)
                continue

            dd_channels[channel_name] = prx
    else:
        # We need more complicated error handling if target_components are given.
        all_components = builder.chip_topology.qubits | builder.chip_topology.computational_resonators
        diff = target_components - all_components
        if diff:
            raise DDError(f"Unknown target components {list(diff)}.")

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
                prx = builder.get_implementation("prx", (component_name,))
            except ValueError:
                errors.append(f"Drive channel {channel_name} of component {component_name} has no PRX available.")
                continue

            dd_channels[channel_name] = prx

        if errors:
            raise DDError("\n".join(errors))

    return dd_channels


def insert_dd_sequences(
    builder: ScheduleBuilder,
    schedule: Schedule,
    strategy: DDStrategy,
    *,
    per_wait_gate_sequences: dict[tuple[str, int], Iterable[DDGateSequence]] | None = None,
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
        per_wait_gate_sequences: Optional gate sequences for individual eligible ``Wait`` instructions,
            keyed by ``(channel_name, instruction_index)`` in ``schedule``. A list overrides
            ``strategy.gate_sequences`` for that ``Wait``, an empty list leaves that ``Wait`` without DD, and
            ``Wait`` instructions not in the mapping use ``strategy.gate_sequences``.

    """

    def _get_channel_segment(builder: ScheduleBuilder, timebox: TimeBox, drive_channel: str) -> Segment:
        """Extract the ``drive_channel`` segment of the schedule generated from the given timebox."""
        schedule = builder.resolve_timebox(timebox, neighborhood=0)
        return schedule[drive_channel]

    def _prepare(gate_sequences: Iterable[tuple[int, str | PRXSequence, str]]) -> list[tuple[int, PRXSequence, str]]:
        """Sort the sequences by ratio (descending) and translate "X"/"Y" gate patterns into PRX args."""
        return [
            (
                ratio,
                (_gate_pattern_to_prx_sequence(pattern) if isinstance(pattern, str) else pattern),
                alignment,
            )
            for ratio, pattern, alignment in sorted(gate_sequences, key=lambda seq: seq[0], reverse=True)
        ]

    strategy_dd_sequences = _prepare(strategy.gate_sequences)

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

        if per_wait_gate_sequences is not None:
            wait_gate_sequences = per_wait_gate_sequences.get((channel_name, wait.index))
            if not wait_gate_sequences:
                cpc_logger.debug("DD disabled for Wait on channel %s, index %d", channel_name, wait.index)
                continue
            dd_sequences = _prepare(wait_gate_sequences)
        else:
            dd_sequences = strategy_dd_sequences

        # NOTE: Duration is always calculated using "X"
        # FIXME PRX duration should become pre-calculated from _find_dd_channels
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
                    "Applying dynamical decoupling sequence %s with alignment %s",
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


def _insert_qubit_specific_dd_sequences(
    builder: ScheduleBuilder,
    schedule: Schedule,
    qubit_dd_map: dict[str, list[DDGateSequence] | None],
    *,
    skip_leading_wait: bool = True,
    skip_trailing_wait: bool = True,
    default_gate_sequences: Iterable[DDGateSequence] = (),
) -> None:
    """Insert qubit-specific dynamical decoupling sequences into the given schedule.

    For every eligible idle window on ``schedule`` (identified via the same rules
    used by :func:`_identify_idle_windows`), look up the qubit of the window in ``qubit_dd_map``:

    * value is a list of gate sequences → apply the longest of them that fits the idle,
    * value is empty or ``None`` → explicitly skip DD on that qubit,
    * qubit is missing → fall back to ``default_gate_sequences`` (or skip if that is empty).

    .. note:: Modifies ``schedule`` in-place.

    .. note::

       This helper function assumes that any desired merging of contiguous ``Wait`` instructions
       has already been performed on ``schedule`` (e.g. by
       :func:`merge_wait_instructions_in_schedule`).

    Args:
        builder: Schedule builder used to build ``schedule``, containing channel information.
        schedule: Schedule to modify.
        qubit_dd_map: Mapping from physical qubit name to a list of DD sequences, or ``None``
            meaning no DD is applied on that qubit.
        skip_leading_wait: Whether to skip the first ``Wait`` on each drive channel.
        skip_trailing_wait: Whether to skip the last ``Wait`` on each drive channel.
        default_gate_sequences: Fallback DD sequences for qubits not in ``qubit_dd_map``.
            Empty iterable means "no DD" for unlisted qubits.

    """
    per_wait_gate_sequences: dict[tuple[str, int], Iterable[DDGateSequence]] = {}
    for qubit, channel_name, wait_loc in _enumerate_eligible_waits_by_qubit(
        builder,
        schedule,
        skip_leading_wait=skip_leading_wait,
        skip_trailing_wait=skip_trailing_wait,
    ):
        per_wait_gate_sequences[channel_name, wait_loc.index] = qubit_dd_map.get(qubit, default_gate_sequences) or ()

    # With target_qubits=None, insert_dd_sequences visits exactly the Waits enumerated above.
    # Its own gate_sequences are empty, so no Wait gets DD unless the mapping above assigns it.
    strategy = DDStrategy(
        merge_contiguous_waits=False,
        target_qubits=None,
        skip_leading_wait=skip_leading_wait,
        skip_trailing_wait=skip_trailing_wait,
        gate_sequences=[],
    )
    insert_dd_sequences(builder, schedule, strategy, per_wait_gate_sequences=per_wait_gate_sequences)


# ─────────────────────────────────────────────────────────────────────────────
# Idling-specific DD
# ─────────────────────────────────────────────────────────────────────────────


class IdleWindowKey(NamedTuple):
    """Key type for :type:`IdleDDMap`: unique identifier for one eligible idle window."""

    schedule_index: int
    """Index of the schedule in the playlist."""
    qubit: str
    """Physical qubit name."""
    idle_index: int
    """Index of the eligible idle window on this qubit within this schedule."""


class EligibleWait(NamedTuple):
    """One eligible ``Wait`` window enumerated by :func:`_enumerate_eligible_waits_by_qubit`."""

    qubit: str
    """Physical qubit name driving the channel the wait sits on."""
    channel_name: str
    """Drive channel the wait sits on."""
    wait_location: InstructionLocation
    """Location of the ``Wait`` instruction inside the channel's segment."""


IdleDDMap: TypeAlias = dict[IdleWindowKey, list[DDGateSequence] | None]
"""Maps each eligible idle window in a schedule playlist to a DD strategy.

The value is either a priority list of DD gate sequences to apply in that window,
or ``None``, in which case no DD is applied.
Any eligible idle windows absent from the map fall back to default DD strategy.
"""


@dataclass
class IdleWindow:
    """A single eligible idle (``Wait``) period on a specific qubit's drive channel.

    Instances are used as keys (via their :attr:`key`) in :type:`IdleDDMap`.
    """

    schedule_index: int
    """Index of the schedule in the compiled list (= circuit index for ``sweeps=None``)."""
    qubit: str
    """Physical qubit name, e.g. ``"QB1"``."""
    channel_name: str
    """Drive channel associated with ``qubit``."""
    idle_index: int
    """0-based index among eligible idles on this qubit within this schedule."""
    instruction_index: int
    """Index of the ``Wait`` instruction in the channel's segment."""
    duration_samples: int
    """Duration of the ``Wait`` in samples."""

    @property
    def key(self) -> IdleWindowKey:
        """Identifier for this window."""
        return IdleWindowKey(self.schedule_index, self.qubit, self.idle_index)


class IdleDDMapBuilder(Protocol):
    """Callable form of the ``idling_specific_dd`` argument, used for automated per-window decisions."""

    def __call__(
        self,
        idle_windows: list[IdleWindow],
        schedules: list[Schedule],
        builder: ScheduleBuilder,
    ) -> IdleDDMap | None:
        """Callable form of the ``idling_specific_dd`` argument.

        Invoked by :func:`~iqm.cpc.compiler.standard_stages.apply_dd_strategy`
        after waits have been merged and idle windows
        enumerated but before any DD pulse is applied.

        Args:
            idle_windows: produced by :func:`_identify_idle_windows` on ``schedules``.
                Ordered by :class:`IdleWindowKey`.
            schedules: Pre-DD (waits-merged) schedules, in the same order as the
                original input to the compile call. **Do not mutate.**
            builder: Current :class:`.ScheduleBuilder`.

        Returns:
            An :type:`IdleDDMap` to be applied immediately, or ``None`` or ``{}`` to skip per-idle overrides
            in which case the standard ``DDStrategy`` is used, if provided).

        """
        ...


def _enumerate_eligible_waits_by_qubit(
    builder: ScheduleBuilder,
    schedule: Schedule,
    skip_leading_wait: bool,
    skip_trailing_wait: bool,
) -> list[EligibleWait]:
    """Enumerate eligible ``Wait`` windows in ``schedule`` in ``(qubit, channel, wait)`` form.

    The order matches :func:`_identify_idle_windows`: channels are visited in sorted
    order; within each channel, waits are visited in the natural (index-ascending)
    order returned by :func:`_find_wait_instructions_on_channels`.

    Args:
        builder: Schedule builder used to build ``schedule``.
        schedule: Schedule to search. Should already have had any desired wait-merging
            applied (see :func:`merge_wait_instructions_in_schedule`).
        skip_leading_wait: Skip the first ``Wait`` on each drive channel.
        skip_trailing_wait: Skip the last ``Wait`` on each drive channel.

    Returns:
        Possible ``Wait`` windows as ``(qubit_name, channel_name, wait_location)`` tuples.

    """
    dd_channels = _find_dd_channels(builder, schedule, target_components=None)
    result: list[EligibleWait] = []
    for channel_name in sorted(dd_channels):
        try:
            qubit = _channel_to_component(builder, channel_name)
        except DDError:
            continue
        waits = _find_wait_instructions_on_channels(
            schedule,
            channels=[channel_name],
            min_duration=0,
            skip_leading_wait=skip_leading_wait,
            skip_trailing_wait=skip_trailing_wait,
        )
        result.extend(EligibleWait(qubit, channel_name, wait_loc) for wait_loc in waits)
    return result


def _identify_idle_windows(
    builder: ScheduleBuilder,
    schedules: list[Schedule],
    skip_leading_wait: bool = True,
    skip_trailing_wait: bool = True,
) -> list[IdleWindow]:
    """Enumerate eligible idle (``Wait``) windows in the given schedules.

    ``schedules`` must already have their waits merged as needed (see
    :func:`_prepare_idling_specific_dd_schedules`). ``skip_leading_wait`` and
    ``skip_trailing_wait`` must match the values later used by
    :func:`_insert_idle_specific_dd_sequences` so that ``idle_index`` values stay consistent.

    Args:
        builder: Schedule builder used to build ``schedules``.
        schedules: Prepared pre-DD schedules. They are not modified.
        skip_leading_wait: Exclude the first ``Wait`` on each channel.
        skip_trailing_wait: Exclude the last ``Wait`` on each channel.

    Returns:
        Idle windows, sorted by :class:`IdleWindowKey`.

    """
    windows: list[IdleWindow] = []
    for sched_idx, schedule in enumerate(schedules):
        qubit_idle_counter: dict[str, int] = {}
        for qubit, channel_name, wait_loc in _enumerate_eligible_waits_by_qubit(
            builder,
            schedule,
            skip_leading_wait=skip_leading_wait,
            skip_trailing_wait=skip_trailing_wait,
        ):
            idle_idx = qubit_idle_counter.get(qubit, 0)
            qubit_idle_counter[qubit] = idle_idx + 1
            windows.append(
                IdleWindow(
                    schedule_index=sched_idx,
                    qubit=qubit,
                    channel_name=channel_name,
                    idle_index=idle_idx,
                    instruction_index=wait_loc.index,
                    duration_samples=wait_loc.duration,
                )
            )

    windows.sort(key=lambda w: (w.schedule_index, w.qubit, w.idle_index))
    return windows


def _prepare_idling_specific_dd_schedules(
    builder: ScheduleBuilder,
    schedules: list[Schedule],
    merge_contiguous_waits: bool = True,
    skip_leading_wait: bool = True,
    skip_trailing_wait: bool = True,
) -> tuple[list[Schedule], list[IdleWindow]]:
    """Prepare pre-DD schedules and enumerate their idle windows.

    This is the first half of the idling-specific DD workflow. It is shared by the
    DD pass and by :func:`identify_idle_windows_for_settings` so that the windows shown
    to the user are exactly the windows that later DD insertion will use.

    Args:
        builder: Schedule builder used to build ``schedules``.
        schedules: Pre-DD schedules. They are not modified.
        merge_contiguous_waits: Merge adjacent ``Wait`` instructions before enumeration.
        skip_leading_wait: Exclude the first ``Wait`` on each channel.
        skip_trailing_wait: Exclude the last ``Wait`` on each channel.

    Returns:
        Copies of ``schedules`` with waits merged (if requested) and idling-only channels removed,
        and their idle windows, sorted by :class:`IdleWindowKey`.

    """
    working_schedules: list[Schedule] = []
    for schedule in schedules:
        if merge_contiguous_waits:
            new_schedule = merge_wait_instructions_in_schedule(builder, schedule)
        else:
            new_schedule = schedule.copy()
        new_schedule.cleanup()  # remove idling-only channels
        working_schedules.append(new_schedule)

    idle_windows = _identify_idle_windows(
        builder,
        working_schedules,
        skip_leading_wait=skip_leading_wait,
        skip_trailing_wait=skip_trailing_wait,
    )
    return working_schedules, idle_windows


def identify_idle_windows_for_settings(schedules: list[Schedule], context: dict[str, Any]) -> list[IdleWindow]:
    """Enumerate the idle windows that the DD pass will see in the given pre-DD schedules.

    Use this when compiling in parts (see the module docstring): run the stages up to the
    ``dynamical_decoupling`` stage, call this function on the returned schedules and context,
    and use the keys of the returned windows to build an :type:`IdleDDMap`.

    The window enumeration options (``DDStrategy_merge_contiguous_waits``,
    ``DDStrategy_skip_leading_wait``, ``DDStrategy_skip_trailing_wait``) are read from
    ``context["settings"]`` in the same way as for the
    :func:`~iqm.cpc.compiler.standard_stages.apply_dd_strategy` pass, so the keys match the
    windows the pass will apply the map to.

    Args:
        schedules: Schedules returned by the stages preceding the ``dynamical_decoupling`` stage.
        context: Compiler context returned together with ``schedules``.

    Returns:
        :class:`IdleWindow` list, sorted by :class:`IdleWindowKey`.

    """
    stage_options = setting_node_to_dict(context["settings"].stages)
    common_options = {key: value for key, value in stage_options.items() if not isinstance(value, dict)}
    dd_options = stage_options.get("dynamical_decoupling", {}).get("apply_dd_strategy", {}) | common_options
    _, idle_windows = _prepare_idling_specific_dd_schedules(
        context["builder"],
        schedules,
        merge_contiguous_waits=dd_options.get("DDStrategy_merge_contiguous_waits", True),
        skip_leading_wait=dd_options.get("DDStrategy_skip_leading_wait", True),
        skip_trailing_wait=dd_options.get("DDStrategy_skip_trailing_wait", True),
    )
    return idle_windows


def _resolve_idle_dd_map(
    idle_spec: IdleDDMap | IdleDDMapBuilder | None,
    idle_windows: list[IdleWindow],
    schedules: list[Schedule],
    builder: ScheduleBuilder,
) -> IdleDDMap:
    """Resolve static or callable idling-specific DD input to an :type:`IdleDDMap`.

    Args:
        idle_spec: The ``idling_specific_dd`` setting: a map, a callable that returns one, or ``None``.
        idle_windows: Idle windows of ``schedules``, passed to a callable ``idle_spec``.
        schedules: Prepared pre-DD schedules, passed to a callable ``idle_spec``.
        builder: Schedule builder, passed to a callable ``idle_spec``.

    Returns:
        The map to apply. Empty if ``idle_spec`` is ``None`` or a callable returned ``None``.

    Raises:
        TypeError: A callable ``idle_spec`` returned something other than a dict or ``None``.

    """
    if callable(idle_spec):
        idle_spec = idle_spec(idle_windows, schedules, builder)
        if idle_spec is not None and not isinstance(idle_spec, dict):
            raise TypeError(
                f"idling_specific_dd callable must return an IdleDDMap or None, got {type(idle_spec).__name__}."
            )
    return dict(idle_spec) if idle_spec else {}


def display_idle_windows(idle_windows: list[IdleWindow]) -> None:
    """Pretty-print idle windows grouped by schedule index and qubit.

    Output columns: ``(sched, qubit, idle_idx)`` key, duration in samples,
    channel name, instruction index.

    Args:
        idle_windows: Idle windows to print, e.g. from :func:`identify_idle_windows_for_settings`.

    """
    col1, col2, col3, col4 = 34, 20, 26, 13
    header = (
        f"{'Key (sched, qubit, idle_idx)':<{col1}}"
        f"{'Duration (samples)':>{col2}}"
        f"  {'Channel':<{col3}}"
        f"{'Instr. Index':>{col4}}"
    )
    print(header)
    print("─" * (col1 + col2 + col3 + col4 + 2))

    grouped: dict[tuple[int, str], list[IdleWindow]] = {}
    for w in idle_windows:
        grouped.setdefault((w.schedule_index, w.qubit), []).append(w)

    for (sched_idx, qubit), group in sorted(grouped.items()):
        for w in group:
            key_str = f"({sched_idx}, {qubit!r}, {w.idle_index})"
            print(
                f"{key_str:<{col1}}{w.duration_samples:>{col2}}  {w.channel_name:<{col3}}{w.instruction_index:>{col4}}"
            )


def _insert_idle_specific_dd_sequences(
    builder: ScheduleBuilder,
    schedule: Schedule,
    schedule_index: int,
    idle_dd_map: IdleDDMap,
    default_gate_sequences: Iterable[DDGateSequence] = (),
    skip_leading_wait: bool = True,
    skip_trailing_wait: bool = True,
) -> None:
    """Insert per-idle DD sequences into ``schedule`` (in-place).

    For every eligible idle window on ``schedule`` (identified via the same rules
    used by :func:`_identify_idle_windows`), look up its key
    :class:`IdleWindowKey` in ``idle_dd_map``:

    * value is a list of gate sequences → apply the longest of them that fits the idle,
    * value is ``None`` → explicitly skip DD on that idle,
    * key is missing → fall back to ``default_gate_sequences`` (or skip if that is
      also ``None``).

    .. note:: Modifies ``schedule`` in-place. Any desired merging of contiguous
       ``Wait`` instructions must have already been performed on ``schedule``
       (e.g. by :func:`merge_wait_instructions_in_schedule`), matching what was
       used for :func:`_identify_idle_windows`.

    Args:
        builder: Schedule builder used to build ``schedule``.
        schedule: Schedule to modify.
        schedule_index: Index of this schedule within the compiled list — must match
            the ``schedule_index`` recorded in ``idle_dd_map`` keys.
        idle_dd_map: Mapping from :class:`IdleWindowKey` to a list of
            :class:`DDGateSequence` items, or ``None``. See :type:`IdleDDMap`.
        default_gate_sequences: Fallback DD strategy for idles not in ``idle_dd_map``.
            Empty iterable means "no DD" for unlisted idles.
        skip_leading_wait: Skip the first ``Wait`` on each drive channel.
        skip_trailing_wait: Skip the last ``Wait`` on each drive channel.

    """
    # Enumerate windows exactly as _identify_idle_windows would, to get consistent
    # idle_index values, and translate the idle window keys into Wait locations.
    qubit_idle_counter: dict[str, int] = {}
    per_wait_gate_sequences: dict[tuple[str, int], Iterable[DDGateSequence]] = {}

    for qubit, channel_name, wait_loc in _enumerate_eligible_waits_by_qubit(
        builder,
        schedule,
        skip_leading_wait=skip_leading_wait,
        skip_trailing_wait=skip_trailing_wait,
    ):
        idle_idx = qubit_idle_counter.get(qubit, 0)
        qubit_idle_counter[qubit] = idle_idx + 1

        key = IdleWindowKey(schedule_index, qubit, idle_idx)
        per_wait_gate_sequences[channel_name, wait_loc.index] = idle_dd_map.get(key, default_gate_sequences) or ()

    # With target_qubits=None, insert_dd_sequences visits exactly the Waits enumerated above.
    # Its own gate_sequences are empty, so no Wait gets DD unless the mapping above assigns it.
    strategy = DDStrategy(
        merge_contiguous_waits=False,
        target_qubits=None,
        skip_leading_wait=skip_leading_wait,
        skip_trailing_wait=skip_trailing_wait,
        gate_sequences=[],
    )
    insert_dd_sequences(builder, schedule, strategy, per_wait_gate_sequences=per_wait_gate_sequences)


def _apply_idling_specific_dd_map(
    builder: ScheduleBuilder,
    schedules: list[Schedule],
    idle_dd_map: IdleDDMap,
    *,
    default_gate_sequences: Iterable[DDGateSequence] = (),
    skip_leading_wait: bool = True,
    skip_trailing_wait: bool = True,
) -> list[Schedule]:
    """Apply an :type:`IdleDDMap` to already prepared pre-DD schedules.

    ``schedules`` should come from :func:`_prepare_idling_specific_dd_schedules`,
    or otherwise use the same wait-merging and cleanup rules. The schedules are
    modified in-place and returned for convenience.

    Args:
        builder: Schedule builder used to build ``schedules``.
        schedules: Prepared pre-DD schedules, in compilation order.
        idle_dd_map: DD gate sequences (or ``None`` to skip DD) per idle window.
        default_gate_sequences: Fallback gate sequences for idle windows not in ``idle_dd_map``.
            Empty iterable here means no DD for unlisted idles.
        skip_leading_wait: Skip the first ``Wait`` on each drive channel.
        skip_trailing_wait: Skip the last ``Wait`` on each drive channel.

    Returns:
        ``schedules``, with DD sequences inserted.

    """
    for schedule_index, schedule in enumerate(schedules):
        _insert_idle_specific_dd_sequences(
            builder,
            schedule,
            schedule_index=schedule_index,
            idle_dd_map=idle_dd_map,
            default_gate_sequences=default_gate_sequences,
            skip_leading_wait=skip_leading_wait,
            skip_trailing_wait=skip_trailing_wait,
        )
    return schedules


def generate_phases_of_urn_sequence(n: int) -> list[float]:
    """Generate PRX gate phases for the URn sequence.

    More information on the URn sequence is available in :cite:`Ezzell_2022`, at the end in Appendix A.1.

    Args:
        n: Number of single qubit PRX pulses with different phases to apply, must be a positive even number.

    Returns:
        Phases to be used by the URn sequence.

    """
    if (n <= 0) or (n % 2 != 0):
        raise ValueError("n should be a positive even number.")

    phi = math.pi / (n // 4) if n % 4 == 0 else 2 * (n // 4) * math.pi / (2 * (n // 4) + 1)

    # Normalize phase angles to interval [0, 2 * pi)
    phases = [
        (index_pulse * (index_pulse - 1) * phi / 2 + index_pulse * math.pi / 2) % (2 * math.pi)
        for index_pulse in range(n)
    ]
    return phases
