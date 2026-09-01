#  ********************************************************************************
#
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
"""Reserving QPU resources in instruction scheduling."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, replace
import enum
from functools import reduce
from typing import Self

from iqm.pulse.playlist.instructions import Block, ReadoutTrigger
from iqm.pulse.playlist.schedule import Schedule, Segment


class SchedulingAlgorithm(enum.Enum):
    """Algorithms for resolving composite TimeBoxes into atomic ones."""

    HARD_BOUNDARY = "HARD_BOUNDARY"
    """Respects the ``TimeBox`` boundary such that the longest channel with a box defines
    its boundary and all other channels are padded to this length (using the specified ``SchedulingStrategy``)."""
    TETRIS = "TETRIS"
    """Will pack the schedule as tightly as possible while respecting the defined scheduling neighborhood."""


class SchedulingStrategy(enum.Enum):
    """Different scheduling strategies for the contents of composite TimeBoxes."""

    ASAP = "ASAP"
    """TimeBox contents are scheduled as soon as possible within the box."""
    ALAP = "ALAP"
    """TimeBox contents are scheduled as late as possible within the box."""


@dataclass
class TimeBox:
    """Container for one or more instruction schedule fragments, to be scheduled according to a given strategy.

    Each TimeBox can be labeled using a human-readable *label* describing it, and operates on a number
    of *locus components*, using some of their control channels.  It can be either *atomic* or
    *composite*.

    - An atomic box only contains a single :class:`.Schedule`.
    - A composite box contains a sequence of other TimeBoxes as its children.
      The locus components are the union of the locus components of the children.
      If two children use the same channel so that they cannot happen simultaneously, they must
      happen in the order they occur in the sequence.

    A box can be made atomic by *resolving* it using :class:`.ScheduleBuilder.resolve_timebox`.
    The time duration of the box is determined by its contents and the way they are scheduled during the resolution.

    TimeBoxes can be concatenated with the following rules:

    - Addition concatenates the children of the operands into a single TimeBox.
    - The pipe operation groups two TimeBoxes together without concatenating.
    - Iterables of Boxes are treated as the sum of the elements.

    Let ``a, b, c, d`` be TimeBoxes. Then

    .. code-block:: python

        a_then_b = a + b
        c_then_d = (c + d).set_alap()
        abcd = a_then_b | c_then_d

        abb = a + [b, b]
        ccd = [c, c] | d

        all_together = abcd | abb + ccd

        # is equivalent to:
        all_together = a + b | (c + d).set_alap() | a + b + b + c + (c | d)

    """

    label: str
    """Description the contents of the box for users' convenience. Has no functional effect."""

    locus_components: set[str]
    """Names of the QPU components on which this timebox operates. These can include additional components
    to the ones included in one of the channels occupied by this ``TimeBox``. The components included in this
    attribute will be blocked in scheduling, in addition to the ones dictated by the neighborhood range (see
    :attr:`.neighborhood_components`)."""

    atom: Schedule | None
    """Resolved contents of the TimeBox, or None if not resolved."""

    children: tuple[TimeBox, ...] = field(default_factory=tuple)
    """Further Timeboxes inside this TimeBox."""

    scheduling: SchedulingStrategy = SchedulingStrategy.ASAP
    """Determines how the contents of a composite TimeBox are scheduled by ScheduleBuilder.
    Has no meaning for an atomic TimeBox."""

    scheduling_algorithm: SchedulingAlgorithm = SchedulingAlgorithm.HARD_BOUNDARY
    """Determines the algorithm used in converting the TimeBox to a Schedule."""

    neighborhood_components: dict[int, set[str]] = field(default_factory=dict)
    """Dict of neighborhood range integers mapped to sets of components neighboring the locus of this ``TimeBox``.
     These are used in the scheduling when the corresponding neighborhood range is used.
     The scheduling algorithm computes the neighborhood components (unless it has been already precomputed by
     e.g. the :class:`.GateImplementation`) and caches them under this attribute.
     Neighborhood range 0 means just the components affected by one of the channels in ``self.atom`` + ``self.locus``,
     1 means also neighboring couplers, 2 the components connected to those couplers, and so on.
     Note: range 0 may differ from ``self.locus_components``: it can have additional components that
     have occupied channels in ``self`` but are not defined as a part of the locus of
     this ``TimeBox`` for any reason.
    """

    @classmethod
    def composite(
        cls,
        boxes: TimeBox | Iterable[TimeBox | Iterable[TimeBox]],
        *,
        label: str = "",
        scheduling: SchedulingStrategy = SchedulingStrategy.ASAP,
        scheduling_algorithm: SchedulingAlgorithm = SchedulingAlgorithm.HARD_BOUNDARY,
    ) -> Self:
        """Build a composite timebox from a sequence of timeboxes.

        Args:
            boxes: contents of the new timebox. Any iterables of timeboxes will be flattened (recursively) and extended
                to the contents in the same order.
            label: label of the new timebox
            scheduling: scheduling strategy to use when resolving the new timebox
            scheduling_algorithm: scheduling algorithm to use when resolving the new timebox

        Returns:
            composite timebox containing ``boxes`` as its children

        """
        children = []
        for child in boxes:  # type: ignore[union-attr]
            if isinstance(child, TimeBox):
                child_box = child
                children.append(child_box)
            else:
                child_box = TimeBox.composite(child, scheduling=scheduling, scheduling_algorithm=scheduling_algorithm)
                children.extend(list(child_box.children))
        if boxes:
            locus_components = set.union(*(box.locus_components for box in children))
        else:
            locus_components = set()
        return cls(
            label=label,
            locus_components=locus_components,
            atom=None,
            children=tuple(children),
            scheduling=scheduling,
            scheduling_algorithm=scheduling_algorithm,
        )

    @classmethod
    def atomic(cls, schedule: Schedule, *, locus_components: Iterable[str], label: str) -> Self:
        """Build an atomic timebox from a schedule.

        Args:
            schedule: contents of the new timebox
            locus_components: names QPU components ``schedule`` operates on
            label: label of the new timebox

        Returns:
            atomic timebox containing ``schedule``

        """
        return cls(label=label, locus_components=set(locus_components), atom=schedule, children=())

    def validate(self, path: tuple[str, ...] = ()) -> None:
        """Validate the contents of the TimeBox.

        Args:
            path: Labels of ancestor boxes, to generate a better error message.

        """
        new_path = path + (self.label,)
        if self.atom:
            self.atom.validate(new_path)
            return

        for child in self.children:
            child.validate(new_path)

    def set_asap(self) -> TimeBox:
        """Set the scheduling strategy to As soon as possible (ASAP)."""
        self.scheduling = SchedulingStrategy.ASAP
        return self

    def set_alap(self) -> TimeBox:
        """Set the scheduling strategy to As late as possible (ALAP)."""
        self.scheduling = SchedulingStrategy.ALAP
        return self

    def __getitem__(self, item: int) -> TimeBox:
        """Shortcut for ``self.children[item]``."""
        if not self.children:
            raise ValueError(f"Tried to access a child of {self}, which is atomic.")
        return self.children[item]

    def _add_children(self, other: TimeBox) -> TimeBox:
        """Concat the children of self and other together."""
        left = self.children if self.atom is None else (self,)
        right = other.children if other.atom is None else (other,)
        return TimeBox(
            label=self.label,
            locus_components=self.locus_components.union(other.locus_components),
            atom=None,
            children=left + right,
            scheduling=self.scheduling,
            scheduling_algorithm=self.scheduling_algorithm,
        )

    def __add__(self, other: TimeBox | Iterable[TimeBox]) -> TimeBox:
        """Return a new TimeBox which has the contents of this and another TimeBox concatenated.

        Used to concatenate multiple TimeBoxes, like atomic operations, to a single logical entity.

        The add operation is associative: for boxes ``a, b, c``, these are equivalent:
        ``a+b+c == (a+b)+c == a+(b+c) == a+[b,c] == [a,b]+c``.

        The scheduling strategy and label are given by ``self``, i.e. the leftmost operand.

        Args:
             other: TimeBox or an iterable of TimeBoxes whose contents to merge.

        Returns:
            A new instance containing the children of both boxes.

        """
        if issubclass(type(other), TimeBox) and type(other) is not TimeBox:  # strict subclass
            # allow subclasses to override __add__ such that __radd__ also works consistent with that logic
            return other.__radd__(self)  # type: ignore[union-attr]
        if isinstance(other, TimeBox):
            return self._add_children(other)
        try:
            return reduce(lambda x, y: x + y, other, self)
        except TypeError as err:
            raise TypeError(f"Cannot add a TimeBox and a {type(other)}.") from err

    def __radd__(self, other: TimeBox | Iterable[TimeBox]) -> TimeBox:
        if isinstance(other, TimeBox):
            return other._add_children(self)
        it = iter(other)
        try:
            first = next(it)
        except StopIteration:
            return self
        return reduce(lambda x, y: x + y, it, first) + self

    def __iadd__(self, other: TimeBox | Iterable[TimeBox]) -> TimeBox:
        """Concatenate contents of another TimeBox to this TimeBox.

        Args:
             other: TimeBox whose contents to merge.

        Returns:
            Self, modified to contain the children of both boxes.

        """
        if self.atom is not None:
            raise ValueError("Cannot add content to an atomic TimeBox.")
        new = self + other
        self.children = new.children
        self.locus_components = new.locus_components
        return self

    def __or__(self, other: TimeBox | Iterable[TimeBox]) -> TimeBox:
        """Construct a new composite TimeBox that contains ``self`` and ``other``.

        Used to group two TimeBoxes without mixing their properties.
        Useful for separating boxes which serve a logically distinct purpose.
        For example, for boxes ``a, b, c, d``,  ``a+b|c+d`` results in the content
        ``[[a, b], [c, d]]``, preserving the properties of ``a+b`` and ``c+d``.
        This way, ``a+b`` can be scheduled according to a different strategy than ``c+d``, for example.
        This is in contrast to ``(a+b)+(c+d) == a+b+c+d``, which results in the content ``[a, b, c, d]``.

        This operation is not associative: ``a|b|c != a|(b|c)`` as these result in box contents ``[[a, b], c]``,
        ``[a, [b, c]]``, respectively.

        Args:
             other: TimeBox to append.

        Returns:
            A new TimeBox containing ``self`` and ``other`` as children.

        """
        if isinstance(other, TimeBox):
            other = [other]
        return TimeBox.composite([self, reduce(lambda x, y: x + y, other)])

    def __ror__(self, other: Iterable[TimeBox]) -> TimeBox:
        return TimeBox.composite([reduce(lambda x, y: x + y, other), self])

    def print(self, _idxs: tuple[int, ...] = ()) -> None:
        """Print a simple representation of the contents of this box."""
        location = "".join(f"[{idx}]" for idx in _idxs)
        location = f"{location}:".ljust(12)
        label = self.label or f"(unnamed on {self.locus_components})"
        atomic = " (atomic)" if self.atom else ""
        print(f"{location}{label}{atomic}")
        for i, child in enumerate(self.children):
            child.print(_idxs + (i,))


@dataclass
class MeasureBox(TimeBox):
    """TimeBox associated with a measure implementation.

    Supports :meth:`multiplex` between two class instances. Contains :attr:`boxes_length`
    and :attr:`measure_components` for identifying all class instances associated with a specific measure
    call.

    The purpose of this class is to 1) denote that a TimeBox is associated with a measure implementation 2) identify all
    the MeasureBox instances associated with a given measure call via the boxes_length. Both of these features are
    needed in the circuit-level multiplexing pass (some client-side libraries like Qiskit do not support multiplexed
    measurements out-of-the-box, requiring a dedicated compiler pass on the IQM-side).
    """

    boxes_length: int = 0
    """How many MeasureBox instances the MeasureBoxes instance this MeasureBox belongs to has (used in circuit-level
    multiplexing)."""
    measure_components: frozenset[str] = field(default_factory=frozenset)
    """The :attr:`MeasureBoxes.measure_components` of the MeasureBoxes ``self`` belongs to."""
    additional_components: frozenset[str] = field(default_factory=frozenset)
    """The :attr:`MeasureBoxes.additional_components` of the MeasureBoxes ``self`` belongs to."""

    def multiplex(self, other: MeasureBox) -> MeasureBox:
        """Multiplex two MeasureBox instances together.

        The same as ``self + other`` with the neighborhood components combined.
        """
        left = self.children if self.atom is None else (self,)
        right = other.children if other.atom is None else (other,)
        nb0_self = self.neighborhood_components.get(0, set())
        nb0_other = other.neighborhood_components.get(0, set())
        return MeasureBox(
            label=self.label,
            locus_components=self.locus_components.union(other.locus_components),
            atom=None,
            children=left + right,
            scheduling=self.scheduling,
            scheduling_algorithm=self.scheduling_algorithm,
            neighborhood_components={0: nb0_self.union(nb0_other)},
        )


class MultiplexedProbeTimeBox(MeasureBox):
    """A ``MeasureBox`` for multiplexing IQ pulses in probe channels.

    A MultiplexedProbeTimeBox's atom contains exactly one ``ReadoutTrigger`` for each probe channel. This class
    implements :meth:`multiplex` between `MultiplexedProbeTimeBox` instances such that every ``ReadoutTrigger`` on
    overlapping probe channels is multiplexed with :meth:`ReadoutTrigger.__add__` (non-overlapping probe channel
    contents are just combined).
    """

    def _multiplex(self, other_atom: Schedule) -> dict[str, Segment]:
        new_segments = dict(self.atom.copy().items())  # type: ignore[union-attr]
        for channel, segment in other_atom.items():
            if channel not in new_segments:
                new_segments[channel] = segment
            elif isinstance(segment[0], ReadoutTrigger) and isinstance(new_segments[channel][0], ReadoutTrigger):
                # multiplex the readout triggers together
                new_segments[channel]._instructions[0] = new_segments[channel][0] + segment[0]
            else:
                new_segments[channel].extend(iter(segment))
        return new_segments

    def multiplex(self, other: MeasureBox) -> MeasureBox:
        """Multiplex two atomic ``MultiplexedProbeTimeBox`` instances together.

        The multiplexing is done such that every ``ReadoutTrigger`` belonging to the same probe channel are
        multiplexed together. Otherwise, behaves like ``MeasureBox.__add__``, returning a normal ``MeasureBox``.

        """
        if isinstance(other, MultiplexedProbeTimeBox) and self.atom is not None and other.atom is not None:
            new_segments = self._multiplex(other.atom)
            locus_components = self.locus_components.union(other.locus_components)
            max_nb = max(
                max(self.neighborhood_components, default=-1),
                max(other.neighborhood_components, default=-1),
            )
            # Combine a neighborhood for the two boxes only if precomputed for both.
            # If a neighborhood is not precomputed for either one, we must leave it empty in the multiplexed result
            # for the scheduler to compute correctly.
            neighborhood_components: dict[int, set[str]] = {}
            for nb in range(max_nb + 1):
                if nb in self.neighborhood_components and nb in other.neighborhood_components:
                    neighborhood_components[nb] = self.neighborhood_components[nb].union(
                        other.neighborhood_components[nb]
                    )
            return type(self)(
                label=f"MultiplexedProbeTimeBox on {locus_components}",
                locus_components=locus_components,
                atom=Schedule(new_segments),
                children=(),
                scheduling=self.scheduling,
                scheduling_algorithm=self.scheduling_algorithm,
                neighborhood_components=neighborhood_components,
            )
        return super().multiplex(other)

    @staticmethod
    def from_readout_trigger(
        readout_trigger: ReadoutTrigger,
        probe_channel: str,
        locus_components: Iterable[str],
        *,
        label: str = "",
        block_channels: Iterable[str] = (),
        block_duration: int = 0,
    ) -> MultiplexedProbeTimeBox:
        """Build an atomic MultiplexedProbeTimeBox from a single ReadoutTrigger instruction.

        Args:
            readout_trigger: Readout trigger instruction.
            probe_channel: Name of the probe channel to play ``readout_trigger`` in.
            locus_components: Locus components.
            label: Label of the new timebox.
            block_channels: Names of channels to block.
            block_duration: Duration of the required blocking (in samples).

        Returns:
            Atomic timebox containing ``readout_trigger`` in the channel ``probe_channel``.

        """
        schedule = {probe_channel: [readout_trigger]}
        for channel in block_channels:
            schedule[channel] = [Block(block_duration)]  # type: ignore[list-item]
        box = MultiplexedProbeTimeBox(
            label=label,
            locus_components=set(locus_components),
            atom=Schedule(schedule, duration=readout_trigger.duration),
            children=(),
            scheduling=SchedulingStrategy.ASAP,
            scheduling_algorithm=SchedulingAlgorithm.HARD_BOUNDARY,
        )
        return box


class MeasureMultiplexingError(TypeError):
    """Error in multiplexing two ``MeasureBoxes`` instances together.

    Raised for example when the ``MeasureBoxes`` instances have overlapping :attr:`MeasureBoxes.measure_components`.
    """


class MeasureBoxes(list):
    """Multiplexable list of TimeBoxes associated with a measure implementation.

    All ``measure`` implementations' ``__call__`` method should return a ``MeasureBoxes`` instance. The
    measurements can be multiplexed together via :meth:`multiplex`, allowing parallel measurements
    within a single probe line.

    A ``MeasureBoxes`` instance has the following structure: ``[A] + [P, B, A] * n``, where
    ``P`` is a :class:`.MultiplexedProbeTimeBox` instance (the actual readout triggers),
    ``B`` is a MeasureBox containing extra blocking (if any) on the probe channel after the measurement, and
    ``A`` is an auxiliary MeasureBox associated with a measure implementation (e.g. the ``prx_12``
    shelving operation in shelved measurements). So, there is an initial auxiliary
    box and then any number of triplets ``P, B, A``. The auxiliary boxes ``A`` and the
    extra probe block boxes ``B`` can be empty.``.

    Args:
        measure_boxes: The measure boxes.
        measure_components: Primary component names this measure ``MeasureBoxes`` instance is associated with. Note that
            this may be different from the ``TimeBox.locus_components`` attributes if its member boxes. Typically, this
            is the same as the measured qubits, but in some measure implementations, additional components may also
            be pinned in the operation (see :meth:`.MeasureBoxes.multiplex`).
        additional_components: Additional component names this measure ``MeasureBoxes`` instance is associated with
            (see :meth:`.MeasureBoxes.multiplex`). By default, initiated as an empty set. The attribute can be used e.g.
            If the multiplexing should be blocked in some neighborhood of the ``measure_components`` (next neighbor
            qubits).

    """

    def __init__(
        self,
        measure_boxes: list[MeasureBox],
        measure_components: Iterable[str],
        additional_components: Iterable[str] | None = None,
    ):
        self.measure_components = frozenset(measure_components)
        self.additional_components = frozenset(additional_components) if additional_components else frozenset()
        # validate contents and resolve probe durations
        error = TypeError(
            "Invalid MeasureBoxes pattern: "
            f"{[b.__class__.__name__ + (':atomic' if b.atom else ':composite') for b in measure_boxes]}"
        )
        if (len(measure_boxes) - 1) % 3 != 0:
            raise error
        if not isinstance(measure_boxes[0], MeasureBox):  # the first element is an aux box
            raise error
        probe_indices = []
        probe_durations: list[list[int]] = []
        for idx, box in enumerate(measure_boxes[1:]):
            modulo = idx % 3
            match modulo:
                case 0:  # the actual probe box
                    if not isinstance(box, MultiplexedProbeTimeBox):
                        raise error
                    probe_indices.append(idx + 1)
                    if box.atom:
                        probe_durations.append([next((box.atom[pl].duration for pl in box.atom if "readout" in pl), 0)])
                    else:
                        probe_durations.append([0])
                case 1:  # Probe block box
                    if not isinstance(box, MeasureBox) or box.children:
                        raise error
                    if not box.atom:
                        probe_durations[-1].append(0)
                    else:
                        probe_durations[-1].append(
                            next((box.atom[pl].duration for pl in box.atom if "readout" in pl), 0)
                        )
                case 2:  # aux box
                    if not isinstance(box, MeasureBox):
                        raise error
        self.probe_indices = tuple(probe_indices)
        self.probe_durations = tuple(tuple(t) for t in probe_durations)
        # finally init the list
        # shallow copy the boxes as we need to assign boxes_length to them -- otherwise we could cause conflicts
        # as a single TimeBox object might be a member of multiple MeasureBoxes instances
        super().__init__(
            [
                replace(
                    b,
                    boxes_length=len(measure_boxes),
                    measure_components=self.measure_components,
                    additional_components=self.additional_components,
                )  # type: ignore[call-arg]
                for b in measure_boxes
            ]
        )

    def is_multiplexable_with(self, other: MeasureBoxes) -> bool:
        """Whether ``self`` can be multiplexed with ``other``."""
        if self.measure_components.intersection(other.measure_components):
            return False
        if self.additional_components.intersection(other.measure_components) or self.measure_components.intersection(
            other.additional_components
        ):
            return False
        return True

    def multiplex(self, other: MeasureBoxes) -> MeasureBoxes:
        """Multiplex two ``MeasureBoxes`` instances together.

        The multiplexing works such that the lists are aligned ASAP (as soon as possible) to the left. The auxiliary
        boxes ``A`` in the overlapping parts of the lists are summed together (:meth:`.MeasureBox.multiplex` resolves
        to just the normal :meth:`.TimeBox.__add__` method), the probe boxes ``P`` are multiplexed with
        ``MultiplexedProbeTimeBox.multiplex``. The durations of the ``P`` boxes and their subsequent ``B`` boxes are
        adjusted so that 1) the duration of the ``P`` is the maximum of multiplexed boxes durations 2) the duration
        of the new ``B`` box is
        ``<multiplexed block duration> = max((P1 + B1).duration, P2 + B2).duration) - max(P1.duration, P2.duration)``.

        If the multiplexed ``MeasureBoxes`` lists are of a different length, the tail of the longer one is copied
        as it is into the multiplexed result.

        If the :attr:`.MeasureBoxes.measure_components` of ``self`` and ``other`` have an overlap, an error is raised.
        The error is raised also if :attr:`.MeasureBoxes.additional_components` of ``self`` have an overlap with the
        ``measure_components`` of ``other`` (or vice versa). NOTE: the ``additional_components`` of ``self`` and
        ``other`` may still overlap.

        Args:
            other: MeasureBoxes instance to multiplex with ``self``.

        Returns:
            The multiplexed ``MeasureBoxes`` instance.

        Raises:
            MeasureMultiplexingError: If :attr:`.MeasureBoxes.measure_components` of ``self`` and ``other`` are
                overlapping or if :attr:`.MeasureBoxes.additional_components` of ``self`` overlaps with the
                :attr:`.MeasureBoxes.measure_components` of ``other`` (or vice versa). Note that
                :attr:`.MeasureBoxes.additional_components` may still overlap with that of ``other``.

        """
        if not self.is_multiplexable_with(other):
            raise MeasureMultiplexingError(
                "Multiplexing not supported between two MeasureBoxes with overlapping measure components or if "
                "additional components overlap with measure components."
            )
        # add the first aux box in both
        boxes: list[MeasureBox] = [self[0].multiplex(other[0])]

        if len(self) >= len(other):
            longer = self
            shorter = other
        else:
            longer = other
            shorter = self

        # multiplex the overlapping parts together
        for probe_idx in range(int((len(shorter) - 1) / 3)):
            list_idx = probe_idx * 3 + 1
            boxes.append(self[list_idx].multiplex(other[list_idx]))
            # calculate the new probe wait box durations
            self_duration = sum(self.probe_durations[probe_idx])
            other_duration = sum(other.probe_durations[probe_idx])
            block_duration = max(self_duration, other_duration) - max(
                self.probe_durations[probe_idx][0], other.probe_durations[probe_idx][0]
            )
            if block_duration > 0:
                self_block_channels = self[list_idx + 1].atom.channels() if self[list_idx + 1].atom else []
                other_block_channels = other[list_idx + 1].atom.channels() if other[list_idx + 1].atom else []
                block_channels = [*self_block_channels, *other_block_channels]
                block_box = MeasureBox.atomic(
                    Schedule({ch: Segment([Block(block_duration)]) for ch in block_channels}),
                    locus_components=set(self[list_idx + 1].locus_components).union(
                        other[list_idx + 1].locus_components
                    ),
                    label="Virtual probe channel Block",
                )
            else:
                block_box = MeasureBox.composite([])
            if (self_nb := self[list_idx + 1].neighborhood_components.get(0)) and (
                other_nb := other[list_idx + 1].neighborhood_components.get(0)
            ):
                block_box.neighborhood_components[0] = self_nb.union(other_nb)
            boxes.append(block_box)
            # normal TimeBox add for the aux boxes
            boxes.append(self[list_idx + 2].multiplex(other[list_idx + 2]))

        # copy the tail from the longer one
        boxes += longer[len(shorter) :].copy()
        return MeasureBoxes(
            boxes,
            measure_components=self.measure_components.union(other.measure_components),
            additional_components=self.additional_components.union(other.additional_components),
        )

    @staticmethod
    def from_readout_trigger(
        readout_trigger: ReadoutTrigger,
        probe_channel: str,
        measure_components: Iterable[str],
        *,
        additional_components: Iterable[str] | None = None,
        extra_probe_block_duration: int = 0,
        probe_box_segments: dict[str, Segment] | None = None,
    ) -> MeasureBoxes:
        """Shortcut method for creating a ``MeasureBoxes`` instance from ``ReadoutTrigger instance``.

        The created ``MeasureBoxes`` instance will have a single probe box ``P`` with an optional extra probe block
        box ``B`` superseding it. The auxiliary box slots are occupied with empty boxes.

        Args:
            readout_trigger: ``ReadoutTrigger`` instance.
            probe_channel: channel name of the probe.
            measure_components: The `measure_components` for the created ``MeasureBoxes`` instance.
            additional_components: The `additional_components` for the created ``MeasureBoxes`` instance (by default
                initialized as an empty set).
            extra_probe_block_duration: The duration of the extra probe block, in seconds. If not given, the extra
                probe block box will be empty.
            probe_box_segments: Optional contents for the drive/flux channels of the probe box.

        Returns:
            ``MeasureBoxes`` instance with a single probe box.

        """
        probe_locus_components = set(measure_components).union({probe_channel.split("__")[0]})
        probe_box = MultiplexedProbeTimeBox.from_readout_trigger(
            readout_trigger,
            probe_channel,
            locus_components=probe_locus_components,
            label=f"MeasureBoxes on {probe_locus_components}",
        )
        if extra_probe_block_duration > 0:
            probe_block_box = MeasureBox.atomic(
                Schedule({probe_channel: Segment([Block(extra_probe_block_duration)])}),
                locus_components=probe_channel.split("__")[0],
                label="Virtual probe channel Block",
            )
        else:
            probe_block_box = MeasureBox.composite([])
        if probe_box_segments:
            for channel, segment in probe_box_segments.items():
                probe_box.atom[channel] = segment  # type: ignore[index]
        return MeasureBoxes(
            [MeasureBox.composite([]), probe_box, probe_block_box, MeasureBox.composite([])],
            measure_components=measure_components,
            additional_components=additional_components,
        )

    def __or__(self, other: TimeBox | list[TimeBox]) -> TimeBox:
        if isinstance(other, list):
            return TimeBox.composite(self) | TimeBox.composite(other)
        if isinstance(other, TimeBox):
            return other.__ror__(self)
        raise ValueError(
            f"Nonsupported __or__ operation between MeasureBoxes and an object of the type {other.__class__.__name__}"
        )

    def __ror__(self, other: TimeBox | list[TimeBox]) -> TimeBox:
        if isinstance(other, list):
            return TimeBox.composite(other) | TimeBox.composite(self)
        if isinstance(other, TimeBox):
            return other | self
        raise ValueError(
            "Nonsupported __or__ operation between an object of the type {other.__class__.__name__} "
            "and a MeasureBoxes instance."
        )

    def append(self, __object: object) -> None:
        raise NotImplementedError("Appending to MeasureBoxes is not supported.")

    def extend(self, other: object) -> None:
        raise NotImplementedError("Extending MeasureBoxes is not supported.")

    def __deepcopy__(self, memo: object) -> MeasureBoxes:
        return MeasureBoxes(list(self), measure_components=self.measure_components)
