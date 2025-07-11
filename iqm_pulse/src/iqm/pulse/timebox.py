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
from dataclasses import dataclass, field
import enum
from functools import reduce

from iqm.pulse.playlist.instructions import Block, ReadoutTrigger
from iqm.pulse.playlist.schedule import Schedule


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

    * An atomic box only contains a single :class:`.Schedule`.

    * A composite box contains a sequence of other TimeBoxes as its children.
      The locus components are the union of the locus components of the children.
      If two children use the same channel so that they cannot happen simultaneously, they must
      happen in the order they occur in the sequence.

    A box can be made atomic by *resolving* it using :class:`.ScheduleBuilder.resolve_timebox`.
    The time duration of the box is determined by its contents and the way they are scheduled during the resolution.

    TimeBoxes can be concatenated with the following rules:

    * Addition concatenates the children of the operands into a single TimeBox.

    * The pipe operation groups two TimeBoxes together without concatenating.

    * Iterables of Boxes are treated as the sum of the elements.

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
     e.g. the `GateImplementation`) and caches them under this attribute. Neighborhood range 0 means just the components
     affected by one of the channels in ``self.atom`` + ``self.locus``, 1 means also neighboring couplers, 2 the
     components connected to those couplers, and so on. Note: range 0 may differ from ``self.locus_components``: it can
     have additional components that have occupied channels in ``self`` but are not defined as a part of the 'locus' of
     this ``TimeBox`` for any reason.
    """

    @staticmethod
    def composite(
        boxes: Iterable[TimeBox | Iterable[TimeBox]],
        *,
        label: str = "",
        scheduling: SchedulingStrategy = SchedulingStrategy.ASAP,
        scheduling_algorithm: SchedulingAlgorithm = SchedulingAlgorithm.HARD_BOUNDARY,
    ) -> TimeBox:
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
        for child in boxes:
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
        return TimeBox(
            label=label,
            locus_components=locus_components,
            atom=None,
            children=tuple(children),
            scheduling=scheduling,
            scheduling_algorithm=scheduling_algorithm,
        )

    @staticmethod
    def atomic(schedule: Schedule, *, locus_components: Iterable[str], label: str) -> TimeBox:
        """Build an atomic timebox from a schedule.

        Args:
            schedule: contents of the new timebox
            locus_components: names QPU components ``schedule`` operates on
            label: label of the new timebox

        Returns:
            atomic timebox containing ``schedule``

        """
        return TimeBox(label=label, locus_components=set(locus_components), atom=schedule, children=())

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

    def __add__(self, other: TimeBox | Iterable[TimeBox]) -> TimeBox:
        """Return a new TimeBox which has the contents of this and another TimeBox concatenated.

        Used to concatenate multiple TimeBoxes, like atomic operations, to a single logical entity.

        The add operation is associative: for boxes ``a, b, c``, these are equivalent:
        ``a+b+c == (a+b)+c == a+(b+c) == a+[b,c] == [a,b]+c``.

        The scheduling strategy and label are given by `self`, i.e. the leftmost operand.

        Args:
             other: TimeBox or an iterable of TimeBoxes whose contents to merge.

        Returns:
            A new instance containing the children of both boxes.

        """
        if isinstance(other, TimeBox):
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
        try:
            return reduce(lambda x, y: x + y, other, self)
        except TypeError as err:
            raise TypeError(f"Cannot add a TimeBox and a {type(other)}.") from err

    def __radd__(self, other: Iterable[TimeBox]) -> TimeBox:
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
        """Construct a new composite TimeBox that contains `self` and `other`.

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
            A new TimeBox containing self and `other` as children.

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


class MultiplexedProbeTimeBox(TimeBox):
    """A ``TimeBox`` that contains any number of multiplexed readout pulses for probe channels.

    A ``MultiplexedProbeTimeBox``'s atom contains exactly one ``ReadoutTrigger`` for each probe channel.
    """

    def __add__(self, other: TimeBox | Iterable[TimeBox]) -> TimeBox:
        """Override ``__add__`` for two atomic ``MultiplexedProbeTimeBox`` instances such that ``ReadoutTrigger``s
        belonging to the same probe channel are multiplexed together. Otherwise, behaves exactly like
        ``TimeBox.__add__``, returning a normal ``TimeBox``.
        """
        if isinstance(other, MultiplexedProbeTimeBox) and self.atom is not None and other.atom is not None:
            new_segments = dict(self.atom.copy().items())
            for channel, segment in other.atom.items():
                if channel not in new_segments:
                    new_segments[channel] = segment
                elif isinstance(segment[0], ReadoutTrigger) and isinstance(new_segments[channel][0], ReadoutTrigger):
                    # multiplex the readout triggers together
                    new_segments[channel]._instructions[0] = new_segments[channel][0] + segment[0]
                else:
                    new_segments[channel].extend(iter(segment))
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
            return MultiplexedProbeTimeBox(
                label=f"MultiplexedProbeTimeBox on {locus_components}",
                locus_components=locus_components,
                atom=Schedule(new_segments),
                children=(),
                scheduling=self.scheduling,
                scheduling_algorithm=self.scheduling_algorithm,
                neighborhood_components=neighborhood_components,
            )
        return super().__add__(other)

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
        """Build an atomic ``MultiplexedProbeTimeBox` from a single ``ReadoutTrigger`` instruction.

        Args:
            readout_trigger: Readout trigger instruction.
            probe_channel: Name of the probe channel to play ``readout_trigger`` in.
            locus_components: Locus components.
            label: Label of the new timebox.
            block_channels: Names of channels to block.
            block_duration: Duration of the required blocking (in samples).

        Returns:
            atomic timebox containing ``readout_trigger`` in the channel ``probe_channel``.

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
