# Copyright 2022-2026 IQM
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

"""Cost models for ranking candidate layouts.

Costs are error budgets in nats: sums of ``-log(fidelity)`` rather than products of fidelities.
A routed circuit with a few thousand CZ gates has a fidelity product around ``1e-6``, so
``1 - product`` rounds every candidate to 100% even when they differ by orders of magnitude, while
the sums still tell them apart. Rank on :attr:`LayoutCost.total`; :attr:`LayoutCost.error` is for
display.

A new cost model subclasses :class:`CostCalculatorBase` and implements
:meth:`~CostCalculatorBase.score_single_layout`.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from typing import Generic, TypeVar

from iqm.qubit_selector.core.enumerator import Layout
from iqm.qubit_selector.core.ir import InteractionGraph
from iqm.qubit_selector.core.metrics_data import MetricsData, ReadoutMode, two_qubit_key

_PERFECT_FIDELITY = 1.0


@dataclass(frozen=True)
class LayoutCost:
    """A layout and its total error budget.

    Ranking only needs this, so a cost model can return it as is or subclass it to add a breakdown,
    as :class:`LayoutScore` does.
    """

    layout: Layout
    """The scored layout."""

    total: float
    """Error budget in nats, ``-log`` of the probability that the circuit runs without error.
    Layouts are ranked by it, lowest first."""

    @property
    def error(self) -> float:
        """``1 - exp(-total)``, for display only.

        It saturates at 1.0 for deep circuits, so rank on :attr:`total` instead.
        """
        return 1.0 if math.isinf(self.total) else 1.0 - math.exp(-self.total)

    def __str__(self) -> str:
        """Return a short human-readable summary of the layout and its error."""
        return f"{self.layout}: {self.error:.4%} error ({self.total:.4f} nats)"


@dataclass(frozen=True)
class LayoutScore(LayoutCost):
    """A :class:`LayoutCost` split into the terms :class:`FastEdgeCost` scores.

    The terms are in nats and add up to :attr:`~LayoutCost.total`. A layout ``d`` nats lower than
    another is ``exp(d)`` times more likely to run the circuit without error.
    """

    two_qubit: float = 0.0
    """Two-qubit gate contribution, in nats."""

    single_qubit: float = 0.0
    """Single-qubit gate contribution, in nats."""

    readout: float = 0.0
    """Readout contribution, in nats. Zero when ``readout_mode`` is ``NONE``."""


TScore = TypeVar("TScore", bound=LayoutCost)


class CostCalculatorBase(ABC, Generic[TScore]):
    """Base class of the cost models: subclasses score one layout, this class ranks many.

    Generic in the result type, so :meth:`sort_by_scores` returns the subclass's own result, e.g.
    :class:`LayoutScore` for :class:`FastEdgeCost`.
    """

    @abstractmethod
    def score_single_layout(
        self,
        layout: Layout,
        interaction_graph: InteractionGraph,
        sqg_counts: dict[str, int],
        measured: set[str],
        logical_qubits: list[str],
    ) -> TScore:
        """Return the cost of one layout, where ``logical_qubits[i]`` is placed on ``layout[i]``.

        Args:
            layout: The candidate layout (physical qubits).
            interaction_graph: Circuit's 2-qubit interaction graph (logical qubits).
            sqg_counts: Single-qubit gate counts per logical qubit.
            measured: Set of logical qubit names that are measured.
            logical_qubits: Ordered logical qubit names matching layout indices.

        Returns:
            The layout's cost result.

        """

    def sort_by_scores(
        self,
        layouts: list[Layout],
        interaction_graph: InteractionGraph,
        sqg_counts: dict[str, int],
        measured: set[str],
        logical_qubits: list[str],
    ) -> list[TScore]:
        """Score each layout and return the results, lowest error budget first.

        Args:
            layouts: Candidate layouts to evaluate.
            interaction_graph: Circuit's 2-qubit interaction graph (logical qubits).
            sqg_counts: Single-qubit gate counts per logical qubit.
            measured: Set of logical qubit names that are measured.
            logical_qubits: Ordered logical qubit names matching layout indices.

        Returns:
            One result per layout, lowest :attr:`~LayoutCost.total` first.

        """
        results = [
            self.score_single_layout(layout, interaction_graph, sqg_counts, measured, logical_qubits)
            for layout in layouts
        ]
        results.sort(key=lambda result: result.total)
        return results

    def evaluate(
        self,
        layouts: list[Layout],
        interaction_graph: InteractionGraph,
        sqg_counts: dict[str, int],
        measured: set[str],
        logical_qubits: list[str],
    ) -> list[tuple[Layout, float]]:
        """Return ``(layout, error)`` pairs, lowest error first.

        The errors saturate at 1.0 for deep circuits; use :meth:`sort_by_scores` there.

        Args:
            layouts: Candidate layouts to evaluate.
            interaction_graph: Circuit's 2-qubit interaction graph (logical qubits).
            sqg_counts: Single-qubit gate counts per logical qubit.
            measured: Set of logical qubit names that are measured.
            logical_qubits: Ordered logical qubit names matching layout indices.

        Returns:
            List of (layout, error) tuples, sorted by error ascending.

        """
        return [
            (result.layout, result.error)
            for result in self.sort_by_scores(layouts, interaction_graph, sqg_counts, measured, logical_qubits)
        ]


class FastEdgeCost(CostCalculatorBase[LayoutScore]):
    """Score layouts from QPU metrics alone, without transpiling the circuit.

    Layouts from the enumerators put every 2-qubit gate on a coupled pair, so the circuit needs no
    routing and its gates can be scored as they are. The error budget of a layout is::

        total = sum(count * -log(two_qubit_fidelity[edge]))
              + sum(count * -log(single_qubit_fidelity[qubit]))
              + sum(-log(readout_fidelity[qubit]))     # only when readout_mode is not NONE

    A missing metric adds infinite error, so a layout on an uncalibrated qubit or pair never wins
    instead of being scored as perfect. A fidelity outside ``(0, 1]`` raises ``ValueError``: it means
    the calibration data is corrupt, and no ranking based on it can be trusted.

    Args:
        metrics: QPU metrics used for scoring.
        readout_mode: Which readout metric to include, if any.

    """

    def __init__(
        self,
        metrics: MetricsData,
        readout_mode: ReadoutMode = ReadoutMode.NONE,
    ):
        self.metrics = metrics
        self.readout_mode = readout_mode

    def score_single_layout(
        self,
        layout: Layout,
        interaction_graph: InteractionGraph,
        sqg_counts: dict[str, int],
        measured: set[str],
        logical_qubits: list[str],
    ) -> LayoutScore:
        """Return the error budget of one layout in nats, split by term."""
        to_physical = dict(zip(logical_qubits, layout, strict=True))
        two_qubit = self._two_qubit_term(interaction_graph, to_physical)
        single_qubit = self._single_qubit_term(sqg_counts, to_physical)
        readout = self._readout_term(measured, to_physical, self._get_readout_fidelities())
        return LayoutScore(
            layout=layout,
            total=two_qubit + single_qubit + readout,
            two_qubit=two_qubit,
            single_qubit=single_qubit,
            readout=readout,
        )

    def _two_qubit_term(self, interaction_graph: InteractionGraph, to_physical: dict[str, str]) -> float:
        """Return the two-qubit error budget of one layout, in nats."""
        fidelities = self.metrics.two_qubit
        total = 0.0
        for edge, count in interaction_graph.items():
            logical_a, logical_b = edge
            physical_a = to_physical[logical_a]
            physical_b = to_physical[logical_b]
            key = two_qubit_key(physical_a, physical_b)
            fidelity = fidelities.get(key)
            if fidelity is None:
                fidelity = fidelities.get(two_qubit_key(physical_b, physical_a))
            # A pair the QPU has no CZ metric for cannot host this interaction; exclude the
            # layout with infinite error rather than pretend the gate is free.
            if fidelity is None:
                total += math.inf
            else:
                total += count * self._neg_log(fidelity, key)
        return total

    def _single_qubit_term(self, sqg_counts: dict[str, int], to_physical: dict[str, str]) -> float:
        """Return the single-qubit error budget of one layout, in nats."""
        fidelities = self.metrics.single_qubit
        total = 0.0
        for logical_qubit, count in sqg_counts.items():
            physical_qubit = to_physical[logical_qubit]
            fidelity = fidelities.get(physical_qubit)
            if fidelity is None:
                total += math.inf
            else:
                total += count * self._neg_log(fidelity, physical_qubit)
        return total

    def _readout_term(
        self,
        measured: set[str],
        to_physical: dict[str, str],
        fidelities: dict[str, float] | None,
    ) -> float:
        """Return the readout error budget of one layout, in nats.

        ``fidelities`` is ``None`` when readout is not scored. A measured qubit missing from it adds
        infinite error, as in the other terms.
        """
        if fidelities is None:
            return 0.0
        total = 0.0
        for logical_qubit in measured:
            physical_qubit = to_physical[logical_qubit]
            fidelity = fidelities.get(physical_qubit)
            if fidelity is None:
                total += math.inf
            else:
                total += self._neg_log(fidelity, physical_qubit)
        return total

    def _neg_log(self, fidelity: float, key: str) -> float:
        """Return ``-log(fidelity)`` in nats.

        Raises:
            ValueError: If ``fidelity`` is not in ``(0, 1]``, which means the calibration data is
                corrupt.

        """
        if math.isnan(fidelity) or fidelity > _PERFECT_FIDELITY or fidelity <= 0.0:
            raise ValueError(f"Fidelity metric for {key} is {fidelity}, outside the valid range (0, 1].")
        return -math.log(fidelity)

    def _get_readout_fidelities(self) -> dict[str, float] | None:
        """Return the readout metric selected by ``readout_mode``, or ``None`` for ``ReadoutMode.NONE``.

        Raises:
            ValueError: If the metrics hold no entry for the selected mode. Scoring without readout
                instead would hide the configuration error.

        """
        mode = self.readout_mode
        if mode is ReadoutMode.NONE:
            return None

        fidelities = self.metrics.readout.get(mode)
        if fidelities is None:
            # Also match by value, so a same-named enum from another module still works (e.g. an
            # %autoreload-reloaded ReadoutMode). This still selects the same logical mode, never a
            # different one.
            value = getattr(mode, "value", mode)
            for known_mode, known_fidelities in self.metrics.readout.items():
                if known_mode.value == value:
                    fidelities = known_fidelities
                    break

        if not fidelities:
            raise ValueError(f"Readout mode {mode!r} requested but 'MetricsData.readout' has no entry for it.")
        return fidelities
