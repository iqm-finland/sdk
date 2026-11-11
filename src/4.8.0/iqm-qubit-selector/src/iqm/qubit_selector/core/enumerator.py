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

"""Layout enumeration strategies for qubit selection."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
import itertools
import logging
import math
from typing import TypeAlias

from iqm.qubit_selector.core.ir import InteractionGraph
import networkx as nx
import numpy as np
import rustworkx as rx

logger = logging.getLogger(__name__)

Layout: TypeAlias = list[str]
"""A layout is a list of physical qubits, one per logical qubit."""

_MIN_COMPLETE_GRAPH_NODES = 2


def is_all_to_all(graph: nx.Graph) -> bool:
    """Return whether every pair of nodes in ``graph`` is coupled.

    A single-resonator Star QPU collapses to such a graph and VF2 is the
    wrong tool there; see :func:`make_enumerator`.

    Args:
        graph: Coupling graph, assumed simple (no self-loops or parallel edges).

    Returns:
        ``True`` if the graph is complete and has at least two nodes.

    """
    num_nodes = graph.number_of_nodes()
    if num_nodes < _MIN_COMPLETE_GRAPH_NODES:
        return False
    return graph.number_of_edges() == num_nodes * (num_nodes - 1) // 2


class BaseEnumerator(ABC):
    """Interface of the layout enumerators; :func:`make_enumerator` picks one for a coupling graph.

    Args:
        node_labels: Names of the QPU's qubits.
        max_num_layouts: Maximum number of layouts to return.
        seed: Seed for the enumerator's random choices; ``None`` for a fresh draw on every call.

    """

    def __init__(
        self,
        node_labels: list[str],
        max_num_layouts: int = 2000,
        seed: int | None = 0,
    ):
        self.node_labels = node_labels
        self.max_num_layouts = max_num_layouts
        self.seed = seed

    @abstractmethod
    def enumerate(self, interaction_graph: InteractionGraph, num_circuit_qubits: int) -> list[Layout]:
        """Return up to ``max_num_layouts`` candidate layouts, one per distinct qubit set.

        Layout positions follow ``sorted(core) + sorted(extras)``, where the core qubits are those with
        2-qubit gates. The selector relies on this order to read each qubit's metrics from the right
        physical qubit.

        Args:
            interaction_graph: 2-qubit interaction graph over logical qubit names.
            num_circuit_qubits: Number of qubits in the circuit, including single-qubit-only ones.

        Returns:
            Layouts of physical qubit names, each of length ``num_circuit_qubits``.

        Raises:
            ValueError: If the circuit needs no qubits, more qubits than the QPU has, or fewer
                qubits than its interaction graph involves.

        """

    def _split_qubits(self, interaction_graph: InteractionGraph, num_circuit_qubits: int) -> tuple[list[str], int]:
        """Validate a request; return ``(sorted core qubits, number of single-qubit-only qubits)``."""
        num_qpu_qubits = len(self.node_labels)
        if num_circuit_qubits < 1:
            raise ValueError(
                f"The circuit must act on at least one qubit, got 'num_circuit_qubits'={num_circuit_qubits}."
            )
        if num_circuit_qubits > num_qpu_qubits:
            raise ValueError(f"The circuit needs {num_circuit_qubits} qubits, but the QPU has only {num_qpu_qubits}.")
        core_qubits = sorted({qubit for edge in interaction_graph for qubit in edge})
        if num_circuit_qubits < len(core_qubits):
            raise ValueError(
                f"'num_circuit_qubits' is {num_circuit_qubits}, but the interaction graph alone "
                f"involves {len(core_qubits)} qubits."
            )
        return core_qubits, num_circuit_qubits - len(core_qubits)


class AllToAllEnumerator(BaseEnumerator):
    """Enumerate layouts on a complete coupling graph.

    Every ``k``-subset of the QPU's qubits is a valid layout here, so VF2 would prune nothing; the subsets
    are generated directly instead.

    Only one assignment of logical to physical qubits is tried per set. Trying all ``k!`` would use up
    the budget on reshuffling a few sets, and on a randomised 20-qubit star the best assignment beat
    the canonical one by only about 0.02 nats (under 2% relative error). :class:`VF2Enumerator` makes
    the same trade-off.

    When ``C(N, k)`` exceeds ``max_num_layouts``, a seeded uniform sample of the subsets is returned;
    taking them in label order would never reach the high-numbered qubits.

    Args:
        node_labels: Names of the QPU's qubits.
        max_num_layouts: Maximum number of layouts to return.
        seed: Seed for the sample; ``None`` for a fresh draw on every call.

    """

    def enumerate(self, interaction_graph: InteractionGraph, num_circuit_qubits: int) -> list[Layout]:
        """Return up to ``max_num_layouts`` qubit sets of size ``num_circuit_qubits``.

        The interaction graph is only validated: on a complete graph any qubit set can host it.

        Args:
            interaction_graph: 2-qubit interaction graph over logical qubit names.
            num_circuit_qubits: Number of qubits in the circuit.

        Returns:
            Layouts in ``node_labels`` order.

        Raises:
            ValueError: If the circuit needs no qubits, more qubits than the QPU has, or fewer
                qubits than its interaction graph involves.

        """
        self._split_qubits(interaction_graph, num_circuit_qubits)
        num_qubits = len(self.node_labels)

        total = math.comb(num_qubits, num_circuit_qubits)
        if total <= self.max_num_layouts:
            layouts = [
                list(combination) for combination in itertools.combinations(self.node_labels, num_circuit_qubits)
            ]
            logger.info("'AllToAllEnumerator' found %d layouts (exhaustive)", len(layouts))
            return layouts

        rng = np.random.default_rng(self.seed)
        combinations = sorted(
            itertools.islice(_random_combinations(rng, num_qubits, num_circuit_qubits), self.max_num_layouts)
        )
        layouts = [[self.node_labels[i] for i in combination] for combination in combinations]
        logger.warning(
            "All-to-all coupling graph admits %d qubit sets of size %d; evaluating a random sample of %d "
            "(max_num_layouts=%d). Increase 'max_num_layouts' to widen the search.",
            total,
            num_circuit_qubits,
            len(layouts),
            self.max_num_layouts,
        )
        return layouts


def _random_combinations(rng: np.random.Generator, n: int, k: int) -> Iterator[tuple[int, ...]]:
    """Yield every ``k``-subset of ``range(n)`` once, lazily, in uniformly random order.

    Callers usually stop after a small fraction of the ``C(n, k)`` subsets, so they are drawn by
    rejection sampling with nothing precomputed. Rejection slows down once half have been drawn, so
    the rest are then listed and shuffled; that list is never longer than what was already yielded.

    Args:
        rng: Random number generator to draw from.
        n: Size of the ground set.
        k: Size of each subset.

    Yields:
        The subsets as ascending index tuples.

    """
    total = math.comb(n, k)
    drawn: set[tuple[int, ...]] = set()
    while 2 * len(drawn) < total:
        subset = tuple(sorted(int(i) for i in rng.choice(n, size=k, replace=False)))
        if subset not in drawn:
            drawn.add(subset)
            yield subset
    remaining = [subset for subset in itertools.combinations(range(n), k) if subset not in drawn]
    for index in rng.permutation(len(remaining)):
        yield remaining[index]


_MAX_CONSECUTIVE_DUPLICATES = 64
"""Consecutive duplicate qubit sets after which :func:`_round_robin` drops a source.

Restarted VF2 searches drift into regions other starts already covered, and symmetric patterns map
onto one qubit set in many ways (24 for a five-qubit star), so a source can go on yielding only
duplicates for a long time.
"""

_MIN_VF2_STATES_PER_CORE_QUBIT = 4
"""Floor on each VF2 search's state limit, per core qubit.

A match needs at least one state per core qubit, so with ``call_limit=max_num_layouts`` alone a
circuit larger than the budget would find no layout at all.
"""


def _round_robin(sources: list[Iterator[Layout]], max_num_layouts: int) -> tuple[list[Layout], bool]:
    """Collect up to ``max_num_layouts`` distinct layouts from ``sources``, which take turns.

    Taking turns, rather than draining one source after another, is what spreads the candidates over
    the QPU. The per-round quota starts at an even share of the budget and doubles, so sources that
    run dry drop out early and the remaining budget goes to those that still find new sets.

    Args:
        sources: Lazy layout iterators.
        max_num_layouts: Maximum number of layouts to return.

    Returns:
        ``(layouts, truncated)``; ``truncated`` is ``True`` if the budget ran out first.

    """
    seen: set[frozenset[str]] = set()
    layouts: list[Layout] = []
    quota = max(1, math.ceil(max_num_layouts / max(1, len(sources))))
    while sources:
        live: list[Iterator[Layout]] = []
        for source in sources:
            found = 0
            duplicates = 0
            for layout in source:
                qubits = frozenset(layout)
                if qubits in seen:
                    duplicates += 1
                    if duplicates >= _MAX_CONSECUTIVE_DUPLICATES:
                        break
                    continue
                duplicates = 0
                seen.add(qubits)
                layouts.append(layout)
                found += 1
                if len(layouts) >= max_num_layouts:
                    return layouts, True
                if found >= quota:
                    live.append(source)
                    break
        sources = live
        quota *= 2
    return layouts, False


class VF2Enumerator(BaseEnumerator):
    """Enumerate layouts with rustworkx VF2 subgraph monomorphism.

    VF2 tries the QPU's qubits in index order, so a single capped search returns layouts clustered in one
    corner of the chip. Each search here therefore runs on a copy of the coupling graph renumbered by distance
    from a different start qubit, and the searches take turns (see :func:`_round_robin`), which gives
    every qubit the same share of the budget as a starting point.

    Each search is capped at ``max_num_layouts`` VF2 states. rustworkx reports a capped search the same
    way as an exhausted one, so matches can be missed without a truncation warning.

    Args:
        coupling_graph: QPU coupling graph; nodes are qubit names.
        max_num_layouts: Maximum number of layouts to return, and the VF2 state limit of each search.
            With fewer than the number of QPU qubits, some qubits never serve as a start; 10-15
            times that number gave full coverage on a 150-qubit lattice.
        seed: Seed for the start order and the placement of single-qubit-only qubits; ``None`` for a
            fresh draw on every call.

    """

    def __init__(
        self,
        coupling_graph: nx.Graph,
        max_num_layouts: int = 2000,
        seed: int | None = 0,
    ):
        super().__init__(list(coupling_graph.nodes()), max_num_layouts, seed)
        self.coupling_graph = coupling_graph

        index_of = {label: index for index, label in enumerate(self.node_labels)}
        self._qpu_graph: rx.PyGraph = rx.PyGraph()
        self._qpu_graph.add_nodes_from(range(len(self.node_labels)))
        self._qpu_graph.add_edges_from_no_data([(index_of[u], index_of[v]) for u, v in coupling_graph.edges()])
        self._edges = np.array(self._qpu_graph.edge_list(), dtype=np.int64).reshape(-1, 2)
        # Hop distance between every pair of qubits, used to renumber the coupling graph from each start
        self._distances = rx.distance_matrix(self._qpu_graph, null_value=np.inf)
        np.fill_diagonal(self._distances, 0)

    def enumerate(self, interaction_graph: InteractionGraph, num_circuit_qubits: int) -> list[Layout]:
        """Return up to ``max_num_layouts`` layouts that can host ``interaction_graph``.

        VF2 matches only the core qubits. Single-qubit-only qubits need no coupling, so they are placed
        on unused qubits afterwards; this keeps the VF2 pattern small.

        The interaction graph may be disconnected, e.g. ``CZ(QB1, QB2)`` and ``CZ(QB3, QB4)``. Its
        components may then land on non-adjacent parts of the QPU, which is a valid layout.

        Args:
            interaction_graph: 2-qubit interaction graph over logical qubit names.
            num_circuit_qubits: Number of qubits in the circuit, including single-qubit-only ones.

        Returns:
            Layouts of physical qubit names, in the order :meth:`BaseEnumerator.enumerate` describes.

        Raises:
            ValueError: If the circuit needs no qubits, more qubits than the QPU has, or fewer
                qubits than its interaction graph involves.

        """
        core_qubits, num_extra = self._split_qubits(interaction_graph, num_circuit_qubits)
        rng = np.random.default_rng(self.seed)

        cores: list[Layout] = [[]]
        truncated = False
        if core_qubits:
            core_index = {qubit: index for index, qubit in enumerate(core_qubits)}
            im_graph: rx.PyGraph = rx.PyGraph()
            im_graph.add_nodes_from(range(len(core_qubits)))
            # The graph is undirected, so the endpoint order of each pair does not matter
            im_graph.add_edges_from_no_data([(core_index[q0], core_index[q1]) for q0, q1 in interaction_graph])
            starts = rng.permutation(len(self.node_labels))
            cores, truncated = _round_robin(
                [self._cores_from(int(start), im_graph) for start in starts], self.max_num_layouts
            )

        layouts = cores
        if num_extra:
            layouts, extras_truncated = _round_robin(
                [self._extensions(core, num_extra, rng) for core in cores], self.max_num_layouts
            )
            truncated = truncated or extras_truncated

        if truncated:
            logger.warning(
                "'VF2Enumerator' hit the candidate cap (max_num_layouts=%d); evaluating %d layouts spread "
                "over the QPU. Increase 'max_num_layouts' to widen the search.",
                self.max_num_layouts,
                len(layouts),
            )
        else:
            logger.info("'VF2Enumerator' found %d unique layouts", len(layouts))
        return layouts

    def _cores_from(self, start: int, im_graph: rx.PyGraph) -> Iterator[Layout]:
        """Yield core layouts from a VF2 search whose first matches lie at qubit ``start``.

        VF2 tries qubits in index order, so renumbering the coupling graph by hop distance from ``start``
        makes the search begin there and grow outwards. ``induced=False`` because extra couplings among
        the chosen qubits do no harm.

        Args:
            start: Index of the qubit the search begins at.
            im_graph: Core interaction graph, one node per core qubit in sorted logical order.

        Yields:
            Physical names of the core qubits, in sorted logical order.

        """
        order = np.argsort(self._distances[start], kind="stable")
        position = np.empty_like(order)
        position[order] = np.arange(len(order))
        renumbered: rx.PyGraph = rx.PyGraph()
        renumbered.add_nodes_from(range(len(order)))
        renumbered.add_edges_from_no_data([(u, v) for u, v in position[self._edges].tolist()])
        num_core = im_graph.num_nodes()
        labels = [self.node_labels[index] for index in order]
        for mapping in rx.vf2_mapping(
            renumbered,
            im_graph,
            subgraph=True,
            id_order=False,
            induced=False,
            call_limit=max(self.max_num_layouts, _MIN_VF2_STATES_PER_CORE_QUBIT * num_core),
        ):
            core = [""] * num_core
            for qubit, logical in mapping.items():
                core[logical] = labels[qubit]
            yield core

    def _extensions(self, core: Layout, num_extra: int, rng: np.random.Generator) -> Iterator[Layout]:
        """Yield ``core`` followed by ``num_extra`` unused qubits, in uniformly random order.

        Args:
            core: Physical names of the core qubits.
            num_extra: Number of single-qubit-only qubits to place.
            rng: Random number generator for the placement order.

        Yields:
            The core followed by the physical names of the extra qubits.

        """
        used = set(core)
        unused = [label for label in self.node_labels if label not in used]
        for extra in _random_combinations(rng, len(unused), num_extra):
            yield core + [unused[i] for i in extra]


def make_enumerator(
    coupling_graph: nx.Graph,
    max_num_layouts: int = 2000,
    seed: int | None = 0,
) -> BaseEnumerator:
    """Return the enumerator for ``coupling_graph``.

    A complete graph, such as a collapsed single-resonator Star, gets :class:`AllToAllEnumerator`:
    VF2 would prune nothing there and spend its budget finding the same sets again. Any other graph
    gets :class:`VF2Enumerator`.

    Args:
        coupling_graph: QPU coupling graph; nodes are qubit names.
        max_num_layouts: Maximum number of layouts the enumerator returns.
        seed: Seed for the enumerator's random choices; ``None`` for a fresh draw on every call.

    Returns:
        An :class:`AllToAllEnumerator` or a :class:`VF2Enumerator`.

    """
    if is_all_to_all(coupling_graph):
        return AllToAllEnumerator(list(coupling_graph.nodes()), max_num_layouts, seed)
    return VF2Enumerator(coupling_graph, max_num_layouts, seed)
