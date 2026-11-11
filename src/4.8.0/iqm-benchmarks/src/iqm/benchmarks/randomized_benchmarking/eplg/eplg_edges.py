# Copyright 2024 IQM Benchmarks developers
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

"""Selection of the qubit pairs (edges) that an EPLG run benchmarks.

A selection is scored by the ``-log`` of its predicted layer fidelity::

    cost = sum over edges of -log(F_edge) + sum over layers of sum over qubits without a gate of no_gate(qubit)

where ``no_gate(qubit) = -log(F_1q) - log(F_idle)`` is the cost of a qubit that only runs its single-qubit Clifford
and idles during the CZs of a layer. A chain runs in 2 layers, while a tree with qubit degrees up to ``d`` needs
``d`` layers, so its qubits idle more often.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
import math
from statistics import median
from typing import Literal

from iqm.benchmarks.benchmark_definition import BenchmarkAnalysisResult
from iqm.benchmarks.logging_config import qcvv_logger
from iqm.benchmarks.utils import _relaxation_fidelity, extract_fidelities_unified
from iqm.qiskit_iqm.iqm_backend import IQMBackendBase
import networkx as nx

Edge = tuple[int, int]
"""A pair of backend qubit indices."""

EdgeSelection = Literal["tree", "chain"]
"""The shapes the edge selection can choose."""

CalibrationMetrics = dict[str, dict[int | Edge, float]]
"""Calibration metrics keyed by metric name, then by backend qubit index or pair of indices."""

DEFAULT_SEARCH_BUDGET = 10**6
"""Default maximum number of path extensions the chain search tries."""

CZ_LAYER_DURATION_US = 0.04
"""Duration of a layer of CZs in microseconds, during which the qubits without a gate idle."""


@dataclass(frozen=True)
class EPLGCandidate:
    """A selection of edges to run EPLG on."""

    edges: list[Edge]
    """The two-qubit gate edges, oriented as in the backend coupling map."""

    num_layers: int
    """Number of layers of disjoint edges the selection runs in."""

    cost: float
    """``-log`` of the predicted layer fidelity; lower is better."""

    label: str
    """``"chain"`` or ``"tree"``."""

    edge_fidelity_source: str
    """What the edges were scored with."""


def calibration_metrics_by_backend_index(backend: IQMBackendBase) -> CalibrationMetrics:
    """Return the calibration metrics of the backend, keyed by backend qubit index.

    Args:
        backend: The backend to read the calibration of.

    Returns:
        The metrics of :func:`~iqm.benchmarks.utils.extract_fidelities_unified`, keyed by backend qubit index
        instead of the calibration's own numbering.

    """
    qubit_mapping, metrics = extract_fidelities_unified(backend)
    if getattr(backend, "error_profile", None) is not None:
        to_backend = {index: int(qubit) for qubit, index in qubit_mapping.items()}
    else:
        to_backend = {
            index: backend.qubit_name_to_index(f"QB{qubit}")
            for qubit, index in qubit_mapping.items()
            if f"QB{qubit}" in backend.physical_qubits
        }

    remapped: CalibrationMetrics = {}
    for name, values in metrics.items():
        remapped[name] = {}
        for key, value in values.items():
            if isinstance(key, tuple):
                if key[0] in to_backend and key[1] in to_backend:
                    remapped[name][(to_backend[key[0]], to_backend[key[1]])] = value
            elif key in to_backend:
                remapped[name][to_backend[key]] = value
    return remapped


def edge_fidelities_from_calibration(
    backend: IQMBackendBase,
    metric: Literal["cz_gate_fidelity", "clifford_fidelity"] = "clifford_fidelity",
    metrics: CalibrationMetrics | None = None,
) -> dict[Edge, float]:
    """Return the calibrated fidelity of every coupler, for ``EPLGConfiguration.edge_fidelities``.

    Args:
        backend: The backend to read the calibration of.
        metric: The calibration metric: 2Q Clifford (default) or CZ fidelity.
        metrics: Calibration metrics already read with :func:`calibration_metrics_by_backend_index`.

    Returns:
        The fidelity of each calibrated coupler, keyed by ascending backend qubit indices.

    """
    metrics = calibration_metrics_by_backend_index(backend) if metrics is None else metrics
    couplers = {tuple(sorted(edge)) for edge in backend.coupling_map.get_edges()}
    fidelities = {}
    for key, value in metrics.get(metric, {}).items():
        if isinstance(key, tuple) and tuple(sorted(key)) in couplers:
            fidelities[(min(key), max(key))] = float(value)
    return fidelities


def edge_fidelities_from_eplg(result: BenchmarkAnalysisResult) -> dict[Edge, float]:
    """Return the layered fidelity of every pair an EPLG run measured, for ``EPLGConfiguration.edge_fidelities``.

    Args:
        result: The analysis result of an EPLG run.

    Returns:
        The layered fidelity of each measured pair, keyed by its backend qubit indices.

    """
    fidelities = {}
    for observation in result.observations:
        qubits = observation.identifier.qubit_indices
        if observation.name == "average_gate_fidelity" and qubits is not None and len(qubits) == 2:  # noqa: PLR2004
            fidelities[(int(qubits[0]), int(qubits[1]))] = float(observation.value)
    return fidelities


def orient_to_coupling_map(backend: IQMBackendBase, pairs: Sequence[Edge]) -> list[Edge]:
    """Flip each pair that is given against the orientation of the backend coupling map.

    Args:
        backend: The backend whose coupling map gives the orientation.
        pairs: The pairs to orient.

    Returns:
        The pairs in coupling map orientation.

    """
    coupling_map = set(backend.coupling_map.get_edges())
    return [(a, b) if (a, b) in coupling_map else (b, a) for a, b in pairs]


def _no_gate_costs(metrics: CalibrationMetrics, qubits: Sequence[int]) -> dict[int, float]:
    """The cost of each qubit in a layer where it has no CZ: its single-qubit Clifford and idling."""
    costs = {}
    for q in qubits:
        f_1q = metrics.get("fidelity_1qb_gates_averaged", {}).get(q, 0.999)
        t1 = metrics.get("t1_time", {}).get(q, 40.0)
        t2 = metrics.get("t2_time", {}).get(q, 30.0)
        # Fall back to typical values if the calibration has a placeholder
        f_1q = f_1q if 0 < f_1q <= 1 else 0.999
        t1 = t1 if t1 > 0 else 40.0
        t2 = t2 if t2 > 0 else 30.0
        costs[q] = -math.log(f_1q) - math.log(_relaxation_fidelity(CZ_LAYER_DURATION_US, t1, t2))
    return costs


def _selection_graph(
    backend: IQMBackendBase,
    metrics: CalibrationMetrics,
    excluded_qubits: Sequence[int],
    edge_fidelities: Mapping[Edge, float] | None,
) -> tuple[nx.Graph, str]:
    """The couplers that can be selected, weighted by ``-log`` of their fidelity, and where the fidelities come from."""
    if edge_fidelities is not None:
        fidelities, source = dict(edge_fidelities), "user-supplied edge fidelities"
    else:
        fidelities = edge_fidelities_from_calibration(backend, "clifford_fidelity", metrics)
        source = "calibration 2Q Clifford fidelities"
        if not fidelities:
            qcvv_logger.warning(
                "EPLG edge selection: the calibration has no 2Q Clifford fidelities; using CZ fidelities."
            )
            fidelities = edge_fidelities_from_calibration(backend, "cz_gate_fidelity", metrics)
            source = "calibration CZ fidelities"
        # Couplers without a usable calibrated fidelity, e.g. on star backends, get the median of the others
        valid = [f for f in fidelities.values() if 0 < f <= 1]
        default = median(valid) if valid else 0.99
        for a, b in backend.coupling_map.get_edges():
            edge = (min(a, b), max(a, b))
            if not 0 < fidelities.get(edge, 0) <= 1:
                fidelities[edge] = default

    graph: nx.Graph = nx.Graph()
    for a, b in backend.coupling_map.get_edges():
        fidelity = fidelities.get((a, b), fidelities.get((b, a)))
        if fidelity is not None and 0 < fidelity <= 1 and a not in excluded_qubits and b not in excluded_qubits:
            graph.add_edge(a, b, weight=-math.log(fidelity))
    return graph, source


def layer_cost(graph: nx.Graph, edges: Sequence[Edge], num_layers: int, no_gate: Mapping[int, float]) -> float:
    """Return the ``-log`` of the predicted layer fidelity of the edges run in ``num_layers`` layers.

    Args:
        graph: The selection graph, with the ``-log`` edge fidelities as ``weight``.
        edges: The selected edges.
        num_layers: The number of layers the edges run in.
        no_gate: The cost of each qubit in a layer where it has no CZ.

    Returns:
        The cost of the selection.

    """
    degree = Counter(q for edge in edges for q in edge)
    edge_cost = sum(graph.edges[edge]["weight"] for edge in edges)
    return edge_cost + sum((num_layers - d) * no_gate[q] for q, d in degree.items())


def _longest_possible_chain(graph: nx.Graph) -> int:
    """An upper bound on the number of qubits a chain can have in the graph.

    On a bipartite graph, such as a square lattice, a chain alternates between the two sides, so it has at most one
    more qubit on one side than on the other.
    """
    longest = 0
    for component in nx.connected_components(graph):
        subgraph = graph.subgraph(component)
        if nx.is_bipartite(subgraph):
            side_a, side_b = nx.bipartite.sets(subgraph)
            longest = max(longest, 2 * min(len(side_a), len(side_b)) + (len(side_a) != len(side_b)))
        else:
            longest = max(longest, len(component))
    return longest


def _best_chains(
    graph: nx.Graph,
    num_qubits: int,
    no_gate: Mapping[int, float],
    num_layers: int,
    num_ranked: int,
    search_budget: int,
) -> list[tuple[float, list[int]]]:
    """The cheapest chains of ``num_qubits`` qubits, from a depth-first search over the paths of the graph.

    Neighbours are tried cheapest first, and a branch is cut when it cannot beat the chains kept so far or cannot
    reach enough unused qubits. The search stops after ``search_budget`` path extensions, shared evenly between the
    start qubits.
    """
    weight = nx.get_edge_attributes(graph, "weight")
    weight.update({(b, a): w for (a, b), w in weight.items()})
    cheapest_edge = min(weight.values(), default=0.0)
    neighbours = {q: sorted(graph[q], key=lambda n: weight[q, n]) for q in graph}
    best: list[tuple[float, list[int]]] = []
    expansions = 0
    step_limit = search_budget
    stopped_early = False

    def worst_kept() -> float:
        """The cost a new chain has to beat to be kept."""
        return best[-1][0] if len(best) == num_ranked else math.inf

    def can_reach(path: set[int], end: int, needed: int) -> bool:
        """Whether at least ``needed`` unused qubits are connected to the end of the path."""
        seen, stack = {end}, [end]
        while stack and len(seen) <= needed:
            for n in graph[stack.pop()]:
                if n not in path and n not in seen:
                    seen.add(n)
                    stack.append(n)
        return len(seen) > needed

    def extend(path: list[int], in_path: set[int], cost: float) -> None:
        """Extend the path by every unused neighbour of its end, keeping the complete chains."""
        nonlocal expansions, stopped_early
        if len(path) == num_qubits:
            # Each chain is found from both ends; keep one of the two
            if path[0] < path[-1]:
                total = cost + no_gate[path[0]] + no_gate[path[-1]]
                if total < worst_kept():
                    best.append((total, list(path)))
                    best.sort(key=lambda item: item[0])
                    del best[num_ranked:]
            return
        remaining = num_qubits - len(path)
        if cost + remaining * cheapest_edge >= worst_kept() or not can_reach(in_path, path[-1], remaining):
            return
        for n in neighbours[path[-1]]:
            if n in in_path:
                continue
            if expansions >= step_limit:
                stopped_early = True
                return
            expansions += 1
            path.append(n)
            in_path.add(n)
            extend(path, in_path, cost + weight[path[-2], n] + (num_layers - 2) * no_gate[n])
            path.pop()
            in_path.remove(n)

    # Each start qubit gets an equal share of the budget, so a start that leads into a dead end cannot use it all;
    # steps a start leaves unused carry over to the next ones
    starts = sorted(graph, key=lambda q: weight[q, neighbours[q][0]])
    for i, start in enumerate(starts):
        step_limit = search_budget * (i + 1) // len(starts)
        # Interior qubits have no gate in num_layers - 2 layers, the two ends in one more, added when a chain is done
        extend([start], {start}, (num_layers - 2) * no_gate[start])
    if stopped_early:
        qcvv_logger.warning(
            f"EPLG chain search stopped after search_budget={search_budget} steps: the chains found are the best "
            f"within the budget, not proven optimal."
        )
    return best


def _best_trees(
    graph: nx.Graph,
    num_qubits: int,
    no_gate: Mapping[int, float],
    max_degree: int,
    num_layers: int | None,
) -> list[tuple[float, list[Edge]]]:
    """Trees grown from every qubit by the cheapest edge that keeps the qubit degrees within ``max_degree``.

    A tree is scored in ``num_layers`` layers, or in as many layers as its highest qubit degree.
    """
    layers = num_layers or max_degree
    trees: dict[frozenset[frozenset[int]], list[Edge]] = {}
    for seed in graph:
        in_tree, degree, edges = {seed}, Counter[int](), []
        while len(in_tree) < num_qubits:
            # Adding b below a costs the edge, plus b idling in all but one layer, minus a's layer that gains a gate
            options = [
                (graph.edges[a, b]["weight"] + (layers - 1) * no_gate[b] - no_gate[a], a, b)
                for a in in_tree
                if degree[a] < max_degree
                for b in graph[a]
                if b not in in_tree
            ]
            if not options:
                break
            _, a, b = min(options)
            edges.append((a, b))
            in_tree.add(b)
            degree.update((a, b))
        if len(in_tree) == num_qubits:
            trees[frozenset(frozenset(edge) for edge in edges)] = edges

    scored = []
    for edges in trees.values():
        tree_layers = num_layers or max(Counter(q for edge in edges for q in edge).values())
        scored.append((layer_cost(graph, edges, tree_layers, no_gate), edges))
    return sorted(scored, key=lambda item: item[0])


def rank_eplg_edges(  # noqa: PLR0913
    backend: IQMBackendBase,
    num_qubits: int,
    edge_selection: EdgeSelection = "tree",
    *,
    excluded_qubits: Sequence[int] = (),
    num_layers: int | None = None,
    edge_fidelities: Mapping[Edge, float] | None = None,
    num_ranked: int = 5,
    search_budget: int = DEFAULT_SEARCH_BUDGET,
    metrics: CalibrationMetrics | None = None,
) -> list[EPLGCandidate]:
    """Rank selections of ``num_qubits`` connected qubits by predicted layer fidelity, best first.

    ``"chain"`` ranks chains, which run in 2 layers. ``"tree"`` ranks the chains together with trees needing up to 3
    or 4 layers, or exactly ``num_layers`` layers if given.

    Args:
        backend: The backend whose coupling map defines the connectivity.
        num_qubits: How many qubits each selection connects.
        edge_selection: ``"tree"`` or ``"chain"``.
        excluded_qubits: Qubits the selections must avoid.
        num_layers: The number of layers each selection runs in; ``None`` uses the fewest.
        edge_fidelities: Edge fidelities to score with; only these edges can be selected. ``None`` uses the
            calibrated 2Q Clifford fidelities, or the CZ fidelities if there are none.
        num_ranked: How many selections to return.
        search_budget: The maximum number of path extensions of the chain search.
        metrics: Calibration metrics already read with :func:`calibration_metrics_by_backend_index`.

    Returns:
        The best selections, best first.

    Raises:
        ValueError: If ``edge_selection``, ``num_qubits`` or ``num_layers`` is invalid, or no selection was found.

    """
    if edge_selection not in ("tree", "chain"):
        raise ValueError(f"EPLG edge selection must be 'tree' or 'chain', got {edge_selection!r}.")
    fewest_layers = 1 if num_qubits == 2 else 2  # noqa: PLR2004
    if num_layers is not None and not fewest_layers <= num_layers <= num_qubits - 1:
        raise ValueError(
            f"num_disjoint_layers must be between {fewest_layers} and {num_qubits - 1} for {num_qubits} qubits."
        )

    metrics = calibration_metrics_by_backend_index(backend) if metrics is None else metrics
    graph, source = _selection_graph(backend, metrics, excluded_qubits, edge_fidelities)
    if not 2 <= num_qubits <= graph.number_of_nodes():  # noqa: PLR2004
        raise ValueError(f"EPLG edge selection needs between 2 and {graph.number_of_nodes()} qubits, got {num_qubits}.")
    no_gate = _no_gate_costs(metrics, list(graph.nodes))

    chain_layers = num_layers or fewest_layers
    found = []
    longest_chain = _longest_possible_chain(graph)
    if num_qubits <= longest_chain:
        for cost, path in _best_chains(graph, num_qubits, no_gate, chain_layers, num_ranked, search_budget):
            found.append((cost, list(pairwise(path)), chain_layers))
    elif edge_selection == "chain":
        raise ValueError(f"No chain of {num_qubits} qubits exists: the longest possible chain has {longest_chain}.")
    if edge_selection == "tree":
        caps = (3, 4) if num_layers is None else (num_layers,) if num_layers > 2 else ()  # noqa: PLR2004
        for cap in caps:
            if cap < num_qubits:
                for cost, edges in _best_trees(graph, num_qubits, no_gate, cap, num_layers)[:num_ranked]:
                    degree = max(Counter(q for edge in edges for q in edge).values())
                    found.append((cost, edges, num_layers or degree))

    candidates: list[EPLGCandidate] = []
    seen: set[frozenset[frozenset[int]]] = set()
    for cost, edges, layers in sorted(found, key=lambda item: item[0]):
        key = frozenset(frozenset(edge) for edge in edges)
        if key not in seen:
            seen.add(key)
            is_chain = max(Counter(q for edge in edges for q in edge).values()) <= 2  # noqa: PLR2004
            label = "chain" if is_chain else "tree"
            candidates.append(EPLGCandidate(orient_to_coupling_map(backend, edges), layers, cost, label, source))
    if not candidates:
        raise ValueError(
            f"No {edge_selection} of {num_qubits} connected qubits was found among the selectable couplers"
            f"{' within search_budget' if edge_selection == 'chain' else ''}."
        )
    return candidates[:num_ranked]
