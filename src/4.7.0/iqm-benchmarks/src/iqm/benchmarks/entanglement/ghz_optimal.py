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

"""Choosing the qubits, the tree and the gate order of a GHZ state.

The GHZ state is prepared by spreading entanglement from one root qubit along a spanning tree of the
chip's coupling graph, so the tree decides both the gate error paid (its edges are the CZ gates
played) and the circuit depth (its shape decides how many of them can run in parallel). Optimizing
only the first of those -- taking the minimum spanning tree, which minimizes the total
``-log(CZ fidelity)`` and nothing else -- degenerates on a chip of similar couplers into a long
snake, where every qubit is entangled early and then idles through the remaining layers while the
rest of the tree is built.

This module backs the ``"tree"`` state generation routine of
:mod:`iqm.benchmarks.entanglement.ghz`, and picks all three together:

* :func:`_ghz_graph` builds the weighted connectivity graph over the layout;
* :func:`_tree_candidates` enumerates the trees worth considering;
* :func:`_schedule_cost` prices each one against the objective in :class:`TreeSearchOptions`;
* :func:`rank_ghz_trees` returns the best candidate of each root, ordered by that price, and
  :func:`generate_ghz_tree_optimal` turns the requested rank of that ranking into a circuit.

The root qubit is the one the objective weighs most heavily: it carries the extra Hadamard, normally
holds the highest degree of the tree, and is entangled before the first layer, so it is exposed to
the whole idle window. Which root wins therefore depends on its single-qubit gate fidelity, T1 and T2
as well as on the CZ fidelities of the tree edges.

Calibration comes from :func:`~iqm.benchmarks.utils.extract_fidelities_unified`, which does not
report gate durations, so a single :data:`CZ_GATE_DURATION` is assumed for every CZ layer.
"""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace as dc_replace
import heapq

from iqm.benchmarks.logging_config import qcvv_logger
from iqm.qiskit_iqm import IQMCircuit as QuantumCircuit
from iqm.qiskit_iqm.iqm_backend import IQMBackendBase
import networkx
import numpy as np

CZ_GATE_DURATION = 40e-9
"""Duration assumed for every CZ layer, in seconds. Replace by automatic fetching once available."""

TREE_OBJECTIVES = ("estimated_fidelity", "depth_penalty", "min_error")
"""Accepted values of :attr:`TreeSearchOptions.objective`."""

_REQUIRED_METRIC_KEYS: tuple[str, ...] = (
    "cz_gate_fidelity",
    "t1_time",
    "t2_time",
    "fidelity_1qb_gates_averaged",
)
"""Metric keys the tree search reads; a missing one silently falls back to defaults, so warn."""

_HOP_PENALTIES: tuple[float, ...] = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.3, 1.0)
"""Per-hop surcharges sweeping the candidate trees from cheapest (0.0) to shallowest."""

_MAX_PRIM_SUBSETS = 5
"""How many weight-greedy subsets are kept as seeds for the tree search."""

_MAX_RANKING_ROWS = 10
"""How many rows of the candidate ranking are logged; enough to choose a rank from, not a wall."""

_COST_TIE_DECIMALS = 3
"""Decimal places at which two candidate costs count as equal, so that the shallower tree wins."""

_MICROSECONDS = 1e-6
"""Factor converting the microsecond coherence times of the metrics dictionary into seconds."""

_DEFAULT_T1 = 40e-6
"""Qubit relaxation time assumed when the calibration carries none, in seconds."""

_DEFAULT_T2 = 30e-6
"""Qubit coherence time assumed when the calibration carries none, in seconds."""

_DEFAULT_PRX_FIDELITY = 0.999
"""Single-qubit gate fidelity assumed when the calibration carries none."""

_T1_RANGE = (1e-7, 1e-2)
"""Range of T1 times accepted as a measurement, in seconds; anything else is a placeholder."""

_T2_RANGE = (1e-7, 1e-2)
"""Range of T2 times accepted as a measurement, in seconds; anything else is a placeholder."""

_MIN_PRX_FIDELITY = 0.5
"""Lowest single-qubit gate fidelity accepted as a measurement; below this it is a placeholder.

The upper bound is exclusive 1.0, matching what :func:`_ghz_graph` does with CZ fidelities: an
uncharacterized qubit is commonly filled with exactly 1.0, and taking that at face value would make
the worst qubit on the chip look like the best one instead of giving it the median.
"""


@dataclass(frozen=True)
class TreeSearchOptions:
    """How the tree search trades gate fidelity against circuit depth."""

    objective: str = "estimated_fidelity"
    """Which cost to minimize, one of :data:`TREE_OBJECTIVES`.

    ``"estimated_fidelity"`` scores a candidate by its estimated GHZ infidelity,
    ``gate_cost + idle_penalty * sum(idle time * (1/T1 + 1/T2))``, where ``gate_cost`` is
    :func:`_gate_cost`: the ``-log`` fidelity of the tree's CZs plus that of the single-qubit gates
    around them. ``"depth_penalty"`` uses the simpler ``gate_cost + depth_penalty * number of CZ
    layers``, which needs no coherence times. ``"min_error"`` scores by the gate fidelity alone,
    which is what this routine did before the idle term existed: it selects the same edges as a
    minimum spanning tree, so it is the setting to use to reproduce older results.
    """

    idle_penalty: float = 1.0
    """Weight of the idle-decoherence term of the ``"estimated_fidelity"`` objective.

    ``0.0`` disables it (leaving the gate fidelity product), values above ``1.0`` buy depth more
    aggressively than the T1 and T2 estimate alone would justify.
    """

    depth_penalty: float = 0.05
    """Cost of one CZ layer for the ``"depth_penalty"`` objective, in units of ``-log fidelity``.

    The scale that matters is what a layer costs the whole register, roughly
    ``num_qubits * layer duration / T2``; on a 45-qubit chip with 100 ns layers and T2 = 40 us that
    is about ``0.05``, which is the default. Values well below it never buy depth, since one layer of
    a 45-qubit tree is worth several CZ gates' worth of error.
    """

    cz_duration: float = CZ_GATE_DURATION
    """Duration of one CZ layer in seconds, used by the idle term of ``"estimated_fidelity"``."""


@dataclass(frozen=True)
class TreeCandidate:
    """One scored spanning tree, as returned by :func:`rank_ghz_trees`."""

    root: int
    """Qubit holding the initial superposition, i.e. the starting node of the tree."""

    pairs: list[tuple[int, int]]
    """The ``(control, target)`` CX pairs, as physical qubit indices, in gate order."""

    order: list[int]
    """The physical qubits in the order they become entangled, the root first.

    This is the logical-qubit order of the generated circuit: logical index ``i`` is ``order[i]``.
    Passing it to the transpiler as the initial layout is what keeps the tree edges on real
    couplings without a layout search.
    """

    cost: float
    """Value of the search objective for this candidate; lower is better."""

    stats: dict[str, float]
    """The terms behind :attr:`cost`: ``layers``, ``weight``, ``idle_cost`` and ``idle_time``."""

    family: str
    """Which candidate family produced it, one of ``"mst"``, ``"subset+dijkstra"``, ``"dijkstra"``."""

    rank: int = 0
    """Its position in the ranking returned by :func:`rank_ghz_trees`, ``0`` being the best."""


@dataclass(frozen=True)
class _ChipCosts:
    """The per-qubit calibration behind the cost model, in SI units and ``-log F``."""

    t1: dict[int, float]
    """Relaxation time of every qubit of the graph, in seconds."""

    t2: dict[int, float]
    """Coherence time of every qubit of the graph, in seconds."""

    prx_cost: dict[int, float]
    """``-log(single-qubit gate fidelity)`` of every qubit, in the same units as the edge weights."""


def _plausible(value: float, low: float, high: float) -> float | None:
    """Return ``value`` if it lies in ``[low, high]``, else ``None`` (a placeholder, not a measurement)."""
    return value if low <= value <= high else None


def _edge_fidelity(cz_fidelities: Mapping[int | tuple[int, int], float], control: int, target: int) -> float | None:
    """CZ fidelity of one coupling, tried in both directions, or ``None`` when it is not a measurement.

    A value outside ``(0, 1)`` is a placeholder rather than a measurement: an uncharacterized locus is
    commonly filled with exactly 1.0.
    """
    fidelity = cz_fidelities.get((control, target))
    if fidelity is None:
        fidelity = cz_fidelities.get((target, control))
    if fidelity is None:
        return None
    return float(fidelity) if 0.0 < float(fidelity) < 1.0 else None


def _ghz_graph(
    qubit_layout: Sequence[int],
    backend: IQMBackendBase,
    cz_fidelities: Mapping[int | tuple[int, int], float],
) -> networkx.Graph:
    """Build a weighted connectivity graph over the layout.

    Nodes are physical qubit indices; edges are the backend couplings whose endpoints are both in
    ``qubit_layout``. Edge weights are ``-log(CZ fidelity)``.

    Couplings without a usable fidelity take the median of the ones that have it, rather than
    degrading the whole search to an unweighted graph: one uncalibrated coupling costs several
    percent of GHZ fidelity on a large layout if it drags every other weight down with it. This is
    the same median substitution :meth:`~iqm.benchmarks.entanglement.ghz.GHZBenchmark` already
    applies to its own fidelity list.

    Args:
        qubit_layout: The subset of system-qubits used in the protocol, indexed from 0.
        backend: The backend whose coupling map defines the connectivity.
        cz_fidelities: The ``"cz_gate_fidelity"`` entry of the metrics dictionary.

    Returns:
        The weighted graph, carrying ``weight`` and ``fidelity`` on every edge.

    """
    layout = set(qubit_layout)
    edges: list[tuple[int, int]] = []
    seen: set[frozenset[int]] = set()
    for a, b in backend.coupling_map.get_edges():
        if a in layout and b in layout and a != b and frozenset((a, b)) not in seen:
            seen.add(frozenset((a, b)))
            edges.append((a, b))

    fidelities = [_edge_fidelity(cz_fidelities, a, b) for a, b in edges]
    known = [f for f in fidelities if f is not None]
    substitute = float(np.median(known)) if known else None
    if edges and substitute is None:
        qcvv_logger.warning(f"GHZ tree search: no usable CZ fidelities for {list(qubit_layout)}; using unit weights.")
    elif len(known) < len(edges):
        missing = [f"{a}-{b}" for (a, b), fidelity in zip(edges, fidelities, strict=True) if fidelity is None]
        qcvv_logger.warning(
            f"GHZ tree search: {len(edges) - len(known)} of {len(edges)} couplings have no usable CZ fidelity; "
            f"substituting the median {substitute:.4f} for {missing}."
        )

    graph: networkx.Graph = networkx.Graph()
    graph.add_nodes_from(qubit_layout)
    for (a, b), fidelity in zip(edges, fidelities, strict=True):
        used = fidelity if fidelity is not None else substitute
        weight = 1.0 if used is None else -float(np.log(used))
        # ``fidelity`` is carried on the edge so the chosen tree can be logged and compared against
        # another routine's tree for the same layout.
        # weight is set to 1 in the case where no fidelity is found in the calibration data
        graph.add_edge(a, b, weight=weight, fidelity=used)
    return graph


def _per_qubit(
    graph: networkx.Graph,
    values: Mapping[int | tuple[int, int], float],
    scale: float,
    bounds: tuple[float, float],
    default: float,
) -> tuple[dict[int, float], int]:
    """One per-qubit calibration channel over the whole graph, in SI units, and how many were known.

    A value outside ``bounds`` after scaling is a placeholder, not a measurement; it and any qubit
    missing from ``values`` take the median of the plausible ones, or ``default`` when nothing is
    plausible.

    Args:
        graph: The connectivity graph whose nodes are to be covered.
        values: One entry of the metrics dictionary, keyed by physical qubit index.
        scale: Factor converting the reported values into SI units.
        bounds: Lowest and highest scaled value accepted as a measurement.
        default: Value used when no qubit of the graph has a plausible one.

    Returns:
        The value of every node of the graph, and how many of them were known.

    """
    known: dict[int, float] = {}
    for qubit, value in values.items():
        if not isinstance(qubit, int) or qubit not in graph:
            continue
        scaled = _plausible(float(value) * scale, *bounds)
        if scaled is not None:
            known[qubit] = scaled
    fallback = float(np.median(list(known.values()))) if known else default
    return {qubit: known.get(qubit, fallback) for qubit in graph.nodes}, len(known)


def _chip_costs(graph: networkx.Graph, metrics: Mapping[str, Mapping[int | tuple[int, int], float]]) -> _ChipCosts:
    """The per-qubit calibration behind the cost model, in seconds and ``-log F``.

    Values outside the corresponding range are treated as missing rather than used: a placeholder or
    a value in the wrong unit would otherwise silently scale a term into irrelevance, which is
    exactly how a T2 in microseconds passed off as seconds hides the idle term completely.

    Args:
        graph: The connectivity graph whose nodes are to be covered.
        metrics: The metrics dictionary from :func:`~iqm.benchmarks.utils.extract_fidelities_unified`.

    Returns:
        The relaxation times, coherence times and single-qubit gate costs of every node.

    """
    # ``extract_fidelities_unified`` reports coherence times in microseconds; everything here is in
    # seconds, so the values have to be converted, not just copied. Fidelities are dimensionless, so
    # their scale is 1.0; their upper bound is exclusive, hence the nextafter rather than 1.0 itself.
    t1, n_t1 = _per_qubit(graph, metrics.get("t1_time", {}), _MICROSECONDS, _T1_RANGE, _DEFAULT_T1)
    t2, n_t2 = _per_qubit(graph, metrics.get("t2_time", {}), _MICROSECONDS, _T2_RANGE, _DEFAULT_T2)
    prx, n_prx = _per_qubit(
        graph,
        metrics.get("fidelity_1qb_gates_averaged", {}),
        1.0,
        (_MIN_PRX_FIDELITY, float(np.nextafter(1.0, 0.0))),
        _DEFAULT_PRX_FIDELITY,
    )
    n_nodes = graph.number_of_nodes()
    qcvv_logger.info(
        f"GHZ tree search calibration: T1 {1e6 * float(np.median(list(t1.values()))):.1f} us ({n_t1} of {n_nodes} "
        f"qubits), T2 {1e6 * float(np.median(list(t2.values()))):.1f} us ({n_t2} of {n_nodes}), single-qubit gate "
        f"fidelity {float(np.median(list(prx.values()))):.5f} ({n_prx} of {n_nodes}); the rest use the median."
    )
    return _ChipCosts(t1=t1, t2=t2, prx_cost={qubit: -float(np.log(f)) for qubit, f in prx.items()})


def _grow_connected_subset(graph: networkx.Graph, seed: int, num_qubits: int) -> tuple[list[int], float] | None:
    """Grow a connected ``num_qubits``-node set from ``seed``, cheapest edge first (Prim, truncated).

    Args:
        graph: The weighted connectivity graph.
        seed: The qubit to grow the set from.
        num_qubits: How many qubits the set must hold.

    Returns:
        The chosen qubits and the total weight of the tree connecting them, or ``None`` if ``seed``'s
        connected component is smaller than ``num_qubits``.

    """
    chosen = [seed]
    in_set = {seed}
    total = 0.0
    frontier: list[tuple[float, int, int]] = [
        (data["weight"], seed, neighbour) for neighbour, data in graph[seed].items()
    ]
    heapq.heapify(frontier)
    while len(chosen) < num_qubits and frontier:
        weight, _from, qubit = heapq.heappop(frontier)
        if qubit in in_set:
            continue
        chosen.append(qubit)
        in_set.add(qubit)
        total += weight
        for neighbour, data in graph[qubit].items():
            if neighbour not in in_set:
                heapq.heappush(frontier, (data["weight"], qubit, neighbour))
    if len(chosen) < num_qubits:
        return None
    return chosen, total


def _children_of(tree: networkx.Graph, root: int) -> dict[int, list[int]]:
    """Root an (undirected) tree at ``root``, returning the child list of every node.

    Args:
        tree: The tree to root.
        root: The node to root it at.

    Returns:
        The child list of every node.

    """
    children: dict[int, list[int]] = {node: [] for node in tree.nodes}
    seen = {root}
    stack = [root]
    while stack:
        node = stack.pop()
        for neighbour in tree[node]:
            if neighbour in seen:
                continue
            seen.add(neighbour)
            children[node].append(neighbour)
            stack.append(neighbour)
    return children


def _broadcast_schedule(children: Mapping[int, Sequence[int]], root: int) -> tuple[int, list[tuple[int, int]]]:
    """Order the CX gates of a rooted tree into as few layers as the tree allows.

    Spreading entanglement through a tree is the *minimum broadcast time in a tree* problem: a qubit
    can entangle only one child per layer, so a node with ``k`` children needs at least ``k`` layers.
    The greedy rule of Slater, Cockayne and Hedetniemi solves it exactly -- serve the children in
    order of decreasing subtree broadcast time, giving ``b(v) = max_i (i + 1 + b(c_i))`` -- which is
    what this implements.

    Args:
        children: Child list of every node of the rooted tree.
        root: Node holding the initial superposition (the Hadamard).

    Returns:
        The number of layers and the ``(control, target)`` pairs in start-time order, which is the
        order the CX gates are emitted in and :func:`cz_layers` then re-derives.

    """
    visit: list[int] = []
    stack = [root]
    while stack:
        node = stack.pop()
        visit.append(node)
        stack.extend(children.get(node, ()))

    depth_of: dict[int, int] = {}
    ranked: dict[int, list[int]] = {}
    for node in reversed(visit):
        kids = sorted(children.get(node, ()), key=lambda child: -depth_of[child])
        ranked[node] = kids
        depth_of[node] = max((rank + 1 + depth_of[kid] for rank, kid in enumerate(kids)), default=0)

    start = {root: 0}
    events: list[tuple[int, int, int]] = []
    for node in visit:  # parents are visited before their children, so ``start[node]`` is known
        for rank, kid in enumerate(ranked[node]):
            start[kid] = start[node] + rank + 1
            events.append((start[kid], node, kid))
    events.sort()
    return depth_of[root], [(control, target) for _start, control, target in events]


def _tree_from_root(
    graph: networkx.Graph,
    root: int,
    hop_penalty: float,
    limit: int | None = None,
) -> tuple[int, dict[int, list[int]]] | None:
    """Grow a tree from ``root`` with Dijkstra over ``weight + hop_penalty``.

    ``hop_penalty = 0`` gives the shortest-path tree (cheap but deep); a large penalty gives a
    breadth-first tree (shallow but expensive), so sweeping the penalty walks the whole family. With
    ``limit`` set the search stops after that many nodes; a prefix of the Dijkstra order is always
    connected, so this doubles as a qubit-subset chooser.

    Args:
        graph: The weighted connectivity graph.
        root: The qubit to grow the tree from.
        hop_penalty: Surcharge added to every edge, in units of ``-log fidelity`` per hop.
        limit: Stop after this many nodes; ``None`` spans the whole component.

    Returns:
        ``(root, children)``, or ``None`` if fewer than ``limit`` nodes are reachable.

    """
    distance = {root: 0.0}
    parent: dict[int, int | None] = {root: None}
    chosen: list[int] = []
    done: set[int] = set()
    frontier: list[tuple[float, int]] = [(0.0, root)]
    while frontier and (limit is None or len(chosen) < limit):
        dist, node = heapq.heappop(frontier)
        if node in done:
            continue
        done.add(node)
        chosen.append(node)
        for neighbour, data in graph[node].items():
            if neighbour in done:
                continue
            candidate = dist + data["weight"] + hop_penalty
            if candidate < distance.get(neighbour, float("inf")):
                distance[neighbour] = candidate
                parent[neighbour] = node
                heapq.heappush(frontier, (candidate, neighbour))
    if limit is not None and len(chosen) < limit:
        return None
    children: dict[int, list[int]] = {node: [] for node in chosen}
    for node in chosen:
        above = parent[node]
        if above is not None:
            children[above].append(node)
    return root, children


def _gate_cost(graph: networkx.Graph, pairs: Sequence[tuple[int, int]], chip: _ChipCosts) -> float:
    """Total ``-log(gate fidelity)`` of the tree: its CZs plus the single-qubit gates around them.

    Args:
        graph: The weighted connectivity graph.
        pairs: The ``(control, target)`` CX pairs of the tree.
        chip: The per-qubit calibration.

    Returns:
        The summed gate cost.

    """
    cost = sum(float(graph[control][target]["weight"]) for control, target in pairs)
    if not pairs:
        return cost

    # Each CX decomposes to PRX-CZ-PRX on both endpoints, so a qubit pays two PRX per incident tree
    # edge; the root pays one more for the Hadamard that creates the superposition. This is what
    # makes the search prefer a root whose single-qubit gates are good, since the root normally
    # carries the highest degree of the tree as well as the extra gate.
    prx_ops: dict[int, int] = {pairs[0][0]: 1}
    for control, target in pairs:
        prx_ops[control] = prx_ops.get(control, 0) + 2
        prx_ops[target] = prx_ops.get(target, 0) + 2
    return cost + sum(count * chip.prx_cost[qubit] for qubit, count in prx_ops.items())


def _schedule_cost(
    graph: networkx.Graph,
    pairs: Sequence[tuple[int, int]],
    chip: _ChipCosts,
    options: TreeSearchOptions,
) -> tuple[float, dict[str, float]]:
    """Cost of one candidate schedule (lower is better), plus the numbers behind it.

    The idle term is taken from the actual schedule rather than from the layer count: a qubit
    entangled in layer ``l`` decoheres through every later layer it does not act in, so the cost
    rewards entangling qubits late as much as it rewards finishing early. The root is the qubit this
    weighs most heavily, being entangled before layer 0 and so exposed to the whole window.

    Args:
        graph: The weighted connectivity graph.
        pairs: The ``(control, target)`` CX pairs in gate order.
        chip: The per-qubit calibration.
        options: What the search minimizes.

    Returns:
        The cost, and the ``layers``, ``weight``, ``idle_cost`` and ``idle_time`` behind it.

    """
    layer_of = cz_layers(pairs)
    n_layers = max(layer_of) + 1 if layer_of else 0
    weight = _gate_cost(graph, pairs, chip)

    entangled_in: dict[int, int] = {pairs[0][0]: -1} if pairs else {}
    busy: dict[int, set[int]] = {}
    for (control, target), layer in zip(pairs, layer_of, strict=True):
        entangled_in[target] = layer
        busy.setdefault(control, set()).add(layer)
        busy.setdefault(target, set()).add(layer)
    idle = 0.0
    total_idle_time = 0.0
    for qubit, entangled in entangled_in.items():
        active = busy.get(qubit, set())
        # Every layer is the same length, so the idle time is just how many of them this qubit sits
        # out after it has been entangled.
        idle_layers = sum(1 for layer in range(entangled + 1, n_layers) if layer not in active)
        idle_time = options.cz_duration * idle_layers
        total_idle_time += idle_time
        # The multiple-quantum-coherences fidelity has two terms and they decay differently: the
        # |0...0> and |1...1> populations relax at T1 while the coherence between them dephases at
        # T2. Both are per idling qubit, so the idle rate is their sum. This is not a double count of
        # the T1 that already limits T2 -- these are two separate terms of the same fidelity.
        idle += idle_time * (1.0 / chip.t2[qubit] + 1.0 / chip.t1[qubit])

    if options.objective == "depth_penalty":
        cost = weight + options.depth_penalty * n_layers
    elif options.objective == "min_error":
        cost = weight
    else:
        cost = weight + options.idle_penalty * idle
    return cost, {
        "layers": float(n_layers),
        "weight": weight,
        "idle_cost": idle,
        "idle_time": total_idle_time,
    }


def _tree_candidates(
    graph: networkx.Graph,
    qubit_layout: Sequence[int],
    num_qubits: int,
) -> Iterator[tuple[str, int, dict[int, list[int]]]]:
    """Yield ``(family, root, children)`` for every tree worth scoring.

    Three families: the minimum spanning tree rooted at each node (so the search can never come out
    behind minimizing gate error alone), Dijkstra trees inside the cheapest weight-greedy subsets, and
    Dijkstra trees truncated to ``num_qubits`` nodes over the whole graph. The last two disagree
    about which qubits to use, and which one wins depends on the calibration.

    Args:
        graph: The weighted connectivity graph.
        qubit_layout: The qubits available to the run.
        num_qubits: Size of the GHZ state.

    Yields:
        The candidate family, its root and the child list of every node.

    """
    if num_qubits >= len(qubit_layout):
        subsets: list[list[int]] = [list(qubit_layout)]
    else:
        grown = [
            candidate
            for candidate in (_grow_connected_subset(graph, seed, num_qubits) for seed in qubit_layout)
            if candidate is not None
        ]
        grown.sort(key=lambda item: item[1])
        subsets = [chosen for chosen, _weight in grown[:_MAX_PRIM_SUBSETS]]

    for chosen in subsets:
        sub = graph.subgraph(chosen)
        if len(chosen) > 1 and not networkx.is_connected(sub):
            continue
        span = networkx.minimum_spanning_tree(sub)
        for root in chosen:
            yield "mst", root, _children_of(span, root)
            for hop_penalty in _HOP_PENALTIES:
                grown_tree = _tree_from_root(sub, root, hop_penalty)
                if grown_tree is not None:
                    yield "subset+dijkstra", grown_tree[0], grown_tree[1]

    if num_qubits < len(qubit_layout):
        for root in qubit_layout:
            for hop_penalty in _HOP_PENALTIES:
                grown_tree = _tree_from_root(graph, root, hop_penalty, limit=num_qubits)
                if grown_tree is not None:
                    yield "dijkstra", grown_tree[0], grown_tree[1]


def _cz_fidelity_product(graph: networkx.Graph, pairs: Sequence[tuple[int, int]]) -> float:
    """Product of the CZ fidelities of the tree edges, or ``nan`` when any of them is unknown.

    Args:
        graph: The weighted connectivity graph.
        pairs: The ``(control, target)`` CX pairs of the tree.

    Returns:
        The fidelity product.

    """
    known: list[float] = [
        float(fidelity)
        for control, target in pairs
        if graph.has_edge(control, target) and (fidelity := graph.edges[control, target].get("fidelity")) is not None
    ]
    if not known or len(known) != len(pairs):
        return float("nan")
    return float(np.prod(known))


def _log_ranking(ranked: Sequence[TreeCandidate], options: TreeSearchOptions) -> None:
    """Log the top of the candidate ranking, so a rank can be chosen for the next run.

    Args:
        ranked: The candidates, best first.
        options: What the search minimized.

    """
    # The cost is printed to more decimals than ties are taken at, so that rows sharing a rounded cost
    # are visibly the same to _COST_TIE_DECIMALS and the layer ordering between them reads as
    # deliberate. The idle cost is the raw one rather than idle_penalty * idle_cost, which would be
    # 0.0000 on every row precisely when idle_penalty is 0 and the tie rule matters most.
    rows = "\n".join(
        f"  rank {candidate.rank}  root {candidate.root}  cost {candidate.cost:.6f}  "
        f"(gates {candidate.stats['weight']:.6f}, idle cost {candidate.stats['idle_cost']:.6f})  "
        f"{int(candidate.stats['layers'])} layer(s)"
        for candidate in ranked[:_MAX_RANKING_ROWS]
    )
    qcvv_logger.info(
        f"GHZ tree ranking (showing {min(len(ranked), _MAX_RANKING_ROWS)} of {len(ranked)} candidate roots, "
        f"objective={options.objective}, idle_penalty={options.idle_penalty:g}). Costs equal to "
        f"{_COST_TIE_DECIMALS} decimals count as a tie and are ordered by CZ layer count. Pass the rank of the "
        f"one to run as GHZConfiguration.tree_rank:\n{rows}"
    )


def _log_tree(candidate: TreeCandidate, n_candidates: int, options: TreeSearchOptions) -> None:
    """Log the chosen tree, why the search preferred it, and what it is expected to cost.

    The CZ fidelity product is the dominant term of the expected GHZ fidelity, so it is the number to
    compare when two routines disagree about the fidelity of the same layout; the layer count and the
    idle cost are what the search traded it against. The root's own calibration is included because
    it is the qubit the objective weighs most heavily, and without it there is no way to tell a real
    preference from a median substitution.

    Args:
        candidate: The chosen candidate.
        n_candidates: How many candidates it was chosen from.
        options: What the search minimized.

    """
    stats = candidate.stats
    product = "n/a" if np.isnan(stats["cz_fidelity_product"]) else f"{stats['cz_fidelity_product']:.4f}"
    qcvv_logger.info(
        f"GHZ tree: {len(candidate.order)} qubits, root={candidate.root} "
        f"(1QB gate fidelity {stats['root_prx_fidelity']:.5f}, T1 {1e6 * stats['root_t1']:.1f} us, "
        f"T2 {1e6 * stats['root_t2']:.1f} us), {len(candidate.pairs)} CX in {int(stats['layers'])} CZ layer(s), "
        f"CZ fidelity product={product}, total idle time={1e6 * stats['idle_time']:.1f} us, "
        f"idle cost={stats['idle_cost']:.4g}, total cost={candidate.cost:.4f} "
        f"(rank {candidate.rank} of {n_candidates}, objective={options.objective}, family={candidate.family})"
    )


def _warn_missing_metric_keys(metrics: Mapping[str, Mapping[int | tuple[int, int], float]]) -> None:
    """Warn once for every :data:`_REQUIRED_METRIC_KEYS` entry absent from ``metrics``."""
    for key in _REQUIRED_METRIC_KEYS:
        if key not in metrics:
            qcvv_logger.warning(
                f"WARNING: GHZ tree search metric {key!r} is missing; falling back to default values "
                f"for it. Expected keys are {_REQUIRED_METRIC_KEYS}. For best results, all metrics must be found."
            )


def rank_ghz_trees(
    qubit_layout: Sequence[int],
    backend: IQMBackendBase,
    metrics: Mapping[str, Mapping[int | tuple[int, int], float]],
    num_qubits: int | None = None,
    options: TreeSearchOptions | None = None,
) -> list[TreeCandidate]:
    """Score one spanning tree per possible root qubit and return them best first.

    Subset and tree are chosen together: which ``num_qubits`` qubits are cheapest depends on the
    shape of the tree that will connect them, so scoring the two separately picks an elongated set
    and then pays for it in depth.

    Only the best candidate of each root is kept. The search generates many trees per root that
    differ by a single edge, and keeping them would fill the ranking with near-duplicates -- one per
    root is what makes rank ``n`` a genuinely different starting node rather than a rounding
    difference.

    Candidates whose costs agree to :data:`_COST_TIE_DECIMALS` are ordered by CZ layer count, so that
    where the objective cannot tell two trees apart the shallower one ranks first.

    Args:
        qubit_layout: The subset of system-qubits used in the protocol, indexed from 0.
        backend: The backend whose coupling map defines the connectivity.
        metrics: The metrics dictionary from :func:`~iqm.benchmarks.utils.extract_fidelities_unified`,
            whose keys are physical qubit indices.
        num_qubits: Size of the GHZ state; ``None`` uses every qubit of the layout.
        options: What the search minimizes; defaults to :class:`TreeSearchOptions`.

    Returns:
        The candidates sorted by non-decreasing cost, one per root, never empty: when no connected
        set of the requested size exists it holds a single linear-chain fallback.

    Raises:
        ValueError: If ``options.objective`` is not one of :data:`TREE_OBJECTIVES`.

    """
    options = options or TreeSearchOptions()
    if options.objective not in TREE_OBJECTIVES:
        raise ValueError(f"Unknown GHZ tree objective {options.objective!r}; expected one of {TREE_OBJECTIVES}.")
    _warn_missing_metric_keys(metrics)
    graph = _ghz_graph(qubit_layout, backend, metrics.get("cz_gate_fidelity", {}))
    target = len(qubit_layout) if num_qubits is None else min(int(num_qubits), len(qubit_layout))
    if num_qubits is not None and int(num_qubits) > len(qubit_layout):
        qcvv_logger.warning(
            f"GHZ tree search: num_qubits={int(num_qubits)} exceeds the {len(qubit_layout)} qubits available to this "
            f"run; using all of them."
        )
    chip = _chip_costs(graph, metrics)

    # Costs that agree to _COST_TIE_DECIMALS are treated as equal and the shallower tree wins, then
    # the one whose idling lands on better qubits, then the genuinely cheaper one, then the lowest
    # root so the order is total and reproducible. Depth has to outrank idle cost rather than the
    # other way round: a deeper tree that happens to idle good qubits would otherwise beat a shallower
    # one, which is the opposite of what anyone reading the ranking wants. The unrounded cost sits
    # below idle cost rather than above it, because differences that small are what the tolerance
    # already declared meaningless; it is here only to settle rows that are otherwise identical.
    best_per_root: dict[int, tuple[tuple[float, float, float, float, int], TreeCandidate]] = {}
    for family, root, children in _tree_candidates(graph, qubit_layout, target):
        _depth, pairs = _broadcast_schedule(children, root)
        if len(pairs) != target - 1:  # the candidate does not span the requested number of qubits
            continue
        cost, stats = _schedule_cost(graph, pairs, chip, options)
        key = (round(cost, _COST_TIE_DECIMALS), stats["layers"], stats["idle_cost"], cost, root)
        previous = best_per_root.get(root)
        if previous is None or key < previous[0]:
            order = [root] + [target_qubit for _control, target_qubit in pairs]
            # Carry the numbers the log needs on the candidate itself, so reporting the chosen tree
            # does not mean rebuilding the graph and re-reading the calibration.
            stats["cz_fidelity_product"] = _cz_fidelity_product(graph, pairs)
            stats["root_prx_fidelity"] = float(np.exp(-chip.prx_cost[root]))
            stats["root_t1"] = chip.t1[root]
            stats["root_t2"] = chip.t2[root]
            best_per_root[root] = (
                key,
                TreeCandidate(root=root, pairs=pairs, order=order, cost=cost, stats=stats, family=family),
            )

    if not best_per_root:
        qcvv_logger.warning(
            f"GHZ tree search: no connected set of {target} qubits exists in {list(qubit_layout)}; falling back to a "
            f"linear chain."
        )
        order = list(qubit_layout[:target])
        pairs = [(order[i], order[i + 1]) for i in range(len(order) - 1)]
        stats = {"layers": float(len(pairs)), "weight": 0.0, "idle_cost": 0.0, "idle_time": 0.0}
        return [
            TreeCandidate(
                root=order[0] if order else -1,
                pairs=pairs,
                order=order,
                cost=float("inf"),
                stats=stats,
                family="linear-chain",
            )
        ]

    ranked = [
        dc_replace(candidate, rank=rank)
        for rank, (_key, candidate) in enumerate(sorted(best_per_root.values(), key=lambda item: item[0]))
    ]
    _log_ranking(ranked, options)
    return ranked


def select_ghz_tree(  # noqa: PLR0913
    qubit_layout: Sequence[int],
    backend: IQMBackendBase,
    metrics: Mapping[str, Mapping[int | tuple[int, int], float]],
    num_qubits: int | None = None,
    options: TreeSearchOptions | None = None,
    rank: int = 0,
) -> TreeCandidate:
    """Choose the qubits, the tree and the gate order of the GHZ state.

    Args:
        qubit_layout: The subset of system-qubits used in the protocol, indexed from 0.
        backend: The backend whose coupling map defines the connectivity.
        metrics: The metrics dictionary from :func:`~iqm.benchmarks.utils.extract_fidelities_unified`.
        num_qubits: Size of the GHZ state; ``None`` uses every qubit of the layout.
        options: What the search minimizes; defaults to :class:`TreeSearchOptions`.
        rank: Which candidate of :func:`rank_ghz_trees` to use, ``0`` being the best-scoring one. A
            rank beyond the last candidate falls back to the last one with a warning, rather than
            failing a run that has already built its circuits.

    Returns:
        The chosen candidate.

    Raises:
        ValueError: If ``options.objective`` is not one of :data:`TREE_OBJECTIVES`.

    """
    ranked = rank_ghz_trees(qubit_layout, backend, metrics, num_qubits, options)
    chosen = max(0, int(rank))
    if chosen >= len(ranked):
        qcvv_logger.warning(
            f"GHZ tree search: rank {chosen} was requested but only {len(ranked)} candidate roots exist; using rank "
            f"{len(ranked) - 1} instead."
        )
        chosen = len(ranked) - 1
    candidate = ranked[chosen]
    if candidate.family != "linear-chain":
        _log_tree(candidate, len(ranked), options or TreeSearchOptions())
    return candidate


def generate_ghz_tree_optimal(  # noqa: PLR0913
    qubit_layout: Sequence[int],
    backend: IQMBackendBase,
    metrics: Mapping[str, Mapping[int | tuple[int, int], float]],
    num_qubits: int | None = None,
    tree_options: TreeSearchOptions | None = None,
    rank: int = 0,
) -> tuple[QuantumCircuit, list[int], TreeCandidate | None]:
    """Build a shallow GHZ circuit from one of the ranked trees over the layout.

    Args:
        qubit_layout: The subset of system-qubits used in the protocol, indexed from 0.
        backend: The backend whose coupling map defines the connectivity.
        metrics: The metrics dictionary from :func:`~iqm.benchmarks.utils.extract_fidelities_unified`.
        num_qubits: Size of the GHZ state; ``None`` uses every qubit of the layout. When smaller than
            the layout, the tree search chooses which qubits to use.
        tree_options: What the tree search minimizes; defaults to :class:`TreeSearchOptions`.
        rank: Which of the ranked candidates to build, ``0`` being the best-scoring one.

    Returns:
        The GHZ circuit (measured), the physical qubits in logical-qubit order, and the chosen
        candidate (``None`` for a single-qubit layout, where there is no tree to choose). Logical
        index ``i`` corresponds to the ``i`` th returned qubit; that order must be given to the
        transpiler as the initial layout, so the tree edges land on physical couplings without a
        layout search.

    """
    candidate: TreeCandidate | None = None
    if len(qubit_layout) > 1:
        candidate = select_ghz_tree(qubit_layout, backend, metrics, num_qubits, tree_options, rank)
        cx_map, order = candidate.pairs, candidate.order
    else:
        order = list(qubit_layout)
        cx_map = []

    relabel = {qubit: idx for idx, qubit in enumerate(order)}
    qc = QuantumCircuit(len(order), name="ghz_tree_optimal")
    qc.h(relabel[order[0]])
    for control, target in cx_map:
        pair = [relabel[control], relabel[target]]
        # Barrier keeps Hadamards from migrating to the circuit front (phase-error prone).
        qc.barrier(pair)
        qc.cx(*pair)
    qc.measure_all()

    return qc, order, candidate


def cx_pairs(direct_ghz: QuantumCircuit, order: Sequence[int]) -> list[tuple[int, int]]:
    """Recover the physical-index CX pairs of a generated GHZ circuit, in circuit order.

    Args:
        direct_ghz: A circuit built by :func:`generate_ghz_tree_optimal`.
        order: The physical qubits in logical-qubit order, as returned alongside it.

    Returns:
        The ``(control, target)`` pairs as physical qubit indices.

    """
    pairs: list[tuple[int, int]] = []
    for instruction in direct_ghz.data:
        if instruction.operation.name != "cx":
            continue
        control, target = (direct_ghz.find_bit(qubit)[0] for qubit in instruction.qubits)
        pairs.append((order[control], order[target]))
    return pairs


def cz_layers(pairs: Sequence[tuple[int, int]]) -> list[int]:
    """Assign each CX of the tree to the earliest layer it can run in.

    The GHZ tree is built breadth first, but a breadth-first level is not a layer of the executed
    circuit: two CXs of the same level that share their control qubit have to be serialized, while
    CXs of different levels can overlap. Scheduling each gate as early as its two qubits allow (ASAP)
    reproduces what the barriers and the transpiler actually do, so gates sharing a layer index are
    the ones played in parallel.

    Args:
        pairs: ``(control, target)`` qubit indices in circuit order.

    Returns:
        The 0-based layer index of every pair.

    """
    next_free: dict[int, int] = {}
    layers: list[int] = []
    for control, target in pairs:
        layer = max(next_free.get(control, 0), next_free.get(target, 0))
        layers.append(layer)
        next_free[control] = next_free[target] = layer + 1
    return layers
