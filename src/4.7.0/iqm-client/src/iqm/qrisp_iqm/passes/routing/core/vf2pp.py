# Copyright 2026 IQM
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

"""VF2++ subgraph isomorphism for quantum circuit layout.

This module implements a Numba-accelerated VF2++-style subgraph
isomorphism search for placing quantum circuits onto hardware topologies.
The search uses BFS-based node ordering, T-set candidate narrowing,
and parallel exploration of first-node candidates across CPU cores.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from iqm.qrisp_iqm.passes.routing.core.permutation_tools import invert_permutation
from iqm.qrisp_iqm.passes.routing.core.qubit_padding import pad_qubits
import networkx as nx
from numba import njit, prange
import numpy as np
from qrisp import QuantumCircuit

if TYPE_CHECKING:
    from iqm.qrisp_iqm.passes.routing.core.graph_processing_tools import QPUTopology


# ================================================================
# Helper functions
# ================================================================


@njit(cache=True)
def _degree_array(indptr: np.ndarray) -> np.ndarray:
    """Compute the degree (number of neighbors) for each node.

    Uses a graph represented in CSR (Compressed Sparse Row) format.

    Args:
        indptr: CSR index pointer array, length = n_nodes + 1.

    Returns:
        Degree of each node.

    """
    n = indptr.size - 1
    deg = np.empty(n, dtype=np.int64)
    for i in range(n):
        deg[i] = indptr[i + 1] - indptr[i]
    return deg


@njit(cache=True)
def _select_order_by_degree(deg: np.ndarray) -> np.ndarray:
    """Produce a descending degree-based node order for matching.

    A simple VF2++ heuristic (higher-degree nodes first).

    Args:
        deg: Degree array of the pattern graph.

    Returns:
        Indices of pattern nodes sorted by descending degree.

    """
    n = deg.shape[0]
    order = np.empty(n, dtype=np.int64)
    used = np.zeros(n, dtype=np.uint8)
    for k in range(n):
        best = -1
        bestd = -1
        for i in range(n):
            if used[i] == 0 and deg[i] > bestd:
                bestd = deg[i]
                best = i
        order[k] = best
        used[best] = 1
    return order


@njit(cache=True)
def _vf2pp_bfs_order(indices: np.ndarray, indptr: np.ndarray) -> np.ndarray:  # noqa: PLR0912
    """VF2++ BFS-based node ordering.

    Produces a matching order that keeps the search tree connected:
    each successive node has the maximum number of neighbors already
    in the order, with total degree as tiebreaker.

    The algorithm processes one connected component at a time (starting
    from the highest-degree node in each), so disconnected pattern
    graphs are handled correctly.

    Args:
        indices: CSR column indices of the pattern graph.
        indptr: CSR row pointers of the pattern graph.

    Returns:
        Pattern node indices in VF2++ BFS order.

    """
    n = indptr.shape[0] - 1
    deg = _degree_array(indptr)
    order = np.empty(n, dtype=np.int64)
    in_order = np.zeros(n, dtype=np.uint8)  # 1 if node is in the order
    conn = np.zeros(n, dtype=np.int64)  # connections to ordered nodes
    pos = 0  # next write position in order

    while pos < n:
        # Pick the highest-degree unordered node as BFS root
        # (starts a new connected component if previous one is done)
        best_root = -1
        best_deg = -1
        for i in range(n):
            if in_order[i] == 0 and deg[i] > best_deg:
                best_deg = deg[i]
                best_root = i

        # BFS queue for the current component
        # We use a simple array-based queue
        queue = np.empty(n, dtype=np.int64)
        q_start = 0
        q_end = 0

        # Enqueue root
        queue[q_end] = best_root
        q_end += 1
        in_order[best_root] = 2  # 2 = enqueued but not yet ordered

        while q_start < q_end:
            # Collect all nodes at the current BFS level
            level_start = q_start
            level_end = q_end

            # Selection sort within this BFS level:
            # repeatedly pick the best node and add it to the order
            level_ordered = np.zeros(q_end - q_start, dtype=np.uint8)

            for _ in range(level_end - level_start):
                best_idx = -1
                best_conn = -1
                best_d = -1

                for li in range(level_end - level_start):
                    if level_ordered[li] == 1:
                        continue
                    node = queue[level_start + li]
                    c = conn[node]
                    d = deg[node]
                    if (c > best_conn) or (c == best_conn and d > best_d):
                        best_conn = c
                        best_d = d
                        best_idx = li

                node = queue[level_start + best_idx]
                level_ordered[best_idx] = 1

                # Add to order
                order[pos] = node
                in_order[node] = 1
                pos += 1

                # Update connectivity counts for neighbors
                for p in range(indptr[node], indptr[node + 1]):
                    nb = indices[p]
                    if in_order[nb] == 0:
                        conn[nb] += 1

            # Enqueue unvisited neighbors of ALL nodes in this level
            for li in range(level_end - level_start):
                node = queue[level_start + li]
                for p in range(indptr[node], indptr[node + 1]):
                    nb = indices[p]
                    if in_order[nb] == 0:
                        queue[q_end] = nb
                        q_end += 1
                        in_order[nb] = 2  # mark enqueued

            q_start = level_end

    return order


@njit(cache=True)
def _binary_adjacent(v: int, w: int, indices: np.ndarray, indptr: np.ndarray) -> bool:
    """Check adjacency between two nodes using binary search.

    Searches the neighbor list of v for w. Neighbor lists must be sorted.

    Args:
        v: Source node index.
        w: Target node index.
        indices: CSR column indices of the graph.
        indptr: CSR row pointers of the graph.

    Returns:
        True if edge (v, w) exists, False otherwise.

    """
    start = indptr[v]
    end = indptr[v + 1] - 1
    # Standard binary search
    while start <= end:
        mid = (start + end) // 2
        val = indices[mid]
        if val == w:
            return True
        elif val < w:
            start = mid + 1
        else:
            end = mid - 1
    return False


@njit(cache=True)
def _feasible(  # noqa: PLR0913
    u: int,
    v: int,
    mapping: np.ndarray,
    pattern_indices: np.ndarray,
    pattern_indptr: np.ndarray,
    target_indices: np.ndarray,
    target_indptr: np.ndarray,
) -> bool:
    """Check if mapping pattern node to target node is feasible.

    Given the current partial mapping.

    Feasibility checks:
    1. Degree pruning: deg(u) <= deg(v)
    2. Adjacency consistency: for every already-mapped neighbor u_n of u,
       its image v_n must be adjacent to v in the target graph.

    Args:
        u: Candidate node in the pattern graph.
        v: Candidate node in the target graph.
        mapping: Current mapping array, -1 for unmapped pattern nodes.
        pattern_indices: CSR column indices of the pattern graph.
        pattern_indptr: CSR row pointers of the pattern graph.
        target_indices: CSR column indices of the target graph.
        target_indptr: CSR row pointers of the target graph.

    Returns:
        True if feasible, False otherwise.

    """
    deg_u = pattern_indptr[u + 1] - pattern_indptr[u]
    deg_v = target_indptr[v + 1] - target_indptr[v]
    if deg_u > deg_v:
        return False

    # Check adjacency consistency with already-mapped neighbors
    start = pattern_indptr[u]
    end = pattern_indptr[u + 1]
    for p in range(start, end):
        u_n = pattern_indices[p]
        v_n = mapping[u_n]
        if v_n != -1 and not _binary_adjacent(v, v_n, target_indices, target_indptr):
            return False
    return True


# ================================================================
# Recursive matching (core of VF2)
# ================================================================


@njit(cache=False)
def _match_recursive(  # noqa: PLR0913, PLR0912
    pos: int,
    order: np.ndarray,
    mapping: np.ndarray,
    used_target: np.ndarray,
    pattern_indices: np.ndarray,
    pattern_indptr: np.ndarray,
    target_indices: np.ndarray,
    target_indptr: np.ndarray,
    counter: np.ndarray,
) -> bool:
    """Recursive search for subgraph isomorphism.

    Uses T-set candidate narrowing.

    When the current pattern node ``u`` has at least one already-mapped
    neighbor, only target nodes adjacent to that mapped neighbor are
    tried (T-set narrowing).  This drastically reduces branching on
    grid-like topologies.  If ``u`` has no mapped neighbors (start of a
    new connected component), all unused target nodes are tried.

    Args:
        pos: Current position in the pattern node order.
        order: Pattern node order (BFS heuristic order).
        mapping: Partial mapping from pattern -> target.
        used_target: Flags for used target nodes.
        pattern_indices: CSR column indices of the pattern graph.
        pattern_indptr: CSR row pointers of the pattern graph.
        target_indices: CSR column indices of the target graph.
        target_indptr: CSR row pointers of the target graph.
        counter: Array of shape ``(2,)``.  ``counter[0]`` is the remaining
            attempts, decremented on each candidate trial; when it reaches 0
            the search aborts.  ``counter[1]`` is the max_attempts limit
            (0 means unlimited).

    Returns:
        True if a full mapping is found, otherwise False.

    """
    num_pattern_nodes = order.shape[0]
    num_target_nodes = target_indptr.shape[0] - 1

    # All pattern nodes matched -> success
    if pos == num_pattern_nodes:
        return True

    u = order[pos]
    deg_u = pattern_indptr[u + 1] - pattern_indptr[u]

    # ── Build candidate set via T-set narrowing ──────────────────
    # Find the first already-mapped neighbor of u in the pattern graph.
    # Its image in the target graph defines the T-set: only neighbors
    # of that image are candidates for v.
    anchor = np.int64(-1)
    for p in range(pattern_indptr[u], pattern_indptr[u + 1]):
        u_n = pattern_indices[p]
        if mapping[u_n] != -1:
            anchor = mapping[u_n]
            break

    if anchor != -1:
        # T-set: only try neighbors of the anchor in the target graph
        for p in range(target_indptr[anchor], target_indptr[anchor + 1]):
            v = target_indices[p]
            if used_target[v]:
                continue

            deg_v = target_indptr[v + 1] - target_indptr[v]
            if deg_v < deg_u:
                continue

            if _feasible(u, v, mapping, pattern_indices, pattern_indptr, target_indices, target_indptr):
                # Check attempt budget
                if counter[1] > 0:
                    counter[0] -= 1
                    if counter[0] <= 0:
                        return False

                mapping[u] = v
                used_target[v] = 1

                if _match_recursive(
                    pos + 1,
                    order,
                    mapping,
                    used_target,
                    pattern_indices,
                    pattern_indptr,
                    target_indices,
                    target_indptr,
                    counter,
                ):
                    return True

                mapping[u] = -1
                used_target[v] = 0
    else:
        # No mapped neighbor — new connected component, try all nodes
        for v in range(num_target_nodes):
            if used_target[v]:
                continue

            deg_v = target_indptr[v + 1] - target_indptr[v]
            if deg_v < deg_u:
                continue

            if _feasible(u, v, mapping, pattern_indices, pattern_indptr, target_indices, target_indptr):
                if counter[1] > 0:
                    counter[0] -= 1
                    if counter[0] <= 0:
                        return False

                mapping[u] = v
                used_target[v] = 1

                if _match_recursive(
                    pos + 1,
                    order,
                    mapping,
                    used_target,
                    pattern_indices,
                    pattern_indptr,
                    target_indices,
                    target_indptr,
                    counter,
                ):
                    return True

                mapping[u] = -1
                used_target[v] = 0

    return False


# ================================================================
# Parallel top-level search
# ================================================================


@njit(cache=False, parallel=True)
def _match_parallel(  # noqa: PLR0913
    order: np.ndarray,
    pattern_indices: np.ndarray,
    pattern_indptr: np.ndarray,
    target_indices: np.ndarray,
    target_indptr: np.ndarray,
    max_attempts_per_thread: int,
) -> np.ndarray:
    """Parallel VF2++ search.

    Each thread tries a different candidate target node for the first
    pattern node in *order* and runs
    _match_recursive from pos=1 with its own budget.  The recursive
    search uses T-set candidate narrowing to limit branching.

    Args:
        order: Pattern node order (VF2++ BFS order).
        pattern_indices: CSR column indices of the pattern graph.
        pattern_indptr: CSR row pointers of the pattern graph.
        target_indices: CSR column indices of the target graph.
        target_indptr: CSR row pointers of the target graph.
        max_attempts_per_thread: Budget per thread (0 = unlimited).

    Returns:
        Array of shape ``(num_target_nodes, num_pattern_nodes)``.  Row *v*
        holds the mapping found when the first pattern node was assigned to
        target node *v*.  All -1 if that thread failed.

    """
    num_pattern_nodes = order.shape[0]
    num_target_nodes = target_indptr.shape[0] - 1

    # Each row is one thread's mapping result
    results = -np.ones((num_target_nodes, num_pattern_nodes), dtype=np.int64)

    u0 = order[0]
    deg_u0 = pattern_indptr[u0 + 1] - pattern_indptr[u0]

    for v in prange(num_target_nodes):
        deg_v = target_indptr[v + 1] - target_indptr[v]
        if deg_v < deg_u0:
            continue

        # Thread-local state
        mapping = -np.ones(num_pattern_nodes, dtype=np.int64)
        used_target = np.zeros(num_target_nodes, dtype=np.uint8)
        counter = np.array([max_attempts_per_thread, max_attempts_per_thread], dtype=np.int64)

        # Check feasibility for the root assignment (no mapped neighbors yet,
        # so only degree check matters — already done above)
        mapping[u0] = v
        used_target[v] = 1

        if num_pattern_nodes == 1 or _match_recursive(
            1, order, mapping, used_target, pattern_indices, pattern_indptr, target_indices, target_indptr, counter
        ):
            results[v] = mapping

    return results


# ================================================================
# Public interface
# ================================================================


def is_identity_embedding(
    pattern_indices: np.ndarray,
    pattern_indptr: np.ndarray,
    target_indices: np.ndarray,
    target_indptr: np.ndarray,
) -> bool:
    """Check whether the pattern graph is a subgraph of the target graph.

    Tests the *identity* mapping, i.e. whether node ``i`` of the pattern graph
    can stay on node ``i`` of the target graph.  For the layout passes this
    answers "does the circuit already have a valid layout?",
    so circuit qubit ``i`` maps directly to physical qubit ``i``.

    Args:
        pattern_indices: CSR neighbor indices of the circuit's interaction graph, with one node per
            circuit qubit.
        pattern_indptr: CSR index pointer of the circuit's interaction graph.
        target_indices: CSR neighbor indices of the target graph (device topology with one node per
            physical qubit).  Neighbor lists must be **sorted** — the same requirement the VF2++
            search makes.
        target_indptr: CSR index pointer of the target graph.

    Returns:
        ``True`` if every edge of the pattern graph is also an edge of the
        target graph between the nodes of the same indices, ``False``
        otherwise (including when the pattern graph has more nodes than the
        target graph).

    """
    num_pattern_nodes = pattern_indptr.shape[0] - 1
    num_target_nodes = target_indptr.shape[0] - 1

    if num_pattern_nodes > num_target_nodes:
        return False

    for u in range(num_pattern_nodes):
        target_neighbors = target_indices[target_indptr[u] : target_indptr[u + 1]]
        for k in range(pattern_indptr[u], pattern_indptr[u + 1]):
            v = pattern_indices[k]
            # The graphs are undirected, so inspect every edge once, from
            # its lower-numbered endpoint.
            if v < u:
                continue
            pos = np.searchsorted(target_neighbors, v)
            if pos >= target_neighbors.size or target_neighbors[pos] != v:
                return False

    return True


def vf2pp_subgraph_isomorphism(
    pattern_indices: np.ndarray,
    pattern_indptr: np.ndarray,
    target_indices: np.ndarray,
    target_indptr: np.ndarray,
    max_attempts: int = 0,
) -> np.ndarray:
    """Find one subgraph isomorphism mapping from pattern graph to target.

    Uses a Numba-optimized VF2++-style search.

    The first pattern node's candidates are explored in parallel
    across CPU cores, each thread with its own budget slice.

    Args:
        pattern_indices: CSR column indices of the pattern graph.
        pattern_indptr: CSR row pointers of the pattern graph.
        target_indices: CSR column indices of the target graph.  Neighbor lists must
            be **sorted** for binary search to work.
        target_indptr: CSR row pointers of the target graph.
        max_attempts: Total maximum number of feasible candidate assignments to
            try (split equally across parallel threads) before giving up.
            0 (default) means unlimited.

    Returns:
        ``mapping[i]`` is the target node matched to pattern node ``i``.
        If no mapping is found, returns an array of all -1.
        If the identity mapping is itself an embedding it is always the one
        returned, so a pattern that already sits on the target is left
        where it is.

    """
    # Convert to Numba-friendly types
    pattern_indices = np.asarray(pattern_indices, dtype=np.int64)
    pattern_indptr = np.asarray(pattern_indptr, dtype=np.int64)
    target_indices = np.asarray(target_indices, dtype=np.int64)
    target_indptr = np.asarray(target_indptr, dtype=np.int64)

    num_pattern_nodes = pattern_indptr.shape[0] - 1
    num_target_nodes = target_indptr.shape[0] - 1

    # Quick rejects
    if num_pattern_nodes == 0:
        return np.empty(0, dtype=np.int64)
    if num_pattern_nodes > num_target_nodes:
        return -np.ones(num_pattern_nodes, dtype=np.int64)

    # Identity fast path.  The VF2++ search returns whichever isomorphism it
    # happens to find first, which need not be the identity even when the
    # identity is valid — it tends to compact the pattern onto the
    # lowest-numbered target nodes.  For the layout passes that would
    # relabel a circuit whose placement is already valid, so check the
    # identity first and hand it back unchanged.  Any embedding is equally
    # good here: all of them need zero SWAPs.
    if is_identity_embedding(pattern_indices, pattern_indptr, target_indices, target_indptr):
        return np.arange(num_pattern_nodes, dtype=np.int64)

    # VF2++ BFS-based ordering: maximises connectivity to already-ordered
    # nodes, processes one connected component at a time.
    order = _vf2pp_bfs_order(pattern_indices, pattern_indptr)

    # Distribute budget across threads.  Each thread (one per candidate
    # target node for the first pattern node) gets an equal share.
    # With 0 (unlimited) we pass 0 through so each thread is also unlimited.
    max_per_thread = max(1, max_attempts // max(1, num_target_nodes)) if max_attempts > 0 else 0

    results = _match_parallel(
        order, pattern_indices, pattern_indptr, target_indices, target_indptr, np.int64(max_per_thread)
    )

    # Pick the first successful mapping (any row that isn't all -1)
    for v in range(num_target_nodes):
        if results[v, 0] != -1:
            return results[v]

    return -np.ones(num_pattern_nodes, dtype=np.int64)


def qc_to_interaction_graph(qc: QuantumCircuit) -> tuple[np.ndarray, np.ndarray]:
    """Extract the qubit-interaction graph of a quantum circuit.

    Builds an undirected graph with one node per circuit qubit, where
    edges represent two-qubit interactions in the circuit.  Single-qubit
    gates, barriers, and measurements are ignored.

    Args:
        qc: The quantum circuit to extract the interaction graph from.

    Returns:
        * CSR index pointer array of the interaction graph
        * CSR indices array of the interaction graph

    """
    G: nx.Graph = nx.Graph()
    G.add_nodes_from(list(range(qc.num_qubits())))

    qubit_dic = {qc.qubits[i]: i for i in range(qc.num_qubits())}

    for instr in qc.data:
        if instr.op.num_qubits <= 1:
            continue

        # Skip barriers — they carry no connectivity information
        if instr.op.name == "barrier":
            continue

        if instr.op.num_qubits == 2:  # noqa: PLR2004
            G.add_edge(qubit_dic[instr.qubits[0]], qubit_dic[instr.qubits[1]])

        else:
            raise Exception(f"Tried to transpile quantum circuit containing a {instr.op.num_qubits}-qubit gate")

    mat = nx.to_scipy_sparse_array(G, format="csr")
    return mat.indptr.astype(np.int64), mat.indices.astype(np.int64)


def vf2pp_layout_and_route(
    qc: QuantumCircuit,
    topology: QPUTopology,
    max_attempts: int = 1_000_000_000,
) -> QuantumCircuit | None:
    """Attempt to place *qc* onto *topology* via subgraph isomorphism.

    Args:
        qc: The circuit to place.
        topology: Target topology in CSR form.
        max_attempts: Total budget of feasible candidate assignments the VF2++
            search may explore (split across parallel threads) before giving up
            and returning None.  0 means unlimited.  Default is 1 000 000 000.
            Because the search is parallelised over first-node candidates,
            wall-clock time scales as ``max_attempts / n_target_nodes``.

    Returns:
        The placed circuit, or None if no isomorphism was found within
        the attempt budget.

    """
    qc_indptr, qc_indices = qc_to_interaction_graph(qc)

    mapping = vf2pp_subgraph_isomorphism(
        qc_indices, qc_indptr, topology.indices, topology.indptr, max_attempts=max_attempts
    )

    is_iso = mapping[0] != -1

    if is_iso:
        qubit_amount = topology.dist_matrix.shape[0]
        n_virtual = qc.num_qubits()

        # Extend the partial mapping (n_virtual → n_physical) to a full
        # permutation by assigning unused physical positions to padding qubits.
        used = {int(mapping[i]) for i in range(n_virtual)}
        unused = [p for p in range(qubit_amount) if p not in used]
        full_mapping = np.empty(qubit_amount, dtype=np.int32)
        full_mapping[:n_virtual] = mapping
        for k, p in enumerate(unused):
            full_mapping[n_virtual + k] = p

        new_qc = qc.copy()

        pad_qubits(new_qc, qubit_amount)

        inv_mapping = invert_permutation(full_mapping)
        new_qc.qubits = [new_qc.qubits[inv_mapping[i]] for i in range(new_qc.num_qubits())]

        return new_qc
    else:
        return None
