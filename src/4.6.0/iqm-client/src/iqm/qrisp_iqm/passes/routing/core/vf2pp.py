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
import networkx as nx
from numba import njit, prange
import numpy as np
from qrisp import QuantumCircuit, Qubit

if TYPE_CHECKING:
    from iqm.qrisp_iqm.passes.routing.core.graph_processing_tools import QPUTopology


# ================================================================
# Helper functions
# ================================================================


@njit(cache=True)
def _degree_array(indptr: np.ndarray) -> np.ndarray:
    """Compute the degree (number of neighbors) for each node.

    Uses a graph represented in CSR (Compressed Sparse Row) format.

    Parameters
    ----------
    indptr : np.ndarray[int64]
        CSR index pointer array, length = n_nodes + 1

    Returns
    -------
    deg : np.ndarray[int64]
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

    Parameters
    ----------
    deg : np.ndarray[int64]
        Degree array of the pattern graph.

    Returns
    -------
    order : np.ndarray[int64]
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

    Parameters
    ----------
    indices, indptr : np.ndarray[int64]
        CSR adjacency of the pattern graph.

    Returns
    -------
    order : np.ndarray[int64]
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

    Parameters
    ----------
    v : int
        Source node index.
    w : int
        Target node index.
    indices : np.ndarray
        CSR column indices of the graph.
    indptr : np.ndarray
        CSR row pointers of the graph.

    Returns
    -------
    bool
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
    indices0: np.ndarray,
    indptr0: np.ndarray,
    indices1: np.ndarray,
    indptr1: np.ndarray,
) -> bool:
    """Check if mapping pattern node to target node is feasible.

    Given the current partial mapping.

    Feasibility checks:
    1. Degree pruning: deg(u) <= deg(v)
    2. Adjacency consistency: for every already-mapped neighbor u_n of u,
       its image v_n must be adjacent to v in the target graph.

    Parameters
    ----------
    u : int
        Candidate node in G0 (pattern).
    v : int
        Candidate node in G1 (target).
    mapping : np.ndarray[int64]
        Current mapping array, -1 for unmapped pattern nodes.
    indices0 : np.ndarray
        CSR column indices of G0.
    indptr0 : np.ndarray
        CSR row pointers of G0.
    indices1 : np.ndarray
        CSR column indices of G1.
    indptr1 : np.ndarray
        CSR row pointers of G1.

    Returns
    -------
    bool : True if feasible, False otherwise.

    """
    deg_u = indptr0[u + 1] - indptr0[u]
    deg_v = indptr1[v + 1] - indptr1[v]
    if deg_u > deg_v:
        return False

    # Check adjacency consistency with already-mapped neighbors
    start = indptr0[u]
    end = indptr0[u + 1]
    for p in range(start, end):
        u_n = indices0[p]
        v_n = mapping[u_n]
        if v_n != -1:
            if not _binary_adjacent(v, v_n, indices1, indptr1):
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
    used1: np.ndarray,
    indices0: np.ndarray,
    indptr0: np.ndarray,
    indices1: np.ndarray,
    indptr1: np.ndarray,
    counter: np.ndarray,
) -> bool:
    """Recursive search for subgraph isomorphism.

    Uses T-set candidate narrowing.

    When the current pattern node ``u`` has at least one already-mapped
    neighbor, only target nodes adjacent to that mapped neighbor are
    tried (T-set narrowing).  This drastically reduces branching on
    grid-like topologies.  If ``u`` has no mapped neighbors (start of a
    new connected component), all unused target nodes are tried.

    Parameters
    ----------
    pos : int
        Current position in the pattern node order.
    order : np.ndarray[int64]
        Pattern node order (BFS heuristic order).
    mapping : np.ndarray[int64]
        Partial mapping from pattern -> target.
    used1 : np.ndarray[uint8]
        Flags for used target nodes.
    indices0 : np.ndarray
        CSR column indices of pattern graph G0.
    indptr0 : np.ndarray
        CSR row pointers of pattern graph G0.
    indices1 : np.ndarray
        CSR column indices of target graph G1.
    indptr1 : np.ndarray
        CSR row pointers of target graph G1.
    counter : np.ndarray[int64], shape (2,)
        counter[0] = remaining attempts (decremented on each candidate
        trial).  When it reaches 0 the search aborts.
        counter[1] = max_attempts limit (0 means unlimited).

    Returns
    -------
    bool
        True if a full mapping is found, otherwise False.

    """
    n0 = order.shape[0]
    n1 = indptr1.shape[0] - 1

    # All pattern nodes matched -> success
    if pos == n0:
        return True

    u = order[pos]
    deg_u = indptr0[u + 1] - indptr0[u]

    # ── Build candidate set via T-set narrowing ──────────────────
    # Find the first already-mapped neighbor of u in the pattern graph.
    # Its image in the target graph defines the T-set: only neighbors
    # of that image are candidates for v.
    anchor = np.int64(-1)
    for p in range(indptr0[u], indptr0[u + 1]):
        u_n = indices0[p]
        if mapping[u_n] != -1:
            anchor = mapping[u_n]
            break

    if anchor != -1:
        # T-set: only try neighbors of the anchor in the target graph
        for p in range(indptr1[anchor], indptr1[anchor + 1]):
            v = indices1[p]
            if used1[v]:
                continue

            deg_v = indptr1[v + 1] - indptr1[v]
            if deg_v < deg_u:
                continue

            if _feasible(u, v, mapping, indices0, indptr0, indices1, indptr1):
                # Check attempt budget
                if counter[1] > 0:
                    counter[0] -= 1
                    if counter[0] <= 0:
                        return False

                mapping[u] = v
                used1[v] = 1

                if _match_recursive(pos + 1, order, mapping, used1, indices0, indptr0, indices1, indptr1, counter):
                    return True

                mapping[u] = -1
                used1[v] = 0
    else:
        # No mapped neighbor — new connected component, try all nodes
        for v in range(n1):
            if used1[v]:
                continue

            deg_v = indptr1[v + 1] - indptr1[v]
            if deg_v < deg_u:
                continue

            if _feasible(u, v, mapping, indices0, indptr0, indices1, indptr1):
                if counter[1] > 0:
                    counter[0] -= 1
                    if counter[0] <= 0:
                        return False

                mapping[u] = v
                used1[v] = 1

                if _match_recursive(pos + 1, order, mapping, used1, indices0, indptr0, indices1, indptr1, counter):
                    return True

                mapping[u] = -1
                used1[v] = 0

    return False


# ================================================================
# Parallel top-level search
# ================================================================


@njit(cache=False, parallel=True)
def _match_parallel(  # noqa: PLR0913
    order: np.ndarray,
    indices0: np.ndarray,
    indptr0: np.ndarray,
    indices1: np.ndarray,
    indptr1: np.ndarray,
    max_attempts_per_thread: int,
) -> np.ndarray:
    """Parallel VF2++ search.

    Each thread tries a different candidate target node for the first
    pattern node in *order* and runs
    _match_recursive from pos=1 with its own budget.  The recursive
    search uses T-set candidate narrowing to limit branching.

    Parameters
    ----------
    order : np.ndarray[int64]
        Pattern node order (VF2++ BFS order).
    indices0 : np.ndarray
        CSR column indices of pattern graph G0.
    indptr0 : np.ndarray
        CSR row pointers of pattern graph G0.
    indices1 : np.ndarray
        CSR column indices of target graph G1.
    indptr1 : np.ndarray
        CSR row pointers of target graph G1.
    max_attempts_per_thread : int64
        Budget per thread (0 = unlimited).

    Returns
    -------
    results : np.ndarray[int64], shape (n1, n0)
        Row *v* holds the mapping found when the first pattern node
        was assigned to target node *v*.  All -1 if that thread failed.

    """
    n0 = order.shape[0]
    n1 = indptr1.shape[0] - 1

    # Each row is one thread's mapping result
    results = -np.ones((n1, n0), dtype=np.int64)

    u0 = order[0]
    deg_u0 = indptr0[u0 + 1] - indptr0[u0]

    for v in prange(n1):
        deg_v = indptr1[v + 1] - indptr1[v]
        if deg_v < deg_u0:
            continue

        # Thread-local state
        mapping = -np.ones(n0, dtype=np.int64)
        used1 = np.zeros(n1, dtype=np.uint8)
        counter = np.array([max_attempts_per_thread, max_attempts_per_thread], dtype=np.int64)

        # Check feasibility for the root assignment (no mapped neighbors yet,
        # so only degree check matters — already done above)
        mapping[u0] = v
        used1[v] = 1

        if n0 == 1 or _match_recursive(1, order, mapping, used1, indices0, indptr0, indices1, indptr1, counter):
            results[v] = mapping

    return results


# ================================================================
# Public interface
# ================================================================


def vf2pp_subgraph_isomorphism(
    indices0: np.ndarray,
    indptr0: np.ndarray,
    indices1: np.ndarray,
    indptr1: np.ndarray,
    max_attempts: int = 0,
) -> np.ndarray:
    """Find one subgraph isomorphism mapping from pattern graph to target.

    Uses a Numba-optimized VF2++-style search on graphs G0 and G1.

    The first pattern node's candidates are explored in parallel
    across CPU cores, each thread with its own budget slice.

    Parameters
    ----------
    indices0, indptr0 : np.ndarray[int64]
        CSR representation of pattern graph G0.
    indices1, indptr1 : np.ndarray[int64]
        CSR representation of target graph G1.
        Neighbor lists must be **sorted** for binary search to work.
    max_attempts : int, optional
        Total maximum number of feasible candidate assignments to try
        (split equally across parallel threads) before giving up.
        0 (default) means unlimited.

    Returns
    -------
    mapping : np.ndarray[int64]
        mapping[i] = node in G1 matched to node i in G0.
        If no mapping is found, returns an array of all -1.

    """
    # Convert to Numba-friendly types
    indices0 = np.asarray(indices0, dtype=np.int64)
    indptr0 = np.asarray(indptr0, dtype=np.int64)
    indices1 = np.asarray(indices1, dtype=np.int64)
    indptr1 = np.asarray(indptr1, dtype=np.int64)

    n0 = indptr0.shape[0] - 1
    n1 = indptr1.shape[0] - 1

    # Quick rejects
    if n0 == 0:
        return np.empty(0, dtype=np.int64)
    if n0 > n1:
        return -np.ones(n0, dtype=np.int64)

    # VF2++ BFS-based ordering: maximises connectivity to already-ordered
    # nodes, processes one connected component at a time.
    order = _vf2pp_bfs_order(indices0, indptr0)

    # Distribute budget across threads.  Each thread (one per candidate
    # target node for the first pattern node) gets an equal share.
    # With 0 (unlimited) we pass 0 through so each thread is also unlimited.
    if max_attempts > 0:
        max_per_thread = max(1, max_attempts // max(1, n1))
    else:
        max_per_thread = 0

    results = _match_parallel(order, indices0, indptr0, indices1, indptr1, np.int64(max_per_thread))

    # Pick the first successful mapping (any row that isn't all -1)
    for v in range(n1):
        if results[v, 0] != -1:
            return results[v]

    return -np.ones(n0, dtype=np.int64)


def qc_to_connectivity(qc: QuantumCircuit) -> tuple[np.ndarray, np.ndarray]:
    """Extract a connectivity graph from a quantum circuit.

    Builds an undirected graph where edges represent two-qubit
    interactions in the circuit.  Single-qubit gates, barriers,
    and measurements are ignored.

    Parameters
    ----------
    qc : QuantumCircuit
        The quantum circuit to extract connectivity from.

    Returns
    -------
    indptr : np.ndarray
        CSR index pointer array of the connectivity graph.
    indices : np.ndarray
        CSR indices array of the connectivity graph.

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

    Parameters
    ----------
    qc : QuantumCircuit
        The circuit to place.
    topology : tuple
        Target topology in CSR form.
    max_attempts : int, optional
        Total budget of feasible candidate assignments the VF2++
        search may explore (split across parallel threads) before
        giving up and returning None.  0 means unlimited.
        Default is 1 000 000 000.  Because the search is parallelised
        over first-node candidates, wall-clock time scales as
        ``max_attempts / n_target_nodes``.

    Returns
    -------
    QuantumCircuit or None
        The placed circuit, or None if no isomorphism was found within
        the attempt budget.

    """
    qc_indptr, qc_indices = qc_to_connectivity(qc)

    mapping = vf2pp_subgraph_isomorphism(
        qc_indices, qc_indptr, topology.indices, topology.indptr, max_attempts=max_attempts
    )

    is_iso = mapping[0] != -1

    if is_iso:
        qubit_amount = topology.dist_matrix.shape[0]
        n_virtual = qc.num_qubits()

        # Extend the partial mapping (n_virtual → n_physical) to a full
        # permutation by assigning unused physical positions to amended qubits.
        used = set(int(mapping[i]) for i in range(n_virtual))
        unused = [p for p in range(qubit_amount) if p not in used]
        full_mapping = np.empty(qubit_amount, dtype=np.int32)
        full_mapping[:n_virtual] = mapping
        for k, p in enumerate(unused):
            full_mapping[n_virtual + k] = p

        new_qc = qc.copy()

        amended_counter = 0
        while new_qc.num_qubits() < qubit_amount:
            new_qc.add_qubit(Qubit("amended_qb_" + str(amended_counter)))
            amended_counter += 1

        inv_mapping = invert_permutation(full_mapping)
        new_qc.qubits = [new_qc.qubits[inv_mapping[i]] for i in range(new_qc.num_qubits())]

        return new_qc
    else:
        return None
