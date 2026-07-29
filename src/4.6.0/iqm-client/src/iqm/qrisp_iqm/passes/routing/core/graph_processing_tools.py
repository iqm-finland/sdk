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

"""Graph processing tools for plasma-sabre routing passes.

This module implements functions that operate on two different graphs:

1. The hardware topology graph :class:`QPUTopology`.
2. The quantum circuit in DAG form :class:`CircuitDAG`.

"""

from __future__ import annotations

from collections import namedtuple
from collections.abc import Callable

from iqm.qrisp_iqm.passes.routing.core.types import data_dtype, qubit_dtype
import networkx as nx
from numba import njit
import numpy as np
from qrisp import PermeabilityGraph, QuantumCircuit
from qrisp.circuit import Instruction
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

# TODO consider @numba.experimental.jitclass
QPUTopology = namedtuple("QPUTopology", ["indptr", "indices", "dist_matrix", "predecessors"])
"""Describes the QPU topology.

All the items are of the type ``np.ndarray[int]``.
The adjacency matrix A of the QPU coupling graph is stored here in
`CSR form <https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)>`__.
The nodes are qubits, and the edges correspond to available native two-qubit gates.

* :attr:`indices` is a 1-D array containing column indices of the nonzero elements of A.
* :attr:`indptr` is a 1-D array, where ``indptr[k]`` is the number of nonzero elements in A above row ``k``.
* :attr:`dist_matrix` is a dense matrix where ``dist_matrix[a, b]`` is the length of the shortest
  path between nodes ``a`` and ``b``.
* :attr:`predecessors` is a dense matrix where ``predecessors[a, b]`` is the index of the next
  node to take when traversing the shortest path from node ``a`` to node ``b``.

:attr:`dist_matrix` and :attr:`predecessors` could be computed on demand,
but since we care for performance and they are required A LOT, we precompute
them and pass them around along the CSR matrix.
"""

CircuitDAG = namedtuple("CircuitDAG", ["indptr", "indices", "instruction_data"])
"""Describes the quantum circuit as a DAG.

All the items are of the type ``np.ndarray[int]``.
The ``m * m`` adjacency matrix A of the DAG is stored here in
`CSR form <https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)>`__.
The nodes are instructions in the circuit, and the edges correspond to qubits the instructions
are acting on. Edge direction represents causality (relative time ordering).

* :attr:`indices` is a 1-D array containing column indices of the nonzero elements of A.
* :attr:`indptr` is a 1-D array, where ``indptr[k]`` is the number of nonzero elements in A above row ``k``.
* :attr:`instruction_data` is a ``shape==(m, 3)`` array describing the instructions:

  * ``instruction_data[i, 0]``: index of the first qubit of the instruction locus
  * ``instruction_data[i, 1]``: index of the second qubit of the instruction locus
    (for one-qubit gates the same as first)
  * ``instruction_data[i, 2]``: index that identifies the :class:`qrisp.Instruction` the node corresponds to.
    The list of Instruction objects is kept outside of the jitted code, and is used to reconstruct the
    :class:`qrisp.QuantumCircuit` object once jit-based compilation has concluded.
"""


def connectivity_to_topology(
    connectivity: list[tuple[int, int]],
) -> QPUTopology:
    """Take a connectivity list and compute the QPU topology.

    The topology graph preserves the original qubit indices from
    ``connectivity`` as node labels.  Qubits that do not participate in
    any edge are included as isolated nodes so that the output circuit's
    qubit indices correspond directly to the original indices — no
    renumbering occurs.

    Args:
        connectivity: Pairs of qubits that are connected to each other.

    Returns:
        QPU topology description.

    """
    if not connectivity:
        return QPUTopology(
            indptr=np.array([0], dtype=np.int32),
            indices=np.array([], dtype=np.int32),
            dist_matrix=np.empty((0, 0), dtype=np.int32),
            predecessors=np.empty((0, 0), dtype=np.int32),
        )

    # Include all qubits from 0 to max_index so that node labels in the
    # graph equal the original qubit indices.  Qubits without any edges
    # appear as isolated nodes — the router's distance matrix marks them
    # as unreachable (-1).
    max_qubit = max(max(a, b) for a, b in connectivity)
    n_qubits = max_qubit + 1

    # Model the connectivity as a networkx Graph to extract the CSR matrix.
    G: nx.Graph = nx.Graph()
    G.add_nodes_from(range(n_qubits))
    G.add_edges_from(connectivity)
    sprs_mat = nx.to_scipy_sparse_array(G, format="csr")

    # Use the scipy shortest path function to compute both distance and
    # predecessors.
    dist_matrix, predecessors = shortest_path(csgraph=sprs_mat, directed=False, return_predecessors=True)

    # Replace inf (unreachable nodes) with a sentinel that survives the
    # integer cast below without triggering "invalid value encountered".
    # Isolated qubits (no edges) remain marked as unreachable.
    dist_matrix[np.isinf(dist_matrix)] = -1

    qdt = qubit_dtype(n_qubits)
    indptr = sprs_mat.indptr.astype(qdt)
    indices = sprs_mat.indices.astype(qdt)
    return QPUTopology(indptr, indices, dist_matrix.astype(qdt), predecessors.astype(qdt))


def convert_qc_to_sparse_dag(
    qc: QuantumCircuit,
    topo_sort: Callable | None = None,
) -> tuple[CircuitDAG, list[Instruction | None]]:
    """Convert a quantum circuit into a DAG.

    The DAG representation used here is the permeability DAG, which allows the
    routing algorithms to leverage permeability-induced commutation relations
    for shortcuts. See the preamble of this file for a full description.

    Args:
        qc: Quantum circuit that is to be converted.
        topo_sort: Function that performs a topological sort on the PermeabilityGraph.
            Must accept a networkx DiGraph and return an iterable of nodes in
            topological order. The node ordering determines which instructions
            end up in which section when sectionized routing is used (prune_dag
            removes nodes by index threshold). By default, nx.topological_sort
            is used.

    Returns:
        * circuit DAG
        * corresponding Qrisp Instruction objects

    """
    if topo_sort is None:
        topo_sort = nx.topological_sort

    # We don't want the transpiler to move around measurements (this seems to
    # deteriorate performance on IQM devices)
    # Therefore we mark the permeability of measurements as False.
    new_qc = qc.copy()

    for i in range(len(new_qc.data)):
        instr = new_qc.data[i]
        if instr.op.name == "measure":
            new_op = instr.op.copy()
            new_op.permeability[0] = False
            new_instr = instr.copy()
            new_instr.op = new_op
            new_qc.data[i] = new_instr

    # Compute the Permeability DAG
    dag = PermeabilityGraph(new_qc, remove_artificials=True)

    # Determine dtypes based on circuit size
    n_physical = qc.num_qubits()
    n_nodes = len(dag.nodes())
    ndt = data_dtype(n_physical, n_nodes)

    # Handle empty circuits (no instructions)
    if n_nodes == 0:
        # Return empty CSR format arrays
        empty_indptr = np.array([0], dtype=ndt)
        empty_indices = np.array([], dtype=ndt)
        empty_instruction_data = np.array([], dtype=ndt).reshape(0, 3)
        return CircuitDAG(empty_indptr, empty_indices, empty_instruction_data), []

    # We need the node indexing of the CSR representation of the DAG to be sorted,
    # otherwise the prune_dag function can do "non-causal" pruning.
    sorted_dag = topo_sort(dag)
    new_dag: nx.DiGraph = nx.DiGraph()
    new_dag.add_nodes_from(sorted_dag)
    new_dag.add_edges_from(dag.edges())

    # Get the CSR representation
    sprs_mat = nx.to_scipy_sparse_array(new_dag, format="csr")
    # Determine the instruction list in order to be able to reconstruct the
    # quantum circuit.
    indptr = sprs_mat.indptr.astype(ndt)
    indices = sprs_mat.indices.astype(ndt)
    instruction_data = []
    instruction_list = []
    node_list = list(new_dag.nodes())

    for i in range(len(node_list)):
        node = node_list[i]

        if node.instr:
            instr = node.instr
            instruction_list.append(instr)

            # Single qubit instruction are indicated by holding the same qubit
            # address twice.
            if instr.op.name == "barrier":
                # Barriers are no-ops that span an arbitrary number of qubits.
                # The DAG already enforces their ordering constraints (barrier
                # permeability is False on all qubits).  In instruction_data
                # we store them as single-qubit-like entries so the router
                # does not attempt to route them — they need no routing.
                # The ordering constraints of the barrier are still enforced
                # via the DAG representation - gates that happen before the
                # barrier are ancestors of the barrier node, so the router
                # is forced to route them first.
                instruction_data.append([qc.qubits.index(instr.qubits[0]), qc.qubits.index(instr.qubits[0]), i])
            elif instr.op.num_qubits == 2:  # noqa: PLR2004
                instruction_data.append([qc.qubits.index(instr.qubits[0]), qc.qubits.index(instr.qubits[1]), i])
            else:
                instruction_data.append([qc.qubits.index(instr.qubits[0]), qc.qubits.index(instr.qubits[0]), i])

        # If the node is a non-instruction node (such as a terminator), the
        # instruction list entry is None.
        else:
            instruction_list.append(None)
            instruction_data.append([qc.qubits.index(node.qubit), qc.qubits.index(node.qubit), i])

    # Turn the instruction data into an array to increase performance
    return CircuitDAG(indptr, indices, np.array(instruction_data, dtype=ndt)), instruction_list


@njit(cache=True)
def find_path_length(index_0: int, index_1: int, topology: QPUTopology) -> int:
    """Return the shortest-path distance between two qubit indices."""
    return topology.dist_matrix[index_0, index_1]


_NO_PATH_SENTINEL = -9999
"""Sentinel object (no qubit may have this index)."""


@njit(cache=True)
def find_path(a: int, b: int, topology: QPUTopology) -> np.ndarray:
    """Reconstruct the shortest path between two qubits.

    Uses the predecessor matrix from the topology tuple to reconstruct
    the shortest path from node ``a`` to node ``b``.
    """
    predecessors = topology.predecessors
    pdt = predecessors.dtype
    path = [np.int16(x) for x in range(0)]
    curr = b
    while curr != _NO_PATH_SENTINEL:
        path.append(np.int16(curr))
        if curr == a:
            break
        curr = predecessors[a, curr]
    if path[-1] != a:
        return np.array([np.int16(x) for x in range(0)], dtype=pdt)  # No path exists
    return np.array(path, dtype=pdt)[::-1]  # Reverse to get path from a to b


@njit(cache=True)
def gen_swap_candidates(
    instruction_data: np.ndarray,
    front_layer_indices: np.ndarray,
    topology: QPUTopology,
    c2a: np.ndarray,
    a2c: np.ndarray,
) -> list[tuple[int, int]]:
    """Generate deduplicated swap candidates for the front layer.

    Returns all swaps that involve a qubit from an instruction in the
    front layer. This
    corresponds to all swaps that involve a qubit that is part of an
    instruction of the front layer.

    Deduplication is performed via a boolean seen-matrix, normalizing each
    swap to (min, max) citizen indices before appending. This avoids
    scoring the same physical swap multiple times.

    Args:
        instruction_data: The instruction data indicating which instructions are globally available.
        front_layer_indices: Indices indicating which entries of ``instruction_data``
            constitute the front layer.
        topology: QPU topology.
        c2a: A "citizan to address" array indicating the current layout.
        a2c: An "address to citizen" array indicating the current layout.

    Returns:
        Potentially viable swaps (deduplicated).

    """
    n = c2a.shape[0]
    seen = np.zeros((n, n), dtype=np.bool_)
    res = []

    topology_indptr = topology[0]
    topology_indices = topology[1]

    # Iterate through the front layer and add any viable swap
    for i in range(len(front_layer_indices)):
        instruction = instruction_data[front_layer_indices[i]]

        address_0 = c2a[instruction[0]]
        address_1 = c2a[instruction[1]]

        # Iterate through the neighboring qubits of both instruction partners
        # and add the swap.
        # We use the fast neighbor iteration property of the CSR format for this.
        # Deduplication: normalize to (min, max) and check the seen matrix.

        for idx in topology_indices[topology_indptr[address_0] : topology_indptr[address_0 + 1]]:
            c_a = a2c[address_0]
            c_b = a2c[idx]
            lo = min(c_a, c_b)
            hi = max(c_a, c_b)
            if not seen[lo, hi]:
                seen[lo, hi] = True
                res.append((c_a, c_b))

        for idx in topology_indices[topology_indptr[address_1] : topology_indptr[address_1 + 1]]:
            c_a = a2c[address_1]
            c_b = a2c[idx]
            lo = min(c_a, c_b)
            hi = max(c_a, c_b)
            if not seen[lo, hi]:
                seen[lo, hi] = True
                res.append((c_a, c_b))

    return res


@njit(cache=True)
def prune_dag(circuit_dag: CircuitDAG, pruning_threshold: int) -> CircuitDAG:
    """Prune instructions below a threshold from an instruction DAG.

    All instructions below the given threshold are removed.

    Args:
        circuit_dag: Quantum circuit as a DAG.
        pruning_threshold: Any node that has an instruction index less than this number will be removed.

    Returns:
        The pruned quantum circuit DAG. Note that the indices array can still contain
        nodes that are pruned, however these indices are negative, which is
        considered to be "invalid"

    """
    # Inherit dtype from the circuit_dag so we stay narrow
    pt = np.int16(pruning_threshold)

    # The pruned instruction data is computed by simply slicing the old
    # instruction data
    instruction_data = circuit_dag.instruction_data[pruning_threshold:]

    # To properly prune the indptr array, we first remove all entries that are
    # below the pruning threshold
    indptr = circuit_dag.indptr[pruning_threshold:].copy()

    # The index pruning threshold is the index of the indices array below which
    # everything is cut.
    index_pruning_threshold = indptr[0]
    indices = circuit_dag.indices[index_pruning_threshold:] - pt

    # To reflect the pruned indices array, we subtract the index pruning
    # threshold from the indptr. It now starts at 0 again like any valid
    # CSR matrix.
    indptr = indptr - index_pruning_threshold

    # Combine the result
    return CircuitDAG(indptr, indices, instruction_data)


@njit(cache=True)
def permute_instruction_dag(circuit_dag: CircuitDAG, permutation: np.ndarray) -> CircuitDAG:
    """Apply a qubit layout permutation to an instruction DAG.

    Args:
        circuit_dag: Quantum circuit as a DAG.
        permutation: An array indicating a permutation of qubit indices.

    Returns:
        Quantum circuit DAG with the permuted layout applied.

    """
    instruction_data = circuit_dag.instruction_data
    permuted_instruction_data = instruction_data.copy()

    # Iterate through the instruction data and apply the permutation
    # to all involved qubit indices.
    for i in range(instruction_data.shape[0]):
        instr = instruction_data[i]
        permuted_instruction_data[i][0] = permutation[instr[0]]
        permuted_instruction_data[i][1] = permutation[instr[1]]
        permuted_instruction_data[i][2] = instr[2]

    return CircuitDAG(circuit_dag.indptr, circuit_dag.indices, permuted_instruction_data)


@njit(cache=True)
def reachability_counts_csr(  # noqa: PLR0912
    indptr: np.ndarray,
    indices: np.ndarray,
    ignore_nodes: np.ndarray,
    predecessors: bool = False,
) -> np.ndarray:
    """Count how many nodes each node can reach in a CSR-encoded DAG.

    By default (``predecessors=False``) this counts **descendants**:
    for every node *u* it returns the number of nodes reachable by
    following edges forward.  When ``predecessors=True`` it counts
    **predecessors** (ancestors) instead, i.e. the number of nodes
    from which *u* is reachable — equivalent to computing descendant
    counts on the transposed graph.

    Nodes listed in ``ignore_nodes`` are excluded from every count
    (useful for skipping single-qubit gates in a circuit DAG).

    Args:
        indptr: CSR row-pointer array of the DAG.
        indices: CSR column-index array of the DAG.
        ignore_nodes: Boolean (0/1) mask — nodes where the entry is 1 are
            excluded from counting.
        predecessors:
            Iff True, count predecessors (ancestors) instead of
            descendants.

    Returns:
        Per-node reachability count (self excluded).

    """
    n = indptr.shape[0] - 1
    # Derive the index dtype from the CSR arrays so we stay consistent
    ndt = indptr.dtype
    # Compute outdegree for topological sort of reversed graph
    outdegree = np.zeros(n, dtype=ndt)
    for u in range(n):
        for i in range(indptr[u], indptr[u + 1]):
            v = indices[i]
            outdegree[u] += 1

    # Topological sort (Kahn's algorithm, forward)
    indegree = np.zeros(n, dtype=ndt)
    for u in range(n):
        for i in range(indptr[u], indptr[u + 1]):
            v = indices[i]
            indegree[v] += 1
    queue = [u for u in range(n) if indegree[u] == 0]
    order = []
    while queue:
        u = queue.pop()
        order.append(u)
        for i in range(indptr[u], indptr[u + 1]):
            v = indices[i]
            indegree[v] -= 1
            if indegree[v] == 0:
                queue.append(v)

    # Reachability bitset: reachability_bits[u, k] is True iff node k is
    # reachable from u (descendants mode) or u is reachable from k
    # (predecessors mode).
    reachability_bits = np.zeros((n, n), dtype=np.bool_)
    reachability_count = np.zeros(n, dtype=ndt)

    if not predecessors:
        # --- Descendant mode ---
        # Traverse in *reverse* topological order so that when we
        # process u, all of its children v already carry their full
        # descendant sets.  We union those sets into u's row.
        for u in order[::-1]:
            reachability_bits[u, u] = True
            for i in range(indptr[u], indptr[u + 1]):
                v = indices[i]
                for k in range(n):
                    if reachability_bits[v, k]:
                        reachability_bits[u, k] = True
    else:
        # --- Predecessor mode ---
        # Traverse in *forward* topological order so that when we
        # process u, its predecessor set is complete.  For every
        # child v of u we propagate u's full predecessor set into v.
        for u in order:
            reachability_bits[u, u] = True
            for i in range(indptr[u], indptr[u + 1]):
                v = indices[i]
                for k in range(n):
                    if reachability_bits[u, k]:
                        reachability_bits[v, k] = True

    not_ignore_nodes = (ignore_nodes + 1) % 2
    for u in range(n):
        # Subtract 1 to exclude the node itself from its own count.
        reachability_count[u] = np.sum(reachability_bits[u] * not_ignore_nodes) - 1

    return reachability_count


def reverse_dag(circuit_dag: CircuitDAG) -> CircuitDAG:
    """Build a backward instruction DAG by reversing the topological order.

    Given a forward CSR instruction DAG
    whose rows are in topological order, return a new DAG where:

    - All dependency edges are reversed (transposed).
    - The row ordering is reversed so that the last forward node becomes
      backward row 0.

    Mathematically the backward adjacency matrix is ``B = P · A^T · P``
    where ``A`` is the forward adjacency matrix and ``P`` is the order-reversal
    permutation.  This ensures that :func:`prune_dag` (which slices the
    first ``k`` rows) peels off gates in backward order.

    Args:
        circuit_dag: CSR instruction DAG.

    Returns:
        CSR instruction DAG with reversed topological order and reversed edges.

    """
    indptr = circuit_dag.indptr
    indices = circuit_dag.indices
    n_nodes = indptr.shape[0] - 1

    if n_nodes == 0:
        return circuit_dag

    # Build scipy CSR from the adjacency part (all-ones data)
    # Inherit the dtype from the forward DAG's indptr so the backward
    # DAG uses the same narrow type.
    dt = indptr.dtype
    fwd_csr = csr_matrix(
        (np.ones(indices.shape[0], dtype=dt), indices, indptr),
        shape=(n_nodes, n_nodes),
    )

    # B = P · A^T · P  (transpose + reverse row/column order)
    rev = np.arange(n_nodes - 1, -1, -1)
    bwd_csr = fwd_csr.T.tocsr()[rev][:, rev].tocsr()

    # Reverse instruction_data ordering (last forward gate → first backward)
    bwd_instruction_data = circuit_dag.instruction_data[::-1].copy()

    return CircuitDAG(
        bwd_csr.indptr.astype(dt),
        bwd_csr.indices.astype(dt),
        bwd_instruction_data,
    )
