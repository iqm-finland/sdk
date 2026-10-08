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

"""Automatic section-count heuristic for SABRE routing.

This module determines *how many sections* a circuit should be split into
when sectionalized routing is used.  The answer is derived from the
**2-qubit critical path** (CP₂) of the circuit's PermeabilityGraph
(PDAG).

Formula
-------
::

    n_sections = floor(CP₂ / 15) + 1

where CP₂ is the longest path through the PDAG counting only 2-qubit
gate nodes (single-qubit gates, measurements, and TerminatorNodes are
weighted 0).


What is the critical path and why does it matter?
--------------------------------------------------
The PermeabilityGraph is a DAG that encodes data-dependency and
commutation relations between circuit gates.  The **critical path** is
the longest directed path through this DAG — it represents the maximum
number of *sequentially dependent* routing decisions that SABRE must
make without any reordering freedom.

Circuits with a short critical path (e.g. QAOA, where almost everything
commutes) give SABRE enormous scheduling flexibility: it can reorder
gates freely to avoid congestion.  Circuits with a long critical path
(e.g. Grover, qwalk) force SABRE into a rigid sequence where early
mistakes cascade through the whole compilation.

Sectionalization helps precisely when the critical path is long: by
resetting the random-seed search at each section boundary, mistakes in
one section cannot propagate beyond it.  For short-CP circuits,
sectionalization adds overhead without benefit.


Why count only 2-qubit gates?
-----------------------------
Only 2-qubit gates create routing constraints — they require their two
operand qubits to be physically adjacent.  Single-qubit gates, barriers,
measurements, and TerminatorNodes (synthetic PDAG nodes marking
commutation-streak boundaries) inflate the naive critical path without
adding routing difficulty.

For example, ``cdkm_ripple_carry_adder@20`` has CP_all = 192 when
counting all nodes, but CP₂ = 109 when counting only 2-qubit gates.
The naive metric suggested this circuit barely benefits from sections
(CP < 200 threshold from naive analysis), while the corrected metric
correctly places it well above the "sections help" threshold.

This insight was discovered when early sweep experiments targeting
CP-per-section values of 20–50 (using the naive all-node CP) produced
optimal section counts that seemed far too aggressive for small
circuits.  Re-examining the CP calculation revealed that 1-qubit gates,
measurements, and terminators were inflating the numbers by ~2×.  After
switching to the 2-qubit-only metric, the optimal CP₂/section
converged to ~10–15 across diverse benchmarks.

Implementation
--------------
The CP₂ computation uses a numba-JIT-compiled Kahn's-algorithm
topological sort with dynamic programming on the CSR adjacency list
that ``convert_qc_to_sparse_dag`` already produces — no redundant
graph construction is needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import njit
import numpy as np

if TYPE_CHECKING:
    from iqm.qrisp_iqm.passes.routing.core.graph_processing_tools import CircuitDAG

# ---------------------------------------------------------------------------
# Numba-accelerated weighted longest path on CSR DAG
# ---------------------------------------------------------------------------


@njit(cache=True)
def _weighted_longest_path_csr(indptr: np.ndarray, indices: np.ndarray, weights: np.ndarray, n: int) -> int:
    """Kahn's topo-sort + DP weighted longest path on a CSR adjacency list.

    Parameters
    ----------
    indptr : np.ndarray
        CSR row pointers (dtype inherited from call site).
    indices : np.ndarray
        CSR column indices (successor nodes).
    weights : np.ndarray
        Per-node weight (1 for nodes that count, 0 otherwise).
    n : int
        Number of nodes.

    Returns
    -------
    int
        Maximum sum-of-weights along any path in the DAG.

    """
    # Inherit the narrow dtype from the CSR arrays
    dt = indptr.dtype
    in_deg = np.zeros(n, dtype=dt)
    for i in range(indptr.shape[0] - 1):
        for j in range(indptr[i], indptr[i + 1]):
            in_deg[indices[j]] += 1

    queue = np.empty(n, dtype=dt)
    head = 0
    tail = 0
    dist = np.zeros(n, dtype=dt)
    for i in range(n):
        dist[i] = weights[i]
        if in_deg[i] == 0:
            queue[tail] = i
            tail += 1

    while head < tail:
        u = queue[head]
        head += 1
        for j in range(indptr[u], indptr[u + 1]):
            v = indices[j]
            cand = dist[u] + weights[v]
            dist[v] = max(dist[v], cand)
            in_deg[v] -= 1
            if in_deg[v] == 0:
                queue[tail] = v
                tail += 1

    max_d = 0
    for i in range(n):
        max_d = max(max_d, dist[i])
    return max_d


# ---------------------------------------------------------------------------
# 2-qubit critical path
# ---------------------------------------------------------------------------


def compute_cp_2qb(forward_dag: CircuitDAG, instruction_list: list) -> int:
    """Compute the routing-relevant critical path length.

    The longest path through the instruction DAG counting only 2-qubit gate nodes.

    Single-qubit gates, measurements, and TerminatorNodes contribute
    weight 0 — they don't create routing constraints.

    Parameters
    ----------
    forward_dag : tuple
        Circuit DAG as returned by :func:`convert_qc_to_sparse_dag`.
    instruction_list : list
        Instruction list from ``convert_qc_to_sparse_dag``.  Entries are
        ``Instruction`` objects or ``None`` (for terminator nodes).

    Returns
    -------
    int
        The 2-qubit critical path length.

    """
    indptr = forward_dag.indptr
    n = len(instruction_list)
    dt = indptr.dtype
    weights = np.zeros(n, dtype=dt)
    for i, instr in enumerate(instruction_list):
        if instr is not None and len(instr.qubits) == _TWO_QUBITS:
            weights[i] = 1
    return int(_weighted_longest_path_csr(indptr, forward_dag.indices, weights, n))


# ---------------------------------------------------------------------------
# Section count heuristic
# ---------------------------------------------------------------------------

# Target CP₂ per section.  Derived from sweep experiments on mqt.bench
# circuits (cdkm, randomcircuit, grover, rg_qft_multiplier, qwalk) routed
# on a 7×7 grid.  The sweet spot across these benchmarks is ~10–20;
# 15 is a balanced middle ground.
_TWO_QUBITS = 2

CP_2QB_PER_SECTION = 15


def auto_section_count(forward_dag: CircuitDAG, instruction_list: list) -> int:
    """Determine the number of sections automatically.

    Derived from the 2-qubit critical path.

    Parameters
    ----------
    forward_dag : tuple
        CSR instruction DAG from :func:`convert_qc_to_sparse_dag`.
    instruction_list : list
        Corresponding instruction list.

    Returns
    -------
    int
        Recommended number of sections (≥ 1).

    """
    cp_2qb = compute_cp_2qb(forward_dag, instruction_list)
    return int(np.floor(cp_2qb / CP_2QB_PER_SECTION)) + 1
