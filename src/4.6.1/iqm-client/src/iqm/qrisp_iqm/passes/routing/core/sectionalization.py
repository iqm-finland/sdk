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

"""Sectionalization strategies for SABRE routing.

Background
----------
The SABRE routing algorithm compiles a quantum circuit by greedily selecting
SWAP gates that reduce the distance between operands of upcoming 2-qubit
gates. Compiling the entire circuit in one pass with a single random seed
is fast but sensitive to the initial random choices. *Sectionalization*
mitigates this by splitting the circuit into consecutive sections and
compiling each one independently with multiple random seeds (parallel
trials). The best trial is kept, its final qubit permutation is forwarded
to the next section, and the procedure repeats. This way early mistakes
don't propagate through the whole circuit.

PermeabilityGraph and TerminatorNodes
-------------------------------------
Qrisp's PermeabilityGraph is a DAG representation of the quantum circuit
that encodes *commutation relations* between gates. Two gates that commute
(e.g. two RZ gates on the same qubit) are not forced into a strict order —
they form a "commutation streak". A **TerminatorNode** is a synthetic node
inserted when such a streak ends, i.e. a non-commuting gate arrives at a
qubit that was participating in a commutation chain. Concretely:

- Edges labelled "Z" or "X" connect gates that commute via Z- or
  X-diagonal permeability.
- An "anti_dependency" edge connects streak members to the TerminatorNode
  that marks the streak's end.
- A "neutral" edge connects the TerminatorNode to the first
  non-commuting gate.

TerminatorNodes thus mark *natural breakpoints* in the circuit's algebraic
structure: everything before the terminator commutes freely, everything
after does not.

Why place section boundaries at terminators?
---------------------------------------------
When SABRE compiles a section, it processes DAG nodes in topological order
up to a node-count threshold (``prune_dag`` removes the first *k* nodes).
If a section boundary falls in the middle of a commutation streak, the
routing of the first half cannot exploit the freedom that commutation
provides — the streak is artificially cut. Placing the boundary *after* a
TerminatorNode ensures each section contains complete commutation blocks,
giving the router maximum freedom within each section.

Topological sort strategy — Kahn's algorithm with terminator deferral
----------------------------------------------------------------------
``convert_qc_to_sparse_dag`` builds the PermeabilityGraph, topologically
sorts its nodes, then converts to a CSR adjacency matrix. The node ordering
in this CSR representation is what ``prune_dag`` uses to determine which
nodes belong to which section (it simply slices the first *k* rows). This
means the topological sort order directly controls section composition.

A plain ``nx.topological_sort`` spreads TerminatorNodes evenly throughout
the ordering. This is problematic because section boundaries are snapped
to terminator positions — if terminators are scattered, cuts can land
inside a commutation streak (splitting it across sections) or in regions
that don't reflect the actual routing difficulty.

The custom sort (``_sectionalization_topo_sort``) uses Kahn's algorithm
with a two-queue dequeuing heuristic:

1. Maintain two ready-queues: one for regular (non-terminator) nodes and
   one for TerminatorNodes.
2. **Always dequeue regular nodes first**, as long as any are available.
   This lets the algorithm "drain" all streak members and ordinary gates
   before emitting any terminators.
3. **When only terminators remain** in the ready set, flush all of them
   in a burst.

The effect is that terminators become *concentrated* into clusters rather
than spread out. Each cluster marks a natural breakpoint where several
commutation streaks end simultaneously. Section boundaries placed at
these clusters are guaranteed to sit between complete commutation blocks,
never in the middle of one.

Section boundary selection algorithm
-------------------------------------
Given a target of ``N`` sections (either user-specified or auto-derived via
the heuristic in :mod:`iqm.qrisp_iqm.passes.core.section_count_heuristic`),
we need ``N − 1`` cut points:

1. Scan the ``instruction_list`` for ``None`` entries — these correspond to
   TerminatorNodes in the CSR node order.
2. Compute ``N − 1`` ideal (uniformly spaced) cut positions in
   **2-qubit-gate-count space** — i.e. the positions that split the
   circuit's 2-qubit workload into equal parts. Only 2-qubit gates matter
   for routing difficulty; single-qubit gates are essentially free. A
   circuit whose 2-qubit gates are all in the first half will therefore
   have its midpoint cut at the ~25 % mark of the total node list, not
   at 50 %.
3. Snap each ideal position to the nearest TerminatorNode (without reuse),
   where "nearest" is measured in the cumulative 2-qubit-gate metric.
4. Place the section boundary immediately *after* the chosen terminator
   (so the terminator is the last node of one section, and the next
   section starts with the first non-commuting gate after it).

This keeps sections balanced in routing workload while respecting the
circuit's commutation structure.

API
---
The ``sections`` parameter controls behavior:

- ``0`` (auto): Derive the section count from the circuit's 2-qubit
  critical path length (``floor(cp_2qb / 15) + 1``) and place boundaries
  at TerminatorNode positions. The critical path counts only 2-qubit
  gates on the longest path through the PDAG — single-qubit gates,
  measurements, and TerminatorNodes are ignored since they don't create
  routing constraints.
- ``1``: No sectionalization — compile in a single pass.
- ``N > 1``: Exactly *N* sections with boundaries at TerminatorNode
  positions.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from iqm.qrisp_iqm.passes.routing.core.graph_processing_tools import CircuitDAG, convert_qc_to_sparse_dag
from iqm.qrisp_iqm.passes.routing.core.section_count_heuristic import auto_section_count
from iqm.qrisp_iqm.passes.routing.core.types import index_dtype
import networkx as nx
import numpy as np
from qrisp import QuantumCircuit
from qrisp.permeability import TerminatorNode

if TYPE_CHECKING:
    from qrisp.circuit import Instruction

# ---------------------------------------------------------------------------
# Section boundaries returned alongside the CSR DAG
# ---------------------------------------------------------------------------
# prepare_dag_and_sections returns section_lengths (np.ndarray | None)
# directly as its third element.


# ---------------------------------------------------------------------------
# Sectionalization-aware topological sort
# ---------------------------------------------------------------------------


def _sectionalization_topo_sort(G: nx.DiGraph) -> list:  # noqa: PLR0912
    """Kahn's-algorithm topological sort that concentrates TerminatorNodes.

    The dequeuing heuristic is:

    1. Always dequeue **non-terminator** nodes from the ready set first,
       as long as any are available.
    2. When only TerminatorNodes remain in the ready set, dequeue **all**
       of them in one burst.

    This produces a topological order where terminators are clustered
    together rather than spread out.  Section boundaries can then be
    placed at these terminator clusters, ensuring that commutativity
    streaks are never cut in the middle.

    Parameters
    ----------
    G : nx.DiGraph
        A :class:`PermeabilityGraph` (or any DAG whose nodes may include
        ``TerminatorNode`` instances).

    Returns
    -------
    list
        Nodes of *G* in topological order with terminators concentrated.

    """
    in_degree = {n: 0 for n in G}
    for u, v in G.edges():
        in_degree[v] += 1

    # Split initial zero-in-degree nodes into two queues
    regular_q: deque = deque()
    terminator_q: deque = deque()
    for n, d in in_degree.items():
        if d == 0:
            if isinstance(n, TerminatorNode):
                terminator_q.append(n)
            else:
                regular_q.append(n)

    result = []

    while regular_q or terminator_q:
        # Phase 1: drain all available non-terminator nodes
        while regular_q:
            node = regular_q.popleft()
            result.append(node)
            for succ in G.successors(node):
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    if isinstance(succ, TerminatorNode):
                        terminator_q.append(succ)
                    else:
                        regular_q.append(succ)

        # Phase 2: no regular nodes available — flush all terminators
        while terminator_q and not regular_q:
            node = terminator_q.popleft()
            result.append(node)
            for succ in G.successors(node):
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    if isinstance(succ, TerminatorNode):
                        terminator_q.append(succ)
                    else:
                        regular_q.append(succ)

    return result


# ---------------------------------------------------------------------------
# Terminator-based variable sections
# ---------------------------------------------------------------------------


def _compute_variable_section_lengths(instruction_list: list, n_sections: int) -> np.ndarray | None:
    """Compute variable-length section boundaries placed at terminator positions.

    Given the *instruction_list* from :func:`convert_qc_to_sparse_dag` (where
    ``None`` entries mark TerminatorNodes), pick a subset of terminator
    positions as section boundaries.  Ideal (uniform) cut positions in
    node-index space are computed first, then each is snapped to the nearest
    terminator so that section sizes stay balanced even when terminators
    cluster.

    Parameters
    ----------
    instruction_list : list
        Instruction list from ``convert_qc_to_sparse_dag``.  Entries are
        ``Instruction`` objects or ``None`` (for terminator nodes).
    n_sections : int
        Target number of sections.

    Returns
    -------
    np.ndarray | None
        Int32 array of section lengths, or ``None`` if no terminators exist
        or ``n_sections <= 1``.

    """
    _TWO_QUBITS = 2
    if n_sections <= 1:
        return None

    total = len(instruction_list)

    # Terminator positions are None entries in instruction_list
    term_positions = [i for i, instr in enumerate(instruction_list) if instr is None]

    if len(term_positions) == 0:
        return None

    n_cuts = n_sections - 1

    # Build a cumulative 2-qubit-gate count at each node index.
    # Only 2-qubit gates matter for routing difficulty, so we place
    # section boundaries to split the *2-qubit workload* evenly —
    # not the raw node count.
    cum_2qb = np.zeros(total, dtype=index_dtype(total))
    running = 0
    for i, instr in enumerate(instruction_list):
        if instr is not None and len(instr.qubits) == _TWO_QUBITS:
            running += 1
        cum_2qb[i] = running
    total_2qb = running

    if total_2qb == 0:
        # No 2-qubit gates → sectionalization has no effect
        return None

    # Ideal cut positions in 2-qubit-gate-count space
    ideal_2qb_cuts = np.linspace(
        total_2qb / n_sections,
        total_2qb * (n_sections - 1) / n_sections,
        n_cuts,
    )

    # For each terminator, look up its cumulative 2QB count
    term_arr = np.array(term_positions)
    term_2qb = cum_2qb[term_arr]

    # Snap each ideal cut to the nearest (unused) terminator in 2QB space
    chosen = []
    used = set()
    for target in ideal_2qb_cuts:
        dists = np.abs(term_2qb - target)
        order = np.argsort(dists)
        for idx in order:
            if idx not in used:
                chosen.append(term_positions[idx])
                used.add(idx)
                break
    chosen.sort()

    # Convert chosen positions to section lengths.
    # The cut is placed *after* the terminator node.
    boundaries = [c + 1 for c in chosen]

    section_lengths = []
    prev = 0
    for b in boundaries:
        length = b - prev
        if length > 0:
            section_lengths.append(length)
            prev = b
    remaining = total - prev
    if remaining > 0:
        section_lengths.append(remaining)
    elif not section_lengths:
        return None

    return np.array(section_lengths, dtype=index_dtype(total))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def prepare_dag_and_sections(
    qc: QuantumCircuit,
    *,
    sections: int = 0,
) -> tuple[CircuitDAG, list[Instruction | None], np.ndarray]:
    """Convert a quantum circuit into a CSR instruction DAG.

    Computes sectionalization information for the router.

    This is the single entry point that combines DAG construction (via
    :func:`convert_qc_to_sparse_dag`) with section-boundary computation.
    The returned ``section_lengths`` array tells the router how to split
    the compilation into sections.

    Parameters
    ----------
    qc : QuantumCircuit
        The circuit to prepare.
    sections : int, optional
        ``0`` (default) for automatic CP-based sections
        (``floor(cp_2qb / 15) + 1``),
        ``1`` for no sectionalization, ``N > 1`` for exactly *N*
        terminator-based sections.

    Returns
    -------
    forward_dag : tuple
        CSR instruction DAG for the forward circuit.
    forward_instruction_list : list
        Instruction list (entries are ``Instruction`` or ``None``).
    section_lengths : np.ndarray
        Int32 array of per-section node counts.  When sectionalization
        is inactive (``sections=1`` or no terminators found), this is a
        single-element array covering the entire DAG.

    """
    # Use sectionalization-aware Kahn's sort when sectioning is active,
    # otherwise fall through to the default (nx.topological_sort).
    topo_sort = _sectionalization_topo_sort if sections != 1 else None

    # --- Build forward CSR DAG ---
    forward_dag, forward_instruction_list = convert_qc_to_sparse_dag(qc, topo_sort=topo_sort)

    # Total DAG node count (indptr length minus 1)
    n_nodes = forward_dag.indptr.shape[0] - 1

    # --- No sectionalization ---
    if sections == 1:
        return (
            forward_dag,
            forward_instruction_list,
            np.array([n_nodes], dtype=index_dtype(n_nodes)),
        )

    # --- Determine target section count ---
    if sections > 1:
        n_sections = sections
    else:
        # Automatic: derive from 2-qubit critical path length.
        # See section_count_heuristic.py for the full rationale.
        n_sections = auto_section_count(forward_dag, forward_instruction_list)

    # --- Terminator-based variable sections ---
    var_lengths = _compute_variable_section_lengths(forward_instruction_list, n_sections)
    if var_lengths is not None:
        return (
            forward_dag,
            forward_instruction_list,
            var_lengths,
        )

    # No terminators found or n_sections <= 1 — single section
    return (
        forward_dag,
        forward_instruction_list,
        np.array([n_nodes], dtype=index_dtype(n_nodes)),
    )
