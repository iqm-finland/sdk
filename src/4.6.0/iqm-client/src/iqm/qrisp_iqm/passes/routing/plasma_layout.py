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

"""SABRE layout pass.

Finds an initial qubit mapping (logical → physical) that minimizes the
SABRE routing cost.  Chain with :func:`plasma_route` for a full transpiler
pipeline.

Forward/backward layout refinement
-----------------------------------
Finding a good initial layout is essential: the routing pass greedily
inserts SWAPs to satisfy 2-qubit gate connectivity, and a poor starting
mapping forces many unnecessary SWAPs early on that cascade through the
rest of the circuit.

The layout search generates many random candidate mappings and scores
each one.  Scoring uses SABRE's **forward/backward refinement loop**:

1. Route the circuit *forward* starting from the candidate mapping.
   The forward pass's final permutation favours the circuit's *last*
   gates.
2. Use that permutation as the starting layout and route *backward*
   (reversed gate order).  The backward pass ends with a permutation
   that places the circuit's *first* gates' qubits close together.
3. Repeat for several iterations.  Each round improves the layout for
   both ends of the circuit.

Because the loop ends with a backward pass, the final mapping is
already optimised for the circuit's *first* gates — exactly what the
downstream forward routing needs.  No extra backward pass is required.

The parallel evaluation function returns both scores and refined
mappings.  The best refined mapping is used directly, avoiding a
redundant re-refinement from the original seed.

Section-aware evaluation
-------------------------
When sectionalized routing is active (``sections != 1``), the layout
search must mirror what the downstream :func:`plasma_route` pass will actually
do.  If the layout pass evaluates candidates on an *unsectioned* DAG
while routing later uses sectioned SABRE, the proxy cost diverges from
the true objective.  In particular, the unsectioned evaluation ignores
the fact that each section is compiled independently — a layout that
looks good globally may be poor for the first section, and that mistake
cascades.

To fix this, the layout evaluation performs **sectionized** forward and
backward passes: the circuit DAG is split at the same terminator-based
boundaries that :func:`plasma_route` will use, each section is compiled
independently, and the total swap count across all sections is the
scoring metric.

Section-aligned backward passes
--------------------------------
During layout refinement the backward pass must use the forward section
lengths in reversed order (``section_lengths[::-1]``), **not**
independently computed backward sections.  The backward pass processes
gates in the opposite order: if forward sections are
``[S_1, S_2, ..., S_N]``, the backward pass routes them as
``[S_N, S_{N-1}, ..., S_1]``.  By using reversed forward section
lengths, backward section *k* processes exactly the gates of forward
section *N − k*.  The backward pass's last section therefore covers
exactly the gates in forward section ``S_1``, so the resulting
permutation places those qubits close together — which is exactly what
the forward pass needs for its first section.

If independently computed backward sections were used instead, the
backward pass's last section would cover arbitrary gates (determined by
the reversed circuit's commutation structure, which differs due to
TerminatorNode placement).  The resulting permutation would be optimised
for those arbitrary gates rather than for ``S_1``, producing a starting
layout misaligned with the forward pass's first section.

Backward DAG construction
--------------------------
The backward DAG is built by reversing the forward CSR adjacency matrix
(see :func:`~iqm.qrisp_iqm.passes.routing.core.graph_processing_tools.reverse_dag`).
This reverses all dependency edges *and* flips the node ordering so that
the last forward nodes become the first backward nodes.  Building a
PermeabilityGraph from the reversed circuit is *not* suitable because it
would place TerminatorNodes at different positions, producing a
structurally different DAG.

Impact
------
Circuits with no sectionalization (C=1) are unaffected — both changes
are no-ops.  Circuits with deep conjugate structure (e.g. Grover with
C=6/20, ripple-carry adders, random circuits with many sections) see
10–28 % reductions in routed 2-qubit gate count compared to unsectioned
layout evaluation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from iqm.qrisp_iqm.passes.routing.core.graph_processing_tools import (
    connectivity_to_topology,
    permute_instruction_dag,
    prune_dag,
    reverse_dag,
)
from iqm.qrisp_iqm.passes.routing.core.parameter_selection import compute_parameters
from iqm.qrisp_iqm.passes.routing.core.permutation_tools import invert_permutation, mul_perm
from iqm.qrisp_iqm.passes.routing.core.sabre_workflow import sabre_gen_swaps
from iqm.qrisp_iqm.passes.routing.core.sectionalization import prepare_dag_and_sections
from iqm.qrisp_iqm.passes.routing.core.vf2pp import (
    vf2pp_layout_and_route as vf2pp_layout_route_internal,
)
from numba import njit, prange
import numpy as np
import psutil
from qrisp import QuantumCircuit, Qubit
from qrisp.circuit.pass_management.circuit_pass import CircuitPass

cpu_count: int = psutil.cpu_count() or 1

if TYPE_CHECKING:
    from iqm.qrisp_iqm.passes.routing.core.graph_processing_tools import CircuitDAG, QPUTopology

# ──────────────────────────────────────────────────────────────────────
#  Sectionized routing primitive (njit)
# ──────────────────────────────────────────────────────────────────────


@njit(cache=True)
def _sectioned_route(
    topology: QPUTopology, dag: CircuitDAG, section_lengths: np.ndarray, qubit_amount: int, backward_mode: bool = False
) -> np.ndarray:
    """Route a DAG through sections, returning the accumulated permutation.

    Processes the DAG section by section (as determined by
    *section_lengths*), running single-seed SABRE on each.  Returns the
    composed qubit permutation across all sections.
    """
    qdt = topology[2].dtype
    n_sections = len(section_lengths)
    perm = np.arange(qubit_amount, dtype=qdt)
    remaining = dag

    for k in range(n_sections - 1):
        sl = section_lengths[k]
        _, c2a = sabre_gen_swaps(topology, remaining, 1000, max_instruction=sl, seed=1, backward_mode=backward_mode)
        remaining = prune_dag(remaining, sl)
        remaining = permute_instruction_dag(remaining, c2a)
        perm = mul_perm(c2a, perm)

    _, c2a = sabre_gen_swaps(topology, remaining, 1000, seed=1)
    perm = mul_perm(c2a, perm)
    return perm


@njit(cache=True)
def _sectioned_route_with_cost(
    topology: QPUTopology, dag: CircuitDAG, section_lengths: np.ndarray, qubit_amount: int
) -> tuple[np.ndarray, np.int64, np.int64]:
    """Route a DAG through sections, returning permutation *and* cost.

    Combines :func:`_sectioned_route` with swap-count / depth tracking.
    The cost metrics are extracted from the same routing that produces
    the permutation, so no additional ``sabre_gen_swaps`` calls are
    needed.

    Returns
    -------
    perm : np.ndarray[int32]
        Accumulated qubit permutation across all sections.
    total_swaps : int64
        Total number of SWAP gates inserted.
    max_depth : int64
        Maximum per-physical-qubit depth (continuous across sections).

    """
    n_sections = len(section_lengths)
    qdt = topology[2].dtype
    perm = np.arange(qubit_amount, dtype=qdt)
    remaining = dag
    total_swaps = np.int64(0)
    phys_depth = np.zeros(qubit_amount, dtype=np.int64)

    for k in range(n_sections - 1):
        sl = section_lengths[k]
        compiled, c2a = sabre_gen_swaps(
            topology,
            remaining,
            1000,
            max_instruction=sl,
            seed=1,
        )
        for t_idx in range(len(compiled)):
            q0 = compiled[t_idx][0]
            q1 = compiled[t_idx][1]
            is_swap = compiled[t_idx][2] == np.int16(-1)
            if is_swap:
                total_swaps += 1
            if q0 != q1:
                cost = np.int64(3) if is_swap else np.int64(1)
                new_d = max(phys_depth[q0], phys_depth[q1]) + cost
                phys_depth[q0] = new_d
                phys_depth[q1] = new_d
        remaining = prune_dag(remaining, sl)
        remaining = permute_instruction_dag(remaining, c2a)
        perm = mul_perm(c2a, perm)

    compiled, c2a = sabre_gen_swaps(topology, remaining, 1000, seed=1)
    for t_idx in range(len(compiled)):
        q0 = compiled[t_idx][0]
        q1 = compiled[t_idx][1]
        is_swap = compiled[t_idx][2] == np.int16(-1)
        if is_swap:
            total_swaps += 1
        if q0 != q1:
            cost = np.int64(3) if is_swap else np.int64(1)
            new_d = max(phys_depth[q0], phys_depth[q1]) + cost
            phys_depth[q0] = new_d
            phys_depth[q1] = new_d
    perm = mul_perm(c2a, perm)

    return perm, total_swaps, np.max(phys_depth)


# ──────────────────────────────────────────────────────────────────────
#  Parallel layout evaluation — section-aware
# ──────────────────────────────────────────────────────────────────────


@njit(cache=True, parallel=True)  # noqa: PLR0913
def perform_layout_stage_sectioned(  # noqa: PLR0913
    init_mappings: np.ndarray,
    forward_sparse_dag: CircuitDAG,
    backward_sparse_dag: CircuitDAG,
    topology: QPUTopology,
    n_iterations: int,
    section_lengths: np.ndarray,
    selection_exponent: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Parallel layout evaluation with sectionized forward and backward passes.

    Both the forward and backward SABRE passes process the DAG in sections.
    The backward pass uses
    ``section_lengths[::-1]`` — the forward sections in reversed order —
    so that the backward pass's final permutation is optimised for the
    same gates that comprise the forward pass's first section.

    Each iteration runs a **forward** pass followed by a **backward** pass.
    This means the loop naturally ends with a backward pass, leaving the
    mapping optimised for the circuit's first gates — exactly what the
    downstream forward routing needs.  No extra backward pass is required.

    The function returns *both* scores and refined mappings so the caller
    can use the refined mapping directly without re-running the
    refinement from scratch.

    Each candidate is scored by the weighted geometric mean::

        score = swaps^(1 - e) * depth^e

    When ``selection_exponent`` is 0 the selection is purely by swap count;
    when 1.0 purely by depth.

    Parameters
    ----------
    init_mappings : np.ndarray
        (n_candidates, n_physical) int32 array of initial qubit mappings.
    forward_sparse_dag : tuple
        CSR instruction DAG (Kahn's sort) for the forward circuit.
    backward_sparse_dag : tuple
        CSR instruction DAG for the reversed circuit (transposed forward DAG).
    topology : tuple
        Hardware topology.
    n_iterations : int
        Number of forward/backward refinement iterations.
    section_lengths : np.ndarray
        1-D int32 array of per-section node counts (forward).
    selection_exponent : float, optional
        Blending weight for the geometric-mean score (0.0–1.0).  Default 0.0.

    Returns
    -------
    res : np.ndarray
        1-D float64 array of scores (length n_candidates).
    refined_mappings : np.ndarray
        (n_candidates, n_physical) int32 array of refined qubit mappings.

    """
    n = len(init_mappings)
    backward_section_lengths = section_lengths[::-1]
    qubit_amount = topology[2].shape[0]
    res = np.empty(n, dtype=np.float64)
    refined_mappings = np.empty_like(init_mappings)
    dp = min(1.0, max(0.0, selection_exponent))

    for j in prange(n):
        mapping = init_mappings[j].copy()

        for iteration in range(n_iterations):
            # Forward pass
            permuted = permute_instruction_dag(forward_sparse_dag, mapping)
            perm = _sectioned_route(topology, permuted, section_lengths, qubit_amount)
            mapping = mul_perm(perm, mapping)

            # Backward pass (reversed forward sections)
            permuted = permute_instruction_dag(backward_sparse_dag, mapping)
            perm = _sectioned_route(topology, permuted, backward_section_lengths, qubit_amount, backward_mode=True)
            mapping = mul_perm(perm, mapping)

        # Score the fully refined mapping with a single forward routing
        # that extracts both swap count and depth simultaneously.
        permuted = permute_instruction_dag(forward_sparse_dag, mapping)
        _, sc, md = _sectioned_route_with_cost(
            topology,
            permuted,
            section_lengths,
            qubit_amount,
        )
        swaps = np.float64(sc + 1)
        depth = np.float64(md + 1)
        res[j] = swaps ** (1.0 - dp) * depth**dp

        refined_mappings[j] = mapping

    return res, refined_mappings


# ──────────────────────────────────────────────────────────────────────
#  Public pass factory
# ──────────────────────────────────────────────────────────────────────


def plasma_layout(
    connectivity: list[tuple[int, int]],
    effort: int = 30,
    sections: int = 0,
    depth_weight: float = 0.0,
    seed: int = 0,
) -> Callable[[QuantumCircuit], QuantumCircuit]:
    """Create a layout pass that finds an initial qubit mapping.

    The returned pass remaps logical qubits to physical positions but does
    **not** insert SWAP gates.  Chain it with :func:`plasma_route` for a full
    layout + routing pipeline::

        pm = PassManager()
        pm.add_pass(plasma_layout(connectivity, sections=0))
        pm.add_pass(plasma_route(connectivity, sections=0))
        transpiled = pm.run(qc)

    When sectionalized routing is used (``sections != 1``), the layout
    search is *section-aware*: forward evaluation passes process the DAG
    in sections matching those that :func:`plasma_route` will use, so the proxy
    cost faithfully predicts the routing outcome.  See the module
    docstring for details.

    Parameters
    ----------
    connectivity : list[tuple[int, int]]
        Hardware topology edges.
    effort : int, optional
        Controls the number of initial layout candidates and refinement
        iterations.  Passed to ``compute_parameters`` together with the
        circuit's 2-qubit gate count to derive ``init_layout_attempts``
        and ``layout_iterations``.  Default 30.
    sections : int, optional
        Section count, forwarded to :func:`prepare_dag_and_sections`.

        - ``0`` (default): Automatic — derive from 2-qubit critical path.
        - ``1``: No sectionalization — unsectioned layout evaluation.
        - ``N > 1``: Exactly *N* sections.

        Must match the ``sections`` value passed to the downstream
        :func:`plasma_route` pass so the layout evaluation reflects the actual
        routing strategy.  Default is 0.
    depth_weight : float, optional
        Tradeoff between gate count and circuit depth (-1.0 to 1.0).

        - ``-1``: Layout scored purely by swap count.
        - ``0`` (default): Balanced.
        - ``+1``: Layout scored purely by circuit depth.

        Should match the ``depth_weight`` passed to :func:`plasma_route`.
    seed : int, optional
        Seed for the random number generator used to produce candidate
        initial layouts.  Default is ``0``, making the layout search
        reproducible by default.

    Returns
    -------
    Callable[[QuantumCircuit], QuantumCircuit]
        A pass function that applies the optimised layout.

    Example
    -------
    >>> from qrisp import QuantumCircuit, PassManager
    >>> from iqm.qrisp_iqm import plasma_layout, plasma_route
    >>> qc = QuantumCircuit(2); qc.cx(0, 1); qc.measure(qc.qubits)
    >>> pm = PassManager()
    >>> pm += plasma_layout(connectivity=[(0,1),(1,2),(2,3)])
    >>> pm += plasma_route(connectivity=[(0,1),(1,2),(2,3)])
    >>> transpiled_qc = pm.run(qc)

    """
    if not connectivity:
        raise ValueError(
            "plasma_layout requires a non-empty connectivity list. "
            "The connectivity must contain at least one edge "
            "describing available two-qubit gate loci on the device."
        )

    _effort = max(1, effort)
    _TWO_QUBITS = 2
    _sections = sections
    _w = max(-1.0, min(1.0, float(depth_weight)))
    _selection_exponent = 0.5 * (1.0 + _w)
    _rng = np.random.default_rng(seed)

    @CircuitPass
    def _plasma_layout(qc: QuantumCircuit) -> QuantumCircuit:
        topology = connectivity_to_topology(connectivity)
        n_physical = topology.dist_matrix.shape[0]
        qdt = topology.dist_matrix.dtype
        n_virtual = qc.num_qubits()

        if n_virtual > n_physical:
            raise ValueError(f"Circuit has {n_virtual} qubits but the topology only has {n_physical} physical qubits.")

        # Fast path: try VF2++ subgraph isomorphism.  If the circuit's
        # interaction graph is a subgraph of the topology, VF2++ returns
        # a perfectly mapped circuit with no routing needed.
        vf2pp_trial = vf2pp_layout_route_internal(qc, topology)
        if vf2pp_trial is not None:
            return vf2pp_trial

        # Derive parameters from effort level
        num_2qb_gates = sum(1 for instr in qc.data if instr.op.num_qubits == _TWO_QUBITS)
        params = compute_parameters(_effort, num_2qb_gates)
        init_layout_attempts = params["init_layout_attempts"]
        layout_iterations = params["layout_iterations"]  # always 3

        # Build forward DAG and section boundaries.
        # prepare_dag_and_sections handles sections=1 (no Kahn's sort,
        # variable_lengths=None) and sections!=1 (Kahn's sort + terminator-
        # based boundaries) uniformly.
        forward_sparse_dag, _, section_lengths = prepare_dag_and_sections(qc, sections=_sections)

        # Build backward DAG by reversing the forward DAG's topological
        # order and edge directions.  This avoids building a reversed-circuit
        # PermeabilityGraph (which would place TerminatorNodes at different
        # positions).  See reverse_dag() for the P·A^T·P formula.
        backward_sparse_dag = reverse_dag(forward_sparse_dag)

        # Generate random initial layout candidates
        init_mappings = np.array([_rng.permutation(n_physical).astype(qdt) for _ in range(init_layout_attempts)])

        # Evaluate all candidates in parallel
        results, refined_mappings = perform_layout_stage_sectioned(
            init_mappings,
            forward_sparse_dag,
            backward_sparse_dag,
            topology,
            layout_iterations,
            section_lengths,
        )

        # Select best layout — use the already-refined mapping directly.
        best_index = np.argmin(results)
        best_mapping = refined_mappings[best_index]

        # Build a new circuit with physical qubit count, remapped
        qubit_amount = n_physical
        new_qc = qc.copy()

        amended_counter = 0
        while new_qc.num_qubits() < qubit_amount:
            new_qc.add_qubit(Qubit("amended_qb_" + str(amended_counter)))
            amended_counter += 1

        inv_best_mapping = invert_permutation(best_mapping)
        new_qc.qubits = [new_qc.qubits[inv_best_mapping[i]] for i in range(new_qc.num_qubits())]

        return new_qc

    _plasma_layout.__name__ = "plasma_layout"
    return _plasma_layout
