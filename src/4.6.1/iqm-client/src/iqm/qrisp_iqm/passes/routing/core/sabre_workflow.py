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


"""SABRE routing workflow.

This module implements the SABRE workflow, described in
`https://arxiv.org/pdf/1809.02573`__

Extended with:
- Per-physical-qubit depth tracking for depth-aware swap selection
- Descendant-count weighting for DAG-criticality-aware gate prioritization
- A congestion penalty mechanism that steers swaps towards idle physical qubits
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

from iqm.qrisp_iqm.passes.routing.core.graph_processing_tools import (
    find_path,
    find_path_length,
    gen_swap_candidates,
    reachability_counts_csr,
)
from iqm.qrisp_iqm.passes.routing.core.permutation_tools import invert_permutation, permute_array
from iqm.qrisp_iqm.passes.routing.core.sabre_metric import (
    compute_depth_array,
    compute_extended_set_prefactor,
    compute_front_layer_prefactor,
    compute_sabre_cost_change,
    filter_involvement_list,
    precompute_exp_depth,
)
from numba import njit
import numpy as np

if TYPE_CHECKING:
    from iqm.qrisp_iqm.passes.routing.core.graph_processing_tools import CircuitDAG, QPUTopology


InstructionRef: TypeAlias = tuple[np.int16, np.int16, np.int16]
"""Represents an instruction in a quantum circuit as a
(locus_qubit_0_index, locus_qubit_1_index, instr_index) tuple."""


metric_mode = np.zeros(1)


@njit(cache=True)
def sabre_gen_swaps(  # noqa: PLR0913, PLR0912, PLR0915
    topology: QPUTopology,
    circuit_dag: CircuitDAG,
    greediness: int,
    max_instruction: float = np.inf,
    seed: int = 0,
    initial_phys_depth: np.ndarray = np.zeros(0, dtype=np.int32),
    congestion_penalty: float = 0.0,
    backward_mode: bool = False,
    scaling_threshold: float = 0.0,
) -> tuple[list[InstructionRef], np.ndarray]:
    """Core SABRE routing loop.

    Iteratively moves through the DAG front layer,
    compiling gates that are already placed on adjacent physical qubits and
    inserting SWAPs to bring non-adjacent operands together.

    Args:
        topology:
            Hardware topology.
        circuit_dag:
            Circuit DAG in CSR format.
        greediness:
            Controls swap selection randomness. Higher = more deterministic.
        max_instruction:
            Only compile DAG nodes with index < max_instruction (for sectionized routing).
        seed:
            RNG seed for stochastic swap selection.
        initial_phys_depth:
            Starting per-physical-qubit depth. The dtype is inherited from the
            topology (int16 on NISQ-scale devices).  Non-empty when routing a
            section that follows a previously routed section, so depth
            accounting is continuous across sections.
        congestion_penalty:
            Strength of the congestion penalty applied to swap candidate scores.
            When > 0, swaps touching high-depth physical qubits receive a score
            penalty proportional to their local depth, encouraging the router
            to spread work across idle qubits and reduce critical-path depth.
            0.0 disables the penalty (pure gate-count optimization).
        backward_mode:
            If True, operate in backward mode (layout refinement).
        scaling_threshold:
            Instructions whose exp_depth_array weight falls below this threshold
            are skipped during cost-change evaluation.  For deep circuits this
            prunes near-zero contributions from the hot loop, avoiding expensive
            lookups on instructions whose weight has decayed to insignificance.
            Default 0.0 (no skipping) for backward compatibility.

    Returns:
        * Compiled instructions. SWAPs have instr_index = -1.
        * Final logical-to-physical qubit mapping.  The dtype is inherited
          from the topology (int16 on NISQ-scale devices).

    """
    # Ensure greediness >= 2 for correct stochastic selection behavior.
    # greediness=1 would cause seed % 1 = 0 always, skipping all candidates.
    greediness = max(greediness, 2)

    qubit_amount = topology[2].shape[0]
    distance_matrix = topology[2]

    # Derive dtypes from the incoming data so the entire routing loop
    # uses the narrowest possible integer type.  The topology tuples and
    # instruction DAGs are constructed with int16 when the problem size
    # permits (see graph_processing_tools.coupling_to_topology and
    # convert_qc_to_sparse_dag).
    qdt = topology[2].dtype  # per-qubit dtype (c2a, phys_depth)

    # Per-physical-qubit ASAP depth tracker. Records the schedule depth at
    # each physical qubit position, updated inline whenever a gate is compiled
    # (+1 for a two-qubit gate) or a SWAP is inserted (+3, since a SWAP
    # decomposes into 3 CX gates). This information serves two purposes:
    #   1. It feeds the depth bias during swap candidate scoring, steering
    #      the router away from congested qubits.
    #   2. It is propagated across sections via initial_phys_depth so that
    #      depth accounting remains continuous in the sectionized strategy.
    phys_depth = np.zeros(qubit_amount, dtype=qdt)
    if len(initial_phys_depth) == qubit_amount:
        for _pd in range(qubit_amount):
            phys_depth[_pd] = initial_phys_depth[_pd]

    # The DAG is represented as a sparse matrix in CSR format
    # for more information check graph_processing_tools.py
    indptr = circuit_dag[0]
    indices = circuit_dag[1]
    instruction_data = circuit_dag[2]
    idt = instruction_data.dtype  # per-node dtype (in_degree, reachability, ...)

    # Reachability-based gate prioritisation.
    # Gates that lie on long dependency chains are more "mission critical" —
    # delaying them cascades into deeper circuits.  We compute per-node
    # reachability counts (descendants in forward mode, predecessors in
    # backward mode) and derive exponential weights:
    #
    #   criticality_weight[i] = exp(sgn * alpha * reachability_count[i])
    #
    # where alpha is normalised so the maximum exponent stays bounded (~10)
    # and sgn flips the sign in backward mode.  In forward mode, high-
    # descendant nodes receive *high* weight (they are mission critical and
    # should be routed first).  In backward mode (layout refinement), the
    # goal is reversed: mission-critical nodes should end up being executed
    # *last*, so high-predecessor nodes receive *low* weight.

    # We first identify the single qubit gates since they should not
    # contribute to our counting.

    single_qubit_nodes = np.zeros(len(instruction_data), dtype=idt)
    for i in range(len(instruction_data)):
        if instruction_data[i][0] == instruction_data[i][1]:
            single_qubit_nodes[i] = 1

    # Compute per-node reachability counts (descendants or predecessors).
    reachability_counts = reachability_counts_csr(indptr, indices, single_qubit_nodes, predecessors=backward_mode)
    criticality_weights = np.zeros(len(instruction_data), dtype=np.float32)

    # Compute exponential criticality weights.
    # alpha is normalized so the maximum exponent stays bounded (~10).
    # sgn = +1 (forward) or -1 (backward) flips the weighting direction.
    # Using a plain ternary instead of (-1)**backward_mode keeps the
    # expression integer-valued and avoids a numpy scalar exponentiation.
    alpha = 10 / (np.max(reachability_counts) + 2)
    sgn = -1 if backward_mode else 1
    for i in range(len(instruction_data)):
        criticality_weights[i] = np.exp(sgn * alpha * reachability_counts[i])

    # In the next step, we set up the core data structures of the algorithm:
    # The front layer and the extended set.

    # To keep track of the front layer, we initialize an array that counts
    # the amount of "in" connections of each node. When a node is processed,
    # all children of this node have their in-degree reduced by one.
    node_amount = len(indptr) - 1
    in_degree = np.zeros(node_amount, dtype=idt)

    # Iterate through the sparse dag to compute the in degree of each node
    for i in range(node_amount):
        for j in indices[indptr[i] : indptr[i + 1]]:
            # If an index is less than 0, this means it has been pruned
            # by the prune_dag function. In this case it will be ignored
            # by all processing steps.
            if j < 0:
                continue
            in_degree[j] += 1

    extended_set_involvement = [[np.int16(x) for x in range(0)] for i in range(qubit_amount)]
    front_layer_involvement = [[np.int16(x) for x in range(0)] for i in range(qubit_amount)]

    # Create a queue and enqueue all vertices with
    # indegree 0
    front_layer_indices = []

    # To increase the efficiency of computing the metric we compute only the CHANGE in the metric
    # induced by each swap. This is more efficient because only the instructions
    # that are affected by this particular swap contribute to this quantity.
    # Because of this, we keep track of an "involvement" list of each qubit,
    # i.e. each qubit has a list of all the instructions that happen on it at some point.

    for i in range(node_amount):
        instruction = instruction_data[i]
        # Single-qubit gates do not contribute to the SABRE metric (they have
        # distance 0 regardless of qubit placement), so we skip adding them
        # to the involvement lists.  This avoids loading their indices in
        # the compute_sabre_cost_change hot loop only to skip them via continue.
        is_single_qubit_gate = instruction[0] == instruction[1]
        if in_degree[i] == 0:
            front_layer_indices.append(np.int16(i))
            if not is_single_qubit_gate:
                front_layer_involvement[instruction[0]].append(np.int16(i))
                front_layer_involvement[instruction[1]].append(np.int16(i))
        elif not is_single_qubit_gate:
            extended_set_involvement[instruction[0]].append(np.int16(i))
            extended_set_involvement[instruction[1]].append(np.int16(i))

    # keeps track of how many times the release valve has been called.
    # this is important because if the release valve makes up a majority
    # of the routing, the strategy is essentially greedy, which is less
    # efficient.
    valve_calls = 0

    # This list will contain the compiled instructions as a list of tuples.
    # There are three possible cases to be distinguished here:

    # 1. A single qubit gate will be the tuple (qubit_index_0, qubit_index_0, instruction_index)
    # 2. A two qubit gate will be the tuple (qubit_index_0, qubit_index_1, instruction_index)
    # 3. A swap will be the tuple (qubit_index_0, qubit_index_1, -1)

    # The instruction index is the position of the corresponding Qrisp instruction
    # in the .data attribute of the input QuantumCircuit.
    compiled_instruction_list = []

    # c2a stands for "citizen to address". This nomenclature essentially distinguishes
    # between physical and logical qubits but might have less potential for confusion.
    # Citizens correspond to logical qubits and addresses to physical qubits.
    # This conversion is stored as a numpy array, representing a member of the permutation group.
    # Several processing functions are available for permutation processing.
    # See permutation_tools.py for more details.
    c2a = np.arange(qubit_amount, dtype=qdt)

    depth_array_update_required = True
    # The depth array indicates at what layer each
    # instruction lives in (the front layer has depth 1),
    # the following layer depth 2 etc.
    # The depth array only needs to be updated if an instruction
    # is compiled. In the case the last iteration only compiled a
    # SWAP, no update is needed.
    # This is reflected by this boolean, which will be flipped
    # to True if necessary.

    # We iterate until the front layer is empty
    while True:
        # Scan the front layer for executable gates (path_length <= 1 and
        # within the section boundary).  Among all executable gates, pick
        # the one whose physical operand qubits become free earliest, i.e.
        # the gate with the minimum max(phys_depth[p0], phys_depth[p1]).
        # This avoids needlessly pushing idle qubits deeper while busy
        # qubits remain occupied, reducing overall circuit depth.
        best_i = -1
        best_depth_val = np.int64(2**60)

        for i in range(len(front_layer_indices)):
            fli = front_layer_indices[i]
            if fli >= max_instruction:
                continue
            lnk = instruction_data[fli]
            path_length = find_path_length(c2a[lnk[0]], c2a[lnk[1]], topology)
            if path_length <= 1:
                p0 = c2a[lnk[0]]
                p1 = c2a[lnk[1]]
                # Among all executable gates, prefer the most mission-critical
                # one.  In forward mode (sgn=+1) high descendant count means
                # more negative dv → picked first.  In backward mode (sgn=-1)
                # the sign flips so high predecessor count yields positive dv
                # → picked last, letting less critical gates go first.
                if p0 == p1:
                    dv = -sgn * reachability_counts[fli]
                else:
                    dv = -sgn * reachability_counts[fli]
                if dv < best_depth_val:
                    best_depth_val = dv
                    best_i = i

        # If an executable gate was found, compile it and restart the
        # outer loop.  Restarting is important because compiling this gate
        # may unlock new front-layer entries that are also immediately
        # executable — those should be considered before inserting any SWAP.
        if best_i >= 0:
            fli = front_layer_indices[best_i]
            lnk = instruction_data[fli]

            # Remove from front layer and update involvement lists.
            # Single-qubit gates are not tracked in the involvement lists,
            # so only two-qubit gates trigger a removal.
            front_layer_indices.pop(best_i)
            if lnk[0] != lnk[1]:
                front_layer_involvement[lnk[0]].remove(np.int16(fli))
                front_layer_involvement[lnk[1]].remove(np.int16(fli))

            compiled_instruction_list.append((c2a[lnk[0]], c2a[lnk[1]], np.int16(lnk[2])))

            # Update physical depth tracking for the compiled gate.
            p0 = c2a[lnk[0]]
            p1 = c2a[lnk[1]]
            if p0 != p1:
                new_d = np.maximum(phys_depth[p0], phys_depth[p1]) + np.int16(1)
                phys_depth[p0] = new_d
                phys_depth[p1] = new_d

            depth_array_update_required = True

            # Promote successors whose in-degree drops to zero into the
            # front layer.
            for j in indices[indptr[fli] : indptr[fli + 1]]:
                if j < 0:
                    continue
                in_degree[j] -= 1
                if in_degree[j] == 0:
                    front_layer_indices.append(np.int16(j))

                    instruction = instruction_data[j]
                    # Single-qubit gates are not tracked in the involvement
                    # lists — they contribute zero to the metric.
                    if instruction[0] != instruction[1]:
                        extended_set_involvement[instruction[0]].remove(np.int16(j))
                        front_layer_involvement[instruction[0]].append(np.int16(j))
                        extended_set_involvement[instruction[1]].remove(np.int16(j))
                        front_layer_involvement[instruction[1]].append(np.int16(j))

            continue

        # If the front layer is empty, we are done
        if len(front_layer_indices) == 0:
            break

        # If the front layer consists of only instructions larger than the
        # maximum instruction index, we are done
        if max_instruction != np.inf:
            if np.all(np.array(front_layer_indices) >= max_instruction):
                break

        # We now compute the value of the Sabre metric for a given list of swaps
        a2c = invert_permutation(c2a)
        swap_candidates = gen_swap_candidates(instruction_data, front_layer_indices, topology, c2a, a2c)

        # We compute the depth array and the extended set if required.
        # The extended set consists of all instructions with non-zero in-degree.
        if depth_array_update_required:
            extended_set = np.nonzero(in_degree)[0].astype(np.int16)
            depth_array = compute_depth_array(front_layer_indices, extended_set, instruction_data, topology)
            exp_depth_array = precompute_exp_depth(depth_array)

            # The prefactors of the terms for both the front layer and the extended set
            # also stay the same for each swap so they can also be precomputed.
            front_layer_prefactor = compute_front_layer_prefactor(
                front_layer_indices, instruction_data, exp_depth_array
            )

            extended_set_prefactor = compute_extended_set_prefactor(extended_set, instruction_data, exp_depth_array)

            depth_array_update_required = False

            scaling_array = exp_depth_array * criticality_weights

            # Build filtered involvement lists once per depth-array update.
            # Instructions whose combined weight (exp_depth * criticality) has
            # decayed below the threshold are pruned from the lists that the
            # swap-scoring loop iterates over.  Filtering once here is far
            # cheaper than checking the threshold inside the per-candidate
            # hot loop — the involvement lists are stable between gate
            # compilations, so this work is amortised over many swap evaluations.
            if scaling_threshold > np.float32(0.0):
                fl_inv_filtered = [[np.int16(x) for x in range(0)] for _ in range(qubit_amount)]
                es_inv_filtered = [[np.int16(x) for x in range(0)] for _ in range(qubit_amount)]
                for q in range(qubit_amount):
                    fl_inv_filtered[q] = filter_involvement_list(
                        front_layer_involvement[q], scaling_array, scaling_threshold
                    )
                    es_inv_filtered[q] = filter_involvement_list(
                        extended_set_involvement[q], scaling_array, scaling_threshold
                    )

        # Keeps track of the score of each swap
        score_array = np.zeros(len(swap_candidates), dtype=np.float32)

        # Choose the appropriate involvement lists: filtered when threshold > 0,
        # original otherwise (no allocation overhead for the default path).
        if scaling_threshold > np.float32(0.0):
            fl_inv_ref = fl_inv_filtered
            es_inv_ref = es_inv_filtered
        else:
            fl_inv_ref = front_layer_involvement
            es_inv_ref = extended_set_involvement

        for i in range(len(swap_candidates)):
            swap = swap_candidates[i]

            # Compute the cost (check sabre_metric.py for more details)
            # The involvement lists are passed as 4 separate references,
            # avoiding 2 typed-list allocations per candidate.
            score = compute_sabre_cost_change(
                instruction_data,
                fl_inv_ref[swap[0]],
                fl_inv_ref[swap[1]],
                es_inv_ref[swap[0]],
                es_inv_ref[swap[1]],
                c2a,
                swap[0],
                swap[1],
                scaling_array,
                front_layer_prefactor,
                extended_set_prefactor,
                distance_matrix,
            )

            score_array[i] = score

        # Congestion penalty: steer swap selection towards physically idle qubits.
        #
        # After the standard SABRE score is computed for each swap candidate,
        # we add a penalty term proportional to the local physical depth at
        # the swap's qubits. This discourages routing through qubits that are
        # already on the critical path, effectively spreading operations
        # across the device and reducing circuit depth.
        #
        # The penalty magnitude is anchored to the best (most negative) score
        # improvement, scaled by congestion_penalty. This makes the penalty
        # unit-agnostic: regardless of circuit size, a congestion_penalty of X
        # means the penalty for the hottest qubit equals X times the best
        # improvement.
        if congestion_penalty > 0:
            score_min = np.min(score_array)
            if score_min < 0:
                max_pd = np.max(phys_depth)
                if max_pd > 0:
                    penalty_budget = -score_min * congestion_penalty
                    inv_max_pd = np.float32(1.0) / np.float32(max_pd)
                    for i in range(len(swap_candidates)):
                        swap = swap_candidates[i]
                        p0 = c2a[swap[0]]
                        p1 = c2a[swap[1]]

                        # If the depth of both qubits agrees, this
                        # is not causing any congestion.
                        # This is (most likely) because another 2qb
                        # gate has been executed, which can be
                        # used as a "booster-pad". We want to
                        # encourage the algorithm to use this,
                        # which is why we don't penalize this
                        # candidate.
                        if phys_depth[p0] == phys_depth[p1]:
                            continue

                        # Use the worse (higher) depth of the two qubits involved
                        local_d = np.maximum(phys_depth[p0], phys_depth[p1])

                        # Penalty is 0 for the coldest qubit, penalty_budget for the hottest
                        score_array[i] += penalty_budget * np.float32(local_d) * inv_max_pd

        # If both depths are the same, chances are this pair has recently been
        # acted on by a two qubit gate. Most two qubit gates (cp, cx) have
        # some simplification rule with swaps, so we want to encourage the algorithm
        # to take this swap in order for the cancellation to reduce gate count.
        # We call this a "booster pad".

        for i in range(len(swap_candidates)):
            swap = swap_candidates[i]
            p0 = c2a[swap[0]]
            p1 = c2a[swap[1]]

            if phys_depth[p0] == phys_depth[p1]:
                score_array[i] *= 2

        # Stochastic swap selection with geometric distribution over candidate ranks.
        # With probability (greediness-1)/greediness we pick the current best;
        # otherwise we skip to the next-best candidate and repeat.
        # This allows occasional "exploration" of suboptimal swaps, which can
        # help escape local minima in the routing.
        #
        # The distribution is approximately:
        #   ~80% pick best, ~16% pick 2nd best, ~3.2% pick 3rd best, etc.
        # (for greediness=5)
        #
        # We use a threshold comparison against the LCG output (range [0, 65536]):
        # skip when seed < 65537/greediness, i.e., with probability 1/greediness.

        skip_threshold = 65537 // greediness
        n_candidates = len(swap_candidates)
        best_swap: tuple[int, int] = (-1, -1)
        for _ in range(n_candidates):
            best_candidate = np.argmin(score_array)
            best_swap = swap_candidates[best_candidate]

            seed = lcg_next(seed)
            if seed >= skip_threshold:
                break  # Use this candidate

            # Skip to next candidate by marking this one as infinity
            score_array[best_candidate] = np.inf

        # After the loop, best_swap holds our selection
        # (either we broke early, or exhausted all candidates and use the last one)
        best_swap = (c2a[best_swap[0]], c2a[best_swap[1]])

        # This section treats the release valve i.e. the mechanism that forcibly
        # routes a gate if the sabre metric is trapped in a deadlock.
        if len(compiled_instruction_list) >= 1:
            # A deadlock can only appear if the last compiled instruction was a swap
            if compiled_instruction_list[-1][2] == -1:
                # Check if the last compiled instruction agrees with the chosen swap
                match_exactly = compiled_instruction_list[-1][:2] == best_swap
                match_swapped = compiled_instruction_list[-1][:2] == best_swap[::-1]

                if match_exactly or match_swapped:
                    valve_calls += 1

                    # Remove the compiled swap and undo the induced permutation
                    compiled_instruction_list.pop(-1)
                    c2a[a2c[best_swap[0]]], c2a[a2c[best_swap[1]]] = best_swap[1], best_swap[0]

                    a2c = invert_permutation(c2a)

                    # Determine the shortest link to route
                    path_lengths = np.zeros(len(front_layer_indices))

                    for k in range(len(front_layer_indices)):
                        lnk = instruction_data[front_layer_indices[k]]
                        if front_layer_indices[k] >= max_instruction:
                            path_lengths[k] = np.inf
                        else:
                            path_lengths[k] = find_path_length(c2a[lnk[0]], c2a[lnk[1]], topology)

                    best_link_index = np.argmin(path_lengths)
                    best_link = instruction_data[front_layer_indices[best_link_index]]

                    # Determine the available paths
                    path = find_path(c2a[best_link[0]], c2a[best_link[1]], topology)

                    temp_a2c = a2c.copy()
                    # Perform the swaps
                    for i in range(len(path) - 2):
                        swap = (path[i], path[i + 1])
                        compiled_instruction_list.append((swap[0], swap[1], np.int16(-1)))
                        # Update phys_depth for release-valve swap
                        # (+3 because a SWAP decomposes into 3 CX gates)
                        rv_d = np.maximum(phys_depth[swap[0]], phys_depth[swap[1]]) + np.int16(3)
                        phys_depth[swap[0]] = rv_d
                        phys_depth[swap[1]] = rv_d

                    temp = permute_array(temp_a2c, path)
                    c2a = invert_permutation(temp)
                    continue

        # If the release valve has not been triggered, compile the swap.
        c2a[a2c[best_swap[0]]], c2a[a2c[best_swap[1]]] = best_swap[1], best_swap[0]
        compiled_instruction_list.append((np.int16(best_swap[0]), np.int16(best_swap[1]), np.int16(-1)))

        # Update phys_depth for the compiled swap
        # (+3 because a SWAP decomposes into 3 CX gates)
        sw_d = np.maximum(phys_depth[best_swap[0]], phys_depth[best_swap[1]]) + np.int16(3)
        phys_depth[best_swap[0]] = sw_d
        phys_depth[best_swap[1]] = sw_d

    # print("valve calls")
    # print(valve_calls/len(instruction_data))
    return compiled_instruction_list, c2a


@njit(cache=True)
def lcg_next(seed: int) -> int:
    """Linear Congruential Generator for stochastic swap selection.

    Returns a value in [0, 65536] with approximately uniform distribution.
    Used with threshold comparison: skip candidate when
    ``seed < 65537 / greediness``.

    The parameters a=75, c=74, m=65537 are chosen for good distribution
    properties; m=65537 is a prime modulus (2^16 + 1) ensuring full period.

    Args:
        seed: Current RNG seed value.

    Returns:
        Next value in the LCG sequence.

    """
    return (75 * seed + 74) % 65537
