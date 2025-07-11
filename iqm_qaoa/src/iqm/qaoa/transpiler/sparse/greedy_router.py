# Copyright (c) 2024-2025 IQM Quantum Computers
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted (subject to the
# limitations in the disclaimer below) provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this list of conditions and the following
#   disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
#   disclaimer in the documentation and/or other materials provided with the distribution.
# * Neither the name of IQM Quantum Computers nor the names of its contributors may be used to endorse or promote
#   products derived from this software without specific prior written permission.
#
# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY
# THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""The main module of the greedy routing algorithm.

The main function to be called for routing is :func:`greedy_router`. The rest are helper functions.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import chain
import math

from dimod import BinaryQuadraticModel
from iqm.qaoa.transpiler.quantum_hardware import QPU, HardEdge, HardQubit, LogEdge, LogQubit
from iqm.qaoa.transpiler.routing import Layer, Routing
from iqm.qaoa.transpiler.sparse.two_color_mapper import two_color_mapper
import networkx as nx


def _int_pair_distance(
    routing: Routing, buffer_interactions: set[LogEdge], swap_pair: HardEdge | None = None, any_distance: bool = False
) -> int:
    """Function for the distance between hardware qubits of interaction pairs in ``buffer_interactions``.

    The function operates in a few modes based on the input variables:

    * If ``swap_pair`` (of hardware qubits) is given, the function only considers distances involving those qubits,
      otherwise it considers the distances between all pairs of logical qubits in ``buffer_interactions``
    * If ``any_distance`` is True, the function returns the first non-zero distance it finds.
    * If ``any_distance`` is False, the function sums up all of the considered distances.

    Args:
        routing: The :class:`~iqm.qaoa.transpiler.routing.Routing` object that the whole algorithm is working on.
        buffer_interactions: The interactions in the 'buffer' waiting to be executed.
        swap_pair: The pair of hardware qubits whose distances (to their interaction partners) are considered.
        any_distance: True to return the first non-zero distance, False to sum up the distances and return the sum.

    Returns:
        Distance, explained above.

    """
    r = 0
    for log_qb0, log_qb1 in buffer_interactions:
        if any_distance and r > 0:
            return r
        hard_qb0, hard_qb1 = routing.mapping.log2hard[log_qb0], routing.mapping.log2hard[log_qb1]
        if swap_pair is None or swap_pair is not None and (hard_qb0 in swap_pair or hard_qb1 in swap_pair):
            r += len(routing.qpu.shortest_path[hard_qb0][hard_qb1]) - 2
    return r


def _int_pair_distance_change(routing: Routing, buffer_interactions: set[LogEdge], swap: HardEdge) -> int:
    """Calculate the change of distances between logical qubits in ``buffer_interactions`` after a swap.

    Args:
        routing: The :class:`~iqm.qaoa.transpiler.routing.Routing` object that the whole algorithm is working on.
        buffer_interactions: The interactions in the 'buffer' waiting to be executed.
        swap: The pair of qubits to be swapped.

    Returns:
        The distance change before - after the swap.

    """
    dist_before = _int_pair_distance(routing, buffer_interactions, swap)
    routing.mapping.swap_hard(swap)
    dist_after = _int_pair_distance(routing, buffer_interactions, swap)
    routing.mapping.swap_hard(swap)
    return dist_before - dist_after


def _execute_all_possible_int_gates(
    routing: Routing, buffer_interactions: set[LogEdge], buffer_involved_qubits: set[LogQubit], problem_graph: nx.Graph
) -> bool:
    """As the name suggests, this executes all possible interaction gates (and then changes the buffer accordingly).

    Looks for all gates left over in ``routing.remaining_interactions``, finds those that can be executed, and
    executes them.
    This function potentially modifies :class:`~iqm.qaoa.transpiler.routing.Routing`, ``buffer_interactions``,
    ``buffer_involved_qubits`` and ``problem_graph`` in-place.

    Args:
        routing: The :class:`~iqm.qaoa.transpiler.routing.Routing` object that the whole algorithm is working on.
        buffer_interactions: The interactions in the 'buffer' waiting to be executed.
        buffer_involved_qubits: The qubits involved in the gates in ``buffer_interactions``
        problem_graph: A graph containing all not-yet-executed interactions (with edge bias corresponding
            to the distance on the HW graph)

    Returns:
        Bool indicating whether any gate was executed.

    """
    int_graph = nx.Graph()  # type: ignore[var-annotated]
    # This bool tracks if any gate was executed at all
    gate_executed = False

    # We check if there are any interactions in ``remaining_interactions`` which we can execute 'for free'.
    for log_0, log_1 in routing.remaining_interactions.edges:
        hard_qb0, hard_qb1 = routing.mapping.log2hard[log_0], routing.mapping.log2hard[log_1]

        # The qubits are neighboring on the HW graph ...
        if routing.qpu.hardware_graph.has_edge(hard_qb0, hard_qb1):
            # ... and they aren't occupied by other gates.
            if routing.layers[-1].int_gate_applicable(HardEdge((hard_qb0, hard_qb1))):
                int_graph.add_edge(hard_qb1, hard_qb0)
                gate_executed = True
    # Only choose a subset of the applicable gates which can be executed in parallel (i.e., a matching).
    matching = nx.maximal_matching(int_graph)

    for hard_qb0, hard_qb1 in matching:
        log_qb0 = routing.mapping.hard2log[hard_qb0]
        log_qb1 = routing.mapping.hard2log[hard_qb1]
        int_pair = LogEdge((log_qb0, log_qb1))
        routing.apply_int(HardEdge((hard_qb0, hard_qb1)))
        if int_pair in buffer_interactions:
            # The interaction is in the buffer, so it should be removed.
            _remove_int(int_pair, buffer_interactions, buffer_involved_qubits, problem_graph)
        else:
            problem_graph.remove_edge(log_qb0, log_qb1)

    return gate_executed


def _remove_int(
    gate_to_be_removed: LogEdge,
    buffer_interactions: set[LogEdge],
    buffer_involved_qubits: set[LogQubit],
    problem_graph: nx.Graph,
) -> None:
    """The function to remove an interaction gate from the buffer of interactions.

    After removal, this triggers the function ``update_after_removal`` for both of the logical qubits involved
    in the removed interaction.
    This function modifies ``buffer_interactions``, ``buffer_involved_qubits`` and ``problem_graph`` in-place.

    Args:
        gate_to_be_removed: The interaction that is about to be removed from the buffer.
        buffer_interactions: The interactions in the 'buffer' waiting to be executed.
        buffer_involved_qubits: The qubits involved in the gates in ``buffer_interactions``.
        problem_graph: A graph containing all not-yet-executed interactions (with edge bias corresponding
            to the distance on the HW graph).

    """
    log_q1, log_q2 = gate_to_be_removed
    problem_graph.remove_edge(log_q1, log_q2)

    buffer_interactions.remove(gate_to_be_removed)
    buffer_involved_qubits.remove(log_q1)
    buffer_involved_qubits.remove(log_q2)

    _update_after_removal(log_q1, buffer_involved_qubits, buffer_interactions, problem_graph)
    _update_after_removal(log_q2, buffer_involved_qubits, buffer_interactions, problem_graph)


def _update_after_removal(
    log_q: LogQubit, buffer_involved_qubits: set[LogQubit], buffer_interactions: set[LogEdge], problem_graph: nx.Graph
) -> None:
    """The function to update the buffer, after the qubit ``log_q`` is removed from it.

    It checks for all interactions involving ``log_q``, finds the qubit closest on the hardware and adds both
    it and ``log_q`` back to the buffer. If ``log_q`` has no more unrealized interactions, this does nothing.
    This function potentially modifies ``buffer_interactions`` and ``buffer_involved_qubits`` in-place.

    Args:
        log_q: The logical qubit just removed from the buffer
        buffer_interactions: The interactions in the 'buffer' waiting to be executed.
        buffer_involved_qubits: The qubits involved in the gates in ``buffer_interactions``.
        problem_graph: A graph containing all not-yet-executed interactions (with edge bias corresponding
            to the distance on the HW graph).

    """
    qubits_that_interact_with_log_q = problem_graph.neighbors(log_q)
    best_candidate = None
    best_distance = math.inf
    for inter in qubits_that_interact_with_log_q:
        if inter not in buffer_involved_qubits:
            dist = problem_graph.get_edge_data(inter, log_q)["bias"]
            if dist < best_distance:
                best_distance = dist
                best_candidate = inter

    if best_candidate is not None:
        buffer_interactions.add(LogEdge((log_q, best_candidate)))
        buffer_involved_qubits.add(log_q)
        buffer_involved_qubits.add(best_candidate)


def _execute_swaps(
    matching: set[tuple[HardQubit, HardQubit]],
    routing: Routing,
    buffer_interactions: set[LogEdge],
    problem_graph: nx.Graph,
) -> None:
    """Takes a set of swaps in ``matching`` and applies them to the routing.

    Checks if the swaps decrease the distances between logical qubits in ``buffer_interactions``. Afterwards
    updates the distances in ``problem_graph``.
    This function modifies ``problem_graph`` and :class:`~iqm.qaoa.transpiler.routing.Routing` in-place.

    Args:
        matching: A set of logical gates (presumably a matching).
        routing: The :class:`~iqm.qaoa.transpiler.routing.Routing` object that the whole algorithm is working on.
        buffer_interactions: The interactions in the 'buffer' waiting to be executed.
        problem_graph: A graph containing all not-yet-executed interactions (with edge bias corresponding
            to the distance on the HW graph)

    """
    for edge in matching:
        # Once more check that the distance really decreases by applying this swap
        if _int_pair_distance_change(routing, buffer_interactions, HardEdge(edge)) >= 1:
            routing.apply_swap(HardEdge(edge))
            log1, log2 = routing.mapping.hard2log[edge[0]], routing.mapping.hard2log[edge[1]]
            _update_distances(routing, log1, problem_graph)
            _update_distances(routing, log2, problem_graph)


def _decrease_int_pair_distance(routing: Routing, buffer_interactions: set[LogEdge], problem_graph: nx.Graph) -> bool:
    """Attempts to decrease the distances between logical qubit pairs in ``buffer_interactions`` by applying swaps.

    Iterates over all possible HW graph edges and checks if the swaps along these decrease the sum of distances
    between logical qubits in ``buffer_interactions``. The swaps which decrease the distance are collected in
    ``swap_graph`` and executed. The whole cycle is repeated as long as there are swaps which decrease the sum
    of distances.
    This function modifies ``problem_graph`` and ``routing`` in-place.

    Args:
        routing: The :class:`~iqm.qaoa.transpiler.routing.Routing` object that the whole algorithm is working on.
        buffer_interactions: The interactions in the 'buffer' waiting to be executed.
        problem_graph: A graph containing all not-yet-executed interactions (with edge bias corresponding
            to the distance on the HW graph).

    Returns:
        True if any swaps were executed, False otherwise.

    """
    gate_executed = False
    for _ in range(len(routing.mapping.hard2log)):
        swap_graph = nx.Graph()  # type: ignore[var-annotated]
        for hard_qb0, hard_qb1 in routing.active_subgraph.edges():
            swap_gate = HardEdge((hard_qb0, hard_qb1))
            if routing.layers[-1].swap_gate_applicable(swap_gate):
                diff = _int_pair_distance_change(routing, buffer_interactions, swap_gate)
                # If ``swap_gate`` decreases the distances, it is added to ``swap_graph``
                if diff in {2, 1}:
                    swap_graph.add_edge(hard_qb0, hard_qb1, weight=diff)

        if not nx.is_empty(swap_graph):
            # We can only execute the swaps that are compatible (i.e., they don't swap the same qubit).
            # The matching is weighted by how much do the swaps decrease the distances.
            matching = nx.max_weight_matching(swap_graph)
            _execute_swaps(matching, routing, buffer_interactions, problem_graph)
            gate_executed = True
        else:
            break
    return gate_executed


def _fallback_routine(routing: Routing, buffer_interactions: set[LogEdge], problem_graph: nx.Graph) -> None:
    """The fallback strategy for if no swap can decrease the distance of the interactions in the buffer.

    The strategy is to pick a random qubit pair to swap, where one of the qubits is guaranteed to move closer to its
    partner after the swap. Presumably, the other qubit will move further away from its partner.
    This function modifies :class:`~iqm.qaoa.transpiler.routing.Routing` and ``problem_graph`` in-place.

    Args:
        routing: The :class:`~iqm.qaoa.transpiler.routing.Routing` object that the whole algorithm is working on.
        buffer_interactions: The interactions in the 'buffer' waiting to be executed.
        problem_graph: A graph containing all not-yet-executed interactions (with edge bias corresponding
            to the distance on the HW graph).

    """

    @lru_cache(maxsize=None)
    def _deterministic_pair(buffer_interactions: frozenset[LogEdge]) -> LogEdge:
        """Deterministically picks an arbitrary interaction pair from ``buffer_interactions``.

        Note that ``buffer_interactions`` has to be converted to a hashable type (e.g., frozenset) before it can
        be passed to this function because of how ``@lru_cache`` works.

        Args:
            buffer_interactions: The interactions in the 'buffer' waiting to be executed.

        Returns:
            A :class:`LogEdge` object, picked from ``buffer_interactions``.

        """
        int_pairs_copy = set(buffer_interactions)
        return int_pairs_copy.pop()

    det_log_qb0, det_log_qb1 = _deterministic_pair(frozenset(buffer_interactions))
    hard_qb0 = routing.mapping.log2hard[det_log_qb0]
    hard_qb1 = routing.mapping.log2hard[det_log_qb1]
    # Find the shortest path between the corresponding hardware qubits
    shortest_path = routing.qpu.shortest_path[hard_qb0][hard_qb1]
    # Swap the first two qubits along the shortest path, guaranteeing that the first qubit from ``det_pair``
    # gets closer to the second qubit. Presumably, the second qubit being swapped will get further away from
    # its logical interaction partner (otherwise the fallback routine wouldn't be triggered).
    routing.apply_swap(HardEdge(shortest_path[:2]))
    _update_distances(routing, routing.mapping.hard2log[shortest_path[0]], problem_graph)
    _update_distances(routing, routing.mapping.hard2log[shortest_path[1]], problem_graph)


def _find_best_replacement(
    lq: LogQubit, buffer_involved_qubits: set[LogQubit], problem_graph: nx.Graph
) -> tuple[float, tuple[LogQubit, LogQubit]] | tuple[float, None]:
    """Finds the best qubit not in ``buffer_involved_qubits`` which has an un-realized interaction with ``lq``.

    Iterates over qubits with unrealized interactions with ``lq``, looks at their distance from ``lq`` on the hardware
    and finds the closest one. If it's NOT in ``buffer_involved_qubits``, returs the distance and the logical
    qubit pair. Otherwise returns ``math.inf`` and ``None``.

    Args:
        lq: A logical qubit.
        buffer_involved_qubits: The qubits involved in the gates in ``buffer_interactions``.
        problem_graph: A graph containing all not-yet-executed interactions (with edge bias corresponding
            to the distance on the HW graph).

    Returns:
        Distance to the best logical neighbor of ``lq`` and the qubit pair as a tuple. Or ``math.inf`` and ``None``.

    """
    min_dist = math.inf
    lq_best_pair = None

    for neighbor in problem_graph.neighbors(lq):
        dist = problem_graph.get_edge_data(neighbor, lq)["bias"]
        if dist < min_dist:
            min_dist = dist
            lq_best_pair = neighbor
    if lq_best_pair not in buffer_involved_qubits:
        return min_dist, (lq, lq_best_pair)  # type: ignore[return-value]
    return math.inf, None


def _update_buffer(
    buffer_interactions: set[LogEdge], buffer_involved_qubits: set[LogQubit], problem_graph: nx.Graph
) -> None:
    """Updates the buffer set of gates (and the set of involved qubits).

    For every interaction in the buffer, this searches for interactions involving one of the qubits, which are
    outside the buffer and closer on the QPU (i.e., lower edge weight in ``problem_graph``). If such an interaction is
    found, it replaces the original interaction in the buffer.
    This function potentially modifies ``buffer_interactions`` and ``buffer-involved_qubits`` in-place.

    Args:
        buffer_interactions: The interactions in the 'buffer' waiting to be executed.
        buffer_involved_qubits: The qubits involved in the gates in ``buffer_interactions``.
        problem_graph: A graph containing all not-yet-executed interactions (with edge bias corresponding
            to the distance on the HW graph).

    """
    pairs_to_be_added = set()
    pairs_to_be_removed = set()
    for lq_1, lq_2 in buffer_interactions:
        # Consider replacements for both ``lq_1`` and ``lq_2``.
        lq1_repl = _find_best_replacement(lq_1, buffer_involved_qubits, problem_graph)
        lq2_repl = _find_best_replacement(lq_2, buffer_involved_qubits, problem_graph)
        best_repl = lq1_repl if lq1_repl[0] <= lq2_repl[0] else lq2_repl
        # If a replacement for either was found ...
        if best_repl[1] is not None:
            # ... remove both qubits from ``buffer_involved_qubits`` ...
            buffer_involved_qubits.remove(lq_1)
            buffer_involved_qubits.remove(lq_2)

            # ... and then add the best replacement.
            pairs_to_be_added.add(LogEdge(best_repl[1]))
            pairs_to_be_removed.add(LogEdge((lq_1, lq_2)))
            buffer_involved_qubits.add(best_repl[1][0])
            buffer_involved_qubits.add(best_repl[1][1])

    # The set of ``buffer_interactions`` is only updated after the for loop is finished.
    buffer_interactions.update(pairs_to_be_added)
    buffer_interactions.difference_update(pairs_to_be_removed)


def _update_distances(routing: Routing, log_q: LogQubit, problem_graph: nx.Graph) -> None:
    """Updates the distances in ``problem_graph``.

    This function is called after the logical qubit ``log_q`` has been swapped to a different hardware qubit.
    Therefore the distances to its neighbors have changed and need to be updated.
    This function modifies the edge weights "bias" of the graph ``problem_graph`` in-place.

    Args:
        routing: The :class:`~iqm.qaoa.transpiler.routing.Routing` that the whole algorithm is working on.
        log_q: The logical qubit that has just been swapped.
        problem_graph: A graph containing all not-yet-executed interactions (with edge bias corresponding
            to the distance on the HW graph).

    """
    for neighbor in problem_graph.neighbors(log_q):
        distance = (
            len(routing.qpu.shortest_path[routing.mapping.log2hard[log_q]][routing.mapping.log2hard[neighbor]]) - 2
        )
        problem_graph[neighbor][log_q]["bias"] = distance


def _greedy_pair_mapper(
    routing: Routing, buffer_interactions: set[LogEdge], buffer_involved_qubits: set[LogQubit], problem_graph: nx.Graph
) -> None:
    """This function performs the main iteration of applying swap gates and interactions until they're all applied.

    As long as there are interactions to be executed (in :meth:`remaining_interactions`), this loops through
    executing gates, updating the gate buffer and applying swaps to decrease the 'distances' between interactions
    in the buffer.
    This function extensively modifies :class:`~iqm.qaoa.transpiler.routing.Routing`, ``buffer_interactions``,
    ``buffer_involved_qubits`` and ``problem_graph`` in-place.

    Args:
        routing: The :class:`~iqm.qaoa.transpiler.routing.Routing` object that the whole algorithm is working on.
        buffer_interactions: The interactions in the 'buffer' waiting to be executed.
        buffer_involved_qubits: The qubits involved in the gates in ``buffer_interactions``.
        problem_graph: A graph containing all not-yet-executed interactions (with edge bias corresponding
            to the distance on the HW graph).

    """
    # This is called just after ``get_initial_objects``, so we necessarily need to add at least one more layer.
    routing.layers.append(Layer(routing.qpu))

    while len(routing.remaining_interactions.edges) != 0:
        gate_executed = _execute_all_possible_int_gates(
            routing, buffer_interactions, buffer_involved_qubits, problem_graph
        )
        _update_buffer(buffer_interactions, buffer_involved_qubits, problem_graph)
        if len(buffer_interactions) > 0:
            # If there are any non-neighboring logical qubit pairs in ``buffer_interactions`` ...
            if _int_pair_distance(routing, buffer_interactions, any_distance=True) > 0:
                decreased = _decrease_int_pair_distance(routing, buffer_interactions, problem_graph)
                if not (gate_executed or decreased):
                    _fallback_routine(routing, buffer_interactions, problem_graph)
            routing.layers.append(Layer(routing.qpu))

        # If there are no more interactions in the buffer but there are still interactions to be done ...
        elif len(buffer_interactions) == 0 and len(routing.remaining_interactions.edges) != 0:
            # One of the remaining edges
            dummy = list(routing.remaining_interactions.edges)[0]
            buffer_interactions.add(LogEdge(dummy))
            buffer_involved_qubits.add(dummy[0])
            buffer_involved_qubits.add(dummy[1])
        # If there are no more interactions in the buffer or in the listr of interactions to be done ...
        else:
            # ... finish the algorithm.
            break


def _get_initial_objects(
    problem_bqm: BinaryQuadraticModel, qpu: QPU
) -> tuple[set[LogEdge], set[LogQubit], nx.Graph, Routing]:
    """The initialization function for the greedy mapper.

    First it calls the ``two_color_mapper`` which handles the edge coloring of the problem graph and creates
    a :class:`~iqm.qaoa.transpiler.routing.Mapping` based on that. Consequently,
    a :class:`~iqm.qaoa.transpiler.routing.Routing` is created, a ``problem_graph`` (with edge weights corresponding to
    distances on the HW graph). Then the first two interaction layers are executed. Lastly, the initial version of
    ``buffer_interactions`` is built (and from it, the initial ``buffer_involved_qubits``).

    Args:
        problem_bqm: The :class:`~dimod.BinaryQuadraticModel` of the problem we're trying to solve.
        qpu: The QPU that we're going to solve the problem on.

    Returns:
        The initial set of buffer edges.
        The set of logical qubits involved in these edges.
        The problem graph with edge weights corresponding to distances between the logical qubits on the hardware.
        The :class:`~iqm.qaoa.transpiler.routing.Routing` object with two first interaction layers executed.

    """
    # Find an initial mapping using edge coloring.
    initial_mapping, first_two_int_layers = two_color_mapper(problem_bqm, qpu)

    route = Routing(problem_bqm, qpu, initial_mapping=initial_mapping)

    # Produce problem graph for virtual interactions, with distance included as edge bias.
    problem_graph = nx.Graph()  # type: ignore[var-annotated]
    for q1, q2 in route.remaining_interactions.edges:
        hard1, hard2 = route.mapping.log2hard[q1], route.mapping.log2hard[q2]
        # The distance of the interaction in terms of number of swaps needed to be able to execute the interaction.
        # It corresponds to the length of the shortest path minus two (excluding the first and last node of the path).
        problem_graph.add_edge(q1, q2, bias=len(route.qpu.shortest_path[hard1][hard2]) - 2)

    # Iterate over all gates in the first two layers (chain unpacks the two layers and all gates in them)
    for int_gate in chain.from_iterable(first_two_int_layers):
        route.apply_int(int_gate)
        log1, log2 = route.mapping.hard2log[list(int_gate)[0]], route.mapping.hard2log[list(int_gate)[1]]
        problem_graph.remove_edge(log1, log2)

    initial_buffer = nx.maximal_matching(problem_graph)
    initial_buffer = set(LogEdge(edge) for edge in initial_buffer)
    initial_involved_qubits = set()
    for log1, log2 in initial_buffer:
        initial_involved_qubits.add(log1)
        initial_involved_qubits.add(log2)

    return initial_buffer, initial_involved_qubits, problem_graph, route


def greedy_router(problem_bqm: BinaryQuadraticModel, qpu: QPU) -> Routing:
    """The function which takes a problem BQM ``problem_bqm`` and a QPU ``qpu`` and returns a routing.

    This serves as a 'wrapper' for the entire greedy routing algorithm. For details of the algorithm, see
    :cite:`Kotil_2023`.

    Args:
        problem_bqm: The :class:`~dimod.BinaryQuadraticModel` of the problem we're trying to solve.
        qpu: The :class:`~iqm.qaoa.transpiler.quantum_hardware.QPU` that we're going to solve the problem on.

    Returns:
        A routing object containing all the swap and interaction layers needed to execute one QAOA layer.

    """
    buffer_interactions, buffer_involved_qubits, problem_graph, route = _get_initial_objects(problem_bqm, qpu)
    _greedy_pair_mapper(route, buffer_interactions, buffer_involved_qubits, problem_graph)
    return route
