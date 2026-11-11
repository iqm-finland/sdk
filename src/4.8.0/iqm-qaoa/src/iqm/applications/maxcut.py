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
"""Module containing the maxcut problem instance class and related functions.

Contains the iterator function :func:`maxcut_generator` which yields random instances of :class:`MaxCutInstance`, useful
for applications such as calculating the Q-score. Also contains two classical solvers for maxcut problems:

- A simple and fast greedy algorithm.
- A solver based on the standard *Goemans-Williamson* :cite:`Goemans_1995`, fast and with a performance guarantee.

Example:

    .. code-block:: python

        from iqm.applications.maxcut import MaxCutInstance
        from iqm.applications.maxcut import maxcut_generator
        from iqm.applications.maxcut import goemans_williamson
        from iqm.applications.maxcut import greedy_max_cut

        for my_instance in maxcut_generator(graph_size, n_instances):  # Generates problem instances.
            gw_solution = goemans_williamson(my_instance)  # Solution bitstring obtained from GW.
            greedy_solution = greedy_max_cut(my_instance)  # Solution bitstring obtained from greedy.
            my_instance.cut_size(gw_solution)  # Check the GW solution cut size.

        my_maxcut_instance = MaxCutInstance(my_graph)  # Problem instance from a graph.

"""

from collections.abc import Iterator
from typing import Literal, cast

import cvxpy as cp
from dimod import BinaryQuadraticModel
from iqm.applications.graph_utils import (
    EDGE_ATTR_PRIORITY,
    _generate_desired_graph,
    _get_attr_with_priority,
    draw_problem,
)
from iqm.applications.qubo import QUBOInstance
import networkx as nx
import numpy as np
from scipy.linalg import eigh


class MaxCutInstance(QUBOInstance):
    r"""The maxcut instance class.

    This class is initialized with a graph whose maxcut we try to find. A maxcut is a division of the graph nodes
    into two groups, such that the number of edges connecting nodes from different groups is maximized.
    The optional ``break_z2`` variable indicates whether the :math:`\mathbb{Z}_2` symmetry of the problem is to be
    broken by pre-assigning one of the nodes to one of the groups.

    Args:
        graph: The :class:`~networkx.Graph` describing the maxcut problem.
        break_z2: Boolean variable indicating whether the :math:`\mathbb{Z}_2` symmetry of the problem should be
            artificially broken, reducing the number of problem variables by 1.

    """

    def __init__(self, graph: nx.Graph, break_z2: bool = False) -> None:
        self._graph = graph
        self._break_z2 = break_z2

        linear = dict.fromkeys(self._graph.nodes(), 0)
        quadratic: dict[tuple[int, int], int] = {}
        for i, j in self._graph.edges():
            quadratic[(i, j)] = quadratic.get((i, j), 0) + 2
            linear[i] -= 1
            linear[j] -= 1

        bqm = BinaryQuadraticModel(linear, quadratic, 0.0, vartype="BINARY")

        super().__init__(bqm)

        if self._break_z2:
            var_to_fix = cast(int, max(self._bqm.variables, key=self._bqm.degree))
            self.fix_variables({var_to_fix: 1})

        # The average quality and the upper bound are trivial
        self._average_quality = -self._graph.size() / 2
        self._upper_bound = 0

    @property
    def graph(self) -> nx.Graph:
        """The graph of the problem.

        Equals the graph that was given on initialization of :class:`MaxCutInstance` and shouldn't be modified.
        Instead of modifying the graph, the user should instantiate a new object of :class:`MaxCutInstance`.
        """
        return self._graph

    def cut_size(self, bit_str: str) -> int:
        """Calculates the cut size of a solution represented by a bitstring.

        The calculation simply iterates over edges of the graph and adds +1 for each edge cut according to
        the bitstring. Since it uses the original graph, the input bitstring needs to have the same length as
        the graph has nodes.

        Args:
            bit_str: A string of 0's and 1's (or any two distinct characters) representing
                the division of the graph into two sets.

        Returns:
            The number of edges cut.

        Raises:
            ValueError: If the length of the input bitstring isn't equal to the number of nodes of :attr:`_graph`.
            ValueError: If the bitstring contains more than 2 different characters (it doesn't have to be 0's and 1's).

        """
        if len(bit_str) != self._graph.number_of_nodes():
            raise ValueError(
                f"The input bitstring has length {len(bit_str)} whereas the graph "
                f"has {self._graph.number_of_nodes()} nodes."
            )
        if len(set(bit_str)) > 2:  # noqa: PLR2004
            raise ValueError(
                f"The string {bit_str} contains more than 2 distinct characters, "
                f"so it doesn't represent a partition of the graph into 2 distinct sets."
            )
        cut = 0
        for v1, v2 in self._graph.edges():
            if bit_str[self.bqm.variables.index(v1)] != bit_str[self.bqm.variables.index(v2)]:
                cut += 1

        return cut

    def draw_problem(
        self,
        solution_bitstring: str | None = None,
        seed: int | None = None,
        # ``frozenset({1})`` means that we highlight cut edges.
        highlight_edge_by_node_count: frozenset[int] = frozenset({1}),
    ) -> None:
        """The method to draw an instance of maxcut, with the solution highlighted.

        Args:
            solution_bitstring: A bitstring representing a solution to the problem. Selected nodes are represented as
                1's, the other nodes are represented as 0's. This skips nodes that have been fixed, e.g., by breaking
                the Z2 symmetry. The full bitstring including the fixed nodes is restored internally by calling
                :meth:`~iqm.applications.applications.ProblemInstance.restore_fixed_variables_bitstring`.
            seed: The seed to derandomize the layout of the graph.
            highlight_edge_by_node_count: How many of the endnodes of an edge need to be highlighted to also make
                the edge highlighted. For maxcut, this should be ``frozenset({1})`` so that cut edges are highlighted.

        """
        if solution_bitstring is not None:
            full_bitstring = self.restore_fixed_variables_bitstring(solution_bitstring)
        else:
            full_bitstring = None
        draw_problem(
            graph_to_plot=self._graph,
            fixed_vars=frozenset(self._fixed_variables.keys()),
            bitstring=full_bitstring,
            seed=seed,
            highlight_edge_by_node_count=highlight_edge_by_node_count,
        )


class WeightedMaxCutInstance(QUBOInstance):
    r"""The weighted maxcut instance class.

    The weighted maxcut problem is very similar to the standard maxcut, with the only difference being that the edges
    of the graph now have weights. Each cut edge contributes its weight to the quality of the cut.

    Args:
        graph: The :class:`~networkx.Graph` describing the weighted maxcut problem. Each edge of the graph needs to have
            an attribute (something like "weight") storing a number. Specifically, the initialization method looks for
            edge attributes from the tuple :py:data:`~iqm.applications.graph_utils.EDGE_ATTR_PRIORITY` (in order) and
            takes the first one as the edge weight.
        break_z2: Boolean variable indicating whether the :math:`\mathbb{Z}_2` symmetry of the problem should be
            artificially broken, reducing the number of problem variables by 1.

    Raises:
        ValueError: If any of the input graph's edges doesn't have any attribute from the tuple of attributes
            :py:data:`~iqm.applications.graph_utils.EDGE_ATTR_PRIORITY`.
        TypeError: If the weight of any edge is a wrong data type (neither :class:`float` nor :class:`int`).

    """

    def __init__(self, graph: nx.Graph, break_z2: bool = False) -> None:
        self._graph = graph
        self._break_z2 = break_z2

        linear = dict.fromkeys(self._graph.nodes(), 0.0)
        quadratic: dict[tuple[int, int], float | int] = {}
        for n1, n2, edge_data in self._graph.edges(data=True):
            value = _get_attr_with_priority(edge_data, EDGE_ATTR_PRIORITY)

            if value is None:
                raise ValueError(
                    f"The edge between nodes {n1} and {n2} is missing"
                    f" one of the required attributes ({', '.join(EDGE_ATTR_PRIORITY)})."
                )
            if not isinstance(value, (float, int)):
                raise TypeError(
                    f"The edge between nodes {n1} and {n2} has a "
                    f"value of type {type(value).__name__}, expected ``float`` or ``int``."
                )

            quadratic[(n1, n2)] = quadratic.get((n1, n2), 0) + 2 * value
            linear[n1] -= value
            linear[n2] -= value

        bqm = BinaryQuadraticModel(linear, quadratic, 0.0, vartype="BINARY")
        super().__init__(bqm)

        if self._break_z2:
            node_to_fix = cast(int, max(bqm.variables, key=bqm.degree))
            self.fix_variables({node_to_fix: 1})

    @property
    def graph(self) -> nx.Graph:
        """The graph of the problem.

        Equals the graph that was given on initialization of :class:`WeightedMaxCutInstance` and shouldn't be modified.
        Instead of modifying the graph, the user should instantiate a new object of :class:`WeightedMaxCutInstance`.
        """
        return self._graph

    def cut_size(self, bit_str: str) -> float:
        """Calculates the cut size of a solution represented by a bitstring.

        The calculation simply iterates over edges of the graph and adds the weight of each edge cut according to
        the bitstring. Since it uses the original graph, the input bitstring needs to have the same length as
        the graph has nodes.

        Args:
            bit_str: A string of 0's and 1's (or any two distinct characters) representing
                the division of the graph into two sets.

        Returns:
            The weight of edges cut.

        Raises:
            ValueError: If the length of the input bitstring isn't equal to the number of nodes of :attr:`graph`.
            ValueError: If the bitstring contains more than 2 different characters (it doesn't have to be 0's and 1's).

        """
        if len(bit_str) != self._graph.number_of_nodes():
            raise ValueError(
                f"The input bitstring has length {len(bit_str)} whereas the graph "
                f"has {self._graph.number_of_nodes()} nodes."
            )
        if len(set(bit_str)) > 2:  # noqa: PLR2004
            raise ValueError(
                f"The string {bit_str} contains more than 2 distinct characters, "
                f"so it doesn't represent a partition of the graph into 2 distinct sets."
            )
        cut = 0
        for v1, v2, edge_data in self._graph.edges(data=True):
            if bit_str[v1] != bit_str[v2]:
                cut += edge_data["weight"]

        return cut

    def draw_problem(
        self,
        solution_bitstring: str | None = None,
        seed: int | None = None,
        # ``frozenset({1})`` means that we highlight cut edges.
        highlight_edge_by_node_count: frozenset[int] = frozenset({1}),
    ) -> None:
        """The method to draw an instance of weighted maxcut, with the solution highlighted.

        Args:
            solution_bitstring: A bitstring representing a solution to the problem. Selected nodes are represented as
                1's, the other nodes are represented as 0's. This skips nodes that have been fixed, e.g., by breaking
                the Z2 symmetry. The full bitstring including the fixed nodes is restored internally by calling
                :meth:`~iqm.applications.applications.ProblemInstance.restore_fixed_variables_bitstring`.
            seed: The seed to derandomize the layout of the graph.
            highlight_edge_by_node_count: How many of the endnodes of an edge need to be highlighted to also make
                the edge highlighted. For weighted maxcut, this should be ``frozenset({1})`` so that cut edges are
                highlighted.

        """
        if solution_bitstring is not None:
            full_bitstring = self.restore_fixed_variables_bitstring(solution_bitstring)
        else:
            full_bitstring = None
        draw_problem(
            graph_to_plot=self._graph,
            fixed_vars=frozenset(self._fixed_variables.keys()),
            bitstring=full_bitstring,
            seed=seed,
            highlight_edge_by_node_count=highlight_edge_by_node_count,
        )


def maxcut_generator(  # noqa: PLR0913
    n: int,
    n_instances: int,
    *,
    graph_family: Literal["regular", "erdos-renyi"] = "erdos-renyi",
    p: float = 0.5,
    d: int = 3,
    break_z2: bool = False,
    seed: int | None | np.random.Generator = None,
    enforce_connected: bool = False,
    max_iterations: int = 1000,
    weighted: bool = False,
    distribution_of_weights: Literal["uniform", "integers"] = "uniform",
    maximum: int | float = 1.0,
) -> Iterator[MaxCutInstance | WeightedMaxCutInstance]:
    r"""The generator function for generating random maxcut problem instances.

    The generator yields maxcut problem instances using random graphs, created according to the input parameters.
    If ``enforce_connected`` is set to ``True``, then the resulting graphs are checked for connectivity and
    regenerated if the check fails. In that case, the output graphs are not strictly speaking Erdős–Rényi or uniformly
    random regular graphs anymore.

    Args:
        n: The number of nodes of the graph.
        n_instances: The number of maxcut instances to generate.
        graph_family: A string describing the random graph family to generate.
            Possible graph families include 'erdos-renyi' and 'regular'.
        p: For the Erdős–Rényi graph, this is the edge probability. For other graph families, it's ignored.
        d: For the random regular graph, this is the degree of each node in the graph. For other graph families, it's
            ignored.
        break_z2: Optional bool indicating whether the :math:`\mathbb{Z}_2` symmetry should be explicitly broken
            in the problem instances.
        seed: Optional random seed for generating the problem instances.
        enforce_connected: ``True`` iff it is required that the random graphs are connected.
        max_iterations: In case ``enforce_connected`` is ``True``, the function generates random graphs in a ``while``
            loop until it finds a connected one. If it doesn't find a connected one after ``max_iterations``, it raises
            an error.
        weighted: ``True`` iff we want to generate weighted maxcut instances (as opposed to unweighted maxcut).
        distribution_of_weights: A string describing the distribution of the random weights in case weighted maxcu is
            generated. Otherwise ignored.
        maximum: A parameter of the distribution used if the distribution is "uniform" or "integers", otherwise it's
            ignored.

    Yields:
        Problem instances of :class:`MaxCutInstance` randomly constructed in accordance to the input parameters. Or
        instances of :class:`WeightedMaxCutInstance` if ``weighted`` is ``True``.

    """
    rng = np.random.default_rng(seed=seed)
    for _ in range(n_instances):
        g = _generate_desired_graph(graph_family, n, p, d, rng, enforce_connected, max_iterations)

        if weighted:
            if distribution_of_weights == "uniform":
                for u, v in g.edges():
                    # Converted to standard Python ``float`` to avoid typing headaches.
                    g[u][v]["weight"] = float(rng.uniform(0, maximum))
            elif distribution_of_weights == "integers":
                for u, v in g.edges():
                    # Converted to standard Python ``int`` to avoid typing headaches.
                    g[u][v]["weight"] = int(rng.integers(1, int(maximum)))
            else:
                raise ValueError("Invalid distribution of weights. Choose either 'uniform' or 'integers'.")
            yield WeightedMaxCutInstance(g)

        else:
            yield MaxCutInstance(g, break_z2=break_z2)


def greedy_max_cut(max_cut_problem: MaxCutInstance, seed: int | None = None) -> str:
    """Standard greedy algorithm for maxcut problem class.

    Steps:

    1. Start with a random assignment of nodes in two groups.
    2. Iterate over all nodes
    3. For each node, switch it to the other group if it improves the cost function.
    4. Repeat steps 2-3 until no node can be switched.
    5. Return the final assignment.

    Args:
        max_cut_problem: A problem instance of maxcut to be solved (approximately).
        seed: Seed for randomizing the initial assignment of nodes.

    Returns:
        A bitstring solution.

    """
    max_cut_graph = max_cut_problem.graph

    rng = np.random.default_rng(seed)
    current_solution = rng.integers(0, 2, size=max_cut_graph.number_of_nodes()).astype(str).tolist()

    while True:
        for node in max_cut_graph.nodes():
            node_idx = max_cut_problem.bqm.variables.index(node)
            n_cut = sum(
                abs(
                    int(current_solution[node_idx])
                    - int(current_solution[max_cut_problem.bqm.variables.index(neighbor)])
                )
                for neighbor in max_cut_graph.neighbors(node)
            )
            n_uncut = cast(int, max_cut_graph.degree(node)) - n_cut  # ``cast`` to satisfy type checker.
            if n_cut < n_uncut:
                current_solution[node_idx] = "1" if current_solution[node_idx] == "0" else "0"
                break
        else:
            break
    return "".join(current_solution)


def goemans_williamson(max_cut_problem: MaxCutInstance, seed: int | None = None) -> str:
    """Runs the Goemans-Williamson algorithm for maxcut, returning a solution bitstring.

    The Goemans-Williamson is a randomized algorithm for maxcut, with a guaranteed approximation ratio of around
    ``0.87856``. The algorithm was first described in :cite:`Goemans_1995`.

    Steps:

    1. Relax the problem to a semidefinite program (SDP) with each variable represented by a multi-dimensional vector
       on a unit sphere.
    2. Solve the SDP.
    3. Generate a random hyperplane through the origin.
    4. Assign the variables to ``0`` or ``1`` based on which side of the plane their multi-dimensional vector lies.

    Args:
        max_cut_problem: A problem instance of maxcut to be solved (approximately).
        seed: An optional seed to fix the rotation of the random hyperplane.

    Returns:
        A bitstring solution.

    """
    max_cut_graph = max_cut_problem.graph

    adjacency = nx.linalg.adjacency_matrix(max_cut_graph, nodelist=list(max_cut_problem.bqm.variables))
    adjacency = adjacency.toarray()

    size = len(adjacency)
    ones_matrix = np.ones((size, size))
    products = cp.Variable((size, size), PSD=True)
    cut_size = 0.5 * cp.sum(cp.multiply(adjacency, ones_matrix - products))

    objective = cp.Maximize(cut_size)
    constraints = [cp.diag(products) == 1]
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # ``mypy`` complains that this could be ``None``.
    eigenvalues, eigenvectors = eigh(cast(np.ndarray, products.value))
    eigenvalues = np.maximum(eigenvalues, 0)
    diagonal_root = np.diag(np.sqrt(eigenvalues))
    assignment = diagonal_root @ eigenvectors.T

    size = len(assignment)
    rng = np.random.default_rng(seed=seed)
    partition = rng.standard_normal(size=size)
    projections = assignment.T @ partition

    sides = (np.sign(projections).astype(int) + 1) // 2

    nodes = list(max_cut_graph.nodes)
    sides_ordered = sides[nodes]
    result = "".join(map(str, sides_ordered))

    return result


def max_cut_exact(problem: MaxCutInstance | WeightedMaxCutInstance) -> set[str]:
    """Runs an exact solver for the (weighted) maxcut problem, using branch-and-bound.

    Args:
        problem: The (weighted) problem instance to be solved.

    Returns:
        A set containing all of the bitstrings strings that minimize the energy of the problem.

    """
    # Aliases to make code more readable.
    nodes = problem.variables
    n = len(nodes)

    if n == 0:
        return {""}

    # Weighted, symmetric adjacency matrix. Computed manually, so that we can use `_get_attr_with_priority`.
    adjacency_matrix = np.zeros((n, n))
    weighted = isinstance(problem, WeightedMaxCutInstance)
    for i, j, data in problem.graph.edges(data=True):
        w = _get_attr_with_priority(data, EDGE_ATTR_PRIORITY) if weighted else 1
        a, b = nodes.index(i), nodes.index(j)
        adjacency_matrix[a, b] += w
        adjacency_matrix[b, a] += w

    # Place high (weighted) degree nodes first -> tighter bounds sooner. This reordering needs to be undone later.
    order = np.argsort(-np.abs(adjacency_matrix).sum(axis=1))
    adjacency_matrix = adjacency_matrix[np.ix_(order, order)]

    # ``tail[d]`` = sum of POSITIVE edge weights among nodes with index >= d.
    # Basically, this is an upper bound on how much can still be cut.
    tail = np.zeros(n + 1)
    for d in range(n - 1, -1, -1):
        tail[d] = tail[d + 1] + np.maximum(adjacency_matrix[d, d + 1 :], 0.0).sum()

    best = 0.0
    sols = []  # Incumbent list for cuts whose quality equals ``best``

    # As we solve the problem, we assign nodes to two groups (called 0 and 1 here).
    # ``contrib_0[v]`` = total edge weight between ``v`` and already-placed side-0 nodes
    # ``contrib_1[v]`` = total edge weight between ``v`` and already-placed side-1 nodes
    # These contribution arrays begin with all zeros, because we begin with all nodes unassigned.
    contrib_0 = np.zeros(n)
    contrib_1 = np.zeros(n)
    assignment = np.zeros(n, dtype=np.int8)  # To be gradually filled with the node assignment.

    def recurse(d: int, cur_cut: int | float) -> None:
        """Recursively explore assignment from node ``d`` onward.

        Args:
            d: The number of already-placed nodes. As the nodes are labelled by integers starting from 0, this is also
                equal to the label of the node that we're going to place next. That is, unless ``d`` is equal to the
                number of nodes, in which case there is no mode nodes to be placed.
            cur_cut: Cut value realized among nodes the first ``d`` nodes.

        """
        nonlocal best, sols
        tol = 1e-9 * max(1.0, abs(best))

        if d == n:
            if cur_cut > best + tol:
                best = cur_cut
                sols = [assignment.copy()]  # Strictly better cut -> discard old incumbent solutions.
            elif abs(cur_cut - best) <= tol:
                sols.append(assignment.copy())  # Equally good (up to tolerance) -> add to incumbents.
            return

        # Next, we construct an upper bound on any cut reachable from the current state.
        # The upper bound has three contributions:
        # ``cur_cut``: Already-realized cut among placed nodes.
        # ``tail[d]``: Upper bound on cut among unplaced nodes.
        # ``to_placed``: For each unplaced node, how much can be cut between it and placed nodes (on either side).
        to_placed = np.maximum(contrib_0[d:], contrib_1[d:]).sum()
        if cur_cut + tail[d] + to_placed < best - tol:
            return

        weights_to_d = adjacency_matrix[:, d]  # Edge weights from node ``d`` to all nodes.

        # Z2 symmetry break. The first node goes to side 0.
        if d == 0:
            assignment[0] = 0
            contrib_0[:] += weights_to_d  # Register the first node as a placed side-0 node
            recurse(1, cur_cut)
            contrib_0[:] -= weights_to_d  # Undo before returning (backtrack).
            return

        # The ``branches`` variable contains the two options for placing node ``d`` as the following information:
        # 1. The side it's placed on.
        # 2. By how much the current cut value should be changed.
        # 3. Which contribution array should be updated as a result of placing the node.
        branches = ((0, contrib_1[d], contrib_0), (1, contrib_0[d], contrib_1))
        # Explore the higher-cut side first so a strong incumbent appears early, making the pruning fire more often.
        if contrib_0[d] > contrib_1[d]:
            branches = branches[::-1]

        for side_to_assign_d, current_cut_change, array_to_update in branches:
            assignment[d] = side_to_assign_d
            array_to_update[:] += weights_to_d  # Update the correct contribution array.
            recurse(d + 1, cur_cut + current_cut_change)
            array_to_update[:] -= weights_to_d  # Backtrack.

    recurse(0, 0.0)

    # Translate solutions back to the correct node order.
    inv = np.argsort(order)
    set_of_solutions = {"".join(map(str, sol[inv])) for sol in sols}
    # Restore the half of the solution space removed by the Z2 symmetry break.
    _flip = str.maketrans("01", "10")
    set_of_solutions = {s.translate(_flip) for s in set_of_solutions} | set_of_solutions
    return set_of_solutions
