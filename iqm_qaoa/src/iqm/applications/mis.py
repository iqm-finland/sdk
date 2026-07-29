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
"""Module containing the MIS problem instance class.

Two solvers for MIS problems are also provided:

- A simple and fast greedy solver.
- Exact solver based on *Bron-Kerbosch* algorithm to find a graph's cliques.

Example:

    .. code-block:: python

        from iqm.applications.mis import MISInstance
        from iqm.applications.mis import greedy_mis
        from iqm.applications.mis import bron_kerbosch

        my_mis_instance = MISInstance(my_graph, penalty = 2)  # Problem instance from a graph.
        my_mis_instance.constraints_checker("101011000")  # Check if this corresponds to an independent set.

        greedy_solution = greedy_mis(my_instance)  # Solution bitstring obtained from greedy.
        bk_solution = bron_kerbosch(my_instance)  # Solution bitstring obtained from BK.

"""

from collections.abc import Iterator, Mapping
from itertools import combinations
from typing import Literal, cast

from dimod import BinaryQuadraticModel, ConstrainedQuadraticModel
from dimod.typing import Variable
from iqm.applications.graph_utils import (
    NODE_ATTR_PRIORITY,
    _generate_desired_graph,
    _get_attr_with_priority,
    draw_problem,
)
from iqm.applications.qubo import ConstrainedQuadraticInstance
import networkx as nx
import numpy as np


class ISInstance(ConstrainedQuadraticInstance):
    """The instance class for independent set problems on a graph with a custom cost function.

    The objective of this class of problems is selecting a subset of nodes in a graph, such that no two nodes in
    the selected subset are connected by an edge, while minimizing some cost function. This class can be instantiated by
    giving it a graph, a penalty to the cost function to be incurred for each violation of the constraint (when
    the problem is converted to QUBO) and an objective function as a :class:`~dimod.BinaryQuadraticModel`. This class is
    not intended to be instantiated directly, but rather by its subclasses :class:`~iqm.applications.mis.MISInstance`
    and :class:`~iqm.applications.mis.MaximumWeightISInstance`. But if the user does the work of defining their own
    ``objective`` function, there is no harm in instantiating an object of :class:`ISInstance`.

    Args:
        graph: The graph from which independent sets are to be picked.
        penalty: The penalty to be incurred per each edge present in the solution. This is needed when the problem
            formulation is transformed into QUBO.
        objective: The objective function to be minimized by the independent set.
        allow_custom_var_names: Determines whether it's allowed for ``graph`` to have nodes labelled by anything other
            than consecutive integers starting at 0.

    """

    def __init__(
        self,
        graph: nx.Graph,
        penalty: float | int,
        objective: BinaryQuadraticModel,
        allow_custom_var_names: bool = False,
    ) -> None:
        self._graph = graph
        self._objective = objective
        cqm = ConstrainedQuadraticModel()
        cqm.set_objective(self._objective)
        for u, v in self._graph.edges():
            cqm.add_constraint_from_iterable([(u, v, 1)], "==", rhs=0)
        super().__init__(cqm, penalty=penalty, allow_custom_var_names=allow_custom_var_names)

    @property
    def graph(self) -> nx.Graph:
        """The graph corresponding to the problem.

        Equals the graph that was given on initialization of :class:`MISInstance` and shouldn't be modified. Instead of
        modifying the graph, the user should instantiate a new object of :class:`MISInstance`.
        """
        return self._graph

    def fix_variables(
        self,
        variables: list[int] | list[Variable] | dict[int, int] | dict[Variable, int],
        original_labels: bool = False,
    ) -> None:
        """A method to fix (assign) some of the problem variables.

        When a variable is fixed to 1, all of its neighboring variables are fixed to 0 (because of the constraints).

        Args:
            variables: Either a list of variables (which get all fixed to the value 1) or a dictionary with keys equal
                to the variables to fix and whose values are equal to the values to fix them to (either 1 or 0).
            original_labels: Is the input using the original variable labels (in case they were labeled with e.g.,
                strings or non-consecutive integers)? This refers to the fact that the input to the problem instance
                (in this case the input ``graph`` of :class:`ISInstance`) may have variables labelled by any hashable,
                aliased as :class:`dimod.typing.Variable` in :mod:`dimod`. On initialization of the problem instance,
                the problem variables are re-labeled by consecutive integers starting from 0 (for internal consistancy).
                If ``original_labels`` is set to ``True``, this method :meth:`fix_variables` is expected to receive
                the original names of the variables instead of their new integer labels.

        Raises:
            ValueError: If the user is trying to fix two neighbouring nodes to 1, violating the independence
                constraint.
            ValueError: If one of the variables has already been fixed previously to a different value.

        """
        if isinstance(variables, list):
            variables = dict.fromkeys(variables, 1)

        if original_labels:
            variables = {self.orig_to_new_labels[var_orig_label]: value for var_orig_label, value in variables.items()}
        else:
            for var_label in variables:
                if not isinstance(var_label, int):
                    raise TypeError(
                        f"When `original_labels` is set to False (default), the types of variables to fix "
                        f"need to be `int`. The variable {var_label} has type {type(var_label)}."
                    )

        # At this point ``variables`` should have the type ``dict[int, int]``, but mypy doesn't know that.
        variables = cast(dict[int, int], variables)

        for var in variables.keys():
            if var in self._fixed_variables and self._fixed_variables[var] != variables[var]:
                raise ValueError(
                    f"The variable {var} has been fixed previously to {self._fixed_variables[var]}, "
                    f"but now it's attempted to be fixed to {variables[var]}."
                )
        for node1, node2 in combinations(variables.keys(), 2):
            if self._graph.has_edge(node1, node2) and variables[node1] == 1 and variables[node2] == 1:
                raise ValueError("Can't fix two neighboring nodes to 1 in an independent set problem.")
        # If a variables is fixed to 1, all its neighbors need to be fixed to 0.
        to_be_fixed_as_0 = set()
        for var in variables:
            if variables[var] == 1:
                var_neighbors = self._graph.neighbors(self.new_to_orig_labels[var])
                for neighbor in var_neighbors:
                    if self.orig_to_new_labels[neighbor] in self._cqm.variables:
                        to_be_fixed_as_0.add(self.orig_to_new_labels[neighbor])
        # This needs to be done in a separate step, so that the dictionary isn't modified in the for loop above.
        for nb in to_be_fixed_as_0:
            variables[nb] = 0

        self._fixed_variables.update(variables)

        # If a variable is fixed, remove all its edges (i.e., constraints that it appears in).
        for constraint in set(self._cqm.constraint_labels):
            if set(self._cqm.constraints[constraint].lhs.variables) & set(variables.keys()):
                self._cqm.remove_constraint(constraint)
        self._cqm.fix_variables(cast(Mapping[Variable, int], variables))

    def _recalculate_bqm(self) -> BinaryQuadraticModel:
        """The function calculating the BQM is relatively simple for independent set problems."""
        bqm_to_return = self._objective.copy()
        for i, j in self._graph.edges():
            bqm_to_return.add_quadratic(i, j, self._penalty)
        bqm_to_return.fix_variables(cast(Mapping[Variable, int], self._fixed_variables))

        return bqm_to_return

    def _induced_subgraph_from_bitstring(self, bit_str: str) -> nx.Graph:
        """Helper method that takes a bitstring representing a solution and returns the graph induced from it.

        The input bitstring indicates which nodes from the original :attr:`_graph` are selected. Selecting a subset
        of nodes defines a subgraph, and this function is used to return the subgraph.

        Args:
            bit_str: The bitstring which defines a subgraph of :attr:`_graph`.

        Returns:
            A :class:`~networkx.Graph` corresponding to the bitstring.

        """
        selected_nodes = [self.new_to_orig_labels[i] for i, c in enumerate(bit_str) if c == "1"]
        induced_subgraph = self._graph.subgraph(selected_nodes).copy()
        return induced_subgraph

    def draw_problem(
        self,
        solution_bitstring: str | None = None,
        seed: int | None = None,
    ) -> None:
        """The method to draw an instance of independent set, with the solution highlighted.

        Args:
            solution_bitstring: A bitstring representing a solution to the problem. Selected nodes are represented as
                1's, the other nodes are represented as 0's. This skips nodes that have been fixed. The full bitstring
                including the fixed nodes is restored internally by calling
                :meth:`~iqm.applications.applications.ProblemInstance.restore_fixed_variables_bitstring`.
            seed: The seed to derandomize the layout of the graph.

        """
        if solution_bitstring is not None:
            full_bitstring = self.restore_fixed_variables_bitstring(solution_bitstring)
        else:
            full_bitstring = None
        draw_problem(
            graph_to_plot=self._graph,
            orig_to_new_mapping=self.orig_to_new_labels,
            fixed_vars=frozenset(self._fixed_variables.keys()),
            bitstring=full_bitstring,
            seed=seed,
            # ``frozenset({2})`` means that we highlight edges violating the independence constraint.
            highlight_edge_by_node_count=frozenset({2}),
        )


class MISInstance(ISInstance):
    r"""The instance class for maximum independent set problems.

    The maximum independent set problem refers to finding the largest subset of nodes of a graph, such that no nodes
    in the subset are connected by an edge. It is completely equivalent to finding the largest clique on the complement
    graph. The class is initialized by initializing its parent class :class:`~iqm.applications.mis.ISInstance` with
    a simple objective function (aiming at maximizing the number of "selected" nodes).

    Args:
        graph: The :class:`~networkx.Graph` describing the MIS problem.
        penalty: The penalty to be incurred per each edge present in the solution, sometimes referred to as
            :math:`\lambda` in the literature. The higher it is, the less likely the algorithm is to include an edge in
            the solution. It needs to be set above 1 to insure that the solution is a maximum independent set. It's
            typically set at 2. At 1, the correct solution will be degenerate with non-independent sets.
        allow_custom_var_names: Determines whether it's allowed for ``graph`` to have nodes labelled by anything other
            than consecutive integers starting at 0.

    """

    def __init__(self, graph: nx.Graph, penalty: float = 1, allow_custom_var_names: bool = False) -> None:
        objective = BinaryQuadraticModel(vartype="BINARY")
        objective.add_linear_from(dict.fromkeys(graph.nodes(), -1))
        super().__init__(graph, penalty, objective, allow_custom_var_names)
        self._upper_bound = 0  # The worse case (not violating the constraints) is selecting no nodes.

    @property
    def best_quality(self) -> float:
        """The best quality for the MIS problem.

        Can be calculated together with other properties of the solution using the exhaustive Bron-Kerbosch algorithm.
        """
        if self._best_quality is None:
            self.solve_with_bron_kerbosch()
        if self._best_quality is None:  # For if :meth:`solve_with_bron_kerbosch` for some reason fails.
            raise ValueError("Expected 'best_quality' to be set, but it is ``None``.")
        return self._best_quality

    @property
    def lowest_quality_bitstrings(self) -> set[str]:
        """The best bitstrings for the MIS problem.

        Can be calculated together with other properties of the solution using the exhaustive Bron-Kerbosch algorithm.
        """
        if self._lowest_quality_bitstrings is None:
            self.solve_with_bron_kerbosch()
        if self._lowest_quality_bitstrings is None:  # For if :meth:`solve_with_bron_kerbosch` for some reason fails.
            raise ValueError("Expected 'lowest_quality_bitstrings' to be set, but it is ``None``.")
        return self._lowest_quality_bitstrings

    @property
    def lower_bound(self) -> float:
        """The lowest quality value among all possible bitstrings.

        Can be calculated together with other properties of the solution using the exhaustive Bron-Kerbosch algorithm.
        """
        if self._lower_bound is None:
            self.solve_with_bron_kerbosch()
        if self._lower_bound is None:  # For if :meth:`solve_with_bron_kerbosch` for some reason fails.
            raise ValueError("Expected 'lower_bound' to be set, but it is ``None``.")
        return self._lower_bound

    def solve_with_bron_kerbosch(self) -> None:
        """Solve the MIS problem using Bron-Kerbosch algorithm and save some properties of the solution."""
        solution_bitstrings = bron_kerbosch(self, all_solutions=True)
        # The output of ``bron_kerbosch`` here should be a set, but ``mypy`` doesn't see that.
        self._lowest_quality_bitstrings = solution_bitstrings  # type: ignore
        # The solution bitstrings should all have the same ``loss``, so we can use an arbitrary one.
        self._lower_bound = self.loss(next(iter(solution_bitstrings)))
        self._best_quality = self.loss(next(iter(solution_bitstrings)))

    @property
    def highest_quality_bitstrings(self) -> set[str]:
        """The worst bitstring for the MIS problem, determined trivially.

        The worst bitstring satisfying all constraints is always selecting no nodes.
        """
        return {"0" * self.dim}

    def fix_constraint_violation_bitstring(self, bit_str: str) -> str:
        """Postprocessing function that fixes a single bitstring, making it satisfy the constraints.

        It works in the following way:

        1. Get the subgraph induced by the bitstring ``bit_str``.
        2. Find the node with the highest degree.
        3. Remove the node from the subgraph (i.e., flip the corresponding bit from "1" to "0").
        4. Repeat 2-3 until the subgraph contains no edges.
        5. Return the bitstring corresponding to the remaining subgraph (which is an independent subset of the original
           graph).

        For ``penalty = 1``, this guarantees that the output bitstring has energy at least as low as the input
        bitstring. For ``penalty > 1``, the energy is expected to be even lower.

        Args:
            bit_str: The bitstring to be modified to satisfy the independence constraint.

        Returns:
            The modified bitstring, corresponding to an independent set of the problem graph.

        """
        # If the bitstring already satisfies the constraints, just return it unchanged and don't waste any time.
        if self.constraints_checker(bit_str):
            return bit_str

        induced_subgraph = self._induced_subgraph_from_bitstring(bit_str)
        while induced_subgraph.number_of_edges() > 0:
            highest_degree_node = max(induced_subgraph.degree, key=lambda x: x[1])[0]
            induced_subgraph.remove_node(highest_degree_node)

        result = ["0"] * self._graph.number_of_nodes()
        for pos in list(induced_subgraph.nodes()):
            result[pos] = "1"  # Set the corresponding position to '1'
        return "".join(result)


class MaximumWeightISInstance(ISInstance):
    r"""The instance class for maximum-weight independent set problems.

    The maximum-weight independent set problem refers to finding a subset of nodes of a graph, such that no nodes in
    the subset are connected by an edge and sum of the weights of the nodes in the subset is maximized. The class is
    initialized by initializing its parent class :class:`~iqm.applications.mis.ISInstance` with a custom objective
    function (carrying the weights of the graph nodes).

    Args:
        graph: The :class:`~networkx.Graph` describing the maximum-weight independent set problem. Each node of the
            graph needs to have an attribute (something like "weight") storing a number. Specifically, the
            initialization method looks fornode attributes from the tuple
            :py:data:`~iqm.applications.graph_utils.NODE_ATTR_PRIORITY` (in order) and takes the first one as the node
            weight.
        penalty: The penalty to be incurred per each edge present in the solution, sometimes referred to as
            :math:`\lambda` in the literature. The higher it is, the less likely the algorithm is to include an edge in
            the solution. This is needed when the problem formulation is transformed into QUBO.
        allow_custom_var_names: Determines whether it's allowed for ``graph`` to have nodes labelled by anything other
            than consecutive integers starting at 0.

    Raises:
        ValueError: If any of the input graph's nodes doesn't have any attribute from the tuple of attributes
            :py:data:`~iqm.applications.graph_utils.NODE_ATTR_PRIORITY`.
        TypeError: If the weight of any node is a wrong data type (neither :class:`float` nor :class:`int`).

    """

    def __init__(self, graph: nx.Graph, penalty: float | int, allow_custom_var_names: bool = False) -> None:
        # Define the objective function to minimize
        objective = BinaryQuadraticModel(vartype="BINARY")

        for node, data in graph.nodes(data=True):
            value = _get_attr_with_priority(data, NODE_ATTR_PRIORITY)

            if value is None:
                raise ValueError(
                    f"The node {node} is missing one of the required attributes ({', '.join(NODE_ATTR_PRIORITY)})."
                )

            if not isinstance(value, (float, int)):
                raise TypeError(
                    f"The local term at node {node} has a "
                    f"value of type {type(value).__name__}, expected ``float`` or ``int``."
                )

            objective.add_linear(node, -value)

        super().__init__(graph, penalty, objective, allow_custom_var_names=allow_custom_var_names)

        # If the nodes have only non-negative weight (i.e., ``objective`` has only non-positive entries) ...
        if max(objective.linear) <= 0:
            self._upper_bound = 0  # ... the worst case is selecting no nodes.

    @property
    def highest_quality_bitstrings(self) -> set[str]:
        """The worst bitstring(s) for the MWIS problem, determined trivially if all weights are non-negative."""
        if min(self._objective.linear.values()) >= 0:
            return {"0" * self.dim}
        else:
            return super().highest_quality_bitstrings  # This will trigger the brute-forcing ``initialize_properties``.


def greedy_mis(mis_problem: MISInstance | nx.Graph) -> str:
    """Standard greedy algorithm for maximum independent set problem class.

    Steps:

    1. Pick the lowest-degree node in the graph.
    2. Add it to the independent set.
    3. Remove it and all its neighbors from the graph.
    4. Repeat steps 1-3 until the graph is empty.
    5. Return the independent set.

    Args:
        mis_problem: A problem instance of maximum independent set or a :class:`~networkx.Graph`.

    Returns:
        A bitstring solution.

    """
    if isinstance(mis_problem, MISInstance):
        mis_problem = mis_problem.graph
    elif not isinstance(mis_problem, nx.Graph):
        raise TypeError(
            f"Supported input is either a NetworkX graph or a MISInstance. Given type: {type(mis_problem).__name__}"
        )

    working_graph = mis_problem.copy()
    independent = set()
    while working_graph.number_of_nodes() > 0:
        lowest_degree_node = min(working_graph.degree, key=lambda x: x[1])
        independent.add(lowest_degree_node[0])
        to_remove = {lowest_degree_node[0]} | set(working_graph.neighbors(lowest_degree_node[0]))
        working_graph.remove_nodes_from(to_remove)
    bitstring = ["0"] * mis_problem.number_of_nodes()
    for pos in list(independent):
        bitstring[pos] = "1"
    return "".join(bitstring)


def bron_kerbosch(mis_problem: MISInstance | nx.Graph, all_solutions: bool = False) -> str | set[str]:
    """Bron-Kerbosch algorithm for finding the maximum independent set.

    The algorithm finds all maximal cliques in a graph recursively. Cliques in complement graph correspond
    to independent sets in the problem graph. For details see `find_cliques — NetworkX documentation
    <https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.clique.find_cliques.html>`_
    (and the references therein). If ``all_solutions`` is False (default), return one maximum independent set as
    a bitstring. If ``all_solutions`` is True, return a set of bitstrings corresponding to all maximum independent sets
    found (not all **maximal** independent sets).

    Args:
        mis_problem: A problem instance of maximum independent set or a :class:`~networkx.Graph`.
        all_solutions: Should the algorithm return all solutions?

    Returns:
        A bitstring solution.

    """
    if isinstance(mis_problem, MISInstance):
        mis_problem = mis_problem.graph
    elif not isinstance(mis_problem, nx.Graph):
        raise TypeError(
            f"Supported input is either a NetworkX graph or a MISInstance. Given type: {type(mis_problem).__name__}"
        )

    if mis_problem.number_of_nodes() == 0:  # Check for empty graph.
        return "" if not all_solutions else {""}  # Return an empty string (or a set with an empty string).

    max_size = 0
    max_sets = []

    # ``nx.find_cliques()`` is an iterator, so we want to go through it in one pass.
    for clique in nx.find_cliques(nx.complement(mis_problem)):
        clique_size = len(clique)
        if clique_size > max_size:
            # New max found: discard old, start new list.
            max_size = clique_size
            max_sets = [clique]
        elif clique_size == max_size:
            # Another clique of same max size: add it.
            max_sets.append(clique)

    def _to_bitstring(nodes: list[int]) -> str:
        bitstring = ["0"] * mis_problem.number_of_nodes()
        for n in nodes:
            bitstring[n] = "1"
        return "".join(bitstring)

    if not all_solutions:
        # Return just one max independent set.
        return _to_bitstring(max_sets[0])

    # Return *all* max independent sets as bitstrings.
    return {_to_bitstring(s) for s in max_sets}


def mis_generator(  # noqa: PLR0913
    n: int,
    n_instances: int,
    *,
    graph_family: Literal["regular", "erdos-renyi"] = "erdos-renyi",
    p: float = 0.5,
    d: int = 3,
    seed: int | None | np.random.Generator = None,
    enforce_connected: bool = False,
    max_iterations: int = 1000,
    penalty: int = 1,
) -> Iterator[MISInstance]:
    r"""The generator function for generating random MIS problem instances.

    The generator yields MIS problem instances using random graphs, created according to the input parameters. If
    ``enforce_connected`` is set to ``True``, then the resulting graphs are checked for connectivity and regenerated if
    the check fails. In that case, the output graphs are not strictly speaking Erdős–Rényi or uniformly random regular
    graphs anymore.

    Args:
        n: The number of nodes of the graph.
        n_instances: The number of MIS instances to generate.
        graph_family: A string describing the random graph family to generate.
            Possible graph families include 'erdos-renyi' and 'regular'.
        p: For the Erdős–Rényi graph, this is the edge probability. For other graph families, it's ignored.
        d: For the random regular graph, this is the degree of each node in the graph. For other graph families, it's
            ignored.
        seed: Optional random seed for generating the problem instances.
        enforce_connected: ``True`` iff it is required that the random graphs are connected.
        max_iterations: In case ``enforce_connected`` is ``True``, the function generates random graphs in a ``while``
            loop until it finds a connected one. If it doesn't find a connected one after ``max_iterations``, it raises
            an error.
        penalty: The penalty to the energy for violating the independence constraint.

    Yields:
        Problem instances of :class:`MISInstance` randomly constructed in accordance to the input parameters.

    """
    for _ in range(n_instances):
        g = _generate_desired_graph(graph_family, n, p, d, seed, enforce_connected, max_iterations)

        yield MISInstance(g, penalty=penalty)
