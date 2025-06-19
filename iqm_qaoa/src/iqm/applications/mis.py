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

from itertools import combinations

from dimod import BinaryQuadraticModel, ConstrainedQuadraticModel
from dimod.typing import Variable
from iqm.applications.qubo import ConstrainedQuadraticInstance, relabel_graph_nodes
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

    """

    def __init__(self, graph: nx.Graph, penalty: float | int, objective: BinaryQuadraticModel) -> None:
        self._graph, self.orig_to_new_labels, self.new_to_orig_labels = relabel_graph_nodes(graph)
        self._objective = objective
        cqm = ConstrainedQuadraticModel()
        cqm.set_objective(self._objective)
        for u, v in self._graph.edges():
            cqm.add_constraint_from_iterable([(u, v, 1)], "==", rhs=0)
        super().__init__(cqm, penalty=penalty)

    @property
    def graph(self) -> nx.Graph:
        """The graph corresponding to the problem.

        Equals the graph that was given on initialization of :class:`MISInstance` and shouldn't be modified. Instead of
        modifying the graph, the user should instantiate a new object of :class:`MISInstance`.
        """
        return self._graph

    def fix_variables(self, variables: list[Variable] | dict[Variable, int]) -> None:
        """A method to fix some of the problem variables.

        When a variable is fixed to 1, all of its neighboring variables are fixed to 0 (because of the constraints).

        Args:
            variables: Either a list of variables (which get all fixed to the value 1) or a dictionary with keys equal
                to the variables to fix and whose values are equal to the values to fix them to (either 1 or 0).

        Raises:
            ValueError: If the user is trying to fix two neighbouring nodes to 1, violating the independence
                constraint.
            ValueError: If one of the variables has already been fixed previously to a different value.

        """
        if isinstance(variables, list):
            variables = {var: 1 for var in variables}
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
                var_neighbors = self._graph.neighbors(var)
                for neighbor in var_neighbors:
                    if neighbor in self._cqm.variables:
                        to_be_fixed_as_0.add(neighbor)
        # This needs to be done in a separate step, so that the dictionary isn't modified in the iteration.
        for nb in to_be_fixed_as_0:
            variables[nb] = 0

        self._fixed_variables.update(variables)

        # If a variable is fixed, remove all its edges (i.e., constraints that it appears in).
        for constraint in set(self._cqm.constraint_labels):
            if set(self._cqm.constraints[constraint].lhs.variables) & set(variables.keys()):
                self._cqm.remove_constraint(constraint)
        self._cqm.fix_variables(variables)

    def _recalculate_bqm(self) -> BinaryQuadraticModel:
        """The function calculating the BQM is relatively simple for independent set problems."""
        bqm_to_return = self._objective.copy()
        for i, j in self._graph.edges():
            bqm_to_return.add_quadratic(i, j, self._penalty)
        bqm_to_return.fix_variables(self._fixed_variables)

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
        selected_nodes = [i for i, c in enumerate(bit_str) if c == "1"]
        induced_subgraph = self._graph.subgraph(selected_nodes).copy()
        return induced_subgraph


# pylint: disable=anomalous-backslash-in-string
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

    """

    def __init__(self, graph: nx.Graph, penalty: float | int = 1) -> None:
        objective = BinaryQuadraticModel(-np.eye(graph.number_of_nodes(), dtype=int), vartype="BINARY")
        super().__init__(graph, penalty, objective)
        self._upper_bound = 0  # The worse case (not violating the constraints) is selecting no nodes.

    @property
    def best_quality(self) -> float:
        """The best quality for the MIS problem, calculated using the Bron-Kerbosch algorithm.

        Instead of brute-forcing over all possible bitstrings, this uses an exhaustive algorithm that finds
        the best solution more efficiently (although it also has exponential scaling).
        """
        return self.loss(bron_kerbosch(self))

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


# pylint: disable=anomalous-backslash-in-string
class MaximumWeightISInstance(ISInstance):
    r"""The instance class for maximum-weight independent set problems.

    The maximum-weight independent set problem refers to finding a subset of nodes of a graph, such that no nodes in
    the subset are connected by an edge and sum of the weights of the nodes in the subset is maximized. The class is
    initialized by initializing its parent class :class:`~iqm.applications.mis.ISInstance` with a custom objective
    function (carrying the weights of the graph nodes).

    Args:
        graph: The :class:`~networkx.Graph` describing the maximum-weight independent set problem. Each node has to have
            an attribute ``weight`` storing a number.
        penalty: The penalty to be incurred per each edge present in the solution, sometimes referred to as
            :math:`\lambda` in the literature. The higher it is, the less likely the algorithm is to include an edge in
            the solution. This is needed when the problem formulation is transformed into QUBO.

    Raises:
        ValueError: If any node of the input ``graph`` is missing the ``weight`` attribute.
        TypeError: If the weight of any node is a wrong data type (neither :class:`float` nor :class:`int`).

    """

    def __init__(self, graph: nx.Graph, penalty: float | int) -> None:
        # Check that all nodes of the graph have a correct weight assigned.
        for node, data in graph.nodes(data=True):
            if "weight" not in data:
                raise ValueError(f"Node {node} is missing the 'weight' attribute.")
            if not isinstance(data["weight"], float | int):
                raise TypeError(
                    f"Node {node} has a 'weight' of type {type(data['weight']).__name__}, expected"
                    f" ``float`` or ``int``."
                )

        # Define the objective function to minimize
        obj_matrix = np.zeros((graph.number_of_nodes(), graph.number_of_nodes()))
        for node in graph.nodes():
            obj_matrix[node, node] = -graph.nodes[node]["weight"]
        objective = BinaryQuadraticModel(obj_matrix, vartype="BINARY")

        super().__init__(graph, penalty, objective)

        # If the nodes have only positive weight (i.e., ``obj_matrix`` has only non-negative entries) ...
        if (obj_matrix >= 0).all():
            self._upper_bound = 0  # ... the worst case is selecting no nodes.


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


# pylint: disable = line-too-long
def bron_kerbosch(mis_problem: MISInstance | nx.Graph) -> str:
    """Bron-Kerbosch algorithm for finding the maximum independent set.

    The algorithm finds all maximal cliques in a graph recursively. Cliques in complement graph correspond
    to independent sets in the problem graph. We pick the largest of these.
    For details see `find_cliques — NetworkX documentation <https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.clique.find_cliques.html>`_
    (and the references therein):



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

    independent_sets = nx.find_cliques(nx.complement(mis_problem))
    maximum_independent_set: list[int] = max(independent_sets, key=len, default=[])

    bitstring = ["0"] * mis_problem.number_of_nodes()
    for pos in maximum_independent_set:
        bitstring[pos] = "1"
    return "".join(bitstring)
