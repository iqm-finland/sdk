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
"""This module contains the maxcut problem instance class and related functions.

Contains the iterator function :func:`maxcut_generator` which yields random instances of :class:`MaxCutInstance`, useful
for applications such as calculating the Q-score. Also contains two classical solvers for maxcut problems:

- A simple and fast greedy algorithm.
- A solver based on the standard *Goemans-Williamson* :cite:`Goemans_1995`, fast and with a performance guarantee.

Example:

    .. code-block:: python

        from iqm.applications.maxcut import MaxCutInstance
        from iqm.applications.maxcut import maxcut_generator
        from iqm.applications.maxcut import goemans_williamson
        from iqm.applications.maxcut import greedy_maxcut

        for my_instance in maxcut_generator(graph_size, n_instances):  # Generates problem instances.
            gw_solution = goemans_williamson(my_instance)  # Solution bitstring obtained from GW.
            greedy_solution = greedy_maxcut(my_instance)  # Solution bitstring obtained from greedy.
            my_instance.cut_size(gw_solution)  # Check the GW solution cut size.

        my_maxcut_instance = MaxCutInstance(my_graph)  # Problem instance from a graph.

"""

from collections.abc import Iterator
import random

import cvxpy as cp
from dimod import BinaryQuadraticModel
from iqm.applications.qubo import QUBOInstance, relabel_graph_nodes
import networkx as nx
import numpy as np
from scipy.linalg import eigh


# pylint: disable=anomalous-backslash-in-string
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

    Raises:
        ValueError: If the input graph's nodes aren't labelled by integers starting from 0.

    """

    def __init__(self, graph: nx.Graph, break_z2: bool = False) -> None:
        self._graph, self.orig_to_new_labels, self.new_to_orig_labels = relabel_graph_nodes(graph)
        self._break_z2 = break_z2
        qubo_mat = np.zeros((self._graph.number_of_nodes(), self._graph.number_of_nodes()), dtype=int)
        for i, j in self._graph.edges():
            qubo_mat[i, j] += 2
            qubo_mat[i, i] += -1
            qubo_mat[j, j] += -1
        bqm = BinaryQuadraticModel(qubo_mat, vartype="BINARY")
        super().__init__(bqm)

        if self._break_z2:
            node_to_fix = max(bqm.variables, key=bqm.degree)
            self.fix_variables({node_to_fix: 1})

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
        if len(set(bit_str)) > 2:
            raise ValueError(
                f"The string {bit_str} contains more than 2 distinct characters, "
                f"so it doesn't represent a partition of the graph into 2 distinct sets."
            )
        cut = 0
        for v1, v2 in self._graph.edges():
            if bit_str[v1] != bit_str[v2]:
                cut += 1

        return cut


# pylint: disable=anomalous-backslash-in-string
class WeightedMaxCutInstance(QUBOInstance):
    r"""The weighted maxcut instance class.

    The weighted maxcut problem is very similar to the standard maxcut, with the only difference being that the edges
    of the graph now have weights. Each cut edge contributes its weight to the quality of the cut.

    Args:
        graph: The :class:`~networkx.Graph` describing the weighted maxcut problem. Each edge of the graph needs to have
            an attribute called ``weight`` storing a number.
        break_z2: Boolean variable indicating whether the :math:`\mathbb{Z}_2` symmetry of the problem should be
            artificially broken, reducing the number of problem variables by 1.

    Raises:
        ValueError: If the input graph's nodes aren't labelled by integers starting from 0.
        ValueError: If the input graph's edges don't all have an attribute ``weight``.
        TypeError: If the weight of any node is a wrong data type (neither :class:`float` nor :class:`int`).

    """

    def __init__(self, graph: nx.Graph, break_z2: bool = False) -> None:
        if set(graph.nodes) != set(range(len(graph))):
            raise ValueError("Graph nodes should be labelled by integers starting with 0.")
        for n1, n2, data in graph.edges(data=True):
            if "weight" not in data:
                raise ValueError(f"The edge between nodes {n1} and {n2} is missing the 'weight' attribute.")
            if not isinstance(data["weight"], float | int):
                raise TypeError(
                    f"The edge between nodes {n1} and {n2} has a 'weight' of type {type(data['weight']).__name__}, "
                    f"expected ``float`` or ``int``."
                )

        self._graph = graph
        self._break_z2 = break_z2

        qubo_mat = np.zeros((self._graph.number_of_nodes(), self._graph.number_of_nodes()), dtype=int)
        for i, j, edge_data in self._graph.edges(data=True):
            qubo_mat[i, j] += 2 * edge_data["weight"]
            qubo_mat[i, i] += -edge_data["weight"]
            qubo_mat[j, j] += -edge_data["weight"]
        bqm = BinaryQuadraticModel(qubo_mat, vartype="BINARY")
        super().__init__(bqm)
        if self._break_z2:
            node_to_fix = max(bqm.variables, key=bqm.degree)
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
        if len(set(bit_str)) > 2:
            raise ValueError(
                f"The string {bit_str} contains more than 2 distinct characters, "
                f"so it doesn't represent a partition of the graph into 2 distinct sets."
            )
        cut = 0
        for v1, v2, edge_data in self._graph.edges(data=True):
            if bit_str[v1] != bit_str[v2]:
                cut += edge_data["weight"]

        return cut


# pylint: disable=anomalous-backslash-in-string
def maxcut_generator(
    n: int,
    n_instances: int,
    *,
    graph_family: str = "erdos-renyi",
    p: float = 0.5,
    d: int = 3,
    break_z2: bool = False,
    seed: int = 1337,
    enforce_connected: bool = False,
) -> Iterator[MaxCutInstance]:
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
        enforce_connected: A bool stating whether it is required that the random graphs are connected.

    Yields:
        Problem instances of :class:`MaxCutInstance` randomly constructed in accordance to the input parameters.

    """

    def _generate_desired_graph(graph_family: str, n: int, p: float, d: int, seed: int) -> nx.Graph:
        """Wrapper helper function to encapsulate the graph generation logic."""
        if graph_family == "erdos-renyi":
            return nx.erdos_renyi_graph(n, p, seed)
        if graph_family == "regular":
            return nx.random_regular_graph(d, n, seed)
        raise ValueError("Invalid random graph type. Choose either 'regular' or 'erdos-renyi'.")

    for _ in range(n_instances):
        while True:
            g = _generate_desired_graph(graph_family, n, p, d, seed)
            seed = seed + 1  # Increment ``seed``, so that we don't keep generating the same graph over and over again.
            if nx.is_connected(g) or not enforce_connected:
                break
        yield MaxCutInstance(g, break_z2=break_z2)


def greedy_max_cut(max_cut_problem: MaxCutInstance | nx.Graph) -> str:
    """Standard greedy algorithm for maxcut problem class.

    Steps:

    1. Start with a random assignment of nodes in two groups.
    2. Iterate over all nodes
    3. For each node, switch it to the other group if it improves the cost function.
    4. Repeat steps 2-3 until no node can be switched.
    5. Return the final assignment.

    Args:
        max_cut_problem: A problem instance of maxcut or a :class:`~networkx.Graph`.

    Returns:
        A bitstring solution.

    """
    if isinstance(max_cut_problem, MaxCutInstance):
        max_cut_problem = max_cut_problem.graph
    elif not isinstance(max_cut_problem, nx.Graph):
        raise TypeError(
            f"Supported input is either a NetworkX graph or a MaxCutInstance. "
            f"Given type: {type(max_cut_problem).__name__}"
        )

    current_solution = list("".join(random.choice("01") for _ in range(max_cut_problem.number_of_nodes())))
    while True:
        for node in max_cut_problem.nodes():
            n_cut = sum(
                abs(int(current_solution[node]) - int(current_solution[neighbor]))
                for neighbor in max_cut_problem.neighbors(node)
            )
            n_uncut = max_cut_problem.degree(node) - n_cut  # type: ignore[operator]
            if n_cut < n_uncut:
                current_solution[node] = "1" if current_solution[node] == "0" else "0"
                break
        else:
            break
    return "".join(current_solution)


# pylint: disable=too-many-locals
def goemans_williamson(max_cut_problem: MaxCutInstance | nx.Graph) -> str:
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
        max_cut_problem: A problem instance of maxcut or a :class:`~networkx.Graph`.

    Returns:
        A bitstring solution.

    """
    if isinstance(max_cut_problem, MaxCutInstance):
        max_cut_problem = max_cut_problem.graph
    elif not isinstance(max_cut_problem, nx.Graph):
        raise TypeError(
            f"Supported input is either a NetworkX graph or a MaxCutInstance. "
            f"Given type: {type(max_cut_problem).__name__}"
        )

    adjacency = nx.linalg.adjacency_matrix(max_cut_problem)
    adjacency = adjacency.toarray()

    size = len(adjacency)
    ones_matrix = np.ones((size, size))
    products = cp.Variable((size, size), PSD=True)
    cut_size = 0.5 * cp.sum(cp.multiply(adjacency, ones_matrix - products))

    objective = cp.Maximize(cut_size)
    constraints = [cp.diag(products) == 1]
    problem = cp.Problem(objective, constraints)
    problem.solve()

    eigenvalues, eigenvectors = eigh(products.value)  # type: ignore[arg-type]
    eigenvalues = np.maximum(eigenvalues, 0)
    diagonal_root = np.diag(np.sqrt(eigenvalues))
    assignment = diagonal_root @ eigenvectors.T

    size = len(assignment)
    partition = np.random.normal(size=size)
    projections = assignment.T @ partition

    sides = (np.sign(projections).astype(int) + 1) // 2

    nodes = list(max_cut_problem.nodes)
    sides_ordered = sides[nodes]
    result = "".join(map(str, sides_ordered))

    return result
