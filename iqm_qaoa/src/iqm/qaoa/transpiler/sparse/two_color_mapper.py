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
"""This file contains the :func:`two_color_mapper` function which creates the initial
:class:`~iqm.qaoa.transpiler.routing.Mapping`.

The :class:`~iqm.qaoa.transpiler.routing.Mapping` is based on the edge coloring of the graph created by
the :func:`~iqm.qaoa.transpiler.sparse.edge_coloring.find_edge_coloring` function.
"""

from __future__ import annotations

from collections.abc import Iterator
import itertools

from dimod import BinaryQuadraticModel, to_networkx_graph
from iqm.qaoa.transpiler.quantum_hardware import QPU, HardQubit, LogEdge, LogQubit
from iqm.qaoa.transpiler.routing import Mapping
from iqm.qaoa.transpiler.sparse.edge_coloring import find_edge_coloring
import networkx as nx
import numpy as np


# pylint: disable=anomalous-backslash-in-string
def _decompose_into_chains_and_loops(
    problem_bqm: BinaryQuadraticModel,
) -> tuple[list[list[LogQubit]], list[list[LogQubit]]]:
    r"""Decomposes a subgraph of the problem graph into chains and loops.

    This function calls ``find_edge_coloring``. Then, it takes a subgraph of the problem graph defined by the two
    biggest colors (by the number of edges) and decomposes this subgraph into ``loops`` and ``chains`` which is always
    possible.

    Args:
        problem_bqm: The problem defined as :class:`~dimod.BinaryQuadraticModel`. The problem graph is obtained
            from this variable.

    Returns:
        Two lists containing the chains and loops of :class:`LogQubit`\s defined by the two biggest colors.

    """
    problem_graph = to_networkx_graph(problem_bqm)

    # Mark all nodes in the graph as NOT endnodes
    nx.set_node_attributes(problem_graph, False, "endnode")  # type: ignore[call-overload]

    # Find the color sets (i.e., sets of edges of the same color) and sort them by size
    color_sets, _ = find_edge_coloring(problem_graph)
    color_sets = sorted(color_sets, key=len, reverse=True)

    # A list of edges of the two largest colors (remade so that they're tuples and not frozensets)
    twocolor_edges = [tuple(edge) for edge in color_sets[0] | color_sets[1]]
    # Create a subgraph, containing only the edges of the two largest colors.
    twocolor_graph = problem_graph.edge_subgraph(twocolor_edges)
    components = [twocolor_graph.subgraph(c).copy() for c in nx.connected_components(twocolor_graph)]
    chains, loops = [], []

    # Go through the components.
    for component in components:
        deg = 3
        # For chains, find one of the endpoints, for loops just select one of its nodes.
        for node in component:
            if component.degree(node) < deg:
                node0 = node
                deg = component.degree(node)
        node1 = next(component.neighbors(node0))
        lst = [node0, node1]
        # Go through all of the nodes of the component, and add them into a list in order.
        for _ in range(len(component) - 2):
            for neighbor in component.neighbors(node1):
                if neighbor != node0:
                    node0 = node1
                    node1 = neighbor
                    lst.append(node1)
                    break
        # Based on whether the component is a loop or a chain, add the list of nodes to ``loops`` / ``chains``.
        if deg == 1:
            chains.append(lst)
        elif deg == 2:
            loops.append(lst)

    return chains, loops


# pylint: disable=anomalous-backslash-in-string
def _embed_chain(chain: list[LogQubit], hardware_graph: nx.Graph) -> dict[HardQubit, LogQubit]:
    r"""This function attempts to embed a chain of :class:`LogQubit`\s into an arbitrary hardware topology.

    Steps:

    1. Start with the node with the lowest degree.
    2. Add the neighboring node with the lowest degree.

    This function however does not produce optimal results, which is why a handmade embedding provided a method of
    a ``QPU`` subclass is preferred.

    Args:
        chain: A list of :class:`LogQubit`\s forming a chain in the problem graph.
        hardware_graph: A :class:`~networkx.Graph` describing the hardware topology.

    Returns:
        An embedding in the form of a dictionary.

    Raises:
        RuntimeError: If the algorithm reaches a dead end before assigning all logical qubits from the ``chain``.

    """
    working_hw_graph = hardware_graph.copy()
    # Select the lowest-degree node of the graph.
    current_node = min(working_hw_graph.nodes(), key=working_hw_graph.degree)  # type: ignore[arg-type]

    # Start building the embedding (mapping of physical HW qubits to logical qubits).
    embedding = {current_node: chain[0]}

    # For each logical qubit in the chain, find the lowest-degree neighbor of the last used HW qubit.
    for log_qb in chain[1:]:
        # Convert the ``Graph.neighbors`` generator into a list (so that it doesn't get exhausted by checking it)
        neighbors = list(working_hw_graph.neighbors(current_node))
        # Check if the current node even has any neighbors.
        if not neighbors:
            raise RuntimeError(
                "The greedy algorithm for embedding the chain failed (``current_node`` has no "
                "neighbors). Check if there is enough :class:`HardQubit`\s in ``hardware_graph`` or define"
                " a custom ``embedded_chain`` method of the ``QPU`` subclass."
            )
        new_node = min(neighbors, key=working_hw_graph.degree)  # type: ignore[arg-type]
        embedding[new_node] = log_qb
        working_hw_graph.remove_node(current_node)
        current_node = new_node
        new_node = None
    working_hw_graph.remove_node(current_node)
    return embedding


def _append_to_layer(mapping: Mapping, log_qb0: LogQubit, log_qb1: LogQubit, int_layer: list[LogEdge]) -> None:
    """Appends an interaction between ``log_qb0`` and ``log_qb1`` to ``int_layer`` (using ``mapping``).

    This function modifies ``int_layer`` in-place.

    Args:
        mapping: The log <-> hard :class:`~iqm.qaoa.transpiler.routing.Mapping` used.
        log_qb0: The first qubit in the interaction to be added to ``int_layer``.
        log_qb1: The second qubit in the interaction to be added to ``int_layer``.
        int_layer: The list of interactions to be applied in one layer.

    """
    hard_qb0, hard_qb1 = mapping.log2hard[log_qb0], mapping.log2hard[log_qb1]
    int_layer.append(frozenset((hard_qb0, hard_qb1)))


def two_color_mapper(problem_bqm: BinaryQuadraticModel, qpu: QPU) -> tuple[Mapping, list[list[LogEdge]]]:
    """Finds an initial mapping between logical and hardware qubits.

    The mapping is constructed so that almost all interactions of two colors of an edge coloring of the problem graph
    can be executed in the first two layers of the phase separator. It uses an internal function to find
    a (near-)optimal coloring of the problem graph, take the subgraph induced by the two largest colors and decompose
    the subgraph into chains and loops. The loops are then broken down into chains and all these small chains are then
    placed onto a big chain embedded along the ``qpu``.

    Args:
        problem_bqm: The :class:`~dimod.BinaryQuadraticModel` representation of the problem.
        qpu: The QPU, an object of the :class:`~iqm.qaoa.transpiler.quantum_hardware.QPU` class (or any of its
            subclasses).

    Returns:
        The initial mapping (as :class:`~iqm.qaoa.transpiler.routing.Mapping`) and a list of two lists of
        :class:`~iqm.qaoa.transpiler.quantum_hardware.LogEdge` interactions that can be implemented in the first two
        layers of the routing algorithm.

    """
    chains, loops = _decompose_into_chains_and_loops(problem_bqm)

    # Concatenate all of the chains and loops in one long list.
    concatenated_chains_and_loops = list(itertools.chain(*chains, *loops))

    # For some cases we want to save time and attach a suitable embedding to the QPU object.
    if hasattr(qpu, "embedded_chain"):
        embedding = dict(zip(qpu.embedded_chain(), concatenated_chains_and_loops))
    else:
        # Use a helper function to find the embedding (because that depends a bit on the QPU).
        embedding = _get_embedding(qpu, concatenated_chains_and_loops, problem_bqm.num_variables)

    # Initialize a mapping from the embedding (which doesn't necessarily define all hard <-> log qubit pairs).
    # This ``mapping`` will assign the remaining :class:`LogQubit`\s and :class:`HardQubit`\s.
    mapping = Mapping(qpu, problem_bqm, embedding)

    # Create the first two layers of interactions from the chains and loops
    int_layer0: list[LogEdge] = []
    int_layer1: list[LogEdge] = []
    for chain in chains:
        for log_qb0, log_qb1 in zip(chain[::2], chain[1::2]):
            _append_to_layer(mapping, log_qb0, log_qb1, int_layer0)
        for log_qb0, log_qb1 in zip(chain[1::2], chain[2::2]):
            _append_to_layer(mapping, log_qb0, log_qb1, int_layer1)
    for loop in loops:
        for log_qb0, log_qb1 in zip(loop[::2], loop[1::2]):
            _append_to_layer(mapping, log_qb0, log_qb1, int_layer0)
        for log_qb0, log_qb1 in zip(loop[1::2], loop[2::2]):
            _append_to_layer(mapping, log_qb0, log_qb1, int_layer1)

    return mapping, [int_layer0, int_layer1]


def _get_embedding(qpu: QPU, long_chain: list[LogQubit], n: int) -> dict[HardQubit, LogQubit]:
    """A helper function for finding an embedded path through the QPU graph.

    The goal is not just to embed the chain of logical qubits on the QPU, but to embed them close to each other (so that
    not many swaps are needed). This is done in two steps. We iterate over some possible shapes of qubits close to each
    other (assuming square grid topology) using ``_subgraph_iterator``. For each such subgraph, we embed the chain in it
    and check if it can fit into the QPU graph. If both checks pass, then this is returned as the embedding.

    Args:
        qpu: A QPU whose graph we want to embed a path into.
        long_chain: A list of variables which is meant to be embedded as a path onto the QPU.
        n: The number of qubits we need to select for the routing.


    Returns:
        A mapping between the qubits of the QPU and the variables provided in ``long_chain``.

    Raises:
        RuntimeError: If none of the subgraphs from ``_subgraph_iterator`` passes the checks (chain can be embedded in
            it and it fits in the QPU).

    """
    for subgraph in _subgraph_iterator(n):
        try:
            # Try to embed a chain in the subgraph greedily.
            embedding_on_subgraph = _embed_chain(long_chain, subgraph)

            # The following lines make it so that the embedding covers the full ``subgraph``.
            unassigned_nodes = set(subgraph.nodes) - set(embedding_on_subgraph.keys())
            available_values = set(range(n)) - set(long_chain)
            for key, value in zip(unassigned_nodes, available_values):
                embedding_on_subgraph[key] = value

        except RuntimeError:
            continue
        # Find a subgraph isomorphism of the subgraph into the QPU graph (if it exists).
        gm = nx.isomorphism.GraphMatcher(qpu.hardware_graph, subgraph)
        try:
            isomorphism = next(gm.subgraph_monomorphisms_iter())
        except StopIteration:
            continue
        # If both ``try`` blocks are passed, a suitable subgraph was found.
        embedding = {qpu_qb: embedding_on_subgraph[subgraph_node] for qpu_qb, subgraph_node in isomorphism.items()}
        return embedding

    raise RuntimeError("No more subgraphs.")


def _subgraph_iterator(n: int) -> Iterator[nx.Graph]:
    """An iterator that yields :class:`networkx.Graph` of ``n`` nodes to be embedded on a QPU.

    Because these graphs are to be embedded on a QPU, they need to be subgraphs of the 2D square lattice. They should
    also be "somewhat packed together", so that when routing is performed on them, not many swap gates are needed.

    Args:
        n: The size of the graphs to be yielded.

    Yields:
        An almost-circle (in Euclidean metric) of size ``n``, followed by an almost-square and an almost-rectangle
        whose sides have 1:2 length ratio.

    """
    # Start with a square 2D grid (which we will remove nodes from).
    # To be sure that this will work, start with a relatively big square grid, to have some "buffer".
    almost_circle_graph = nx.grid_2d_graph(int(np.sqrt(2 * n) + 1), int(np.sqrt(2 * n) + 1))
    # In order for there to be any chance of an embedded path, we center the circle at the center of a 2-by-2 square.
    center_coords = np.ceil(np.sqrt(2 * n) / 2) - 0.5
    # We sort the nodes by their Euclidean distance from the center of the circle (squared).
    sorted_nodes = sorted(
        list(almost_circle_graph.nodes()),
        key=lambda x: ((x[0] - center_coords) ** 2 + (x[1] - center_coords) ** 2, x[0], x[1]),
        reverse=True,
    )
    # And we keep only ``n`` nodes closest to the center.
    nodes_to_remove = sorted_nodes[: len(almost_circle_graph.nodes) - n]
    almost_circle_graph.remove_nodes_from(nodes_to_remove)

    yield almost_circle_graph

    # Start with a larger square and then remove nodes along the "highest-order" row and column.
    # For example for n = 11, the result should look like this (. are removed nodes, o are kept nodes):
    #           . . . .
    #           . o o o
    #           o o o o
    #           o o o o
    almost_square_graph = nx.grid_2d_graph(int(np.sqrt(n) + 1), int(np.sqrt(n) + 1))
    sorted_nodes = sorted(list(almost_square_graph.nodes()), key=lambda x: (-max(x), -x[0], -x[1]))
    nodes_to_remove = sorted_nodes[: len(almost_square_graph.nodes) - n]
    almost_square_graph.remove_nodes_from(nodes_to_remove)

    yield almost_square_graph

    # Same as above, except for now we remove two rows and one column (because we start with a 2-by-1 rectangle).
    almost_1_to_2_rectangle_graph = nx.grid_2d_graph(int(np.sqrt(2 * n) + 1), int(np.sqrt(n / 2) + 1))
    sorted_nodes = sorted(
        list(almost_1_to_2_rectangle_graph.nodes()), key=lambda x: (-max(x[0], 2 * x[1]), -x[0], x[1])
    )
    nodes_to_remove = sorted_nodes[: len(almost_1_to_2_rectangle_graph.nodes) - n]
    almost_1_to_2_rectangle_graph.remove_nodes_from(nodes_to_remove)

    yield almost_1_to_2_rectangle_graph
