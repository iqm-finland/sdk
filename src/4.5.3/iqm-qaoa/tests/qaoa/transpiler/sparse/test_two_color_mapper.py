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
"""Module for testing the usage of the edge coloring to find the first two interaction layers."""

from dimod import BinaryQuadraticModel, to_networkx_graph
from iqm.qaoa.transpiler.quantum_hardware import QPU, Grid2DQPU
from iqm.qaoa.transpiler.sparse.edge_coloring import find_edge_coloring
from iqm.qaoa.transpiler.sparse.two_color_mapper import (
    _decompose_into_chains_and_loops,
    _embed_chain,
    _get_embedding,
    _greedy_longest_path_basic,
    _greedy_longest_path_with_backtracking,
    _subgraph_iterator,
    two_color_mapper,
)
import networkx as nx
import pytest


def test_decompose_into_chains_and_loops(bqms: list[BinaryQuadraticModel]) -> None:
    """Test function for ``decompose_into_chains_and_loops``.

    Tests several properties of the chains and loops.
    """
    for bqm in bqms:
        problem_graph = to_networkx_graph(bqm)
        color_sets, _ = find_edge_coloring(problem_graph)
        color_sets = sorted(color_sets, key=len, reverse=True)
        int_gate_count = len(color_sets[0]) + len(color_sets[1])
        chains, loops = _decompose_into_chains_and_loops(bqm, (color_sets[0], color_sets[1]))
        alternate_int_gate_count = 0
        for chain in chains:
            alternate_int_gate_count += len(chain) - 1
        for loop in loops:
            alternate_int_gate_count += len(loop)
        assert int_gate_count == alternate_int_gate_count, (
            "The number of interactions doesn't match the length of chains and loops."
        )
        # Check if no node has been assigned twice.
        node_set = set()
        node_sum = 0
        for chain in chains:
            assert len(chain) > 1, "Chain found with a single element."
            node_set.update(chain)
            node_sum += len(chain)
        for loop in loops:
            assert len(loop) > 1, "Loop found with single element."
            node_set.update(loop)
            node_sum += len(loop)
        assert node_sum == len(node_set), "At least a node has been assigned at least twice."
        # Check if all nodes in the chains and nodes are actually adjacent to each other in ``problem_graph``
        for chain in chains:
            for log_qb0, log_qb1 in zip(chain[:-1], chain[1:], strict=True):
                assert problem_graph.has_edge(log_qb0, log_qb1), (
                    "Adjacent nodes in chain are not adjacent in ``problem_graph``."
                )
        for loop in loops:
            for log_qb0, log_qb1 in zip(loop[:-1], loop[1:], strict=True):
                assert problem_graph.has_edge(log_qb0, log_qb1), (
                    "Adjacent nodes in loop are not adjacent in ``problem_graph``."
                )
            assert problem_graph.has_edge(log_qb0, log_qb1), (
                "First and last node of loop are not adjacent in ``problem_graph``."
            )


def test_embed_chain_neighbors_remain_neighbors(bqms: list[BinaryQuadraticModel], square_qpu: QPU) -> None:
    """Check that a chain is embedded correctly on the QPU.

    In particular, it checks the neighboring variables in the chain are assigned to neighboring qubits on the QPU.
    """
    for bqm in bqms:
        problem_graph = to_networkx_graph(bqm)
        color_sets, _ = find_edge_coloring(problem_graph)
        color_sets = sorted(color_sets, key=len, reverse=True)
        chains, _ = _decompose_into_chains_and_loops(bqm, (color_sets[0], color_sets[1]))
        qpu_graph = square_qpu.hardware_graph
        for chain in chains:
            embedding = _embed_chain(chain, qpu_graph.copy())
            rev_embedding = {log_qb: hard_qb for hard_qb, log_qb in embedding.items()}
            for log_qb0, log_qb1 in zip(chain[:-1], chain[1:], strict=True):
                assert qpu_graph.has_edge(rev_embedding[log_qb0], rev_embedding[log_qb1]), (
                    "Chain embeddeding incorrect -- neighboring chain variables aren't neighboring on the QPU."
                )


def test_embed_chain_should_not_work() -> None:
    """Tests the function ``_embed_chain`` on a graph on which it should not work.

    0---1---7---5---6
        |   |   |
        2---3---4
    """
    test_graph = nx.path_graph(7)
    test_graph.add_edges_from({(1, 7), (3, 7), (5, 7)})

    chain_to_embed = list(range(8))

    # The ``match`` keyword makes it so that we only check for part of the error message, not the entire error message.
    with pytest.raises(RuntimeError, match="The greedy algorithm with"):
        _embed_chain(chain_to_embed, test_graph)


def test_two_color_mapper_biggest_two_colors_can_interact(bqms: list[BinaryQuadraticModel], square_qpu: QPU) -> None:
    """Tests ``two_color_mapper``.

    Checks that the each qubit is involved in each interaction layer only once.
    Checks that the number of qubit pairs which can interact immediately is at least as big as the number of
    interactions in the first two interaction layers.
    """
    for bqm in bqms:
        problem_graph = to_networkx_graph(bqm)
        color_sets, _ = find_edge_coloring(problem_graph)
        color_sets = sorted(color_sets, key=len, reverse=True)
        mapping, int_layers = two_color_mapper(bqm, square_qpu, (color_sets[0], color_sets[1]))
        int_gate_count = [len(int_layers[0]), len(int_layers[1])]
        for layer_ind, int_layer in enumerate(int_layers):
            qbs_set = set()
            for int_gate in int_layer:
                qbs_set.add(list(int_gate)[0])
                qbs_set.add(list(int_gate)[1])
            assert 2 * int_gate_count[layer_ind] == len(qbs_set), (
                f"``int_layer``{layer_ind} involves at least one qubit twice."
            )
        int_partners = 0
        problem_graph = to_networkx_graph(bqm)
        for hard_qb0, hard_qb1 in square_qpu.hardware_graph.edges():
            if problem_graph.has_edge(mapping.hard2log[hard_qb0], mapping.hard2log[hard_qb1]):
                int_partners += 1
        assert int_partners >= int_gate_count[0] + int_gate_count[1], (
            "``two_color_mapper`` did not map as many logical qubits next to each other as advertised."
        )


def test_subgraph_iterator() -> None:
    """Tests if the graphs of size 10 from the iterator look as expected.

    They're expected to look like this (circle, square, 1-to-2 rectangle):

          +---+---+---+       +---+---+          +---+---+---+
          |   |   |   |       |   |   |          |   |   |   |
          +---+---+---+       +---+---+          +---+---+---+
              |   |           |   |   |          |   |
              +---+           +---+---+---+      +---+
    """
    graph_iterator = _subgraph_iterator(10)

    # The graphs are generated by removing nodes from ``grid_2d_graph``.
    # That is why their nodes are labelled by their 2D integer coordinates.

    circ_graph = next(graph_iterator)
    assert set(circ_graph.nodes()) == {(2, 1), (2, 2), (3, 1), (3, 2), (2, 3), (3, 3), (3, 4), (2, 4), (4, 2), (4, 3)}

    sqr_graph = next(graph_iterator)
    assert set(sqr_graph.nodes()) == {(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 0), (3, 1), (3, 2), (3, 3)}

    rec_graph = next(graph_iterator)
    assert set(rec_graph.nodes()) == {(1, 1), (1, 2), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2)}


def test_get_embedding_finds_correct_embedding(qpu_with_hole: QPU) -> None:
    """We test the function `_get_embedding` on the fixture `qpu_with_hole`.

    We ask to embedd a list of 15, 18 and 48 qubits in order. The list of 15 qubits should be mapped onto a near-square
    (such as the one highlighted by the 1's below). The list of 18 qubits should be mapped onto a 3-by-6 rectangle (such
    as the one highlighted by the 2's below). The list of 48 qubits should be mapped along a Hamiltonian path onto
    the entire QPU.

    +---2---2---2---2---2---2
    |   |   |   |   |   |   |
    +---2---2---2---2---2---2
    |   |   |   |   |   |   |
    +---2---2---2---2---2---2
    |   |   |       |   |   |
    +---+---+       1---1---1
    |   |   |       |   |   |
    +---+---+---1---1---1---1
    |   |   |   |   |   |   |
    +---+---+---1---1---1---1
    |   |   |   |   |   |   |
    +---+---+---1---1---1---1
    """
    list_of_15_qubits = list(range(15))
    list_of_18_qubits = list(range(18))
    list_of_48_qubits = list(range(48))

    embed_15_qubits = _get_embedding(qpu_with_hole, list_of_15_qubits, 15)
    embed_18_qubits = _get_embedding(qpu_with_hole, list_of_18_qubits, 18)
    embed_48_qubits = _get_embedding(qpu_with_hole, list_of_48_qubits, 48)

    # Check if the selected subgraph is isomorphic with the graph we expect.
    subgraph_15 = qpu_with_hole.hardware_graph.subgraph(set(embed_15_qubits.keys()))
    expected_graph_15 = nx.grid_2d_graph(4, 4)
    expected_graph_15.remove_node((0, 0))
    assert nx.is_isomorphic(subgraph_15, expected_graph_15)

    # Now for the 18 logical qubits.
    subgraph_18 = qpu_with_hole.hardware_graph.subgraph(set(embed_18_qubits.keys()))
    expected_graph_18 = nx.grid_2d_graph(3, 6)
    assert nx.is_isomorphic(subgraph_18, expected_graph_18)

    # Just a reality check. If `_get_embedding` returns anything for 48 qubits, it must be correct.
    assert len(embed_48_qubits) == 48


def test_get_embedding_fails() -> None:
    """Tests that `_get_embedding` returns the correct error when the qubits can't be embedded on the QPU.

    First, when there's too many logical qubits for the QPU:

           +---+
    QPU:   |   |   Logical qubits: [0, 1, 2, 3, 4]
           +---+

    Second, when there's no good way to embedd the qubits onto the QPU (and have a Hamiltonian path go through them):

              +
              |
    QPU:  +---+---+   Logical qubits: [0, 1, 2, 3]
              |
              +
    """
    qpu_square = Grid2DQPU(2, 2)
    qpu_star = QPU(nx.star_graph(4))  # 4 spokes -> 5 nodes total.

    with pytest.raises(RuntimeError, match="No embedding"):
        _get_embedding(qpu_square, list(range(5)), 5)

    with pytest.raises(RuntimeError, match="No embedding"):
        _get_embedding(qpu_star, list(range(4)), 4)


def test_backtracking_vs_no_backtracking() -> None:
    r"""Tests that the backtracking approach for finding graph paths can find paths that the simple approach can't.

    In particular, here we test the algorithms on the following graph:

    0-----2---5---6
    |\   /|   |
    | \ / |   |
    | / \ |   |
    |/   \|   |
    1-----3---4---7

    The expected behavior is that the basic greedy algorithm will find the path [6, 5, 4, 7] whereas the algorithm with
    backtracking will find the full Hamiltonian path. The backtracking algorithm needs to be called with
    ``search_all_neighbors`` set to ``True``, so that when it reaches node 4 or 5, it goes towards the higher-degree
    node next (unlike the basic algorith).
    """
    test_graph = nx.complete_graph(4)
    test_graph.add_edges_from([(3, 4), (2, 5), (4, 5), (4, 7), (5, 6)])

    path_from_simple_algorithm = _greedy_longest_path_basic(test_graph)
    path_from_backtracking_algorithm = _greedy_longest_path_with_backtracking(test_graph, search_all_neighbors=True)

    assert path_from_simple_algorithm == [6, 5, 4, 7]
    assert set(path_from_backtracking_algorithm) == {0, 1, 2, 3, 4, 5, 6, 7}
