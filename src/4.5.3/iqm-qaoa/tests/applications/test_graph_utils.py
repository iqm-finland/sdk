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
"""The testing module for the iqm/applications/graph_utils.py file, testing the that graph with exotic labels work."""

import math

from iqm.applications.graph_utils import ProblemData, _extract_problem_info, get_top_n_color_pairs
from iqm.applications.maxcut import WeightedMaxCutInstance
from iqm.applications.mis import MaximumWeightISInstance, MISInstance
from iqm.applications.qubo import QUBOInstance
from iqm.qaoa.transpiler.sparse.edge_coloring import find_edge_coloring
import networkx as nx
import pytest


def test_graph_combination_of_labels() -> None:
    """Tests the expected behavior if the node / edge attributes contain a combination of different attributes."""
    test_graph = nx.Graph()
    test_graph.add_node(0, h=0.8)
    test_graph.add_node(1, bias=2)
    test_graph.add_node(2, field=0.3, weight=10)  # The `weight` should be given priority over `field`.
    test_graph.add_edge(1, 2, weight=0, interaction=12)  # The `weight` should be given priority over `interaction`.
    test_graph.add_edge(2, 0, J=3, coupling=2.7)  # The `coupling` should be given priority over `J`.

    test_qubo = QUBOInstance(test_graph)

    assert test_qubo.bqm.quadratic[(1, 2)] == 0  # This interaction should be 0 (that is the weight of that edge).
    assert (1, 0) not in test_qubo.bqm.quadratic  # This interaction shouldn't be present at all.
    assert test_qubo.bqm.quadratic[(2, 0)] == 2.7
    assert test_qubo.bqm.linear[0] == 0.8
    assert test_qubo.bqm.linear[1] == 2
    assert test_qubo.bqm.linear[2] == 10


def test_graph_missing_labels() -> None:
    """Tests that the correct error gets raised if the edge or node bias is missing."""
    test_graph = nx.Graph()
    test_graph.add_node(0)  # No node attribute here.
    test_graph.add_node(1, bias=2)
    test_graph.add_node(2, bias=0.3, weight=10)  # The weight should be ignored because of the priority.
    test_graph.add_edge(1, 2)  # No edge attribute here.
    test_graph.add_edge(2, 0, bias=3, coupling=2.7)  # The coupling should be ignored because of the priority.

    with pytest.raises(ValueError) as error_name:
        QUBOInstance(test_graph)

    assert str(error_name.value).startswith("The node 0 is missing one of the required attributes")

    test_graph.nodes[0]["weight"] = 5

    with pytest.raises(ValueError) as error_name:
        QUBOInstance(test_graph)

    assert str(error_name.value).startswith("The edge between nodes 1 and 2 is missing") or str(
        error_name.value
    ).startswith("The edge between nodes 2 and 1 is missing")


def test_wrong_label_type() -> None:
    """Tests that the correct error gets raised if the edge bias is a wrong type (not ``int`` or ``float``)."""
    test_graph = nx.Graph()
    test_graph.add_node(1, bias=2)
    test_graph.add_node(2, bias=0.3)
    test_graph.add_node(3, bias={})  # Empty dictionary instead of ``int`` or ``float``.
    test_graph.add_edge(1, 2, bias="1337")  # String instead of ``int`` or ``float``.

    # Creating a weighted maxcut from the above graph should detect the incorrect edge bias type.
    with pytest.raises(TypeError) as error_name:
        WeightedMaxCutInstance(test_graph)

    assert str(error_name.value).startswith("The edge between nodes ")

    # Creating a weighted MIS from the above graph should detect the incorrect edge bias type.
    with pytest.raises(TypeError) as error_name:
        MaximumWeightISInstance(test_graph, penalty=1)

    assert str(error_name.value).startswith("The local term at node ")


def test_max_number_of_color_pairs() -> None:
    r"""Tests some properties of ``get_top_n_color_pairs`` using the following graph.

    3-------2
    |      /|
    |     / |
    |    /  |
    |   4   |
    |  / \  |
    | /   \ |
    |/     \|
    0-------1
    """
    test_graph = nx.cycle_graph(4)
    test_graph.add_edges_from([(0, 4), (1, 4), (2, 4)])

    sets_of_colors, _ = find_edge_coloring(test_graph)
    num_of_colors = len(sets_of_colors)

    assert num_of_colors == max(dict(test_graph.degree()).values()) + 1  # Sanity check.
    total_color_pairs = math.comb(num_of_colors, 2)

    nums_to_check = [0, 3, 5, 9, 17]
    lists_of_pairs = []

    for num_pairs in nums_to_check:
        list_of_pairs = get_top_n_color_pairs(problem_graph=test_graph, max_color_pairs=num_pairs)
        # Check that the list of pairs of colors has the expected length.
        assert len(list_of_pairs) == min(num_pairs, total_color_pairs)
        lists_of_pairs.append(list_of_pairs)

    longest_list_of_pairs = max(lists_of_pairs, key=len)
    assert all(longest_list_of_pairs[: len(lst)] == lst for lst in lists_of_pairs)


def test_extract_problem_info_invalid_bitstring_length() -> None:
    """It should raise a ``ValueError`` if bitstring length does not match number of nodes."""
    sample_graph = nx.path_graph(4)
    mapping = {i: i for i in sample_graph.nodes()}

    with pytest.raises(ValueError, match="Bitstring length"):
        _extract_problem_info(
            graph_to_plot=sample_graph,
            orig_to_new_mapping=mapping,
            bitstring="01",  # Too short.
        )


def test_extract_problem_info_basic(sparse_maxcut_instance: MISInstance) -> None:
    """It should return a ``ProblemData`` object with data extracted from the problem instance."""
    sparse_maxcut_instance.fix_variables({4: 1})
    test_btstr = "01011000101000"

    result = _extract_problem_info(problem_instance=sparse_maxcut_instance, bitstring=test_btstr)

    assert isinstance(result, ProblemData)
    assert result.graph is sparse_maxcut_instance.graph
    assert result.orig_to_new_mapping == sparse_maxcut_instance.orig_to_new_labels
    assert result.bitstring == test_btstr
    assert result.highlight_edge_by_node_count == frozenset({1})
    assert result.fixed_vars == {4}
