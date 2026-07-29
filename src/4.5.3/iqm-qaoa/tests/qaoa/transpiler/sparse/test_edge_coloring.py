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
"""Module which tests the edge coloring according to Vizing's theorem."""

from iqm.qaoa.transpiler.sparse.edge_coloring import ec_is_complete, ec_is_valid, find_edge_coloring
import networkx as nx


def test_correct_number_of_colors(graphs_for_ec: list[nx.Graph]) -> None:
    """Tests that the number of colors used satisfies Vizing's theorem.

    Vizing's theorem says that the number of colors needed is at most one more than the maximum degree of the graph.
    Trivially, it is greater than or equal to the maximum degree of the graph.
    """
    assert len(find_edge_coloring(graphs_for_ec[0])[0]) in {19, 20}
    assert len(find_edge_coloring(graphs_for_ec[1])[0]) in {2, 3}
    assert len(find_edge_coloring(graphs_for_ec[2])[0]) in {12, 13}
    assert len(find_edge_coloring(graphs_for_ec[3])[0]) in {19, 20}
    assert len(find_edge_coloring(graphs_for_ec[4])[0]) in {4, 5}
    max_degree_random_graph = max(dict(graphs_for_ec[5].degree()).values())
    assert len(find_edge_coloring(graphs_for_ec[5])[0]) in {max_degree_random_graph, max_degree_random_graph + 1}
    max_degree_disco_graph = max(dict(graphs_for_ec[6].degree()).values())
    assert len(find_edge_coloring(graphs_for_ec[6])[0]) in {max_degree_disco_graph, max_degree_disco_graph + 1}


def test_that_coloring_is_valid_and_complete(graphs_for_ec: list[nx.Graph]) -> None:
    """Tests that the coloring is always valid and complete."""
    for graph in graphs_for_ec:
        _, colored_graph = find_edge_coloring(graph)
        assert ec_is_valid(colored_graph)
        assert ec_is_complete(colored_graph)
