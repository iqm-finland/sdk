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
"""The testing module for the iqm/applications/max_cut.py file, testing the MaxCutInstance class."""

from collections import Counter
import random

from iqm.applications.maxcut import MaxCutInstance, WeightedMaxCutInstance, greedy_max_cut, maxcut_generator
import networkx as nx
import pytest


def test_break_z2() -> None:
    """Testing that breaking Z2 changes the local fields.

    Breaking the Z2 symmetry removes one of the nodes of the graph and changes its 2-body interaction terms
    to single-site interaction terms for the neighboring nodes.
    """
    g = nx.erdos_renyi_graph(30, p=0.5, seed=137)
    max_cut_instance = MaxCutInstance(g, break_z2=True)
    sum_of_linear_biases = sum(max_cut_instance.bqm.spin.linear.values())
    # I'm comparing a float with an integer, so I use approx() to avoid rounding errors.
    assert max(dict(g.degree()).values()) == pytest.approx(2 * sum_of_linear_biases, rel=1e-9)


def test_greedy_max_cut_generic_cases() -> None:
    """Testing that the greedy algorithm performs at least as well as random guessing.

    Creates 3 instaces each of 4 different types of quasi-random graphs:
    - Erdős–Rényi
    - Barabasi-Albert
    - Watts-Strogatz
    - 3-regular
    For these, checks that the size of the cut found by ``greedy_max_cut`` is at least as good as random guessing.

    """
    for i in range(3):
        er_graph = nx.erdos_renyi_graph(80, 0.5, seed=1337 + i)
        ba_graph = nx.barabasi_albert_graph(80, m=2, seed=1337 + i)
        ws_graph = nx.watts_strogatz_graph(80, k=4, p=0.5, seed=1337 + i)
        rr_graph = nx.random_regular_graph(d=3, n=80, seed=1337 + i)
        max_cut_instance_er = MaxCutInstance(er_graph)
        max_cut_instance_ba = MaxCutInstance(ba_graph)
        max_cut_instance_ws = MaxCutInstance(ws_graph)
        max_cut_instance_rr = MaxCutInstance(rr_graph)
        assert max_cut_instance_er.cut_size(greedy_max_cut(er_graph)) >= er_graph.number_of_edges() / 2
        assert max_cut_instance_ba.cut_size(greedy_max_cut(ba_graph)) >= ba_graph.number_of_edges() / 2
        assert max_cut_instance_ws.cut_size(greedy_max_cut(ws_graph)) >= ws_graph.number_of_edges() / 2
        assert max_cut_instance_rr.cut_size(greedy_max_cut(rr_graph)) >= rr_graph.number_of_edges() / 2


def test_cut_size_equal_to_quality() -> None:
    """Testing if the naive cut size calculation matches the output of the quality method."""
    for i in range(10):
        graph = nx.erdos_renyi_graph(100, 0.5, seed=1337 + i)
        local_instance = MaxCutInstance(graph)
        random.seed(1337 + i)
        random_bitstring = "".join(random.choice("01") for _ in range(100))

        assert local_instance.cut_size(random_bitstring) == -local_instance.quality(random_bitstring)


def test_cut_size_weighted_maxcut() -> None:
    """Tests weighted maxcut on a custom graph.

    The graph looks like this:
       (4)
       / \
     10   50
     /     \
   (3)-10--(2)
    |       |
    25     18
    |       |
    (0)--25-(1)
    """
    weighted_graph = nx.Graph()
    weighted_graph.add_edges_from(
        [
            (0, 1, {"weight": 25}),
            (1, 2, {"weight": 18}),
            (2, 3, {"weight": 10}),
            (3, 0, {"weight": 25}),
            (2, 4, {"weight": 50}),
            (3, 4, {"weight": 10}),
        ]
    )

    weighted_maxcut = WeightedMaxCutInstance(weighted_graph)

    assert weighted_maxcut.best_quality == -128  # The cut size obtained when nodes [0, 2] are picked out.
    assert weighted_maxcut.cut_size("00101") == 38  # The cut size obtained when nodes [2, 4] are picked out.


def test_maxcut_generator_produces_multiple_instances() -> None:
    """Checks that the generator generates instances of ``MaxCutInstance`` that can be iterated over."""
    count = 0
    maxcut_gen = maxcut_generator(graph_family="regular", n=10, n_instances=5, d=3)
    for maxcut_inst in maxcut_gen:
        assert isinstance(maxcut_inst, MaxCutInstance)
        count += 1

    assert count > 0  # Check that the iterator produced at least a single instance.


def test_maxcut_generator_respects_graph_params() -> None:
    """Checks that the generated instances of ``MaxCutInstance`` respect the input parameters."""
    maxcut_gen_1 = maxcut_generator(graph_family="regular", n=10, n_instances=1, d=3, seed=1337)
    maxcut_gen_2 = maxcut_generator(graph_family="regular", n=10, n_instances=1, d=3, seed=1337)
    maxcut_1 = next(maxcut_gen_1)
    maxcut_2 = next(maxcut_gen_2)
    assert nx.is_isomorphic(maxcut_1.graph, maxcut_2.graph)  # Check that seed works
    assert maxcut_1.dim == 10
    degrees = {d for _, d in maxcut_1.graph.degree()}
    assert degrees == {3}


def test_maxcut_generator_fails_to_find_connected_graph() -> None:
    """Checks that the generator raises the correct error when asked to find a connected graph that is impossible."""
    # A 1-regular graph can't be connected (if it has more than 2 vertices).
    maxcut_gen = maxcut_generator(n=6, n_instances=1, graph_family="regular", d=1, enforce_connected=True)
    with pytest.raises(RuntimeError) as err_info:
        _ = next(maxcut_gen)  # Since it's a generator, it needs to be called to trigger the error.
    assert str(err_info.value).startswith("Failed to generate a connected graph")


def test_weighted_maxcut_generator() -> None:
    """Checks that the weighted maxcut generator generates problem instances like it's supposed to."""
    maxim = 5
    n_nodes = 14
    maxcut_gen = maxcut_generator(
        n=n_nodes,
        n_instances=1,
        graph_family="regular",
        d=3,
        weighted=True,
        distribution_of_weights="integers",
        maximum=maxim,
    )
    test_maxcut = next(maxcut_gen)

    # Collect weights into a Counter.
    weights = Counter(data["weight"] for _, _, data in test_maxcut._graph.edges(data=True))

    # Validation: total count of weights must match the dimension of the problem.
    assert sum(weights.values()) == test_maxcut.dim * 3 / 2 == n_nodes * 3 / 2
    # Check that weights are in the correct range.
    assert set(weights.keys()) <= set(range(maxim))
