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
"""The testing module for the iqm/applications/mis.py file, testing the MISInstance class."""

from collections.abc import Callable
import math
import random

from iqm.applications.mis import MaximumWeightISInstance, MISInstance, bron_kerbosch, greedy_mis, mis_generator
import networkx as nx
import pytest


def test_greedy_suboptimal(special_g: nx.Graph) -> None:
    """Testing whether the greedy algorithm finds the incorrect MIS (as expected).

    It uses the ``special_g`` fixture, which provides the smallest graph for which the greedy algorithm
    doesn't find the correct MIS.
    """
    solution = greedy_mis(special_g)
    assert solution in {"1000100", "1000010", "1000001"}


def test_bron_kerbosch_suboptimal(special_g: nx.Graph) -> None:
    """Testing whether the Bron-Kerbosch algorithm finds the correct MIS (as expected).

    It uses the ``special_g`` fixture, which provides the smallest graph for which the greedy algorithm
    doesn't find the correct MIS, but the Bron-Kerbosch algorithm presumably does.
    """
    solution = bron_kerbosch(special_g)
    assert solution == "0111000"


def test_best_quality_edge_cases(edge_cases_graph_generator: Callable[[int], tuple[nx.Graph, ...]]) -> None:
    """Testing that the ``best_quality`` method finds the correct MIS for a set of edge cases.

    The ``best_quality`` method calls the Bron-Kerbosch algorithm. The special graphs have maximum
    independent sets easily found by hand.
    """
    expected_quality = [
        -1,
        -math.ceil(30 / 2),
        -math.floor(30 / 2),
        -30,
    ]  #  Assuming (complete_graph, cycle_graph, linear_graph, disconnected_graph)
    graphs = edge_cases_graph_generator(30)
    for i in range(len(edge_cases_graph_generator(30))):
        local_mis_instance = MISInstance(graphs[i])
        quality_exact_calculated = local_mis_instance.best_quality
        assert quality_exact_calculated == expected_quality[i]

    empty_graph = nx.Graph()
    empty_mis_instance = MISInstance(empty_graph)
    quality_exact_empty = empty_mis_instance.best_quality
    assert quality_exact_empty == 0


def test_constraints_edge_cases(edge_cases_graph_generator: Callable[[int], tuple[nx.Graph, ...]]) -> None:
    """Testing that the ``constraints_checker`` method of MISInstance works correctly for a set of edge cases.

    The ``constraints_checker`` method goes over the graph and checks that no edge connects two 'selected' nodes.
    For the special cases, the constraint can be checked easily in alternative ways (see the assert statements).
    The edge cases are various graphs with 100 nodes. For the purpose of the test, a list of 101 quasi-random
    bistrings is generated. Each bitstring has a different number of nodes selected to proprely "probe" the edge cases.
    """
    graphs = edge_cases_graph_generator(100)
    complete_graph_mis_instance = MISInstance(graphs[0])
    cycle_graph_mis_instance = MISInstance(graphs[1])
    linear_graph_mis_instance = MISInstance(graphs[2])
    disconnected_graph_mis_instance = MISInstance(graphs[3])
    bitstrings_to_test = []

    for i in range(100 + 1):
        ones = ["1"] * i
        zeros = ["0"] * (100 - i)
        combined = ones + zeros
        random.shuffle(combined)
        bitstrings_to_test.append("".join(combined))

    for bitstring in bitstrings_to_test:
        assert complete_graph_mis_instance.constraints_checker(bitstring) == (bitstring.count("1") < 2)
        assert cycle_graph_mis_instance.constraints_checker(bitstring) == (
            ("11" not in bitstring) and (bitstring[0] == "0" or bitstring[-1] == "0")
        )
        assert linear_graph_mis_instance.constraints_checker(bitstring) == ("11" not in bitstring)
        assert disconnected_graph_mis_instance.constraints_checker(bitstring)


def test_greedy_bound_generic_cases() -> None:
    """Testing the performance of the greedy algorithm against lower bounds.

    The lower bound on the performance of the greedy algorithm is n/(d+1) where
    n is the number of nodes and d is the average degree (Turan theorem).
    """
    for i in range(10):
        er_graph = nx.erdos_renyi_graph(100, 0.5, seed=1337 + i)
        ba_graph = nx.barabasi_albert_graph(100, m=2, seed=1337 + i)
        ws_graph = nx.watts_strogatz_graph(100, k=4, p=0.5, seed=1337 + i)
        avg_deg_er = er_graph.number_of_edges() / er_graph.number_of_nodes()
        avg_deg_ba = ba_graph.number_of_edges() / ba_graph.number_of_nodes()
        avg_deg_ws = ws_graph.number_of_edges() / ws_graph.number_of_nodes()
        assert greedy_mis(er_graph).count("1") >= 100 / (avg_deg_er + 1)
        assert greedy_mis(ba_graph).count("1") >= 100 / (avg_deg_ba + 1)
        assert greedy_mis(ws_graph).count("1") >= 100 / (avg_deg_ws + 1)


def test_fix_constraint_violation_postprocessing(
    samples_dict: dict[str, int], sparse_mis_instance: MISInstance
) -> None:
    """Tests that the constraints are now satisfied by setting the penalty high and checking for the maximum energy."""
    sparse_mis_instance.penalty = 10000

    new_dict = sparse_mis_instance.fix_constraint_violation(samples_dict)
    worst_quality = max(sparse_mis_instance.loss(bitstr) for bitstr in new_dict.keys())

    assert worst_quality < 0
    assert sum(new_dict.values()) == sum(samples_dict.values())


def test_fix_variables_correct_error(special_g: nx.Graph) -> None:
    """Testing if fixing variables returns the correct error when done on a MIS."""
    my_mis = MISInstance(special_g)
    with pytest.raises(ValueError, match="Can't fix two neighboring nodes to 1 in an independent set problem."):
        my_mis.fix_variables([0, 2])

    with pytest.raises(ValueError, match="Can't fix two neighboring nodes to 1 in an independent set problem."):
        my_mis.fix_variables({5: 1, 4: 0, 6: 1})


def test_fix_variables_to_zero(special_g: nx.Graph, sparse_mis_instance: MISInstance) -> None:
    """Testing if fixing variables correctly removes the node from the problem."""
    my_mis = MISInstance(special_g)
    # Removing the first 3 nodes make the problem be fully-connected on 4 nodes.
    my_mis.fix_variables({0: 0, 1: 0, 2: 0})

    original_num_nodes = sparse_mis_instance.dim
    sparse_mis_instance.fix_variables({0: 0, 2: 0, 3: 0, 6: 0})

    assert sparse_mis_instance.bqm.num_variables == original_num_nodes - 4
    assert my_mis.bqm.num_variables == 4
    assert my_mis.bqm.num_interactions == 6


def test_fix_variables_to_one_removes_all_neighbors(special_g: nx.Graph, sparse_mis_instance: MISInstance) -> None:
    """Testing if fixing variables correctly removes the node and its neighbours from the problem."""
    my_mis = MISInstance(special_g)
    # Fixing the node 0 to 1 should also fix nodes 1, 2 and 3 to 0 (leaving only the fully-connected nodes 4, 5 and 6).
    my_mis.fix_variables([0])

    # This is an arbitrary 3-regular graph, so we fix one node to 1 and expect 3 more to be fixed to 0.
    original_num_nodes = sparse_mis_instance.dim
    sparse_mis_instance.fix_variables([2])

    assert sparse_mis_instance.bqm.num_variables == original_num_nodes - 4
    assert my_mis.bqm.num_variables == 3
    assert my_mis.bqm.num_interactions == 3


def test_weighted_mis() -> None:
    """Tests initializing ``MaximumWeightISInstance`` with a weighted graph (and brute-forcing its quality)."""
    weighted_star_graph = nx.star_graph(4)  # Star graph with 4 spokes, so 5 total nodes.

    # The central node has huge weight, so it alone will be the maximum weight independent set
    weights = {0: 6, 1: 1, 2: 1, 3: 1, 4: 1}
    nx.set_node_attributes(weighted_star_graph, weights, "weight")

    # Define the problem instances, with high penalty to make sure the constraints aren't violated.
    mis = MISInstance(weighted_star_graph, penalty=10)
    weighted_is = MaximumWeightISInstance(weighted_star_graph, penalty=10)

    # Regular MIS ignores the weights, so it finds MIS of size 4 (the spokes of the star)
    assert mis.best_quality == -4
    # Maximum weight IS should be just the center of the star (with weight 6)
    assert weighted_is.best_quality == -6


def test_mis_generator_produces_multiple_instances() -> None:
    """Checks that the generator generates instances of ``MISInstance`` that can be iterated over."""
    count = 0
    mis_gen = mis_generator(graph_family="regular", n=10, n_instances=5, d=3)
    for mis_inst in mis_gen:
        assert isinstance(mis_inst, MISInstance)
        count += 1

    assert count > 0  # Check that the iterator produced at least a single instance.


def test_mis_generator_respects_graph_params() -> None:  # seed, d, n
    """Checks that the generated instances of ``MISInstance`` respect the input parameters."""
    mis_gen_1 = mis_generator(graph_family="regular", n=10, n_instances=1, d=3, seed=1337)
    mis_gen_2 = mis_generator(graph_family="regular", n=10, n_instances=1, d=3, seed=1337)
    mis_1 = next(mis_gen_1)
    mis_2 = next(mis_gen_2)
    assert nx.is_isomorphic(mis_1.graph, mis_2.graph)  # Check that seed works
    assert mis_1.dim == 10
    degrees = {d for _, d in mis_1.graph.degree()}
    assert degrees == {3}


def test_mis_generator_fails_to_find_connected_graph() -> None:
    """Checks that the generator raises the correct error when asked to find a connected graph that is impossible."""
    # A 1-regular graph can't be connected (if it has more than 2 vertices).
    mis_gen = mis_generator(graph_family="regular", n=6, n_instances=1, d=1, enforce_connected=True)
    with pytest.raises(RuntimeError) as err_info:
        _ = next(mis_gen)  # Since it's a generator, it needs to be called to trigger the error.
    assert str(err_info.value).startswith("Failed to generate a connected graph")


def test_mis_fix_variables_orig_labels(graph_with_uncommon_node_labels: nx.Graph) -> None:
    """Test that the method ``fix_variables`` works when using original (non-integer) labels of variables."""
    node_labels = list(graph_with_uncommon_node_labels)

    mis_from_graph = MISInstance(graph_with_uncommon_node_labels, allow_custom_var_names=True)

    node_to_fix = node_labels[0]  # Deterministically pick an arbitrary node to fix.

    with pytest.raises(TypeError) as err_info:
        mis_from_graph.fix_variables({node_to_fix: 1})  # Not overriding the default ``original_labels = False``.
    assert str(err_info.value).startswith("When `original_labels` is set to False (default), the types of variables")

    mis_from_graph.fix_variables({node_to_fix: 1}, original_labels=True)

    expected_number_of_nodes_left = (
        graph_with_uncommon_node_labels.number_of_nodes() - 1 - graph_with_uncommon_node_labels.degree[node_to_fix]
    )
    assert mis_from_graph.cqm.num_variables() == expected_number_of_nodes_left
