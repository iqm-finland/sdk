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
"""Some tests for the classes in ``src/iqm/applications/qubo.py``."""

from collections import defaultdict

from dimod import BinaryQuadraticModel, ConstrainedQuadraticModel
from iqm.applications.maxcut import MaxCutInstance
from iqm.applications.qubo import ConstrainedQuadraticInstance, QUBOInstance
import networkx as nx
import numpy as np
import pytest


def test_fix_variables(sparse_maxcut_instance: MaxCutInstance) -> None:
    """A simple test testing that fixing the variables reduces the number of variables correctly."""
    vars_to_fix = [1, 3, 6]
    original_num_vars = sparse_maxcut_instance.bqm.num_variables
    sparse_maxcut_instance.fix_variables(vars_to_fix)
    assert sparse_maxcut_instance.bqm.num_variables == original_num_vars - len(vars_to_fix)


def test_restore_fix_variables(samples_dict: dict[str, int], sparse_maxcut_instance: MaxCutInstance) -> None:
    """Testing that the fixed variables are correctly restored into the bitstrings."""
    vars_to_fix = {1: 1, 3: 0, 6: 0}
    # We create a dictionary of "shorter samples", i.e., what the QC would return if it had solved the reduced problem.
    shorter_samples: defaultdict[str, int] = defaultdict(int)
    for bit_str, sample in samples_dict.items():
        shorter_bit_str = "".join(c for i, c in enumerate(bit_str) if i not in vars_to_fix)
        # If ``shorter_bit_str`` is not in ``shorter_samples`` yet, it is automatically created with a value of 0.
        shorter_samples[shorter_bit_str] += sample

    for shorter_sample in shorter_samples:
        # Correct length, equal to the dimension of the full problem, minus the variables that will be fixed.
        assert len(shorter_sample) == sparse_maxcut_instance.dim - len(vars_to_fix)

    # This reduces ``sparse_maxcut_instance.dim`` by ``len(vars_to_fix)``
    sparse_maxcut_instance.fix_variables(vars_to_fix)

    restored_samples = sparse_maxcut_instance.restore_fixed_variables(shorter_samples)

    for restored_sample in restored_samples:
        # Correct length, equal to the dimension of the problem, plus the variables that were fixed.
        assert len(restored_sample) == sparse_maxcut_instance.dim + len(vars_to_fix)
        # Correct values in the correct place.
        assert restored_sample[1] == "1" and restored_sample[3] == "0" and restored_sample[6] == "0"


def test_relabel_graph_nodes(graph_with_uncommon_node_labels: nx.Graph) -> None:
    """Testing that relabeling the problem variables works.

    In the current version, the graph nodes don't get relabelled, but the problem variables (when saved as a BQM) do.
    """
    max_cut_instance = MaxCutInstance(graph_with_uncommon_node_labels, allow_custom_var_names=True)
    assert list(max_cut_instance.bqm.variables) == list(range(max_cut_instance.graph.number_of_nodes()))
    assert set(max_cut_instance.graph.nodes) == set(graph_with_uncommon_node_labels)


def test_qubo_representations_for_constrainedquadraticinstance() -> None:
    """Tests that the different QUBO representations (graph / matrix / BQM) for `ConstrainedQuadraticInstance` agree."""
    # This is tested on a toy use case of portfolio optimization.
    n_assets = 7
    expected_return = [0.9, 1.1, 1.2, 0.7, 1.5, 1.8, 1.1]
    a = np.random.randn(n_assets, n_assets)
    cov_matrix = a @ a.T
    risk_aversion = 2.0
    budget = 3

    my_cqm = ConstrainedQuadraticModel()
    objective = -np.diag(expected_return) + risk_aversion * cov_matrix
    my_cqm.set_objective(BinaryQuadraticModel(objective, "BINARY"))
    my_cqm.add_constraint_from_model(qm=BinaryQuadraticModel(np.eye(n_assets), "BINARY"), sense="==", rhs=budget)
    my_problem = ConstrainedQuadraticInstance(my_cqm, penalty=1)

    # We'll just check 10 random entries.
    num_checks = 10
    rng = np.random.default_rng(1337)  # Fix the seed for reproducibility.
    for _ in range(num_checks):
        i = rng.integers(0, n_assets - 1)
        j = rng.integers(i, n_assets - 1)

        mat_val = my_problem.qubo_matrix[i, j]
        if i == j:
            graph_val = my_problem.qubo_graph.nodes[i]["bias"]
            bqm_val = my_problem.bqm.linear[i]
        else:
            graph_val = my_problem.qubo_graph[i][j]["bias"]
            bqm_val = my_problem.bqm.quadratic[(i, j)]

        assert np.isclose(mat_val, graph_val)
        assert np.isclose(graph_val, bqm_val)


def test_instantiating_qubo_vartypes() -> None:
    """Testing that instantiating ``QUBOInstance`` with different vartypes works correctly."""
    test_graph = nx.Graph()
    test_graph.add_node(0, bias=0)
    test_graph.add_node(1, bias=0)
    test_graph.add_edge(0, 1, bias=1)

    qubo_1 = QUBOInstance(test_graph, vartype="BINARY")
    qubo_2 = QUBOInstance(test_graph, vartype="SPIN")

    # Initiating the qubo as BINARY means that the internal BQM corresponds to the input graph.
    assert qubo_1.bqm.quadratic[(0, 1)] == 1
    assert qubo_1.bqm.linear[0] == 0

    # Now the input graph is interpreted as hamiltonian :math:`\mathcal{H} = Z_1 \otimes Z_2`.
    # This translates to QUBO cost function :math:`C(\mathbb{x}) = - 2 x_1 - 2 x_2 + 4 x_1 x_2 + 1`.
    assert qubo_2.bqm.quadratic[(0, 1)] == 4
    assert qubo_2.bqm.linear[0] == -2
    assert qubo_2.bqm.offset == 1


def test_fix_variables_orig_labels(graph_with_uncommon_node_labels: nx.Graph) -> None:
    """Test that the method ``fix_variables`` works when using original (non-integer) labels of variables."""
    node_labels = list(graph_with_uncommon_node_labels)

    maxcut_from_graph = MaxCutInstance(graph_with_uncommon_node_labels, allow_custom_var_names=True)

    var_to_fix = {node_labels[0]: 1, node_labels[1]: 0}  # Deterministically pick some arbitrary nodes to fix.

    with pytest.raises(TypeError) as err_info:
        maxcut_from_graph.fix_variables(var_to_fix)  # Not overriding the default ``original_labels = False``.
    assert str(err_info.value).startswith("When `original_labels` is set to False (default), the types of variables")

    maxcut_from_graph.fix_variables(var_to_fix, original_labels=True)

    expected_number_of_nodes_left = graph_with_uncommon_node_labels.number_of_nodes() - len(var_to_fix)
    assert maxcut_from_graph.bqm.num_variables == expected_number_of_nodes_left
