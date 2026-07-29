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
"""Tests for the tree_qaoa.py file."""

from iqm.applications.maxcut import MaxCutInstance
from iqm.applications.mis import MISInstance
from iqm.applications.qubo import QUBOInstance
from iqm.qaoa.backends import EstimatorSingleLayer
from iqm.qaoa.tree_qaoa import TreeQAOA
import networkx as nx
import numpy as np
from pytest import CaptureFixture


def test_angles_scaling_max_cut() -> None:
    """Testing that the angles scale properly with the Hamiltonian of the graph.

    Creates a quasi-random instance of maxcut (my_instance) and another problem instance (my_new_instance),
    which is identical, but has its cost function scaled by "scale".
    """
    scale = 10
    p = 6
    g = nx.random_regular_graph(3, 50, seed=1337)
    my_instance = MaxCutInstance(g)
    my_new_bqm = my_instance.bqm.copy()
    my_new_bqm.scale(scale)
    my_new_instance = QUBOInstance(my_new_bqm)
    my_qaoa = TreeQAOA(my_instance, num_layers=p)
    my_new_qaoa = TreeQAOA(my_new_instance, num_layers=p)

    my_qaoa.set_tree_angles()
    my_new_qaoa.set_tree_angles()

    assert np.allclose(my_qaoa.angles[0::2], my_new_qaoa.angles[0::2] * scale, rtol=1e-5, atol=1e-5)


def test_angles_scaling_mis() -> None:
    """Testing that the angles scale properly with the Hamiltonian of the graph.

    Creates a quasi-random instance of maxcut (my_instance) and another problem instance (my_new_instance),
    which is identical, but has its cost function scaled by "scale".
    """
    scale = 10
    p = 6
    g = nx.random_regular_graph(3, 50, seed=1337)
    my_instance = MISInstance(g)
    my_new_bqm = my_instance.bqm.copy()
    my_new_bqm.scale(scale)
    my_new_instance = QUBOInstance(my_new_bqm)
    my_qaoa = TreeQAOA(my_instance, num_layers=p)
    my_new_qaoa = TreeQAOA(my_new_instance, num_layers=p)

    my_qaoa.set_tree_angles()
    my_new_qaoa.set_tree_angles()

    assert np.allclose(my_qaoa.angles[0::2], my_new_qaoa.angles[0::2] * scale, rtol=1e-5, atol=1e-5)


def test_exp_val_p_one_scaling() -> None:
    """Testing that the scaled angles give the correct expectation value."""
    scale = 10
    p = 1
    g = nx.random_regular_graph(3, 50, seed=1337)
    my_instance = MaxCutInstance(g)
    my_new_bqm = my_instance.bqm.copy()
    my_new_bqm.scale(scale)
    my_new_instance = QUBOInstance(my_new_bqm)
    my_qaoa = TreeQAOA(my_instance, num_layers=p)
    my_new_qaoa = TreeQAOA(my_new_instance, num_layers=p)

    my_qaoa.set_tree_angles()
    my_new_qaoa.set_tree_angles()

    my_esti = EstimatorSingleLayer()

    assert np.isclose(my_esti.estimate(my_new_qaoa), my_esti.estimate(my_qaoa) * scale, rtol=1e-5, atol=1e-5)


def test_angles_same_from_optimization() -> None:
    """Testing whether the angles calculated from the ``set_tree_angles`` agree with the single-layer angles."""
    p = 1
    g = nx.random_regular_graph(3, 50, seed=1337)
    my_instance = MaxCutInstance(g)
    my_qaoa = TreeQAOA(my_instance, num_layers=p)

    my_qaoa.angles = [0.5, -0.4]  # We nudge the training a bit (to not get stuck in a distant local minimum)
    my_qaoa.train()
    angles_from_normal_training = my_qaoa.angles.copy()
    my_qaoa.set_tree_angles()

    assert np.allclose(my_qaoa.angles, angles_from_normal_training, rtol=1e-2, atol=1e-2)


def test_verbosity_of_set_tree_angles(capsys: CaptureFixture[str], sparse_mis_instance: MISInstance) -> None:
    """Tests that the `verbose` input parameter of `set_tree_angles` works as intended by reading the printed output."""
    p = 4
    my_qaoa = TreeQAOA(sparse_mis_instance, num_layers=p)

    my_qaoa.set_tree_angles(verbose=True)
    captured = capsys.readouterr()
    assert "QAOA depth p = " in captured.out

    my_qaoa.set_tree_angles(verbose=False)
    captured = capsys.readouterr()
    assert captured.out == ""
