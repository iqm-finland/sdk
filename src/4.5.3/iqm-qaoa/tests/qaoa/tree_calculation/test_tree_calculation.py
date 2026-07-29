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
"""Tests for functions inside of the ``tree_calculation.py`` file."""

from collections.abc import Callable

from dimod import BinaryQuadraticModel
from iqm.applications.maxcut import MaxCutInstance
from iqm.applications.mis import MISInstance
from iqm.applications.qubo import QUBOInstance
from iqm.qaoa.backends import EstimatorSingleLayer
from iqm.qaoa.qubo_qaoa import QUBOQAOA
from iqm.qaoa.tree_calculation.tree_calculation import (
    a24_a25,
    exp_gamma,
    get_big_h_fixed_points_from_scratch,
    get_exp_vals,
    prune_angles,
    small_g,
)
import networkx as nx
import numpy as np
from numpy.typing import NDArray
import pytest
from scipy.optimize import minimize


def test_small_g_zero_from_sin() -> None:
    r"""Tests the function `small_g` with a special input, guaranteeing the output 0.

    Since `b[0]` doesn't equal `b[1]`, the first term of the product will be :math:`i \sin{\beta_1}`. Because `beta[0]`
    is 0, this term will be 0 and so will be the entire product.
    """
    b = np.array([1, -1, -1, -1, 1, -1, 1, 1])
    m_minus_1 = 4
    betas = np.array([0, 42, 1337, -1337, -42, 0])
    assert np.isclose(small_g(m_minus_1, b, betas), 0)


def test_small_g_zero_from_cos() -> None:
    r"""Tests the function `small_g` with a special input, guaranteeing the output 0.

    Since `b[0]` equals `b[1]`, the first term of the product will be :math:`\cos{\beta_1}`. Because `beta[0]` is π/2,
    this term will be 0 and so will be the entire product.
    """
    b = np.array([1, 1, -1, -1, 1, -1, 1, -1])
    m_minus_1 = 4
    betas = np.array([np.pi / 2, 42, 1337, -1337, -42, -np.pi / 2])
    assert np.isclose(small_g(m_minus_1, b, betas), 0)


def test_small_g_power_of_sin_cos() -> None:
    """Tests the function `small_g` with a special input, guaranteeing a simple output.

    It does two tests:
    * In the first test, all `b` elements are the same, the function simplifies to product of the cosines of `betas`.
      Since all `betas` are also the same, this will be just one cosine raised to the power of `m_minus_1 - 1`.
    * In the second test, each consecutive `b` element is different from the previous one. So the function will be just
      the product of sines. We just need to account for the imaginary units, and for the fact that some `betas` are
      negative.
    """
    for m_minus_1 in range(4, 10):
        b_pos = np.ones(2 * m_minus_1, dtype=np.int8)
        betas = 1337 * np.concatenate([np.ones(m_minus_1 - 1), -np.ones(m_minus_1 - 1)])
        assert np.isclose(small_g(m_minus_1, b_pos, betas), np.cos(1337) ** (2 * m_minus_1 - 2))

        b_neg = np.tile([1, -1], m_minus_1)  # An array of altenating +1 and -1
        assert np.isclose(
            small_g(m_minus_1, b_neg, betas), (1j * np.sin(1337)) ** (2 * m_minus_1 - 2) * (-1) ** (m_minus_1 - 1)
        )


def test_prune_angles() -> None:
    """Tests that the output of `prune_angles` has some basic properties.

    Specifically, tests if the output angles have correct length (twice the input `layer` for the Gamma angles and two
    less for the Beta angles). It also tests that the output Beta and Gamma angles are antisymmetric as expected.
    """
    rng = np.random.default_rng(1337)
    betas = rng.uniform(0, np.pi, size=8)
    big_betas = np.concatenate([betas, [0], -betas[::-1]])
    gammas = rng.uniform(0, np.pi, size=8)
    big_gammas = np.concatenate([gammas, [0], -gammas[::-1]])
    layer = 4

    pruned_big_gammas, pruned_big_betas = prune_angles(layer=layer, big_gamma=big_gammas, big_beta=big_betas)
    assert len(pruned_big_gammas) == 8
    assert len(pruned_big_betas) == 6
    assert np.all(pruned_big_gammas + pruned_big_gammas[::-1] == 0)  # Anti-symmetric.
    assert np.all(pruned_big_betas + pruned_big_betas[::-1] == 0)  # Anti-symmetric.


def test_a24_a25_special_cases() -> None:
    """Tests that `a24_a25` gives expected result when given some special cases of inputs."""
    # There's a sine of angle 0 in each term, so the result is 0.
    assert a24_a25(1, -1, 1, -1, np.float64(0), np.float64(1337), 4) == 0
    # Exponential disappears, cosines of angle 0 make up the result 1.
    assert a24_a25(1, -1, 1, 1, np.float64(0), np.float64(0), 4) == 1
    # Exponential disappears, the cosines and sines from the `pair` functions cancel each other.
    assert a24_a25(1, 1, -1, 1, np.float64(1337), np.float64(1337), 4) == 0
    # Exponential disappears, sines and cosines square, adding up to 1. Although the addition isn't 100% accurate.
    assert np.isclose(a24_a25(1, 1, 1, 1, np.float64(1337), np.float64(1337), 4), 1)


def test_exp_gamma() -> None:
    """Tests some properties of the function `exp_gamma`."""
    # It's all product of imaginary exponentials, so it has to have absolute value 1.
    assert abs(
        exp_gamma(
            big_d=4,
            h=42.0,
            big_gamma_pruned=np.array([1, 3, 3, 7], dtype=np.float64),
            a=np.array([1, -1, 1, 1], dtype=np.float64),
            b=np.array([1, 1, 1, -1], dtype=np.float64),
        )
    ) == pytest.approx(1.0, abs=1e-9)
    # The first exponential disappears because of the inner product.
    # The second exponential disappears because `h` is zero.
    assert exp_gamma(
        big_d=4,
        h=0.0,
        big_gamma_pruned=np.array([5, 3, 5, 7], dtype=np.float64),
        a=np.array([1, -1, 1, 1], dtype=np.float64),
        b=np.array([1, 1, 1, -1], dtype=np.float64),
    ) == pytest.approx(1.0, abs=1e-9)
    # The element-wise product will be just `b`. The negative local interaction will make the two exponentials cancel.
    assert exp_gamma(
        big_d=4,
        h=-1.0,
        big_gamma_pruned=np.array([3, 1.2, 10, 0.08], dtype=np.float64),
        a=np.array([1, 1, 1, 1], dtype=np.float64),
        b=np.array([1, 1, 1, -1], dtype=np.float64),
    ) == pytest.approx(1.0, abs=1e-9)
    # The second exponential disappears because `h` is zero.
    # The scalar product adds up to `np.pi/2`, divided by the square root of `D` gives `np.pi/4`.
    # Using Euler's formula, the real part of the exponential is the cosine of `np.pi/4`, which is `np.sqrt(2)/2`.
    assert np.real(
        exp_gamma(
            big_d=4,
            h=0.0,
            big_gamma_pruned=np.array([np.pi / 2, 0, 0, 0], dtype=np.float64),
            a=np.array([1, 1, 1, 1], dtype=np.float64),
            b=np.array([1, 1, -1, -1], dtype=np.float64),
        )
    ) == pytest.approx(np.sqrt(2) / 2, 1e-9)


def test_get_h_fixed_points_small_p() -> None:
    """Checks for the first couple of entries in the generated fixed points."""
    p = 1
    big_d = 3
    h = 1.0
    big_gamma = np.array([0.5])
    big_beta = np.array([0.3])

    fixed_points, fixed_points_onsite = get_big_h_fixed_points_from_scratch(p, big_d, h, big_gamma, big_beta)

    expected_fp_p0 = np.array([1], dtype=np.complex128)
    expected_fp_p1 = np.array([np.cos(2 * big_gamma[0] / np.sqrt(big_d)) ** big_d], dtype=np.complex128)
    expected_fp_onsite_p1 = np.array([np.cos(2 * big_gamma[0] / np.sqrt(big_d)) ** (big_d + 1)], dtype=np.complex128)

    np.testing.assert_allclose(fixed_points[0], expected_fp_p0, rtol=1e-12)
    np.testing.assert_allclose(fixed_points[1], expected_fp_p1, rtol=1e-12)
    np.testing.assert_allclose(fixed_points_onsite[1], expected_fp_onsite_p1, rtol=1e-12)


@pytest.mark.parametrize("p", [2, 3])
def test_fixed_point_dimensions(p: int) -> None:
    """Checks that the fixed point generated lists have the correct dimensions."""
    big_d = 3
    h = 1.0
    gamma = np.linspace(0.1, 0.9, p)
    beta = np.linspace(0.1, 0.9, p)
    big_gamma = np.concatenate([gamma, [0], -gamma[::-1]])
    big_beta = np.concatenate([beta, [0], -beta[::-1]])

    fixed_points, fixed_points_onsite = get_big_h_fixed_points_from_scratch(p, big_d, h, big_gamma, big_beta)

    assert len(fixed_points) == p + 1
    assert len(fixed_points_onsite) == p + 1

    for m in range(2, p + 1):
        expected_size = 2 ** (2 * m - 2)
        assert fixed_points[m].shape == (expected_size,)
        assert fixed_points_onsite[m].shape == (expected_size,)


def test_exp_val_agree_for_p1_d2_maxcut() -> None:
    """Tests that the expectation values calculated from the tree-angle calculation agree with `EstimatorSingleLayer`.

    This particular tests checks the equality for maxcut on a 2-regular graph (i.e., a cycle).
    """
    n = 7
    test_graph_cycle = nx.cycle_graph(n)
    d = 2
    big_d = d - 1

    my_problem = MaxCutInstance(test_graph_cycle)
    # By default, the BQM is scaled so that its energy equals the cut size.
    # It has to be rescaled so that the interections equal to 1.
    my_problem.bqm.scale(2)

    h = 0
    assert h == my_problem.bqm.spin.linear[0]  # Sanity check that the `h` is what we expect it to be.
    p = 1

    # Create a `QUBOQAOA` instance and estimate its energy using `EstimatorSingleLayer`.
    my_qaoa = QUBOQAOA(my_problem, num_layers=p, initial_angles=[0.5, 0.4])
    my_p1_esti = EstimatorSingleLayer()
    exp_val_from_estimator = my_qaoa.estimate(my_p1_esti)

    # Prepare the angles for the tree calculation (see equation A8).
    big_gamma = np.concatenate([my_qaoa.gammas, [0], -my_qaoa.gammas[::-1]])
    big_beta = np.concatenate([my_qaoa.betas, [0], -my_qaoa.betas[::-1]])

    # Calculating expectation values as a tuple (first element is <Z>, second element is <ZZ>).
    z_and_zz = get_exp_vals(p=p, big_d=big_d, h=h, big_gamma=big_gamma, big_beta=big_beta)

    # Get the energy from the <Z> and <ZZ> expectation values.
    # 1. Multiply by the number of nodes / edges (both `n` here)
    # 2. Multiply local term by `h` (here it's 0 though).
    # 3. Add the offset (constant) term.
    exp_val_from_tree_angles = h * z_and_zz[0] * n + z_and_zz[1] * n + my_problem.bqm.spin.offset

    # Do the same as above, but calculate the <Z> and <ZZ> using the formulas for `p` equal to 1 from A51, A52 and A53.
    exp_val_from_a51_a52_a53 = (
        n * h * _a53(big_beta[0], big_gamma[0], h, big_d)
        + n * (_a51(big_beta[0], big_gamma[0], h, big_d) + _a52(big_beta[0], big_gamma[0], h, big_d))
        + my_problem.bqm.spin.offset
    )

    assert np.isclose(exp_val_from_tree_angles, exp_val_from_a51_a52_a53)  # Reality check.
    assert np.isclose(exp_val_from_estimator, exp_val_from_tree_angles)


def test_exp_val_agree_for_p1_d3_maxcut() -> None:
    r"""Tests that the expectation values calculated from the tree-angle calculation agree with `EstimatorSingleLayer`.

    This particular tests checks the equality for maxcut on this 3-regular graph:

     /‾‾‾‾‾‾‾‾‾‾‾\
    0----1---2----3
    |    |   |    |
    |    8---9    |
    |    |   |    |
    7----6---5----4
     \___________/

    """
    test_graph_3reg = nx.cycle_graph(8)  # Starts with the cycle of nodes 0 to 7.
    test_graph_3reg.add_edges_from([(0, 3), (4, 7), (1, 8), (2, 9), (5, 9), (6, 8), (8, 9)])  # Add the missing edges.

    n = test_graph_3reg.number_of_nodes()  # This will be 10.
    d = 3
    big_d = d - 1
    # Reality check that `d` is indeed the degree of the graph.
    assert d == sum(dict(test_graph_3reg.degree()).values()) / n

    my_problem = MaxCutInstance(test_graph_3reg)
    my_problem.bqm.scale(2)  # Scale the BQM so that the interactions are 1.

    p = 1
    h = 0
    assert h == my_problem.bqm.spin.linear[0]  # Reality check that the local term is equal to `h` as expected.

    # The tree angle calculation is made for the Hamiltonian rescaled by `1 / np.sqrt(d - 1)` (almost equation 1).
    my_problem.bqm.scale(1 / np.sqrt(d - 1))

    # Create a `QUBOQAOA` instance and estimate its energy using `EstimatorSingleLayer`.
    my_qaoa = QUBOQAOA(my_problem, num_layers=p, initial_angles=[0.5, 0.4])
    my_p1_esti = EstimatorSingleLayer()
    exp_val_from_estimator = my_qaoa.estimate(my_p1_esti)

    # Prepare the angles for the tree calculation (see equation A8).
    big_gamma = np.concatenate([my_qaoa.gammas, [0], -my_qaoa.gammas[::-1]])
    big_beta = np.concatenate([my_qaoa.betas, [0], -my_qaoa.betas[::-1]])

    # Calculating expectation values as a tuple (first element is <Z>, second element is <ZZ>).
    z_and_zz = get_exp_vals(p=p, big_d=big_d, h=h, big_gamma=big_gamma, big_beta=big_beta)

    # Get the energy from the <Z> and <ZZ> expectation values.
    # 1. Multiply by the number of nodes / edges (`n` and `n * d / 2` respectively).
    # 2. Multiply local term by `h` (here it's 0 though).
    # 3. Add the offset (constant) term.
    # 4. Multiply the <Z> and <ZZ> terms by `1 / np.sqrt(d - 1)` as the Hamiltonian is scaled like that.
    exp_val_from_tree_angles = (
        h / np.sqrt(d - 1) * z_and_zz[0] * n + 1 / np.sqrt(d - 1) * z_and_zz[1] * n * d / 2 + my_problem.bqm.spin.offset
    )

    # Do the same as above, but calculate the <Z> and <ZZ> using the formulas for `p` equal to 1 from A51, A52 and A53.
    exp_val_from_a51_a52_a53 = (
        n * h / np.sqrt(d - 1) * _a53(big_beta[0], big_gamma[0], h, big_d)
        + n
        * d
        / 2
        / np.sqrt(d - 1)
        * (_a51(big_beta[0], big_gamma[0], h, big_d) + _a52(big_beta[0], big_gamma[0], h, big_d))
        + my_problem.bqm.spin.offset
    )

    assert np.isclose(exp_val_from_tree_angles, exp_val_from_a51_a52_a53)  # Reality check.
    assert np.isclose(exp_val_from_estimator, exp_val_from_tree_angles)


def test_exp_val_agree_for_p1_d2_mis() -> None:
    """Tests that the expectation values calculated from the tree-angle calculation agree with `EstimatorSingleLayer`.

    This particular tests checks the equality for MIS on a 2-regular graph (i.e., a cycle).
    """
    n = 7
    test_graph_cycle = nx.cycle_graph(n)
    d = 2
    big_d = d - 1

    lam = 2  # The penalty for violating the independence constraint. Needed for transforming the MIS into a QUBO.
    # First we create a MISInstance to get a QUBO out of the graph.
    my_problem = MISInstance(test_graph_cycle, penalty=lam)
    # We can't modify the BQM of a MISInstance, so we take the BQM and create a QUBOInstance out of it.
    my_problem_scaled = QUBOInstance(my_problem.bqm)
    # We can scale the BQM of this QUBOInstance now, so that the interactions are 1.
    my_problem_scaled.bqm.scale(2)

    p = 1
    h = d - 2 / lam  # Formula 18 (and the text directly after it).

    assert h == my_problem_scaled.bqm.spin.linear[0]  # Sanity check.

    # Create a `QUBOQAOA` instance and estimate its energy using `EstimatorSingleLayer`.
    my_qaoa = QUBOQAOA(my_problem_scaled, num_layers=p, initial_angles=[0.5, 0.4])
    my_p1_esti = EstimatorSingleLayer()
    exp_val_from_estimator = my_qaoa.estimate(my_p1_esti)

    # Prepare the angles for the tree calculation (see equation A8).
    big_gamma = np.concatenate([my_qaoa.gammas, [0], -my_qaoa.gammas[::-1]])
    big_beta = np.concatenate([my_qaoa.betas, [0], -my_qaoa.betas[::-1]])

    # Calculating expectation values as a tuple (first element is <Z>, second element is <ZZ>).
    z_and_zz = get_exp_vals(p=p, big_d=big_d, h=h, big_gamma=big_gamma, big_beta=big_beta)

    # Get the energy from the <Z> and <ZZ> expectation values.
    # 1. Multiply by the number of nodes / edges (both `n` here)
    # 2. Multiply local term by `h`.
    # 3. Add the offset (constant) term.
    exp_val_from_tree_angles = h * z_and_zz[0] * n + z_and_zz[1] * n + my_problem_scaled.bqm.spin.offset

    # Do the same as above, but calculate the <Z> and <ZZ> using the formulas for `p` equal to 1 from A51, A52 and A53.
    exp_val_from_a51_a52_a53 = (
        n * h * _a53(big_beta[0], big_gamma[0], h, big_d)
        + n * (_a51(big_beta[0], big_gamma[0], h, big_d) + _a52(big_beta[0], big_gamma[0], h, big_d))
        + my_problem.bqm.spin.offset
    )

    assert np.isclose(exp_val_from_tree_angles, exp_val_from_a51_a52_a53)  # Reality check.
    assert np.isclose(exp_val_from_estimator, exp_val_from_tree_angles)


def test_exp_val_agree_for_p1_d3_mis() -> None:
    r"""Tests that the expectation values calculated from the tree-angle calculation agree with `EstimatorSingleLayer`.

    This particular tests checks the equality for MIS on this 3-regular graph:

     /‾‾‾‾‾‾‾‾‾‾‾\
    0----1---2----3
    |    |   |    |
    |    8---9    |
    |    |   |    |
    7----6---5----4
     \___________/

    """
    test_graph_3reg = nx.cycle_graph(8)  # Starts with the cycle of nodes 0 to 7.
    test_graph_3reg.add_edges_from([(0, 3), (4, 7), (1, 8), (2, 9), (5, 9), (6, 8), (8, 9)])  # Add the missing edges.

    n = test_graph_3reg.number_of_nodes()  # This will be 10.
    d = 3
    big_d = d - 1
    # Reality check that `d` is indeed the degree of the graph.
    assert d == sum(dict(test_graph_3reg.degree()).values()) / n

    lam = 2  # The penalty for violating the independence constraint. Needed for transforming the MIS into a QUBO.
    # First we create a MISInstance to get a QUBO out of the graph.
    my_problem = MISInstance(test_graph_3reg, penalty=lam)
    # We can't modify the BQM of a MISInstance, so we take the BQM and create a QUBOInstance out of it.
    my_problem_scaled = QUBOInstance(my_problem.bqm)
    # We can scale the BQM of this QUBOInstance now, so that the interactions are 1.
    my_problem_scaled.bqm.scale(2)

    p = 1
    h = d - 2 / lam  # Formula 18 (and the text directly after it).

    assert h == my_problem_scaled.bqm.spin.linear[0]  # Sanity check.

    # The tree angle calculation is made for the Hamiltonian rescaled by `1 / np.sqrt(d - 1)` (almost equation 1).
    my_problem_scaled.bqm.scale(1 / np.sqrt(d - 1))

    # Create a `QUBOQAOA` instance and estimate its energy using `EstimatorSingleLayer`.
    my_qaoa = QUBOQAOA(my_problem_scaled, num_layers=p, initial_angles=[0.5, 0.4])
    my_p1_esti = EstimatorSingleLayer()
    exp_val_from_estimator = my_qaoa.estimate(my_p1_esti)

    # Prepare the angles for the tree calculation (see equation A8).
    big_gamma = np.concatenate([my_qaoa.gammas, [0], -my_qaoa.gammas[::-1]])
    big_beta = np.concatenate([my_qaoa.betas, [0], -my_qaoa.betas[::-1]])

    # Calculating expectation values as a tuple (first element is <Z>, second element is <ZZ>).
    z_and_zz = get_exp_vals(p=p, big_d=big_d, h=h, big_gamma=big_gamma, big_beta=big_beta)

    # Get the energy from the <Z> and <ZZ> expectation values.
    # 1. Multiply by the number of nodes / edges (`n` and `n * d / 2` respectively).
    # 2. Multiply local term by `h`.
    # 3. Add the offset (constant) term.
    # 4. Multiply the <Z> and <ZZ> terms by `1 / np.sqrt(d - 1)` as the Hamiltonian is scaled like that.
    exp_val_from_tree_angles = (
        h / np.sqrt(d - 1) * z_and_zz[0] * n
        + 1 / np.sqrt(d - 1) * z_and_zz[1] * n * d / 2
        + my_problem_scaled.bqm.spin.offset
    )

    # Do the same as above, but calculate the <Z> and <ZZ> using the formulas for `p` equal to 1 from A51, A52 and A53.
    exp_val_from_a51_a52_a53 = (
        n * h / np.sqrt(d - 1) * _a53(big_beta[0], big_gamma[0], h, big_d)
        + n
        * d
        / 2
        / np.sqrt(d - 1)
        * (_a51(big_beta[0], big_gamma[0], h, big_d) + _a52(big_beta[0], big_gamma[0], h, big_d))
        + my_problem_scaled.bqm.spin.offset
    )

    assert np.isclose(exp_val_from_tree_angles, exp_val_from_a51_a52_a53)  # Reality check.
    assert np.isclose(exp_val_from_estimator, exp_val_from_tree_angles)


def test_exp_val_agree_for_p1_d2_generic() -> None:
    """Tests that the expectation values calculated from the tree-angle calculation agree with `EstimatorSingleLayer`.

    This particular tests checks the equality for a problem with a generic local term on a 2-regular graph (i.e.,
    a cycle).
    """
    n = 7
    d = 2
    big_d = d - 1
    p = 1

    rng = np.random.default_rng(seed=1337)
    array_of_h = rng.random(5)  # Generate 5 arbitrary values for the local term `h`.

    for h in array_of_h:
        my_bqm = BinaryQuadraticModel(
            np.array([h, h, h, h, h, h, h]),  # Local terms of the Hamiltonian.
            np.array(
                [
                    [0, 1, 0, 0, 0, 0, 1],
                    [0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0],
                ]
            ),  # Interactions of the Hamiltonian.
            vartype="SPIN",
        )

        my_problem = QUBOInstance(my_bqm.binary)
        my_qaoa = QUBOQAOA(my_problem, num_layers=p, initial_angles=[0.5, 0.4])
        my_p1_esti = EstimatorSingleLayer()
        exp_val_from_estimator = my_qaoa.estimate(my_p1_esti)

        # Prepare the angles for the tree calculation (see equation A8).
        big_gamma = np.concatenate([my_qaoa.gammas, [0], -my_qaoa.gammas[::-1]])
        big_beta = np.concatenate([my_qaoa.betas, [0], -my_qaoa.betas[::-1]])

        z_and_zz = get_exp_vals(p=p, big_d=big_d, h=h, big_gamma=big_gamma, big_beta=big_beta)

        # Get the energy from the <Z> and <ZZ> expectation values.
        # 1. Multiply by the number of nodes / edges (`n` and `n` respectively).
        # 2. Multiply local term by `h`.
        # 3. Add the offset (constant) term.
        exp_val_from_tree_angles = h * z_and_zz[0] * n + z_and_zz[1] * n + my_problem.bqm.spin.offset

        # Do the same as above, but calculate the <Z> and <ZZ> using the formulas from A51, A52 and A53.
        exp_val_from_a51_a52_a53 = (
            n * h * _a53(big_beta[0], big_gamma[0], h, big_d)
            + n * (_a51(big_beta[0], big_gamma[0], h, big_d) + _a52(big_beta[0], big_gamma[0], h, big_d))
            + my_problem.bqm.spin.offset
        )

        assert np.isclose(exp_val_from_tree_angles, exp_val_from_a51_a52_a53)
        assert np.isclose(exp_val_from_estimator, exp_val_from_tree_angles)


def _a51(beta: float, gamma: float, h: float, big_d: int) -> np.complex128:
    """A helper function containing the formula from A51."""
    return (
        -1
        / 2
        * np.sin(2 * beta) ** 2
        * (np.cos(4 * h * gamma / np.sqrt(big_d)) - 1)
        * np.cos(2 * gamma / np.sqrt(big_d)) ** (2 * big_d)
    )


def _a52(beta: float, gamma: float, h: float, big_d: int) -> np.complex128:
    """A helper function containing the formula from A52."""
    return (
        np.sin(4 * beta)
        * np.sin(2 * gamma / np.sqrt(big_d))
        * np.cos(2 * h * gamma / np.sqrt(big_d))
        * np.cos(2 * gamma / np.sqrt(big_d)) ** big_d
    )


def _a53(beta: float, gamma: float, h: float, big_d: int) -> np.complex128:
    """A helper function containing the formula from A53."""
    return np.sin(2 * beta) * np.sin(2 * h * gamma / np.sqrt(big_d)) * np.cos(2 * gamma / np.sqrt(big_d)) ** (big_d + 1)


@pytest.mark.parametrize(
    "p, d, h, starting_angles, angles_from_paper",
    [
        # MaxCut p=2
        (2, 3, 0, np.array([-0.2222, -0.8888, 0.5555, 0.4444]), np.array([-0.4225, -0.7776, 0.5549, 0.2924])),
        # MaxCut p=3
        (
            3,
            3,
            0,
            np.array([-0.2222, -0.8888, -0.5555, 0.4444, 0.6666, 0.3333]),
            np.array([-0.3653, -0.6914, -0.8114, 0.6090, 0.4596, 0.2357]),
        ),
        # MIS p=2
        (
            2,
            3,
            1,  # h = d - 2 / lam, so this is for lam = 1.
            np.array([-0.2222, -0.8888, 0.5555, 0.4444]),
            np.array([-0.3678, -0.7957, 0.5175, 0.2642]),
        ),
        # MIS p=3
        (
            3,
            3,
            1,  # h = d - 2 / lam, so this is for lam = 1.
            np.array([-0.2222, -0.8888, -0.5555, 0.4444, 0.6666, 0.3333]),
            np.array([-0.3260, -0.6720, -0.7582, 0.5777, 0.3680, 0.2103]),
        ),
        # MIS p=2, d=4
        (
            2,
            4,
            2,  # h = d - 2 / lam, so this is for lam = 1.
            np.array([-0.2222, -0.8888, 0.4444, 0.6666]),
            np.array([-0.3123, -0.8352, 0.5169, 0.2407]),
        ),
    ],
    ids=["maxcut_p2_d3", "maxcut_p3_d3", "mis_p2_d3", "mis_p3_d3", "mis_p2_d4"],
)
def test_angle_agreement_with_paper(
    make_fun_to_min: Callable[[], Callable[[int, int, float], Callable[[NDArray[np.float64]], np.float64]]],
    p: int,
    d: int,
    h: float,
    starting_angles: NDArray[np.float64],
    angles_from_paper: NDArray[np.float64],
) -> None:
    """Tests that the optimized angles agree with published values (Appendix B)."""
    fun_to_min = make_fun_to_min(p, d, h)
    res = minimize(fun_to_min, starting_angles, method="BFGS")

    np.testing.assert_almost_equal(res.x, angles_from_paper, decimal=3)
    np.testing.assert_almost_equal(res.fun, fun_to_min(angles_from_paper), decimal=7)
