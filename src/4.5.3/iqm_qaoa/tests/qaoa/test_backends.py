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
"""The testing module for the iqm/qaoa/backends.py file."""

from collections.abc import Callable
import math

from iqm.applications.maxcut import MaxCutInstance
from iqm.applications.mis import MISInstance
from iqm.qaoa.backends import (
    EstimatorFromSampler,
    EstimatorQUIMB,
    EstimatorSingleLayer,
    EstimatorStateVector,
    SamplerBackend,
    SamplerSimulation,
)
from iqm.qaoa.qubo_qaoa import QUBOQAOA
from iqm.qiskit_iqm import IQMFakeApollo
import networkx as nx
import numpy as np
import pytest


def test_exp_val_equal_maxcut(tolerance: float, sparse_maxcut_instance: MaxCutInstance) -> None:
    """Testing that the estimators agree for maxcut QAOA."""
    p = 1
    rng = np.random.default_rng(1337)
    angles = rng.uniform(0, np.pi, size=2 * p)

    my_qaoa_maxcut = QUBOQAOA(sparse_maxcut_instance, num_layers=p, initial_angles=angles)

    estimators = [EstimatorSingleLayer(), EstimatorStateVector(), EstimatorQUIMB()]

    results = [estimator.estimate(my_qaoa_maxcut) for estimator in estimators]

    assert math.isclose(max(results), min(results), rel_tol=tolerance, abs_tol=tolerance)


def test_estimator_from_custom_sampler(
    special_g: nx.Graph, custom_rigged_sampler: Callable[[dict[str, float]], SamplerBackend]
) -> None:
    """Testing the ``EstimatorFromSampler`` class by using it on a specific problem with a specific rigged sampler.

    The MIS problem on the graph ``special_g`` has a known solution. We take this problem and define a "sampler" that
    always outputs the same bitstring. This sampler is imported as fixture. Then we use the ``EstimatorFromSampler``
    object to take the output of this sampler and estimate the energy based on that.
    """
    my_mis_instance = MISInstance(special_g)

    p = 1
    rng = np.random.default_rng(1337)
    angles = rng.uniform(0, np.pi, size=2 * p)
    my_mis_qaoa = QUBOQAOA(my_mis_instance, num_layers=p, initial_angles=angles)

    my_rigged_sampler = custom_rigged_sampler(
        {"0111000": 1.0}
    )  # Create a rigged sampler that only samples the solution.
    estimator_from_rigged_sampler = EstimatorFromSampler(my_rigged_sampler, shots=1000)
    energy = estimator_from_rigged_sampler.estimate(my_mis_qaoa)

    assert energy == my_mis_qaoa.hamiltonian_bqm.energy([1, -1, -1, -1, 1, 1, 1])
    assert energy == my_mis_instance.bqm.energy([0, 1, 1, 1, 0, 0, 0])  # Sanity check.


def test_sampler_simulation_does_not_fail_with_iqmfakeapollo(small_maxcut_instance: MaxCutInstance) -> None:
    """Testing that `SamplerSimulation` accepts our `IQMFakeApollo` simulator.

    Previously, `SamplerSimulation` was written in such a way that `IQMFakeApollo` wasn't compatible with it, so after
    fixing it, this test was added to check that it runs. As the results are random (and noisy), we just test that it
    runs, not anything about the results.
    """
    shots = 20000

    # We don't even care about training the QAOA.
    my_qaoa = QUBOQAOA(problem=small_maxcut_instance, num_layers=1, initial_angles=[0.1, 0.2])

    crystal_qpu_sampler = SamplerSimulation(IQMFakeApollo(), transpiler="SparseTranspiler")

    crystal_qpu_estimator = EstimatorFromSampler(crystal_qpu_sampler, shots=shots)

    # We don't care about the results of the sampling and the estimation, we just test that it runs.
    my_qaoa.sample(crystal_qpu_sampler, shots=shots)
    my_qaoa.estimate(crystal_qpu_estimator)


@pytest.mark.parametrize(
    "problem_instance_name",
    [
        ("sparse_mis_instance"),
        ("sparse_maxcut_instance"),
    ],
)
def test_estimators_of_arbitrary_zz_agree(problem_instance_name: str, request: pytest.FixtureRequest) -> None:
    """Testing that the estimators agree for maxcut QAOA."""
    prob_inst = request.getfixturevalue(problem_instance_name)

    rng = np.random.default_rng(1337)

    n_vars = prob_inst.dim

    esl = EstimatorSingleLayer()
    esv = EstimatorStateVector()
    equ = EstimatorQUIMB()

    for p in range(1, 4):
        angles = rng.uniform(0, np.pi, size=2 * p)

        my_qaoa_maxcut = QUBOQAOA(prob_inst, num_layers=p, initial_angles=angles)

        for wght_corr in range(1, 5):
            n_cors_to_check = 2  # Somewhat arbitrary number of correlations to check.

            # This looks a bit clunky, but we need standard Python integers, not ``np.int64``.
            target_qubits = [
                {int(x) for x in rng.choice(n_vars, wght_corr, replace=False)} for _ in range(n_cors_to_check)
            ]

            esv_value = esv.estimate_correlations_z(qaoa_object=my_qaoa_maxcut, target_qubits=target_qubits)
            equ_value = equ.estimate_correlations_z(qaoa_object=my_qaoa_maxcut, target_qubits=target_qubits)
            if p == 1 and wght_corr <= 2:
                esl_value = esl.estimate_correlations_z(qaoa_object=my_qaoa_maxcut, target_qubits=target_qubits)

                assert np.allclose(esl_value, equ_value)
                assert np.allclose(esl_value, esv_value)
            elif p == 1:
                with pytest.raises(ValueError) as error_name:
                    esl_value = esl.estimate_correlations_z(qaoa_object=my_qaoa_maxcut, target_qubits=target_qubits)

                assert (
                    str(error_name.value)
                    == "The ``EstimatorSingleLayer`` can only calculate expectation values of Z or ZZ."
                )
                assert np.allclose(esv_value, equ_value)
            else:
                with pytest.raises(ValueError) as error_name:
                    esl_value = esl.estimate_correlations_z(qaoa_object=my_qaoa_maxcut, target_qubits=target_qubits)

                assert str(error_name.value).startswith("The number of layers is not 1")
                assert np.allclose(esv_value, equ_value)


def test_estimator_from_sampler_zz_correlations(
    small_mis_instance: MISInstance, custom_rigged_sampler: Callable[[dict[str, float]], SamplerBackend]
) -> None:
    """Checks the method ``estimate_correlations_z`` of :class:`~iqm.qaoa.backends.EstimatorFromSampler`.

    Uses a custom rigged sampler to generate deterministic samples. Then checks that
    :class:`~iqm.qaoa.backends.EstimatorFromSampler` calculates the correlations from the samples correctly.
    """
    p = 1
    rng = np.random.default_rng(1337)
    angles = rng.uniform(0, np.pi, size=2 * p)
    my_mis_qaoa = QUBOQAOA(small_mis_instance, num_layers=p, initial_angles=angles)

    bitstr_dist = {"0100": 0.5, "1110": 0.25, "1000": 0.25}
    my_rigged_sampler = custom_rigged_sampler(bitstr_dist)
    efs = EstimatorFromSampler(my_rigged_sampler, 100)

    assert efs.estimate_correlations_z(my_mis_qaoa, {0}) == 0
    assert efs.estimate_correlations_z(my_mis_qaoa, {0, 1}) == -0.5
    assert efs.estimate_correlations_z(my_mis_qaoa, {0, 2}) == 0.5
    assert efs.estimate_correlations_z(my_mis_qaoa, {3}) == 1
    assert efs.estimate_correlations_z(my_mis_qaoa, {0, 1, 2}) == -1
    assert efs.estimate_correlations_z(my_mis_qaoa, {0, 3, 2}) == 0.5
