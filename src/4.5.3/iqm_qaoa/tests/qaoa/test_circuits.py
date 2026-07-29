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
"""Tests for functions inside of the ``circuits.py`` file."""

from iqm.applications.sk import SherringtonKirkpatrick
from iqm.qaoa.circuits import transpiled_circuit
from iqm.qaoa.qubo_qaoa import QUBOQAOA
from iqm.qaoa.transpiler.hardwired.hardwired import hardwired_router
from iqm.qaoa.transpiler.quantum_hardware import CrystalQPUFromBackend, StarQPU
from iqm.qaoa.transpiler.sn.sn import sn_router
from iqm.qaoa.transpiler.sparse.greedy_router import greedy_router
from iqm.qaoa.transpiler.star.star import star_router
from iqm.qiskit_iqm.iqm_backend import IQMBackendBase
from iqm.qiskit_iqm.iqm_provider import IQMBackend
import pytest


@pytest.mark.parametrize(
    "problem_instance_name",
    [
        ("small_mis_instance"),
        ("sparse_mis_instance"),
        ("small_maxcut_instance"),
        ("sparse_maxcut_instance"),
        ("small_sk_instance"),
    ],
)
def test_transpilation_does_not_mess_up_the_circuit_star(
    problem_instance_name: str, request: pytest.FixtureRequest, sirius_mock_backend: IQMBackendBase
) -> None:
    """Tests whether using ``transpiled_circuit`` doesn't mess up the circuit (star QPU)."""
    problem_instance = request.getfixturevalue(problem_instance_name)
    qaoa = QUBOQAOA(problem_instance, num_layers=2, initial_angles=[0.1, 0.2, 0.1, 0.2])

    qpu = StarQPU(problem_instance.dim)
    routed = star_router(qaoa.hamiltonian_bqm, qpu)

    circuit_pre_transpilation = routed.build_qiskit(qaoa.betas, qaoa.gammas)

    circuit_post_transpilation = transpiled_circuit(qaoa, backend=sirius_mock_backend, transpiler="MinimumVertexCover")

    circuit_pre_ops = dict(circuit_pre_transpilation.count_ops())
    circuit_post_ops = dict(circuit_post_transpilation.count_ops())

    # The 2QB gates get changed from CX to CZ, but their number stays the same.
    assert circuit_pre_ops["cx"] == circuit_post_ops["cz"]
    # The number of MOVE gates stays the same.
    assert circuit_pre_ops["move"] == circuit_post_ops["move"]


@pytest.mark.parametrize(
    "problem_instance_name",
    [
        ("small_mis_instance"),
        ("sparse_mis_instance"),
        ("small_maxcut_instance"),
        ("sparse_maxcut_instance"),
    ],
)
def test_transpilation_does_not_mess_up_the_greedy_routed_circuits(
    problem_instance_name: str, request: pytest.FixtureRequest, apollo_backend: IQMBackendBase
) -> None:
    """Tests whether using ``transpiled_circuit`` doesn't mess up the circuit (using greedy router)."""
    problem_instance = request.getfixturevalue(problem_instance_name)
    qaoa = QUBOQAOA(problem_instance, num_layers=2, initial_angles=[0.1, 0.2, 0.1, 0.2])

    qpu = CrystalQPUFromBackend(apollo_backend)
    routed = greedy_router(qaoa.hamiltonian_bqm, qpu)

    circuit_post_transpilation = transpiled_circuit(qaoa, backend=apollo_backend, transpiler="SparseTranspiler")

    circuit_post_ops = dict(circuit_post_transpilation.count_ops())

    number_of_expected_2qb_gates = 0
    for layer in routed.layers:
        for i in layer.gates.edges(data=True):
            if i[2]["swap"]:  # If there is a swap or a swap combined with interaction
                number_of_expected_2qb_gates += 3
            elif i[2]["int"]:  # If there is an interaction only
                number_of_expected_2qb_gates += 2

    # The 2QB gates get changed to CZ, but their number stays the same.
    assert number_of_expected_2qb_gates * qaoa.num_layers >= circuit_post_ops["cz"]


def test_transpilation_does_not_mess_up_the_dense_routed_circuits(
    small_sk_instance: SherringtonKirkpatrick, apollo_backend: IQMBackend
) -> None:
    """Tests whether using ``transpiled_circuit`` doesn't mess up the circuit (using SN and hardwired router)."""
    qaoa = QUBOQAOA(small_sk_instance, num_layers=2, initial_angles=[0.1, 0.2, 0.1, 0.2])

    qpu = CrystalQPUFromBackend(apollo_backend)
    routed_sn = sn_router(qaoa.hamiltonian_bqm, qpu)
    routed_hw = hardwired_router(qaoa.hamiltonian_bqm, qpu)

    sn_circuit_post_transpilation = transpiled_circuit(qaoa, backend=apollo_backend, transpiler="SwapNetwork")
    hw_circuit_post_transpilation = transpiled_circuit(qaoa, backend=apollo_backend, transpiler="HardwiredTranspiler")

    hw_circuit_post_ops = dict(hw_circuit_post_transpilation.count_ops())
    sn_circuit_post_ops = dict(sn_circuit_post_transpilation.count_ops())

    sn_number_of_expected_2qb_gates = 0
    for layer in routed_sn.layers:
        for i in layer.gates.edges(data=True):
            if i[2]["swap"]:  # If there is a swap or a swap combined with interaction.
                sn_number_of_expected_2qb_gates += 3
            elif i[2]["int"]:  # If there is an interaction only.
                sn_number_of_expected_2qb_gates += 2

    hw_number_of_expected_2qb_gates = 0
    for layer in routed_hw.layers:
        for i in layer.gates.edges(data=True):
            if i[2]["swap"]:  # If there is a swap or a swap combined with interaction.
                hw_number_of_expected_2qb_gates += 3
            elif i[2]["int"]:  # If there is an interaction only.
                hw_number_of_expected_2qb_gates += 2

    # If the inequality is satisfied strictly, it means that transpilation even saves us some gates!
    assert sn_number_of_expected_2qb_gates * qaoa.num_layers >= sn_circuit_post_ops["cz"]
    assert hw_number_of_expected_2qb_gates * qaoa.num_layers >= hw_circuit_post_ops["cz"]
