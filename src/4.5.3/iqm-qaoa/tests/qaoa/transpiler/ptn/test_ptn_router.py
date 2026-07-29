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
"""Tests for the ParityTwineNetwork router."""

import math

from iqm.applications.sk import sk_generator
from iqm.qaoa.qubo_qaoa import QUBOQAOA
from iqm.qaoa.transpiler.ptn.ptn import ptn_router
from iqm.qaoa.transpiler.quantum_hardware import CrystalQPUFromBackend, Grid2DQPU
from iqm.qaoa.transpiler.sn.sn import sn_router
from iqm.qiskit_iqm.iqm_provider import IQMBackend
import numpy as np
from qiskit.quantum_info import Statevector


def test_ptn_routing_number_of_cnots_1_qaoa_layer(apollo_backend: IQMBackend) -> None:
    """Tests that the ptn router creates circuits with the correct number of cnots."""
    expected_cnot_numbers = {n: n**2 - 1 for n in range(6, 20, 1)}
    number_of_cnots = 0
    number_of_ints = 0
    for num_qubits in range(6, 20, 1):
        sk_problem = next(sk_generator(n=num_qubits, n_instances=1, seed=1337))
        sk_qaoa = QUBOQAOA(problem=sk_problem, num_layers=1, initial_angles=[0.3, -0.3])
        apollo = CrystalQPUFromBackend(apollo_backend)
        routing = ptn_router(sk_qaoa.hamiltonian_bqm, qpu=apollo, strategy="Line")
        qc = routing.build_qiskit(sk_qaoa.betas.tolist(), sk_qaoa.gammas.tolist())
        number_of_cnots = qc.count_ops()["cx"]
        number_of_ints = qc.count_ops()["rz"]

        assert number_of_cnots == expected_cnot_numbers[num_qubits]
        assert number_of_ints == math.comb(num_qubits, 2)


def test_ptn_routing_depth(apollo_backend: IQMBackend) -> None:
    """Tests that the ptn router creates circuits with the correct depth."""

    def expected_depth_ptn(n: int, p: int, local_terms: bool) -> int:
        """The expected depth of PTN circuit, obtained by visually inspecting the circuit for n=6 and p=2.

        It is assumed that the problem has all-to-all interactions. If some interactions are missing, the depth is
        lower. The function is intentionally written in a very explicit way, to make it easier to check its correctness.

        Args:
            n: The number of qubits / problem variables.
            p: The number of QAOA layers.
            local_terms: Does the circuit include local terms? Set this to ``False`` for SK model, which doesn't contain
                local terms (just 2-body interactions).

        Returns:
            The depth of the PTN quantum circuit (without any transpilation).

        """
        depth = 0
        depth += 1  # First layer of Hadamards.
        depth += 1 * local_terms  # First layer of interactions.
        depth += 2 * (n - 4)  # First part of the PTN triangle (just DCNOTs).
        second_part_of_triangle = 0
        second_part_of_triangle += 4  # Four CNOTs between the two parts of the triangle.
        second_part_of_triangle += 3 * (n - 1)  # The bulk of the 2nd part of the triangle (2 CNOTs + 1 RZ).
        depth += p * (second_part_of_triangle)
        depth += p * 1  # Line of CNOTs to restore the variables.
        depth += (p - 1) * 0  # Line of RX (they perfectly fit in between the gaps in the triangles).
        depth += 1  # The final line of RX doesn't have two triangles to fit in between.
        depth += (p - 1) * local_terms  # Interactions of the next layer.
        return depth

    for p in range(1, 5):
        for num_qubits in range(6, 20, 1):
            sk_problem = next(sk_generator(n=num_qubits, n_instances=1, seed=1337))
            sk_qaoa = QUBOQAOA(problem=sk_problem, num_layers=p, initial_angles=0.1 * np.arange(2 * p))
            apollo = CrystalQPUFromBackend(apollo_backend)
            routing = ptn_router(sk_qaoa.hamiltonian_bqm, qpu=apollo, strategy="Line")
            qc = routing.build_qiskit(sk_qaoa.betas.tolist(), sk_qaoa.gammas.tolist(), measurement=False)
            depth = qc.depth()

            assert depth == expected_depth_ptn(n=num_qubits, p=p, local_terms=False)


def test_ptn_circuit_is_equivalent_to_sn() -> None:
    """Tests that the ptn router creates a circuit logically equivalent to the circuit produced by swap networks."""
    for p in range(1, 3):
        for num_qubits in (4, 6, 8, 9):
            sk_problem = next(sk_generator(n=num_qubits, n_instances=1, seed=1337))
            sk_qaoa = QUBOQAOA(problem=sk_problem, num_layers=p, initial_angles=list(range(1, 2 * p + 1)))
            qpu = Grid2DQPU(3, 4)

            # We skip the measurements in the qiskit circuit because we want to compare statevectors.
            routing_ptn = ptn_router(sk_qaoa.hamiltonian_bqm, qpu=qpu, strategy="Line")
            qc_ptn = routing_ptn.build_qiskit(sk_qaoa.betas.tolist(), sk_qaoa.gammas.tolist(), measurement=False)

            routing_sn = sn_router(sk_qaoa.hamiltonian_bqm, qpu=qpu)
            qc_sn = routing_sn.build_qiskit(sk_qaoa.betas.tolist(), sk_qaoa.gammas.tolist(), measurement=False)

            # In the circuits with measurements, the measurements make sure of ordering the information consistently.
            # Since we skipped the measurements, the qubits of the two circuits are in different orders.
            # Therefore, we need to figure out what's the difference and add a few swaps to fix that.
            ptn_l2h = routing_ptn.mapping._line2hard.copy()
            if p % 2:  # Odd number of layers.
                sn_l2h = routing_sn.mapping.log2hard.copy()
            else:  # Even number of layers means that the mapping was done and un-done the same amoutn of times.
                sn_l2h = routing_sn.initial_mapping.log2hard.copy()

            # Go through all logical qubits.
            for log_var in routing_sn.mapping.log2hard:
                if ptn_l2h[log_var] == sn_l2h[log_var]:
                    # If the logical qubit corresponds to the same physical qubit in both mappings, all is good.
                    continue
                else:
                    # Otherwise, apply swap (to one of the circuits) and update the list `ptn_l2h`.
                    qc_ptn.swap(ptn_l2h[log_var], sn_l2h[log_var])
                    if sn_l2h[log_var] not in ptn_l2h:
                        ptn_l2h[log_var] = sn_l2h[log_var]
                        continue
                    swapped_log = ptn_l2h.index(sn_l2h[log_var])
                    ptn_l2h[log_var], ptn_l2h[swapped_log] = ptn_l2h[swapped_log], ptn_l2h[log_var]

            sv_sn = Statevector.from_instruction(qc_sn)
            sv_ptn = Statevector.from_instruction(qc_ptn)

            assert sv_sn.equiv(sv_ptn)
