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

"""Module containing the implementation of the parity twine network strategies based on :cite:`Dreier_2025`."""

from __future__ import annotations

from dimod import BinaryQuadraticModel
from iqm.qaoa.transpiler.quantum_hardware import QPU
from iqm.qaoa.transpiler.routing import CircuitSynthesis, ParityMapping
from iqm.qaoa.transpiler.sparse.two_color_mapper import _greedy_longest_path_with_backtracking


def ptn_router(problem_bqm: BinaryQuadraticModel, qpu: QPU) -> CircuitSynthesis:
    """Construct a routing object for the ParityTwineNetwork (PTN).

    Args:
        problem_bqm: The binary quadratic model representing the optimization problem.
        qpu: QPU instance.

    Returns:
        A ``CircuitSynthesis`` object implementing the requested routing strategy.

    """
    num_vars = problem_bqm.num_variables

    # PTN works on a line, so we need to find a line of qubits on the QPU.
    path_of_hw_qubits = _greedy_longest_path_with_backtracking(qpu.hardware_graph)[:num_vars]

    # We place the problem variables along this line (the order doesn't matter).
    init_map = {hw_qb: {problem_bqm.variables[idx]} for idx, hw_qb in enumerate(path_of_hw_qubits)}
    mapping = ParityMapping(qpu, init_map)

    cs = CircuitSynthesis(mapping)

    # This is the construction of the parity twine network. Several layers of CNOTs in a triangular shape.
    for layer in range(num_vars - 1):  # One fewer layers than qubits.
        for qbt in range(num_vars - layer - 1):
            cs.cnot(path_of_hw_qubits[qbt + 1], path_of_hw_qubits[qbt], False)
            cs.cnot(path_of_hw_qubits[qbt], path_of_hw_qubits[qbt + 1], True)

    # The uncomputing of the parities at the end.
    cs.begin_uncompute()
    for qbt in range(num_vars - 1, 0, -1):
        cs.cnot(path_of_hw_qubits[qbt], path_of_hw_qubits[qbt - 1])

    return cs
