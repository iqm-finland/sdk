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
"""The testing module for the iqm/qaoa/transforming_functions.py file."""

from dimod import to_networkx_graph
from iqm.applications.mis import MISInstance
from iqm.qaoa.transforming_functions import ham_graph_to_ham_operator


def test_ham_graph_to_ham_operator(small_mis_instance: MISInstance) -> None:
    """A test for the ``ham_graph_to_ham_operator`` function.

    It takes the Hamiltonian operator obtained from the function, deconstructs it into strings (containing interaction
    sites) and coefficients (containing intraction strength) and compares with interactions saved in the BQM.
    """
    bqm_with_spins = small_mis_instance.bqm.spin

    ham_op = ham_graph_to_ham_operator(to_networkx_graph(bqm_with_spins))
    pl = ham_op.paulis
    cfs = ham_op.coeffs
    for i in range(len(pl)):
        qubits_being_acted_on = [index for index, char in enumerate(pl[i]) if char == "Z"]

        if len(qubits_being_acted_on) == 1:
            assert bqm_with_spins.get_linear(*qubits_being_acted_on) == cfs[i]
        elif len(qubits_being_acted_on) == 2:
            assert bqm_with_spins.get_quadratic(*qubits_being_acted_on) == cfs[i]
