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
"""A module for custom functions that transform data from one format to another format"""

import networkx as nx
from qiskit.quantum_info import PauliList, SparsePauliOp


def ham_graph_to_ham_operator(ham_graph: nx.Graph) -> SparsePauliOp:
    """A function to transform Hamiltonian represented as a :class:`~networkx.Graph` into
    a :class:`~qiskit.quantum_info.SparsePauliOp` object.

    A Hamiltonian as :class:`~qiskit.quantum_info.SparsePauliOp` may be used by :mod:`qiskit` functions that e.g.,
    calculate expectation values.

    Args:
        ham_graph: A :class:`~networkx.Graph` whose nodes and edges have a parameter ``bias`` whose value corresponds
            to the coefficients before the corresponding *Z* and *ZZ* operators in the problem Hamiltonian.

    Returns:
        The Hamiltonian as :class:`~qiskit.quantum_info.SparsePauliOp` to be used by :mod:`qiskit`.

    """
    pauli_strings: list[str] = []
    coefficients: list[float] = []
    for node in ham_graph.nodes:
        string_list = ["I"] * (max(ham_graph.nodes()) + 1)
        string_list[node] = "Z"
        string_to_add = "".join(string_list)
        if string_to_add in pauli_strings:
            coefficients[pauli_strings.index(string_to_add)] += ham_graph.nodes[node]["bias"]
        else:
            pauli_strings.append(string_to_add)
            coefficients.append(ham_graph.nodes[node]["bias"])
    for n1, n2 in ham_graph.edges:
        string_list = ["I"] * (max(ham_graph.nodes()) + 1)
        string_list[n1] = "Z"
        string_list[n2] = "Z"
        string_to_add = "".join(string_list)
        if string_to_add in pauli_strings:
            coefficients[pauli_strings.index(string_to_add)] += ham_graph[n1][n2]["bias"]
        else:
            pauli_strings.append(string_to_add)
            coefficients.append(ham_graph[n1][n2]["bias"])

    pauli_list = PauliList(pauli_strings)
    return SparsePauliOp(pauli_list, coefficients)
