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
"""A module for custom functions that transform data from one format to another format."""

from dimod import BinaryQuadraticModel
from dimod.higherorder import BinaryPolynomial
from qiskit.quantum_info import PauliList, SparsePauliOp


def ham_bqm_to_ham_operator(ham_bqm: BinaryQuadraticModel) -> SparsePauliOp:
    """A function to transform Hamiltonian represented as a BQM into a :class:`~qiskit.quantum_info.SparsePauliOp`.

    A Hamiltonian as :class:`~qiskit.quantum_info.SparsePauliOp` may be used by :mod:`qiskit` functions that e.g.,
    calculate expectation values.

    Args:
        ham_bqm: A :class:`~dimod.BinaryQuadraticModel` with ``vartype=SPIN``, i.e., the linear and quadratic terms
            correspond to the coefficients before the corresponding *Z* and *ZZ* operators in the problem Hamiltonian.

    Returns:
        The Hamiltonian as :class:`~qiskit.quantum_info.SparsePauliOp` to be used by :mod:`qiskit`.

    """
    pauli_strings: list[str] = []
    coefficients: list[float] = []

    n = ham_bqm.num_variables

    # Start with the constant term.
    if ham_bqm.offset:  # If it it's non-zero.
        string_to_add = positions_to_pauli_string(set(), n)
        pauli_strings.append(string_to_add)
        coefficients.append(float(ham_bqm.offset))

    for i, var in enumerate(ham_bqm.variables):
        string_to_add = positions_to_pauli_string({i}, n)
        if string_to_add in pauli_strings:
            coefficients[pauli_strings.index(string_to_add)] += float(ham_bqm.get_linear(var))
        else:
            pauli_strings.append(string_to_add)
            coefficients.append(float(ham_bqm.get_linear(var)))

    for v1, v2 in ham_bqm.quadratic.keys():
        string_to_add = positions_to_pauli_string({ham_bqm.variables.index(v1), ham_bqm.variables.index(v2)}, n)
        if string_to_add in pauli_strings:
            coefficients[pauli_strings.index(string_to_add)] += ham_bqm.get_quadratic(v1, v2)
        else:
            pauli_strings.append(string_to_add)
            coefficients.append(float(ham_bqm.get_quadratic(v1, v2)))

    pauli_list = PauliList(pauli_strings)
    return SparsePauliOp(pauli_list, coefficients)


def ham_bp_to_ham_operator(ham_bp: BinaryPolynomial) -> SparsePauliOp:
    """A function to transform a Hamiltonian represented as a BP into a :class:`~qiskit.quantum_info.SparsePauliOp`.

    A Hamiltonian as :class:`~qiskit.quantum_info.SparsePauliOp` may be used by :mod:`qiskit` functions that e.g.,
    calculate expectation values. The input :class:`dimod.higherorder.BinaryPolynomial` needs to have sortable
    variables. This is already the case if the ``ham_bp`` comes from a :class:`~iqm.applications.hubo.HUBOInstance`.

    Args:
        ham_bp: A :class:`~dimod.higherorder.BinaryPolynomial` with ``vartype=SPIN``, where each term corresponds to a
            product of spin variables and its coefficient represents the prefactor of the associated multi-qubit *Z...Z*
            Pauli operator in the problem Hamiltonian.

    Returns:
        The Hamiltonian as :class:`~qiskit.quantum_info.SparsePauliOp` to be used by :mod:`qiskit`.

    Raises:
        TypeError: If the variables of the input ``ham_bp`` are not sortable.

    """
    pauli_strings: list[str] = []
    coefficients: list[float] = []

    n = len(ham_bp.variables)
    try:
        sorted_vars = sorted(ham_bp.variables)
    except TypeError as exc:
        raise TypeError("Variables of 'ham_bp' must be sortable (mutually comparable).") from exc

    for term, coeff in ham_bp.items():
        qb_indices = {sorted_vars.index(var) for var in term}
        string_to_add = positions_to_pauli_string(qb_indices, n)

        if string_to_add in pauli_strings:
            coefficients[pauli_strings.index(string_to_add)] += float(coeff)
        else:
            pauli_strings.append(string_to_add)
            coefficients.append(float(coeff))

    pauli_list = PauliList(pauli_strings)
    return SparsePauliOp(pauli_list, coefficients)


def positions_to_pauli_string(positions: set[int], num_qubits: int) -> str:
    """Create a string with the character "Z" on the specified positions and the character "I" elsewhere.

    The string is intended to be used to construct a :class:`~qiskit.quantum_info.PauliList` and from it a
    :class:`~qiskit.quantum_info.SparsePauliOp`.

    Args:
        positions: Set of qubit indices where "Z" should be.
        num_qubits: Total number of qubits in the system (i.e., the length of the string).

    Returns:
        The Pauli string.

    Raises:
        ValueError: If any of the qubit indices is negative or too large (more than ``num_qubits-1``).

    """
    pauli_str_list = ["I"] * num_qubits
    for q in positions:
        if q < 0 or q >= num_qubits:
            raise ValueError(f"Qubit index {q} out of bounds for {num_qubits} qubits.")
        pauli_str_list[q] = "Z"

    pauli_str = "".join(pauli_str_list)

    return pauli_str
