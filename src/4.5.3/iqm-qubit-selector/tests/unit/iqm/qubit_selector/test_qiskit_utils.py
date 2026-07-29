# Copyright 2022-2025 IQM
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for ``qiskit_utils`` module."""

from iqm.qiskit_iqm.iqm_backend import IQMBackendBase
from iqm.qubit_selector.qiskit_utils import (
    CircuitType,
    _cost_operator,
    _mixer_operator,
    active_bits,
    deflate_circuit,
    extract_2q_interactions,
    get_circuit,
)
import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile


@pytest.mark.parametrize("circuit_type", CircuitType)
def test_get_circuit_returns_quantum_circuit_of_predefined_type(circuit_type: CircuitType) -> None:
    """Test that ``get_circuit`` function creates circuits of types predefined in ``CircuitType``."""
    expected_num_qubits = 5

    qc = get_circuit(circuit_type, num_qubits=expected_num_qubits)
    last_instructions = qc.data[-expected_num_qubits:]

    assert isinstance(qc, QuantumCircuit)
    assert qc.num_qubits == expected_num_qubits
    assert all(
        instruction.name == "measure" for instruction in last_instructions
    )  # Because of call to ``measure_all()``


def test_get_circuit_raises_value_error_upon_unsupported_circuit_type() -> None:
    """Test that ``get_circuit`` function can handle names of unsupported circuit types by raising exceptions."""
    unsupported_circuit_type = "superresisting_qubits"

    with pytest.raises(ValueError) as supported_circuit_types:
        get_circuit(unsupported_circuit_type, 2)

    assert unsupported_circuit_type not in str(supported_circuit_types.value)


def test_active_bits_finds_active_and_classical_bits_in_circuit() -> None:
    """Test that ``active_bits`` finds active bits (quantum and classical) in a circuit with multiple gates."""
    qc = QuantumCircuit(3, 2)
    qc.x(0)
    qc.cx(0, 1)
    qc.measure([1, 2], [0, 1])

    active_qubits, active_clbits = active_bits(qc)

    assert len(active_qubits) == 3
    assert len(active_clbits) == 2


def test_deflate_circuit_does_not_count_barrier_as_active_qubits_in_transpiled_circuit(
    fake_apollo_backend: IQMBackendBase,
) -> None:
    """Test that a barrier itself does not mean active qubits."""
    qc = QuantumCircuit(10)
    qc.barrier()
    expected_qc = QuantumCircuit()

    transpiled_qc = transpile(qc, fake_apollo_backend)
    deflated_qc = deflate_circuit(transpiled_qc)

    assert deflated_qc == expected_qc


def test_cost_hamiltonian_operator_is_applied_onto_circuit() -> None:
    """Test that a Cost Hamiltonian operator is applied onto a quantum circuit."""
    num_qubits = 3
    gamma = 1.72414847
    qc = QuantumCircuit(num_qubits)
    expected_operator_instruction_pattern = ["cx", "rz", "cx", "cx", "rz", "cx"]
    expected_rz_angle = -2 * gamma

    _cost_operator(qc, gamma, num_qubits)

    assert [instruction.name for instruction in qc.data] == expected_operator_instruction_pattern

    for instruction in qc.data:
        if instruction.name == "rz":
            rz_angle = instruction.params[0]
            assert pytest.approx(rz_angle) == expected_rz_angle


def test_mixer_hamiltonian_operator_is_applied_onto_circuit() -> None:
    """Test that a Mixer Hamiltonian operator is applied onto a quantum circuit."""
    num_qubits = 3
    beta = 2.24683366
    qc = QuantumCircuit(num_qubits)
    expected_operator_instruction_pattern = ["rx", "rx", "rx"]
    expected_rx_angle = 2 * beta

    _mixer_operator(qc, beta, num_qubits)

    assert [instruction.name for instruction in qc.data] == expected_operator_instruction_pattern

    for instruction in qc.data:
        rx_angle = instruction.params[0]
        assert pytest.approx(rx_angle) == expected_rx_angle


def test_2q_interactions_are_extracted(ghz_circuit_with_4_qubits: QuantumCircuit) -> None:
    """Test that 2-qubit interactions are extracted correctly."""
    expected_interactions = [(2, 3), (1, 2), (0, 1)]

    interactions = extract_2q_interactions(ghz_circuit_with_4_qubits)

    assert interactions == expected_interactions


def test_no_2q_interactions_are_extracted_from_circuit_that_has_no_gates() -> None:
    """Test that no 2-qubit interactions are extracted from a circuit that has no gates."""
    num_qubits = 3
    qc = QuantumCircuit(num_qubits)
    expected_interactions = []

    interactions = extract_2q_interactions(qc)

    assert interactions == expected_interactions


def test_no_2q_interactions_are_extracted_from_circuit_that_has_1q_gates_only() -> None:
    """Test that no 2-qubit interactions are extracted from a circuit that has only single qubit gates."""
    num_qubits = 3
    qc = QuantumCircuit(num_qubits)
    qc.x(0)
    qc.h(1)
    qc.rx(np.pi, 0)
    expected_interactions = []

    interactions = extract_2q_interactions(qc)

    assert interactions == expected_interactions
