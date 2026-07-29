# Copyright 2022-2026 IQM
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

"""End-to-end tests for circuit randomization workflow.

This module tests the complete pipeline of:
1. Circuit definition and transpilation
2. Conversion from Qiskit to IQM representation
3. Circuit randomization with Pauli twirling
4. Conversion back to Qiskit
5. Untwirling of measurement results

Tests validate that the randomization process preserves measurement statistics
when un-twirling is applied.
"""

from iqm.error_reduction_tools.twirling.twirling_modifiers import randomize_circuit
from iqm.error_reduction_tools.twirling.twirling_processors import untwirl_counts
from iqm.error_reduction_tools.utils.general_utils import total_variational_distance
from iqm.error_reduction_tools.utils.qiskit_utils import (
    from_iqm_to_qiskit,
    from_qiskit_to_iqm,
)
from iqm.qiskit_iqm import optimize_single_qubit_gates
from iqm.qiskit_iqm.fake_backends import IQMFakeApollo
import numpy as np
import pytest
from qiskit import ClassicalRegister, QuantumCircuit, transpile
from qiskit.circuit.library import QuantumVolume

from ...utils_test import simulate_statevector_outcomes

THRESHOLD = 1e-10


@pytest.fixture(scope="module")
def circuits():
    """Create test circuits A and QV."""
    circs = {}

    # Circuit A: Custom 5-qubit circuit
    circs["A"] = QuantumCircuit(5, 4)
    circs["A"].ry(-2, 0)
    circs["A"].rx(-0.5, 1)
    circs["A"].ry(0.2, 2)
    circs["A"].rx(3, 3)
    circs["A"].ry(1.5, 4)
    circs["A"].cz(0, 1)
    circs["A"].rx(1.6, 1)
    circs["A"].cz(1, 2)
    circs["A"].cz(3, 0)
    circs["A"].ry(0.7, [0, 1, 2])
    circs["A"].cz(4, 3)
    circs["A"].cz(0, 1)
    circs["A"].ry(-1.7, [0, 2, 4])
    circs["A"].cz(2, 3)
    circs["A"].cz(4, 1)
    circs["A"].rx(2, 1)
    circs["A"].cz(4, 3)
    circs["A"].ry(2, 3)
    circs["A"].cz(1, 4)
    circs["A"].measure([0, 1, 4, 3], [0, 1, 2, 3])

    # Circuit QV: Quantum Volume circuit
    num_qubits = 6
    depth = 6
    repetitions = 1
    unit_circuit = QuantumVolume(num_qubits, depth, seed=0)
    circs["QV"] = unit_circuit.copy()
    for r in range(repetitions - 1):
        circs["QV"].compose(unit_circuit, inplace=True)

    c = ClassicalRegister(num_qubits, "c")
    circs["QV"].add_register(c)
    circs["QV"].measure(range(num_qubits), [c[j] for j in range(num_qubits)])

    return circs


@pytest.fixture(scope="module")
def apollo_backend():
    """Create IQM Apollo fake backend."""
    return IQMFakeApollo()


@pytest.fixture(scope="module")
def transpiled_circuits(circuits, apollo_backend):
    """Transpile circuits for both aer and apollo backends."""
    transpiled = {}

    # Transpile for aer (generic basis gates)
    transpiled["aer"] = {}
    for name, circuit in circuits.items():
        transpiled["aer"][name] = optimize_single_qubit_gates(
            transpile(
                circuit,
                basis_gates=["r", "cz"],
                optimization_level=3,
            )
        )

    # Transpile for apollo backend
    transpiled["apollo"] = {}
    for name, circuit in circuits.items():
        transpiled["apollo"][name] = transpile(
            circuit,
            backend=apollo_backend,
            optimization_level=3,
        )

    return transpiled


def test_qiskit_iqm_conversion(transpiled_circuits):
    """Test that conversion between Qiskit and IQM representations preserves circuit behavior.

    This is a sanity check that the conversion utilities work correctly.
    """
    for backend_type in ["aer", "apollo"]:
        for key in transpiled_circuits[backend_type].keys():
            qiskit_circuit = transpiled_circuits[backend_type][key]
            pulse_circuit, qubit_index_to_name = from_qiskit_to_iqm(qiskit_circuit)
            recovered_qiskit_circuit = from_iqm_to_qiskit(pulse_circuit, qubit_index_to_name, qiskit_circuit)

            original_counts = simulate_statevector_outcomes(qiskit_circuit)
            recovered_counts = simulate_statevector_outcomes(recovered_qiskit_circuit)

            distance = total_variational_distance(original_counts, recovered_counts)
            assert distance < THRESHOLD, (
                f"Conversion round-trip should preserve circuit behavior for "
                f"{backend_type}/{key}. L1 distance: {distance}"
            )


@pytest.mark.parametrize("backend_type", ["aer", "apollo"])
@pytest.mark.parametrize(
    "readout_twirling",
    [
        pytest.param(False, id="no_twirl"),
        pytest.param(True, id="full_twirl"),
        pytest.param({"QB1": "I", "QB2": "X", "QB6": "X"}, id="enforced_twirl"),
    ],
)
@pytest.mark.parametrize(
    "twirling_probabilities",
    [
        pytest.param(None, id="prob_none"),
        pytest.param("Random", id="prob_random"),
        pytest.param(0.0, id="prob_zero"),
    ],
)
def test_circuit_randomization(
    backend_type,
    readout_twirling,
    twirling_probabilities,
    transpiled_circuits,
    apollo_backend,
):
    """Test that circuit randomization preserves measurement statistics after untwirling.

    This test validates the complete randomization workflow:
    1. Converts Qiskit circuit to pulse representation
    2. Randomizes the circuit with specified parameters
    3. Converts back to Qiskit
    4. Verifies that untwirling recovers the original statistics

    Args:
        backend_type: Either "aer" or "apollo" backend type.
        readout_twirling: False, True, or dict specifying enforced twirlings.
        twirling_probabilities: None, "Random", or float value.
        transpiled_circuits: Fixture providing transpiled circuits.
        apollo_backend: Fixture providing Apollo backend.
    """

    # Skip apollo backend for all but one parameter combination to save time
    if backend_type == "apollo" and not (readout_twirling is True and twirling_probabilities == "Random"):
        pytest.skip("Skipping apollo backend for this parameter combination to reduce test time")

    # Determine number of random seeds based on backend
    num_rand = 10 if backend_type == "aer" else 1

    for seed in range(num_rand):
        rgen = np.random.default_rng(seed=seed)

        for circuit_type, qc in transpiled_circuits[backend_type].items():
            # Get exact counts from original circuit
            exact_counts = simulate_statevector_outcomes(qc)

            # Conversion to IQM format
            backend = None if backend_type == "aer" else apollo_backend
            pulse_circuit, qubit_index_to_name = from_qiskit_to_iqm(qc, backend)

            # Randomization
            if isinstance(twirling_probabilities, str) and twirling_probabilities == "Random":
                tp = rgen.random(size=sum(pulse_circuit.sqg_counter.values()))
            else:
                tp = twirling_probabilities

            randomized_pulse_circuit = randomize_circuit(
                circuit=pulse_circuit,
                rgen=rgen,
                readout_twirling=readout_twirling,
                twirling_probabilities=tp,
            )

            # Back to Qiskit
            randomized_qiskit_circuit = from_iqm_to_qiskit(randomized_pulse_circuit, qubit_index_to_name, qc)
            randomized_counts = simulate_statevector_outcomes(randomized_qiskit_circuit)

            # Assertions

            # If readout_twirling is a dict, verify enforced twirlings are respected
            if isinstance(readout_twirling, dict):
                for key, val in readout_twirling.items():
                    actual_twirl = randomized_pulse_circuit.rot_dict.get(key, 0)
                    assert actual_twirl in [
                        val,
                        0,
                    ], f"Enforced twirling not respected: {key} should be {val}, got {actual_twirl}"

            # If no readout twirling, randomized counts should match exact counts
            if readout_twirling is False:
                assert total_variational_distance(randomized_counts, exact_counts) < THRESHOLD, (
                    f"Without readout twirling, randomized counts should match exact counts. "
                    f"L1 distance: {total_variational_distance(randomized_counts, exact_counts)}"
                )

            # Untwirling should always recover the original statistics
            untwirled_counts = untwirl_counts(randomized_counts, randomized_qiskit_circuit.metadata["rot_string"])
            distance = total_variational_distance(untwirled_counts, exact_counts)
            assert distance < THRESHOLD, (
                f"Untwirled counts should match exact counts for {backend_type}/{circuit_type}. "
                f"L1 distance: {distance}, seed: {seed}, "
                f"readout_twirling: {readout_twirling}, twirling_prob: {twirling_probabilities}"
            )
