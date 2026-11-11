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

"""Tests for the Stim-translation and Clifford-validation primitives CDR relies on."""

from iqm.error_reduction_tools.cdr.cdr_training_circuits import (
    build_clbit_to_stim_qubit_map,
    exact_expectation_from_stim,
    is_instruction_allowed_in_training_circuit,
    qubit_order,
    stim_circuit_without_measurements,
    training_circuit_to_stim,
    validate_training_circuit_is_clifford,
)
import numpy as np
import pytest
import stim

from iqm.pulse import Circuit, CircuitOperation

ATOL = 1e-10


@pytest.fixture
def native_target_circuit() -> Circuit:
    """
    Construct a model 2-qubit pulse circuit in native prx/CZ basis with one non-Clifford prx gate.
    """
    ops = (
        CircuitOperation(name="prx", locus=("QB1",), args={"angle": np.pi / 2, "phase": 0.0}),  # Clifford
        CircuitOperation(name="prx", locus=("QB2",), args={"angle": np.pi / 2, "phase": 0.0}),  # Clifford
        CircuitOperation(name="cz", locus=("QB1", "QB2")),
        CircuitOperation(name="prx", locus=("QB1",), args={"angle": 0.3, "phase": 0.0}),  # non-Clifford (slot)
        CircuitOperation(name="measure", locus=("QB1",), args={"key": "m0"}),
        CircuitOperation(name="measure", locus=("QB2",), args={"key": "m1"}),
    )
    return Circuit(name="native_target", instructions=ops)


def clifford_training_circuit() -> Circuit:
    """Construct a fully-Clifford 2-qubit pulse circuit that training_circuit_to_stim can translate directly."""
    ops = (
        CircuitOperation(name="prx", locus=("QB1",), args={"angle": np.pi / 2, "phase": 0.0}),
        CircuitOperation(name="prx", locus=("QB2",), args={"angle": np.pi / 2, "phase": 0.0}),
        CircuitOperation(name="cz", locus=("QB1", "QB2")),
        CircuitOperation(name="measure", locus=("QB1",), args={"key": "m0"}),
        CircuitOperation(name="measure", locus=("QB2",), args={"key": "m1"}),
    )
    return Circuit(name="clifford_training_circuit", instructions=ops)


def test_qubit_order_follows_first_appearance(native_target_circuit):
    """Test that qubit_order lists qubits in order of first appearance."""
    assert qubit_order(native_target_circuit) == ["QB1", "QB2"]


def test_qubit_order_does_not_repeat_a_qubit_seen_again_later():
    """Test that qubit_order lists a qubit only once, at its first appearance."""
    circuit = Circuit(
        name="c",
        instructions=(
            CircuitOperation(name="prx", locus=("QB2",), args={"angle": np.pi / 2, "phase": 0.0}),
            CircuitOperation(name="cz", locus=("QB1", "QB2")),
        ),
    )
    assert qubit_order(circuit) == ["QB2", "QB1"]


class TestIsInstructionAllowed:
    """Tests for :func:`is_instruction_allowed_in_training_circuit`."""

    def test_clifford_prx_gate_allowed(self):
        """Test that a Clifford prx gate is allowed."""
        op = CircuitOperation(name="prx", locus=("QB1",), args={"angle": np.pi / 2, "phase": 0.0})
        assert is_instruction_allowed_in_training_circuit(op)

    def test_non_clifford_prx_gate_rejected(self):
        """Test that a non-Clifford prx gate is rejected."""
        op = CircuitOperation(name="prx", locus=("QB1",), args={"angle": 0.3, "phase": 0.0})
        assert not is_instruction_allowed_in_training_circuit(op)

    def test_cz_delay_measure_and_barrier_are_allowed(self):
        """Test that cz, delay, measure, and barrier are always allowed, regardless of arguments."""
        ops = (
            CircuitOperation(name="cz", locus=("QB1", "QB2"), args={}),
            CircuitOperation(name="delay", locus=("QB1",), args={"duration": 1e-8}),
            CircuitOperation(name="measure", locus=("QB1",), args={"key": "m0"}),
            CircuitOperation(name="barrier", locus=("QB1",), args={}),
        )
        assert all(is_instruction_allowed_in_training_circuit(op) for op in ops)

    def test_an_unrecognized_instruction_is_rejected(self):
        """Test that an operation outside the native-gate set (e.g. reset) is rejected."""
        op = CircuitOperation(name="reset", locus=("QB1",), args={})
        assert not is_instruction_allowed_in_training_circuit(op)


class TestValidateTrainingCircuitIsClifford:
    """Tests for :func:`validate_training_circuit_is_clifford`."""

    def test_returns_true_on_clifford(self):
        """Test that a Clifford circuit passes validation."""
        circ = Circuit(
            name="c",
            instructions=(CircuitOperation(name="prx", locus=("QB1",), args={"angle": np.pi / 2, "phase": 0.0}),),
        )
        assert validate_training_circuit_is_clifford(circ)

    def test_raises_on_non_clifford(self):
        """Test that a non-Clifford circuit raises."""
        circ = Circuit(
            name="c",
            instructions=(CircuitOperation(name="prx", locus=("QB1",), args={"angle": 0.3, "phase": 0.0}),),
        )
        with pytest.raises(ValueError, match="Training circuit validation failed"):
            validate_training_circuit_is_clifford(circ, raise_on_failure=True)

    def test_returns_false_on_non_clifford_when_not_raising(self):
        """Test that a non-Clifford circuit returns False when raise_on_failure is False."""
        circ = Circuit(
            name="c",
            instructions=(CircuitOperation(name="prx", locus=("QB1",), args={"angle": 0.3, "phase": 0.0}),),
        )
        assert not validate_training_circuit_is_clifford(circ, raise_on_failure=False)


class TestTrainingCircuitToStim:
    """Tests for :func:`training_circuit_to_stim`."""

    def test_translates_a_clifford_training_circuit(self):
        """Test that the Stim circuit has one map entry per measurement and the same number of qubits."""
        training_circuit = clifford_training_circuit()
        stim_circuit, meas_map = training_circuit_to_stim(training_circuit, return_measurement_map=True)
        target_meas = sum(1 for op in training_circuit.instructions if op.name == "measure")
        assert len(meas_map) == target_meas
        assert stim_circuit.num_qubits == len(qubit_order(training_circuit))

    def test_clbit_index_follows_measurement_order(self):
        """Test that measurements without a clbit in their key get clbits in measurement order."""
        _, meas_map = training_circuit_to_stim(clifford_training_circuit(), return_measurement_map=True)
        clbit_indices = [entry["clbit_index"] for entry in meas_map]
        assert clbit_indices == list(range(len(meas_map)))

    @pytest.mark.parametrize(
        ("operation", "message"),
        [
            (CircuitOperation(name="measure", locus=("QB1", "QB2"), args={"key": "m0"}), "Measurement expects 1 qubit"),
            (CircuitOperation(name="cz", locus=("QB1",), args={}), "CZ expects 2 qubits"),
        ],
    )
    def test_measure_or_cz_with_the_wrong_number_of_qubits_is_rejected(self, operation, message):
        """Test that a measure not acting on 1 qubit, or a cz not acting on 2, raises ValueError."""
        with pytest.raises(ValueError, match=message):
            training_circuit_to_stim(Circuit(name="c", instructions=(operation,)))

    def test_an_unsupported_instruction_is_rejected(self):
        """Test that an operation outside prx/cz/measure/delay/barrier raises ValueError."""
        circ = Circuit(name="c", instructions=(CircuitOperation(name="reset", locus=("QB1",), args={}),))
        with pytest.raises(ValueError, match="Cannot convert training circuit operation to Stim"):
            training_circuit_to_stim(circ)

    def test_delay_and_barrier_have_no_effect_on_the_stim_circuit(self):
        """Test that delay and barrier instructions are dropped, leaving the Stim circuit unchanged."""
        with_ops = Circuit(
            name="c",
            instructions=(
                CircuitOperation(name="delay", locus=("QB1",), args={"duration": 1e-8}),
                CircuitOperation(name="barrier", locus=("QB1",), args={}),
                CircuitOperation(name="measure", locus=("QB1",), args={"key": "m0"}),
            ),
        )
        without_ops = Circuit(
            name="c",
            instructions=(CircuitOperation(name="measure", locus=("QB1",), args={"key": "m0"}),),
        )
        stim_with, _ = training_circuit_to_stim(with_ops, return_measurement_map=True)
        stim_without, _ = training_circuit_to_stim(without_ops, return_measurement_map=True)
        assert str(stim_with) == str(stim_without)

    def test_preserve_num_qubits_keeps_a_qubit_only_touched_by_a_delay(self):
        """Test that preserve_num_qubits keeps a qubit whose only instruction is a delay in the Stim qubit count."""
        circ = Circuit(
            name="c",
            instructions=(
                CircuitOperation(name="prx", locus=("QB1",), args={"angle": np.pi, "phase": 0.0}),
                CircuitOperation(name="delay", locus=("QB2",), args={"duration": 1e-8}),
                CircuitOperation(name="measure", locus=("QB1",), args={"key": "m0"}),
            ),
        )
        preserved = training_circuit_to_stim(circ, preserve_num_qubits=True, return_measurement_map=False)
        assert preserved.num_qubits == 2

        not_preserved = training_circuit_to_stim(circ, preserve_num_qubits=False, return_measurement_map=False)
        assert not_preserved.num_qubits == 1


class TestExactTrainingCircuitExpectation:
    """Tests for :func:`exact_expectation_from_stim`."""

    def test_zero_state_gives_plus_one_for_z(self):
        """Test that the |00> state gives <IZ> = 1."""
        circ = Circuit(
            name="c",
            instructions=(
                CircuitOperation(name="measure", locus=("QB1",), args={"key": "m0"}),
                CircuitOperation(name="measure", locus=("QB2",), args={"key": "m1"}),
            ),
        )
        stim_circuit, meas_map = training_circuit_to_stim(circ, return_measurement_map=True)
        val = exact_expectation_from_stim(stim_circuit, meas_map, "IZ")
        assert np.isclose(val, 1.0, atol=ATOL)

    def test_flipped_qubit_gives_minus_one(self):
        """Test that a flipped qubit gives <Z> = -1."""
        circ = Circuit(
            name="c",
            instructions=(
                CircuitOperation(name="prx", locus=("QB1",), args={"angle": np.pi, "phase": 0.0}),  # X flip
                CircuitOperation(name="measure", locus=("QB1",), args={"key": "m0"}),
            ),
        )
        stim_circuit, meas_map = training_circuit_to_stim(circ, return_measurement_map=True)
        val = exact_expectation_from_stim(stim_circuit, meas_map, "Z")
        assert np.isclose(val, -1.0, atol=ATOL)

    def test_a_weighted_sum_of_terms_combines_correctly(self):
        """Test that a weighted sum of Pauli terms is combined into a single expectation value."""
        circ = Circuit(
            name="c",
            instructions=(
                CircuitOperation(name="prx", locus=("QB1",), args={"angle": np.pi, "phase": 0.0}),  # QB1 -> -1
                CircuitOperation(name="measure", locus=("QB1",), args={"key": "m0"}),
                CircuitOperation(name="measure", locus=("QB2",), args={"key": "m1"}),  # QB2 stays +1
            ),
        )
        stim_circuit, meas_map = training_circuit_to_stim(circ, return_measurement_map=True)
        val = exact_expectation_from_stim(stim_circuit, meas_map, [[0.4, "IZ"], [0.6, "ZI"]])
        assert np.isclose(val, 0.2, atol=ATOL)

    def test_wrong_expectation_num_bits_is_rejected(self):
        """Test that a mismatched expectation_num_bits raises ValueError."""
        circ = Circuit(
            name="c",
            instructions=(CircuitOperation(name="measure", locus=("QB1",), args={"key": "m0"}),),
        )
        stim_circuit, meas_map = training_circuit_to_stim(circ, return_measurement_map=True)
        with pytest.raises(ValueError, match="does not match expectation_num_bits"):
            exact_expectation_from_stim(stim_circuit, meas_map, "Z", expectation_num_bits=2)

    def test_observable_on_an_unmeasured_clbit_is_rejected(self):
        """Test that an observable acting on a classical bit no measurement writes to raises ValueError."""
        circ = Circuit(name="c", instructions=(CircuitOperation(name="measure", locus=("QB1",), args={"key": "m0"}),))
        stim_circuit, meas_map = training_circuit_to_stim(circ, return_measurement_map=True)
        with pytest.raises(ValueError, match="index 1 has no corresponding measurement"):
            exact_expectation_from_stim(stim_circuit, meas_map, "ZZ")

    def test_measurement_map_pointing_outside_the_circuit_is_rejected(self):
        """Test that a measurement map naming a Stim qubit the circuit does not have raises ValueError."""
        circ = Circuit(name="c", instructions=(CircuitOperation(name="measure", locus=("QB1",), args={"key": "m0"}),))
        stim_circuit, meas_map = training_circuit_to_stim(circ, return_measurement_map=True)
        with pytest.raises(ValueError, match="Mapped Stim qubit 7 is out of range"):
            exact_expectation_from_stim(stim_circuit, [{**meas_map[0], "qubit": 7}], "Z")

    def test_complex_coefficient_warns(self):
        """Test that a complex coefficient left in the value warns and is returned as complex."""
        circ = Circuit(name="c", instructions=(CircuitOperation(name="measure", locus=("QB1",), args={"key": "m0"}),))
        stim_circuit, meas_map = training_circuit_to_stim(circ, return_measurement_map=True)
        with pytest.warns(UserWarning, match="non-negligible imaginary part"):
            value = exact_expectation_from_stim(stim_circuit, meas_map, [[1j, "Z"]])
        assert value == 1j


class TestStimCircuitWithoutMeasurements:
    """Tests for :func:`stim_circuit_without_measurements`."""

    def test_removes_measurements_but_keeps_other_gates(self):
        """Test that measurement instructions are dropped and every other instruction is kept, in order."""
        circuit = stim.Circuit("H 0\nCX 0 1\nM 0 1")
        cleaned = stim_circuit_without_measurements(circuit)
        assert [instruction.name for instruction in cleaned] == ["H", "CX"]

    def test_strips_measurements_nested_inside_a_repeat_block(self):
        """Test that measurements nested inside a REPEAT block are also stripped."""
        circuit = stim.Circuit("REPEAT 2 {\n    X 0\n    M 0\n}")
        cleaned = stim_circuit_without_measurements(circuit)
        assert "M" not in str(cleaned)
        # Two X gates cancel out
        simulator = stim.TableauSimulator()
        simulator.do_circuit(cleaned)
        assert simulator.peek_observable_expectation(stim.PauliString("Z")) == pytest.approx(1.0)


class TestBuildClbitToStimQubitMap:
    """Tests for :func:`build_clbit_to_stim_qubit_map`."""

    def test_maps_each_classical_bit_to_its_measured_qubit(self):
        """Test that each classical bit is mapped to the Stim qubit index it was measured from."""
        measurement_map = [
            {"stim_measurement_index": 0, "qubit": 2, "clbit_index": 0, "instruction_index": 0},
            {"stim_measurement_index": 1, "qubit": 0, "clbit_index": 1, "instruction_index": 1},
        ]
        assert build_clbit_to_stim_qubit_map(measurement_map) == {0: 2, 1: 0}

    def test_a_measurement_entry_without_a_clbit_index_is_rejected(self):
        """Test that an entry with clbit_index=None raises ValueError."""
        measurement_map = [{"stim_measurement_index": 0, "qubit": 0, "clbit_index": None, "instruction_index": 0}]
        with pytest.raises(ValueError, match="has no classical bit"):
            build_clbit_to_stim_qubit_map(measurement_map)

    def test_two_measurements_writing_the_same_classical_bit_are_rejected(self):
        """Test that two measurements assigned to the same classical bit raise ValueError."""
        measurement_map = [
            {"stim_measurement_index": 0, "qubit": 0, "clbit_index": 0, "instruction_index": 0},
            {"stim_measurement_index": 1, "qubit": 1, "clbit_index": 0, "instruction_index": 1},
        ]
        with pytest.raises(ValueError, match="assigns classical bit 0"):
            build_clbit_to_stim_qubit_map(measurement_map)
