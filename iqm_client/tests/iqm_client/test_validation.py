# Copyright 2024 IQM client developers
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
"""Tests for circuit validation."""

from math import pi

from iqm.iqm_client import (
    CircuitValidationError,
)
from iqm.iqm_client.validation import validate_circuit_instructions, validate_circuit_moves, validate_instruction
import pytest

from iqm.pulse import Circuit, CircuitOperation
from iqm.station_control.interface.models import DynamicQuantumArchitecture, MoveGateValidationMode

sample_qb_mapping = {"0": "CR1", "1": "QB1", "2": "QB2", "3": "QB3", "100": "CR2"}
reverse_qb_mapping = {value: key for key, value in sample_qb_mapping.items()}


@pytest.mark.parametrize(
    "instruction",
    [
        CircuitOperation(name="barrier", locus=("QB1",), args={}),
        CircuitOperation(name="barrier", locus=("QB1", "QB2"), args={}),
        CircuitOperation(name="barrier", locus=("QB2", "QB1"), args={}),  # barrier can use any loci
        CircuitOperation(name="delay", locus=("QB1",), args={"duration": 80e-9}),
        CircuitOperation(name="delay", locus=("QB1", "QB2"), args={"duration": 40e-9}),
        CircuitOperation(name="delay", locus=("QB2", "QB1"), args={"duration": 100e-9}),  # delay can use any loci
        CircuitOperation(name="prx", locus=("QB1",), args={"phase": 0.6 * pi, "angle": -0.4 * pi}),
        CircuitOperation(
            name="cc_prx",
            locus=("QB1",),
            args={"phase": 0.6 * pi, "angle": -0.4 * pi, "feedback_key": "f1", "feedback_qubit": "QB2"},
        ),
        CircuitOperation(name="reset", locus=("QB1",), args={}),
        CircuitOperation(name="cz", locus=("QB1", "QB2"), args={}),
        CircuitOperation(name="cz", locus=("QB2", "QB1"), args={}),  # CZ is symmetric
        CircuitOperation(name="measure", locus=("QB1",), args={"key": "m"}),
        CircuitOperation(name="measure", locus=("QB1", "QB2"), args={"key": "m"}),  # measure is factorizable
        CircuitOperation(name="measure", locus=("QB2", "QB1"), args={"key": "m"}),  # measure is factorizable
    ],
)
def test_valid_instruction(sample_dynamic_architecture, instruction):
    """Valid instructions must pass validation."""
    validate_instruction(sample_dynamic_architecture, instruction, None)


@pytest.mark.parametrize(
    "instruction,match",
    [
        [CircuitOperation(name="barrier", locus=("QB1", "QB2", "XXX"), args={}), "does not exist"],
        [CircuitOperation(name="delay", locus=("YYY",), args={"duration": 40e-9}), "does not exist"],
        [
            CircuitOperation(name="prx", locus=("QB4",), args={"phase": 0.6 * pi, "angle": -0.4 * pi}),
            "not allowed as locus for 'prx'",
        ],
        [CircuitOperation(name="cz", locus=("QB2", "QB4"), args={}), "not allowed as locus for 'cz'"],
        [
            CircuitOperation(name="measure", locus=("QB1", "QB4"), args={"key": "m"}),
            "not allowed as locus for 'measure'",
        ],
        [CircuitOperation(name="measure", locus=("QB4",), args={"key": "m"}), "not allowed as locus for 'measure'"],
        [
            CircuitOperation(name="cz", locus=("QB1", "QB2"), args={}, implementation="xyz"),
            "'cz' implementation 'xyz' is not supported",
        ],
        [
            CircuitOperation(
                name="prx", locus=("QB2",), args={"phase": 0.6 * pi, "angle": -0.4 * pi}, implementation="drag_crf"
            ),
            "not allowed as locus for 'prx.drag_crf'",
        ],
    ],
)
def test_invalid_instruction(sample_dynamic_architecture, instruction, match):
    """Invalid instructions must not pass validation."""
    with pytest.raises(CircuitValidationError, match=match):
        validate_instruction(sample_dynamic_architecture, instruction, None)


@pytest.mark.parametrize("qubit_mapping", [None, sample_qb_mapping])
@pytest.mark.parametrize("qubits", [("QB1", "CR1"), ("CR1", "QB1"), ("CR1", "QB2")])
def test_allowed_cz_qubits(sample_move_architecture, qubits, qubit_mapping):
    """
    Tests that instruction validation passes for allowed CZ loci
    """
    if qubit_mapping:
        qubits = [reverse_qb_mapping[q] for q in qubits]
    validate_instruction(
        sample_move_architecture,
        CircuitOperation(name="cz", locus=qubits, args={}),
        qubit_mapping,
    )


@pytest.mark.parametrize("qubit_mapping", [None, sample_qb_mapping])
@pytest.mark.parametrize(
    "qubits", [["QB1", "QB2"], ["QB2", "QB1"], ["QB1", "QB1"], ["QB3", "QB1"], ["CR1", "CR1"], ["CR1", "QB3"]]
)
def test_disallowed_cz_qubits(sample_move_architecture, qubits, qubit_mapping):
    """
    Tests that instruction validation fails for loci that are not allowed for CZ by the quantum architecture
    """
    if qubit_mapping:
        qubits = [reverse_qb_mapping[q] for q in qubits]
    with pytest.raises(CircuitValidationError, match="not allowed as locus for 'cz'"):
        validate_instruction(
            sample_move_architecture,
            CircuitOperation(name="cz", locus=qubits, args={}),
            qubit_mapping,
        )


@pytest.mark.parametrize("qubit_mapping", [None, sample_qb_mapping])
@pytest.mark.parametrize("qubits", [("QB3", "CR1")])
def test_allowed_move_qubits(sample_move_architecture, qubits, qubit_mapping):
    """
    Tests that instruction validation passes for allowed MOVE loci
    """
    if qubit_mapping:
        qubits = [reverse_qb_mapping[q] for q in qubits]

    validate_instruction(
        sample_move_architecture,
        CircuitOperation(name="move", locus=qubits, args={}),
        qubit_mapping,
    )


@pytest.mark.parametrize("qubit_mapping", [None, sample_qb_mapping])
@pytest.mark.parametrize(
    "qubits",
    [["QB1", "QB2"], ["QB2", "QB1"], ["QB1", "QB1"], ["QB1", "CR1"], ["CR1", "CR1"], ["CR1", "QB3"]],
)
def test_disallowed_move_qubits(sample_move_architecture, qubits, qubit_mapping):
    """
    Tests that instruction validation fails for loci that are not allowed for MOVE by the quantum architecture
    """
    if qubit_mapping:
        qubits = [reverse_qb_mapping[q] for q in qubits]

    with pytest.raises(CircuitValidationError, match="not allowed as locus for 'move'"):
        validate_instruction(
            sample_move_architecture,
            CircuitOperation(name="move", locus=qubits, args={}),
            qubit_mapping,
        )


@pytest.mark.parametrize("qubits", [["QB1", "QB2"], ["QB2"], ["QB1", "QB2", "QB3"], ["QB3", "QB1"], ["QB1"]])
def test_allowed_measure_qubits(sample_move_architecture, qubits):
    """
    Tests that instruction validation succeeds for loci that are any combination of valid measure qubits
    """
    validate_instruction(
        sample_move_architecture,
        CircuitOperation(name="measure", locus=qubits, args={"key": "measure_1"}),
        None,
    )


@pytest.mark.parametrize("qubits", [["QB1", "CR1"], ["CR1"], ["QB1", "QB2", "QB4"], ["QB4"]])
def test_disallowed_measure_qubits(sample_move_architecture, qubits):
    """
    Tests that instruction validation fails for loci containing any qubits that are not valid measure qubits
    """
    with pytest.raises(CircuitValidationError, match="is not allowed as locus for 'measure'"):
        validate_instruction(
            sample_move_architecture,
            CircuitOperation(name="measure", locus=qubits, args={"key": "measure_1"}),
            None,
        )


def test_measurement_keys_must_be_unique(sample_move_architecture):
    """
    Tests that all measure instructions in a circuit must have unique keys.
    """
    circuit = Circuit(
        name="Test circuit",
        instructions=[
            CircuitOperation(name="measure", locus=("QB1",), args={"key": "a"}),
            CircuitOperation(name="measure", locus=("QB2",), args={"key": "a"}),
        ],
    )
    with pytest.raises(CircuitValidationError, match="has a non-unique measurement key"):
        validate_circuit_instructions(
            sample_move_architecture,
            [circuit],
        )


def test_same_measurement_key_in_different_circuits(sample_move_architecture):
    """
    Tests that the same measurement key can be used in different circuits.
    """
    circuits = [
        Circuit(
            name="Test circuit 1",
            instructions=[
                CircuitOperation(name="measure", locus=("QB1",), args={"key": "a"}),
            ],
        ),
        Circuit(
            name="Test circuit 2",
            instructions=[
                CircuitOperation(name="measure", locus=("QB1",), args={"key": "a"}),
            ],
        ),
    ]
    validate_circuit_instructions(
        sample_move_architecture,
        circuits,
    )


def test_qir_mixed_with_iqm_circuits(sample_dynamic_architecture):
    """
    Tests that the same measurement key can be used in different circuits.
    """
    circuits = [
        "This is QIR",
        Circuit(
            name="Test circuit 1",
            instructions=[
                CircuitOperation(name="measure", locus=("QB1",), args={"key": "a"}),
            ],
        ),
        "This is also QIR",
        Circuit(
            name="Test circuit 2",
            instructions=[
                CircuitOperation(name="measure", locus=("QB1",), args={"key": "a"}),
            ],
        ),
        "And this is QIR as well",
    ]
    validate_circuit_instructions(
        sample_dynamic_architecture,
        circuits,
    )


@pytest.mark.parametrize(
    "qubits",
    [
        ["CR1", "QB1", "QB2", "QB3"],
        ["QB1", "CR1", "QB2", "QB3"],
        ["QB1", "CR1", "QB2"],
    ],
)
def test_barrier(sample_move_architecture, qubits):
    """
    Tests that instruction validation passes for the barrier operation
    """
    validate_instruction(
        sample_move_architecture,
        CircuitOperation(name="barrier", locus=qubits, args={}),
        None,
    )


class TestMoveValidation:
    """Tests the validation of MOVE instructions."""

    @staticmethod
    def make_circuit_and_check(
        instructions: tuple[CircuitOperation, ...],
        arch: DynamicQuantumArchitecture,
        validate_moves: MoveGateValidationMode,
        qubit_mapping=None,
    ):
        """Validate the given instructions (as a circuit)."""
        circuit = Circuit(name="Move validation circuit", instructions=instructions)
        validate_circuit_instructions(
            arch,
            [circuit],
            qubit_mapping,
            validate_moves=validate_moves,
        )

    @pytest.mark.parametrize("validate_moves", list(MoveGateValidationMode))
    @pytest.mark.parametrize(
        "instructions",
        [
            (CircuitOperation(name="move", locus=("QB3", "CR1"), args={}),),
            (CircuitOperation(name="move", locus=("QB3", "CR1"), args={}),) * 3,
        ],
    )
    def test_non_sandwich_move(self, sample_move_architecture, validate_moves, instructions):
        """Non-sandwich MOVEs are not allowed."""
        if validate_moves != MoveGateValidationMode.NONE:
            with pytest.raises(CircuitValidationError, match=r"qubit state\(s\) are still in a resonator"):
                TestMoveValidation.make_circuit_and_check(instructions, sample_move_architecture, validate_moves)
        else:
            TestMoveValidation.make_circuit_and_check(instructions, sample_move_architecture, validate_moves)

    @pytest.mark.parametrize("validate_moves", list(MoveGateValidationMode))
    def test_move_sandwich(self, sample_move_architecture, validate_moves):
        """Valid pair of MOVEs."""
        move = CircuitOperation(name="move", locus=("QB3", "CR1"), args={})
        TestMoveValidation.make_circuit_and_check((move, move), sample_move_architecture, validate_moves)

    @pytest.mark.parametrize("validate_moves", list(MoveGateValidationMode))
    def test_bad_move_occupied_resonator(self, sample_move_architecture, validate_moves):
        """Moving a qubit state into an occupied resonator."""
        move = CircuitOperation(name="move", locus=("QB3", "CR1"), args={})
        invalid_sandwich_circuit = Circuit(
            name="Move validation circuit",
            instructions=(
                move,
                CircuitOperation(name="move", locus=("QB2", "CR1"), args={}),
            ),  # this MOVE locus is not in the architecture, but only checking MOVE validation
        )
        if validate_moves != MoveGateValidationMode.NONE:
            with pytest.raises(CircuitValidationError, match="already occupied resonator"):
                validate_circuit_moves(
                    sample_move_architecture,
                    invalid_sandwich_circuit,
                    validate_moves=validate_moves,
                )
        else:
            validate_circuit_moves(
                sample_move_architecture,
                invalid_sandwich_circuit,
                validate_moves=validate_moves,
            )

    @pytest.mark.parametrize("validate_moves", list(MoveGateValidationMode))
    def test_bad_move_qubit_already_moved(self, sample_move_architecture, validate_moves):
        """Moving the state of a qubit which is already moved to another resonator."""
        move = CircuitOperation(name="move", locus=("QB3", "CR1"), args={})
        invalid_sandwich_circuit = Circuit(
            name="Move validation circuit",
            instructions=(
                move,
                CircuitOperation(name="move", locus=("QB3", "CR2"), args={}),
            ),  # this MOVE locus is not in the architecture, but only checking MOVE validation
        )
        if validate_moves != MoveGateValidationMode.NONE:
            with pytest.raises(CircuitValidationError, match="is in another resonator"):
                validate_circuit_moves(
                    sample_move_architecture,
                    invalid_sandwich_circuit,
                    validate_moves=validate_moves,
                )
        else:
            validate_circuit_moves(
                sample_move_architecture,
                invalid_sandwich_circuit,
                validate_moves=validate_moves,
            )

    @pytest.mark.parametrize("validation_mode", list(MoveGateValidationMode))
    @pytest.mark.parametrize(
        "gate, allowed_modes, disallowed_modes",
        [
            (
                CircuitOperation(name="prx", locus=("QB3",), args={"phase": 0.6 * pi, "angle": -0.4 * pi}),
                (MoveGateValidationMode.ALLOW_PRX, MoveGateValidationMode.NONE),
                (MoveGateValidationMode.STRICT,),
            ),
            (
                CircuitOperation(name="cz", locus=("QB2", "CR1"), args={}),
                (MoveGateValidationMode.STRICT, MoveGateValidationMode.ALLOW_PRX, MoveGateValidationMode.NONE),
                (),
            ),
        ],
    )
    def test_gates_in_move_sandwich(
        self, sample_move_architecture, validation_mode, gate, allowed_modes, disallowed_modes
    ):
        """Only some gates can be applied on the qubit or resonator inside a MOVE sandwich."""
        move = CircuitOperation(name="move", locus=("QB3", "CR1"), args={})
        instructions = (move, gate, move)
        if validation_mode in disallowed_modes:
            with pytest.raises(CircuitValidationError, match=r"while the state\(s\) of (.+) are in a resonator"):
                TestMoveValidation.make_circuit_and_check(
                    instructions,
                    sample_move_architecture,
                    validation_mode,
                )
        elif validation_mode in allowed_modes:
            TestMoveValidation.make_circuit_and_check(
                instructions,
                sample_move_architecture,
                validation_mode,
            )
        else:
            raise ValueError(f"Unexpected validation mode: {validation_mode}")

    @pytest.mark.parametrize("validation_mode", list(MoveGateValidationMode))
    def test_device_without_resonator(self, sample_dynamic_architecture, sample_circuit, validation_mode):
        """MOVEs cannot be used on a device that does not support them."""
        move = CircuitOperation(name="move", locus=("QB3", "CR1"), args={})
        with pytest.raises(CircuitValidationError, match="'move' is not supported"):
            TestMoveValidation.make_circuit_and_check((move,), sample_dynamic_architecture, validation_mode)
        # But validation passes if there are no MOVE gates
        TestMoveValidation.make_circuit_and_check(
            sample_circuit.instructions, sample_dynamic_architecture, validation_mode
        )

    @pytest.mark.parametrize("validation_mode", list(MoveGateValidationMode))
    def test_qubit_mapping(self, sample_move_architecture, validation_mode):
        """Test that MOVE circuit validation works with an explicit qubit mapping given."""
        move = CircuitOperation(name="move", locus=tuple(reverse_qb_mapping[qb] for qb in ["QB3", "CR1"]), args={})
        prx = CircuitOperation(
            name="prx",
            locus=tuple(reverse_qb_mapping[qb] for qb in ["QB3"]),
            args={"phase": 0.6 * pi, "angle": -0.4 * pi},
        )
        cz = CircuitOperation(name="cz", locus=tuple(reverse_qb_mapping[qb] for qb in ["QB2", "CR1"]), args={})
        TestMoveValidation.make_circuit_and_check(
            (move, move), sample_move_architecture, validation_mode, sample_qb_mapping
        )
        TestMoveValidation.make_circuit_and_check(
            (prx, move, cz, move), sample_move_architecture, validation_mode, sample_qb_mapping
        )
        # qubit mapping without all qubits/resonators in the architecture
        partial_qb_mapping = {k: v for k, v in sample_qb_mapping.items() if v in ["QB2", "QB3", "CR1"]}
        TestMoveValidation.make_circuit_and_check(
            (move, move), sample_move_architecture, validation_mode, partial_qb_mapping
        )
        TestMoveValidation.make_circuit_and_check(
            (prx, move, cz, move), sample_move_architecture, validation_mode, partial_qb_mapping
        )
