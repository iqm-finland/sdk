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
import pytest

from iqm.iqm_client import (
    Circuit,
    CircuitValidationError,
    DynamicQuantumArchitecture,
    Instruction,
    IQMClient,
    MoveGateValidationMode,
)

sample_qb_mapping = {'0': 'CR1', '1': 'QB1', '2': 'QB2', '3': 'QB3', '100': 'CR2'}
reverse_qb_mapping = {value: key for key, value in sample_qb_mapping.items()}


@pytest.mark.parametrize(
    'instruction',
    [
        Instruction(name='barrier', qubits=['QB1'], args={}),
        Instruction(name='barrier', qubits=['QB1', 'QB2'], args={}),
        Instruction(name='barrier', qubits=['QB2', 'QB1'], args={}),  # barrier can use any loci
        Instruction(name='delay', qubits=['QB1'], args={'duration': 80e-9}),
        Instruction(name='delay', qubits=['QB1', 'QB2'], args={'duration': 40e-9}),
        Instruction(name='delay', qubits=['QB2', 'QB1'], args={'duration': 100e-9}),  # delay can use any loci
        Instruction(name='prx', qubits=['QB1'], args={'phase_t': 0.3, 'angle_t': -0.2}),
        Instruction(
            name='cc_prx',
            qubits=['QB1'],
            args={'phase_t': 0.3, 'angle_t': -0.2, 'feedback_key': 'f1', 'feedback_qubit': 'QB2'},
        ),
        Instruction(name='reset', qubits=['QB1'], args={}),
        Instruction(name='cz', qubits=['QB1', 'QB2'], args={}),
        Instruction(name='cz', qubits=['QB2', 'QB1'], args={}),  # CZ is symmetric
        Instruction(name='measure', qubits=['QB1'], args={'key': 'm'}),
        Instruction(name='measure', qubits=['QB1', 'QB2'], args={'key': 'm'}),  # measure is factorizable
        Instruction(name='measure', qubits=['QB2', 'QB1'], args={'key': 'm'}),  # measure is factorizable
    ],
)
def test_valid_instruction(sample_dynamic_architecture, instruction):
    """Valid instructions must pass validation."""
    IQMClient._validate_instruction(sample_dynamic_architecture, instruction, None)


@pytest.mark.parametrize(
    'instruction,match',
    [
        [Instruction(name='barrier', qubits=['QB1', 'QB2', 'XXX'], args={}), 'does not exist'],
        [Instruction(name='delay', qubits=['YYY'], args={'duration': 40e-9}), 'does not exist'],
        [
            Instruction(name='prx', qubits=['QB4'], args={'phase_t': 0.3, 'angle_t': -0.2}),
            "not allowed as locus for 'prx'",
        ],
        [Instruction(name='cz', qubits=['QB2', 'QB4'], args={}), "not allowed as locus for 'cz'"],
        [Instruction(name='measure', qubits=['QB1', 'QB4'], args={'key': 'm'}), "not allowed as locus for 'measure'"],
        [Instruction(name='measure', qubits=['QB4'], args={'key': 'm'}), "not allowed as locus for 'measure'"],
        [
            Instruction(name='cz', qubits=['QB1', 'QB2'], args={}, implementation='xyz'),
            "'cz' implementation 'xyz' is not supported",
        ],
        [
            Instruction(name='prx', qubits=['QB2'], args={'phase_t': 0.3, 'angle_t': -0.2}, implementation='drag_crf'),
            "not allowed as locus for 'prx.drag_crf'",
        ],
    ],
)
def test_invalid_instruction(sample_dynamic_architecture, instruction, match):
    """Invalid instructions must not pass validation."""
    with pytest.raises(CircuitValidationError, match=match):
        IQMClient._validate_instruction(sample_dynamic_architecture, instruction, None)


@pytest.mark.parametrize('qubit_mapping', [None, sample_qb_mapping])
@pytest.mark.parametrize('qubits', [['QB1', 'CR1'], ['CR1', 'QB1'], ['CR1', 'QB2']])
def test_allowed_cz_qubits(sample_move_architecture, qubits, qubit_mapping):
    """
    Tests that instruction validation passes for allowed CZ loci
    """
    if qubit_mapping:
        qubits = [reverse_qb_mapping[q] for q in qubits]
    IQMClient._validate_instruction(
        sample_move_architecture,
        Instruction(name='cz', qubits=qubits, args={}),
        qubit_mapping,
    )


@pytest.mark.parametrize('qubit_mapping', [None, sample_qb_mapping])
@pytest.mark.parametrize(
    'qubits', [['QB1', 'QB2'], ['QB2', 'QB1'], ['QB1', 'QB1'], ['QB3', 'QB1'], ['CR1', 'CR1'], ['CR1', 'QB3']]
)
def test_disallowed_cz_qubits(sample_move_architecture, qubits, qubit_mapping):
    """
    Tests that instruction validation fails for loci that are not allowed for CZ by the quantum architecture
    """
    if qubit_mapping:
        qubits = [reverse_qb_mapping[q] for q in qubits]
    with pytest.raises(CircuitValidationError, match="not allowed as locus for 'cz'"):
        IQMClient._validate_instruction(
            sample_move_architecture,
            Instruction(name='cz', qubits=qubits, args={}),
            qubit_mapping,
        )


@pytest.mark.parametrize('qubit_mapping', [None, sample_qb_mapping])
@pytest.mark.parametrize('qubits', [['QB3', 'CR1']])
def test_allowed_move_qubits(sample_move_architecture, qubits, qubit_mapping):
    """
    Tests that instruction validation passes for allowed MOVE loci
    """
    if qubit_mapping:
        qubits = [reverse_qb_mapping[q] for q in qubits]

    IQMClient._validate_instruction(
        sample_move_architecture,
        Instruction(name='move', qubits=qubits, args={}),
        qubit_mapping,
    )


@pytest.mark.parametrize('qubit_mapping', [None, sample_qb_mapping])
@pytest.mark.parametrize(
    'qubits',
    [['QB1', 'QB2'], ['QB2', 'QB1'], ['QB1', 'QB1'], ['QB1', 'CR1'], ['CR1', 'CR1'], ['CR1', 'QB3']],
)
def test_disallowed_move_qubits(sample_move_architecture, qubits, qubit_mapping):
    """
    Tests that instruction validation fails for loci that are not allowed for MOVE by the quantum architecture
    """
    if qubit_mapping:
        qubits = [reverse_qb_mapping[q] for q in qubits]

    with pytest.raises(CircuitValidationError, match="not allowed as locus for 'move'"):
        IQMClient._validate_instruction(
            sample_move_architecture,
            Instruction(name='move', qubits=qubits, args={}),
            qubit_mapping,
        )


@pytest.mark.parametrize('qubits', [['QB1', 'QB2'], ['QB2'], ['QB1', 'QB2', 'QB3'], ['QB3', 'QB1'], ['QB1']])
def test_allowed_measure_qubits(sample_move_architecture, qubits):
    """
    Tests that instruction validation succeeds for loci that are any combination of valid measure qubits
    """
    IQMClient._validate_instruction(
        sample_move_architecture,
        Instruction(name='measure', qubits=qubits, args={'key': 'measure_1'}),
        None,
    )


@pytest.mark.parametrize('qubits', [['QB1', 'CR1'], ['CR1'], ['QB1', 'QB2', 'QB4'], ['QB4']])
def test_disallowed_measure_qubits(sample_move_architecture, qubits):
    """
    Tests that instruction validation fails for loci containing any qubits that are not valid measure qubits
    """
    with pytest.raises(CircuitValidationError, match="is not allowed as locus for 'measure'"):
        IQMClient._validate_instruction(
            sample_move_architecture,
            Instruction(name='measure', qubits=qubits, args={'key': 'measure_1'}),
            None,
        )


def test_measurement_keys_must_be_unique(sample_move_architecture):
    """
    Tests that all measure instructions in a circuit must have unique keys.
    """
    circuit = Circuit(
        name='Test circuit',
        instructions=[
            Instruction(name='measure', qubits=['QB1'], args={'key': 'a'}),
            Instruction(name='measure', qubits=['QB2'], args={'key': 'a'}),
        ],
    )
    with pytest.raises(CircuitValidationError, match='has a non-unique measurement key'):
        IQMClient._validate_circuit_instructions(
            sample_move_architecture,
            [circuit],
        )


def test_same_measurement_key_in_different_circuits(sample_move_architecture):
    """
    Tests that the same measurement key can be used in different circuits.
    """
    circuits = [
        Circuit(
            name='Test circuit 1',
            instructions=[
                Instruction(name='measure', qubits=['QB1'], args={'key': 'a'}),
            ],
        ),
        Circuit(
            name='Test circuit 2',
            instructions=[
                Instruction(name='measure', qubits=['QB1'], args={'key': 'a'}),
            ],
        ),
    ]
    IQMClient._validate_circuit_instructions(
        sample_move_architecture,
        circuits,
    )


@pytest.mark.parametrize(
    'qubits',
    [
        ['CR1', 'QB1', 'QB2', 'QB3'],
        ['QB1', 'CR1', 'QB2', 'QB3'],
        ['QB1', 'CR1', 'QB2'],
    ],
)
def test_barrier(sample_move_architecture, qubits):
    """
    Tests that instruction validation passes for the barrier operation
    """
    IQMClient._validate_instruction(
        sample_move_architecture,
        Instruction(name='barrier', qubits=qubits, args={}),
        None,
    )


class TestMoveValidation:
    """Tests the validation of MOVE instructions."""

    @staticmethod
    def make_circuit_and_check(
        instructions: tuple[Instruction, ...],
        arch: DynamicQuantumArchitecture,
        validate_moves: MoveGateValidationMode,
        qubit_mapping=None,
    ):
        """Validate the given instructions (as a circuit)."""
        circuit = Circuit(name='Move validation circuit', instructions=instructions)
        IQMClient._validate_circuit_instructions(
            arch,
            [circuit],
            qubit_mapping,
            validate_moves=validate_moves,
        )

    @pytest.mark.parametrize('validate_moves', list(MoveGateValidationMode))
    @pytest.mark.parametrize(
        'instructions',
        [
            (Instruction(name='move', qubits=['QB3', 'CR1'], args={}),),
            (Instruction(name='move', qubits=['QB3', 'CR1'], args={}),) * 3,
        ],
    )
    def test_non_sandwich_move(self, sample_move_architecture, validate_moves, instructions):
        """Non-sandwich MOVEs are not allowed."""
        if validate_moves != MoveGateValidationMode.NONE:
            with pytest.raises(CircuitValidationError, match=r'qubit state\(s\) are still in a resonator'):
                TestMoveValidation.make_circuit_and_check(instructions, sample_move_architecture, validate_moves)
        else:
            TestMoveValidation.make_circuit_and_check(instructions, sample_move_architecture, validate_moves)

    @pytest.mark.parametrize('validate_moves', list(MoveGateValidationMode))
    def test_move_sandwich(self, sample_move_architecture, validate_moves):
        """Valid pair of MOVEs."""
        move = Instruction(name='move', qubits=['QB3', 'CR1'], args={})
        TestMoveValidation.make_circuit_and_check((move, move), sample_move_architecture, validate_moves)

    @pytest.mark.parametrize('validate_moves', list(MoveGateValidationMode))
    def test_bad_move_occupied_resonator(self, sample_move_architecture, validate_moves):
        """Moving a qubit state into an occupied resonator."""
        move = Instruction(name='move', qubits=['QB3', 'CR1'], args={})
        invalid_sandwich_circuit = Circuit(
            name='Move validation circuit',
            instructions=(
                move,
                Instruction(name='move', qubits=['QB2', 'CR1'], args={}),
            ),  # this MOVE locus is not in the architecture, but only checking MOVE validation
        )
        if validate_moves != MoveGateValidationMode.NONE:
            with pytest.raises(CircuitValidationError, match='already occupied resonator'):
                IQMClient._validate_circuit_moves(
                    sample_move_architecture,
                    invalid_sandwich_circuit,
                    validate_moves=validate_moves,
                )
        else:
            IQMClient._validate_circuit_moves(
                sample_move_architecture,
                invalid_sandwich_circuit,
                validate_moves=validate_moves,
            )

    @pytest.mark.parametrize('validate_moves', list(MoveGateValidationMode))
    def test_bad_move_qubit_already_moved(self, sample_move_architecture, validate_moves):
        """Moving the state of a qubit which is already moved to another resonator."""
        move = Instruction(name='move', qubits=['QB3', 'CR1'], args={})
        invalid_sandwich_circuit = Circuit(
            name='Move validation circuit',
            instructions=(
                move,
                Instruction(name='move', qubits=['QB3', 'CR2'], args={}),
            ),  # this MOVE locus is not in the architecture, but only checking MOVE validation
        )
        if validate_moves != MoveGateValidationMode.NONE:
            with pytest.raises(CircuitValidationError, match='is in another resonator'):
                IQMClient._validate_circuit_moves(
                    sample_move_architecture,
                    invalid_sandwich_circuit,
                    validate_moves=validate_moves,
                )
        else:
            IQMClient._validate_circuit_moves(
                sample_move_architecture,
                invalid_sandwich_circuit,
                validate_moves=validate_moves,
            )

    @pytest.mark.parametrize('validation_mode', list(MoveGateValidationMode))
    @pytest.mark.parametrize(
        'gate, allowed_modes, disallowed_modes',
        [
            (
                Instruction(name='prx', qubits=['QB3'], args={'phase_t': 0.3, 'angle_t': -0.2}),
                (MoveGateValidationMode.ALLOW_PRX, MoveGateValidationMode.NONE),
                (MoveGateValidationMode.STRICT,),
            ),
            (
                Instruction(name='cz', qubits=['QB2', 'CR1'], args={}),
                (MoveGateValidationMode.STRICT, MoveGateValidationMode.ALLOW_PRX, MoveGateValidationMode.NONE),
                (),
            ),
        ],
    )
    def test_gates_in_move_sandwich(
        self, sample_move_architecture, validation_mode, gate, allowed_modes, disallowed_modes
    ):
        """Only some gates can be applied on the qubit or resonator inside a MOVE sandwich."""
        # pylint: disable=too-many-arguments
        move = Instruction(name='move', qubits=['QB3', 'CR1'], args={})
        instructions = (move, gate, move)
        if validation_mode in disallowed_modes:
            with pytest.raises(CircuitValidationError, match=r'while the state\(s\) of (.+) are in a resonator'):
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
            raise ValueError(f'Unexpected validation mode: {validation_mode}')

    @pytest.mark.parametrize('validation_mode', list(MoveGateValidationMode))
    def test_device_without_resonator(self, sample_dynamic_architecture, sample_circuit, validation_mode):
        """MOVEs cannot be used on a device that does not support them."""
        move = Instruction(name='move', qubits=['QB3', 'CR1'], args={})
        with pytest.raises(CircuitValidationError, match="'move' is not supported"):
            TestMoveValidation.make_circuit_and_check((move,), sample_dynamic_architecture, validation_mode)
        # But validation passes if there are no MOVE gates
        TestMoveValidation.make_circuit_and_check(
            sample_circuit.instructions, sample_dynamic_architecture, validation_mode
        )

    @pytest.mark.parametrize('validation_mode', list(MoveGateValidationMode))
    def test_qubit_mapping(self, sample_move_architecture, validation_mode):
        """Test that MOVE circuit validation works with an explicit qubit mapping given."""
        move = Instruction(name='move', qubits=[reverse_qb_mapping[qb] for qb in ['QB3', 'CR1']], args={})
        prx = Instruction(
            name='prx', qubits=[reverse_qb_mapping[qb] for qb in ['QB3']], args={'phase_t': 0.3, 'angle_t': -0.2}
        )
        cz = Instruction(name='cz', qubits=[reverse_qb_mapping[qb] for qb in ['QB2', 'CR1']], args={})
        TestMoveValidation.make_circuit_and_check(
            (move, move), sample_move_architecture, validation_mode, sample_qb_mapping
        )
        TestMoveValidation.make_circuit_and_check(
            (prx, move, cz, move), sample_move_architecture, validation_mode, sample_qb_mapping
        )
        # qubit mapping without all qubits/resonators in the architecture
        partial_qb_mapping = {k: v for k, v in sample_qb_mapping.items() if v in ['QB2', 'QB3', 'CR1']}
        TestMoveValidation.make_circuit_and_check(
            (move, move), sample_move_architecture, validation_mode, partial_qb_mapping
        )
        TestMoveValidation.make_circuit_and_check(
            (prx, move, cz, move), sample_move_architecture, validation_mode, partial_qb_mapping
        )
