# Copyright 2024-2025 IQM client developers
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
from collections import Counter
import re
from typing import Optional
from uuid import UUID

import pytest

from iqm.iqm_client import (
    Circuit,
    CircuitTranspilationError,
    DynamicQuantumArchitecture,
    ExistingMoveHandlingOptions,
    GateImplementationInfo,
    GateInfo,
    Instruction,
    IQMClient,
    simplify_architecture,
    transpile_insert_moves,
    transpile_remove_moves,
)
from iqm.iqm_client.transpile import _ResonatorStateTracker as ResonatorStateTracker

I = Instruction  # shorthand


class MoveTranspilerBase:
    """Base class for transpiler tests, containing some utility methods."""

    arch: DynamicQuantumArchitecture

    def insert(
        self,
        circuit: Circuit,
        existing_moves: Optional[ExistingMoveHandlingOptions] = None,
        qb_map: Optional[dict[str, str]] = None,
        restore_states: bool = True,
    ):
        """Call transpile_insert_moves on the given circuit."""
        kwargs = {}
        if existing_moves is not None:
            kwargs['existing_moves'] = existing_moves
        return transpile_insert_moves(
            circuit,
            arch=self.arch,
            qubit_mapping=qb_map,
            restore_states=restore_states,
            **kwargs,
        )

    def check_equiv_without_moves(self, c1: Circuit, c2: Circuit) -> bool:
        """After removing MOVEs, True iff c1 and c2 are equivalent.
        Symmetric gates may have been flipped.
        """
        c1 = transpile_remove_moves(c1)
        c2 = transpile_remove_moves(c2)
        for i1, i2 in zip(c1.instructions, c2.instructions):
            if i1.name != i2.name or i1.args != i2.args:
                return False
            if i1.qubits != i2.qubits:
                if i1.name != 'cz':
                    return False
                if i1.qubits != i2.qubits[::-1]:
                    return False
        return True

    def assert_valid_circuit(self, circuit: Circuit, qb_map=None) -> None:
        """Raises an error if circuit is not valid."""
        if qb_map:
            for q in self.arch.qubits:
                if q not in qb_map.values():
                    qb_map[q] = q
        IQMClient._validate_circuit_instructions(
            self.arch, [circuit], qubit_mapping=qb_map, must_close_sandwiches=False
        )

    def check_moves_in_circuit(self, circuit: Circuit, moves: tuple[Instruction]) -> bool:
        """True iff ``moves`` all appear in ``circuit`` in that order."""
        idx = 0
        for instr in circuit.instructions:
            if idx < len(moves) and moves[idx] == instr:
                idx += 1
        return idx == len(moves)


class TestMoveTranspilerHybrid(MoveTranspilerBase):
    """A more complicated hybrid quantum architecture, involving both real q-r and real q-q loci."""

    @pytest.fixture(autouse=True)
    def init_arch(self, hybrid_move_architecture):
        # pylint: disable=attribute-defined-outside-init
        self.arch: DynamicQuantumArchitecture = hybrid_move_architecture

    @pytest.mark.parametrize('handling_option', ExistingMoveHandlingOptions)
    def test_normal_usage_no_moves_added(self, handling_option):
        """No MOVE insertion required."""

        circuit = Circuit(
            name='test',
            instructions=(
                I(name='prx', qubits=('QB1',), args={'phase_t': 0.3, 'angle_t': -0.2}),
                I(name='cz', qubits=('QB1', 'CR1'), args={}),
                I(name='cz', qubits=('QB3', 'CR1'), args={}),
                I(name='cz', qubits=('QB3', 'QB4'), args={}),
            ),
        )
        c1 = self.insert(circuit, handling_option)
        self.assert_valid_circuit(c1)
        assert self.check_equiv_without_moves(c1, circuit)

    @pytest.mark.parametrize('handling_option', ExistingMoveHandlingOptions)
    def test_normal_usage(self, handling_option):
        """Tests basic usage of the transpile method"""

        circuit = Circuit(
            name='test',
            instructions=(
                I(name='prx', qubits=('QB1',), args={'phase_t': 0.3, 'angle_t': -0.2}),
                I(name='cz', qubits=('QB1', 'QB2'), args={}),
                I(name='cz', qubits=('QB3', 'QB2'), args={}),
                I(name='cz', qubits=('QB3', 'QB4'), args={}),
            ),
        )
        c1 = self.insert(circuit, handling_option)
        self.assert_valid_circuit(c1)
        assert self.check_equiv_without_moves(c1, circuit)

    @pytest.mark.parametrize('handling_option', ExistingMoveHandlingOptions)
    def test_close_sandwich_for_cz(self, handling_option):
        """Tests basic usage of the transpile method"""

        # all: cz is invalid because QB3 is not reset, so transpiler must reset it
        circuit = Circuit(
            name='test',
            instructions=(
                I(name='move', qubits=('QB3', 'CR1'), args={}),
                I(name='cz', qubits=('QB3', 'CR2'), args={}),
            ),
        )
        if handling_option == ExistingMoveHandlingOptions.KEEP:
            with pytest.raises(
                CircuitTranspilationError,
                match=re.escape("cz acts on ('QB3', 'CR2') while the state(s) of {'QB3'} are"),
            ):
                self.insert(circuit, handling_option)
        else:
            c1 = self.insert(circuit, handling_option)
            self.assert_valid_circuit(c1)
            assert self.check_equiv_without_moves(c1, circuit)

    @pytest.mark.parametrize('handling_option', ExistingMoveHandlingOptions)
    def test_close_sandwich_for_move(self, handling_option):
        """MOVE sandwiches are automatically closed when a new one needs to start."""
        circuit = Circuit(
            name='test',
            instructions=(
                # without this prx transpiler_remove_moves would leave an empty, invalid circuit
                I(name='prx', qubits=('QB2',), args={'phase_t': 0.3, 'angle_t': -0.2}),
                I(name='move', qubits=('QB2', 'CR1'), args={}),  # opens a sandwich
                I(name='move', qubits=('QB2', 'CR2'), args={}),  # opens a new sandwich on the same qubit
            ),
        )
        if handling_option == ExistingMoveHandlingOptions.KEEP:
            with pytest.raises(
                CircuitTranspilationError,
                match=re.escape("MOVE instruction ('QB2', 'CR2'): state of QB2 is in another"),
            ):
                self.insert(circuit, handling_option)
        else:
            c1 = self.insert(circuit, handling_option)
            self.assert_valid_circuit(c1)
            assert self.check_equiv_without_moves(c1, circuit)

    @pytest.mark.parametrize('handling_option', ExistingMoveHandlingOptions)
    def test_heuristic_reuse_moved_state(self, handling_option):
        """Heuristic chooses the optimal MOVE locus with multiple resonators."""
        circuit = Circuit(
            name='test',
            instructions=(
                I(name='cz', qubits=('QB3', 'QB2'), args={}),  # can happen via moving QB2 to either CR
                I(name='cz', qubits=('QB5', 'QB2'), args={}),  # requires QB2 state in CR2
            ),
        )
        c1 = self.insert(circuit, handling_option)
        self.assert_valid_circuit(c1)
        assert self.check_equiv_without_moves(c1, circuit)
        assert len(c1.instructions) == 4  # prx(QB2), move(QB2, CR2), cz(QB3, CR2), cz(QB5, CR2)

    @pytest.mark.parametrize('handling_option', ExistingMoveHandlingOptions)
    def test_heuristic_move_correct_qubit(self, handling_option):
        """Heuristic chooses the optimal MOVE locus with multiple resonators."""
        circuit = Circuit(
            name='test',
            instructions=(
                I(name='cz', qubits=('QB3', 'QB5'), args={}),  # can happen via moving either QB3 or QB5 to CR2
                I(name='cz', qubits=('QB3', 'QB1'), args={}),  # requires moving QB3 to CR1
            ),
        )
        c1 = self.insert(circuit, handling_option, restore_states=False)
        self.assert_valid_circuit(c1)
        assert self.check_equiv_without_moves(c1, circuit)
        assert len(c1.instructions) == 4  # move(QB5, CR2), cz(QB3, CR2), move(QB3, CR1), cz(QB1, CR1)

    def test_simplify_architecture_with_insert_moves(self):
        """Conversions between simplified architecture circuits and corresponding full arch circuits."""

        simple_arch = simplify_architecture(self.arch)
        qc_simple = Circuit(
            name='simple',
            instructions=(
                tuple(I(name='prx', qubits=(q,), args={'phase_t': 0.3, 'angle_t': -0.2}) for q in self.arch.qubits)
                + (
                    # fictional cz is available both ways, since it is symmetric
                    I(name='cz', qubits=('QB1', 'QB2'), args={}),
                    I(name='cz', qubits=('QB2', 'QB1'), args={}),
                    I(name='cz', qubits=('QB3', 'QB2'), args={}),
                    I(name='cz', qubits=('QB3', 'QB4'), args={}),
                    I(name='cz', qubits=('QB3', 'QB5'), args={}),
                )
            ),
        )
        IQMClient._validate_circuit_instructions(simple_arch, [qc_simple])

        qc_with_moves = transpile_insert_moves(qc_simple, self.arch)
        IQMClient._validate_circuit_instructions(self.arch, [qc_with_moves])

        qc = transpile_remove_moves(qc_with_moves)
        IQMClient._validate_circuit_instructions(simple_arch, [qc])


class TestMoveTranspiler(MoveTranspilerBase):
    # pylint: disable=too-many-public-methods

    @pytest.fixture(autouse=True)
    def init_arch(self, sample_move_architecture):
        # pylint: disable=attribute-defined-outside-init
        self.arch: DynamicQuantumArchitecture = sample_move_architecture

    @pytest.fixture
    def unsafe_circuit(self):
        """A circuit with a prx in the middle of a MOVE sandwich."""
        instructions = (
            I(
                name='prx',
                qubits=('QB1',),
                args={'phase_t': 0.3, 'angle_t': -0.2},
            ),
            I(
                name='move',
                qubits=('QB3', 'CR1'),
                args={},
            ),
            I(
                name='prx',
                qubits=('QB3',),
                args={'phase_t': 0.3, 'angle_t': -0.2},
            ),
            I(
                name='move',
                qubits=('QB3', 'CR1'),
                args={},
            ),
        )
        return Circuit(name='unsafe', instructions=instructions)

    @pytest.fixture
    def safe_circuit(self):
        """A partially transpiled circuit."""
        instructions = (
            I(
                name='prx',
                qubits=('QB1',),
                args={'phase_t': 0.3, 'angle_t': -0.2},
            ),
            I(
                name='cz',
                qubits=('QB1', 'CR1'),
                args={},
            ),
            I(
                name='move',
                qubits=('QB3', 'CR1'),
                args={},
            ),
            I(
                name='cz',
                qubits=('QB2', 'CR1'),
                args={},
            ),
            I(
                name='move',
                qubits=('QB3', 'CR1'),
                args={},
            ),
            I(
                name='cz',
                qubits=('QB3', 'QB1'),
                args={},
            ),
        )
        return Circuit(name='safe', instructions=instructions)

    @pytest.fixture
    def simple_circuit(self):
        """An untranspiled circuit in the simplified architecture."""
        instructions = (
            I(
                name='prx',
                qubits=('QB1',),
                args={'phase_t': 0.3, 'angle_t': -0.2},
            ),
            I(
                name='cz',
                qubits=('QB1', 'CR1'),
                args={},
            ),
            I(
                name='cz',
                qubits=('QB2', 'CR1'),
                args={},
            ),
            I(
                name='cz',
                qubits=('QB3', 'QB1'),
                args={},
            ),
            I(
                name='prx',
                qubits=('QB3',),
                args={'phase_t': 0.3, 'angle_t': -0.2},
            ),
        )
        return Circuit(name='safe', instructions=instructions)

    @pytest.fixture
    def ambiguous_circuit(self):
        """A circuit that is unclear how to compile it because there is only one move"""
        instructions = (
            I(
                name='prx',
                qubits=('QB1',),
                args={'phase_t': 0.3, 'angle_t': -0.2},
            ),
            I(
                name='cz',
                qubits=('QB1', 'CR1'),
                args={},
            ),
            I(
                name='move',
                qubits=('QB3', 'CR1'),
                args={},
            ),
            I(
                name='cz',
                qubits=('QB2', 'CR1'),
                args={},
            ),
            I(
                name='cz',
                qubits=('QB3', 'QB1'),
                args={},
            ),
        )
        return Circuit(name='ambiguous', instructions=instructions)

    @pytest.mark.parametrize('handling_option', ExistingMoveHandlingOptions)
    def test_simple_architecture_simple_circuit(self, sample_dynamic_architecture, handling_option, simple_circuit):
        """Tests transpiler for architectures without a resonator"""
        c1 = transpile_insert_moves(simple_circuit, sample_dynamic_architecture, existing_moves=handling_option)
        # no changes
        assert c1 == simple_circuit

    @pytest.mark.parametrize('handling_option', ExistingMoveHandlingOptions)
    def test_simple_architecture_moves_in_circuit(self, sample_dynamic_architecture, handling_option, safe_circuit):
        """MOVEs in the circuit cause an error if architecture does not support them."""
        with pytest.raises(
            ValueError,
            match='Circuit contains MOVE instructions, but the architecture does not support them',
        ):
            transpile_insert_moves(safe_circuit, sample_dynamic_architecture, existing_moves=handling_option)

    @pytest.mark.parametrize('circuit', ['simple_circuit', 'safe_circuit'])
    def test_no_handling_option(self, circuit, request):
        """Tests transpiler in case the handling option is not specified."""
        circuit = request.getfixturevalue(circuit)
        c1 = self.insert(circuit)
        self.assert_valid_circuit(c1)
        assert self.check_equiv_without_moves(c1, circuit)

    @pytest.mark.parametrize('handling_option', ExistingMoveHandlingOptions)
    def test_normal_usage(self, simple_circuit, handling_option):
        """Tests basic usage of the transpile method"""
        c1 = self.insert(simple_circuit, handling_option)
        self.assert_valid_circuit(c1)
        assert self.check_equiv_without_moves(c1, simple_circuit)

    @pytest.mark.parametrize(
        'circuit,error',
        [
            ('safe_circuit', None),
            ('unsafe_circuit', "prx acts on ('QB3',) while the state(s) of {'QB3'} are in a resonator"),
            ('ambiguous_circuit', "cz acts on ('QB3', 'QB1') while the state(s) of {'QB3'} are in a resonator"),
        ],
    )
    def test_keep(self, circuit, error, request):
        """Tests special cases for the KEEP option"""
        c = request.getfixturevalue(circuit)
        moves = tuple(i for i in c.instructions if i.name == 'move')
        if error:
            with pytest.raises(
                CircuitTranspilationError,
                match=re.escape(error),
            ):
                self.insert(c, ExistingMoveHandlingOptions.KEEP)
        else:
            c1 = self.insert(c, ExistingMoveHandlingOptions.KEEP)
            self.assert_valid_circuit(c1)
            assert self.check_moves_in_circuit(c1, moves)

    @pytest.mark.parametrize('circuit', ['safe_circuit', 'unsafe_circuit', 'ambiguous_circuit'])
    def test_remove_moves(self, circuit, request):
        """Tests if removing MOVEs works as intended."""
        c = request.getfixturevalue(circuit)
        moves = tuple(i for i in c.instructions if i.name == 'move')
        c1 = transpile_remove_moves(c)
        assert not self.check_moves_in_circuit(c1, moves)
        c1_with = self.insert(c1, ExistingMoveHandlingOptions.REMOVE)
        c1_direct = self.insert(c, ExistingMoveHandlingOptions.REMOVE)
        assert c1_with == c1_direct
        assert self.check_equiv_without_moves(c1, c1_with)
        assert self.check_equiv_without_moves(c1, c1_direct)

    @pytest.mark.parametrize('circuit', ['safe_circuit', 'unsafe_circuit', 'ambiguous_circuit'])
    def test_trust(self, circuit, request):
        """Tests if the TRUST option works as intended."""
        # Unsafe PRX is made safe since insert adds a MOVE that brings the qubit state back
        # before the PRX is applied. If you want to use unsafe PRXs, do not use transpile_insert_moves.
        # ambiguous MOVE sandwich is automatically closed
        c = request.getfixturevalue(circuit)
        moves = tuple(i for i in c.instructions if i.name == 'move')
        c1 = self.insert(c, ExistingMoveHandlingOptions.TRUST)
        self.assert_valid_circuit(c1)
        assert self.check_moves_in_circuit(c1, moves)

    @pytest.mark.parametrize('handling_option', ExistingMoveHandlingOptions)
    @pytest.mark.parametrize('n', range(2, 6))
    def test_star_ghz(self, handling_option, n: int):
        """Test that the Star transpiler produces an optimal n-qubit GHZ circuit."""
        # all gates available for all loci
        qubits = [f'QB{k+1}' for k in range(n)]
        q_loci = tuple((q,) for q in qubits)
        qr_loci = tuple((q, 'CR') for q in qubits)
        arch = DynamicQuantumArchitecture(
            calibration_set_id=UUID('26c5e70f-bea0-43af-bd37-6212ec7d04cb'),
            qubits=qubits,
            computational_resonators=['CR'],
            gates={
                'prx': GateInfo(
                    implementations={'drag_gaussian': GateImplementationInfo(loci=q_loci)},
                    default_implementation='drag_gaussian',
                    override_default_implementation={},
                ),
                'cz': GateInfo(
                    implementations={'tgss': GateImplementationInfo(loci=qr_loci)},
                    default_implementation='tgss',
                    override_default_implementation={},
                ),
                'move': GateInfo(
                    implementations={'tgss_crf': GateImplementationInfo(loci=qr_loci)},
                    default_implementation='tgss_crf',
                    override_default_implementation={},
                ),
                'measure': GateInfo(
                    implementations={'constant': GateImplementationInfo(loci=q_loci)},
                    default_implementation='constant',
                    override_default_implementation={},
                ),
            },
        )

        instructions = [I(name='prx', qubits=(qubits[0],), args={'phase_t': 0.25, 'angle_t': 0.0})]
        for q in qubits[1:]:
            instructions.extend(
                [
                    I(name='prx', qubits=(q,), args={'phase_t': 0.25, 'angle_t': 0.0}),
                    I(name='cz', qubits=(qubits[0], q), args={}),
                    I(name='prx', qubits=(q,), args={'phase_t': -0.25, 'angle_t': 0.0}),
                ]
            )
        circuit = Circuit(
            name='simplified',
            instructions=instructions,
        )
        c = transpile_insert_moves(
            circuit,
            arch=arch,
            existing_moves=handling_option,
        )
        counter = Counter([inst.name for inst in c.instructions])
        # optimal circuit is a cz comb where QB1 is moved into CR, and then moved back at the end
        assert counter['prx'] == 2 * n - 1
        assert counter['move'] == 2
        assert counter['cz'] == n - 1

    @pytest.mark.parametrize('handling_option', ExistingMoveHandlingOptions)
    def test_with_qubit_map(self, handling_option):
        """Test if qubit mapping works as intended"""

        circuit = Circuit(
            name='mapped',
            instructions=(
                I(
                    name='prx',
                    qubits=('A',),
                    args={'phase_t': 0.3, 'angle_t': -0.2},
                ),
                I(
                    name='cz',
                    qubits=('A', 'B'),
                    args={},
                ),
            ),
        )
        qb_map = {'A': 'QB3', 'B': 'QB1'}

        c1 = self.insert(circuit, handling_option, qb_map)
        self.assert_valid_circuit(c1, qb_map)
        assert self.check_equiv_without_moves(c1, circuit)

    def test_multiple_resonators(self, sample_move_architecture):
        """Test if multiple resonators works."""
        # add MOVE loci to the architecture
        default_move_impl = sample_move_architecture.gates['move'].default_implementation
        sample_move_architecture.gates['move'].implementations[default_move_impl].loci += (('QB1', 'CR2'),)

        # Test with bad architecture
        circuit = Circuit(
            name='multi resonators',
            instructions=(
                I(name='cz', qubits=('QB1', 'QB2'), args={}),
                I(name='cz', qubits=('QB2', 'QB3'), args={}),
                I(name='cz', qubits=('QB1', 'QB3'), args={}),
            ),
        )
        bad_architecture = sample_move_architecture.model_copy(deep=True)
        with pytest.raises(
            CircuitTranspilationError,
            match=re.escape("Unable to find native gate sequence to enable fictional gate cz at ('QB1', 'QB2')"),
        ):
            # CZ(QB1, QB2) is not possible
            # Create a new copy of the DQA to ensure the cached properties are computed only for this architecture.
            transpiled_circuit = transpile_insert_moves(circuit, bad_architecture)

        # Add the CZ loci to the architecture make it ok for this circuit.
        default_cz_impl = sample_move_architecture.gates['cz'].default_implementation
        sample_move_architecture.gates['cz'].implementations[default_cz_impl].loci += tuple(
            (qb, 'CR2') for qb in sample_move_architecture.qubits
        )

        # Create a new copy of the DQA to ensure the cached properties are computed only for this architecture.
        good_architecture = sample_move_architecture.model_copy(deep=True)
        transpiled_circuit = transpile_insert_moves(circuit, good_architecture)
        IQMClient._validate_circuit_instructions(good_architecture, [transpiled_circuit])
        assert self.check_equiv_without_moves(circuit, transpiled_circuit)

    def test_circuit_on_nonexisting_qubits(self):
        """Test for a broken circuit on a non-existing qubit in the architecture."""
        c = Circuit(
            name='QB5 does not exist',
            instructions=(
                I(
                    name='prx',
                    qubits=('QB5',),
                    args={'phase_t': 0.3, 'angle_t': -0.2},
                ),
            ),
        )
        with pytest.raises(CircuitTranspilationError, match=re.escape("('QB5',) is not allowed as locus for 'prx'")):
            self.insert(c, qb_map={'QB5': 'QB5'})

    @pytest.mark.parametrize(
        'circuit',
        [
            Circuit(
                name='bell',
                instructions=(  # prx uses wrong values for the H gate, but that's not the point of this test
                    I(
                        name='prx',
                        qubits=('QB1',),
                        args={'phase_t': 0.3, 'angle_t': -0.2},
                    ),
                    I(
                        name='prx',
                        qubits=('QB2',),
                        args={'phase_t': 0.3, 'angle_t': -0.2},
                    ),
                    I(
                        name='cz',
                        qubits=('QB1', 'QB2'),
                        args={},
                    ),
                    I(
                        name='prx',
                        qubits=('QB2',),
                        args={'phase_t': 0.3, 'angle_t': -0.2},
                    ),
                ),
            ),
            Circuit(
                name='bell',
                instructions=(  # prx uses wrong values for the H gate, but that's not the point of this test
                    I(
                        name='prx',
                        qubits=('QB1',),
                        args={'phase_t': 0.3, 'angle_t': -0.2},
                    ),
                    I(
                        name='prx',
                        qubits=('QB2',),
                        args={'phase_t': 0.3, 'angle_t': -0.2},
                    ),
                    I(
                        name='cz',
                        qubits=('QB2', 'QB1'),  # Swapped qubits
                        args={},
                    ),
                    I(
                        name='prx',
                        qubits=('QB2',),
                        args={'phase_t': 0.3, 'angle_t': -0.2},
                    ),
                ),
            ),
        ],
    )
    def test_can_reverse_cz_locus(self, circuit):
        """Circuit requires unavailable CZ locus, but the reversed locus is available in the DQA,
        and CZ is symmetric. This test reproduces the bug COMP-1485."""
        arch = DynamicQuantumArchitecture(
            calibration_set_id=UUID('0c5a5624-2faf-4885-888c-805af891479c'),
            qubits=['QB1', 'QB2'],
            computational_resonators=['CR1'],
            gates={
                'prx': GateInfo(
                    implementations={'drag_gaussian': GateImplementationInfo(loci=(('QB1',), ('QB2',)))},
                    default_implementation='drag_gaussian',
                    override_default_implementation={},
                ),
                'cz': GateInfo(
                    implementations={'tgss': GateImplementationInfo(loci=(('QB2', 'CR1'),))},
                    default_implementation='tgss',
                    override_default_implementation={},
                ),
                'move': GateInfo(
                    implementations={'tgss_crf': GateImplementationInfo(loci=(('QB1', 'CR1'), ('QB2', 'CR1')))},
                    default_implementation='tgss_crf',
                    override_default_implementation={},
                ),
                'measure': GateInfo(
                    implementations={'constant': GateImplementationInfo(loci=(('QB1',), ('QB2',)))},
                    default_implementation='constant',
                    override_default_implementation={},
                ),
            },
        )
        c1 = transpile_insert_moves(circuit, arch=arch)
        IQMClient._validate_circuit_instructions(arch, [c1])

    @pytest.mark.parametrize(
        'locus', [(qb1, qb2) for qb1 in ['QB1', 'QB2', 'QB3'] for qb2 in ['QB1', 'QB2', 'QB3'] if qb1 != qb2]
    )
    def test_pass_always_picks_correct_move_gate(self, locus):
        circuit = Circuit(
            name='test',
            instructions=(I(name='cz', qubits=locus, args={}),),
        )
        if set(locus) == {'QB1', 'QB2'}:
            # There is no MOVE gate available between this pair of qubits
            with pytest.raises(
                CircuitTranspilationError,
                match=re.escape(f'Unable to find native gate sequence to enable fictional gate cz at {locus}'),
            ):
                transpile_insert_moves(circuit, self.arch)
        else:
            transpiled_circuit = transpile_insert_moves(circuit, self.arch)
            IQMClient._validate_circuit_instructions(self.arch, [transpiled_circuit])


class TestResonatorStateTracker:
    def test_apply_move_no_resonators(self, sample_dynamic_architecture):
        # Check handling of an architecture without a resonator
        no_move_status = ResonatorStateTracker.from_dynamic_architecture(sample_dynamic_architecture)
        with pytest.raises(CircuitTranspilationError, match=re.escape("MOVE locus ('QB1', 'QB2') is not allowed")):
            no_move_status.apply_move('QB1', 'QB2')

    def test_apply_move(self, sample_move_architecture):
        # Check handling of an architecture with resonator
        status = ResonatorStateTracker.from_dynamic_architecture(sample_move_architecture)
        status.apply_move('QB3', 'CR1')
        assert status.res_state_owner['CR1'] == 'QB3'
        status.apply_move('QB3', 'CR1')
        assert status.res_state_owner['CR1'] == 'CR1'
        with pytest.raises(CircuitTranspilationError, match=re.escape("MOVE locus ('QB1', 'CR1') is not allowed")):
            status.apply_move('QB1', 'CR1')
        with pytest.raises(CircuitTranspilationError, match=re.escape("MOVE locus ('QB1', 'QB2') is not allowed")):
            status.apply_move('QB1', 'QB2')
        status.res_state_owner['CR1'] = 'QB1'
        with pytest.raises(CircuitTranspilationError, match=re.escape("MOVE locus ('QB3', 'CR1') is not allowed")):
            status.apply_move('QB3', 'CR1')

    def test_create_move_instructions(self, sample_move_architecture):
        default_move_impl = sample_move_architecture.gates['move'].default_implementation
        sample_move_architecture.gates['move'].implementations[default_move_impl].loci += (('QB1', 'CR1'),)
        status = ResonatorStateTracker.from_dynamic_architecture(sample_move_architecture)
        instr = I(name='move', qubits=('QB3', 'CR1'), args={})
        # Check insertion
        gen_instr = tuple(status.create_move_instructions('QB3', 'CR1'))
        assert len(gen_instr) == 1
        assert gen_instr[0] == instr
        assert status.res_state_owner['CR1'] == 'QB3'
        gen_instr = tuple(status.create_move_instructions('QB3', 'CR1'))
        assert len(gen_instr) == 1
        assert gen_instr[0] == instr
        assert status.res_state_owner['CR1'] == 'CR1'
        status.res_state_owner['CR1'] = 'QB1'
        # Check removal
        gen_instr = tuple(status.create_move_instructions('QB3', 'CR1'))
        assert len(gen_instr) == 2
        assert gen_instr[0] == I(name='move', qubits=('QB1', 'CR1'), args={})
        assert gen_instr[1] == instr
        assert status.res_state_owner['CR1'] == 'QB3'

    def test_reset_as_move_instructions(self, sample_move_architecture):
        status = ResonatorStateTracker.from_dynamic_architecture(sample_move_architecture)
        # No reset needed
        gen_instr = tuple(status.reset_as_move_instructions())
        assert len(gen_instr) == 0
        # Reset with argument
        status.apply_move('QB3', 'CR1')
        gen_instr = tuple(status.reset_as_move_instructions(['CR1']))
        assert len(gen_instr) == 1
        assert gen_instr[0] == I(name='move', qubits=('QB3', 'CR1'), args={})
        assert status.res_state_owner['CR1'] == 'CR1'
        # Reset without arguments
        status.apply_move('QB3', 'CR1')
        gen_instr = tuple(status.reset_as_move_instructions())
        assert len(gen_instr) == 1
        assert gen_instr[0] == I(name='move', qubits=('QB3', 'CR1'), args={})
        assert status.res_state_owner['CR1'] == 'CR1'

    def test_qubits_in_resonator(self, sample_move_architecture):
        components = sample_move_architecture.components
        status = ResonatorStateTracker.from_dynamic_architecture(sample_move_architecture)
        assert status.resonators_holding_qubits(components) == []
        status.apply_move('QB3', 'CR1')
        assert status.resonators_holding_qubits(components) == ['CR1']

    def test_map_resonators_in_locus(self, sample_move_architecture):
        components = sample_move_architecture.components
        status = ResonatorStateTracker.from_dynamic_architecture(sample_move_architecture)
        status.apply_move('QB3', 'CR1')
        assert status.map_resonators_in_locus(components) == ('QB3', 'CR2', 'QB1', 'QB2', 'QB3')


def test_simplify_architecture(sample_move_architecture):
    """Resonators and MOVE gates are eliminated, q-r gates are replaced with q-q gates."""
    simple = simplify_architecture(sample_move_architecture)

    assert simple.qubits == sample_move_architecture.qubits
    assert not simple.computational_resonators

    assert len(simple.gates) == 3
    assert 'move' not in simple.gates
    assert simple.gates['measure'].loci == (('QB1',), ('QB2',), ('QB3',))
    assert simple.gates['prx'].loci == (('QB1',), ('QB2',), ('QB3',))
    assert simple.gates['cz'].loci == (
        ('QB1', 'QB3'),
        ('QB2', 'QB3'),
    )


def test_simplify_architecture_hybrid(hybrid_move_architecture):
    """Resonators and MOVE gates are eliminated, q-r gates are replaced with q-q gates."""
    simple = simplify_architecture(hybrid_move_architecture)

    qubit_loci = (('QB1',), ('QB2',), ('QB3',), ('QB4',), ('QB5',))
    assert simple.qubits == hybrid_move_architecture.qubits
    assert not simple.computational_resonators
    assert len(simple.gates) == len(hybrid_move_architecture.gates) - 1
    assert 'move' not in simple.gates

    # default implementations have not changed
    for name, info in simple.gates.items():
        orig_info = hybrid_move_architecture.gates[name]
        assert orig_info.default_implementation == info.default_implementation
        assert orig_info.override_default_implementation == info.override_default_implementation

    # non-ficitional gates retain their implementations
    impls = simple.gates['prx'].implementations
    assert len(impls) == 1
    assert impls['drag_gaussian'].loci == qubit_loci

    impls = simple.gates['measure'].implementations
    assert len(impls) == 1
    assert impls['constant'].loci == qubit_loci

    impls = simple.gates['cz'].implementations
    assert len(impls) == 2
    assert impls['tgss'].loci == (('QB3', 'QB4'),)
    # fictional gates lose their implementation info
    assert set(impls['__fictional'].loci) == {
        ('QB1', 'QB2'),
        ('QB1', 'QB3'),
        ('QB3', 'QB2'),
        ('QB5', 'QB2'),
        ('QB5', 'QB2'),
        ('QB5', 'QB3'),
        ('QB3', 'QB5'),
    }


@pytest.mark.parametrize('locus', [('QB1', 'QB2'), ('CR1', 'CR2'), ('CR1', 'QB1')])
def test_simplify_architecture_bad_move_locus(locus):
    """MOVE gate with a locus that isn't (qubit, resonator)."""
    dqa = DynamicQuantumArchitecture(
        calibration_set_id=UUID('26c5e70f-bea0-43af-bd37-6212ec7d04cb'),
        qubits=['QB1', 'QB2'],
        computational_resonators=['CR1', 'CR2'],
        gates={
            'move': GateInfo(
                implementations={'tgss_crf': GateImplementationInfo(loci=(locus,))},
                default_implementation='tgss_crf',
                override_default_implementation={},
            ),
        },
    )
    with pytest.raises(ValueError, match=re.escape(f'MOVE locus {locus} is not of the form')):
        simplify_architecture(dqa)


def test_simplify_architecture_no_resonators(sample_dynamic_architecture):
    """Architectures with no resonators are not changed."""
    simple = simplify_architecture(sample_dynamic_architecture)
    assert simple == sample_dynamic_architecture
