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
import pytest

from iqm.iqm_client import QuantumArchitectureSpecification


@pytest.mark.parametrize(
    'operations',
    [
        {'prx': [['QB1'], ['QB2'], ['QB3']]},
        {'prx': [['QB2'], ['QB1'], ['QB3']]},
        {'prx': [['QB3'], ['QB1'], ['QB2']]},
    ],
)
def test_equivalent_prx_ops(operations):
    verify_ops_match({'prx': [['QB1'], ['QB2'], ['QB3']]}, operations, True)


@pytest.mark.parametrize(
    'operations',
    [
        {'prx': [['QB2'], ['QB2'], ['QB3']]},
        {'prx': [['QB2'], ['QB3']]},
        {'prx': [['QB3'], ['QB1'], ['QB2'], ['QB4']]},
        {'prx': [['QB1'], ['QB2'], ['QB3']], 'move': [['QB1', 'QB2']]},
    ],
)
def test_different_prx_ops(operations):
    verify_ops_match({'prx': [['QB1'], ['QB2'], ['QB3']]}, operations, False)


@pytest.mark.parametrize(
    'operations',
    [
        {'prx': [['QB1'], ['QB2'], ['QB3']], 'move': [['QB1', 'QB2']]},
        {'prx': [['QB2'], ['QB1'], ['QB3']], 'move': [['QB1', 'QB2']]},
        {'prx': [['QB3'], ['QB1'], ['QB2']], 'move': [['QB1', 'QB2']]},
    ],
)
def test_equivalent_prx_move_ops(operations):
    verify_ops_match({'prx': [['QB1'], ['QB2'], ['QB3']], 'move': [['QB1', 'QB2']]}, operations, True)


@pytest.mark.parametrize(
    'operations',
    [
        {'prx': [['QB1'], ['QB2'], ['QB3']], 'move': [['QB2', 'QB1']]},
        {'prx': [['QB2'], ['QB1']], 'move': [['QB1', 'QB2']]},
        {'prx': [['QB3'], ['QB1'], ['QB2']], 'move': [['QB2', 'QB3']]},
        {'prx': [['QB3'], ['QB1'], ['QB2']], 'move': [['QB1', 'QB2'], ['QB2', 'QB3']]},
    ],
)
def test_different_prx_move_ops(operations):
    verify_ops_match({'prx': [['QB1'], ['QB2'], ['QB3']], 'move': [['QB1', 'QB2']]}, operations, False)


@pytest.mark.parametrize(
    'operations',
    [
        {'prx': [['QB1'], ['QB2'], ['QB3']], 'cz': [['QB2', 'QB1']]},
        {'prx': [['QB2'], ['QB1'], ['QB3']], 'cz': [['QB2', 'QB1']]},
        {'prx': [['QB3'], ['QB1'], ['QB2']], 'cz': [['QB1', 'QB2']]},
    ],
)
def test_equivalent_prx_cz_ops(operations):
    verify_ops_match({'prx': [['QB1'], ['QB2'], ['QB3']], 'cz': [['QB1', 'QB2']]}, operations, True)


def verify_ops_match(ops1, ops2, should_match: bool):
    assert QuantumArchitectureSpecification.compare_operations(ops1=ops1, ops2=ops2) == should_match
    assert QuantumArchitectureSpecification.compare_operations(ops1=ops2, ops2=ops1) == should_match
    arch1 = to_arch(ops1)
    arch2 = to_arch(ops2)
    assert arch1.has_equivalent_operations(arch2) == should_match
    assert arch2.has_equivalent_operations(arch1) == should_match


def to_arch(operations):
    return QuantumArchitectureSpecification(
        name='hercules',
        qubits=['QB1', 'QB2', 'QB3'],
        qubit_connectivity=[['QB1', 'QB2'], ['QB2', 'QB3']],
        operations=operations,
    )


def test_simplified_quantum_architecture():
    simplified_architecture = QuantumArchitectureSpecification(
        name='hercules',
        qubits=['QB1', 'QB2', 'QB3'],
        qubit_connectivity=[['QB1', 'QB2'], ['QB2', 'QB3']],
        operations=['prx', 'cz'],
    )
    assert simplified_architecture.operations == {
        'prx': [['QB1'], ['QB2'], ['QB3']],
        'cz': [['QB1', 'QB2'], ['QB2', 'QB3']],
    }
