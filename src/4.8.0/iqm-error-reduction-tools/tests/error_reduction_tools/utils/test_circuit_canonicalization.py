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

"""Tests for the single entry-point circuit converter."""

import math
from unittest.mock import MagicMock

from iqm.error_reduction_tools.utils.circuit_utils import TwirledCircuit
from iqm.error_reduction_tools.utils.qiskit_utils import (
    canonicalize_circuit,
    from_qiskit_to_iqm,
    qubit_index_to_name_map,
)
from iqm.qiskit_iqm.fake_backends import IQMFakeApollo
from iqm.qrisp_iqm import transpile_to_iqm
import pytest
from qiskit import QuantumCircuit
from qrisp import QuantumCircuit as QrispQuantumCircuit

from iqm.pulse import Circuit as PulseCircuit
from iqm.pulse import CircuitOperation


def native_qiskit_circuit(num_qubits=2):
    """A circuit already in the IQM native gate set, so serialization accepts it."""
    circuit = QuantumCircuit(num_qubits, num_qubits)
    circuit.r(math.pi / 2, 0.0, 0)
    for qubit in range(num_qubits - 1):
        circuit.cz(qubit, qubit + 1)
    circuit.measure(range(num_qubits), range(num_qubits))
    return circuit


def bell_operations():
    return [
        CircuitOperation(name="prx", locus=("QB1",), args={"angle": math.pi / 2, "phase": 0.0}),
        CircuitOperation(name="cz", locus=("QB1", "QB2"), args={}),
        CircuitOperation(name="measure", locus=("QB1",), args={"key": "m_1_2_0_0"}),
        CircuitOperation(name="measure", locus=("QB2",), args={"key": "m_1_2_1_1"}),
    ]


def test_a_qiskit_circuit_becomes_a_pulse_circuit_with_its_true_index_mapping():
    """Test that a Qiskit circuit becomes a pulse circuit, with the qubit names of its indices."""
    circuit = native_qiskit_circuit(3)

    canonical, mapping = canonicalize_circuit(circuit, name="target")

    assert isinstance(canonical, PulseCircuit)
    assert canonical.name == "target"
    assert mapping == qubit_index_to_name_map(circuit)
    assert mapping == {0: "QB1", 1: "QB2", 2: "QB3"}


def test_canonicalizing_produces_the_same_operations_as_the_twirler_always_did():
    """Test that canonicalization gives the same operations and mapping as from_qiskit_to_iqm."""
    circuit = native_qiskit_circuit(3)

    canonical, mapping = canonicalize_circuit(circuit)
    twirled, legacy_mapping = from_qiskit_to_iqm(circuit)

    assert [operation.name for operation in canonical.instructions] == [
        operation.name for operation in twirled.operations
    ]
    assert [operation.locus for operation in canonical.instructions] == [
        operation.locus for operation in twirled.operations
    ]
    assert mapping == legacy_mapping


def test_a_pulse_circuit_is_returned_untouched():
    """Test that a pulse circuit is returned as it is."""
    circuit = PulseCircuit(name="bell", instructions=tuple(bell_operations()))

    canonical, mapping = canonicalize_circuit(circuit)

    assert canonical is circuit
    assert mapping == {0: "QB1", 1: "QB2"}


def test_a_twirled_circuit_is_converted_and_keeps_its_operations():
    """Test that a TwirledCircuit is converted to a pulse circuit with the same operations."""
    circuit = TwirledCircuit(bell_operations())

    canonical, mapping = canonicalize_circuit(circuit, name="from_twirled")

    assert isinstance(canonical, PulseCircuit)
    assert canonical.name == "from_twirled"
    assert [operation.name for operation in canonical.instructions] == [
        operation.name for operation in circuit.operations
    ]
    assert mapping == {0: "QB1", 1: "QB2"}


def test_canonicalizing_a_randomized_circuit_drops_its_twirling_state():
    """Test that canonicalizing a randomized circuit drops its rot_dict."""
    circuit = TwirledCircuit(bell_operations())
    circuit.rot_dict = {"QB1": "X", "QB2": "I"}

    canonical, _mapping = canonicalize_circuit(circuit)

    assert not hasattr(canonical, "rot_dict")


def test_an_object_exposing_to_qiskit_is_accepted():
    """Test that an object with a to_qiskit() method is accepted."""

    class Wrapped:
        def __init__(self, circuit):
            self._circuit = circuit

        def to_qiskit(self):
            return self._circuit

    canonical, mapping = canonicalize_circuit(Wrapped(native_qiskit_circuit()))

    assert isinstance(canonical, PulseCircuit)
    assert mapping == {0: "QB1", 1: "QB2"}


def _transpiled_qrisp_ghz(num_qubits=3):
    """Return a GHZ circuit in Qrisp, transpiled for a line of qubits."""
    qc = QrispQuantumCircuit(num_qubits)
    qc.h(0)
    for qubit in range(1, num_qubits):
        qc.cx(qubit - 1, qubit)
    qc.measure(list(range(num_qubits)))
    return transpile_to_iqm(qc, [(i, i + 1) for i in range(num_qubits - 1)])


def test_a_qrisp_circuit_is_named_after_the_backend_architecture():
    """Test that Qrisp qubit i becomes the i-th qubit of the architecture, with qiskit-style measurement keys."""
    backend = IQMFakeApollo()
    canonical, mapping = canonicalize_circuit(_transpiled_qrisp_ghz(), name="ghz", backend=backend)
    assert isinstance(canonical, PulseCircuit)
    assert canonical.name == "ghz"
    assert mapping == dict(enumerate(backend.architecture.qubits[:3]))
    keys = sorted(op.args["key"] for op in canonical.instructions if op.name == "measure")
    assert keys == ["c_3_0_0", "c_3_0_1", "c_3_0_2"]


def test_a_qrisp_circuit_without_a_backend_is_converted_through_qiskit():
    """Test that a Qrisp circuit without a backend is converted through to_qiskit()."""
    circuit = _transpiled_qrisp_ghz()
    assert canonicalize_circuit(circuit) == canonicalize_circuit(circuit.to_qiskit())


def test_an_unsupported_type_names_every_accepted_representation():
    """Test that an unsupported input type is rejected."""
    with pytest.raises(TypeError, match="Unsupported circuit type"):
        canonicalize_circuit(42)


def test_a_to_qiskit_returning_a_non_qiskit_object_raises_rather_than_recursing():
    # `to_qiskit()` is called once, never on its own result. Recursing instead would not
    # terminate for a type like this one and would exhaust memory before raising.
    """Test that a to_qiskit() that does not return a Qiskit circuit raises."""

    class Endless:
        def to_qiskit(self):
            return Endless()

    with pytest.raises(TypeError, match=r"Endless\.to_qiskit\(\) returned Endless"):
        canonicalize_circuit(Endless())


def test_a_mock_circuit_raises_rather_than_exhausting_memory():
    # A MagicMock auto-creates `to_qiskit`, and its result auto-creates one too. This is the
    # shape that took a whole test session down with a MemoryError.
    """Test that a mock passed as a circuit raises."""
    with pytest.raises(TypeError, match="not a qiskit QuantumCircuit"):
        canonicalize_circuit(MagicMock())


def test_the_invented_mapping_covers_every_active_qubit():
    """Test that the index mapping of a pulse circuit covers every qubit it acts on."""
    operations = [
        CircuitOperation(name="prx", locus=("QB5",), args={"angle": 0.4, "phase": 0.0}),
        CircuitOperation(name="cz", locus=("QB5", "QB2"), args={}),
        CircuitOperation(name="measure", locus=("QB2",), args={"key": "m_1_1_0_0"}),
    ]

    _canonical, mapping = canonicalize_circuit(PulseCircuit(name="c", instructions=tuple(operations)))

    assert mapping == {0: "QB2", 1: "QB5"}
