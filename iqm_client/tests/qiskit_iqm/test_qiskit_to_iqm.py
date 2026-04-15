# Copyright 2022 Qiskit on IQM developers
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

"""Testing Qiskit to IQM conversion tools."""

from math import pi
import re

from iqm.iqm_client import CircuitValidationError
from iqm.qiskit_iqm.iqm_provider import IQMBackend
from iqm.qiskit_iqm.qiskit_to_iqm import MeasurementKey, deserialize_instructions, serialize_instructions
from mockito import when
from packaging.version import Version
import pytest
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit import __version__ as qiskit_version
from qiskit.circuit.library import CZGate, RGate, RXGate, RYGate, XGate, YGate
from qiskit.transpiler.layout import Layout

from iqm.pulse import Circuit, CircuitOperation

from .conftest import get_mocked_backend
from .utils import get_transpiled_circuit_json

pytestmark = pytest.mark.usefixtures("unstub")


@pytest.fixture
def backend(linear_3q_architecture, request):
    return get_mocked_backend(linear_3q_architecture, request)


@pytest.fixture()
def circuit() -> QuantumCircuit:
    return QuantumCircuit(3, 3)


def test_measurement_key_to_str():
    mk = MeasurementKey("abc", 1, 2, 3)
    assert str(mk) == "abc_1_2_3"


def test_measurement_key_from_clbit():
    qreg = QuantumRegister(3)
    creg1, creg2 = ClassicalRegister(2, name="cr1"), ClassicalRegister(1, name="cr2")

    circuit = QuantumCircuit(qreg, creg1, creg2)
    mk1 = MeasurementKey.from_clbit(creg1[0], circuit)
    mk2 = MeasurementKey.from_clbit(creg1[1], circuit)
    mk3 = MeasurementKey.from_clbit(creg2[0], circuit)
    assert str(mk1) == "cr1_2_0_0"
    assert str(mk2) == "cr1_2_0_1"
    assert str(mk3) == "cr2_1_1_0"


@pytest.mark.parametrize("key_str", ["abc_4_5_6", "a_bc_4_5_6"])
def test_measurement_key_from_string(key_str):
    mk = MeasurementKey.from_string(key_str)
    assert str(mk) == key_str


def test_circuit_to_iqm_json(adonis_architecture, request):
    """Test that a circuit submitted via IQM backend gets transpiled into proper JSON."""
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()

    # The transpilation seed could be expected to make the routing result deterministic,
    # but it seems to vary between Qiskit versions at least.
    submitted_circuit = get_transpiled_circuit_json(circuit, adonis_architecture, seed_transpiler=123, request=request)
    assert isinstance(submitted_circuit, Circuit)

    instr_names = [instr.name for instr in submitted_circuit.instructions]
    # Depending on the Qiskit library version, the gates act on different qubits, so we only check the instruction names
    instr_qubits = {qubit for instr in submitted_circuit.instructions for qubit in instr.locus}
    assert len(instr_qubits) == 2
    assert instr_names == [
        # Hadamard on qubit 0
        "prx",
        # CX: Hadamard on qubit 1
        "prx",
        # CX: CZ on 0,1
        "cz",
        # CX: Hadamard on qubit 1
        "prx",
        # Barrier before measurements
        "barrier",
        # Measurement on both qubits
        "measure",
        "measure",
    ]


def test_serialize_instructions_can_allow_nonnative_gates():
    # Majority of the logic is tested in test_iqm_backend, here we only test the non-default behavior
    nonnative_gate = QuantumCircuit(3, name="nonnative").to_gate()
    circuit = QuantumCircuit(5)
    circuit.append(nonnative_gate, [1, 2, 4])
    circuit.measure_all()
    mapping = {i: f"QB{i + 1}" for i in range(5)}

    with pytest.raises(ValueError, match="is not natively supported. You need to transpile"):
        serialize_instructions(circuit, mapping)

    instructions = serialize_instructions(circuit, mapping, allowed_nonnative_gates={"nonnative"})
    assert instructions[0] == CircuitOperation(name="nonnative", locus=("QB2", "QB3", "QB5"), args={})


def test_deserialize_instructions_empty():
    """Check that default input creates an empty qiskit quantum circuit."""
    circuit = deserialize_instructions([], {}, Layout())
    assert isinstance(circuit, QuantumCircuit)
    assert circuit.num_qubits == 0


def test_deserialize_instructions_without_layout():
    """Check that instructions are parsed for default layout."""
    instructions = [
        CircuitOperation(name="prx", locus=("QB1",), args={"phase": 0.0, "angle": 0.0}),
        CircuitOperation(name="cz", locus=("QB1", "QB2"), args={}),
        CircuitOperation(name="move", locus=("QB1", "CR1"), args={}),
        CircuitOperation(name="barrier", locus=("QB1", "QB2"), args={}),
        CircuitOperation(
            name="measure",
            locus=("QB1",),
            args={"key": "m_3_2_1", "feedback_key": "m_3_2_1"},
        ),
        CircuitOperation(name="delay", locus=("QB1",), args={"duration": 50e-9}),
        CircuitOperation(
            name="cc_prx",
            locus=("QB1",),
            args={
                "phase": 0.0,
                "angle": 0.0,
                "feedback_qubit": "QB1",
                "feedback_key": "m_3_2_1",
            },
        ),
        CircuitOperation(name="reset", locus=("QB2",), args={}),
    ]
    circuit = deserialize_instructions(instructions, {"QB1": 0, "QB2": 1, "CR1": 2}, Layout())
    assert isinstance(circuit, QuantumCircuit)
    assert len(circuit.count_ops()) == len(instructions)
    assert circuit.num_qubits == 3
    assert circuit.num_nonlocal_gates() == 2
    assert circuit.num_ancillas == 0
    assert circuit.num_clbits == 3
    assert len(circuit.cregs) == 3
    for circuit_instruction, name in zip(
        circuit.data, ["r", "cz", "move", "barrier", "measure", "delay", "if_else", "reset"]
    ):
        assert circuit_instruction.operation.name == name


def test_deserialize_instructions_roundtrip():
    """Check that native instructions are retrieved after a deserialize-serialize roundtrip."""
    instructions = [
        CircuitOperation(name="prx", locus=("QB1",), args={"phase": 0.2 * pi, "angle": 0.0}),
        CircuitOperation(name="cz", locus=("QB1", "QB2"), args={}),
        CircuitOperation(name="move", locus=("QB1", "CR1"), args={}),
        CircuitOperation(name="barrier", locus=("QB1", "QB2"), args={}),
        CircuitOperation(
            name="measure",
            locus=("QB1",),
            args={"key": "m_3_2_1", "feedback_key": "m_3_2_1"},
        ),
        CircuitOperation(name="delay", locus=("QB1",), args={"duration": 50e-9}),
        CircuitOperation(
            name="cc_prx",
            locus=("QB2",),
            args={
                "angle": 0.4 * pi,
                "phase": 0.6 * pi,
                "feedback_qubit": "QB1",
                "feedback_key": "m_3_2_1",
            },
        ),
        CircuitOperation(name="reset", locus=("QB2",), args={}),
    ]
    circuit = deserialize_instructions(instructions, {"QB1": 0, "QB2": 1, "CR1": 2}, Layout())
    new_instructions = serialize_instructions(circuit, qubit_index_to_name={0: "QB1", 1: "QB2", 2: "CR1"})
    assert new_instructions == instructions


def test_deserialize_instructions_unsupported_instruction():
    """Check that invalid instruction raises an error."""
    instruction = CircuitOperation(name="cz", locus=("QB1", "QB2"), args={})
    instruction.name = "cx"  # Purposely creating an instruction with an unsupported name.
    with pytest.raises(ValueError, match="Unsupported instruction cx in the circuit."):
        deserialize_instructions([instruction], {"QB1": 0, "QB2": 1, "CR1": 2}, Layout())


def test_serialize_circuit_raises_error_for_non_transpiled_circuit(iqm_client_mock, circuit, linear_3q_architecture):
    when(iqm_client_mock).get_dynamic_quantum_architecture(None).thenReturn(linear_3q_architecture)
    when(iqm_client_mock).get_dynamic_quantum_architecture(linear_3q_architecture.calibration_set_id).thenReturn(
        linear_3q_architecture
    )

    backend = IQMBackend(iqm_client_mock)
    circuit = QuantumCircuit(3)
    circuit.cz(0, 2)
    with pytest.raises(CircuitValidationError, match=re.escape("('QB1', 'QB3') is not allowed as locus for 'cz'")):
        backend.run(circuit)


def test_serialize_circuit_raises_error_for_unsupported_instruction(backend, circuit):
    circuit.sx(0)
    with pytest.raises(ValueError, match=f"Instruction 'sx' in the circuit '{circuit.name}' is not natively supported"):
        backend.serialize_circuit(circuit)


def test_serialize_circuit_does_not_raise_for_x_rx_y_ry(backend, circuit):
    circuit.x(0)
    circuit.rx(0.123, 0)
    circuit.y(0)
    circuit.ry(0.321, 0)
    backend.serialize_circuit(circuit)


def test_serialize_circuit_raises_error_for_unsupported_metadata(backend, circuit):
    circuit.append(RGate(theta=pi, phi=0), [0])
    circuit.metadata = {"some-key": complex(1.0, 2.0)}
    with pytest.warns(UserWarning):
        serialized_circuit = backend.serialize_circuit(circuit)
    assert serialized_circuit.metadata is None


@pytest.mark.parametrize(
    "gate, expected_angle, expected_phase",
    [
        (RGate(theta=pi, phi=0), pi, 0),
        (RGate(theta=0, phi=pi), 0, pi),
        (RGate(theta=0, phi=2 * pi), 0, 2 * pi),
        (RGate(theta=2 * pi, phi=pi), 2 * pi, pi),
    ],
)
def test_serialize_circuit_maps_r_gate(circuit, gate, expected_angle, expected_phase, backend):
    circuit.append(gate, [0])
    circuit_ser = backend.serialize_circuit(circuit)
    assert len(circuit_ser.instructions) == 1
    instr = circuit_ser.instructions[0]
    assert instr.name == "prx"
    assert instr.locus == ("QB1",)
    # Serialized angles should be in radians
    assert instr.args["angle"] == expected_angle
    assert instr.args["phase"] == expected_phase


@pytest.mark.parametrize(
    "gate, expected_angle, expected_phase",
    [
        (XGate(), pi, 0),
        (RXGate(theta=pi / 2), (1 / 2) * pi, 0),
        (RXGate(theta=2 * pi / 3), (2 / 3) * pi, 0),
        (YGate(), pi, (1 / 2) * pi),
        (RYGate(theta=pi / 2), (1 / 2) * pi, (1 / 2) * pi),
        (RYGate(theta=2 * pi / 3), (2 / 3) * pi, (1 / 2) * pi),
    ],
)
def test_serialize_circuit_maps_x_rx_y_ry_gates(backend, circuit, gate, expected_angle, expected_phase):
    circuit.append(gate, [0])
    circuit_ser = backend.serialize_circuit(circuit)
    assert len(circuit_ser.instructions) == 1
    instr = circuit_ser.instructions[0]
    assert instr.name == "prx"
    assert instr.locus == ("QB1",)
    assert instr.args["angle"] == expected_angle
    assert instr.args["phase"] == expected_phase


def test_serialize_circuit_maps_cz_gate(circuit, backend):
    circuit.cz(0, 2)
    circuit_ser = backend.serialize_circuit(circuit)
    assert len(circuit_ser.instructions) == 1
    assert circuit_ser.instructions[0].name == "cz"
    assert circuit_ser.instructions[0].locus == ("QB1", "QB3")
    assert circuit_ser.instructions[0].args == {}


def test_serialize_circuit_maps_individual_measurements(circuit, backend):
    circuit.measure(0, 0)
    circuit.measure(1, 1)
    circuit.measure(2, 2)
    circuit_ser = backend.serialize_circuit(circuit)
    assert len(circuit_ser.instructions) == 3
    for i, instruction in enumerate(circuit_ser.instructions):
        assert instruction.name == "measure"
        assert instruction.locus == (f"QB{i + 1}",)
        key = f"c_3_0_{i}"
        assert instruction.args == {"key": key}


def test_serialize_circuit_batch_measurement(circuit, backend):
    circuit.measure([0, 1, 2], [0, 1, 2])
    circuit_ser = backend.serialize_circuit(circuit)
    assert len(circuit_ser.instructions) == 3
    for i, instruction in enumerate(circuit_ser.instructions):
        assert instruction.name == "measure"
        assert instruction.locus == (f"QB{i + 1}",)
        key = f"c_3_0_{i}"
        assert instruction.args == {"key": key}


def test_serialize_circuit_barrier(circuit, backend):
    circuit.r(theta=pi, phi=0, qubit=0)
    circuit.barrier([0, 1])
    circuit_ser = backend.serialize_circuit(circuit)
    assert len(circuit_ser.instructions) == 2
    assert circuit_ser.instructions[1].name == "barrier"
    assert circuit_ser.instructions[1].locus == ("QB1", "QB2")
    assert circuit_ser.instructions[1].args == {}


@pytest.mark.parametrize(
    "duration,unit,in_seconds",
    [
        (3, "dt", 3e-9),
        (0.4, "s", 0.4),
        (1.23, "ms", 1.23e-3),
        (240, "us", 240e-6),
        (25, "ns", 25e-9),
        (50, "ps", 50e-12),
    ],
)
def test_serialize_circuit_delay(circuit, backend, duration, unit, in_seconds):
    circuit.delay(duration, 0, unit=unit)
    circuit_ser = backend.serialize_circuit(circuit)
    assert len(circuit_ser.instructions) == 1
    assert circuit_ser.instructions[0].name == "delay"
    assert circuit_ser.instructions[0].locus == ("QB1",)
    assert circuit_ser.instructions[0].args == pytest.approx({"duration": in_seconds})


def test_serialize_circuit_id(circuit, backend):
    circuit.r(theta=pi, phi=0, qubit=0)
    circuit.id(0)
    circuit_ser = backend.serialize_circuit(circuit)
    assert len(circuit_ser.instructions) == 1
    assert circuit_ser.instructions[0].name == "prx"


def check_measure_cc_prx_pair(measure, cc_prx, check_key: bool = True):
    """Makes sure the given measure instruction provides control to the given cc_prx instruction."""
    assert measure.name == "measure"
    assert cc_prx.name == "cc_prx"
    feedback_key = cc_prx.args["feedback_key"]
    assert measure.args["feedback_key"] == feedback_key
    if check_key:
        assert measure.args["key"] == feedback_key
    assert cc_prx.args["feedback_qubit"] == "QB1"


@pytest.mark.skipif(Version(qiskit_version) >= Version("2.0"), reason="Qiskit 2.0 no longer supports c_if on gates.")
@pytest.mark.parametrize(
    "gate",
    [
        XGate(),
        RXGate(theta=pi / 2),
        YGate(),
        RYGate(theta=pi / 3),
        RGate(theta=2 * pi / 3, phi=0.176),
    ],
)
def test_serialize_circuit_c_if_different_qubit(backend, gate):
    """Test that the c_if classical control method works with the supported gates."""
    q = QuantumRegister(2, "q")
    control = ClassicalRegister(1, "c")
    result = ClassicalRegister(2, "r")
    qc = QuantumCircuit(q, control, result)

    # classically controlled gate on different qubit
    qc.measure(q[0], control[0])
    qc.append(gate.c_if(control, 1), [1])
    # final measurement
    qc.measure(q, result)

    circuit_ser = backend.serialize_circuit(qc)
    assert len(circuit_ser.instructions) == 4
    check_measure_cc_prx_pair(circuit_ser.instructions[0], circuit_ser.instructions[1])


@pytest.mark.parametrize(
    "gate",
    [
        XGate(),
        RXGate(theta=pi / 2),
        YGate(),
        RYGate(theta=pi / 3),
        RGate(theta=2 * pi / 3, phi=0.176),
    ],
)
def test_serialize_circuit_with_if_test_different_qubit(backend, gate):
    """Test that the with if_test classical control method works with the supported gates."""
    q = QuantumRegister(2, "q")
    control = ClassicalRegister(1, "c")
    result = ClassicalRegister(2, "r")
    qc = QuantumCircuit(q, control, result)

    # classically controlled gate on different qubit
    qc.measure(q[0], control[0])
    with qc.if_test((control, 1)):
        qc.append(gate, [1])
    # final measurement
    qc.measure(q, result)

    circuit_ser = backend.serialize_circuit(qc)
    assert len(circuit_ser.instructions) == 4
    check_measure_cc_prx_pair(circuit_ser.instructions[0], circuit_ser.instructions[1])


@pytest.mark.skipif(Version(qiskit_version) >= Version("2.0"), reason="Qiskit 2.0 no longer supports c_if on gates.")
@pytest.mark.parametrize(
    "gate",
    [
        XGate(),
        RXGate(theta=pi / 2),
        YGate(),
        RYGate(theta=pi / 3),
        RGate(theta=2 * pi / 3, phi=0.176),
    ],
)
def test_serialize_circuit_c_if_same_qubit(backend, gate):
    """Test that the c_if classical control method works with the supported gates."""
    q = QuantumRegister(2, "q")
    control = ClassicalRegister(1, "c")
    result = ClassicalRegister(2, "r")
    qc = QuantumCircuit(q, control, result)

    # classically controlled gate on same qubit
    qc.measure(q[0], control[0])
    qc.append(gate.c_if(control, 1), [0])
    # final measurement
    qc.measure(q, result)

    circuit_ser = backend.serialize_circuit(qc)
    assert len(circuit_ser.instructions) == 4
    check_measure_cc_prx_pair(circuit_ser.instructions[0], circuit_ser.instructions[1])


@pytest.mark.parametrize(
    "gate",
    [
        XGate(),
        RXGate(theta=pi / 2),
        YGate(),
        RYGate(theta=pi / 3),
        RGate(theta=2 * pi / 3, phi=0.176),
    ],
)
def test_serialize_circuit_with_if_test_same_qubit(backend, gate):
    """Test that the with if_test classical control method works with the supported gates."""
    q = QuantumRegister(2, "q")
    control = ClassicalRegister(1, "c")
    result = ClassicalRegister(2, "r")
    qc = QuantumCircuit(q, control, result)

    # classically controlled gate on same qubit
    qc.measure(q[0], control[0])
    with qc.if_test((control, 1)):
        qc.append(gate, [0])
    # final measurement
    qc.measure(q, result)

    circuit_ser = backend.serialize_circuit(qc)
    assert len(circuit_ser.instructions) == 4
    check_measure_cc_prx_pair(circuit_ser.instructions[0], circuit_ser.instructions[1])


@pytest.mark.skipif(Version(qiskit_version) >= Version("2.0"), reason="Qiskit 2.0 no longer supports c_if on gates.")
@pytest.mark.parametrize(
    "gate, arity",
    [
        (CZGate(), 2),
    ],
)
def test_serialize_circuit_c_if_unsupported(backend, gate, arity):
    """Test that the c_if with unsupported gate gives an error."""
    q = QuantumRegister(2, "q")
    control = ClassicalRegister(1, "c")
    qc = QuantumCircuit(q, control)
    qc.measure(q[0], control[0])
    qc.append(gate.c_if(control, 1), list(range(arity)))

    with pytest.raises(ValueError, match="only supports conditionals on"):
        backend.serialize_circuit(qc)


@pytest.mark.parametrize(
    "gate, arity",
    [
        (CZGate(), 2),
    ],
)
def test_serialize_circuit_with_if_test_unsupported(backend, gate, arity):
    """Test that the with if_test classical control method works with unsupported gates."""
    q = QuantumRegister(2, "q")
    control = ClassicalRegister(1, "c")
    qc = QuantumCircuit(q, control)
    qc.measure(q[0], control[0])
    with qc.if_test((control, 1)):
        qc.append(gate, list(range(arity)))

    with pytest.raises(ValueError, match="only supports conditionals on"):
        backend.serialize_circuit(qc)


@pytest.mark.skipif(Version(qiskit_version) >= Version("2.0"), reason="Qiskit 2.0 no longer supports c_if on gates.")
@pytest.mark.parametrize("value", [0, 2, 100])
def test_serialize_circuit_c_if_bad_value(backend, value):
    """Test that the c_if with a control value != 1 gives an error."""
    q = QuantumRegister(2, "q")
    control = ClassicalRegister(1, "c")
    qc = QuantumCircuit(q, control)
    qc.measure(q[0], control[0])
    qc.x(q[0]).c_if(control, value)

    with pytest.raises(ValueError, match="only value 1 is supported"):
        backend.serialize_circuit(qc)


@pytest.mark.parametrize("value", [0, 2, 100])
def test_serialize_circuit_with_if_test_bad_value(backend, value):
    """Test that the if_test with a control value != 1 gives an error."""
    q = QuantumRegister(2, "q")
    control = ClassicalRegister(1, "c")
    qc = QuantumCircuit(q, control)
    qc.measure(q[0], control[0])
    with qc.if_test((control, value)):
        qc.x(q[0])

    with pytest.raises(ValueError, match="only value 1 is supported"):
        backend.serialize_circuit(qc)


@pytest.mark.skipif(Version(qiskit_version) >= Version("2.0"), reason="Qiskit 2.0 no longer supports c_if on gates.")
@pytest.mark.parametrize("cbits", [2, 5])
def test_serialize_circuit_c_if_multiple_cbits(backend, cbits):
    """Test that the c_if using a classical register with more than one bit gives an error."""
    q = QuantumRegister(2, "q")
    control = ClassicalRegister(cbits, "c")
    qc = QuantumCircuit(q, control)
    qc.measure(q[0], control[0])
    qc.x(q[0]).c_if(control, 0)

    with pytest.raises(ValueError, match="conditioned on multiple bits"):
        backend.serialize_circuit(qc)


@pytest.mark.parametrize("cbits", [2, 5])
def test_serialize_circuit_with_if_test_multiple_cbits(backend, cbits):
    """Test that the with if_test using a classical register with more than one bit gives an error."""
    q = QuantumRegister(2, "q")
    control = ClassicalRegister(cbits, "c")
    qc = QuantumCircuit(q, control)
    qc.measure(q[0], control[0])
    with qc.if_test((control, 0)):
        qc.x(q[0])

    with pytest.raises(ValueError, match="conditioned on multiple bits"):
        backend.serialize_circuit(qc)


def test_serialize_circuit_reset(backend):
    """Test that the reset operation is accepted."""
    qc = QuantumCircuit(2, 2)
    qc.ry(pi / 2, 0)
    qc.ry(pi / 2, 1)
    qc.cz(0, 1)
    qc.ry(-pi / 2, 0)
    qc.reset(0)
    # final measurement
    qc.measure_all()
    circuit_ser = backend.serialize_circuit(qc)

    assert len(circuit_ser.instructions) == 8
    reset = circuit_ser.instructions[4]
    assert reset.name == "reset"


def test_serialize_if_test_qubit_mapping(backend):
    """Test that the serialization maps the circuit to the correct qubits."""
    q = QuantumRegister(2, "q")
    control = ClassicalRegister(1, "c")
    result = ClassicalRegister(2, "r")
    qc = QuantumCircuit(q, control, result)
    qc.measure(q[0], control[0])
    with qc.if_test((control, 1)):
        qc.x(1)
    qc.measure(q, result)
    # Choose qubits to map to
    layout = [0, 1]
    transpiled_circuit = transpile(qc, backend=backend, initial_layout=layout)
    circuit_ser = backend.serialize_circuit(transpiled_circuit)
    # Check the final circuit only uses the mapped qubits
    for instruction in circuit_ser.instructions:
        for locus in instruction.locus:
            assert locus in {backend.index_to_qubit_name(i) for i in layout}
