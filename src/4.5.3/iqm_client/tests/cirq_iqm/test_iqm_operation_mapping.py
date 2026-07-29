# Copyright 2020–2021 Cirq on IQM developers
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
from math import pi

import cirq
from cirq import (
    ClassicallyControlledOperation,
    CZPowGate,
    GateOperation,
    MeasurementGate,
    PhasedXPowGate,
    ResetChannel,
    XPowGate,
    YPowGate,
    ZPowGate,
)
from iqm.cirq_iqm.iqm_gates import IQMMoveGate
from iqm.cirq_iqm.serialize import (
    OperationNotSupportedError,
    circuit_operation_to_operation,
    operation_to_circuit_operation,
    serialize_circuit,
)
from mockito import mock
import pytest
from sympy import Eq, symbols  # type: ignore

from iqm.pulse import CircuitOperation

pytestmark = pytest.mark.usefixtures("unstub")


@pytest.fixture()
def qubit_1() -> cirq.NamedQubit:
    return cirq.NamedQubit("QB1")


@pytest.fixture()
def qubit_2() -> cirq.NamedQubit:
    return cirq.NamedQubit("QB2")


def test_raises_error_for_unsupported_operation(qubit_1):
    operation = GateOperation(ZPowGate(), [qubit_1])
    with pytest.raises(OperationNotSupportedError):
        operation_to_circuit_operation(operation)


def test_maps_measurement_gate(qubit_1):
    key = "test measurement"
    operation = GateOperation(MeasurementGate(1, key), [qubit_1])
    mapped = operation_to_circuit_operation(operation)
    expected = CircuitOperation(name="measure", locus=(str(qubit_1),), args={"key": key})
    assert expected == mapped


@pytest.mark.parametrize(
    "gate, expected_angle, expected_phase",
    [
        (XPowGate(exponent=0.5), 0.5 * pi, 0),
        (YPowGate(exponent=0.75), 0.75 * pi, 0.5 * pi),
        (PhasedXPowGate(exponent=0.25, phase_exponent=0.5), 0.25 * pi, 0.5 * pi),
    ],
)
def test_maps_to_phased_rx(qubit_1, gate, expected_angle, expected_phase):
    operation = GateOperation(gate, [qubit_1])
    mapped = operation_to_circuit_operation(operation)
    assert mapped.name == "prx"
    assert mapped.locus == (str(qubit_1),)

    # The unit for angle and phase is radians
    assert mapped.args["angle"] == expected_angle
    assert mapped.args["phase"] == expected_phase


def test_maps_cz_gate(qubit_1, qubit_2):
    operation = GateOperation(CZPowGate(), [qubit_1, qubit_2])
    mapped = operation_to_circuit_operation(operation)
    expected = CircuitOperation(name="cz", locus=(str(qubit_1), str(qubit_2)), args={})
    assert expected == mapped


def test_maps_r_gate():
    operation = ResetChannel().on(cirq.NamedQubit("QB1"))
    instruction = operation_to_circuit_operation(operation)
    assert instruction.name == "reset"
    assert instruction.locus == ("QB1",)


def test_raises_error_for_general_cz_pow_gate(qubit_1, qubit_2):
    operation = GateOperation(CZPowGate(exponent=0.5), [qubit_1, qubit_2])
    with pytest.raises(OperationNotSupportedError):
        operation_to_circuit_operation(operation)


def test_raises_error_for_non_trivial_invert_mask(qubit_1, qubit_2):
    operation = GateOperation(MeasurementGate(2, "measurement key", invert_mask=(True, False)), [qubit_1, qubit_2])
    with pytest.raises(OperationNotSupportedError):
        operation_to_circuit_operation(operation)


def test_circuit_operation_to_operation():
    instruction = CircuitOperation(name="prx", locus=("QB1",), args={"angle": 1.0 * pi, "phase": 0.5 * pi})
    operation = circuit_operation_to_operation(instruction)
    assert isinstance(operation.gate, PhasedXPowGate)
    assert operation.qubits == (cirq.NamedQubit("QB1"),)
    assert operation.gate.exponent == 1.0
    assert operation.gate.phase_exponent == 0.5

    instruction = CircuitOperation(name="cz", locus=("QB1", "QB2"), args={})
    operation = circuit_operation_to_operation(instruction)
    assert isinstance(operation.gate, CZPowGate)
    assert operation.qubits == (cirq.NamedQubit("QB1"), cirq.NamedQubit("QB2"))
    assert operation.gate.exponent == 1.0
    assert operation.gate.global_shift == 0.0

    instruction = CircuitOperation(name="measure", locus=("QB1",), args={"key": "test key"})
    operation = circuit_operation_to_operation(instruction)
    assert isinstance(operation.gate, MeasurementGate)
    assert operation.qubits == (cirq.NamedQubit("QB1"),)
    assert operation.gate.key == "test key"

    instruction = CircuitOperation(name="reset", locus=("QB1",))
    operation = circuit_operation_to_operation(instruction)
    assert isinstance(operation.gate, ResetChannel)
    assert operation.qubits == (cirq.NamedQubit("QB1"),)

    instruction = mock({"name": "unsupported", "locus": ("QB1",), "args": {}}, spec=CircuitOperation)
    with pytest.raises(OperationNotSupportedError):
        operation = circuit_operation_to_operation(instruction)

    instruction = CircuitOperation(name="move", locus=("QB1", "COMP_R"), args={})
    operation = circuit_operation_to_operation(instruction)
    assert isinstance(operation.gate, IQMMoveGate)
    assert operation.qubits == (cirq.NamedQubit("QB1"), cirq.NamedQid("COMP_R", dimension=2))


def test_cc_prx_operation():
    instruction = CircuitOperation(
        name="cc_prx",
        locus=("QB1",),
        args={"angle": 1.0 * pi, "phase": 1.5 * pi, "feedback_qubit": "COMP_R", "feedback_key": "test key"},
    )
    operation = circuit_operation_to_operation(instruction)
    assert isinstance(operation, ClassicallyControlledOperation)
    assert isinstance(operation._sub_operation.gate, PhasedXPowGate)


def test_cc_prx_error_circuits():
    qubits = cirq.LineQubit.range(2)
    late_measurement_circuit = cirq.Circuit(
        cirq.X(qubits[1]).with_classical_controls("f"), cirq.measure(qubits[0], key="f")
    )
    with pytest.raises(
        OperationNotSupportedError,
        match="cc_prx has feedback_key f, but no measure operation with that key precedes it.",
    ):
        serialize_circuit(late_measurement_circuit)

    multiple_conditions = cirq.Circuit(
        cirq.measure(qubits[0], key="f"),
        cirq.measure(qubits[1], key="g"),
        cirq.X(qubits[1]).with_classical_controls("f", "g"),
    )
    with pytest.raises(
        OperationNotSupportedError, match="Classically controlled gates can currently only have one condition."
    ):
        serialize_circuit(multiple_conditions)

    same_key_circuit = cirq.Circuit(
        cirq.measure(qubits[0], key="f"),
        cirq.measure(qubits[1], key="f"),
        cirq.X(qubits[1]).with_classical_controls("f"),
    )

    with pytest.raises(OperationNotSupportedError, match="Cannot use the same key for multiple measurements."):
        serialize_circuit(same_key_circuit)

    long_measurement = cirq.Circuit(
        cirq.measure(qubits[0], qubits[1], key="f"), cirq.X(qubits[1]).with_classical_controls("f")
    )
    with pytest.raises(
        OperationNotSupportedError, match="cc_prx must depend on the measurement result of a single qubit."
    ):
        serialize_circuit(long_measurement)

    f = symbols("f")
    condition = Eq(f, 0)
    wrong_condition_circuit = cirq.Circuit(
        cirq.measure(qubits[0], key="f"), cirq.X(qubits[1]).with_classical_controls(condition)
    )
    with pytest.raises(OperationNotSupportedError, match="Only KeyConditions are supported as classical controls."):
        serialize_circuit(wrong_condition_circuit)

    cc_z_circuit = cirq.Circuit(cirq.measure(qubits[0], key="f"), cirq.Z(qubits[1]).with_classical_controls("f"))
    with pytest.raises(OperationNotSupportedError, match="Classical control on the Z gate is not supported."):
        serialize_circuit(cc_z_circuit)
