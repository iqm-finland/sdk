# Copyright 2020–2025 Cirq on IQM developers
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

"""Helper functions for serializing and deserializing quantum circuits between Cirq and IQM circuit formats."""

from math import pi

from cirq import Circuit, KeyCondition, NamedQid
from cirq.ops import (
    ClassicallyControlledOperation,
    CZPowGate,
    Gate,
    MeasurementGate,
    Operation,
    PhasedXPowGate,
    ResetChannel,
    XPowGate,
    YPowGate,
)
from iqm.cirq_iqm.iqm_gates import IQMMoveGate

import iqm.pulse
from iqm.pulse import CircuitOperation

# Mapping from IQM operation names to cirq operations
_IQM_CIRQ_OP_MAP: dict[str, tuple[type[Gate], ...]] = {
    # XPow and YPow kept for convenience, Cirq does not know how to decompose them into PhasedXPow
    # so we would have to add those rules...
    "prx": (PhasedXPowGate, XPowGate, YPowGate),
    "cc_prx": (),  # cc_prx is special, it does not directly map to any single Cirq op
    "cz": (CZPowGate,),
    "move": (IQMMoveGate,),
    "measure": (MeasurementGate,),
    "reset": (ResetChannel,),
}


def circuit_operation_to_operation(circuit_operation: CircuitOperation) -> Operation:
    """Convert an IQM circuit operation to a Cirq Operation.

    Args:
        circuit_operation: the IQM circuit_operation

    Returns:
        Operation: the converted operation

    Raises:
        OperationNotSupportedError When the circuit contains an unsupported operation.

    """
    if circuit_operation.name not in _IQM_CIRQ_OP_MAP:
        raise OperationNotSupportedError(f"Operation {circuit_operation.name} not supported.")

    qubits = [NamedQid(qubit, dimension=2) for qubit in circuit_operation.locus]

    if circuit_operation.name == "cc_prx":
        # special case
        args = {
            "exponent": circuit_operation.args["angle"] / pi,
            "phase_exponent": circuit_operation.args["phase"] / pi,
        }
        # ignore feedback_qubit, we currently only support 1-qubit KeyConditions so it can be
        # added in serialize_circuit
        return PhasedXPowGate(**args)(*qubits).with_classical_controls(circuit_operation.args["feedback_key"])

    cirq_op = _IQM_CIRQ_OP_MAP[circuit_operation.name][0]
    if circuit_operation.name == "prx":
        args = {
            "exponent": circuit_operation.args["angle"] / pi,
            "phase_exponent": circuit_operation.args["phase"] / pi,
        }
    elif circuit_operation.name == "measure":
        args = {"num_qubits": len(qubits), "key": circuit_operation.args["key"]}
        # ignore feedback_key, in Cirq it has to be the same as key
    else:
        # cz, move, reset have no args
        args = {}
    return cirq_op(**args)(*qubits)


class OperationNotSupportedError(RuntimeError):
    """Raised when a given operation is not supported by the IQM Server."""


def operation_to_circuit_operation(operation: Operation) -> CircuitOperation:
    """Map a Cirq Operation to the IQM data transfer format.

    Assumes the circuit has been transpiled so that it only contains operations natively supported by the
    given IQM quantum architecture.

    Args:
        operation: a Cirq Operation

    Returns:
        CircuitOperation: the converted operation

    Raises:
        OperationNotSupportedError When the circuit contains an unsupported operation.

    """
    locus = tuple(qubit.name if isinstance(qubit, NamedQid) else str(qubit) for qubit in operation.qubits)
    if isinstance(operation.gate, (PhasedXPowGate, XPowGate, YPowGate)):
        return CircuitOperation(
            name="prx",
            locus=locus,
            args={"angle": operation.gate.exponent * pi, "phase": operation.gate.phase_exponent * pi},
        )
    if isinstance(operation.gate, MeasurementGate):
        if any(operation.gate.full_invert_mask()):
            raise OperationNotSupportedError("Invert mask not supported")

        return CircuitOperation(
            name="measure",
            locus=locus,
            args={"key": operation.gate.key},
        )
    if isinstance(operation.gate, CZPowGate):
        if operation.gate.exponent == 1.0:
            return CircuitOperation(
                name="cz",
                locus=locus,
                args={},
            )
        raise OperationNotSupportedError(
            f"CZPowGate exponent was {operation.gate.exponent}, but only 1 is natively supported."
        )

    if isinstance(operation.gate, IQMMoveGate):
        return CircuitOperation(
            name="move",
            locus=locus,
            args={},
        )

    if isinstance(operation.gate, ResetChannel):
        return CircuitOperation(name="reset", locus=locus)

    if isinstance(operation, ClassicallyControlledOperation):
        if len(operation._conditions) > 1:
            raise OperationNotSupportedError("Classically controlled gates can currently only have one condition.")
        if not isinstance(operation._conditions[0], KeyCondition):
            raise OperationNotSupportedError("Only KeyConditions are supported as classical controls.")
        if isinstance(operation._sub_operation.gate, (PhasedXPowGate, XPowGate, YPowGate)):
            return CircuitOperation(
                name="cc_prx",
                locus=locus,
                args={
                    "angle": operation._sub_operation.gate.exponent * pi,
                    "phase": operation._sub_operation.gate.phase_exponent * pi,
                    "feedback_qubit": "",
                    "feedback_key": str(operation._conditions[0]),
                },
            )
        raise OperationNotSupportedError(
            f"Classical control on the {operation._sub_operation.gate} gate is not supported."
        )

        # skipping feedback_qubit and feedback_key information until total circuit serialization

    raise OperationNotSupportedError(f"{type(operation.gate)} not natively supported.")


def serialize_circuit(circuit: Circuit) -> iqm.pulse.Circuit:
    """Serializes a quantum circuit into the IQM data transfer format.

    Args:
        circuit: quantum circuit to serialize

    Returns:
        data transfer object representing the circuit

    """
    total_ops_list = [op for moment in circuit for op in moment]
    cc_prx_support = any(isinstance(op, ClassicallyControlledOperation) for op in total_ops_list)
    instructions = tuple(map(operation_to_circuit_operation, total_ops_list))

    if cc_prx_support:
        mkey_to_measurement: dict[str, CircuitOperation] = {}
        for inst in instructions:
            if inst.name == "measure":
                if inst.args["key"] in mkey_to_measurement:
                    raise OperationNotSupportedError("Cannot use the same key for multiple measurements.")
                mkey_to_measurement[inst.args["key"]] = inst
            elif inst.name == "cc_prx":
                feedback_key = inst.args["feedback_key"]
                measurement = mkey_to_measurement.get(feedback_key)
                if measurement is None:
                    raise OperationNotSupportedError(
                        f"cc_prx has feedback_key {feedback_key}, but no measure operation with that key precedes it."
                    )
                if len(measurement.locus) != 1:
                    raise OperationNotSupportedError("cc_prx must depend on the measurement result of a single qubit.")
                inst.args["feedback_qubit"] = measurement.locus[0]
                measurement.args["feedback_key"] = feedback_key

    return iqm.pulse.Circuit(name="Serialized from Cirq", instructions=instructions, metadata=None)


def deserialize_circuit(circuit: iqm.pulse.Circuit) -> Circuit:
    """Deserializes a quantum circuit from the IQM data transfer format to a Cirq Circuit.

    Args:
        circuit: data transfer object representing the circuit

    Returns:
        quantum circuit

    """
    return Circuit(
        map(
            circuit_operation_to_operation,
            circuit.instructions,
        )
    )
