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
from iqm import iqm_client
from iqm.cirq_iqm.iqm_gates import IQMMoveGate
from iqm.iqm_client import Instruction

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


def instruction_to_operation(instr: Instruction) -> Operation:
    """Convert an IQM instruction to a Cirq Operation.

    Args:
        instr: the IQM instruction

    Returns:
        Operation: the converted operation

    Raises:
        OperationNotSupportedError When the circuit contains an unsupported operation.

    """
    if instr.name not in _IQM_CIRQ_OP_MAP:
        raise OperationNotSupportedError(f"Operation {instr.name} not supported.")

    qubits = [NamedQid(qubit, dimension=2) for qubit in instr.qubits]

    if instr.name == "cc_prx":
        # special case
        args = {"exponent": 2 * instr.args["angle_t"], "phase_exponent": 2 * instr.args["phase_t"]}
        # ignore feedback_qubit, we currently only support 1-qubit KeyConditions so it can be
        # added in serialize_circuit
        return PhasedXPowGate(**args)(*qubits).with_classical_controls(instr.args["feedback_key"])

    cirq_op = _IQM_CIRQ_OP_MAP[instr.name][0]
    if instr.name == "prx":
        args = {"exponent": 2 * instr.args["angle_t"], "phase_exponent": 2 * instr.args["phase_t"]}
    elif instr.name == "measure":
        args = {"num_qubits": len(qubits), "key": instr.args["key"]}
        # ignore feedback_key, in Cirq it has to be the same as key
    else:
        # cz, move, reset have no args
        args = {}
    return cirq_op(**args)(*qubits)


class OperationNotSupportedError(RuntimeError):
    """Raised when a given operation is not supported by the IQM server."""


def map_operation(operation: Operation) -> Instruction:
    """Map a Cirq Operation to the IQM data transfer format.

    Assumes the circuit has been transpiled so that it only contains operations natively supported by the
    given IQM quantum architecture.

    Args:
        operation: a Cirq Operation

    Returns:
        Instruction: the converted operation

    Raises:
        OperationNotSupportedError When the circuit contains an unsupported operation.

    """
    locus = tuple(qubit.name if isinstance(qubit, NamedQid) else str(qubit) for qubit in operation.qubits)
    if isinstance(operation.gate, (PhasedXPowGate, XPowGate, YPowGate)):
        return Instruction(
            name="prx",
            qubits=locus,
            args={"angle_t": operation.gate.exponent / 2, "phase_t": operation.gate.phase_exponent / 2},
        )
    if isinstance(operation.gate, MeasurementGate):
        if any(operation.gate.full_invert_mask()):
            raise OperationNotSupportedError("Invert mask not supported")

        return Instruction(
            name="measure",
            qubits=locus,
            args={"key": operation.gate.key},
        )
    if isinstance(operation.gate, CZPowGate):
        if operation.gate.exponent == 1.0:
            return Instruction(
                name="cz",
                qubits=locus,
                args={},
            )
        raise OperationNotSupportedError(
            f"CZPowGate exponent was {operation.gate.exponent}, but only 1 is natively supported."
        )

    if isinstance(operation.gate, IQMMoveGate):
        return Instruction(
            name="move",
            qubits=locus,
            args={},
        )

    if isinstance(operation.gate, ResetChannel):
        return Instruction(name="reset", qubits=locus)

    if isinstance(operation, ClassicallyControlledOperation):
        if len(operation._conditions) > 1:
            raise OperationNotSupportedError("Classically controlled gates can currently only have one condition.")
        if not isinstance(operation._conditions[0], KeyCondition):
            raise OperationNotSupportedError("Only KeyConditions are supported as classical controls.")
        if isinstance(operation._sub_operation.gate, (PhasedXPowGate, XPowGate, YPowGate)):
            return Instruction(
                name="cc_prx",
                qubits=locus,
                args={
                    "angle_t": operation._sub_operation.gate.exponent / 2,
                    "phase_t": operation._sub_operation.gate.phase_exponent / 2,
                    "feedback_qubit": "",
                    "feedback_key": str(operation._conditions[0]),
                },
            )
        raise OperationNotSupportedError(
            f"Classical control on the {operation._sub_operation.gate} gate is not supported."
        )

        # skipping feedback_qubit and feedback_key information until total circuit serialization

    raise OperationNotSupportedError(f"{type(operation.gate)} not natively supported.")


def serialize_circuit(circuit: Circuit) -> iqm_client.Circuit:
    """Serializes a quantum circuit into the IQM data transfer format.

    Args:
        circuit: quantum circuit to serialize

    Returns:
        data transfer object representing the circuit

    """
    total_ops_list = [op for moment in circuit for op in moment]
    cc_prx_support = any(isinstance(op, ClassicallyControlledOperation) for op in total_ops_list)
    instructions = list(map(map_operation, total_ops_list))

    if cc_prx_support:
        mkey_to_measurement = {}
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
                if len(measurement.qubits) != 1:
                    raise OperationNotSupportedError("cc_prx must depend on the measurement result of a single qubit.")
                inst.args["feedback_qubit"] = measurement.qubits[0]
                measurement.args["feedback_key"] = feedback_key

    return iqm_client.Circuit(name="Serialized from Cirq", instructions=instructions, metadata=None)


def deserialize_circuit(circuit: iqm_client.Circuit) -> Circuit:
    """Deserializes a quantum circuit from the IQM data transfer format to a Cirq Circuit.

    Args:
        circuit: data transfer object representing the circuit

    Returns:
        quantum circuit

    """
    return Circuit(
        map(
            instruction_to_operation,
            circuit.instructions,
        )
    )
