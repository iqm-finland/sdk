# Copyright 2022-2025 Qiskit on IQM developers
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
"""Conversion tools from Qiskit to IQM representation."""

from __future__ import annotations

from collections.abc import Collection, Iterable
from dataclasses import dataclass
from math import pi
import re
import warnings

from iqm.qiskit_iqm.move_gate import MoveGate
from packaging.version import Version
from qiskit import QuantumCircuit as QiskitQuantumCircuit
from qiskit import __version__ as qiskit_version
from qiskit.circuit import ClassicalRegister, Clbit, Operation, QuantumRegister
from qiskit.transpiler.layout import Layout

from iqm.pulse import CircuitOperation


class InstructionNotSupportedError(RuntimeError):
    """Raised when a given instruction is not supported by the IQM Server."""


@dataclass(frozen=True)
class MeasurementKey:
    """Unique key associated with a measurement instruction.

    Qiskit stores the results of quantum measurements in classical registers consisting of bits.
    The circuit execution results are presented as bitstrings of a certain structure so that the classical
    register and the index within that register for each bit is implied from its position in the bitstring.

    For example, if you have two classical registers in the circuit with lengths 3 and 2, then the
    measurement results will look like '01 101' if the classical register of length 3 was added to
    the circuit first, and '101 01' otherwise. If a bit in a classical register is not used in any
    measurement operation it will still show up in the results with the default value of '0'.

    To be able to handle measurement results in a Qiskit-friendly way, we need to keep around some
    information about how the circuit was constructed. This can, for example, be achieved by keeping
    around the original Qiskit quantum circuit and using it when constructing the results in
    :class:`qiskit_IQMJob`. This should be done so that the circuit is saved on the server side and not in
    ``IQMJob``, since otherwise users will not be able to retrieve results from a detached Python
    environment solely based on the job id. Another option is to use measurement key strings to
    store the required info. Qiskit does not use measurement keys, so we are free to use them
    internally in the communication with the IQM Server, and can encode the necessary information in
    them.

    This class encapsulates the necessary info, and provides methods to transform between this
    representation and the measurement key string representation.

    Args:
        creg_name: name of the classical register
        creg_len: number of bits in the classical register
        creg_idx: Index of the classical register in the circuit. Determines the order in which this register was added
            to the circuit relative to the others.
        clbit_idx: index of the classical bit within the classical register

    """

    creg_name: str
    creg_len: int
    creg_idx: int
    clbit_idx: int

    def __str__(self):
        return f"{self.creg_name}_{self.creg_len}_{self.creg_idx}_{self.clbit_idx}"

    @classmethod
    def from_string(cls, string: str) -> MeasurementKey:
        """Create a MeasurementKey from its string representation."""
        match = re.match(r"^(.*)_(\d+)_(\d+)_(\d+)$", string)
        if match is None:
            raise ValueError("Invalid measurement key string representation.")
        return cls(match.group(1), int(match.group(2)), int(match.group(3)), int(match.group(4)))

    @classmethod
    def from_clbit(cls, clbit: Clbit, circuit: QiskitQuantumCircuit) -> MeasurementKey:
        """Create a MeasurementKey for a classical bit in a quantum circuit."""
        bitloc = circuit.find_bit(clbit)
        creg = bitloc.registers[0][0]
        creg_idx = circuit.cregs.index(creg)
        clbit_idx = bitloc.registers[0][1]
        return cls(creg.name, len(creg), creg_idx, clbit_idx)


def _apply_condition(
    operation: Operation,
    native_instructions: Iterable[CircuitOperation],
    clbit_to_measure: dict[Clbit, CircuitOperation],
) -> None:
    """Apply a classical condition to circuit instructions.

    Modifies the instructions in place.

    Args:
        operation: Operation containing the classical condition.
        native_instructions: Instructions to apply the condition to.
        clbit_to_measure: Maps bits in the classical register to the measurement operation that
            last wrote something into them.

    """
    # check that the condition is supported
    creg, value = operation.condition
    if isinstance(creg, ClassicalRegister):
        if len(creg) != 1:
            raise ValueError(f"{operation.name} is conditioned on multiple bits, this is not supported.")
        clbit = creg[0]
    else:
        clbit = creg  # it is a Clbit
    if value != 1:
        raise ValueError(f"{operation.name} is conditioned on integer value {value}, only value 1 is supported.")

    # Set up feedback routing.
    # The latest "measure" instruction to write to that classical bit is modified, it is
    # given an explicit feedback_key equal to its measurement key.
    # The same feedback_key is given to the controlled instruction, along with the feedback qubit.
    if (measure_inst := clbit_to_measure.get(clbit)) is None:
        raise ValueError(f"{operation.name} conditioned on {clbit}, which does not contain a measurement result yet.")
    feedback_key = measure_inst.args["key"]
    measure_inst.args["feedback_key"] = feedback_key  # this measure is used to provide feedback
    physical_qubit_name = measure_inst.locus[0]  # single-qubit measurement

    for inst in native_instructions:
        # TODO we do not check anywhere if cc_prx is available for this locus!
        if inst.name != "prx":
            raise ValueError(f"This backend only supports conditionals on r, x, y, rx and ry gates, not on {inst.name}")
        inst.name = "cc_prx"
        inst.args["feedback_key"] = feedback_key
        inst.args["feedback_qubit"] = physical_qubit_name


def serialize_instructions(  # noqa: PLR0912, PLR0915
    circuit: QiskitQuantumCircuit,
    qubit_index_to_name: dict[int, str],
    allowed_nonnative_gates: Collection[str] = (),
    *,
    clbit_to_measure: dict[Clbit, CircuitOperation] | None = None,
) -> list[CircuitOperation]:
    """Serialize a quantum circuit into the IQM data transfer format.

    This is IQM's internal helper for :meth:`.IQMBackend.serialize_circuit` that gives slightly more control.
    See :meth:`.IQMBackend.serialize_circuit` for details.

    The position of a qubit in ``circuit`` is used as its physical qubit index, which is what Qiskit's
    transpiler produces.

    Args:
        circuit: quantum circuit to serialize
        qubit_index_to_name: Mapping from qubit indices to the corresponding qubit names.
        allowed_nonnative_gates: Names of gates that are converted as-is without validation.
            By default, any gate that can't be converted will raise an error.
            If such gates are present in the circuit, the caller must edit the result to be valid and executable.
            Notably, since IQM transfer format requires named parameters and qiskit parameters don't have names, the
            `i` th parameter of an unrecognized instruction is given the name ``"p<i>"``.
        clbit_to_measure: Maps clbits to the latest "measure" instruction to store its result there, or
            None if nothing has been measured yet.

    Returns:
        list of IQM instructions representing the circuit

    Raises:
        ValueError: circuit contains an unsupported instruction or is not transpiled in general

    """
    instructions: list[CircuitOperation] = []
    # maps clbits to the latest "measure" instruction to store its result there
    if clbit_to_measure is None:
        clbit_to_measure = {}
    for circuit_instruction in circuit.data:
        instruction = circuit_instruction.operation
        qubit_idxs = tuple(circuit.find_bit(qubit).index for qubit in circuit_instruction.qubits)
        qubit_names = tuple(qubit_index_to_name[idx] for idx in qubit_idxs)
        if instruction.name == "r":
            angle = float(instruction.params[0])
            phase = float(instruction.params[1])
            native_inst = CircuitOperation(name="prx", locus=qubit_names, args={"angle": angle, "phase": phase})
        elif instruction.name == "x":
            native_inst = CircuitOperation(name="prx", locus=qubit_names, args={"angle": pi, "phase": 0.0})
        elif instruction.name == "rx":
            angle = float(instruction.params[0])
            native_inst = CircuitOperation(name="prx", locus=qubit_names, args={"angle": angle, "phase": 0.0})
        elif instruction.name == "y":
            native_inst = CircuitOperation(name="prx", locus=qubit_names, args={"angle": pi, "phase": 0.5 * pi})
        elif instruction.name == "ry":
            angle = float(instruction.params[0])
            native_inst = CircuitOperation(name="prx", locus=qubit_names, args={"angle": angle, "phase": 0.5 * pi})
        elif instruction.name == "cz":
            native_inst = CircuitOperation(name="cz", locus=qubit_names, args={})
        elif instruction.name == "move":
            native_inst = CircuitOperation(name="move", locus=qubit_names, args={})
        elif instruction.name == "barrier":
            native_inst = CircuitOperation(name="barrier", locus=qubit_names, args={})
        elif instruction.name == "delay":
            duration = float(instruction.params[0])
            # convert duration to seconds
            unit = instruction.unit
            if unit == "dt":
                duration *= 1e-9  # we arbitrarily pick dt == 1 ns
            elif unit == "s":
                pass
            elif unit == "ms":
                duration *= 1e-3
            elif unit == "us":
                duration *= 1e-6
            elif unit == "ns":
                duration *= 1e-9
            elif unit == "ps":
                duration *= 1e-12
            else:
                raise ValueError(f"Delay: Unsupported unit '{unit}'")
            native_inst = CircuitOperation(name="delay", locus=qubit_names, args={"duration": duration})
        elif instruction.name == "measure":
            if len(circuit_instruction.clbits) != 1:
                raise ValueError(
                    f"Unexpected: measurement instruction {circuit_instruction} uses multiple classical bits."
                )
            clbit = circuit_instruction.clbits[0]  # always a single-qubit measurement
            mk = str(MeasurementKey.from_clbit(clbit, circuit))
            native_inst = CircuitOperation(name="measure", locus=qubit_names, args={"key": mk})
            clbit_to_measure[clbit] = native_inst
        elif instruction.name == "reset":
            native_inst = CircuitOperation(name="reset", locus=qubit_names, args={})
        elif instruction.name == "id":
            continue
        elif instruction.name in allowed_nonnative_gates:
            args = {f"p{i}": param for i, param in enumerate(instruction.params)}
            native_inst = CircuitOperation(name=instruction.name, locus=qubit_names, args=args)
        elif instruction.name == "if_else":
            if_block, else_block = instruction.params
            if else_block is not None and len(else_block) > 0:  # Non-empty circuit in else-block
                raise ValueError("The use of an else-block with if_test is not supported.")
            # The block body keeps its pre-transpilation qubits, only the outer loci are kept up to date.
            if_instructions = serialize_instructions(
                if_block,
                dict(enumerate(qubit_names)),
                allowed_nonnative_gates,
                clbit_to_measure=clbit_to_measure,
            )
            _apply_condition(instruction, if_instructions, clbit_to_measure)
            instructions.extend(if_instructions)
            continue  # Skip the rest of the loop, as we already handled the instructions
        else:
            raise ValueError(
                f"Instruction '{instruction.name}' in the circuit '{circuit.name}' is not natively supported. "
                f"You need to transpile the circuit before execution."
            )
        # classically controlled gates (using the c_if method) need to be updated
        if (
            Version(qiskit_version) < Version("2.0.0") and instruction.condition is not None
        ):  # None means no classical condition
            if Version(qiskit_version) < Version("1.3.0"):
                # Avoid double deprecation warnings.
                warnings.warn(
                    DeprecationWarning(
                        "The use of Qiskit's `c_if` method is deprecated and will be removed in a future release"
                        "of IQM Client. Please use the `with circuit.if_test(...)` construction instead."
                    )
                )
            _apply_condition(instruction, [native_inst], clbit_to_measure)
        instructions.append(native_inst)
    return instructions


def deserialize_instructions(
    instructions: Iterable[CircuitOperation],
    qubit_name_to_index: dict[str, int],
    layout: Layout | None = None,
) -> QiskitQuantumCircuit:
    """Turn a list of IQM Instructions into a Qiskit QuantumCircuit.

    Args:
        instructions: The gates in the circuit.
        qubit_name_to_index: Mapping from qubit names to their indices, as specified in a backend.
        layout: Qiskit representation of a layout for the circuit. The returned Qiskit circuit inherits
            its QuantumRegisters and Qubits from ``layout``.
            It tells which Qubit object in the circuit matches which physical qubit index.
            If None, return a QuantumCircuit with just one register called "q".

    Raises:
        ValueError: Thrown when a given instruction is not supported.

    Returns:
        Qiskit circuit represented by the given instructions.

    """
    # maps measurement key to the corresponding clbit
    mk_to_clbit: dict[str, Clbit] = {}
    # maps feedback key to the corresponding clbit
    fk_to_clbit: dict[str, Clbit] = {}

    # maps creg index to creg in the circuit
    cl_regs: dict[int, ClassicalRegister] = {}

    def register_key(key: str, mapping: dict[str, Clbit]) -> None:
        """Update the classical registers and the given key-to-clbit mapping with the given key."""
        mk = MeasurementKey.from_string(key)
        # find/create the corresponding creg
        creg = cl_regs.setdefault(mk.creg_idx, ClassicalRegister(size=mk.creg_len, name=mk.creg_name))
        # add the key to the given mapping
        if mk.clbit_idx < len(creg):
            mapping[str(mk)] = creg[mk.clbit_idx]
        else:
            raise IndexError(f"{mk}: Clbit index {mk.clbit_idx} is out of range for {creg}.")

    for instr in instructions:
        if instr.name == "measure":
            register_key(instr.args["key"], mk_to_clbit)
            if (key := instr.args.get("feedback_key")) is not None:
                register_key(key, fk_to_clbit)

    if layout is None:
        # create a trivial layout with one quantum register
        qreg = QuantumRegister(len(qubit_name_to_index), "q")
        layout = Layout.generate_trivial_layout(qreg)

    index_to_qiskit_qubit = layout.get_physical_bits()
    # Add an empty Classical register when the original circuit had unused classical registers
    circuit = QiskitQuantumCircuit(
        *layout.get_registers(),
        *(cl_regs.get(i, ClassicalRegister(0)) for i in range(max(cl_regs) + 1 if cl_regs else 0)),
    )
    for instr in instructions:
        locus = [index_to_qiskit_qubit[qubit_name_to_index[q]] for q in instr.locus]
        if instr.name == "prx":
            angle = instr.args["angle"]
            phase = instr.args["phase"]
            circuit.r(angle, phase, locus[0])
        elif instr.name == "cz":
            circuit.cz(*locus)
        elif instr.name == "move":
            circuit.append(MoveGate(), locus)
        elif instr.name == "measure":
            mk = MeasurementKey.from_string(instr.args["key"])
            circuit.measure(locus[0], mk_to_clbit[str(mk)])
        elif instr.name == "barrier":
            circuit.barrier(*locus)
        elif instr.name == "delay":
            duration = instr.args["duration"]
            circuit.delay(duration, locus, unit="s")  # native delay instructions always use seconds
        elif instr.name == "cc_prx":
            angle = instr.args["angle"]
            phase = instr.args["phase"]
            feedback_key = instr.args["feedback_key"]
            # NOTE: 'feedback_qubit' is not needed, because in Qiskit you only have single-qubit measurements.
            with circuit.if_test((fk_to_clbit[feedback_key], 1)):
                circuit.r(angle, phase, locus[0])
        elif instr.name == "reset":
            for qubit in locus:
                circuit.reset(qubit)
        else:
            raise ValueError(f"Unsupported instruction {instr.name} in the circuit.")
    return circuit
