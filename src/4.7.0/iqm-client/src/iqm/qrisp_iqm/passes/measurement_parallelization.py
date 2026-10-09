# Copyright 2026 IQM
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

"""Measurement Parallelization Pass.

This module provides a transpilation pass that combines consecutive measurement
operations into a single multi-qubit measurement. This optimization is more
efficient on IQM hardware because:

1. Multiple single-qubit measurements can be performed simultaneously
2. Reduces the overhead of multiple measurement calls
3. The hardware can execute parallel readout on all specified qubits at once

The pass scans through the circuit and collects consecutive measurement
instructions. When a non-measurement instruction is encountered (or at the end
of the circuit), all collected measurements are flushed as a single combined
multi-qubit measurement operation.

Example::

    >>> from qrisp import QuantumCircuit
    >>> from iqm.qrisp_iqm import measurement_parallelization
    >>>
    >>> qc = QuantumCircuit(3)
    >>> qc.h(0)
    >>> qc.measure([0, 1, 2])  # Three separate measurements
    >>>
    >>> # After parallelization, measurements become one multi-qubit operation
    >>> qc_optimized = measurement_parallelization(qc)

"""

from __future__ import annotations

from iqm.qrisp_iqm.pulse_operation import IQMPulseOperation
from qrisp import QuantumCircuit
from qrisp.circuit import Instruction
from qrisp.circuit.pass_management.circuit_pass import CircuitPass

from iqm.pulse.quantum_ops import QuantumOp


def flush_collection(measurement_collection: list[Instruction]) -> Instruction:
    """Combine collected measurement instructions into a single multi-qubit measurement.

    This helper function takes a list of single-qubit measurement instructions and
    creates a single IQMPulseOperation that measures all qubits simultaneously.

    Parameters
    ----------
    measurement_collection : list[Instruction]
        A list of single-qubit measurement instructions to combine.
        This list will be cleared after processing.

    Returns
    -------
    Instruction
        A single instruction representing a multi-qubit measurement operation
        that measures all qubits from the collection simultaneously.

    Notes
    -----
    The resulting measurement uses a combined key string that concatenates
    all classical bit identifiers, allowing proper result correlation.

    """
    # Collect all qubits and classical bits from the measurement instructions
    qubits = []
    clbits = []

    for instr in measurement_collection:
        qubits.append(instr.qubits[0])
        clbits.append(instr.clbits[0])

    # Clear the collection for reuse
    measurement_collection.clear()

    # Create a multi-qubit measurement operation
    # The arity parameter specifies how many qubits are measured simultaneously
    mm_op = QuantumOp(
        name="measure",
        params={"key": (str,)},
        arity=len(qubits),
    )

    # Build the key string from all classical bit identifiers
    # This allows tracking which measurement result corresponds to which qubit
    arg_str = "".join([cb.identifier + "," for cb in clbits[::-1]])

    multi_qubit_measurement = IQMPulseOperation(mm_op, param_dict={"key": arg_str}, num_clbits=len(clbits))
    return Instruction(multi_qubit_measurement, qubits, clbits)


@CircuitPass
def measurement_parallelization(qc: QuantumCircuit) -> QuantumCircuit:
    """Combine consecutive measurements into single multi-qubit measurements.

    This transpilation pass optimizes circuit execution on IQM hardware by
    collecting consecutive measurement operations and combining them into
    single multi-qubit measurement operations. This allows the hardware to
    perform parallel readout, reducing overall execution time.

    Parameters
    ----------
    qc : QuantumCircuit
        The input quantum circuit containing measurement operations.

    Returns
    -------
    QuantumCircuit
        A new circuit where consecutive measurements have been combined
        into multi-qubit measurement operations.

    Examples
    --------
    >>> from iqm.qrisp_iqm import measurement_parallelization
    >>> from qrisp import QuantumCircuit
    >>>
    >>> # Create a circuit with individual measurements
    >>> qc = QuantumCircuit(3)
    >>> qc.h([0, 1, 2])
    >>> qc.measure(0)
    >>> qc.measure(1)  # Consecutive with previous
    >>> qc.measure(2)  # Consecutive with previous
    >>>
    >>> # Combine into single multi-qubit measurement
    >>> qc_opt = measurement_parallelization(qc)
    >>> # Now has one 3-qubit measurement instead of three 1-qubit measurements

    Notes
    -----
    - Non-consecutive measurements (separated by gates) remain separate
    - The pass preserves circuit semantics while optimizing for hardware
    - Works with IQMPulseOperation for native pulse-level execution

    """
    # Create a new empty circuit with the same structure
    qc_new = qc.clearcopy()

    # Buffer to collect consecutive measurement instructions
    measurement_collection = []

    # Iterate through all instructions in the circuit
    for i in range(len(qc.data)):
        instr = qc.data[i]

        if instr.op.name == "measure":
            # Collect measurement instructions
            measurement_collection.append(instr)
        else:
            # Non-measurement instruction encountered
            # First, flush any collected measurements as a combined operation
            if measurement_collection:
                combined_instruction = flush_collection(measurement_collection)
                qc_new.append(combined_instruction)
            # Then append the current non-measurement instruction
            qc_new.append(instr)

    # Don't forget to flush any remaining measurements at the end
    if measurement_collection:
        combined_instruction = flush_collection(measurement_collection)
        qc_new.append(combined_instruction)

    return qc_new
