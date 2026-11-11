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

"""Build Clifford training circuits, convert them to Stim and compute their exact expectation values."""

from __future__ import annotations

from typing import Any
import warnings

from iqm.error_reduction_tools.cdr.cdr_clifford_utils import (
    append_stim_sequence,
    is_logically_single_qubit_clifford_r,
    stim_sequence_for_r_gate,
)
from iqm.error_reduction_tools.cdr.cdr_observables import (
    diagonal_observable_to_terms,
    get_observable_num_qubits,
)
from iqm.error_reduction_tools.utils.circuit_utils import clbit_index_from_key
import stim

from iqm.pulse import Circuit, CircuitOperation


def qubit_order(circuit: Circuit) -> list[str]:
    """Return the names of the qubits of ``circuit``, in order of first appearance.

    Args:
        circuit: A pulse circuit.

    Returns:
        The qubit names.

    """
    order: list[str] = []
    seen: set[str] = set()
    for operation in circuit.instructions:
        for qubit in operation.locus:
            if qubit not in seen:
                seen.add(qubit)
                order.append(qubit)
    return order


def is_instruction_allowed_in_training_circuit(
    operation: CircuitOperation,
    atol: float = 1e-9,
) -> bool:
    """Check whether an operation may appear in a Clifford training circuit.

    Allowed are Clifford ``prx`` gates, ``cz``, ``delay``, ``measure`` and ``barrier``.

    Args:
        operation: The operation.
        atol: Tolerance for Clifford detection.

    Returns:
        True if the operation is allowed.

    """
    if operation.name == "prx":
        theta = float(operation.args["angle"])
        phi = float(operation.args["phase"])
        return is_logically_single_qubit_clifford_r(theta, phi, atol=atol)

    # A delay acts as an identity.
    return operation.name in {"cz", "delay", "measure", "barrier"}


def validate_training_circuit_is_clifford(
    training_circuit: Circuit,
    atol: float = 1e-9,
    raise_on_failure: bool = True,
) -> bool:
    """Check that every operation of a training circuit passes :func:`is_instruction_allowed_in_training_circuit`.

    Args:
        training_circuit: The training circuit.
        atol: Tolerance for Clifford detection.
        raise_on_failure: Whether to raise on the first operation that is not allowed.

    Returns:
        True if every operation is allowed.

    Raises:
        ValueError: If ``raise_on_failure`` is True and an operation is not allowed.

    """
    for inst_index, operation in enumerate(training_circuit.instructions):
        ok = is_instruction_allowed_in_training_circuit(operation, atol=atol)

        if not ok:
            if raise_on_failure:
                raise ValueError(
                    f"Training circuit validation failed at instruction {inst_index}: {operation.name} "
                    f"with args {operation.args}."
                )

            return False

    return True


def training_circuit_to_stim(
    training_circuit: Circuit,
    clifford_atol: float = 1e-9,
    preserve_num_qubits: bool = True,
    return_measurement_map: bool = True,
) -> Any:
    """Convert a Clifford training circuit into a Stim circuit.

    Args:
        training_circuit: The training circuit.
        clifford_atol: Tolerance for Clifford detection.
        preserve_num_qubits: Whether to add an identity on every qubit, so that idle qubits are counted by Stim.
        return_measurement_map: Whether to also return one dict per measurement, with the keys
            ``"stim_measurement_index"``, ``"qubit"``, ``"clbit_index"`` and ``"instruction_index"``.

    Returns:
        The Stim circuit, and the measurement map if ``return_measurement_map`` is True.

    Raises:
        ValueError: If the circuit contains a non-Clifford prx gate or an unsupported operation.

    """
    stim_circuit = stim.Circuit()

    qubits = qubit_order(training_circuit)
    qubit_index = {q: i for i, q in enumerate(qubits)}

    measurement_map = []
    stim_measurement_index = 0
    clbit_counter = 0

    num_qubits = len(qubits)
    if preserve_num_qubits and num_qubits > 0:
        stim_circuit.append("I", list(range(num_qubits)))

    for inst_index, operation in enumerate(training_circuit.instructions):
        if operation.name == "prx":
            theta = float(operation.args["angle"])
            phi = float(operation.args["phase"])

            # Also verifies that the prx gate is a Clifford.
            stim_sequence = stim_sequence_for_r_gate(
                theta,
                phi,
                atol=clifford_atol,
            )

            append_stim_sequence(stim_circuit, stim_sequence, qubit_index[operation.locus[0]])
            continue

        if operation.name == "cz":
            if len(operation.locus) != 2:  # noqa: PLR2004
                raise ValueError(f"CZ expects 2 qubits at instruction {inst_index}, got locus={operation.locus}.")

            stim_circuit.append("CZ", [qubit_index[q] for q in operation.locus])
            continue

        if operation.name in {"delay", "barrier"}:
            continue

        if operation.name == "measure":
            if len(operation.locus) != 1:
                raise ValueError(
                    f"Measurement expects 1 qubit at instruction {inst_index}, got locus={operation.locus}."
                )

            stim_circuit.append("M", [qubit_index[operation.locus[0]]])

            key = operation.args.get("key")
            from_key = clbit_index_from_key(key) if isinstance(key, str) else None

            measurement_map.append(
                {
                    "stim_measurement_index": stim_measurement_index,
                    "qubit": qubit_index[operation.locus[0]],
                    "clbit_index": clbit_counter if from_key is None else from_key,
                    "instruction_index": inst_index,
                }
            )

            stim_measurement_index += 1
            clbit_counter += 1
            continue

        raise ValueError(
            f"Cannot convert training circuit operation to Stim at instruction {inst_index}: {operation.name} "
            f"with args {operation.args}."
        )

    if return_measurement_map:
        return stim_circuit, measurement_map

    return stim_circuit


def stim_circuit_without_measurements(stim_circuit: stim.Circuit) -> stim.Circuit:
    """Return a copy of a Stim circuit without its measurements.

    Args:
        stim_circuit: The Stim circuit.

    Returns:
        The other instructions, in order.

    """
    measurement_names = {"M", "MX", "MY", "MZ", "MR", "MRX", "MRY", "MRZ"}

    clean = stim.Circuit()
    for instruction in stim_circuit:
        if instruction.name == "REPEAT":
            clean += stim_circuit_without_measurements(instruction.body_copy()) * instruction.repeat_count
            continue
        if instruction.name in measurement_names:
            continue
        clean.append(instruction)

    return clean


def build_clbit_to_stim_qubit_map(measurement_map: list[dict]) -> dict:
    """Map each classical bit to the Stim qubit measured into it.

    Args:
        measurement_map: Map from :func:`training_circuit_to_stim` with ``return_measurement_map=True``.

    Returns:
        The Stim qubit index of each classical bit.

    Raises:
        ValueError: If a measurement has no classical bit, or two measurements write to the same classical bit.

    """
    clbit_to_stim_qubit: dict[int, int] = {}
    for entry in measurement_map:
        clbit = entry["clbit_index"]
        stim_qubit = entry["qubit"]  # Stim qubit index == circuit qubit index.

        if clbit is None:
            raise ValueError(f"A measurement has no classical bit: {entry}.")
        if clbit in clbit_to_stim_qubit:
            raise ValueError(
                f"measurement_map assigns classical bit {clbit} to both qubit {clbit_to_stim_qubit[clbit]} and "
                f"qubit {stim_qubit} (instruction {entry['instruction_index']})."
            )
        clbit_to_stim_qubit[clbit] = stim_qubit

    return clbit_to_stim_qubit


def exact_expectation_from_stim(
    stim_circuit: stim.Circuit,
    measurement_map: list[dict],
    observable: Any,
    expectation_num_bits: int | None = None,
    imaginary_tolerance: float = 1e-12,
    warn_if_complex: bool = True,
) -> float | complex:
    """Compute the exact expectation value of an observable for a Clifford training circuit.

    Args:
        stim_circuit: Circuit from :func:`training_circuit_to_stim`. Its measurements are removed to get the state
            before measurement.
        measurement_map: Map from :func:`training_circuit_to_stim` with ``return_measurement_map=True``.
        observable: Diagonal observable indexed by classical bit, in a format accepted by
            :func:`~iqm.error_reduction_tools.cdr.cdr_observables.to_pauli_terms`.
        expectation_num_bits: Required length of the observable, or ``None`` for no check.
        imaginary_tolerance: Largest imaginary part that is treated as zero.
        warn_if_complex: Whether to warn when the value has a larger imaginary part.

    Returns:
        The expectation value.

    Raises:
        ValueError: If the observable is not diagonal, has the wrong length, or acts on a classical bit the
            measurement map does not cover.

    """
    observable_num_bits = get_observable_num_qubits(observable)
    if expectation_num_bits is not None and observable_num_bits != expectation_num_bits:
        raise ValueError(
            "Observable length does not match expectation_num_bits. "
            f"Observable acts on {observable_num_bits} bits, but "
            f"expectation_num_bits={expectation_num_bits}."
        )

    num_stim_qubits = stim_circuit.num_qubits

    # Stim qubit of each classical bit.
    clbit_to_stim_qubit = build_clbit_to_stim_qubit_map(measurement_map)

    def index_to_stim_qubit(i: int) -> int:
        if i not in clbit_to_stim_qubit:
            raise ValueError(f"Observable classical-bit index {i} has no corresponding measurement in measurement_map.")
        return clbit_to_stim_qubit[i]

    # Reconstruct the pre-measurement stabilizer state once.
    simulator = stim.TableauSimulator()
    simulator.do_circuit(stim_circuit_without_measurements(stim_circuit))

    terms = diagonal_observable_to_terms(observable)

    value = 0.0 + 0.0j
    for coeff, z_bits in terms:
        # Build a Z-type Pauli string over the Stim qubits for this term.
        pauli_chars = ["_"] * num_stim_qubits
        for bit_index in z_bits:
            stim_qubit = index_to_stim_qubit(bit_index)
            if stim_qubit < 0 or stim_qubit >= num_stim_qubits:
                raise ValueError(f"Mapped Stim qubit {stim_qubit} is out of range for {num_stim_qubits} qubits.")
            pauli_chars[stim_qubit] = "Z"

        pauli_string = stim.PauliString("".join(pauli_chars))

        term_expectation = simulator.peek_observable_expectation(pauli_string)

        value += coeff * term_expectation

    if abs(value.imag) <= imaginary_tolerance:
        return float(value.real)

    if warn_if_complex:
        warnings.warn(
            f"The exact expectation value has a non-negligible imaginary part: {value.imag}.",
            UserWarning,
            stacklevel=2,
        )

    return value
