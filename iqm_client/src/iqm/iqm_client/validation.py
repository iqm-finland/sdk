#  ********************************************************************************
#  Copyright (c) 2019-2025 IQM Finland Oy.
#  All rights reserved. Confidential and proprietary.
#
#  Distribution or reproduction of any information contained herein
#  is prohibited without IQM Finland Oy’s prior written permission.
#  ********************************************************************************
"""Validation related helper functions for IQMClient."""

from collections.abc import Iterable
import itertools

from iqm.iqm_client.errors import CircuitValidationError
from iqm.iqm_client.models import (
    _SUPPORTED_OPERATIONS,
    Circuit,
    CircuitBatch,
    DynamicQuantumArchitecture,
    Instruction,
    MoveGateValidationMode,
    QIRCode,
)


def validate_qubit_mapping(
    architecture: DynamicQuantumArchitecture,
    circuits: CircuitBatch,
    qubit_mapping: dict[str, str] | None = None,
) -> None:
    """Validate the given qubit mapping.

    Args:
      architecture: Quantum architecture to check against.
      circuits: Circuits to be checked.
      qubit_mapping: Mapping of logical qubit names to physical qubit names.
          Can be set to ``None`` if all ``circuits`` already use physical qubit names.
          Note that the ``qubit_mapping`` is used for all ``circuits``.

    Raises:
        CircuitValidationError: There was something wrong with ``circuits``.

    """
    if qubit_mapping is None:
        return

    # check if qubit mapping is injective
    target_qubits = set(qubit_mapping.values())
    if not len(target_qubits) == len(qubit_mapping):
        raise CircuitValidationError("Multiple logical qubits map to the same physical qubit.")

    # check if qubit mapping covers all qubits in the circuits
    for i, circuit in enumerate(circuits):
        if isinstance(circuit, (QIRCode)):
            continue
        diff = circuit.all_qubits() - set(qubit_mapping)
        if diff:
            raise CircuitValidationError(
                f"The qubits {diff} in circuit '{circuit.name}' at index {i} "
                f"are not found in the provided qubit mapping."
            )

    # check that each mapped qubit is defined in the quantum architecture
    for _logical, physical in qubit_mapping.items():
        if physical not in architecture.components:
            raise CircuitValidationError(f"Component {physical} not present in dynamic quantum architecture")


def validate_circuit_instructions(
    architecture: DynamicQuantumArchitecture,
    circuits: CircuitBatch,
    qubit_mapping: dict[str, str] | None = None,
    validate_moves: MoveGateValidationMode = MoveGateValidationMode.STRICT,
    *,
    must_close_sandwiches: bool = True,
) -> None:
    """Validate the given circuits against the given quantum architecture.

    Args:
      architecture: Quantum architecture to check against.
      circuits: Circuits to be checked.
      qubit_mapping: Mapping of logical qubit names to physical qubit names.
          Can be set to ``None`` if all ``circuits`` already use physical qubit names.
          Note that the ``qubit_mapping`` is used for all ``circuits``.
      validate_moves: Determines how MOVE gate validation works.
      must_close_sandwiches: Iff True, MOVE sandwiches cannot be left open when the circuit ends.

    Raises:
        CircuitValidationError: validation failed

    """
    for index, circuit in enumerate(circuits):
        if isinstance(circuit, QIRCode):
            continue

        measurement_keys: set[str] = set()
        for instr in circuit.instructions:
            validate_instruction(architecture, instr, qubit_mapping)
            # check measurement key uniqueness
            if instr.name in {"measure", "measurement"}:
                key = instr.args["key"]
                if key in measurement_keys:
                    raise CircuitValidationError(f"Circuit {index}: {instr!r} has a non-unique measurement key.")
                measurement_keys.add(key)
        validate_circuit_moves(
            architecture,
            circuit,
            qubit_mapping,
            validate_moves=validate_moves,
            must_close_sandwiches=must_close_sandwiches,
        )


def validate_instruction(
    architecture: DynamicQuantumArchitecture,
    instruction: Instruction,
    qubit_mapping: dict[str, str] | None = None,
) -> None:
    """Validate an instruction against the dynamic quantum architecture.

    Checks that the instruction uses a valid implementation, and targets a valid locus.

    Args:
      architecture: Quantum architecture to check against.
      instruction: Instruction to check.
      qubit_mapping: Mapping of logical qubit names to physical qubit names.
          Can be set to ``None`` if ``instruction`` already uses physical qubit names.

    Raises:
        CircuitValidationError: validation failed

    """
    op_info = _SUPPORTED_OPERATIONS.get(instruction.name)
    if op_info is None:
        raise CircuitValidationError(f"Unknown quantum operation '{instruction.name}'.")

    # apply the qubit mapping if any
    mapped_qubits = tuple(qubit_mapping[q] for q in instruction.qubits) if qubit_mapping else instruction.qubits

    def check_locus_components(allowed_components: Iterable[str], msg: str) -> None:
        """Checks that the instruction locus consists of the allowed components only."""
        for q, mapped_q in zip(instruction.qubits, mapped_qubits):
            if mapped_q not in allowed_components:
                raise CircuitValidationError(
                    f"{instruction!r}: Component {q} = {mapped_q} {msg}."
                    if qubit_mapping
                    else f"{instruction!r}: Component {q} {msg}."
                )

    if op_info.no_calibration_needed:
        # all QPU loci are allowed
        check_locus_components(architecture.components, msg="does not exist on the QPU")
        return

    gate_info = architecture.gates.get(instruction.name)
    if gate_info is None:
        raise CircuitValidationError(
            f"Operation '{instruction.name}' is not supported by the dynamic quantum architecture."
        )

    if instruction.implementation is not None:
        # specific implementation requested
        impl_info = gate_info.implementations.get(instruction.implementation)
        if impl_info is None:
            raise CircuitValidationError(
                f"Operation '{instruction.name}' implementation '{instruction.implementation}' "
                f"is not supported by the dynamic quantum architecture."
            )
        allowed_loci = impl_info.loci
        instruction_name = f"{instruction.name}.{instruction.implementation}"
    else:
        # any implementation is fine
        allowed_loci = gate_info.loci
        instruction_name = f"{instruction.name}"

    if op_info.factorizable:
        # Check that all the locus components are allowed by the architecture
        check_locus_components(
            {q for locus in allowed_loci for q in locus}, msg=f"is not allowed as locus for '{instruction_name}'"
        )
        return

    # Check that locus matches one of the allowed loci
    all_loci = (
        tuple(tuple(x) for locus in allowed_loci for x in itertools.permutations(locus))
        if op_info.symmetric
        else allowed_loci
    )
    if mapped_qubits not in all_loci:
        raise CircuitValidationError(
            f"{instruction.qubits} = {tuple(mapped_qubits)} is not allowed as locus for '{instruction_name}'"
            if qubit_mapping
            else f"{instruction.qubits} is not allowed as locus for '{instruction_name}'"
        )


def validate_circuit_moves(
    architecture: DynamicQuantumArchitecture,
    circuit: Circuit,
    qubit_mapping: dict[str, str] | None = None,
    validate_moves: MoveGateValidationMode = MoveGateValidationMode.STRICT,
    *,
    must_close_sandwiches: bool = True,
) -> None:
    """Raise an error if the MOVE gates in the circuit are not valid in the given architecture.

    Args:
        architecture: Quantum architecture to check against.
        circuit: Quantum circuit to validate.
        qubit_mapping: Mapping of logical qubit names to physical qubit names.
            Can be set to ``None`` if the ``circuit`` already uses physical qubit names.
        validate_moves: Option for bypassing full or partial MOVE gate validation.
        must_close_sandwiches: Iff True, MOVE sandwiches cannot be left open when the circuit ends.

    Raises:
        CircuitValidationError: validation failed

    """
    if validate_moves == MoveGateValidationMode.NONE:
        return
    move_gate = "move"
    # Check if MOVE gates are allowed on this architecture
    if move_gate not in architecture.gates:
        if any(i.name == move_gate for i in circuit.instructions):
            raise CircuitValidationError("MOVE instruction is not supported by the given device architecture.")
        return

    # some gates are allowed in MOVE sandwiches
    allowed_gates = {"barrier"}
    if validate_moves == MoveGateValidationMode.ALLOW_PRX:
        allowed_gates.add("prx")

    all_resonators = set(architecture.computational_resonators)
    all_qubits = set(architecture.qubits)
    if qubit_mapping:
        reverse_mapping = {phys: log for log, phys in qubit_mapping.items()}
        all_resonators = {reverse_mapping[q] if q in reverse_mapping else q for q in all_resonators}
        all_qubits = {reverse_mapping[q] if q in reverse_mapping else q for q in all_qubits}

    # Mapping from resonator to the qubit whose state it holds. Resonators not in the map hold no qubit state.
    resonator_occupations: dict[str, str] = {}
    # Qubits whose states are currently moved to a resonator
    moved_qubits: set[str] = set()

    for inst in circuit.instructions:
        if inst.name == "move":
            qubit, resonator = inst.qubits
            if not (qubit in all_qubits and resonator in all_resonators):
                raise CircuitValidationError(
                    f"MOVE instructions are only allowed between qubit and resonator, not {inst.qubits}."
                )

            if (resonator_qubit := resonator_occupations.get(resonator)) is None:
                # Beginning MOVE: check that the qubit hasn't been moved to another resonator
                if qubit in moved_qubits:
                    raise CircuitValidationError(
                        f"MOVE instruction {inst.qubits}: state of {qubit} is "
                        f"in another resonator: {resonator_occupations}."
                    )
                resonator_occupations[resonator] = qubit
                moved_qubits.add(qubit)
            else:
                # Ending MOVE: check that the qubit matches to the qubit that was moved to the resonator
                if resonator_qubit != qubit:
                    raise CircuitValidationError(
                        f"MOVE instruction {inst.qubits} to an already occupied resonator: {resonator_occupations}."
                    )
                del resonator_occupations[resonator]
                moved_qubits.remove(qubit)
        elif moved_qubits:
            # Validate that moved qubits are not used during MOVE operations
            if inst.name not in allowed_gates:
                if overlap := set(inst.qubits) & moved_qubits:
                    raise CircuitValidationError(
                        f"Instruction {inst.name} acts on {inst.qubits} while the state(s) of {overlap} "
                        f"are in a resonator. Current resonator occupation: {resonator_occupations}."
                    )

    # Finally validate that all MOVE sandwiches have been ended before the circuit ends
    if must_close_sandwiches and resonator_occupations:
        raise CircuitValidationError(
            f"Circuit ends while qubit state(s) are still in a resonator: {resonator_occupations}."
        )
