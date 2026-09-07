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

"""Module for Poor Man's Pauli Twirling (PMPT) and circuit manipulation utilities.

This module provides functions and classes for randomizing quantum circuits using
PMPT and readout twirling techniques, as well as helpers for circuit analysis and
gate manipulation. It is designed to work with Qiskit circuits transpiled to a
restricted gate set (typically R and CZ gates), and supports readout twirling,
Z rotation tracking, and logical equivalence checks. The main entry point is
`randomize_circuit`, which returns a randomized circuit logically equivalent to
the input (considering readout twirling and suppression of Z rotations before measurement).
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
import warnings

import numpy as np
from numpy.random import Generator

from iqm.pulse import CircuitOperation

from ..utils.circuit_utils import TwirledCircuit

# Module-level constant for numerical tolerance
_ANGLE_TOLERANCE = 1e-10


@dataclass
class TrackingRegisters:
    """Container for tracking registers during circuit randomization."""

    virtual_z: dict[str, float] = field(default_factory=dict)
    """Track pending Z rotations per qubit."""

    extra_phi: dict[str, float] = field(default_factory=dict)
    """Track phase values for π rotations."""

    extra_pi: dict[str, int] = field(default_factory=dict)
    """Track count of π rotations per qubit."""

    sqg_number: dict[str, int] = field(default_factory=dict)
    """Number of single-qubit gates per qubit."""

    already_used_sqg: dict[str, int] = field(default_factory=dict)
    """Number of already processed gates per qubit."""

    rot_chars: dict[str, str] = field(default_factory=dict)
    """Readout twirling character(s) per qubit ('I' or 'X')."""


def absorb_pending_pi_rotation(
    registers: TrackingRegisters, qubit: str, theta: float, phi: float
) -> tuple[float, float]:
    """Absorb a pending π rotation (from the left) into the current R gate.

    Implements the identity::

        --[R(π, ξ)]--[R(θ, φ)]-- = --[R(θ+π, 2ξ - φ)]--[RZ(2(φ - ξ))]--

    The pending π rotation (with phase ``extra_phi``) is combined with the current
    gate R(θ, φ). The resulting RZ remainder is folded into ``virtual_z``, the
    π-rotation counter is incremented (making it even again), and ``extra_phi``
    is reset.

    Args:
        registers: Tracking registers (``extra_phi``, ``extra_pi``, ``virtual_z`` are updated).
        qubit: Qubit identifier.
        theta: Polar rotation angle of the current gate.
        phi: Phase angle of the current gate.

    Returns:
        Tuple (theta, phi) with updated rotation parameters.

    """
    theta = theta + np.pi
    xi = registers.extra_phi.get(qubit, 0)
    registers.virtual_z[qubit] = registers.virtual_z.get(qubit, 0) + 2 * (phi - xi)
    phi = -phi + 2 * xi
    registers.extra_pi[qubit] = registers.extra_pi.get(qubit, 0) + 1
    registers.extra_phi[qubit] = 0
    return theta, phi


def insert_twirling_pi_rotation(registers: TrackingRegisters, qubit: str, theta: float, phi: float, xi: float) -> float:
    """Insert a new π rotation (to the right) after the current R gate for twirling.

    Implements the identity::

        --[R(θ, φ)]--[R(π, ξ)]-- = --[R(θ+π, φ)]--[RZ(2(ξ - φ))]--

    followed by commuting the resulting RZ through the now-pending R(π, …)::

        extra_phi ← ξ - 2(ξ - φ)

    The π-rotation counter is incremented and the RZ remainder is stored in
    ``virtual_z`` for later absorption.

    Args:
        registers: Tracking registers (``extra_pi``, ``virtual_z``, ``extra_phi`` are updated).
        qubit: Qubit identifier.
        theta: Polar rotation angle of the current gate.
        phi: Phase angle of the current gate.
        xi: Phase of the inserted π rotation.

    Returns:
        Updated theta value.

    """
    registers.extra_pi[qubit] = registers.extra_pi.get(qubit, 0) + 1
    theta += np.pi
    registers.virtual_z[qubit] = registers.virtual_z.get(qubit, 0) + 2 * (xi - phi)
    registers.extra_phi[qubit] = xi - 2 * (xi - phi)
    return theta


def take_care_of_pending_operations(
    registers: TrackingRegisters, qubit: str, theta: float, phi: float
) -> tuple[float, float]:
    """Process pending Z and π-rotation operations before the current R gate.

    Before a new R(θ, φ) gate is applied, any accumulated Z rotation and
    pending π rotation on that qubit must be commuted through. This function:

    1. **Z rotation**: absorbs the pending RZ(α) by shifting φ → φ - α, using::

          --[RZ(α)]--[R(θ, φ)]-- = --[R(θ, φ - α)]--[RZ(α)]--

    2. **Pending π rotation** (if odd count): delegates to
       `absorb_pending_pi_rotation`, which combines the pending R(π, ξ) with
       the current gate.

    Args:
        registers: Tracking registers with ``virtual_z``, ``extra_pi``, ``extra_phi``.
        qubit: Qubit identifier.
        theta: Polar rotation angle of the current gate.
        phi: Phase angle of the current gate.

    Returns:
        Tuple (theta, phi) with updated rotation parameters.

    """
    # Z rotation.
    phi = phi - registers.virtual_z.get(qubit, 0)

    # Pi rotation
    if registers.extra_pi.get(qubit, 0) % 2 != 0:
        theta, phi = absorb_pending_pi_rotation(registers, qubit, theta, phi)

    return theta, phi


def process_r_gate(
    operation: CircuitOperation,
    twirled_circuit: TwirledCircuit,
    rgen: Generator,
    registers: TrackingRegisters,
    enforce_readout_twirling_basis: str | None = None,
    twirling_prob: float = 0.5,
    fix_axes: bool = False,
) -> None:
    """Process an R gate: absorb pending operations, then optionally insert a twirling π rotation.

    First delegates to `take_care_of_pending_operations` to commute any accumulated
    Z and π rotations into the gate parameters. Then, with probability
    ``twirling_prob`` (possibly adjusted by parity or readout twirling constraints),
    inserts a new π rotation via `insert_twirling_pi_rotation`. Finally appends
    the resulting PRX gate to the twirled circuit.

    Args:
        operation: The PRX CircuitOperation to process.
        twirled_circuit: The new circuit being built.
        rgen: Random number generator for twirling decisions.
        registers: Tracking registers for Z, π-rotation state, etc.
        enforce_readout_twirling_basis: If set ('X' or 'I'), overrides twirling_prob
            to enforce a specific readout twirling outcome on the last SQG.
        twirling_prob: Probability of inserting a twirling π rotation. Defaults to 0.5.
        fix_axes: If True, the twirling π rotation axis is fixed to match the gate's
            phase rather than being uniformly random. Defaults to False.

    """
    theta = float(operation.args["angle"])
    phi = float(operation.args["phase"])
    qubit = operation.locus[0]

    theta, phi = take_care_of_pending_operations(registers, qubit, theta, phi)

    # Potential override to enforce readout twirling state
    if enforce_readout_twirling_basis is not None:
        if enforce_readout_twirling_basis == "X":
            twirling_prob = 1.0
        else:
            twirling_prob = 0.0

    # Proceed with (potential) twirling
    if registers.extra_pi.get(qubit, 0) % 2 != 0:
        # If we have an odd number of pi rotations, we flip the twirling probability
        twirling_prob = 1.0 - twirling_prob

    # Perform the twirling
    if rgen.random() < twirling_prob:
        if not fix_axes:
            xi = float(rgen.random() * 2 * np.pi)
        else:
            xi = phi

        theta = insert_twirling_pi_rotation(registers, qubit, theta, phi, xi)

    registers.already_used_sqg[qubit] = registers.already_used_sqg.get(qubit, 0) + 1

    # Append the PRX gate to the new circuit
    if np.abs(theta) > _ANGLE_TOLERANCE:
        new_operation = CircuitOperation(
            name="prx",
            locus=operation.locus,
            args={"angle": theta, "phase": phi},
            implementation=operation.implementation,
        )
        twirled_circuit.append_operation(new_operation)


def process_cz_gate(
    operation: CircuitOperation,
    new_circuit: TwirledCircuit,
    registers: TrackingRegisters,
) -> None:
    """Process a CZ gate: propagate pending π-rotation parity as Z kicks.

    A CZ gate commutes with Z rotations but cojugates a R(π, φ) rotations as::

        --[R(π, φ) ⊗ I]--[CZ]-- = --[CZ]--[R(π, φ) ⊗ Z]--

    So, if there is a pending π-rotation on one qubit, a Z rotation of π is added to
    the *other* qubit.

    Args:
        operation: The CZ CircuitOperation to process.
        new_circuit: The new circuit being built.
        registers: Tracking registers for Z and π-rotation state.

    """
    new_circuit.append_operation(operation)

    # Propagate the additional Z rotations
    qubits = operation.locus
    if registers.extra_pi.get(qubits[0], 0) % 2 == 1:
        registers.virtual_z[qubits[1]] = registers.virtual_z.get(qubits[1], 0) + np.pi
    if registers.extra_pi.get(qubits[1], 0) % 2 == 1:
        registers.virtual_z[qubits[0]] = registers.virtual_z.get(qubits[0], 0) + np.pi


def process_measure_gate(
    operation: CircuitOperation,
    new_circuit: TwirledCircuit,
    registers: TrackingRegisters,
    drop_final_rz: bool = True,
) -> None:
    """Process a measurement gate: record readout twirling character and append.

    Determines the net readout twirling character ('I' or 'X') from the parity
    of accumulated π rotations on the measured qubit, stores it in
    ``registers.rot_chars``, and appends the measurement operation.

    Args:
        operation: The measurement CircuitOperation to process.
        new_circuit: The new circuit being built.
        registers: Tracking registers for π-rotation state and readout characters.
        drop_final_rz: If True (default), any pending virtual-Z before measurement
            is silently dropped. If False, raises NotImplementedError.

    """
    if len(operation.locus) > 1:
        raise NotImplementedError("Single measurements operations defined on more than one qubits are not supported.")

    if not drop_final_rz:
        raise NotImplementedError("Leaving RZ rotations before measurement are not supported yet.")

    registers.rot_chars[operation.locus[0]] = "I" if registers.extra_pi.get(operation.locus[0], 0) % 2 == 0 else "X"

    new_circuit.append_operation(operation)


def _determine_twirling_params(
    operation: CircuitOperation,
    registers: TrackingRegisters,
    circuit: TwirledCircuit,
    readout_twirling: bool | dict[str, str],
    twirling_probabilities: list[float] | None,
    processed_sqg: int,
) -> tuple[str | None, float]:
    """Determine twirling state and probability for a gate.

    Args:
        operation: The operation being processed.
        registers: Circuit registers tracking state.
        circuit: The original circuit.
        readout_twirling: Readout twirling configuration.
        twirling_probabilities: Optional list of twirling probabilities.
        processed_sqg: Number of single-qubit gates processed so far.

    Returns:
        Tuple of (enforce_readout_twirling_basis, twirling_prob).

    """
    qubit = operation.locus[0]

    twirling_prob = twirling_probabilities[processed_sqg] if twirling_probabilities is not None else 0.5

    enforce_readout_twirling_basis = None

    # Is this the last SQG on the qubit?
    if registers.already_used_sqg.get(qubit, 0) + 1 == circuit.sqg_counter.get(qubit, 0):
        if isinstance(readout_twirling, dict):
            enforce_readout_twirling_basis = readout_twirling.get(qubit, None)
        elif readout_twirling is False:
            enforce_readout_twirling_basis = "I"
        # Note: if readout_twirling is True, enforce_readout_twirling_basis remains None
        # (which is fine, no override is done in this case)

    return enforce_readout_twirling_basis, twirling_prob


def _add_twirling_gate_if_needed(
    operation: CircuitOperation,
    twirled_circuit: TwirledCircuit,
    registers: TrackingRegisters,
    readout_twirling: bool | dict[str, str],
    measured_qubits_without_sqg: list[str],
) -> None:
    """Add SQG before measurement for readout twirling if needed.

    Args:
        operation: The measurement operation.
        twirled_circuit: The circuit being built.
        registers: Circuit registers tracking state.
        readout_twirling: Readout twirling configuration.
        measured_qubits_without_sqg: List of measured qubits without single-qubit gates.

    """
    qubit = operation.locus[0]

    if (
        isinstance(readout_twirling, dict)
        and qubit in measured_qubits_without_sqg
        and readout_twirling.get(qubit, None) == "X"
    ):
        new_operation = CircuitOperation(
            name="prx",
            locus=(qubit,),
            args={"angle": np.pi, "phase": 0.0},
            implementation=None,
        )
        registers.extra_pi[qubit] = registers.extra_pi.get(qubit, 0) + 1
        twirled_circuit.append_operation(new_operation)


def _phrase_twirling_probabilities(
    tot_sqg_gates: int, twirling_probabilities: Iterable[float] | float | None = None
) -> list[float] | None:
    """Validate and normalize twirling probabilities for single-qubit gates.

    Args:
        tot_sqg_gates: Total number of single-qubit gates in the circuit.
        twirling_probabilities: Twirling probability per gate. Can be:
            - None: Returns None (no twirling probabilities)
            - float: Broadcast to all gates
            - Iterable[float]: Per-gate probabilities (length must match tot_sqg_gates)

    Returns:
        List of probabilities matching tot_sqg_gates length, or None.

    Raises:
        ValueError: If iterable length mismatches tot_sqg_gates or contains non-numeric values.
        TypeError: If input type is invalid.

    """
    if twirling_probabilities is None:
        return None

    # Handle scalar: broadcast to all gates
    if isinstance(twirling_probabilities, (int, float)):
        return [float(twirling_probabilities)] * tot_sqg_gates

    # Handle iterable: validate and convert to list of floats
    try:
        prob_list = list(twirling_probabilities)
    except TypeError:
        raise TypeError("twirling_probabilities must be None, a numeric value, or an iterable of floats") from None

    # Validate length
    if len(prob_list) != tot_sqg_gates:
        raise ValueError(
            f"Length of twirling_probabilities ({len(prob_list)}) "
            f"does not match number of single-qubit gates ({tot_sqg_gates})"
        )

    # Validate each element is numeric and convert to float
    try:
        return [float(p) for p in prob_list]
    except (TypeError, ValueError) as exc:
        raise ValueError("All elements in twirling_probabilities must be numeric") from exc


def randomize_circuit(
    circuit: TwirledCircuit,
    rgen: Generator,
    drop_final_rz: bool = True,
    readout_twirling: bool | dict[str, str] = False,
    twirling_probabilities: Iterable[float] | float | None = None,
    fix_axes: bool = False,
) -> TwirledCircuit:
    """Randomize a quantum circuit using Poor Man's Pauli Twirling (PMPT).

    Produces a new circuit that is logically equivalent to the input (up to
    readout twirling characters recorded in ``TwirledCircuit.rot_dict``),
    but with the θ angle of (arbitrarily/randomly) selected existing single-qubit R gates
    increased by π. The necessary bookkeeping for tracking the changes required downstream
    to maintain logical equivalence is handled by the `TrackingRegisters` class internally,
    so that no extra physical gates are added to the circuit.

    **Algorithm overview** (gate-by-gate processing):

    1. **PRX (single-qubit R gate)** — handled by `process_r_gate`:

       a. Absorb any pending virtual-Z and π rotations (stored in the `TrackingRegisters`)
          into the gate parameters (see `take_care_of_pending_operations`,
          `absorb_pending_pi_rotation`).
       b. With probability ``twirling_prob``, insert a π rotation (with random φ angle) after
          the gate and combine the two together (see `insert_twirling_pi_rotation`).
          On the last SQG of each qubit, the probability may be overridden to enforce a desired
          readout twirling basis.

    2. **CZ (two-qubit gate)** — handled by `process_cz_gate`:
       Propagate (pending) operations through the CZ gate, updating the `TrackingRegisters`.
    3. **Measure** — handled by `process_measure_gate`:
       Record the net readout twirling basis ('I' or 'X') from the parity
       of accumulated π rotations. Pending virtual-Z rotations are dropped. If the measured qubit
       has no SQG and readout twirling enforces 'X', a final PRX(π, 0) is added before measurement
       to ensure the correct readout twirling basis.
    4. **Barrier / Delay** — passed through unchanged.

    The recorded readout twirling characters can later be used by
    ``untwirl_counts`` to recover the original measurement statistics.

    .. note::

       Right now, there is no support for MOVE gates. If the input circuit contains MOVE operations,
       a ``ValueError`` will be raised.


    Args:
        circuit: The quantum circuit to randomize.
        rgen: Random number generator for twirling decisions.
        drop_final_rz: If True, drop final virtual-Z rotations before measurements.
        readout_twirling: Controls readout twirling behavior:
            - ``False``: no readout twirling (last SQG forced to even parity).
            - ``True``: free twirling (no constraint on last SQG).
            - ``dict``: maps qubit names to 'I' or 'X' to enforce a specific basis.
        twirling_probabilities: Probability of inserting a π rotation per SQG.
            Can be a single float (broadcast), a per-gate iterable, or None (default 0.5).
        fix_axes: If True, twirling π rotations use the same phase as the gate
            instead of a uniformly random phase.

    Returns:
        A new TwirledCircuit with randomized gates and ``rot_dict`` set to
        the per-qubit readout twirling characters.

    Raises:
        ValueError: If twirling_probabilities length doesn't match the number of
            single-qubit gates, or if an unsupported operation is encountered.
        TypeError: If readout_twirling has an invalid type.

    """
    # Verify that twirling_probabilities are valid
    tot_sqg_gates = sum(circuit.sqg_counter.values())
    if tot_sqg_gates == 0:
        warnings.warn(
            "PMPT (Poor Man's Pauli Twirling) is applied only to single-qubit gates, but the "
            "circuit contains none. No circuit twirling will be performed.",
            UserWarning,
            stacklevel=2,
        )
    twirling_probabilities_list = _phrase_twirling_probabilities(tot_sqg_gates, twirling_probabilities)

    # Validate readout_twirling type
    if not isinstance(readout_twirling, (bool, dict)):
        raise TypeError(f"readout_twirling must be bool or Dict[str, str], got {type(readout_twirling).__name__}")

    # Initialize the circuit registers
    registers = TrackingRegisters()

    # Create a new circuit
    twirled_circuit = TwirledCircuit([])

    # Check for the existence of measured qubits without SQG
    measured_qubits_without_sqg = [
        meas_qubit for meas_qubit in circuit.measured_qubits if circuit.sqg_counter.get(meas_qubit, 0) == 0
    ]

    # Process each instruction
    processed_sqg = 0
    for operation in circuit.operations:
        if operation.name == "prx":
            enforce_readout_twirling_basis, prob = _determine_twirling_params(
                operation,
                registers,
                circuit,
                readout_twirling,
                twirling_probabilities_list,
                processed_sqg,
            )
            process_r_gate(
                operation,
                twirled_circuit,
                rgen,
                registers,
                enforce_readout_twirling_basis,
                prob,
                fix_axes=fix_axes,
            )
            processed_sqg += 1

        elif operation.name == "cz":
            process_cz_gate(operation, twirled_circuit, registers)

        elif operation.name == "measure":
            _add_twirling_gate_if_needed(
                operation,
                twirled_circuit,
                registers,
                readout_twirling,
                measured_qubits_without_sqg,
            )
            process_measure_gate(operation, twirled_circuit, registers, drop_final_rz)

        elif operation.name in ["barrier", "delay"]:
            twirled_circuit.append_operation(operation)

        else:
            raise ValueError(f"Unsupported instruction type: {operation.name}")

    twirled_circuit.rot_dict = registers.rot_chars

    return twirled_circuit
