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

"""Pulse-level circuit surgery for noise amplification: global folding and operation inversion."""

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import replace

from iqm.error_reduction_tools.utils.circuit_utils import TwirledCircuit
import numpy as np

from iqm.pulse import Circuit as PulseCircuit
from iqm.pulse import CircuitOperation

_SELF_INVERSE_PULSE_OPS = frozenset({"cz", "barrier"})
"""Native IQM Pulse operations that are their own inverse."""

ACHIEVED_FOLDS_KEY = "achieved_folds"
"""Metadata key holding the number of folds of a circuit.

``s`` folds give a noise factor of ``1 + 2s``, which is what the extrapolation is fitted against.
"""

_TERMINAL_PULSE_OPS = frozenset({"measure"})
"""Native IQM Pulse operations that must stay at the end of a circuit and are never folded."""


def compact_factor(value: object) -> int | float:
    """Normalize a noise factor, narrowing a whole number to an ``int``.

    Args:
        value: The factor, as an ``int`` or a ``float``.

    Returns:
        ``int`` for ``2.0``, ``float`` for ``1.5``.

    Raises:
        TypeError: If the value is not a real number.
        ValueError: If the value is not finite.

    """
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"A noise factor must be a real number, got {value!r} of type {type(value).__name__}.")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"A noise factor must be finite, got {value!r}.")
    return int(number) if number.is_integer() else number


def as_fold_count(value: object) -> int:
    """Normalize a fold count: how many times ``U^dagger U`` is inserted.

    Args:
        value: The requested number of folds.

    Returns:
        The count as an ``int`` of at least 0.

    Raises:
        TypeError: If the value is not a real number.
        ValueError: If the value is not a whole number of at least 0.

    """
    count = compact_factor(value)
    if not isinstance(count, int) or count < 0:
        raise ValueError(f"A fold count must be a whole number of at least 0, got {value!r}.")
    return count


def noise_factor_for(folds: int) -> int:
    """Return the noise factor of a circuit folded ``folds`` times.

    Args:
        folds: How many times ``U^dagger U`` was inserted.

    Returns:
        The noise factor the folding realizes.

    """
    return 1 + 2 * folds


def invert_pulse_operation(operation: CircuitOperation) -> CircuitOperation:
    """Invert a single native IQM Pulse operation.

    The IQM native gate set inverts elementwise without any recompilation: ``cz`` is its own
    inverse, and ``prx`` is inverted by negating its rotation angle while keeping its phase.

    Args:
        operation: The native operation to invert.

    Returns:
        The inverted operation. The input operation is not modified.

    Raises:
        ValueError: If the operation has no well-defined inverse (e.g. ``measure``).

    """
    if operation.name == "prx":
        args = dict(operation.args)
        args["angle"] = -args["angle"]
        return replace(operation, args=args)

    if operation.name in _SELF_INVERSE_PULSE_OPS:
        return deepcopy(operation)

    raise ValueError(
        f"Cannot invert native operation '{operation.name}': it has no well-defined inverse. "
        f"Invertible operations are 'prx' and {sorted(_SELF_INVERSE_PULSE_OPS)}."
    )


def invert_pulse_operations(operations: Sequence[CircuitOperation]) -> list[CircuitOperation]:
    """Invert a sequence of native IQM Pulse operations.

    Args:
        operations: The native operations to invert, in circuit order.

    Returns:
        The inverted sequence. The input operations are not modified.

    Raises:
        ValueError: If any operation has no well-defined inverse.

    """
    return [invert_pulse_operation(operation) for operation in reversed(operations)]


def fold_global_pulse(
    target_circuit: PulseCircuit | TwirledCircuit,
    folds: int,
    *,
    name: str | None = None,
) -> PulseCircuit | TwirledCircuit:
    """Perform global folding directly on an IQM Pulse circuit.

    Implements the map ``U -> U (U^dagger U)^s`` on the native IQM Pulse representation, so every
    gate is executed ``1 + 2s`` times.

    The circuit must already be transpiled to the IQM native gate set, with every qubit measured after its last
    gate. The measurements are excluded from the folded body and appended unchanged at the end.

    Args:
        target_circuit: The original noisy circuit to be amplified, as an
            :class:`~iqm.pulse.Circuit` or a
            :class:`~iqm.error_reduction_tools.utils.circuit_utils.TwirledCircuit`.
        folds: How many times to insert ``U^dagger U``. ``0`` returns a copy of the input
            circuit. ``s`` folds give a noise factor of ``1 + 2s``.
        name: Name for the returned circuit. Defaults to the input circuit's name (when
            available) suffixed with the fold count.

    Returns:
        The noise amplified circuit, in the same representation as the input, with a
        :class:`~iqm.error_reduction_tools.utils.circuit_utils.TwirledCircuit` keeping its
        ``rot_dict`` so that a folded randomized circuit can still be untwirled. The original
        circuit is not modified.

    Raises:
        ValueError: If ``folds`` is not a whole number of at least 0, if the circuit contains
            a gate on a qubit after its measurement, if it has no foldable gates, or if it contains an operation
            with no well-defined inverse.

    """
    num_folds = as_fold_count(folds)

    operations, circuit_name = _pulse_operations(target_circuit)
    body, measurements = _split_measurements(operations)

    if not [operation for operation in body if operation.name != "barrier"]:
        raise ValueError("The target circuit contains no foldable gates.")

    # U (U^dagger U)^num_folds: whole circuits only, so every gate is repeated 1 + 2 * folds times.
    folded = list(deepcopy(body))
    for _ in range(num_folds):
        folded.extend(invert_pulse_operations(body))
        folded.extend(deepcopy(body))

    # Folding must not change what the circuit measures, nor in what order.
    folded.extend(deepcopy(measurements))

    if name is None:
        name = f"{circuit_name or 'circuit'}_folded_{num_folds}"

    metadata = {
        # Keep the metadata of the input.
        **(getattr(target_circuit, "metadata", None) or {}),
        ACHIEVED_FOLDS_KEY: num_folds,
    }

    if isinstance(target_circuit, TwirledCircuit):
        folded_circuit = TwirledCircuit(folded)
        folded_circuit.rot_dict = None if target_circuit.rot_dict is None else dict(target_circuit.rot_dict)
        folded_circuit.metadata = metadata
        return folded_circuit

    return PulseCircuit(name=name, instructions=tuple(folded), metadata=metadata)


def _pulse_operations(target_circuit: PulseCircuit | TwirledCircuit) -> tuple[list[CircuitOperation], str | None]:
    """Extract the native operations and the name of a pulse-level circuit."""
    if isinstance(target_circuit, PulseCircuit):
        return list(target_circuit.instructions), target_circuit.name
    if isinstance(target_circuit, TwirledCircuit):
        return list(target_circuit.operations), None
    raise TypeError(
        f"target_circuit must be an iqm.pulse.Circuit or a TwirledCircuit, got {type(target_circuit).__name__}."
    )


def _split_measurements(
    operations: Sequence[CircuitOperation],
) -> tuple[list[CircuitOperation], list[CircuitOperation]]:
    """Split operations into the foldable body and the measurements, which are moved to the end.

    A measurement may be followed by operations on other qubits, since it commutes with them.
    """
    body: list[CircuitOperation] = []
    measurements: list[CircuitOperation] = []
    measured: set[str] = set()
    for operation in operations:
        if operation.name in _TERMINAL_PULSE_OPS:
            measurements.append(operation)
            measured.update(operation.locus)
        elif measured.intersection(operation.locus) and operation.name != "barrier":
            raise ValueError(
                "Global folding requires all measurements to be terminal, but the circuit applies "
                f"{operation.name!r} to {sorted(measured.intersection(operation.locus))} after measuring them."
            )
        else:
            body.append(operation)
    return body, measurements
