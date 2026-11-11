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

"""Target circuits, their variants and twirling instances, each carrying the indices that identify it."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
import math
from typing import NamedTuple

from iqm.pulse import Circuit

from .utils.circuit_utils import TwirledCircuit


class CircuitKey(NamedTuple):
    """Identity of one executed circuit: which target, which variant of it, which twirling instance."""

    target_index: int
    """Position of the target in the workflow's list of targets."""

    variant_index: int
    """Position of the variant among the variants of its target."""

    instance_index: int
    """Position of the twirling instance among the instances of its variant."""


@dataclass(frozen=True, kw_only=True)
class TwirlingInstance:
    """One randomized copy of a circuit variant, the unit that is executed on the quantum computer."""

    target_index: int
    """Index of the target this instance belongs to."""

    variant_index: int
    """Index of the variant this instance belongs to."""

    instance_index: int
    """Index of this instance among the instances of its variant."""

    circuit: Circuit
    """The randomized circuit that is executed."""

    rot_string: str
    """Readout-twirling rotation, one ``"I"`` or ``"X"`` per classical bit in ascending order.

    An ``"X"`` means the outcome of that bit is flipped by the circuit and must be flipped back.
    """

    @property
    def key(self) -> CircuitKey:
        """Return the identity of this instance."""
        return CircuitKey(self.target_index, self.variant_index, self.instance_index)


@dataclass(frozen=True, kw_only=True)
class CircuitVariant:
    """A circuit derived from a target by an error reduction technique, e.g. a noise-amplified copy.

    Variant 0 of every target is the target circuit itself. Techniques that need data per variant subclass this.
    """

    target_index: int
    """Index of the target this variant belongs to."""

    variant_index: int
    """Index of this variant among the variants of its target."""

    circuit: Circuit
    """The variant circuit, before twirling."""

    shots: int | None = None
    """Total shots for this variant, split evenly over its twirling instances. ``None`` until the workflow is
    planned, unless the technique sets it, in which case it overrides the workflow's shots."""

    num_instances: int | None = None
    """Requested number of twirling instances. ``None`` until the workflow is planned, unless the technique sets it,
    in which case it overrides the twirling configuration. The actual number can differ, see
    :attr:`twirling_instances`."""

    twirling_instances: tuple[TwirlingInstance, ...] | None = None
    """The executed instances, or ``None`` before the workflow has been planned."""

    def __post_init__(self) -> None:
        """Reject budgets that cannot be executed."""
        if self.shots is not None and self.shots < 1:
            raise ValueError(f"A variant needs at least 1 shot, got {self.shots}.")
        if self.num_instances is not None and self.num_instances < 1:
            raise ValueError(f"A variant needs at least 1 twirling instance, got {self.num_instances}.")

    @property
    def shots_per_instance(self) -> int:
        """Return the shots each twirling instance is executed with.

        Raises:
            RuntimeError: If the variant has not been planned yet.

        """
        instances = self.instances()
        if self.shots is None:
            raise RuntimeError(f"Variant {self.variant_index} of target {self.target_index} has no shots yet.")
        return math.ceil(self.shots / len(instances))

    def instances(self) -> tuple[TwirlingInstance, ...]:
        """Return the twirling instances.

        Raises:
            RuntimeError: If the variant has not been twirled yet.

        """
        if self.twirling_instances is None:
            raise RuntimeError(f"Variant {self.variant_index} of target {self.target_index} has not been twirled.")
        return self.twirling_instances


@dataclass(frozen=True, kw_only=True)
class TargetCircuit:
    """A circuit whose observables are to be estimated, together with its variants once planned."""

    target_index: int
    """Position of this target in the workflow's list of targets."""

    circuit: Circuit
    """The target circuit, in the native gate set and with terminal measurements."""

    circuit_variants: tuple[CircuitVariant, ...] | None = None
    """The variants built by the technique, or ``None`` before the workflow has been planned."""

    qubit_to_clbit: Mapping[str, int] = field(init=False)
    """Measured qubit name to the classical bit its outcome lands in, read from the measurement keys."""

    def __post_init__(self) -> None:
        """Validate the measurements and derive :attr:`qubit_to_clbit` from them."""
        object.__setattr__(self, "qubit_to_clbit", measurement_map(self.circuit))

    @property
    def num_clbits(self) -> int:
        """Return the number of measured classical bits."""
        return len(self.qubit_to_clbit)

    @property
    def measured_qubits(self) -> tuple[str, ...]:
        """Return the measured qubit names, ordered by classical bit."""
        return tuple(sorted(self.qubit_to_clbit, key=lambda qubit: self.qubit_to_clbit[qubit]))

    def variants(self) -> tuple[CircuitVariant, ...]:
        """Return the variants.

        Raises:
            RuntimeError: If the workflow has not been planned yet.

        """
        if self.circuit_variants is None:
            raise RuntimeError(f"Target {self.target_index} has no variants. Plan the workflow first.")
        return self.circuit_variants

    def instances(self) -> Iterator[TwirlingInstance]:
        """Iterate over all twirling instances of all variants, in variant order.

        Raises:
            RuntimeError: If the workflow has not been planned yet.

        """
        for variant in self.variants():
            yield from variant.instances()


def measurement_map(circuit: Circuit) -> dict[str, int]:
    """Map each measured qubit to its classical bit, after checking the measurements can be processed.

    Args:
        circuit: A circuit in the native gate set.

    Returns:
        Measured qubit name to classical bit index.

    Raises:
        ValueError: If the circuit measures nothing, acts on a qubit after measuring it, or uses classical bits that
            are not ``0, ..., n - 1``.

    """
    measured: set[str] = set()
    for operation in circuit.instructions:
        acted_on = measured.intersection(operation.locus)
        if acted_on and operation.name != "barrier":
            raise ValueError(
                f"Circuit {circuit.name!r} applies {operation.name!r} to {sorted(acted_on)} after measuring them. "
                "Every qubit must be measured once, after its last gate."
            )
        if operation.name == "measure":
            measured.update(operation.locus)
    if not measured:
        raise ValueError(f"Circuit {circuit.name!r} has no measurements.")
    mapping = TwirledCircuit(list(circuit.instructions)).qubit_to_clbit()
    if sorted(mapping.values()) != list(range(len(mapping))):
        raise ValueError(
            f"Circuit {circuit.name!r} writes to classical bits {sorted(mapping.values())}. "
            f"The {len(mapping)} measurements must use the classical bits 0 to {len(mapping) - 1}."
        )
    return mapping
