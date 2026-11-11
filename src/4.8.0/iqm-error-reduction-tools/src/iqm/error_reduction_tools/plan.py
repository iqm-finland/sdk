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

"""The execution plan: every circuit to execute, grouped into jobs, built from the targets and a technique."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING
import warnings

import numpy as np

from iqm.pulse import Circuit

from .circuits import CircuitKey, CircuitVariant, TargetCircuit, TwirlingInstance, measurement_map
from .readout_characterization.data_collection import create_calibration_circuits
from .readout_characterization.readout_error_characterization import RECConfiguration
from .twirling.twirling_api import TwirlingConfiguration, readout_rot_strings, rot_string_for, twirl_circuit
from .utils.circuit_utils import TwirledCircuit

DEFAULT_SHOTS = 20_000
"""Total shots of a variant when neither the workflow nor the technique sets them."""

if TYPE_CHECKING:
    from .observables import ObservableSet
    from .readout_characterization.topologies import QPUTopology
    from .technique import Technique


@dataclass(frozen=True)
class Batch:
    """Circuits executed as one job, all with the same number of shots."""

    shots: int
    """Shots of every circuit in the batch."""

    keys: tuple[CircuitKey, ...]
    """The twirling instances in the batch, in submission order."""


@dataclass(frozen=True)
class CalibrationPlan:
    """The readout error characterization circuits, executed as one job."""

    qubits: tuple[str, ...]
    """Characterized qubits. Qubit ``i`` is measured into classical bit ``i``."""

    circuits: tuple[Circuit, ...]
    """Circuits that prepare a computational basis state and measure it."""

    prep_strings: tuple[str, ...]
    """Prepared state of each circuit, ``"I"`` for 0 and ``"X"`` for 1 on each qubit."""

    shots: int
    """Shots of every circuit."""


@dataclass(frozen=True)
class ExecutionPlan:
    """Every circuit to execute, and how they are grouped into jobs."""

    targets: tuple[TargetCircuit, ...]
    """The targets, with their variants and twirling instances."""

    batches: tuple[Batch, ...]
    """The jobs. Every twirling instance is in exactly one batch."""

    calibration: CalibrationPlan | None
    """The readout error characterization job, or ``None`` when readout errors are not mitigated."""

    @property
    def readout_twirled(self) -> bool:
        """Return whether any instance flips a readout, which means readout twirling was applied."""
        return any("X" in instance.rot_string for target in self.targets for instance in target.instances())

    @property
    def num_circuits(self) -> int:
        """Return the number of executed circuits, calibration included."""
        calibration = 0 if self.calibration is None else len(self.calibration.circuits)
        return sum(len(batch.keys) for batch in self.batches) + calibration

    @property
    def total_shots(self) -> int:
        """Return the total number of shots, calibration included."""
        calibration = 0 if self.calibration is None else len(self.calibration.circuits) * self.calibration.shots
        return sum(len(batch.keys) * batch.shots for batch in self.batches) + calibration

    def instance(self, key: CircuitKey) -> TwirlingInstance:
        """Return the twirling instance with the given key.

        Args:
            key: The identity of the instance.

        Returns:
            The instance.

        """
        target_index, variant_index, instance_index = key
        return self.targets[target_index].variants()[variant_index].instances()[instance_index]

    def summary(self) -> str:
        """Describe the cost of the plan, per job.

        Returns:
            A few lines of text.

        """
        lines = [f"{len(self.targets)} target(s), {self.num_circuits} circuits, {self.total_shots} shots in total."]
        lines += [
            f"  job {index}: {len(batch.keys)} circuits x {batch.shots} shots"
            for index, batch in enumerate(self.batches)
        ]
        if self.calibration is not None:
            lines.append(
                f"  readout calibration: {len(self.calibration.circuits)} circuits x {self.calibration.shots} shots "
                f"on {len(self.calibration.qubits)} qubits"
            )
        return "\n".join(lines)


def build_plan(
    targets: Sequence[TargetCircuit],
    observables: ObservableSet,
    technique: Technique,
    *,
    shots: int | None,
    twirling: TwirlingConfiguration | None,
    rem: RECConfiguration | None,
    topology: QPUTopology | None,
) -> ExecutionPlan:
    """Build the variants of every target, fill in their budgets, twirl them, and group all circuits into jobs.

    A variant whose shots or twirling instances the technique left unset gets the workflow's. A variant that sets
    them keeps its own, with a warning when that overrides a value the user gave explicitly.

    Args:
        targets: The targets, without variants.
        observables: The observables to be estimated.
        technique: The error reduction technique that builds the variants.
        shots: Total shots of a variant, or ``None`` for :data:`DEFAULT_SHOTS`.
        twirling: Twirling configuration, or ``None`` to execute every variant once as it is.
        rem: Readout error characterization configuration, or ``None`` for no readout error mitigation.
        topology: QPU topology, required by the ``"LOCAL"`` readout-twirling strategy.

    Returns:
        The plan.

    Raises:
        ValueError: If the technique builds inconsistent variants, or the ``"LOCAL"`` strategy has no topology.

    """
    if twirling is not None and twirling.readout_twirl_strategy == "LOCAL" and topology is None:
        raise ValueError("The 'LOCAL' readout-twirling strategy needs the QPU topology, but none is available.")
    rng = np.random.default_rng(None if twirling is None else twirling.seed)
    requested_instances = None if twirling is None else twirling.num_twirling_instances

    planned: list[TargetCircuit] = []
    all_variants: list[CircuitVariant] = []
    # Plain loops, so that warnings raised while twirling point at the caller of plan().
    for target in targets:
        variants = technique.build_variants(target, observables)
        _check_variants(target, variants)
        all_variants.extend(variants)
        twirled: list[CircuitVariant] = []
        for variant in variants:
            budgeted = replace(
                variant,
                shots=variant.shots or shots or DEFAULT_SHOTS,
                num_instances=variant.num_instances or requested_instances or 1,
            )
            twirled.append(replace(budgeted, twirling_instances=_twirl(budgeted, twirling, topology, rng)))
        planned.append(replace(target, circuit_variants=tuple(twirled)))
    _warn_on_overrides(all_variants, shots, requested_instances)

    calibration = None if rem is None else _calibration_plan(planned, rem)
    return ExecutionPlan(targets=tuple(planned), batches=_batches(planned), calibration=calibration)


def _warn_on_overrides(variants: Sequence[CircuitVariant], shots: int | None, num_instances: int | None) -> None:
    """Warn once per budget when variants override a value the user set explicitly."""
    for name, given, owner, own in (
        ("shots", shots, "the workflow", [variant.shots for variant in variants]),
        ("num_twirling_instances", num_instances, "the twirling configuration", [v.num_instances for v in variants]),
    ):
        overriding = sorted({value for value in own if value is not None and given is not None and value != given})
        if overriding:
            count = sum(value in overriding for value in own)
            warnings.warn(
                f"{count} variant(s) set their own {name} {overriding}, which override {name}={given} of {owner}.",
                stacklevel=4,
            )


def _check_variants(target: TargetCircuit, variants: Sequence[CircuitVariant]) -> None:
    """Check that the variants of a target are indexed in order, start with the target, and measure alike."""
    if not variants or variants[0].circuit != target.circuit:
        raise ValueError(f"Variant 0 of target {target.target_index} must be the target circuit itself.")
    for index, variant in enumerate(variants):
        if (variant.target_index, variant.variant_index) != (target.target_index, index):
            raise ValueError(
                f"Variant {index} of target {target.target_index} is labelled "
                f"({variant.target_index}, {variant.variant_index})."
            )
        if measurement_map(variant.circuit) != target.qubit_to_clbit:
            raise ValueError(
                f"Variant {index} of target {target.target_index} measures {measurement_map(variant.circuit)}, "
                f"but the target measures {dict(target.qubit_to_clbit)}."
            )


def _twirl(
    variant: CircuitVariant,
    config: TwirlingConfiguration | None,
    topology: QPUTopology | None,
    rng: np.random.Generator,
) -> tuple[TwirlingInstance, ...]:
    """Build the twirling instances of one variant."""
    name = f"t{variant.target_index}_v{variant.variant_index}"
    num_instances = variant.num_instances or 1
    if config is None:
        if num_instances > 1:
            warnings.warn(
                f"Variant {variant.variant_index} of target {variant.target_index} asks for "
                f"{num_instances} twirling instances, but no twirling is configured. It is executed once.",
                stacklevel=4,
            )
        clbits = len(measurement_map(variant.circuit))
        circuit = replace(variant.circuit, name=f"{name}_i0")
        return (_instance(variant, 0, circuit, "I" * clbits),)

    working = TwirledCircuit(list(variant.circuit.instructions))
    rot_strings = readout_rot_strings(config.readout_twirl_strategy, working.measured_qubits, topology)
    randomized = twirl_circuit(working, rot_strings, num_instances, circuit_twirling=config.circuit_twirling, rng=rng)
    return tuple(
        _instance(variant, index, circuit.to_circuit(name=f"{name}_i{index}"), rot_string_for(circuit))
        for index, circuit in enumerate(randomized)
    )


def _instance(variant: CircuitVariant, index: int, circuit: Circuit, rot_string: str) -> TwirlingInstance:
    """Create one twirling instance of a variant."""
    return TwirlingInstance(
        target_index=variant.target_index,
        variant_index=variant.variant_index,
        instance_index=index,
        circuit=circuit,
        rot_string=rot_string,
    )


def _batches(targets: Sequence[TargetCircuit]) -> tuple[Batch, ...]:
    """Group all twirling instances into one job per shot count, in tree order."""
    by_shots: dict[int, list[CircuitKey]] = {}
    for target in targets:
        for variant in target.variants():
            keys = [instance.key for instance in variant.instances()]
            by_shots.setdefault(variant.shots_per_instance, []).extend(keys)
    return tuple(Batch(shots=shots, keys=tuple(keys)) for shots, keys in by_shots.items())


def _calibration_plan(targets: Sequence[TargetCircuit], config: RECConfiguration) -> CalibrationPlan:
    """Build the readout error characterization circuits for the measured qubits."""
    measured = {qubit for target in targets for qubit in target.qubit_to_clbit}
    qubits = tuple(config.qubits) if config.qubits is not None else tuple(sorted(measured))
    missing = sorted(measured - set(qubits))
    if missing:
        raise ValueError(f"The readout characterization does not cover the measured qubits {missing}.")
    shots = config.shots // config.num_circuits
    if shots < 1:
        raise ValueError(f"{config.shots} calibration shots cannot be split over {config.num_circuits} circuits.")
    circuits, prep_strings = create_calibration_circuits(
        list(qubits), config.num_circuits, config.symmetrize, config.seed, config.equatorial_randomization
    )
    return CalibrationPlan(qubits=qubits, circuits=tuple(circuits), prep_strings=tuple(prep_strings), shots=shots)
