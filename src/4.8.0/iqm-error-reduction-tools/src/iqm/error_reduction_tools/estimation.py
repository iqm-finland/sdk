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

"""Pauli term expectation values of circuit variants from twirled counts, and the result of an estimation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import numpy as np

from .circuits import CircuitKey, CircuitVariant, TargetCircuit
from .observables import ObservableSet
from .rem.rem_processors import ReadoutErrorMitigation, SampledDistribution
from .technique import TargetEstimate
from .twirling.twirling_processors import untwirl_and_sum_counts


@dataclass(frozen=True)
class EstimationResult:
    """The estimated observables of every target, with what they were computed from."""

    values: np.ndarray
    """Estimated expectation value of each observable of each target, shape ``(num_targets, num_observables)``."""

    unmitigated: np.ndarray
    """Expectation values measured on the target circuits themselves, shape ``(num_targets, num_observables)``.

    Readout error mitigation is included when it was applied to :attr:`values`.
    """

    sigma: np.ndarray | None
    """Bootstrap standard deviation of :attr:`values`, or ``None`` when no bootstrap samples were drawn."""

    term_values: tuple[np.ndarray, ...]
    """Per target, the measured value of each Pauli term of each variant, shape ``(num_variants, num_terms)``."""

    term_sigma: tuple[np.ndarray, ...] | None
    """Bootstrap standard deviation of :attr:`term_values`, or ``None`` when no bootstrap samples were drawn."""

    targets: tuple[TargetEstimate, ...]
    """Estimate and technique-specific diagnostics of each target."""

    readout_mitigated: bool
    """Whether readout error mitigation was applied."""


def variant_counts(variant: CircuitVariant, counts: Mapping[CircuitKey, Mapping[str, float]]) -> dict[str, float]:
    """Undo the readout twirling of each instance of a variant and sum their counts.

    Args:
        variant: A twirled variant.
        counts: Counts of every executed instance.

    Returns:
        The counts of the variant, as if it had been executed without readout twirling.

    Raises:
        KeyError: If the counts of an instance are missing.

    """
    instances = variant.instances()
    missing = [instance.key for instance in instances if instance.key not in counts]
    if missing:
        raise KeyError(f"No counts for instances {missing}.")
    return untwirl_and_sum_counts(
        [dict(counts[instance.key]) for instance in instances], [instance.rot_string for instance in instances]
    )


def measure_term_values(
    target: TargetCircuit,
    counts: Mapping[CircuitKey, Mapping[str, float]],
    observables: ObservableSet,
    mitigation: ReadoutErrorMitigation | None = None,
    readout_twirled: bool = False,
) -> np.ndarray:
    """Compute the expectation value of every Pauli term of every variant of a target.

    Args:
        target: A planned target.
        counts: Counts of every executed instance.
        observables: The observables, whose :attr:`~ObservableSet.pauli_terms` are evaluated.
        mitigation: Readout error mitigation to apply, or ``None`` for plain parities.
        readout_twirled: Whether the counts were readout-twirled, so that the symmetrized readout errors apply.

    Returns:
        Shape ``(num_variants, num_terms)``.

    """
    per_variant = [variant_counts(variant, counts) for variant in target.variants()]
    if observables.num_terms == 0:
        return np.zeros((len(per_variant), 0), dtype=float)
    if mitigation is None:
        return np.asarray([observables.term_values_from_counts(c) for c in per_variant], dtype=float)
    result = mitigation.mitigate_counts(
        # Untwirled counts are sums of integer counts, stored as floats.
        experiment_counts=cast("list[SampledDistribution]", per_variant),
        qubit_to_bit_mapping=dict(target.qubit_to_clbit),
        observables=list(observables.pauli_terms),
        twirled=readout_twirled,
    )
    return np.asarray(result["expectation_values"], dtype=float)


def resample_counts(
    counts: Mapping[CircuitKey, Mapping[str, float]], rng: np.random.Generator
) -> dict[CircuitKey, dict[str, float]]:
    """Draw one bootstrap sample of every counts distribution, keeping each total number of shots.

    Args:
        counts: Counts of every executed instance.
        rng: Random number generator.

    Returns:
        The resampled counts, with the same keys.

    Raises:
        ValueError: If a distribution has no shots.

    """
    resampled: dict[CircuitKey, dict[str, float]] = {}
    for key, distribution in counts.items():
        labels = list(distribution)
        observed = np.asarray([distribution[label] for label in labels], dtype=float)
        total = float(observed.sum())
        if total <= 0:
            raise ValueError(f"Cannot resample the counts of {key}, which have no shots.")
        drawn = rng.multinomial(round(total), observed / total)
        resampled[key] = {label: float(value) for label, value in zip(labels, drawn) if value}
    return resampled
