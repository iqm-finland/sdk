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

"""Clifford data regression: fit measured against exact values of Clifford training circuits."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal
import warnings

import numpy as np

from .cdr_processors import cdr_fit_and_predict, r_squared
from .cdr_training_set import generate_clifford_training_set
from ..circuits import CircuitVariant, TargetCircuit
from ..technique import Estimation, TargetEstimate, Technique

if TYPE_CHECKING:
    from ..observables import ObservableSet

MIN_TRAINING_CIRCUITS = 2
"""Fewest training circuits a line can be fitted through."""


@dataclass(frozen=True, kw_only=True)
class TrainingVariant(CircuitVariant):
    """A Clifford training circuit of a target, with its exact expectation values."""

    exact_term_values: tuple[float, ...]
    """Exact expectation value of each Pauli term, ordered as the observables' ``pauli_terms``."""


@dataclass(frozen=True)
class CDR(Technique):
    """Configuration of Clifford data regression: which training circuits are executed, and with what budget.

    The shot and instance fields, when set, override the workflow's shots and twirling instances for the target or
    the training circuits.
    """

    num_training_circuits: int = 30
    """Number of Clifford training circuits. Fewer are built if there are fewer distinct candidate circuits."""

    num_target_instances: int | None = None
    """Twirling instances of the target circuit, or ``None`` for the workflow's default."""

    num_training_instances: int | None = None
    """Twirling instances of each training circuit, or ``None`` for the workflow's default."""

    target_shots: int | None = None
    """Total shots of the target circuit, or ``None`` for the workflow's default."""

    training_shots: int | None = None
    """Total shots of each training circuit, or ``None`` for the workflow's default."""

    strategy: Literal["pool_fps", "optimize"] = "pool_fps"
    """How training circuits are chosen: farthest-point selection from a random pool, or optimization of the spread
    of one Pauli term."""

    pool_size: int | None = None
    """Number of candidates ``"pool_fps"`` selects from, or ``None`` for its default."""

    increase_combined_value_spread: bool = False
    """Whether training circuits are chosen to spread the exact values of the first observable."""

    primary_index: int = 0
    """Pauli term whose exact values ``"optimize"`` spreads."""

    identity_as_delay: bool = True
    """Whether an identity replacing a gate is a 20 ns delay rather than prx(0, 0)."""

    seed: int | None = 0
    """Seed of the training circuit construction."""

    strategy_options: Mapping[str, Any] = field(default_factory=dict)
    """Extra keyword arguments for
    :func:`~iqm.error_reduction_tools.cdr.cdr_training_set.generate_clifford_training_set`."""

    def __post_init__(self) -> None:
        """Reject budgets that cannot be executed or fitted."""
        if self.num_training_circuits < MIN_TRAINING_CIRCUITS:
            raise ValueError(
                f"num_training_circuits must be at least {MIN_TRAINING_CIRCUITS}, got {self.num_training_circuits}."
            )
        if self.strategy not in ("pool_fps", "optimize"):
            raise ValueError(f"Unknown strategy {self.strategy!r}. Expected 'pool_fps' or 'optimize'.")
        for name in ("num_target_instances", "num_training_instances", "target_shots", "training_shots"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be at least 1, got {value}.")

    def build_variants(self, target: TargetCircuit, observables: ObservableSet) -> tuple[CircuitVariant, ...]:
        """Build the target variant and the Clifford training circuits, with their exact values.

        Args:
            target: The target, without variants.
            observables: The observables, whose Pauli terms the exact values are computed for.

        Returns:
            The target itself as variant 0, followed by one :class:`TrainingVariant` per training circuit.

        Raises:
            ValueError: If no training circuit could be built, or the observables have no Pauli terms.

        """
        if observables.num_terms == 0:
            raise ValueError("The observables have no Pauli terms besides the identity, so there is nothing to learn.")
        options: dict[str, Any] = {
            "strategy": self.strategy,
            "primary_index": self.primary_index,
            "rng_seed": self.seed,
            "identity_as_delay": self.identity_as_delay,
            "progress": False,
            "build_stim_circuits": False,
            **dict(self.strategy_options),
        }
        if self.pool_size is not None:
            options["pool_size"] = self.pool_size
        if self.increase_combined_value_spread:
            options["coefficients"] = observables.coefficients[0].tolist()
        training_set = generate_clifford_training_set(
            target.circuit, list(observables.pauli_terms), self.num_training_circuits, **options
        )
        exact = np.atleast_2d(np.asarray(training_set.exact_values, dtype=float))
        circuits = list(training_set.pulse_circuits)
        if not circuits:
            raise ValueError("No training circuits could be built. Check that the target has non-Clifford gates.")
        _warn_if_short(len(circuits), self.num_training_circuits)

        target_variant = CircuitVariant(
            target_index=target.target_index,
            variant_index=0,
            circuit=target.circuit,
            shots=self.target_shots,
            num_instances=self.num_target_instances,
        )
        training = tuple(
            TrainingVariant(
                target_index=target.target_index,
                variant_index=index,
                circuit=circuit,
                shots=self.training_shots,
                num_instances=self.num_training_instances,
                exact_term_values=tuple(float(value) for value in exact[index - 1]),
            )
            for index, circuit in enumerate(circuits, start=1)
        )
        return (target_variant, *training)


@dataclass(frozen=True)
class CDRDiagnostics:
    """How the regression of one target was obtained, per fitted column."""

    combine: Literal["per_term", "group"]
    """Whether the columns are Pauli terms or observables."""

    noisy_train: np.ndarray
    """Measured value of each column of each training circuit, shape ``(num_training, num_columns)``."""

    exact_train: np.ndarray
    """Exact value of each column of each training circuit, shape ``(num_training, num_columns)``."""

    noisy_target: np.ndarray
    """Measured value of each column of the target, shape ``(num_columns,)``."""

    slopes: np.ndarray
    """Slope of the fitted line of each column, shape ``(num_columns,)``."""

    intercepts: np.ndarray
    """Intercept of the fitted line of each column, shape ``(num_columns,)``."""

    r_squared: np.ndarray
    """Coefficient of determination of each fit, shape ``(num_columns,)``."""

    @property
    def num_training(self) -> int:
        """Return the number of training circuits."""
        return int(self.noisy_train.shape[0])

    @property
    def predicted(self) -> np.ndarray:
        """Return the corrected value of each column of the target, shape ``(num_columns,)``."""
        return self.slopes * self.noisy_target + self.intercepts


@dataclass(frozen=True)
class CDREstimation(Estimation):
    """Regression settings of Clifford data regression. Can be changed after execution.

    A line ``exact = a * measured + b`` is fitted through the training circuits and evaluated at the measured value of
    the target.
    """

    combine: Literal["per_term", "group"] = "group"
    """``"per_term"`` fits one line per Pauli term and then combines the terms. ``"group"`` combines the terms into
    the observables first and fits one line per observable."""

    min_training_spread: float = 0.05
    """Warn when the measured training values of a column span less than this, since the slope is then
    ill-determined. ``0`` disables the check."""

    warn_on_extrapolation: bool = True
    """Warn when the measured target value lies outside the range of the measured training values."""

    def __post_init__(self) -> None:
        """Reject unknown settings early."""
        if self.combine not in ("per_term", "group"):
            raise ValueError(f"Unknown combine {self.combine!r}. Expected 'per_term' or 'group'.")
        if self.min_training_spread < 0:
            raise ValueError(f"min_training_spread must be at least 0, got {self.min_training_spread}.")

    def accepts(self, technique: Technique) -> bool:
        """Return whether the technique is :class:`CDR`."""
        return isinstance(technique, CDR)

    def estimate(self, target: TargetCircuit, term_values: np.ndarray, observables: ObservableSet) -> TargetEstimate:
        """Fit the training circuits and correct the target.

        Args:
            target: The planned target, whose variants 1 and up are :class:`TrainingVariant`.
            term_values: Measured value of each Pauli term of each variant, shape ``(num_variants, num_terms)``.
            observables: The observables.

        Returns:
            The estimate, with :class:`CDRDiagnostics`.

        Raises:
            ValueError: If there are fewer than two training circuits.

        """
        training = target.variants()[1:]
        if len(training) < MIN_TRAINING_CIRCUITS:
            raise ValueError(f"At least {MIN_TRAINING_CIRCUITS} training circuits are needed, got {len(training)}.")
        exact = np.asarray([_exact_values(variant) for variant in training], dtype=float)
        measured = np.asarray(term_values, dtype=float)
        noisy_target, noisy_train = measured[0], measured[1:]
        if self.combine == "group":
            noisy_target, noisy_train, exact = (observables.combine(a) for a in (noisy_target, noisy_train, exact))

        _warn_on_ill_posed_fit(noisy_train, noisy_target, self, target.target_index)
        predicted, coefficients = cdr_fit_and_predict(
            exact, noisy_train, noisy_target, mode="per_observable", return_coeffs=True
        )
        quality = [r_squared(noisy_train[:, column], exact[:, column])[2] for column in range(exact.shape[1])]
        diagnostics = CDRDiagnostics(
            combine=self.combine,
            noisy_train=noisy_train,
            exact_train=exact,
            noisy_target=noisy_target,
            slopes=np.asarray([a for a, _ in coefficients], dtype=float),
            intercepts=np.asarray([b for _, b in coefficients], dtype=float),
            r_squared=np.asarray(quality, dtype=float),
        )
        predicted = np.atleast_1d(np.asarray(predicted, dtype=float))
        values = predicted if self.combine == "group" else observables.combine(predicted)
        return TargetEstimate(values=values, diagnostics=diagnostics)


def _exact_values(variant: CircuitVariant) -> tuple[float, ...]:
    """Read the exact values of a training variant built by :class:`CDR`."""
    if not isinstance(variant, TrainingVariant):
        raise TypeError(f"Variant {variant.variant_index} is a {type(variant).__name__}, not a TrainingVariant.")
    return variant.exact_term_values


def _warn_if_short(produced: int, requested: int) -> None:
    """Warn when fewer training circuits were built than requested."""
    if produced < requested:
        warnings.warn(
            f"Only {produced} of the requested {requested} training circuits could be built, because the candidate "
            "pool is too small.",
            stacklevel=5,
        )


def _warn_on_ill_posed_fit(
    noisy_train: np.ndarray, noisy_target: np.ndarray, config: CDREstimation, target_index: int
) -> None:
    """Warn when the training values are too flat to fit, or the target lies outside them."""
    kind = "term" if config.combine == "per_term" else "observable"
    spreads = np.ptp(noisy_train, axis=0)
    flat = [column for column, spread in enumerate(spreads) if spread < config.min_training_spread]
    if flat:
        warnings.warn(
            f"Target {target_index}: the measured training values of {kind}(s) {flat} span less than "
            f"{config.min_training_spread:g}, so the slope of the fit is ill-determined.",
            stacklevel=5,
        )
    low, high = noisy_train.min(axis=0), noisy_train.max(axis=0)
    outside = [column for column in range(noisy_target.size) if not low[column] <= noisy_target[column] <= high[column]]
    if config.warn_on_extrapolation and outside:
        warnings.warn(
            f"Target {target_index}: the measured target value of {kind}(s) {outside} lies outside the range of the "
            "training circuits, so the estimate is an extrapolation.",
            stacklevel=5,
        )
