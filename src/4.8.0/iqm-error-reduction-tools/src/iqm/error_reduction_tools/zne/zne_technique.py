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

"""Zero-noise extrapolation: amplify the noise by global folding, then extrapolate to zero noise."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal
import warnings

import numpy as np

from iqm.pulse import Circuit

from .zne_modifiers import fold_global_pulse
from .zne_processors import extrapolate_at_zero
from ..circuits import CircuitVariant, TargetCircuit
from ..technique import Estimation, TargetEstimate, Technique

if TYPE_CHECKING:
    from ..observables import ObservableSet

MIN_FIT_POINTS = 2
"""Fewest noise factors an extrapolation can be fitted through."""


@dataclass(frozen=True, kw_only=True)
class NoiseAmplifiedVariant(CircuitVariant):
    """A target circuit with its noise amplified by global folding."""

    noise_factor: int
    """How many times every gate of the target is executed, ``1 + 2 * folds``."""


@dataclass(frozen=True)
class ZNE(Technique):
    """Configuration of zero-noise extrapolation: which noise factors are executed.

    The noise of the target is amplified by global folding, ``U -> U (U^dagger U)^s``, which gives the odd noise
    factor ``1 + 2s``.
    """

    noise_factors: tuple[int, ...] = (1, 3, 5)
    """Noise factors to execute. Odd integers in increasing order, starting at 1."""

    def __post_init__(self) -> None:
        """Validate the noise factors."""
        factors = tuple(self.noise_factors)
        if len(factors) < MIN_FIT_POINTS:
            raise ValueError(f"At least {MIN_FIT_POINTS} noise factors are needed for a fit, got {factors}.")
        if any(isinstance(f, bool) or not isinstance(f, (int, np.integer)) or f < 1 or f % 2 == 0 for f in factors):
            raise ValueError(f"Global folding only realizes odd integer noise factors, got {factors}.")
        if factors[0] != 1 or list(factors) != sorted(set(factors)):
            raise ValueError(f"Noise factors must start at 1 and strictly increase, got {factors}.")
        object.__setattr__(self, "noise_factors", tuple(int(f) for f in factors))

    def build_variants(self, target: TargetCircuit, observables: ObservableSet) -> tuple[NoiseAmplifiedVariant, ...]:
        """Fold the target once per noise factor.

        Args:
            target: The target, without variants.
            observables: The observables. Folding does not depend on them.

        Returns:
            One variant per noise factor, variant 0 being the target itself.

        """
        return tuple(
            NoiseAmplifiedVariant(
                target_index=target.target_index,
                variant_index=index,
                circuit=_amplified(target.circuit, factor),
                noise_factor=factor,
            )
            for index, factor in enumerate(self.noise_factors)
        )


@dataclass(frozen=True)
class ZNEDiagnostics:
    """How the extrapolation of one target was obtained, per Pauli term."""

    variant_indices: tuple[int, ...]
    """Indices of the variants that were fitted."""

    noise_factors: np.ndarray
    """Noise factor of each fitted variant, shape ``(num_fitted,)``."""

    pauli_terms: tuple[str, ...]
    """Label of each Pauli term."""

    measured: np.ndarray
    """Measured value of each Pauli term at each fitted noise factor, shape ``(num_fitted, num_terms)``."""

    extrapolated: np.ndarray
    """Value of each Pauli term at zero noise, shape ``(num_terms,)``."""

    fit_parameters: tuple[tuple[float, ...], ...]
    """Fit parameters of each Pauli term, as :func:`~iqm.error_reduction_tools.zne.zne_processors.fit_curve` reads
    them."""

    extrapolation_type: Literal["Poly", "Exp"]
    """The fitted model."""

    degree: int | None
    """Polynomial degree, ``None`` for Richardson extrapolation or an exponential fit."""

    num_executed_factors: int
    """Number of executed noise factors, which can exceed the fitted ones."""


@dataclass(frozen=True)
class ZNEEstimation(Estimation):
    """Extrapolation settings of zero-noise extrapolation. Can be changed after execution.

    Every Pauli term is fitted separately against the noise factors, and the observables are combined from the values
    at zero noise.
    """

    extrapolation_type: Literal["Poly", "Exp"] = "Poly"
    """``"Poly"`` for a polynomial fit, ``"Exp"`` for ``a * exp(b * x)``."""

    degree: int | None = None
    """Degree of the polynomial. ``None`` uses one less than the number of fitted points (Richardson extrapolation).
    Must be ``None`` for an exponential fit."""

    noise_factors: Sequence[int] | None = None
    """Executed noise factors to fit, or ``None`` for all of them."""

    def __post_init__(self) -> None:
        """Reject contradictory settings early."""
        if self.extrapolation_type not in ("Poly", "Exp"):
            raise ValueError(f"Unknown extrapolation_type {self.extrapolation_type!r}. Expected 'Poly' or 'Exp'.")
        if self.extrapolation_type == "Exp" and self.degree is not None:
            raise ValueError("An exponential fit has no degree. Set degree=None.")
        if self.degree is not None and self.degree < 1:
            raise ValueError(f"The degree must be at least 1, got {self.degree}.")
        if self.noise_factors is not None:
            selected = tuple(int(factor) for factor in self.noise_factors)
            if len(selected) < MIN_FIT_POINTS or len(set(selected)) != len(selected):
                raise ValueError(f"Select at least {MIN_FIT_POINTS} distinct noise factors, got {selected}.")
            object.__setattr__(self, "noise_factors", selected)

    def accepts(self, technique: Technique) -> bool:
        """Return whether the technique is :class:`ZNE`."""
        return isinstance(technique, ZNE)

    def estimate(self, target: TargetCircuit, term_values: np.ndarray, observables: ObservableSet) -> TargetEstimate:
        """Extrapolate every Pauli term to zero noise and combine them into the observables.

        Args:
            target: The planned target, whose variants are :class:`NoiseAmplifiedVariant`.
            term_values: Measured value of each Pauli term of each variant, shape ``(num_variants, num_terms)``.
            observables: The observables.

        Returns:
            The estimate, with :class:`ZNEDiagnostics`.

        Raises:
            ValueError: If a selected noise factor was not executed, or a polynomial degree is too high.

        """
        factors = [_noise_factor(variant) for variant in target.variants()]
        wanted = factors if self.noise_factors is None else list(self.noise_factors)
        missing = sorted(set(wanted) - set(factors))
        if missing:
            raise ValueError(f"Noise factors {missing} were not executed. Executed: {factors}.")
        selected = tuple(index for index, factor in enumerate(factors) if factor in wanted)
        x = np.asarray([factors[index] for index in selected], dtype=float)
        measured = np.asarray(term_values, dtype=float)[list(selected)]

        extrapolated = np.empty(observables.num_terms, dtype=float)
        parameters: list[tuple[float, ...]] = []
        for column in range(observables.num_terms):
            value, fit = extrapolate_at_zero(
                x.tolist(), measured[:, column].tolist(), extrapolation_type=self.extrapolation_type, degree=self.degree
            )
            extrapolated[column] = value
            parameters.append(tuple(fit))
        _warn_on_unphysical_terms(extrapolated, observables.pauli_terms)

        diagnostics = ZNEDiagnostics(
            variant_indices=selected,
            noise_factors=x,
            pauli_terms=observables.pauli_terms,
            measured=measured,
            extrapolated=extrapolated,
            fit_parameters=tuple(parameters),
            extrapolation_type=self.extrapolation_type,
            degree=self.degree,
            num_executed_factors=len(factors),
        )
        return TargetEstimate(values=observables.combine(extrapolated), diagnostics=diagnostics)


def _amplified(circuit: Circuit, noise_factor: int) -> Circuit:
    """Fold a circuit to the given noise factor, returning the circuit itself for noise factor 1."""
    if noise_factor == 1:
        return circuit
    folded = fold_global_pulse(circuit, (noise_factor - 1) // 2, name=f"{circuit.name}_nf{noise_factor}")
    if not isinstance(folded, Circuit):
        raise TypeError(f"Folding returned a {type(folded).__name__}, not a Circuit.")
    return folded


def _noise_factor(variant: CircuitVariant) -> int:
    """Read the noise factor of a variant built by :class:`ZNE`."""
    if not isinstance(variant, NoiseAmplifiedVariant):
        raise TypeError(f"Variant {variant.variant_index} is a {type(variant).__name__}, not a NoiseAmplifiedVariant.")
    return variant.noise_factor


def _warn_on_unphysical_terms(extrapolated: np.ndarray, labels: Sequence[str]) -> None:
    """Warn about Pauli terms extrapolated outside ``[-1, 1]``."""
    outside = [f"{label!r} = {value:+.4g}" for label, value in zip(labels, extrapolated) if abs(value) > 1.0]
    if outside:
        warnings.warn(
            f"These Pauli terms extrapolated outside [-1, 1], which a Pauli expectation value cannot be: "
            f"{', '.join(outside)}.",
            stacklevel=5,
        )
