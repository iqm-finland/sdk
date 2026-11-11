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

"""Linear fitting for CDR."""

from __future__ import annotations

from typing import Any

import numpy as np

_FLAT_STD = 1e-12
"""Standard deviation below which the noisy training values count as constant."""


def _fit_linear(noisy: np.ndarray, exact: np.ndarray) -> tuple[float, float]:
    """Fit ``exact = a * noisy + b`` by least squares.

    Returns ``a = 0`` and ``b = mean(exact)`` when there are fewer than two points or the noisy values are constant,
    because the slope is then undefined.
    """
    noisy = np.asarray(noisy, dtype=float)
    exact = np.asarray(exact, dtype=float)
    if noisy.size < 2 or np.std(noisy) < _FLAT_STD:  # noqa: PLR2004
        return 0.0, float(np.mean(exact)) if exact.size else 0.0
    a, b = np.polyfit(noisy, exact, deg=1)
    return float(a), float(b)


def cdr_fit_and_predict(
    exact_train: np.ndarray,
    noisy_train: np.ndarray,
    noisy_target: np.ndarray | float,
    mode: str = "per_observable",
    return_coeffs: bool = False,
) -> Any:
    """Fit the CDR line of every column and evaluate it at the noisy target value.

    A column is one Pauli term or one observable.

    Args:
        exact_train: Exact training values, shape ``(num_training_circuits,)`` for one column or
            ``(num_training_circuits, num_columns)``.
        noisy_train: Measured training values, same shape as ``exact_train``.
        noisy_target: Measured target values, a scalar for one column or shape ``(num_columns,)``.
        mode: ``"per_observable"`` fits each column separately. ``"pooled"`` fits one line to the pairs of all
            columns.
        return_coeffs: Whether to also return the fitted ``(a, b)`` of each line.

    Returns:
        The corrected target values, a scalar for one column or shape ``(num_columns,)``. With ``return_coeffs``,
        also the list of ``(a, b)``, one per column or one for ``"pooled"``.

    Raises:
        ValueError: If the shapes of ``noisy_train`` and ``exact_train`` differ, ``noisy_target`` does not have one
            entry per column, or ``mode`` is unknown.

    """
    exact_train = np.asarray(exact_train, dtype=float)
    noisy_train = np.asarray(noisy_train, dtype=float)
    scalar_input = exact_train.ndim == 1
    if scalar_input:
        exact_train = exact_train[:, None]
        noisy_train = noisy_train[:, None]

    noisy_target = np.atleast_1d(np.asarray(noisy_target, dtype=float))
    nt, nobs = exact_train.shape
    if noisy_train.shape != exact_train.shape:
        raise ValueError(f"noisy_train shape {noisy_train.shape} must match exact_train shape {exact_train.shape}.")
    if noisy_target.shape[0] != nobs:
        raise ValueError(f"noisy_target has {noisy_target.shape[0]} entries, but there are {nobs} columns.")

    mitigated = np.zeros(nobs, dtype=float)
    coeffs = []

    if mode == "pooled":
        a, b = _fit_linear(noisy_train.reshape(-1), exact_train.reshape(-1))
        mitigated = a * noisy_target + b
        coeffs = [(a, b)]
    elif mode == "per_observable":
        for j in range(nobs):
            a, b = _fit_linear(noisy_train[:, j], exact_train[:, j])
            mitigated[j] = a * noisy_target[j] + b
            coeffs.append((a, b))
    else:
        raise ValueError(f"Unknown mode {mode!r}. Use 'per_observable' or 'pooled'.")

    result = float(mitigated[0]) if scalar_input else mitigated
    if return_coeffs:
        return result, coeffs
    return result


def r_squared(noisy: np.ndarray, exact: np.ndarray) -> tuple[float, float, float]:
    """Fit the CDR line ``exact = a*noisy + b`` and report its R^2.

    Args:
        noisy: Measured noisy training values.
        exact: Exact training values, same shape as ``noisy``.

    Returns:
        The slope, the intercept and R^2. R^2 is NaN if the exact values are constant.

    """
    noisy = np.asarray(noisy, dtype=float)
    exact = np.asarray(exact, dtype=float)
    slope, intercept = _fit_linear(noisy, exact)
    pred = slope * noisy + intercept
    ss_res = float(np.sum((exact - pred) ** 2))
    ss_tot = float(np.sum((exact - np.mean(exact)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), float(r2)
