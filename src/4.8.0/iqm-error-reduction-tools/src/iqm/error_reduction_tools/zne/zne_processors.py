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

"""Fits for zero-noise extrapolation and their values at zero noise."""

from collections.abc import Sequence
from typing import Literal, cast

import numpy as np
from scipy import optimize


def _polynomial_ansatz(x: float | Sequence[float] | np.ndarray, coefficients: Sequence[float]) -> np.ndarray:
    """Evaluate the polynomial model at ``x``, taking numpy's highest-degree-first coefficients."""
    return np.polyval(np.asarray(coefficients, dtype=float), x)


def _exponential_ansatz(x: float | Sequence[float] | np.ndarray, amplitude: float, rate: float) -> np.ndarray:
    """Evaluate the exponential model ``amplitude * exp(rate * x)``, in curve_fit's parameter form."""
    return amplitude * np.exp(rate * np.asarray(x, dtype=float))


def extrapolate_at_zero(
    x_data: Sequence[float],
    y_data: Sequence[float],
    extrapolation_type: Literal["Poly", "Exp"] = "Poly",
    degree: int | None = None,
) -> tuple[float, list[float]]:
    """Fit ``y = f(x)`` and evaluate the fit at ``x = 0``.

    Args:
        x_data: The x values, e.g. the noise factors.
        y_data: The y values at ``x_data``.
        extrapolation_type: ``"Poly"`` for a polynomial fit, ``"Exp"`` for ``a * exp(b * x)``.
        degree: Degree of the polynomial. ``None`` uses ``len(y_data) - 1`` (Richardson extrapolation).

    Returns:
        The value of the fit at zero, and the fit parameters.

    Raises:
        ValueError: If ``x_data`` and ``y_data`` differ in length, the extrapolation type is unknown, the degree is
            too high for the number of points, or the exponential fit does not converge.

    """
    if len(y_data) != len(x_data):
        raise ValueError(f"Got {len(y_data)} y values for {len(x_data)} x values.")

    coeff: list[float]

    if extrapolation_type == "Poly":
        if degree is None:
            degree = len(y_data) - 1
        elif degree > len(y_data) - 1:
            raise ValueError(
                f"The degree of the polynomial is too high for the given data (len(y_data)={len(y_data)}, "
                f"degree={degree}). It should be at most {len(y_data) - 1}."
            )

        coeff = np.polyfit(x_data, y_data, deg=degree).tolist()
        value_at_zero = _polynomial_ansatz(0.0, coeff)

    elif extrapolation_type == "Exp":
        amplitude = float(y_data[int(np.argmin(x_data))])
        try:
            coeff_arr, *_ = optimize.curve_fit(
                _exponential_ansatz,
                x_data,
                y_data,
                p0=(amplitude, 0.0),
            )
            coeff = [float(x) for x in cast(np.ndarray, coeff_arr).tolist()]
        except RuntimeError as error:
            raise ValueError(f"The exponential fit did not converge on {list(y_data)} at {list(x_data)}.") from error
        value_at_zero = _exponential_ansatz(0.0, *coeff)
    else:
        raise ValueError(f"Unknown extrapolation_type {extrapolation_type!r}. Expected 'Poly' or 'Exp'.")

    return float(value_at_zero), list(coeff)


def polynomial_extrapolation_weights(
    x_data: Sequence[float],
    degree: int | None = None,
) -> list[float]:
    """Return the weights a fixed-degree polynomial extrapolation applies to its ordinates.

    Args:
        x_data: The x data points, e.g. the noise amplification factors.
        degree: Degree of the polynomial. If None, ``len(x_data) - 1`` (Richardson extrapolation).

    Returns:
        One weight per entry of ``x_data``, in the same order.

    Raises:
        ValueError: If ``degree`` is too high for the number of points.

    """
    x = np.asarray(x_data, dtype=float)
    if degree is None:
        degree = len(x) - 1
    elif degree > len(x) - 1:
        raise ValueError(
            f"The degree of the polynomial is too high for the given data (len(x_data)={len(x)}, "
            f"degree={degree}). It should be at most {len(x) - 1}."
        )
    return [float(weight) for weight in np.linalg.pinv(np.vander(x, degree + 1))[-1]]


def fit_curve(parameters: Sequence[float], x: np.ndarray, extrapolation_type: str) -> np.ndarray:
    """Evaluate a fitted model, for plotting the fit next to the measured points.

    Args:
        parameters: The ``fit_parameters`` entry of one observable or Pauli term, as the estimator's
            diagnostics report it.
        x: Where to evaluate.
        extrapolation_type: ``"Poly"`` or ``"Exp"``, matching the fit.

    Returns:
        The model's value at each ``x``.

    Raises:
        ValueError: If the extrapolation type is unknown.

    """
    if extrapolation_type == "Poly":
        return _polynomial_ansatz(x, parameters)
    if extrapolation_type == "Exp":
        amplitude, rate = parameters
        return _exponential_ansatz(x, amplitude, rate)
    raise ValueError(f"Unknown extrapolation_type {extrapolation_type!r}. Expected 'Poly' or 'Exp'.")
