#  ********************************************************************************
#    _____  ____ _
#   / _ \ \/ / _` |  Framework for control
#  |  __/>  < (_| |  and measurement of
#   \___/_/\_\__,_|  superconducting qubits
#
#  Copyright 2019-2026 IQM Finland Oy
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ********************************************************************************
"""Lmfit model for Gaussian."""

from typing import Literal
import warnings

import lmfit
import numpy as np
import scipy.signal

from .model_utils import BaseModel

GAUSSIAN_PARAMS = ["x_0", "y_0", "w", "A", "slope"]


def gaussian(x: np.ndarray, x_0: float = 0, y_0: float = 0, w: float = 1, A: float = 1, slope: float = 0) -> np.ndarray:
    """Gaussian curve.

    Args:
        x: Coordinate.
        x_0: Mean.
        y_0: Offset.
        w: Width.
        A: Amplitude.
        slope: Linear slope.

    Returns:
        Gaussian.

    """
    return A * np.exp(-((x - x_0) ** 2) / (2 * w**2)) + y_0 + slope * x


class GaussianModel(BaseModel):
    """Gaussian fit with offset. Can fit peak (A>0) or dip (A<0), which is automatically guessed.

    .. math::
    f(x) = y_0 + A{\rm e}^{(-(x-x_0)^2/(2 w^2))}
    """  # noqa: D301

    __doc__ = lmfit.models.COMMON_INIT_DOC

    def __init__(self, independent_vars: list[Literal["x"]] | None = None, **kwargs) -> None:
        self.normalize_data = kwargs.pop("normalize_data", False)
        if independent_vars is None:
            independent_vars = ["x"]
        super().__init__(gaussian, independent_vars=independent_vars, **kwargs)
        self.set_param_hint(name="w", min=1e-15)
        self.set_param_hint(name="slope", vary=False)

    def guess(self, data: np.ndarray, x: np.ndarray, *args, **kws) -> lmfit.parameter.Parameters:
        """Initial guess for the model."""
        slope = (data[-1] - data[0]) / (x[-1] - x[0])
        scaled_data = data - slope * x if self.param_hints["slope"]["vary"] else data
        y_0 = np.median(scaled_data)
        scaled_data = scaled_data - y_0
        normalized_data = np.abs(scaled_data) if self.normalize_data else scaled_data
        A = np.max(normalized_data) if np.max(normalized_data) > -np.min(normalized_data) else np.min(normalized_data)

        data_for_guess = normalized_data if self.normalize_data else scaled_data
        data_for_guess = data_for_guess if A > 0 else -data_for_guess
        idx_max_val, max_val = self._robust_max_value(data_for_guess, n=1, rtol=0.25)
        x_0 = x[idx_max_val]
        with warnings.catch_warnings():
            # Suppress peak not found warnings. They are handled below.
            warnings.simplefilter("ignore")
            peak_width_out = scipy.signal.peak_widths(data_for_guess, [idx_max_val], rel_height=0.5)
        w = peak_width_out[0][0] * (x[1] - x[0])

        if w == 0.0:  # Peak could not be found. Fit quadratic function
            # Expand around x=x_0 and take terms up to second order.
            [c, b, a] = np.polynomial.polynomial.polyfit(x, data, 2)
            x_0 = -1 * b / (2 * a)
            y_0 = 0
            A = c - x_0**2 * a
            w = x[1] - x[0] if -A / a / 2 < 0 else np.sqrt(-A / a / 2)

        if self.param_hints["slope"]["vary"]:
            y_0 = np.median(data - slope * x)

        guesses = {"x_0": x_0, "y_0": y_0, "w": np.abs(w) / 2, "A": A, "slope": slope}
        params = self.make_params(**guesses)
        slope_hint = self.param_hints["slope"]
        if not slope_hint.get("vary", True) and not slope_hint.get("expr") and "value" not in slope_hint:
            # The model itself holds the slope fixed without naming a value, meaning the linear background
            # is to be estimated here rather than fitted. A caller who names a value owns it instead, and
            # ``filter_param_guesses`` has already dropped the guess for them from reaching ``make_params``.
            params[f"{self.prefix}slope"].value = slope
        return lmfit.models.update_param_vals(params, self.prefix, **kws)

    def _robust_max_value(self, data: np.ndarray, n: int = 1, rtol: float = 0.2) -> tuple[int, int | float]:
        """Find the index of the maximum value in data such that its n neighbors
        are within a given tolerance. This makes finding the maximum more robust.

        Args:
            data: array with valu
            n: number of neighbors to consider. n=1 means left and right neighbour.
            rtol: relative tolerance to accept a central point. Set to np.inf to directly yield argmax(data).

        Returns:
            Index of the constrained maximum (or argmax fallback),
            Value of the found maximum

        """
        found_in_tol = False
        # Sort indices by descending value that will be checked for neighbours
        sorted_idx = np.argsort(data)[::-1]
        for idx in sorted_idx:
            left = max(0, idx - n)
            right = min(len(data), idx + n + 1)
            neighbors = data[left:right]

            # Check all neighbors within tolerance of central value
            data_range = np.max(data) - np.min(data)
            if np.all(np.abs(neighbors - data[idx]) <= rtol * data_range):
                idx_max = idx
                found_in_tol = True
                break  # only loop through small part of highest values

        if not found_in_tol and rtol < np.inf:
            idx_max = int(np.argmax(data))
            warnings.warn(
                f"No maximum found with neighbors within tolerance= {rtol}; returning plain argmax instead.",
                RuntimeWarning,
            )

        return idx_max, data[idx_max]
