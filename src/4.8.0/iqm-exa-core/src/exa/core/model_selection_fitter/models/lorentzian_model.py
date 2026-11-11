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
"""Lmfit model for Lorentzian."""

from typing import Literal
import warnings

import lmfit
import numpy as np
import scipy.signal

from .model_utils import BaseModel

LORENTZIAN_PARAMS = ["height", "center", "width", "offset", "slope"]


def lorentzian(
    x: np.ndarray, height: float, center: float, width: float, offset: float, slope: float = 0
) -> np.ndarray:
    """Lorentzian function with offset and slope.

    Args:
        x: x-coordinate.
        height: Amplitude of Lorentzian peak.
        center: Location of the peak.
        width: Width of the peak.
        offset: Constant offset.
        slope: Linear slope.

    Returns:
        Lorentzian function.

    """
    return height * width**2 / ((x - center) ** 2 + width**2) + offset + slope * x


class LorentzianModel(BaseModel):
    """Lorentzian model.

    .. math::
      f(x) = height * width**2 / ((x - center) ** 2 + width**2) + offset + slope * x
    """  # noqa: D301

    __doc__ = lmfit.models.COMMON_INIT_DOC

    def __init__(self, independent_vars: list[Literal["x"]] | None = None, **kwargs) -> None:
        if independent_vars is None:
            independent_vars = ["x"]
        super().__init__(lorentzian, independent_vars=independent_vars, **kwargs)
        self.set_param_hint(name="height", value=1)
        self.set_param_hint(name="center", value=0)
        self.set_param_hint(name="offset", value=0)
        self.set_param_hint(name="width", value=0.2, min=1e-15)
        self.set_param_hint(name="slope", value=0, vary=False)

    def guess(self, data: np.ndarray, x: np.ndarray, *args, **kws) -> lmfit.parameter.Parameters:
        """Initial guess for amplitude, center and width based on the data.

        Amplitude and center is based on the max or min of the data, while width is determined using
        ``scipy.signal.peak_widths``.
        If latter fails then fits a second order polynomial and extracts width from the fit.
        """
        temp_params = self.make_params()
        slope = (data[-1] - data[0]) / (x[-1] - x[0])
        scaled_data = data - slope * x if temp_params[self.prefix + "slope"].vary else data
        offset = np.median(scaled_data)
        scaled_data = scaled_data - offset

        data_for_guess = scaled_data
        height = np.max(scaled_data) if np.max(scaled_data) > -np.min(scaled_data) else np.min(scaled_data)
        data_for_guess = data_for_guess if height > 0 else -data_for_guess

        center = x[np.argmax(data_for_guess)]
        with warnings.catch_warnings():
            # Suppress peak not found warnings. They are handled below.
            warnings.simplefilter("ignore")
            peak_width_out = scipy.signal.peak_widths(data_for_guess, [np.argmax(data_for_guess)], rel_height=0.5)
        width = peak_width_out[0][0] * (x[1] - x[0])

        if width == 0.0:  # Peak could not be found. Fit quadratic function
            # Expand around x=center and take terms up to second order.
            [c, b, a] = np.polynomial.polynomial.polyfit(x, data, 2)
            if a == 0:
                # No measurable curvature (flat or perfectly linear data): the peak location is
                # undefined. Fall back to coarse range-based heuristics so the fitter still receives
                # finite initial values.
                center = float((x[0] + x[-1]) / 2)
                offset = float(np.median(data))
                height = float(np.max(data) - offset)
                width = float(x[-1] - x[0]) / 10
            else:
                center = -1 * b / (2 * a)
                offset = 0
                height = c - center**2 * a
                width = x[1] - x[0] if -height / a / 2 < 0 else np.sqrt(-height / a / 2)

        params_map = {}
        params_map[self.prefix + "height"] = height
        params_map[self.prefix + "center"] = center
        params_map[self.prefix + "width"] = np.abs(width) / 2
        if temp_params[self.prefix + "slope"].vary:
            offset = np.median(data - slope * x)
        params_map[self.prefix + "slope"] = slope
        params_map[self.prefix + "offset"] = offset
        params = self.make_params(**params_map)
        return lmfit.models.update_param_vals(params, self.prefix, **kws)
