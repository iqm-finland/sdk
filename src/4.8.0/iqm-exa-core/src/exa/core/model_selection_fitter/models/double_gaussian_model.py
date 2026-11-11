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
"""Lmfit model for Double Gaussian."""

from typing import Literal

import lmfit
import numpy as np
from scipy.signal import find_peaks

from .model_utils import BaseModel

DOUBLE_GAUSSIAN_PARAMS = ["x_01", "x_02", "w1", "w2", "a1", "a2", "y_0"]


def double_gaussian(
    x: np.ndarray, x_01: float, x_02: float, w1: float, w2: float, a1: float, a2: float, y_0: float
) -> np.ndarray:
    """Double Gaussian function with offset.

    Args:
        x: Coordinate.
        x_01: Center of first Gaussian.
        x_02: Center of second Gaussian.
        w1: Standard deviation of first Gaussian.
        w2: Standard deviation of second Gaussian.
        a1: Amplitude of first Gaussian.
        a2: Amplitude of second Gaussian.
        y_0: Offset.

    Returns:
        np.ndarray: Values of the double Gaussian function at given x-coordinates.

    """
    return a1 * np.exp(-((x - x_01) ** 2) / (2 * w1**2)) + a2 * np.exp(-((x - x_02) ** 2) / (2 * w2**2)) + y_0


class DoubleGaussianModel(BaseModel):
    """Lmfit model for double Gaussian function with offset.

    .. math::
    f(x; x_01, x_02, w1, w2, a1, a2, y_0) = y_0 + a1 e^{(-(x-x_01)^2/(2 w1^2))} + a2 e^{(-(x-x_02)^2/(2 w2^2))}

    """

    __doc__ = lmfit.models.COMMON_INIT_DOC

    def __init__(self, independent_vars: list[Literal["x"]] | None = None, **kwargs) -> None:
        """Initializes the DoubleGaussianModel.

        Args:
            args: Passed to parent __init__.
            independent_vars: List of independent variable names.
            kwargs: Passed to parent __init__.

        """
        if independent_vars is None:
            independent_vars = ["x"]
        super().__init__(
            double_gaussian,
            independent_vars=independent_vars,
            **kwargs,
        )
        self.set_param_hint(name="x_01", value=0)
        self.set_param_hint(name="x_02", value=0)
        self.set_param_hint(name="w1", value=1, min=1e-15)
        self.set_param_hint(name="w2", value=1, min=1e-15)
        self.set_param_hint(name="a1", value=1, min=0)
        self.set_param_hint(name="a2", value=1, min=0)
        self.set_param_hint(name="y_0", value=0)

    def guess(self, data: np.ndarray, x: np.ndarray, **kws) -> lmfit.parameter.Parameters:
        """Estimate initial model parameter values from data.

        Uses scipy.find_peaks to locate most prominent peaks. If data contains at least two peaks, sets their
        x-coordinates and amplitudes as centers and amplitudes of the Gaussians. The width of the Gaussians is estimated
        as half a distance between the peaks. If data contains a single peak, sets center of both Gaussians to the
        x-coordinate of the peak, their amplitude to the peak amplitude, and width to one tenth of the x-range.
        The value of the offset is set to the minimum value in the data.

        Args:
            data: Array of data used to guess parameter values.
            x: Array of values for the independent variable.
            kws: Values for the model parameters.

        Returns:
            Estimated parameters for fitting.

        """
        temp_params = self.make_params()
        data = np.array(data)

        peaks_idx, properties = find_peaks(data, prominence=0, width=1)
        if len(peaks_idx) == 0:
            return temp_params

        sorted_indices = np.argsort(properties["prominences"])[::-1]
        highest_peaks_idx = peaks_idx[sorted_indices[:2]]
        x_01 = x[highest_peaks_idx[0]]
        a1 = data[highest_peaks_idx[0]]

        if len(peaks_idx) < 2:
            w = (max(x) - min(x)) / 10
            x_02 = x_01
            a2 = a1
        else:
            w = abs(x[highest_peaks_idx[1]] - x[highest_peaks_idx[0]]) / 2
            x_02 = x[highest_peaks_idx[1]]
            a2 = data[highest_peaks_idx[1]]

        params = self.make_params(
            x_01={"value": x_01, "min": min(x), "max": max(x)},
            x_02={"value": x_02, "min": min(x), "max": max(x)},
            w1=w,
            w2=w,
            a1=a1,
            a2=a2,
            y_0=data.min(),
        )
        return lmfit.models.update_param_vals(params, self.prefix, **kws)
