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
"""lmfit model for absolute value function."""

from typing import Literal

import lmfit
import numpy as np

from .model_utils import BaseModel

ABSOLUTE_VALUE_PARAMS = ["slope", "offset", "center"]


def _absolute_value(x: np.ndarray, slope: float, offset: float, center: float) -> np.ndarray:
    """Scalar distance from a given point, ``center``, scaled by a factor ``slope``,
    plus a constant ``offset``.

    Args:
        x: X-axis coordinate.
        slope: Slope of the absolute value function.
        offset: Y-axis offset of the absolute value function.
        center: X-axis value, where the absolute value function is minimal.


    Returns:
        Absolute value function.

    """
    return slope * np.abs(x - center) + offset


class AbsoluteValueModel(BaseModel):
    r"""Scaled scalar distance from given point, plus offset.

    Defined as:

    .. math:: f(x; m, b, x_0) = m |x - x_0| + b

    with ``slope`` for :math:`m`, ``offset`` for :math:`b` and ``center`` for :math:`x_0`.

    """

    __doc__ = lmfit.models.COMMON_INIT_DOC

    def __init__(self, independent_vars: list[Literal["x"]] | None = None, **kwargs) -> None:
        if independent_vars is None:
            independent_vars = ["x"]
        super().__init__(_absolute_value, independent_vars=independent_vars, **kwargs)

    def guess(self, data: np.ndarray, x: np.ndarray, **kwargs) -> lmfit.parameter.Parameters:
        """Initial guess for the model."""
        slope_guess = np.mean(np.abs(np.diff(data) / np.diff(x)))
        center_guess = x[np.argmin(np.abs(data))]
        offset_guess = np.min(np.abs(data)) * np.sign(data[np.argmin(np.abs(data))])

        params = self.make_params(slope=slope_guess, offset=offset_guess, center=center_guess)
        return lmfit.models.update_param_vals(params, self.prefix, **kwargs)
