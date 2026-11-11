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
"""Lmfit model for Linear fit."""

from typing import Literal

import lmfit
import numpy as np

from .model_utils import BaseModel

LINEAR_PARAMS = ["a", "b"]


def linear(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Linear function.

    Args:
        x: x-coordinate.
        a: Slope.
        b: Offset.

    Returns:
        Linear function.

    """
    return a * x + b


class LinearModel(BaseModel):
    """Linear model.

    .. math::
    f(x) = a * x + b
    """

    __doc__ = lmfit.models.COMMON_INIT_DOC

    def __init__(self, independent_vars: list[Literal["x"]] | None = None, **kwargs) -> None:
        if independent_vars is None:
            independent_vars = ["x"]
        super().__init__(linear, independent_vars=independent_vars, **kwargs)

    def guess(self, data: np.ndarray, x: np.ndarray, *args, **kws) -> lmfit.parameter.Parameters:
        """Initial guess for the model."""
        [b_guess, a_guess] = np.polynomial.polynomial.polyfit(x, data, 1)
        params = self.make_params(a=a_guess, b=b_guess)
        return lmfit.models.update_param_vals(params, self.prefix, **kws)
