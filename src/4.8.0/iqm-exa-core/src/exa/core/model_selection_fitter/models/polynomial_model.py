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
"""Lmfit model for Polynomial fit."""

from typing import Literal

import lmfit
import numpy as np

from .model_utils import BaseModel

THIRD_POLYNOMIAL_PARAMS = ["a_3", "a_2", "a_1", "a_0"]
SECOND_POLYNOMIAL_PARAMS = ["a_2", "a_1", "a_0"]
FIRST_POLYNOMIAL_PARAMS = ["a_1", "a_0"]


def third_order_poly_function(
    x: np.ndarray, a_3: float = 1, a_2: float = 0, a_1: float = 0, a_0: float = 0
) -> np.ndarray:
    """Third order polynomial model.

    Args:
        x: x-coordinate.
        a_3: Coefficient of cubic term.
        a_2: Coefficient of quadratic term.
        a_1: Coefficient of linear term.
        a_0: Constant term.

    Returns:
        Values of the third order polynomial for ``x``.

    """
    return np.polynomial.polynomial.polyval(x, [a_0, a_1, a_2, a_3])


class ThirdOrderPolyModel(BaseModel):
    """Third order polynomial model.

    .. math:: f(x) = a_3 * x^3 + a_2 * x^2 + a_1 * x^1 + a_0
    """

    __doc__ = lmfit.models.COMMON_INIT_DOC

    def __init__(self, independent_vars: list[Literal["x"]] | None = None, **kwargs) -> None:
        if independent_vars is None:
            independent_vars = ["x"]
        super().__init__(third_order_poly_function, independent_vars=independent_vars, **kwargs)


def second_order_poly_function(x: np.ndarray, a_2: float = 1, a_1: float = 0, a_0: float = 0) -> np.ndarray:
    """Second order polynomial model.

    Args:
        x: x-coordinate.
        a_2: Coefficient of quadratic term.
        a_1: Coefficient of linear term.
        a_0: Constant term.

    Returns:
        Values of the second order polynomial for ``x``.

    """
    return np.polynomial.polynomial.polyval(x, [a_0, a_1, a_2])


class SecondOrderPolyModel(BaseModel):
    """Second order polynomial model.

    .. math:: f(x) = a_2 * x^2 + a_1 * x^1 + a_0
    """

    __doc__ = lmfit.models.COMMON_INIT_DOC

    def __init__(self, independent_vars: list[Literal["x"]] | None = None, **kwargs) -> None:
        if independent_vars is None:
            independent_vars = ["x"]
        super().__init__(second_order_poly_function, independent_vars=independent_vars, **kwargs)

    def guess(self, data: np.ndarray, x: np.ndarray, *args, **kws) -> lmfit.parameter.Parameters:
        """Initial guess for the quadratic model."""
        # Estimate a2
        x1, x2, x3 = x[:-2], x[1:-1], x[2:]
        m12 = np.diff(data[:-1]) / np.diff(x[:-1])
        m23 = np.diff(data[1:]) / np.diff(x[1:])
        dm = m23 - m12
        individual_a2_guess_values = dm / (x3 - x1) / 2  # Divide by 2 because d2y/dx2 = 2*a_2
        a2_guess = np.mean(individual_a2_guess_values)

        # Estimate a1
        x1, x2 = x[:-1], x[1:]
        # Correct the slope calculation to account for the estimated a_2
        individual_a1_guess_values = np.diff(data) / np.diff(x) - a2_guess * (x2 + x1)
        a1_guess = np.mean(individual_a1_guess_values)

        # Estimate a0
        individual_a0_guess_values = data - a1_guess * x - a2_guess * x**2
        a0_guess = np.mean(individual_a0_guess_values)

        params = self.make_params(a_2=a2_guess, a_1=a1_guess, a_0=a0_guess)
        return lmfit.models.update_param_vals(params, self.prefix, **kws)


def first_order_poly_function(x: np.ndarray, a_1: float = 1, a_0: float = 0) -> np.ndarray:
    """First order polynomial model.

    Args:
        x: x-coordinate.
        a_1: Coefficient of linear term.
        a_0: Constant term.

    Returns:
        Values of the first order polynomial for ``x``.

    """
    return np.polynomial.polynomial.polyval(x, [a_0, a_1])


class FirstOrderPolyModel(BaseModel):
    """First order polynomial model.

    .. math:: f(x) = a_1 * x^1 + a_0
    """

    __doc__ = lmfit.models.COMMON_INIT_DOC

    def __init__(self, independent_vars: list[Literal["x"]] | None = None, **kwargs) -> None:
        if independent_vars is None:
            independent_vars = ["x"]
        super().__init__(first_order_poly_function, independent_vars=independent_vars, **kwargs)

    def guess(self, data: np.ndarray, x: np.ndarray, *args, **kws) -> lmfit.parameter.Parameters:
        """Initial guess for the linear model."""
        individual_a1_guess_values = np.diff(data) / np.diff(x)
        a1_guess = np.mean(individual_a1_guess_values)

        individual_a0_guess_values = data - a1_guess * x
        a0_guess = np.mean(individual_a0_guess_values)

        params = self.make_params(a_1=a1_guess, a_0=a0_guess)
        return lmfit.models.update_param_vals(params, self.prefix, **kws)
