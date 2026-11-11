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
"""Lmfit model for sine expansion."""

from typing import Literal

import lmfit
import numpy as np

from .model_utils import BaseModel, prefix_expr

SINE_EXPANSION_PARAMS = ["A", "f", "phi", "offset", "T_exp", "T_gauss"]


def sine_expansion(
    t: np.ndarray, A: float, f: float, phi: float, offset: float, T_exp: float = np.inf, T_gauss: float = np.inf
) -> np.ndarray:
    """Returns the first-order Taylor expansion for sine model with decay.

    The parameters of the model are the same as in GaussExpSineModel.

    f(t) = A*sin(2*pi*f*t + phi)*exp(-t/T_exp - t**2/T_exp**2) + offset
    f^{(1)}(t) = f(0) + df(t)/dt*t|t=0

    Args:
        t: Time coordinate.
        A: Amplitude.
        f: Frequency.
        phi: Phase.
        offset: Offset.
        T_exp: Exponential decay time.
        T_gauss: Gaussian decay time.

    Returns:
        The first-order Taylor expansion of a sinusoid.

    """
    return A * np.sin(phi) + offset + A * t * (2 * np.pi * f * np.cos(phi))


class SmallAngleSineExpansion(BaseModel):
    """First order Taylor expansion for GaussExpSineModel from model-selection fitter."""

    __doc__ = lmfit.models.COMMON_INIT_DOC

    def __init__(self, independent_vars: list[Literal["t"]] | None = None, **kwargs) -> None:
        if independent_vars is None:
            independent_vars = ["t"]
        super().__init__(
            sine_expansion, independent_vars=independent_vars, param_names=SINE_EXPANSION_PARAMS, prefix="", **kwargs
        )
        self.set_param_hint(name="A", value=1, min=0, vary=True)
        self.set_param_hint(name="f", value=-3e6, vary=True)
        self.set_param_hint(name="phi", value=0, min=-2 * np.pi, max=2 * np.pi, vary=True)
        self.set_param_hint(name="offset", value=0.5, vary=True)
        self.set_param_hint(name="T_exp", value=np.inf, vary=False)
        self.set_param_hint(name="T_gauss", value=np.inf, vary=False)
        self.set_param_hint(
            name="T_2",
            expr="(-1/T_exp + sqrt(1/T_exp**2 + 4/T_gauss**2))/(2/T_gauss**2) if T_gauss !=inf else T_exp",
            vary=False,
        )

    def guess(self, data: np.ndarray, t: np.ndarray, **kws) -> lmfit.parameter.Parameters:
        """Initial guess for the model.

        Makes a linear fit using polyfit. Since there are more free parameter in the model
        than in linear fit, amplitude, phase, T_exp, and T_gauss are assumed to be fixed.

        Args:
            data: One-dimensional NumPy array to fit.
            t: Time coordinate.
            kws: Values for the model parameters ``f``, ``offset``, ``T_2``.

        """
        a_guess = self.param_hints["A"]["value"]
        phi_guess = self.param_hints["phi"]["value"]

        [b, a] = np.polynomial.polynomial.polyfit(t, data, 1)

        # ``T_2`` is left out: its hint ties it to the decay times with an expression, and a guess for it
        # would cut that tie. ``filter_param_guesses`` drops the guesses for the parameters hints pin.
        params = self.make_params(
            f=a / (np.cos(phi_guess) * a_guess * 2 * np.pi),
            offset=b - a_guess * np.sin(phi_guess),
        )

        return lmfit.models.update_param_vals(params, self.prefix, **kws)

    def post_fit(self, fitresult: lmfit.model.ModelResult) -> None:
        fitresult.params.add(
            f"{self.prefix}T_2",
            expr=prefix_expr(
                self, "(-1/T_exp + sqrt(1/T_exp**2 + 4/T_gauss**2))/(2/T_gauss**2) if T_gauss !=inf else T_exp"
            ),
        )
