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
"""Lmfit model for decaying sine."""

from typing import Literal

import lmfit
import numpy as np

from .model_utils import BaseModel, frequency_guess_with_fft, prefix_expr

DECAYING_SINE_PARAMS = ["A", "f", "phi", "offset", "T_exp", "T_gauss"]
ABS_DECAY_PARAMS = ["pulse_duration", "amplitude", "offset", "exp_decay_time", "gauss_decay_time"]


def decaying_sine(
    t: np.ndarray,
    A: float,
    f: float | np.ndarray,
    phi: float | np.ndarray,
    offset: float | np.ndarray,
    T_exp: float,
    T_gauss: float,
) -> np.ndarray:
    """Decaying sine function.

    Models a sine which contains both Gaussian and exponential decay components.

    Args:
        t: Time coordinate.
        A: Amplitude of the oscillation.
        f: Frequency of the oscillation.
        phi: Phase of the oscillation.
        offset: Constant offset of the oscillation.
        T_exp: Exponential decay time.
        T_gauss: Gaussian decay time.

    Returns:
        Value at given times.

    """
    return A * np.sin(2 * np.pi * t * f + phi) * np.exp(-t / T_exp - (t / T_gauss) ** 2) + offset


class GaussExpSineModel(BaseModel):
    r"""Decaying sine model with both gaussian and exponential decay coefficients.

    .. math:: f(t) = \sin(2\pi f t + \phi) e^{-t/T_{\text{exp}} - (t/T_{\text{gauss}})^2}

    """

    __doc__ = lmfit.models.COMMON_INIT_DOC

    def __init__(self, independent_vars: list[Literal["t"]] | None = None, **kwargs) -> None:
        if independent_vars is None:
            independent_vars = ["t"]
        super().__init__(decaying_sine, independent_vars=independent_vars, param_names=DECAYING_SINE_PARAMS, **kwargs)
        self.set_param_hint(name="A", value=1, min=0, vary=True)
        self.set_param_hint(name="f", value=-3e6, vary=True)
        self.set_param_hint(name="phi", value=0, min=-2 * np.pi, max=2 * np.pi, vary=True)
        self.set_param_hint(name="offset", value=0.5, vary=True)
        self.set_param_hint(name="T_exp", value=np.inf, min=1e-15, vary=True)
        self.set_param_hint(name="T_gauss", value=np.inf, min=1e-15, vary=True)
        # T_2 is the time for which exp(-t/T_exp - (t/T_gauss)**2) = exp(-1)
        self.set_param_hint(
            name="T_2",
            expr="(-1/T_exp + sqrt(1/T_exp**2 + 4/T_gauss**2))/(2/T_gauss**2) if T_gauss !=inf else T_exp",
            vary=False,
        )

    def guess(self, data: np.ndarray, t: np.ndarray, *args, **kws) -> lmfit.parameter.Parameters:
        """Initial guess for the model.

        Uses zero-padded fft and peak-detection to guess initial values for the phase and
        frequency of the oscillating signal.
        """
        (
            f_guess,
            A_guess,
            phi_guess,
            offset_guess,
            T_exp_guess,
            T_gauss_guess,
        ) = frequency_guess_with_fft(data, t)

        # ``T_2`` is left out: its hint ties it to the decay times with an expression, and a guess for it
        # would cut that tie. ``make_params`` drops the guesses for the parameters hints pin.
        params = self.make_params(
            A=A_guess,
            f=f_guess,
            phi=phi_guess,
            offset=offset_guess,
            T_exp=T_exp_guess,
            T_gauss=T_gauss_guess,
        )

        return lmfit.models.update_param_vals(params, self.prefix, **kws)

    def post_fit(self, fitresult: lmfit.model.ModelResult) -> None:
        fitresult.params.add(
            f"{self.prefix}T_2",
            expr=prefix_expr(
                self, "(-1/T_exp + sqrt(1/T_exp**2 + 4/T_gauss**2))/(2/T_gauss**2) if T_gauss !=inf else T_exp"
            ),
        )


def abs_decay(
    n_gates: int, pulse_duration: float, amplitude: float, offset: float, exp_decay_time: float, gauss_decay_time: float
) -> float:
    """Absolute decaying function.

    Models a decay which contains both Gaussian and exponential decay components and can only be positive.

    Args:
        n_gates: Coordinate for number of gates.
        pulse_duration: duration of a single CZ gate.
        amplitude: Amplitude of the oscillation.
        offset: Constant offset of the oscillation.
        exp_decay_time: Exponential decay time.
        gauss_decay_time: Gaussian decay time.

    Returns:
        Value at given times.

    """
    return np.abs(
        amplitude
        * np.exp(-n_gates * pulse_duration / exp_decay_time - (n_gates * pulse_duration / gauss_decay_time) ** 2)
        + offset
    )


class GaussExpAbsModel(BaseModel):
    r"""Absolute value decaying model with both gaussian and exponential decay coefficients.

    .. math::
    f(t) = {\rm e}^{-t/T_{\rm exp} - (t/T_{\rm gauss})^2}

    """

    __doc__ = lmfit.models.COMMON_INIT_DOC

    def __init__(self, independent_vars: list[Literal["n_gates"]] | None = None, **kwargs) -> None:
        if independent_vars is None:
            independent_vars = ["n_gates"]
        super().__init__(abs_decay, independent_vars=independent_vars, param_names=ABS_DECAY_PARAMS, **kwargs)
        self.set_param_hint(name="amplitude", value=0.5, min=0, vary=True)
        self.set_param_hint(name="offset", value=0.0, vary=True)
        self.set_param_hint(name="exp_decay_time", value=np.inf, min=1e-15, vary=True)
        self.set_param_hint(name="gauss_decay_time", value=np.inf, min=1e-15, vary=True)
        self.set_param_hint(name="pulse_duration", value=4e-8, vary=False)
        # T_2 is the time for which exp(-t/exp_decay_time - (t/gauss_decay_time)**2) = exp(-1)
        self.set_param_hint(
            name="T_2",
            expr="(-1/exp_decay_time + sqrt(1/exp_decay_time**2 + 4/"
            "gauss_decay_time**2))/(2/gauss_decay_time**2) if "
            "gauss_decay_time !=inf else exp_decay_time",
            vary=False,
        )
        self.set_param_hint(
            name="error_per_gate",
            expr="pulse_duration/exp_decay_time + (pulse_duration/gauss_decay_time)**2",
            vary=False,
        )

    def guess(self, data: np.ndarray, n_gates: np.ndarray, *args, **kws) -> lmfit.parameter.Parameters:
        """Initial guess for the model.

        Uses zero-padded fft and peak-detection to guess initial values for the phase and
        frequency of the oscillating signal.
        """
        (
            _,
            amplitude_guess,
            _,
            offset_guess,
            exp_decay_time_guess,
            gauss_decay_time_guess,
        ) = frequency_guess_with_fft(data, n_gates)

        pulse_duration = self.param_hints["pulse_duration"]["value"]
        # ``T_2`` and ``error_per_gate`` are left out: their hints tie them to the decay times with
        # expressions, and a guess for either would cut that tie.
        params = self.make_params(
            amplitude=amplitude_guess,
            offset=offset_guess,
            exp_decay_time=exp_decay_time_guess * pulse_duration,
            gauss_decay_time=gauss_decay_time_guess * pulse_duration,
        )

        return lmfit.models.update_param_vals(params, self.prefix, **kws)

    def post_fit(self, fitresult: lmfit.model.ModelResult) -> None:
        fitresult.params.add(
            f"{self.prefix}T_2",
            expr=prefix_expr(
                self,
                "(-1/exp_decay_time + sqrt(1/exp_decay_time**2 + 4/"
                "gauss_decay_time**2))/(2/gauss_decay_time**2) if "
                "gauss_decay_time !=inf else exp_decay_time",
            ),
        )
        fitresult.params.add(
            f"{self.prefix}error_per_gate",
            expr=prefix_expr(self, "pulse_duration/exp_decay_time + (pulse_duration/gauss_decay_time)**2"),
        )
