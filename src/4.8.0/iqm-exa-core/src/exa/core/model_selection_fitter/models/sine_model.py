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

from .model_utils import BaseModel, frequency_guess_with_fft

SINE_PARAMS = ["A", "f", "phi", "offset"]


def sine(
    t: np.ndarray,
    A: float,
    f: float | np.ndarray,
    phi: float | np.ndarray,
    offset: float | np.ndarray = 0,
) -> np.ndarray:
    """Sine function.

    Args:
        t: Time coordinate.
        A: Amplitude of the oscillation.
        f: Frequency of the oscillation.
        phi: Phase of the oscillation.
        offset: Offset of the oscillation.

    Returns:
        Value at given times.

    """
    return A * np.sin(2 * np.pi * t * f + phi) + offset


class SineModel(BaseModel):
    r"""Sine model.

    .. math:: f(t) = \sin(2\pi f t + \phi) + C

    ``phi`` can be fixed to a value by setting the ``vary`` hint to ``False``.
    """

    __doc__ = lmfit.models.COMMON_INIT_DOC

    def __init__(self, independent_vars: list[Literal["t"]] | None = None, **kwargs) -> None:
        if independent_vars is None:
            independent_vars = ["t"]
        super().__init__(sine, independent_vars=independent_vars, param_names=SINE_PARAMS, **kwargs)
        self.set_param_hint(name="A", value=1, min=0, vary=True)
        self.set_param_hint(name="f", value=-3e6, vary=True)
        self.set_param_hint(name="phi", value=0, min=-2 * np.pi, max=2 * np.pi, vary=True)

    def guess(self, data: np.ndarray, t: np.ndarray, *args, **kws) -> lmfit.parameter.Parameters:
        """Initial guess for the model.

        Uses zero-padded fft and peak-detection to guess initial values for the phase and
        frequency of the oscillating signal.
        """
        # ``frequency_guess_with_fft`` also returns decay-time guesses (for decaying-sine models), which the
        # plain sine model does not have and therefore discards.
        (f_guess, A_guess, phi_guess, offset_guess, _, _) = frequency_guess_with_fft(data, t)

        guesses = {"A": A_guess, "f": f_guess, "offset": offset_guess}
        # Respect a user-fixed phase: if the ``phi`` hint has ``vary=False``, keep the hinted value
        # instead of overwriting it with the guess.
        if self.param_hints.get("phi", {}).get("vary", True):
            guesses["phi"] = phi_guess
        params = self.make_params(**guesses)

        return lmfit.models.update_param_vals(params, self.prefix, **kws)
