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
"""Lmfit model for Nyquist sawtooth."""

from typing import Any, Literal

import lmfit
import numpy as np
import scipy.signal

from .model_utils import BaseModel

NYQUIST_PARAMS = ["nyquist_freq", "phi", "width"]


def nyquist_sawtooth(t: np.ndarray, nyquist_freq: float, phi: float, width: float) -> np.ndarray:
    """Sawtooth model oscillating between zero and the Nyquist frequency.

    Args:
        t: The virtual detuning.
        nyquist_freq: The Nyquist frequency.
        phi: The phase offset of the sawtooth waveform.
        width: The width of the rising ramp, see `https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sawtooth.html`.

    Returns:
        Sawtooth waveform.

    """
    if np.isclose(nyquist_freq, 0, atol=1e-15):
        return np.zeros_like(t)
    return nyquist_freq / 2 * scipy.signal.sawtooth(np.pi / nyquist_freq * t + phi, width=width) + nyquist_freq / 2


class NyquistSawtoothModel(BaseModel):
    r"""Fit a sawtooth model oscillating between zero and the Nyquist frequency.

    .. math::
    f(t) = f_{\rm nyquist}/2 {\rm sawtooth} (\pi/ f_{\rm nyquist} t + \phi) + f_{\rm nyquist}/2

    """

    __doc__ = lmfit.models.COMMON_INIT_DOC

    def __init__(self, independent_vars: list[Literal["t"]] | None = None, **kwargs):
        if independent_vars is None:
            independent_vars = ["t"]
        super().__init__(nyquist_sawtooth, independent_vars=independent_vars, **kwargs)
        self.set_param_hint(name="nyquist_freq", value=1e6, min=0)
        self.set_param_hint(name="width", value=0.5, min=0, max=1)
        self.set_param_hint(name="phi", value=0, min=-np.pi, max=np.pi)

    def guess(self, data: np.ndarray, t: np.ndarray, *args, **kws) -> lmfit.parameter.Parameters:
        """Initial guess for the sawtooth model.

        Since the nyquist frequency can be calculated from the sampling rate of the time axis, we do not need to
        implement a guess value for it, but we can fix it to the correct value in the experiment analysis. We typically
        have a symmetric sawtooth, so the width should be 0.5 and if not can be set in the analysis as well.
        Therefore, we only need a guess value for the phase, which can be estimated by finding the value in the t array,
        for which the extracted frequency from FFT (data) is minimal. Close to this t value is a minimum of the
        sawtooth. Since the sawtooth has a minimum e.g. at phi = 0, we can calculate the phase guess by setting phi=0.
        Then as the last step we make sure that the phi_guess is inside the interval -pi to pi.
        """
        temp_params = self.make_params()

        t_min = t[np.argmin(data)]
        phi_guess = -np.pi * t_min / temp_params[f"{self.prefix}nyquist_freq"].value
        phi_guess_modulo = (phi_guess + np.pi) % (2 * np.pi) - np.pi

        phi_guess_attrs: dict[str, Any] = {
            "value": phi_guess_modulo,
            "min": phi_guess_modulo - np.pi,
            "max": phi_guess_modulo + np.pi,
        }

        params = self.make_params(phi=phi_guess_attrs)

        return lmfit.models.update_param_vals(params, self.prefix, **kws)
