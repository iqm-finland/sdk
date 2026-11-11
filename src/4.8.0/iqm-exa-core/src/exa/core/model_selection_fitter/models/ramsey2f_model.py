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
"""Lmfit model for Ramsey2f."""

from typing import Literal

import lmfit
import numpy as np

from .decaying_sine_model import decaying_sine
from .model_utils import BaseModel, frequency_guess_with_fft

RAMSEY_2F_PARAMS = ["f_q", "phi", "tau", "T_exp", "A", "offset_1", "offset_2"]


def ramsey_2f_response(  # noqa: PLR0913
    t: np.ndarray,
    f_d: np.ndarray,
    f_q: float,
    phi: float,
    tau: float,
    T_exp: float,
    A: float,
    offset_1: float,
    offset_2: float,
) -> np.ndarray:
    """Response function for a Ramsey experiment with two drive tones.

    If time coordinate does not start from zero, removes the offset and assumes
    that the phase accumulated before the start time can be accounted for using
    tau, the common additional delay from phase accumulation perspective.

    Args:
        t: The time coordinate values.
        f_d: Frequencies of the two drive tones.
        f_q: Frequency of the qubit.
        phi: Phase of the oscillation.
        tau: A constant offset in time to account for additional frequency-dependnet phase evolution
            not captured by the time coordinate.
        T_exp: Expontial decay coefficient, shared by both sine waves.
        A: Amplitude of the oscillation, shared by both sine waves.
        offset_1: Offset of the first wave.
        offset_2: Offset of the second wave.

    Returns:
        Values corresponding to the sines for both of the drive tones.

    """
    detuning = f_q - f_d
    offsets = np.asarray([offset_1, offset_2])
    y = decaying_sine(
        t[..., None] - t[0], A, detuning[None], 2 * np.pi * detuning[None] * tau + phi, offsets[None], T_exp, np.inf
    )
    return y


class Ramsey2fModel(BaseModel):
    """Model for simulteneously fitting two oscillating components.

    Finds a qubit frequency by fitting two exponentially decaying sine wave to two oscillating signals.
    The signals are assumed to be measured in a refence frame given by the independent variable
    ```f_d```. Simultaneously fits the data from the both measured oscillations, assuming the frequency difference
    calculated as ```f_d[1] - f_d[0]```.
    """

    __doc__ = lmfit.models.COMMON_INIT_DOC

    def __init__(self, independent_vars: list[Literal["t", "f_d"]] | None = None, **kwargs) -> None:
        if independent_vars is None:
            independent_vars = ["t", "f_d"]
        super().__init__(ramsey_2f_response, independent_vars=independent_vars, **kwargs)
        self.set_param_hint(name="f_q", value=0)
        self.set_param_hint(name="phi", value=np.pi / 2, min=-2 * np.pi, max=2 * np.pi, vary=False)
        self.set_param_hint(name="tau", value=0, min=0)
        self.set_param_hint(name="T_exp", value=50e-6, min=1e-15, vary=True)
        self.set_param_hint(name="A", value=1, min=0)
        self.set_param_hint(name="offset_1", value=0.5)
        self.set_param_hint(name="offset_2", value=0.5)

    def guess(self, data: np.ndarray, t: np.ndarray, f_d: np.ndarray, **kws) -> lmfit.parameter.Parameters:
        """Initial guess for the model.

        Uses frequency_guess_with_fft to first get initial parameters for both
        oscillating tones, and then derives the best guess for Ramsey2f model from those.

        Checks if one of the frequencies is likely zero and sets the frequency to be exactly
        zero.

        Values for the parameters the hints hold fixed are taken from those hints rather than guessed here,
        since a value the fit never moves belongs to whoever fixed it. Pinning ``phi`` at ``3 / 2 * np.pi``,
        for instance, is how a trace whose readout phase comes back flipped by pi is fitted.

        Args:
            data: 2D array to fit.
            t: Time coordinate.
            f_d: Frequency of the drive signal.
            kws: Values for the model parameters.

        """
        if len(f_d) != 2:
            raise ValueError("Supports only data with len(f_d) == 2.")

        (
            f_guess_1,
            A_guess_1,
            phi_guess_1,
            offset_guess_1,
            T_exp_guess_1,
            _,
        ) = frequency_guess_with_fft(data[:, 0], t - t[0])
        (
            f_guess_2,
            A_guess_2,
            phi_guess_2,
            offset_guess_2,
            T_exp_guess_2,
            _,
        ) = frequency_guess_with_fft(data[:, 1], t - t[0])

        phi_guess = np.pi / 2
        detuning = f_d[1] - f_d[0]

        better_index = np.argmax([A_guess_1, A_guess_2])
        one_frequency_is_zero = np.max([A_guess_1, A_guess_2]) > 5 * np.min([A_guess_1, A_guess_2])
        if one_frequency_is_zero:
            if better_index == 1:
                f_guess_1 = 0
                phi_guess_1 = phi_guess  # If frequency is zero, phi should be the ideal value
            else:
                f_guess_2 = 0
                phi_guess_2 = phi_guess  # If frequency is zero, phi should be the ideal value

        f_q_guess = (f_guess_1**2 - f_guess_2**2) / (2 * detuning) + f_d[0] + detuning / 2
        detuning_1 = f_q_guess - f_d[0]
        detuning_2 = f_q_guess - f_d[1]
        tau_guess_1 = (np.sign(detuning_1) * (phi_guess_1 - np.pi / 2)) / (2 * np.pi * np.abs(detuning_1))
        tau_guess_2 = (np.sign(detuning_2) * (phi_guess_2 - np.pi / 2)) / (2 * np.pi * np.abs(detuning_2))
        tau_guess = (
            tau_guess_1 if np.abs(detuning_1) > np.abs(detuning_2) else tau_guess_2
        )  # The estimate of tau is better for bigger absolute detuning
        guessed = {
            "f_q": f_q_guess,
            "phi": phi_guess,
            "tau": tau_guess,
            "T_exp": np.mean([T_exp_guess_1, T_exp_guess_2]),
            "A": np.mean([A_guess_1, A_guess_2]),
            "offset_1": offset_guess_1,
            "offset_2": offset_guess_2,
        }
        # Not every parameter is free to move: ``phi`` is fixed by default, and a hint can fix any other.
        # A fixed one keeps the value its hint sets rather than taking this guess.
        params = self.make_params(**guessed)

        return lmfit.models.update_param_vals(params, self.prefix, **kws)
