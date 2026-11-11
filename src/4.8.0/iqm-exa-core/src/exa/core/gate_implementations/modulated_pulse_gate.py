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
"""Implementation of a "gate" which contains a modulated pulse."""

from dataclasses import dataclass

import numpy as np

from exa.common.data import Parameter
from iqm.pulse.gate_implementation import SinglePulseGate
from iqm.pulse.playlist.instructions import IQPulse
from iqm.pulse.playlist.waveforms import TruncatedGaussianSmoothedSquare, Waveform


@dataclass(frozen=True)
class ModulatedGaussianSmoothedSquare(Waveform):
    r"""Product of a cosine pulse with the :class:`iqm.pulse.playlist.waveforms.TruncatedGaussianSmoothedSquare`.

    The waveform is given by

    .. math::
        f(t) = \cos(2\pi \: f \: t + \phi)
                \; \text{TGSS}(t),

    where :math:`f` is the modulation frequency and :math:`\text{TGSS}` is the
    :class:`iqm.pulse.playlist.waveforms.TruncatedGaussianSmoothedSquare`
    Its values are in :math:`(-1, 1)`.

    Args:
        full_width: Duration of the non-zero part of the envelope in samples
        rise_time: Duration of the rising edge in samples
        center_offset: Center offset in samples
        frequency: Frequency of the cosine in samples
        phase: Phase of the cosine in samples

    """

    full_width: float
    rise_time: float
    frequency: float
    phase: float = 0.0
    center_offset: float = 0.0

    def _sample(self, sample_coords: np.ndarray) -> np.ndarray:
        cosine = np.cos(2 * np.pi * self.frequency * sample_coords + self.phase)
        gss = TruncatedGaussianSmoothedSquare(self.n_samples, self.full_width, self.rise_time, self.center_offset)
        return cosine * gss.sample()


class ModulatedTGSS(SinglePulseGate):
    """A "gate" which plays an IQPulse with GaussianSmoothedSquare waveform envelopes, modulated by some frequency.

    Args:
        pulse: The pulse to play.
        channel: name of drive channel on which the pulse is applied

    """

    parameters = {
        "duration": Parameter("", "Gate duration", "s"),
        "amplitude": Parameter("", "Pulse amplitude", ""),
        "full_width": Parameter("", "Pulse width", "s"),
        "rise_time": Parameter("", "Rise time", "s"),
        "center_offset": Parameter("", "Center offset of the gaussian smoothed square pulse", "s"),
        "frequency": Parameter("", "Modulation frequency", "Hz"),
        "phase": Parameter("", "Modulation phase", ""),
    }

    @classmethod
    def _get_pulse(  # type: ignore[override]
        cls,
        *,
        n_samples: int,
        amplitude: float,
        full_width: float,
        rise_time: float,
        center_offset: float,
        frequency: float,
        phase: float,
    ) -> IQPulse:
        """Returns an X_pi pulse for the e-f subspace."""
        # here we assume the sign convention of the mixer to be RF = I*cos - Q*sin,
        # therefore the waves here need to be I~cos and Q~sin
        wave_i = ModulatedGaussianSmoothedSquare(
            n_samples=n_samples,
            full_width=full_width,
            rise_time=rise_time,
            center_offset=center_offset,
            frequency=frequency,
            phase=phase,
        )
        wave_q = ModulatedGaussianSmoothedSquare(
            n_samples=n_samples,
            full_width=full_width,
            rise_time=rise_time,
            center_offset=center_offset,
            frequency=frequency,
            phase=phase - np.pi / 2,
        )
        return IQPulse(n_samples, wave_i=wave_i, wave_q=wave_q, scale_i=amplitude, scale_q=amplitude)
