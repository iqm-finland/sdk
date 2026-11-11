# Copyright 2024 IQM
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Waveform definitions.

This module defines some waveforms that don't have special serialization, and reimports
waveforms that do from :mod:`iqm.models.playlist.waveforms`.
See the link for documentation of waveforms that don't appear here.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

from iqm.models.playlist.waveforms import (
    CanonicalWaveform,  # noqa: F401
    Constant,  # noqa: F401
    CosineRiseFall,  # noqa: F401
    CosineRiseFallDerivative,  # noqa: F401
    Gaussian,  # noqa: F401
    GaussianDerivative,  # noqa: F401
    GaussianSmoothedSquare,  # noqa: F401
    ModulatedCosineRiseFall,  # noqa: F401
    Samples,  # noqa: F401
    Slepian,  # noqa: F401
    TruncatedGaussian,  # noqa: F401
    TruncatedGaussianDerivative,  # noqa: F401
    TruncatedGaussianSmoothedSquare,  # noqa: F401
    Waveform,
)
import numpy as np
import scipy.signal as ss

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Cosine(Waveform):
    r"""Periodic sinusoidal waveform which defaults to cosine.

    The use case for this waveform is to do manual modulation of other waveforms.

    .. math::
        f(t) = \cos(2\pi \: f \: t + \phi)

    where :math:`f` is the frequency, and :math:`\phi` the phase of the wave.

    Args:
        frequency: frequency of the wave, in units of inverse sampling window duration
        phase: phase of the wave, in radians

    """

    frequency: float
    phase: float = 0.0

    def _sample(self, sample_coords: np.ndarray) -> np.ndarray:
        return np.cos(2 * np.pi * self.frequency * sample_coords + self.phase)

    @staticmethod
    def non_timelike_attributes() -> dict[str, str]:
        return {"frequency": "Hz", "phase": "rad"}


@dataclass(frozen=True)
class PolynomialCosine(Waveform):
    r"""Polynomial of a periodic sinusoidal waveform which defaults to cosine.

    .. math::
        f(t) = P(\cos(2\pi \: f \: t + \phi))

    where :math:`P(x)` is a polynomial, :math:`f` is the frequency, and :math:`\phi` the phase of the wave.

    Args:
        frequency: frequency of the wave, in units of inverse sampling window duration
        phase: phase of the wave, in radians

    """

    frequency: float
    coefficients: np.ndarray
    phase: float = 0.0

    def _sample(self, sample_coords: np.ndarray) -> np.ndarray:
        cosine = np.cos(2 * np.pi * self.frequency * sample_coords + self.phase)
        return np.polynomial.polynomial.polyval(cosine, self.coefficients)

    @staticmethod
    def non_timelike_attributes() -> dict[str, str]:
        return {"frequency": "Hz", "phase": "rad", "coefficients": ""}


@dataclass(frozen=True)
class PiecewiseConstant(Waveform):
    r"""Piecewise constant waveform.

    The values are assumed to be in the range :math:`[-1, 1]`, and the changepoints are
    assumed to be in the Nyquist-zone of the duration,
    i.e. in the range [-`duration`/2, `duration`/2]

    Args:
        changepoints: Array of the changepoints of the piecewise constant function.
        values: Array of the values of the piecewise constant function.
        Must have one more element than ``changepoints``.

    """

    changepoints: np.ndarray
    values: np.ndarray

    def __post_init__(self):
        if len(self.values) != len(self.changepoints) + 1:
            raise ValueError("The number of values must be one more than the number of changepoints.")

    @staticmethod
    def non_timelike_attributes() -> dict[str, str]:
        return {
            "values": "",
        }

    def _sample(self, sample_coords: np.ndarray) -> np.ndarray:
        condlist = []
        # Before first changepoint
        condlist.append(sample_coords < self.changepoints[0])

        condlist.extend(
            [
                (sample_coords >= self.changepoints[i]) & (sample_coords < self.changepoints[i + 1])
                for i in range(len(self.changepoints) - 1)
            ]
        )

        condlist.append(sample_coords >= self.changepoints[-1])

        funclist = (
            [self.values[0]] + [self.values[i + 1] for i in range(len(self.changepoints) - 1)] + [self.values[-1]]
        )
        return np.piecewise(sample_coords, condlist, funclist)


@dataclass(frozen=True)
class Chirp(Waveform):
    r"""Linear chirp.

     .. math:: f(t) = A \: \omega[\alpha, N] \: \cos(2\pi \int (f_{0} + (f_{1} - f_{0}) t) \: \mathrm{d}t),

    where :math:`\omega[\alpha, N]` is a cosine-tapered window. For :math:`\alpha = 1` it becomes rectangular,
    and for :math:`\alpha = 0` it becomes a Hann (or raised cosine) window.

    The chirp pulse is valued inside the Nyquist zone, such that :math:`f_{0}` and :math:`f_{1}` are constrained
    in the range :math:`[-0.5, 0.5]`.

    Args:
        freq_start: Initial frequency of the chirp waveform in the Nyquist zone.
        freq_stop: Final frequency of the chirp waveform in the Nyquist zone.
        alpha: Alpha parameter of the cosine-tapered window. Defaults to 0.05.
        phase: Phase of the waveform. Defaults to 0

    """

    freq_start: float
    freq_stop: float
    alpha: float = 0.05
    phase: float = 0

    def _sample(self, sample_coords: np.ndarray) -> np.ndarray:
        chirpfreq = np.linspace(self.freq_start, self.freq_stop, len(sample_coords))
        chirpphase = 2 * np.pi * np.cumsum(chirpfreq) + self.phase
        wave = np.exp(1j * chirpphase) * ss.windows.tukey(len(sample_coords), self.alpha)
        return wave.real

    @staticmethod
    def non_timelike_attributes() -> dict[str, str]:
        return {
            "alpha": "",
            "phase": "",
            "freq_start": "",
            "freq_stop": "",
        }


@dataclass(frozen=True)
class ChirpImag(Chirp):
    """Imaginary part of the linear chirp, which sets the phase to $-\\pi/2$.

    Attributes:
        phase: Phase of the pulse. Defaults to $\\pi/2$

    """  # noqa: D301

    phase: float = -np.pi / 2


@dataclass(frozen=True)
class CosineRise(Waveform):
    r"""Cosine rise waveform.

    This waveform assumes that during its duration, the only thing happening is signal rising to the required
    amplitude.
    The waveform is made for pairing with 'Constant' waveform to enable arbitrarily long pulses with smooth rise part.
    The rise time is equal to pulse duration.

    """

    def _sample(self, sample_coords: np.ndarray) -> np.ndarray:
        return 0.5 + 0.5 * np.sin(np.pi * sample_coords)


@dataclass(frozen=True)
class CosineFall(Waveform):
    r"""Cosine fall waveform.

    This waveform assumes that during its duration, the only thing occurring is signal falling to 0.
    The waveform is made for pairing with 'Constant' waveform to enable arbitrarily long pulses with smooth fall part.
    The fall time is equal to pulse duration.
    """

    def _sample(self, sample_coords: np.ndarray) -> np.ndarray:
        return 0.5 - 0.5 * np.sin(np.pi * sample_coords)


@dataclass(frozen=True)
class CosineRiseFlex(Waveform):
    r"""Cosine Rise waveform with an extra duration buffer.

    The waveform is a piecewise function: (buffer, cosine rise, flat plateau), where:

    - buffer is a 'leftover' constant signal with amplitude = 0, with duration of ``duration - full_width``
    - cosine rise is a cosine rise pulse with a duration of ``rise_time``
    - flat plateau is a constant signal with amplitude = 1, with duration of ``full_width - rise_time``

    Args:
        rise_time: rise time of the waveform
        full_width: combined duration of the cosine rise time and the flat plateau

    Raises:
        ValueError: Error is raised if full_width or rise_time is more than duration

    """

    rise_time: float
    full_width: float

    def _sample(self, sample_coords: np.ndarray) -> np.ndarray:
        flat_part_duration = np.abs(self.full_width) - np.abs(self.rise_time)
        rise_time_duration = np.abs(self.rise_time)
        dead_wait_time = 1 - np.abs(self.full_width)

        if dead_wait_time >= 0:
            return np.piecewise(
                sample_coords,
                [
                    sample_coords <= 0.5 - flat_part_duration - rise_time_duration,
                    sample_coords > 0.5 - flat_part_duration - rise_time_duration,
                    sample_coords >= 0.5 - flat_part_duration,  # flat carry-over from the Constant
                ],
                [
                    0,
                    lambda oc: 0.5 - 0.5 * np.cos(np.pi / rise_time_duration * (oc - dead_wait_time + 0.5)),
                    1,
                ],
            )
        elif (flat_part_duration + dead_wait_time > 0) and (1 - rise_time_duration >= 0):
            raise ValueError("Full width is more than duration")
        else:
            raise ValueError("Rise time is more than duration")


@dataclass(frozen=True)
class CosineFallFlex(Waveform):
    r"""Cosine fall waveform with an extra duration buffer.

    The waveform is a piecewise function: (flat plateau, cosine fall, buffer), where:

    - buffer is a 'leftover' constant signal with amplitude = 0, generally with duration of ``duration - full_width``
    - cosine fall is a cosine fall pulse with a duration of ``rise_time``
    - flat plateau is a constant signal with amplitude = 1, generally with duration of ``full_width - rise_time``

    Args:
        rise_time: rise time of the waveform
        full_width: combined duration of the cosine fall time and the flat plateau

    Raises:
        ValueError: Error is raised if full_width or rise_time is more than duration

    """

    rise_time: float
    full_width: float

    def _sample(self, sample_coords: np.ndarray) -> np.ndarray:
        flat_part_duration = max(np.abs(self.full_width) - np.abs(self.rise_time), 0)
        rise_time_duration = np.abs(self.rise_time)
        dead_wait_time = 1 - flat_part_duration - rise_time_duration

        if dead_wait_time >= 0:
            return np.piecewise(
                sample_coords,
                [
                    sample_coords <= -0.5 + flat_part_duration,  # flat corry-over from the Constant
                    sample_coords > -0.5 + flat_part_duration,
                    sample_coords >= -0.5 + flat_part_duration + rise_time_duration,
                ],
                [
                    1,
                    lambda oc: 0.5 + 0.5 * np.cos(np.pi / rise_time_duration * (oc - flat_part_duration + 0.5)),
                    0,
                ],
            )
        elif (flat_part_duration + dead_wait_time > 0) and (1 - rise_time_duration >= 0):
            raise ValueError("Full width is more than duration")
        else:
            raise ValueError("Rise time is more than duration")


@dataclass(frozen=True)
class SuperGaussianRiseFall(Waveform):
    r"""Super-Gaussian waveform with a flat top, truncated to zero when the waveform drops below a cutoff.

    .. math::
        f(t) =
        \begin{cases}
          \exp\left(-\left(\frac{|t - c| - p/2}{\sqrt{2}\sigma}\right)^{2n}\right) & |t - c| - p/2 \in [0, r]\\
          1 & |t - c| \le p/2\\
          0 & \text{otherwise}
        \end{cases}

    where :math:`c` is :attr:`center_offset`, :math:`n` is :attr:`order`, and
    the plateau width :math:`p = w - 2r`, where :math:`w` is :attr:`full_width` and
    :math:`r` is :attr:`rise_time`.

    The scale parameter :math:`\sigma = r / (\sqrt{2}\ln(1/\epsilon)^{1/(2n)})` is determined by the rise time
    and :attr:`cutoff`, denoted as :math:`\epsilon`.
    During the rise time the rising half of the super-Gaussian rises from :math:`\epsilon` to :math:`1`.

    If the plateau width :math:`p < 0`, a warning is logged and the middle of the waveform is cut:
    the rise and fall still start and end at ``cutoff`` but meet at the center before reaching full amplitude.

    Args:
        rise_time: Duration of each edge from ``cutoff`` to full amplitude (or back).
        order: Order of the super-Gaussian, ``order==1`` produces the standard Gaussian.
        full_width: Duration of the support of the waveform, from the point where the rising edge
            reaches ``cutoff`` to the point where the falling edge returns to ``cutoff``.
        center_offset: The waveform is centered at this offset from the midpoint of the sampling window.
        cutoff: Amplitude below which the waveform is truncated to zero.

    """

    rise_time: float
    order: int
    full_width: float
    center_offset: float = 0.0
    cutoff: float = 1e-5

    @staticmethod
    def non_timelike_attributes() -> dict[str, str]:
        return {"order": "", "cutoff": ""}

    def _sample(self, sample_coords: np.ndarray) -> np.ndarray:
        sigma = self.rise_time / (np.sqrt(2) * np.log(1 / self.cutoff) ** (1 / (2 * self.order)))
        flat_width = self.full_width - 2 * self.rise_time
        if flat_width < 0:
            logger.warning(
                "full_width %s is shorter than the rise and fall (%s); the peak stays below full amplitude.",
                self.full_width,
                2 * self.rise_time,
            )
        distance = np.abs(sample_coords - self.center_offset)
        edge_distance = np.maximum(distance - flat_width * 0.5, 0.0)
        values = np.exp(-((edge_distance / (np.sqrt(2) * sigma)) ** (2 * self.order)))
        return np.where(distance <= self.full_width * 0.5, values, 0.0)
