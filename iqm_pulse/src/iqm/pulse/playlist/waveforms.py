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
    Gaussian,  # noqa: F401
    GaussianDerivative,  # noqa: F401
    GaussianSmoothedSquare,  # noqa: F401
    Samples,  # noqa: F401
    TruncatedGaussian,  # noqa: F401
    TruncatedGaussianDerivative,  # noqa: F401
    TruncatedGaussianSmoothedSquare,  # noqa: F401
    Waveform,
)
import numpy as np
from scipy.integrate import romb
from scipy.interpolate import interp1d
import scipy.signal as ss

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CosineRiseFallDerivative(Waveform):
    r"""Derivative of a waveform that has a sinusoidal rise and fall, and a constant part in between.

    .. math::
        f(t) =
        \begin{cases}
          -\sin(\pi (t - c + p / 2) / r) & t - (c - p / 2) \in [-r, 0]\\
          -\sin(\pi (t - c - p / 2) / r) & t - (c + p / 2) \in [0, r]\\
          0 & \text{otherwise}
        \end{cases}

    where :math:`c` is :attr:`center_offset`, :math:`r` is :attr:`rise_time`, and :math:`p` is the plateau width,
    calculated via :math:`p :=` :attr:`full_width` - 2 * :attr:`rise_time`.

    Its values are in :math:`[-1, 1]`.


    Args:
        full_width: Duration of the support of the pulse, >= 2 * :attr:`rise_time`.
        rise_time: Duration of the sinusoidal rise (and fall) part of the waveform, >= 0.
        center_offset: The waveform is centered at this offset from the midpoint of the sampling window.

    """

    full_width: float
    rise_time: float
    center_offset: float = 0.0

    def _sample(self, sample_coords: np.ndarray) -> np.ndarray:
        offset_coords = sample_coords - self.center_offset
        fw = np.abs(self.full_width)
        rt = np.abs(self.rise_time)
        pw = fw - 2 * rt

        if pw < 0:
            logging.getLogger(__name__).warning(
                "Since full width is smaller than twice the rise time, the derivative of the"
                "CosineRiseFall waveform jumps at the symmetry point of the waveform."
            )
            waveform = np.piecewise(
                offset_coords,
                [
                    offset_coords <= -fw / 2,
                    offset_coords > -fw / 2,
                    offset_coords > 0,
                    offset_coords >= fw / 2,
                ],
                [
                    0,
                    lambda oc: -np.sin(np.pi / rt * (oc + pw / 2)),
                    lambda oc: -np.sin(np.pi / rt * (oc - pw / 2)),
                    0,
                ],
            )
        else:
            waveform = np.piecewise(
                offset_coords,
                [
                    offset_coords <= -pw / 2 - rt,
                    offset_coords > -pw / 2 - rt,
                    offset_coords >= -pw / 2,
                    offset_coords > pw / 2,
                    offset_coords >= pw / 2 + rt,
                ],
                [
                    0,
                    lambda oc: -np.sin(np.pi / rt * (oc + pw / 2)),
                    0,
                    lambda oc: -np.sin(np.pi / rt * (oc - pw / 2)),
                    0,
                ],
            )
        return waveform


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

        for i in range(len(self.changepoints) - 1):
            condlist.append((sample_coords >= self.changepoints[i]) & (sample_coords < self.changepoints[i + 1]))

        condlist.append(sample_coords >= self.changepoints[-1])

        funclist = (
            [self.values[0]] + [self.values[i + 1] for i in range(len(self.changepoints) - 1)] + [self.values[-1]]
        )
        return np.piecewise(sample_coords, condlist, funclist)


@dataclass(frozen=True)
class Slepian(Waveform):
    r"""Slepian waveform, which minimizes non-adiabatic errors during a gate.

    It is assumed that the user has done the minimization in a prior step, such that the optimal :math:`\lambda_n` for
    a specific length :math:`\tau_\text{pulse}` (in the accelerated frame) is known. This class then reconstructs the
    waveform with the following steps:

    1.  Calculate :math:`\theta(\tau)` (Slepian in the accelerated frame and in :math:`\theta` space)
    2.  Calculate :math:`t(\tau)` (mapping time in the accelerated frame to time in the lab frame)
    3.  Interpolate :math:`\theta(t)` (Slepian in the lab frame and in :math:`\theta` space)
    4.  Calculate :math:`f(t)` (Slepian in the lab frame and in frequency space)
    5.  Calculate :math:`V(t)` (Slepian in the lab frame and in voltage space)

    Since the waveform is normalized, any voltage pre-factor should go into the pulse amplitude.

    The user is advised to look up :cite:`Martinis2014` for further details, since the derivation is mathematically
    heavy.

    Args:
        full_width: Duration of the support of the waveform.
        lambda_1: First coefficient of Slepian waveform.
        lambda_2: Second coefficient of Slepian waveform.
        frequency_initial_normalized: Initial frequency of the pulsed component (usually coupler),
            normalized by the maximum frequency of the pulsed component.
        frequency_to_minimize_normalized: Frequency of the static component (usually qubit) which to
            minimize the leakage from/to, normalized by the maximum frequency of the pulsed component.
        coupling_strength_normalized: Coupling strength between pulsed component and static component,
            normalized by the maximum frequency of the pulsed component.
        center_offset: The waveform is centered at this offset from the midpoint of the sampling window.
        squid_asymmetry: Squid asymmetry.

    """

    full_width: float
    lambda_1: float
    lambda_2: float
    frequency_initial_normalized: float
    frequency_to_minimize_normalized: float
    coupling_strength_normalized: float
    center_offset: float = 0
    squid_asymmetry: float = 0

    @staticmethod
    def non_timelike_attributes() -> dict[str, str]:
        return {
            "lambda_1": "",
            "lambda_2": "",
            "frequency_initial_normalized": "",
            "frequency_to_minimize_normalized": "",
            "coupling_strength_normalized": "",
            "squid_asymmetry": "",
        }

    def _sample(self, sample_coords: np.ndarray) -> np.ndarray:
        fw = np.abs(self.full_width)
        offset_coords = sample_coords - self.center_offset
        return np.piecewise(
            offset_coords,
            [(offset_coords <= -fw / 2) | (offset_coords >= fw / 2)],
            [0, lambda coords: self._sample_normalized_slepian(coords / fw + 0.5)],
        )

    def _sample_normalized_slepian(self, coords: np.ndarray) -> np.ndarray:
        """Calculate the samples of the actual Slepian waveform.

        Args:
            coords: normalized sample coordinates, in :math:`[0, 1]`
        Returns:
            samples of the Slepian waveform

        """

        def freq_to_voltage(f_normalized: float | np.ndarray, d: float) -> float | np.ndarray:
            """Map frequencies to normalized voltages.

            Here, we use the formula from `exa.core.calibration_tools.qubit_utils.simple_transmon_ge_frequency`, and
            solve the equation for phi.
            Note that we don't need the voltage period, since it just scales the amplitude, which is anyways normalized
            for every pulse to 1. Furthermore, we don't need the voltage offset, since we are only interested in the
            frequency difference from the initial frequency.

            Args:
                f_normalized: Frequency to be converted to voltage normalized over the sweet spot frequency.
                d: Squid asymmetry.

            Returns:
                Voltage corresponding to the input frequency.

            """
            return -np.arccos(np.sqrt((f_normalized**4 - d**2) / (1 - d**2)))

        # build an interpolating mapping from t to theta
        tau_array = np.linspace(0, 1, 51)  # tau is normalized to [0, 1]
        interp_t_to_theta = interp1d(
            self._t_tau(tau_array), self._theta_tau(tau_array), kind="linear", bounds_error=True
        )

        # the waveform is scaled to the required duration
        t = coords * self._t_tau(1)
        frequency = self.frequency_to_minimize_normalized + 2 * self.coupling_strength_normalized / np.tan(
            interp_t_to_theta(t)
        )
        slepian_samples = freq_to_voltage(frequency, self.squid_asymmetry) - freq_to_voltage(
            self.frequency_initial_normalized, self.squid_asymmetry
        )
        return np.abs(slepian_samples / np.max(np.abs(slepian_samples)))

    def _t_tau(self, tau: float | np.ndarray, num_samples: int = 2**7 + 1) -> np.ndarray:
        r"""Convert time in the accelerated frame to the lab frame (real time).

        Since the conversion is defined via an integral, finding an analytical solution was not possible. Therefore, we
        integrate the expression numerically with :func:`scipy.integrate.romb`.

        Args:
            tau: Time in the accelerated frame.
            num_samples: Number of samples for the numerical integration. Must be of the form 2 ** k + 1.

        Returns:
            ``tau`` converted to the lab frame

        """
        tau = np.atleast_1d(tau)
        t_array = []
        for tau_single in tau:
            if tau_single == 0:
                t_array.append(0)
            else:
                tau_samples = np.linspace(0, tau_single, num_samples)
                integrand = np.sin(self._theta_tau(tau_samples))
                t = romb(integrand, float(tau_samples[1] - tau_samples[0]))
                t_array.append(t)

        return np.abs(np.array(t_array))

    def _theta_tau(self, tau: np.ndarray) -> np.ndarray:
        r"""Parametrization of the Slepian waveform in the accelerated frame.

        The Slepian waveform is parametrized using Fourier base functions, where we only take the cosine terms into
        account (:cite:`Martinis2014` has shown that this is a reasonable assumption).
        Here, :math:`\lambda_n` is the amplitude of the :math:`n`-th term of the Fourier base function.
        It is usually sufficient to take only up to second order terms into account, i.e. only :math:`\lambda_1` and
        :math:`\lambda_2` are non-zero.

        Args:
            tau: Time in the accelerated frame, normalized to [0, 1] with tau_pulse.

        Returns:
            Slepian waveform in the theta space and accelerated frame.

        """
        lambdas = np.array([self.lambda_1, self.lambda_2])
        theta_i = np.arctan(
            2
            * self.coupling_strength_normalized
            / (self.frequency_initial_normalized - self.frequency_to_minimize_normalized)
        )
        return theta_i + np.sum(
            [lambda_i * (1 - np.cos(2 * np.pi * (i + 1) * tau)) for i, lambda_i in enumerate(lambdas)],
            axis=0,
        )


@dataclass(frozen=True)
class Chirp(Waveform):
    r"""Linear chirp, defined as

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

    def _sample(self, sample_coords):  # noqa: ANN001, ANN202
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
class ModulatedCosineRiseFall(Waveform):
    r"""Modulated Cosine Rise Fall waveform.

    This waveform takes the waveform :class:`CosineRiseFall` and modulates it with a cosine signal
    which then has parameters :attr:`frequency` and :attr:`phase`, additional to the parameters :attr:`full_width`,
    :attr:`rise_time`, and :attr:`center_offset`, see description of :class:`TruncatedGaussianSmoothedSquare` for
    further details.


    Args:
        full_width: Full width of the pulse, >= 2 * :attr:`rise_time`.
        rise_time: Duration of the sinusoidal rise (and fall) part of the waveform, >= 0.
        modulation_frequency: Modulation frequency.
        phase: Phase of the modulation.
        center_offset: The waveform is centered around this sampling window coordinate.
            If zero, the pulse is placed in the middle of the sampling window.

    """

    full_width: float
    rise_time: float
    modulation_frequency: float = 0.0
    phase: float = 0
    center_offset: float = 0.0

    def _sample(self, sample_coords: np.ndarray) -> np.ndarray:
        offset_coords = sample_coords - self.center_offset
        fw = np.abs(self.full_width)
        rt = np.abs(self.rise_time)
        pw = fw - 2 * rt

        if pw < 0:
            logging.getLogger(__name__).warning(
                "Since the full width is smaller than twice the rise time, the CosineRiseFall pulse does not reach its"
                "top.",
            )
            envelope = np.piecewise(
                offset_coords,
                [
                    offset_coords <= -fw / 2,
                    offset_coords > -fw / 2,
                    offset_coords > 0,
                    offset_coords >= fw / 2,
                ],
                [
                    0,
                    lambda oc: 0.5 + 0.5 * np.cos(np.pi / rt * (oc + pw / 2)),
                    lambda oc: 0.5 + 0.5 * np.cos(np.pi / rt * (oc - pw / 2)),
                    0,
                ],
            )
        else:
            envelope = np.piecewise(
                offset_coords,
                [
                    offset_coords <= -pw / 2 - rt,
                    offset_coords > -pw / 2 - rt,
                    offset_coords >= -pw / 2,
                    offset_coords > pw / 2,
                    offset_coords >= pw / 2 + rt,
                ],
                [
                    0,
                    lambda oc: 0.5 + 0.5 * np.cos(np.pi / rt * (oc + pw / 2)),
                    1,
                    lambda oc: 0.5 + 0.5 * np.cos(np.pi / rt * (oc - pw / 2)),
                    0,
                ],
            )
        return envelope * np.cos(2 * np.pi * self.modulation_frequency * sample_coords + self.phase)

    @staticmethod
    def non_timelike_attributes() -> dict[str, str]:
        return {"modulation_frequency": "Hz", "phase": "rad"}


@dataclass(frozen=True)
class CosineRise(Waveform):
    r"""Cosine Rise waveform.

    This waveform assumes that during its duration, the only thing happening is signal occurring to the required
    amplitude.
    The waveform is made for pairing with 'Constant' waveform to enable arbitrarily long pulses with smooth rise part.
    The rise time is equal to pulse duration.

    Args:
        rise_time: Dummy parameter, used only as due to a bug. FIXME it is not used, placed for resolving exa bug

    """

    def _sample(self, sample_coords: np.ndarray) -> np.ndarray:
        return 0.5 + 0.5 * np.sin(np.pi * sample_coords)


@dataclass(frozen=True)
class CosineFall(Waveform):
    r"""Cosine Rise waveform.

    This waveform assumes that during its duration, the only thing occurring is signal falling to 0.
    The waveform is made for pairing with 'Constant' waveform to enable arbitrarily long pulses with smooth fall part.
    The fall time is equal to pulse duration.
    """

    def _sample(self, sample_coords: np.ndarray) -> np.ndarray:
        return 0.5 - 0.5 * np.sin(np.pi * sample_coords)


@dataclass(frozen=True)
class CosineRiseFlex(Waveform):
    r"""Cosine Rise waveform with an extra duration buffer.

    The waveform is a piecewise function: |buffer|cosine rise|flat plateau|, where:
    - buffer is a 'leftover' constant signal with amplitude = 0, with duration of duration - full_width
    - cosine rise is a cosine rise pulse with a duration of rise_time
    - flat plateau is a constant signal with amplitude = 1, with duration of full_width - rise_time

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

    The waveform is a piecewise function: |flat plateau|cosine fall|buffer|, where:
    - buffer is a 'leftover' constant signal with amplitude = 0, generally with duration of duration - full_width
    - cosine fall is a cosine fall pulse with a duration of rise_time
    - flat plateau is a constant signal with amplitude = 1, generally with duration of full_width - rise_time

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
