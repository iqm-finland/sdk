# Copyright 2019-2025 IQM
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
"""Waveform definitions."""
# pylint: disable=no-name-in-module
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Waveform:
    r"""Describes the normalized shape of a real-valued control pulse.

    The shape is described by a function :math:`f: \mathbb{R} \to [-1, 1]` that comes with an implicit
    sampling window. :math:`f` maps time (measured in units of the sampling window duration and relative
    to its center point) to the value of the pulse at that time.

    Each Waveform subclass may have attributes that affect its shape. All time-like attributes are
    measured in units of the sampling window duration. The station method ``non_timelike_attributes``
    may be used to define non-timelike attributes and their units (in addition to `"s"`, `"Hz"` has a special
    behaviour in that such attributes will be converted to the units of the inverse of duration). Providing type
    hints to the waveform attributes is mandatory, as they are used in parsing the information for the
    ``GateImplementations``. Supported scalar attribute types are: ``int``, ``float``, ``complex``, ``str``, ``bool``.
    In addition, ``list[<scalar type>]`` is supported for all the aforementioned scalar types, and also
    ``numpy.ndarray``, in which case it is interpreted to contain complex numbers.

    When the Waveform is used by an instrument it is typically sampled using the :meth:`sample` method, which
    converts it into an array of :attr:`n_samples` equidistant samples, generated using the midpoint method,
    by evaluating the function :math:`f(t)` inside the sampling window :math:`t \in [-1/2, 1/2]`.
    The instruments will discretize the values of the samples to a finite, instrument-dependent resolution,
    typically 14--16 bits.

    Usually, it is sufficient for Waveforms to describe normalized waveforms (i.e. using the full
    value range :math:`[-1, 1]`), not including a scaling prefactor in the defining expression.
    Instead, the scaling should be specified as a parameter of the :class:`.Instruction` using the Waveform
    (e.g. :class:`.IQPulse`, :class:`.RealPulse`), thus allowing compilers to more efficiently
    re-use waveforms and utilize the available hardware support to perform such re-scaling in real time.
    """

    n_samples: int
    """Requested number of samples for the waveform. May be different from the duration (in samples) of the
    parent Instruction."""

    @staticmethod
    def non_timelike_attributes() -> dict[str, str]:
        """Mapping from waveform attributes to the units of their calibration data, unless that unit is second.

        Used to construct the parameters for the calibration data required by the waveform.
        By default all the waveform attributes are "timelike" (the unit for their calibration data is s).
        However, some waveform attributes can be dimensionless, e.g. the relative amplitudes in a sum of
        consisting of multiple terms, or frequency-like (calibration data has the unit 'Hz').
        If a Waveform subclass has non-timelike attributes, it needs to redefine this method.

        When the Waveform is constructed, all timelike calibration data is converted to units of the
        sampling window duration, and all frequency-like calibration data into units of inverse sampling
        window duration.
        """
        return {}

    def sample(self) -> np.ndarray:
        """Sample the waveform.

        Contains the boilerplate code for determining the sample coordinates,
        the actual sampling happens in :meth:`._sample`.

        Returns:
            ``self`` sampled in the window [-1/2, 1/2]
        """
        # midpoint sampling
        half_sample_duration = 0.5 / self.n_samples
        bound = 0.5 - half_sample_duration
        sample_coords = np.linspace(-bound, bound, self.n_samples)
        return self._sample(sample_coords)

    def _sample(self, sample_coords: np.ndarray) -> np.ndarray:
        """Actually samples the waveform.

        Args:
            sample_coords: coordinates of the samples to be returned

        Returns:
            array of samples (same shape as ``sample_coords``, ``dtype == float``)
        """
        raise NotImplementedError


CanonicalWaveform = Waveform
"""Alias for Waveform to emphasize the fact the waveforms defined in this module have their own serialisation."""


_CANONICAL_WAVEFORMS: set[type[CanonicalWaveform]] = set()
"""Waveform classes that are considered canonical (have their own serialisation). Use the decorator
``register_canonical_waveform`` to make a Waveform canonical."""


def register_canonical_waveform(cls: type[Waveform]) -> None:
    """Decorator for making a Waveform into a canonical waveform."""
    _CANONICAL_WAVEFORMS.add(cls)
    return cls


@register_canonical_waveform
@dataclass(frozen=True)
class Samples(CanonicalWaveform):
    """Custom pre-sampled waveform.

    This class can be used to represent an arbitrary waveform
    that is not supported with the predefined shapes of waveforms.
    """

    samples: np.ndarray

    def __init__(self, samples: np.ndarray):
        # pylint: disable=super-init-not-called
        # define the __init__ method explicitly to guarantee that self.n_samples has the correct value
        if samples.ndim != 1:
            raise ValueError(f"Incorrect shape {samples.shape} for waveform samples. Samples should be a 1D array.")
        samples.flags.writeable = False
        # frozen dataclass requires this
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "n_samples", len(samples))

    def __hash__(self):
        # NOTE: assumes the array size is not too large, otherwise hash(str(self.samples)) would be smarter
        return hash(self.samples.data.tobytes())

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Samples) and np.array_equal(self.samples, other.samples)

    def sample(self) -> np.ndarray:
        return self.samples

    def _sample(self, sample_coords: np.ndarray) -> np.ndarray:
        raise NotImplementedError  # not needed


@register_canonical_waveform
@dataclass(frozen=True)
class Gaussian(CanonicalWaveform):
    r"""Gaussian pulse.

    .. math::
        f(t) = e^{-\frac{(t - c)^2}{2 \sigma^2}},

    where :math:`c` is :attr:`center_offset`, and :math:`\sigma` is :attr:`sigma`.

    Args:
        sigma: gaussian standard deviation
        center_offset: center offset
    """

    sigma: float
    center_offset: float = 0.0

    def _sample(self, sample_coords: np.ndarray) -> np.ndarray:
        offset_coords = sample_coords - self.center_offset
        return np.exp(-0.5 * (offset_coords / self.sigma) ** 2)


@register_canonical_waveform
@dataclass(frozen=True)
class GaussianDerivative(CanonicalWaveform):
    r"""Derivative of a gaussian pulse.

    Normalized so that values are in :math:`[-1, 1]`.
    The normalization factor is :math:`\sigma \: \sqrt{e}`.

    .. math::
        f(t) = - \sqrt{e} \frac{t - c}{\sigma} e^{-\frac{(t - c)^2}{2 \sigma^2}},

    where :math:`c` is :attr:`center_offset`, and :math:`\sigma` is :attr:`sigma`.

    Args:
        sigma: gaussian standard deviation
        center_offset: center offset
    """

    sigma: float
    center_offset: float = 0.0

    def _sample(self, sample_coords: np.ndarray) -> np.ndarray:
        offset_coords = sample_coords - self.center_offset
        gaussian = np.exp(-0.5 * (offset_coords / self.sigma) ** 2)
        inner_derivative = -np.exp(0.5) * offset_coords / self.sigma
        return inner_derivative * gaussian


@register_canonical_waveform
@dataclass(frozen=True)
class TruncatedGaussian(CanonicalWaveform):
    r"""Gaussian pulse, where the decaying tails are removed by offsetting, truncating, and then rescaling the pulse
    slightly, so that the resulting waveform is zero where the original waveform reaches the threshold level, and
    beyond, while still reaching the same maximal pulse amplitude.  Currently, the threshold is fixed at
    :math:`g_0 = 0.003`.

    .. math::
        g(t) = e^{-\frac{(t - c)^2}{2 \sigma^2}},

    where :math:`c` is :attr:`center_offset`, and :math:`\sigma` is calculated via :math:`\sigma :=` :attr:`full_width`
    :math:`/ \sqrt{8 \text{ln}(1/g_0)}`.

    The waveform after offsetting, truncating and rescaling is given by

    .. math::
        f(t) = \text{max}(g(t) - g_0, 0) / (1 - g_0).

    where :math:`g_0` is the threshold level for the truncation.

    Args:
        full_width: Duration of the support of the pulse, >= 0.
        center_offset: The waveform is centered at this offset from the midpoint of the sampling window.
    """

    full_width: float
    center_offset: float = 0.0

    def _sample(self, sample_coords: np.ndarray) -> np.ndarray:
        threshold: float = 0.003
        offset_coords = sample_coords - self.center_offset
        gaussian_sigma = self.full_width / np.sqrt(8 * np.log(1 / threshold))
        waveform = np.exp(-0.5 * (offset_coords / gaussian_sigma) ** 2)
        return _subtract_threshold_and_rescale(waveform, threshold)


@register_canonical_waveform
@dataclass(frozen=True)
class TruncatedGaussianDerivative(CanonicalWaveform):
    r"""Derivative of a gaussian pulse, where the decaying tails are removed by offsetting, truncating, and then
    rescaling the pulse slightly, so that the resulting waveform is zero where the original waveform reaches the
    threshold level, and beyond, while still reaching the same maximal pulse amplitude. Currently, the threshold is
    fixed at :math:`g_0 = 0.003`.

    Normalized so that values are in :math:`[-1, 1]`.
    The normalization factor is :math:`\sigma \: \sqrt{e}`.

    .. math::
        f(t) = - \sqrt{e} \frac{t - c}{\sigma} e^{-\frac{(t - c)^2}{2 \sigma^2}},

    where :math:`c` is :attr:`center_offset`, and :math:`\sigma` is calculated via :math:`\sigma :=` :attr:`full_width`
    :math:`/ \sqrt{8 \text{ln}(1/g_0)}`.

    The waveform after offsetting, truncating and rescaling is given by

    .. math::
        f(t) = \text{max}(g(t) - g_0, 0) / (1 - g_0).

    where :math:`g_0` is the threshold level for the truncation.

    Args:
        full_width: Duration of the support of the pulse, >= 0.
        center_offset: The waveform is centered at this offset from the midpoint of the sampling window.
    """

    full_width: float
    center_offset: float = 0.0

    def _sample(self, sample_coords: np.ndarray) -> np.ndarray:
        threshold: float = 0.003
        offset_coords = sample_coords - self.center_offset
        gaussian_sigma = self.full_width / np.sqrt(8 * np.log(1 / threshold))
        gaussian = np.exp(-0.5 * (offset_coords / gaussian_sigma) ** 2)
        inner_derivative = -np.exp(0.5) * offset_coords / gaussian_sigma
        return inner_derivative * gaussian


@register_canonical_waveform
@dataclass(frozen=True)
class Constant(CanonicalWaveform):
    """Constant waveform.

    .. math::
        f(t) = 1
    """

    def _sample(self, sample_coords: np.ndarray) -> np.ndarray:
        return np.ones(len(sample_coords))


@register_canonical_waveform
@dataclass(frozen=True)
class GaussianSmoothedSquare(CanonicalWaveform):
    r"""Convolution of a square pulse and a gaussian pulse.

    One can think of it as a square pulse smoothed with a gaussian one, or vice versa.

    .. math::
        f(t) = \frac{1}{2} \left[
                  \text{erf}\left(\frac{t - (c - s / 2)}{\sqrt{2 \sigma^2}}\right)
                - \text{erf}\left(\frac{t - (c + s / 2)}{\sqrt{2 \sigma^2}}\right)
               \right],

    where :math:`\text{erf}` is the error function,
    :math:`c` is :attr:`center_offset`, :math:`s` is :attr:`square_width`, and :math:`\sigma` is :attr:`gaussian_sigma`.

    Its values are in :math:`(0, 1)`.

    Args:
        square_width: square pulse width
        gaussian_sigma: gaussian pulse standard deviation
        center_offset: The waveform is centered at this offset from the midpoint of the sampling window.
    """

    square_width: float
    gaussian_sigma: float
    center_offset: float = 0.0

    def _sample(self, sample_coords: np.ndarray) -> np.ndarray:
        rising_edge = self.center_offset - self.square_width / 2
        falling_edge = rising_edge + self.square_width
        return 0.5 * (
            scipy.special.erf((sample_coords - rising_edge) / (np.sqrt(2) * self.gaussian_sigma))
            - scipy.special.erf((sample_coords - falling_edge) / (np.sqrt(2) * self.gaussian_sigma))
        )


def _subtract_threshold_and_rescale(waveform: np.ndarray, threshold: float) -> np.ndarray:
    """Truncate waveform values smaller than `threshold` to zero and rescale the waveform.

    Subtracts `threshold` from `waveform` and sets negative values to zero, to define the start of the rising edge
    precisely and make the waveform have finite support. Finally, the waveform is rescaled such that the value 1.0 in
    the input waveform is unchanged in the transformation.

    Can only be applied to waveforms which are purely positive-valued, as otherwise the zero-crossing gets altered.
    """
    return np.maximum((waveform - threshold) / (1 - threshold), 0)


@register_canonical_waveform
@dataclass(frozen=True)
class TruncatedGaussianSmoothedSquare(CanonicalWaveform):
    r"""Convolution of a square pulse and a gaussian pulse, offset and truncated so that it has finite support.

    One can think of it as a square pulse smoothed with a gaussian one, or vice versa.
    The decaying tails are removed by offsetting, truncating, and then rescaling the pulse slightly,
    so that the resulting waveform is zero where the original waveform reaches the threshold level, and beyond, while
    still reaching the same maximal pulse amplitude.
    Currently, the threshold is fixed at :math:`g_0 = 0.003`.

    .. math::
        g(t) = \frac{1}{2} \left[
                  \text{erf}\left(\frac{t - (c - w / 2)}{\sqrt{2 \sigma^2}}\right)
                - \text{erf}\left(\frac{t - (c + w / 2)}{\sqrt{2 \sigma^2}}\right)
               \right],

    where :math:`\text{erf}` is the error function, :math:`c` is :attr:`center_offset`, :math:`w` is the
    square width, and :math:`\sigma` is the gaussian standard deviation.

    We set

    * :math:`w :=` :attr:`full_width` - :attr:`rise_time`
    * :math:`\sigma :=` :attr:`rise_time` :math:`/ (\sqrt{8} \: \text{erf}^{-1}(1 - 2 g_0))`

    The cutoff time :math:`t_c = c - w / 2 -` :attr:`rise_time` :math:`/ 2`
    marks the start of the rising segment.
    The waveform after offsetting, truncating and rescaling is given by

    .. math::
        f(t) = \text{max}(g(t) - g_0, 0) / (1 - g_0).

    where :math:`g_0` is the threshold level for the truncation.
    We have :math:`f(t_c) \approx 0`, and the approximation is good if :math:`g_0 < 0.1`.

    The values of the waveform are in :math:`[0, 1]`.

    Args:
        full_width: Duration of the support of the pulse, from start of the rising to the end of the falling segment.
        rise_time: Duration of the rising and falling segments.
        center_offset: The waveform is centered at this offset from the midpoint of the sampling window.
    """

    full_width: float
    rise_time: float
    center_offset: float = 0.0

    def _sample(self, sample_coords: np.ndarray) -> np.ndarray:
        threshold: float = 0.003
        rt = np.abs(self.rise_time)
        # distance between rising and falling edge
        square_width = self.full_width - rt
        # if pulse does not reach 1, we could define sq = fw / 2, otherwise we get a jump
        if 0 < self.full_width <= 2 * rt:
            square_width = self.full_width / (2 * (2 * rt / self.full_width) ** 2)
        # We calculate the sigma, which yields the requested threshold value at the beginning of the rise time.
        gaussian_sigma = rt / (scipy.special.erfinv(1 - 2 * threshold) * np.sqrt(8))

        rising_edge = self.center_offset - square_width / 2
        falling_edge = rising_edge + square_width

        if square_width > 0:
            waveform = 0.5 * (
                scipy.special.erf((sample_coords - rising_edge) / (np.sqrt(2) * gaussian_sigma))
                - scipy.special.erf((sample_coords - falling_edge) / (np.sqrt(2) * gaussian_sigma))
            )
        else:
            waveform = 0.0 * sample_coords

        return _subtract_threshold_and_rescale(waveform, threshold)


@register_canonical_waveform
@dataclass(frozen=True)
class CosineRiseFall(CanonicalWaveform):
    r"""Waveform that has a sinusoidal rise and fall, and a constant part in between.

    .. math::
        f(t) =
        \begin{cases}
          \frac{1}{2}(1 + \cos(\pi (t - c + p / 2) / r)) & t - (c - p / 2) \in [-r, 0]\\
          1 & t - c \in [-p / 2, p / 2]\\
          \frac{1}{2}(1 + \cos(\pi (t - c - p / 2) / r)) & t - (c + p / 2) \in [0, r]\\
          0 & \text{otherwise}
        \end{cases}

    where :math:`c` is :attr:`center_offset`, :math:`r` is :attr:`rise_time`, and :math:`p` is the plateau width,
    calculated via :math:`p :=` :attr:`full_width` - 2 * :attr:`rise_time`.

    Its values are in :math:`[0, 1]`.


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
                "Since the full width is smaller than twice the rise time, the CosineRiseFall pulse does not reach its"
                "top.",
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
                    lambda oc: 0.5 + 0.5 * np.cos(np.pi / rt * (oc + pw / 2)),
                    lambda oc: 0.5 + 0.5 * np.cos(np.pi / rt * (oc - pw / 2)),
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
                    lambda oc: 0.5 + 0.5 * np.cos(np.pi / rt * (oc + pw / 2)),
                    1,
                    lambda oc: 0.5 + 0.5 * np.cos(np.pi / rt * (oc - pw / 2)),
                    0,
                ],
            )
        return waveform


def is_canonical(waveform: type[Waveform]) -> bool:
    """Returns ``True`` if ``waveform`` is a canonical waveform else ``False``"""
    return type(waveform) in _CANONICAL_WAVEFORMS


def to_canonical(waveform: Waveform) -> CanonicalWaveform:
    """Convert the waveform into a canonical version of itself, e.g. for serialization.

    Canonical waveforms are returned as is, non-canonical waveforms are sampled.

    Returns:
        canonical version of the waveform
    """
    if is_canonical(waveform):
        return waveform
    logging.getLogger(__name__).debug("%s is not canonical, will sample it.", waveform)
    return Samples(waveform.sample())
