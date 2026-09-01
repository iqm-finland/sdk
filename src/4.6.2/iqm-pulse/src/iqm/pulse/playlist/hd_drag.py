#  ********************************************************************************
#    _____  ____ _
#   / _ \ \/ / _` |  Framework for control
#  |  __/>  < (_| |  and measurement of
#   \___/_/\_\__,_|  superconducting qubits
#
#  Copyright (c) 2019-2025 IQM Finland Oy.
#  All rights reserved. Confidential and proprietary.
#
#  Distribution or reproduction of any information contained herein
#  is prohibited without IQM Finland Oy’s prior written permission.
#  ********************************************************************************
"""Waveform definitions for a higher-derivative (HD) DRAG pulse based on Appendix B of :cite:`Hyyppa_2024`."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from iqm.pulse.playlist.fast_drag import SuppressedPulse

COSINE_COEFFICIENTS_DICT = {
    0: np.array([1]),
    1: np.array([1.33333333, -0.33333333]),
    2: np.array([1.5, -0.6, 0.1]),
    3: np.array([1.6, -0.8, 0.22857143, -0.02857143]),
    4: np.array([1.66666667, -0.95238095, 0.35714286, -0.07936508, 0.00793651]),
    5: np.array([1.71428571, -1.07142857, 0.47619048, -0.14285714, 0.02597403, -0.0021645]),
    6: np.array(
        [
            1.75000000e00,
            -1.16666667e00,
            5.83333333e-01,
            -2.12121212e-01,
            5.30303030e-02,
            -8.15850816e-03,
            5.82750583e-04,
        ]
    ),
    7: np.array(
        [
            1.77777778e00,
            -1.24444444e00,
            6.78787879e-01,
            -2.82828283e-01,
            8.70240870e-02,
            -1.86480186e-02,
            2.48640249e-03,
            -1.55400155e-04,
        ]
    ),
}
"""
Pre-computed coefficients of the cosine terms in the basis envelope (0th derivative). This dictionary contains a 
mapping from the number of suppressed frequencies to the coefficients of the cosine terms computed using the
function ``solve_cosine_coefs_for_hd_drag``. 
"""


def solve_cosine_coefs_for_hd_drag(
    number_of_suppressed_freqs: int,
) -> np.ndarray:
    r"""Solve cosine coefficients of the basis envelope given the number of suppressed frequencies.

    The cosine coefficients :math:`\{d_k\}_{k=1}^{K+1}` define the basis envelope as
    :math:`g(t) = \sum_{k=1}^{K+1} d_k (1 - \cos(2\pi k t/t_p + k\pi))`, where the pulse is defined on the
    interval :math:`t \in [-t_p/2, t_p/2]`.

    Args:
        number_of_suppressed_freqs: Number of suppressed frequencies

    Returns:
        Coefficient array of length ``number_of_suppressed_freqs + 1``

    """
    # The coefficients x_k can be solved from a matrix equation A*x = b. Let's build A
    if number_of_suppressed_freqs == 0:
        return np.array([1.0])
    n_coefs = number_of_suppressed_freqs + 1
    a_mat = np.zeros((n_coefs, n_coefs))
    for idx_row in range(n_coefs):
        for idx_col in range(n_coefs):
            if idx_row < n_coefs - 1:
                a_mat[idx_row, idx_col] = (-1) ** idx_row * (idx_col + 1) ** (2 * (idx_row + 1))
            else:
                a_mat[idx_row, idx_col] = 1

    b_arr = np.zeros((n_coefs,))
    b_arr[-1] = 1
    cosine_coefs_arr = np.linalg.solve(a_mat, b_arr)
    return cosine_coefs_arr


@lru_cache(maxsize=10000)
def solve_hd_drag_coefficients_from_suppressed_frequencies(
    pulse_duration: float,
    suppressed_freq_arr: tuple[float, ...],
) -> np.ndarray:
    r"""Solve coefficients of the derivative terms in a HD DRAG pulse given pulse duration and frequencies to suppress.

    The coefficients :math:`\{\beta_{2n}}_{n=0}^{K}` of the derivative terms are solved using Eq. (B5) of
    :cite:`Hyyppa_2024` assuming that :math:`\beta_0 = 1`. Here, :math:`K` is the number of suppressed frequencies.

    Note that the duration and frequencies must have matching units, e.g., s and Hz, or ns and GHz.

    Args:
        pulse_duration: Pulse duration (in s).
        suppressed_freq_arr: Frequencies to be suppressed relative to the center drive frequency (in Hz).

    Returns:
        Coefficient array of length ``len(suppressed_freq_arr) + 1``

    """
    # The beta coefficients can be solved from a matrix equation A*beta = b according to Eq. (B5). Let's build A
    suppressed_freqs = np.asarray(suppressed_freq_arr)
    n_coefs = len(suppressed_freqs) + 1
    a_mat = np.zeros((n_coefs, n_coefs))
    a_mat[0, 0] = 1
    a_mat[1:, :] = (-1) ** np.arange(n_coefs)[None] * (pulse_duration * suppressed_freqs[:, None]) ** (
        2 * np.arange(n_coefs)[None]
    )
    b_arr = np.zeros((n_coefs,))
    b_arr[0] = 1
    derivative_coefs_arr = np.linalg.solve(a_mat, b_arr)
    return derivative_coefs_arr


def evaluate_nth_derivative_of_basis_envelope(
    t_arr: np.ndarray, pulse_duration: float, cosine_coefs_arr: np.ndarray, n: int
) -> np.ndarray:
    r"""Evaluate nth derivative of the basis envelope for HD DRAG based on a cosine series.

    The basis envelope is given by :math:`g(t) = \sum_{k=1}^{K+1} d_k (1 - \cos(2\pi k t/t_p + k\pi))`, where the
    pulse is defined on the interval :math:`t \in [-t_p/2, t_p/2]`. The returned derivatives are normalized
    via multiplication by :math:`((t_p/(2\pi))^n`, where :math:`n` is the order of the derivative.


    Args:
        t_arr: Array of time points, at which the function is to be evaluated
        pulse_duration: Pulse duration in the same units as t_arr
        cosine_coefs_arr: Coefficients of the cosine terms in the basis envelope
        n: order of derivative

    Returns:
        Array containing the nth derivative of the basis envelope evaluated at ``t_arr``

    """
    non_zero_indices = np.logical_and(t_arr > -pulse_duration / 2, t_arr < pulse_duration / 2)
    t_arr_nonzero = t_arr[non_zero_indices]
    pulse_samples = np.zeros(t_arr.shape)
    n_cos_coefs = len(cosine_coefs_arr)
    if n == 0:
        non_zero_samples = np.sum(
            cosine_coefs_arr[:, None]
            * (
                1
                - (-1) ** np.arange(1, n_cos_coefs + 1)[:, None]
                * np.cos(2 * np.pi * np.arange(1, n_cos_coefs + 1)[:, None] * t_arr_nonzero[None] / pulse_duration)
            ),
            axis=0,
        )
    elif n % 2 == 0:
        non_zero_samples = np.sum(
            cosine_coefs_arr[:, None]
            * (-1) ** (n // 2 + np.arange(n_cos_coefs)[:, None])
            * np.arange(1, n_cos_coefs + 1)[:, None] ** n
            * np.cos(2 * np.pi * np.arange(1, n_cos_coefs + 1)[:, None] * t_arr_nonzero[None] / pulse_duration),
            axis=0,
        )
    else:
        non_zero_samples = np.sum(
            cosine_coefs_arr[:, None]
            * (-1) ** (n // 2 + np.arange(1, n_cos_coefs + 1)[:, None])
            * np.arange(1, n_cos_coefs + 1)[:, None] ** n
            * np.sin(2 * np.pi * np.arange(1, n_cos_coefs + 1)[:, None] * t_arr_nonzero[None] / pulse_duration),
            axis=0,
        )
    pulse_samples[non_zero_indices] = non_zero_samples
    return pulse_samples


def evaluate_hd_drag_i_envelope(
    t_arr: np.ndarray,
    pulse_duration: float,
    derivative_coefs_arr: np.ndarray,
    cosine_coefs_arr: np.ndarray,
) -> np.ndarray:
    r"""Evaluate I-envelope of HD DRAG given the coefficients of the derivative terms and the cosine terms.

    The I-envelope is defined as :math:`I(t) = \sum_{n=0}^{K} \beta_{2n} g^{(2n)}(t)`, where :math:`K` is the number
    of suppressed frequency ranges, :math:`\{beta_{2n}\}` are the coefficients of the derivative terms, and
    :math:`g(t)` is the basis envelope. The pulse is assumed to start at time ``-pulse_duration/2``,
    and end at time ``pulse_duration/2``.

    Args:
        t_arr: Array of time points, at which the function is to be evaluated
        pulse_duration: Pulse duration in the same units as t_arr
        derivative_coefs_arr: Coefficients of the even derivatives
        cosine_coefs_arr: Coefficients of the cosine terms in the basis envelope

    Returns:
        I-envelope of a HD DRAG pulse evaluated at ``t_arr``

    """
    pulse_samples_i = np.zeros(t_arr.shape)
    for idx, derivative_coef in enumerate(derivative_coefs_arr):
        nth_derivative = evaluate_nth_derivative_of_basis_envelope(t_arr, pulse_duration, cosine_coefs_arr, idx * 2)
        pulse_samples_i += derivative_coef * nth_derivative
    return pulse_samples_i


def evaluate_hd_drag_q_envelope(
    t_arr: np.ndarray,
    pulse_duration: float,
    derivative_coefs_arr: np.ndarray,
    cosine_coefs_arr: np.ndarray,
) -> np.ndarray:
    r"""Evaluate Q-envelope of HD DRAG given the coefficients of the derivative terms and the cosine terms.

    The Q-envelope is defined as :math:`Q(t) = \sum_{n=0}^{K} \beta_{2n} g^{(2n+1)}(t)`, where :math:`K` is the number
    of suppressed frequency ranges, :math:`\{beta_{2n}\}` are the coefficients of the derivative terms, and
    :math:`g(t)` is the basis envelope. The pulse is assumed to start at time ``-pulse_duration/2``,
    and end at time ``pulse_duration/2``.

    Args:
        t_arr: Array of time points, at which the function is to be evaluated
        pulse_duration: Pulse duration in the same units as t_arr
        derivative_coefs_arr: Coefficients of the derivatives
        cosine_coefs_arr: Coefficients of the cosine terms in the basis envelope

    Returns:
        Q-envelope of a HD DRAG pulse evaluated at ``t_arr``

    """
    pulse_samples_i = np.zeros(t_arr.shape)
    for idx, derivative_coef in enumerate(derivative_coefs_arr):
        nth_derivative = evaluate_nth_derivative_of_basis_envelope(t_arr, pulse_duration, cosine_coefs_arr, idx * 2 + 1)
        pulse_samples_i += derivative_coef * nth_derivative
    return pulse_samples_i


@dataclass(frozen=True)
class HdDrag(SuppressedPulse):
    r"""Base class for higher-derivative DRAG based on Eqs. (B1) and (B2) of :cite:`Hyyppa_2024`.

    Base class for IQ components of the higher derivative (HD) drag pulse. Depending on the value of
    ``compute_coefs_from_frequencies``, we compute the coefficients from the suppressed frequencies during the
    post-initialization or use pre-computed coefficients of the derivative terms (neglecting the suppressed
    frequencies). See :class:`SuppressedPulse`.

    """

    center_offset: float = 0

    def __post_init__(self) -> None:
        """Post initialization."""
        if self.compute_coefs_from_frequencies:
            coefficients = solve_hd_drag_coefficients_from_suppressed_frequencies(
                self.full_width,
                tuple(list(self.suppressed_frequencies)),
            )
            object.__setattr__(self, "coefficients", coefficients)

    @staticmethod
    def non_timelike_attributes() -> dict[str, str]:
        return {
            "coefficients": "",
            "suppressed_frequencies": "Hz",
            "compute_coefs_from_frequencies": "",
        }


@dataclass(frozen=True)
class HdDragI(HdDrag):
    r"""I-component of the higher derivative (HD) drag pulse.

    The I-component is defined according to Eq. (B1) of :cite:`Hyyppa_2024` ,

    .. math::
         I(t) = \sum_{n=0}^{K} \beta_{2n} g^{(2n)}(t),

    where :math:`\{\beta_{2n}\}_{n=0}^K` are the coefficients of the derivative terms, :math:`K` is the number of
    suppressed frequencies, and :math:`g(t)` is the basis envelope given by

    .. math::
        g(t) = \sum_{k=1}^K d_k (1 - \cos(2 \pi k t/t_p + k\pi)),

    where :math:`d_k` are pre-computed to ensure continuous derivatives up to order :math:`2K + 1`, :math:`t_p`
    denotes the pulse duration, and the pulse is defined across :math:`t \in (-t_p/2, t_p/2)`.

    The sampled pulse is always normalized to have a maximum value slightly below 1.0.
    """

    def _sample(self, sample_coords: np.ndarray) -> np.ndarray:
        n_suppressed_freqs = len(self.coefficients) - 1
        if (cosine_coefs := COSINE_COEFFICIENTS_DICT.get(n_suppressed_freqs)) is not None:
            samples = evaluate_hd_drag_i_envelope(
                sample_coords - self.center_offset, self.full_width, self.coefficients, cosine_coefs
            )
            normalized_samples = HdDrag._normalize(samples)
            return normalized_samples
        raise ValueError("Too many suppressed frequencies. At most 7 is allowed.")


@dataclass(frozen=True)
class HdDragQ(HdDrag):
    r"""Q-component of the higher derivative (HD) drag pulse.

    The Q-component is defined according to Eq. (B1) of :cite:`Hyyppa_2024` ,

    .. math::
         Q(t) = \sum_{n=0}^{K} \beta_{2n} g^{(2n + 1)}(t),

    where :math:`\{\beta_{2n}\}` are the coefficients, and :math:`g(t)` is the basis envelope given by

    .. math::
        g(t) = \sum_{k=1}^K d_k (1 - \cos(2 \pi k t/t_p + k\pi)),

    where :math:`d_k` are pre-computed to ensure continuous derivatives up to order :math:`2K + 1`, :math:`t_p`
    denotes the pulse duration, and the pulse is defined across :math:`t \in (-t_p/2, t_p/2)`.

    The sampled pulse is always normalized to have a maximum value of slightly below 1.0.
    """

    def _sample(self, sample_coords: np.ndarray) -> np.ndarray:
        n_suppressed_freqs = len(self.coefficients) - 1
        if (cosine_coefs := COSINE_COEFFICIENTS_DICT.get(n_suppressed_freqs)) is not None:
            samples = evaluate_hd_drag_q_envelope(
                sample_coords - self.center_offset, self.full_width, self.coefficients, cosine_coefs
            )
            normalized_samples = HdDrag._normalize(samples)
            return normalized_samples
        raise ValueError("Too many suppressed frequencies. At most 7 is allowed.")
