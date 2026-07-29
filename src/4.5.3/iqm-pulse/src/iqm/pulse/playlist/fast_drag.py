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
"""Waveform definitions for Fourier Ansatz Spectrum Tuning (FAST) DRAG pulse based on :cite:`Hyyppa_2024`."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.integrate import simpson

from iqm.pulse.playlist.waveforms import Waveform


def fourier_transform_of_cos_basis_functions_as_tensor(
    n_arr: np.ndarray, frequencies: np.ndarray, pulse_duration: float
) -> np.ndarray:
    r"""Evaluate Fourier transforms of cosine basis functions for given basis function indices and frequencies.

    The nth cosine basis function is given by :math:`g_n(t) = 1 -  \cos(2 \pi n t/t_p)` defined on the interval
    :math:`[0, t_p]`, where :math:`t_p` is the pulse duration. The Fourier transform can be analytically evaluated, see
    Eq. (A7) of :cite:`Hyyppa_2024`. We evaluate the Fourier transform for the basis function indices given by ``n_arr``
    and for the frequencies given by ``frequency_arr``. We store the Fourier transforms as a tensor of dimension
    ``1+dim(frequency_arr)``, such that the first dimension corresponds to the basis function indices, and the following
    dimensions to those of ``frequency_arr``. Thus, the Fourier transform is evaluated essentially for the cartesian
    product of ``n_arr`` and ``frequency_arr``.


    Args:
        n_arr: 1d array of basis function indices, running typically from 1 to N, where N is the
            number of considered basis functions
        frequencies: N-dimensional array of frequencies (in Hz), at which the Fourier transform is evaluated.
            For the computation of FAST DRAG coefficients, ``N=2``.
        pulse_duration: Pulse duration (in s), without zero padding.

    Returns:
        Array containing the Fourier transform as a tensor with a shape ``(len(n_arr), *frequency_arr.shape)``

    """
    # Note np.sinc = sin(pi*x)/(pi*x)
    # Shape product of frequencies and pulse duration to have same number of dimensions as output tensor.
    # Frequencies will be copied across different basis function indices
    f_arr_tp = pulse_duration * frequencies[None]
    # Shape n_arr to have same number of dimensions as output tensor. Basis function indices will be copied
    # across frequencies
    n_arr_shaped = n_arr[:, *[None] * len(frequencies.shape)]
    fourier_transform_tensor = pulse_duration * (
        np.exp(-1j * np.pi * f_arr_tp) * np.sinc(f_arr_tp)
        - 0.5 * np.exp(1j * np.pi * (n_arr_shaped - f_arr_tp)) * np.sinc(n_arr_shaped - f_arr_tp)
        - 0.5 * np.exp(-1j * np.pi * (n_arr_shaped + f_arr_tp)) * np.sinc(n_arr_shaped + f_arr_tp)
    )
    return fourier_transform_tensor


def compute_matrix_of_summed_fourier_transform_inner_products(
    n_arr: np.ndarray,
    weights: np.ndarray,
    suppressed_freq_ranges_2darr: np.ndarray,
    pulse_duration: float,
    time_scaling_factor: float,
    n_points_for_integration: int = 60,
) -> np.ndarray:
    r"""Evaluate matrix :math:`A` defined in Eq. (5) of :cite:`Hyyppa_2024`.

    The matrix element :math:`A_{nm}` is defined as

    .. math:: A_{nm} = \sum_{j=1}^k w_j \int_{f_{l,j}}^{f_{h,j}} \hat{g}_n(f) \hat{g}_m(f)^* \mathrm{d} f,

    where n,m are the row and column indices starting from 1 and extending to :math:`N` denoting the
    number of basis functions, :math:`f_{l,j}` and :math:`f_{h,j}` denote the starting and ending frequency of the
    j'th frequency range to suppress, :math:`\hat{g}_n(f)` denotes the Fourier transform of
    :math:`g_n(f) = 1 -  \cos(2 \pi n t/t_p)` with :math:`t_p` being the pulse duration.

    Importantly, the computations are fully vectorized for fast evaluation.

    Args:
        n_arr: 1d array containing indices of basis functions used in the series expansion,
            typically ranging from 1 to N
        weights: 1d array of weights corresponding to the suppressed frequency ranges
        suppressed_freq_ranges_2darr: 2d array describing the frequency ranges (in Hz), over which we want to suppress
            the Fourier transform. The array should have two columns, the first giving the starting frequency
            for each range and the second giving the ending frequency for each range.
            Note that frequencies will be symmetrically suppressed around the center frequency.
        pulse_duration: Pulse duration (in s) without zero padding.
        time_scaling_factor: Scaling factor for time to avoid excessively large or small values in the matrix. Scaling
            factor of frequency is obtained as the inverse.
        n_points_for_integration: Number of points at which the integrand is evaluated for each interval. At least 50
            points are recommended for the numerical approximation error to be low.

    Returns:
        2d array representing the ``A`` matrix with dimensions ``(len(n_arr), len(n_arr))``

    """
    # Construct 2d array of frequencies at which the Fourier transforms are evaluated when computing the numerical
    # integrals. Here, scaled_freqs_for_all_ranges[j, i] is the ith point of numerical integration in the jth frequency
    # range.
    scaled_freqs_for_all_ranges = np.transpose(
        np.linspace(
            suppressed_freq_ranges_2darr[:, 0] / time_scaling_factor,
            suppressed_freq_ranges_2darr[:, 1] / time_scaling_factor,
            n_points_for_integration,
        )
    )
    # Tensor comprising Fourier transforms \hat{g}_n for the considered basis functions. Here,
    # g_hat_nji_tensor[n, j, i] is the Fourier transform of the n_arr[n]'th basis function
    # evaluated in the i'th integration point of the j'th frequency interval
    g_hat_nji_tensor = fourier_transform_of_cos_basis_functions_as_tensor(
        n_arr, scaled_freqs_for_all_ranges, pulse_duration * time_scaling_factor
    )
    # Construct tensor comprising the products \hat{g}_n*conj(\hat{g}_m). Here, g_hat_product_nmji_tensor[n, m, j, i]
    # denotes the product of basis functions n and m evaluated at the i'th integration point of the j'th frequency
    # interval
    g_hat_product_nmji_tensor = g_hat_nji_tensor[:, None] * np.conjugate(g_hat_nji_tensor[None])
    # Frequency step for the different suppressed ranges. Needed for scaling the numerically evaluated integral
    frequency_steps = scaled_freqs_for_all_ranges[:, 1] - scaled_freqs_for_all_ranges[:, 0]
    # integral_nmj_tensor[n, m, j]: integral of \hat{g}_n*conj(\hat{g}_m) across the j'th frequency interval
    integral_nmj_tensor = simpson(g_hat_product_nmji_tensor, dx=1, axis=-1)
    # a_matrix[n, m]: matrix element corresponding to the basis function pair (n, m)
    a_matrix = np.sum(integral_nmj_tensor * weights * frequency_steps, axis=-1)
    return a_matrix


@lru_cache(maxsize=10000)
def solve_fast_coefficients_for_given_weights_and_ranges(
    number_of_cosines: int,
    pulse_duration: float,
    weights_tuple: tuple[float, ...],
    suppressed_freq_ranges_2d_tuple: tuple[tuple[float, ...], ...],
    n_points_for_integration: int = 60,
) -> np.ndarray:
    r"""Solve for optimal coefficients of the basis functions in a FAST DRAG pulse.

    Computes the optimal coefficients :math:`\{c_n\}_{n=1}^N` for a FAST DRAG pulse with :math:`N` basis functions
    such that the pulse spectrum is suppressed across the given frequency ranges according to the provided weights.
    The computation is based on  Eqs. (A5), (A7), (A12) and (A13) of :cite:`Hyyppa_2024`.

    Note that this function essentially computes the mapping from frequency-domain parameters of a FAST DRAG pulse
    to time-domain parameters.

    The results must be cached to allow efficient generation of playlists containing a large number of pulses with
    same parameters, as in RB.

    Args:
        number_of_cosines: Number of cosine basis functions used in the FAST DRAG pulse
        pulse_duration: Pulse duration (in s) without zero padding.
        weights_tuple: 1d tuple of weights for each suppressed frequency interval used in the objective function
        suppressed_freq_ranges_2d_tuple: 2d tuple describing the frequency ranges (in Hz), over which we want to
            suppress the Fourier transform. Each element of the outer tuple should be a tuple of two floats, the first
            setting the starting frequency for the given range and the second setting the ending frequency for the
            given range.
        n_points_for_integration: Number of points at which the integrand is evaluated for each interval. At least 50
            points are recommended for the numerical approximation error to be low.

    Returns:
        Coefficients of the basis functions as a 1d array.

    """
    # Convert tuple args to np arrays
    weights_arr = np.asarray(weights_tuple)
    suppressed_freq_ranges_2d_arr = np.array([list(inner_tuple) for inner_tuple in suppressed_freq_ranges_2d_tuple])
    # Target rotation angle used in the constraint of the FAST optimization problem. This affects the integral of the
    # pulse, but in typical experimental scenario, the pulse is further scaled so we can set anything here.
    rotation_angle = np.pi
    # Times and frequencies are scaled to values on the order of unity for better numerical stability.
    time_scaling_factor = 1 / pulse_duration
    # Construct matrix A_tilde = [A + A^T, -b; b^T, 0] and vector b = [0; rotation_angle/tg] for solving the
    # optimal coefficients
    n_arr = np.linspace(1, number_of_cosines, number_of_cosines)
    a_tilde = np.zeros((number_of_cosines + 1, number_of_cosines + 1), dtype=complex)
    b_vect = np.zeros((number_of_cosines + 1,), dtype=complex)
    ones_vect = np.ones((number_of_cosines,))

    a_mat = compute_matrix_of_summed_fourier_transform_inner_products(
        n_arr,
        weights_arr,
        suppressed_freq_ranges_2d_arr,
        pulse_duration,
        time_scaling_factor,
        n_points_for_integration=n_points_for_integration,
    )
    # Construct A_tilde of Eq. (A13)
    a_tilde[0:number_of_cosines, 0:number_of_cosines] = a_mat + np.transpose(a_mat)
    a_tilde[0:number_of_cosines, number_of_cosines] = -ones_vect
    a_tilde[number_of_cosines, 0:number_of_cosines] = ones_vect
    # Construct b vector
    b_vect[number_of_cosines] = rotation_angle / (pulse_duration * time_scaling_factor)
    c_opt_full = np.linalg.solve(a_tilde, b_vect)
    # Note that c_opt contains also the Lagrangian multiplier lambda as the last element, which we ignore.
    c_opt = c_opt_full[0:number_of_cosines]

    return np.real(c_opt)


def evaluate_fast_drag_i_envelope(
    t_arr: np.ndarray,
    pulse_duration: float,
    coefficients: np.ndarray,
) -> np.ndarray:
    r"""Evaluate I-envelope of a FAST DRAG pulse for given coefficients.

    The I-envelope is defined as :math:`I(t) = \sum_{n=1}^{N} c_n [1 - (-1)^n \cos(2\pi n t/t_p)]`, where :math:`N` is
    the number of cosine terms in the series, :math:`\{c_n\}` are the coefficients, and the pulse is defined on the
    interval :math:`t \in [-t_p/2, t_p/2]`.

    Args:
        t_arr: Array of time points, at which the function is to be evaluated
        pulse_duration: Pulse duration in the same units as t_arr
        coefficients: Coefficients of a FAST DRAG pulse

    Returns:
        I-envelope of a FAST DRAG pulse evaluated at ``t_arr``

    """
    non_zero_indices = np.logical_and(t_arr > -pulse_duration / 2, t_arr < pulse_duration / 2)
    pulse_samples = np.zeros(t_arr.shape)

    cosine_terms = np.arange(1, len(coefficients) + 1)
    pm = 2 * (cosine_terms % 2) - 1
    pulse_samples[non_zero_indices] = np.sum(
        coefficients[:, None]
        * (1 + pm[:, None] * np.cos(2 * np.pi * cosine_terms[:, None] * t_arr[non_zero_indices] / pulse_duration)),
        axis=0,
    )
    return pulse_samples


def evaluate_fast_drag_q_envelope(
    t_arr: np.ndarray,
    pulse_duration: float,
    coefficients: np.ndarray,
) -> np.ndarray:
    r"""Evaluate Q-envelope of FAST DRAG for given coefficients.

    The Q-envelope is defined as :math:`Q(t) = \sum_{n=1}^{N} c_n n (-1)^n \sin(2\pi n t/t_p)]`, where :math:`N` is
    the number of cosine terms in the series, :math:`\{c_n\}` are the coefficients, and the pulse is defined on the
    interval :math:`t \in [-t_p/2, t_p/2]`.

    Args:
        t_arr: Array of time points, at which the function is to be evaluated
        pulse_duration: Pulse duration in the same units as t_arr
        coefficients: Coefficients of a FAST DRAG pulse

    Returns:
        Q-envelope of a FAST DRAG pulse evaluated at ``t_arr``

    """
    non_zero_indices = np.logical_and(t_arr > -pulse_duration / 2, t_arr < pulse_duration / 2)
    pulse_samples = np.zeros(t_arr.shape)
    sine_terms = np.arange(1, len(coefficients) + 1)
    pm = 1 - 2 * (sine_terms % 2)
    pulse_samples[non_zero_indices] = np.sum(
        coefficients[:, None]
        * sine_terms[:, None]
        * pm[:, None]
        * np.sin(2 * np.pi * sine_terms[:, None] * t_arr[non_zero_indices] / pulse_duration),
        axis=0,
    )

    return pulse_samples


@dataclass(frozen=True)
class SuppressedPulse(Waveform):
    r"""Base class for a control pulse using a series expansion to suppress certain frequencies in its envelope spectrum

    The base class describes control pulses, in which the coefficients of the basis functions
    are chosen to suppress specific frequencies or frequency ranges in the frequency spectrum of the pulse envelope.
    Examples include FAST DRAG and HD DRAG.

    The pulse argument ``compute_coefs_from_frequencies`` allows the user to choose whether the basis function
    coefficients are computed from ``suppressed_frequencies`` during the post-initialization of the pulse
    (thus, overriding any pre-computed values in ``coefficients``), or if the (pre-computed) values in ``coefficients``
    are directly used and ``suppressed_frequencies`` are neglected. The classes deriving from this base class should
    implement the post-initialization logic specific to the given pulse.

    Args:
        full_width: Full width of the pulse corresponding to the pulse duration with non-zero amplitude (in s).
        coefficients: Pre-computed coefficients of the series expansion.
        suppressed_frequencies: Frequencies to be suppressed or center frequencies of intervals to be suppressed
            (in Hz).
        compute_coefs_from_frequencies: Boolean value indicating whether we compute the coefficients from the
            suppressed frequencies or whether we use the pre-computed coefficients. If True, ``suppressed_frequencies``
            are used to compute and override ``coefficients`` in the post-initialization. If False, pre-computed
            ``coefficients`` are used, and ``suppressed_frequencies`` are ignored.

    """

    full_width: float
    coefficients: np.ndarray
    suppressed_frequencies: np.ndarray
    compute_coefs_from_frequencies: bool

    @staticmethod
    def _normalize(samples: np.ndarray) -> np.ndarray:
        """Scale the pulse samples to the interval (-1, 1) to avoid clipping at the instruments.

        Args:
            samples: Array of pulse samples

        Returns:
            re-scaled Array of pulse samples

        """
        return np.clip(samples / np.max(np.abs(samples)), -1, 1)


@dataclass(frozen=True)
class FastDrag(SuppressedPulse):
    r"""Base class for IQ components of the Fourier Ansatz Spectrum Tuning (FAST) DRAG pulse.

    The FAST DRAG pulse shapes the I-envelope in the frequency domain to suppress specified frequency intervals
    according to given weights. Furthermore, the Q-envelope is obtained as a derivative of the I-envelope
    similarly to ordinary DRAG. This class represents a base class for an implementation using cosine functions
    as the I-envelope basis functions. See :cite:`Hyyppa_2024` for more details on FAST DRAG.

    Args:
        number_of_cos_terms: Number of cosine terms in the Fourier series expression of the I-component
        suppressed_interval_widths: Widths of the suppressed frequency intervals (in Hz). The last element corresponds
            to the width of a potential cutoff interval.
        weights: Weights corresponding to the suppressed frequency intervals

    """

    number_of_cos_terms: int
    suppressed_interval_widths: np.ndarray
    weights: np.ndarray
    center_offset: float = 0

    def __post_init__(self) -> None:
        """Post initialization."""
        if self.compute_coefs_from_frequencies:
            suppressed_frequency_ranges_2d_arr = np.zeros((len(self.suppressed_frequencies), 2))
            suppressed_frequency_ranges_2d_arr[:, 0] = self.suppressed_frequencies - self.suppressed_interval_widths / 2
            suppressed_frequency_ranges_2d_arr[:, 1] = self.suppressed_frequencies + self.suppressed_interval_widths / 2
            coefficients = solve_fast_coefficients_for_given_weights_and_ranges(
                self.number_of_cos_terms,
                self.full_width,
                tuple(list(self.weights)),
                tuple([tuple(freq_pair) for freq_pair in suppressed_frequency_ranges_2d_arr]),
            )
            object.__setattr__(self, "coefficients", coefficients)

    @staticmethod
    def non_timelike_attributes() -> dict[str, str]:
        return {
            "compute_coefs_from_frequencies": "",
            "coefficients": "",
            "suppressed_frequencies": "Hz",
            "number_of_cos_terms": "",
            "suppressed_interval_widths": "Hz",
            "weights": "",
        }


@dataclass(frozen=True)
class FastDragI(FastDrag):
    r"""I-component of the Fourier Ansatz Spectrum Tuning (FAST) drag pulse.

    The I-envelope is defined as

    .. math:: I(t) = \sum_{n=1}^{N} c_n [1 - \cos(2\pi n t/t_p + n\pi)],

    where :math:`N` is the number of cosine terms in the series, :math:`\{c_n\}` are the coefficients, and
    the pulse is defined on the interval :math:`t \in [-t_p/2, t_p/2]`.
    """

    def _sample(self, sample_coords: np.ndarray) -> np.ndarray:
        pulse_samples = evaluate_fast_drag_i_envelope(
            sample_coords - self.center_offset,
            self.full_width,
            self.coefficients,
        )
        normalized_samples = FastDrag._normalize(pulse_samples)
        return normalized_samples


@dataclass(frozen=True)
class FastDragQ(FastDrag):
    r"""Q-component of the Fourier Ansatz Spectrum Tuning (FAST) drag pulse.

    The Q-envelope is defined as

    .. math:: Q(t) = \sum_{n=1}^{N} c_n n \sin(2\pi n t/t_p + n\pi)],

    where :math:`N` is the number of cosine terms in the series, :math:`\{c_n\}` are the coefficients, and
    the pulse is defined on the interval :math:`t \in [-t_p/2, t_p/2]`.
    """

    def _sample(self, sample_coords: np.ndarray) -> np.ndarray:
        pulse_samples = evaluate_fast_drag_q_envelope(
            sample_coords - self.center_offset,
            self.full_width,
            self.coefficients,
        )
        normalized_samples = FastDrag._normalize(pulse_samples)
        return normalized_samples
