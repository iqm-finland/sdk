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
"""ESPRIT algorithm implementation for fitting oscillatory curves."""

import numpy as np
from scipy import linalg


def esprit(
    data: np.ndarray, window: int | None = None, model_size: int | None = None, eps: float | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""ESPRIT method.

    Estimate parameters assuming `data` is a sum of complex exponentials:
    :math:`\Sigma_n A_n \exp (2 \pi i f_n) + N` where :math:`N` is noise.

    :math:`n` is estimated:

    - by some AIC-based guess based on the RSS after denoising
    - if `eps` is given: by truncating based on singular values
      (throw away smallest SVD making up fraction `esp` of sum of all SVDs)
    - if model_size is given: n = model_size

    Args:
        data: A 1-d array of data points.
        window: the averaging window used for denoising. Should be at least a few n, usually the default
            ``(len(data) // 2)`` is a good choice, but can be reduced to speed up if data is large.
        model_size: If given, forces :math:`n` in the model.
        eps: Fraction of smallest SVDs to discard in model size estimation.

    Returns:
        * Array of estimated frequencies :math:`f_n`.
          For oscillating decaying components, two values occur as a complex pair
          :math:`(2 \pi i T - f, 2 \pi i T + f)`; for purely decaying components, a single imaginary value
          :math:`2 \pi i T` occurs,
          where the real part :math:`f` is the frequency and the imaginary part corresponds to the decay constant.
        * Array of corresponding amplitudes :math:`A_n`.
          The phase of the complex amplitudes correspond to the phase of the sine.
        * Model data corresponding to :math:`f_n` and :math:`A_n`.

    """
    if window is None:
        window = len(data) // 2
    hankel = _make_hankel_matrix(data, window)
    signal_components = _signal_subspace_extraction(hankel, model_size, eps)
    frequencies = _prony(signal_components)
    amplitudes, fit_data = _fit_amplitudes(data, frequencies)

    return frequencies, amplitudes, fit_data


def _make_hankel_matrix(x: np.ndarray, window: int) -> np.ndarray:
    return linalg.hankel(x[0:-window], x[-window - 1 :])


def _svd_reconstruction_errors(matrix: np.ndarray, svd: tuple[np.ndarray, ...]) -> np.ndarray:
    # Given a matrix and its svd (singular value decomposition), calculate the residual sum squared
    # when approximating `matrix` with a rank 0...nmax matrix.
    svd_u, svd_vals, svd_vh = svd
    nmax = len(svd_vals) // 2

    ret = np.empty(nmax)
    for rank in range(nmax):
        matrix_est = svd_u[:, :rank] @ np.diag(svd_vals[:rank]) @ svd_vh[:rank]
        ret[rank] = np.sum(np.abs(matrix - matrix_est) ** 2)

    return ret


def _signal_subspace_extraction(hankel_matrix: np.ndarray, model_size: int | None = None, eps=None, aic_fudge=16):  # noqa: ANN001, ANN202
    # Given the hankel matrix of a complex exponential time series with noise, extract the signal subspace components.

    svd_u, svd_vals, svd_vh = linalg.svd(hankel_matrix)

    if model_size is None:
        if eps is not None:
            # model selection based on fraction of sum of svd
            model_size = int(np.sum(svd_vals / svd_vals[0] > eps))
        else:
            # model selection based on AIC of denoised hankel matrix
            reconstruction_errors = _svd_reconstruction_errors(hankel_matrix, (svd_u, svd_vals, svd_vh))
            k = sum(hankel_matrix.shape)

            # AIC:
            # for normal least-squares, AIC = 2k + n logRSS
            # where k is the number of parameters and n is the number of datapoints.
            # Every component comes with 4 parameters (complex frequency and complex amplitude)
            # but if signal is real, not all are independent.
            # Somehow i think a fudgefactor of 2 works better
            # also RSS here is actually for the hankel matrix...
            # So not sure how valid this is but it seems to work fine.

            aics = aic_fudge * np.arange(len(reconstruction_errors)) + k * np.log(reconstruction_errors)
            model_size = np.argmin(aics)  # type:ignore[assignment]

    return svd_vh[:model_size].T


def _prony(components: np.ndarray) -> np.ndarray:
    r"""Prony's method.

    Given a KxT matrix containing K time series of length T which are linearly independent
    superpositions of complex exponentials

    :math:`C_n (t) = \Sum_m a_nm * \exp (2 \pi i f_m t)`

    extract the set of complex frequencies :math:`f_m`.
    """
    component_shift_estimate = linalg.lstsq(components[1:], components[:-1])[0]
    shift_eigenvalues = linalg.eigvals(component_shift_estimate)
    frequencies = -np.log(shift_eigenvalues) / 2j / np.pi
    return frequencies


def _vandermonde(freqs: np.ndarray, length: int) -> np.ndarray:
    # Build a Vandermonde matrix: for each frequency, one row of given length, with the complex exponential time
    # series of that frequency.
    shape = (len(freqs), length)
    return np.fromfunction(lambda k, t: np.exp(2j * np.pi * freqs[k.astype(int)] * t), shape)


def _fit_amplitudes(data: np.ndarray, frequencies: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""Fit amplitudes to a sum of complex exponentials.

    Given ``data`` and ``frequencies`` (f_n),
    find the best least-square estimate for amplitudes A_n assuming:

    :math:`\Sum_n A_n \exp (2 \pi i f_n) + N` where :math:`N` is noise.

    Returns:
        - Array of amplitudes :math:`A_n` corresponding to frequencies in `frequencies`.
        - Fitted data.

    """
    vm_matrix = _vandermonde(frequencies, len(data))
    amp_est = linalg.lstsq(vm_matrix.T, data)[0]
    data_est = np.dot(amp_est, vm_matrix)

    return amp_est, data_est
