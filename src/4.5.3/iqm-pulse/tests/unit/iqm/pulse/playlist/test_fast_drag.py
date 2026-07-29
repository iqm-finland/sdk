#  ********************************************************************************
#
# Copyright 2025 IQM
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

import numpy as np
from numpy.testing import assert_allclose
import pytest

from iqm.pulse.playlist.fast_drag import (
    FastDragI,
    FastDragQ,
    compute_matrix_of_summed_fourier_transform_inner_products,
    evaluate_fast_drag_i_envelope,
    evaluate_fast_drag_q_envelope,
    fourier_transform_of_cos_basis_functions_as_tensor,
    solve_fast_coefficients_for_given_weights_and_ranges,
)


def test_fourier_transform_has_correct_shape_and_values():
    n_arr = np.linspace(1, 4, 4)
    frequency_arr = np.array([[0.03, 0.035, 0.04], [0.19, 0.20, 0.21]])
    pulse_duration = 20
    fourier_transform_tensor = fourier_transform_of_cos_basis_functions_as_tensor(n_arr, frequency_arr, pulse_duration)

    assert fourier_transform_tensor.shape == (4, 2, 3)
    assert np.real(fourier_transform_tensor[0, 0, 0]) == pytest.approx(-4.87234001, abs=1e-5)
    assert np.imag(fourier_transform_tensor[0, 0, 0]) == pytest.approx(-14.995520, abs=1e-5)
    assert np.real(fourier_transform_tensor[2, 1, 1]) == pytest.approx(0, abs=1e-5)
    assert np.imag(fourier_transform_tensor[2, 1, 1]) == pytest.approx(0, abs=1e-5)


def test_matrix_of_inner_products_has_correct_shape_and_values():
    n_arr = np.linspace(1, 4, 4)
    weights_arr = np.array([5, 1])
    suppressed_freq_ranges_2darr = np.array([[190e6, 210e6], [450e6, 2000e6]])
    pulse_duration = 8e-9
    time_scaling_factor = 1 / pulse_duration
    n_points_for_integration = 60

    matrix = compute_matrix_of_summed_fourier_transform_inner_products(
        n_arr,
        weights_arr,
        suppressed_freq_ranges_2darr,
        pulse_duration,
        time_scaling_factor,
        n_points_for_integration,
    )
    assert matrix.shape == (4, 4)
    assert np.real(matrix[0, 0]) == pytest.approx(0.0122816, abs=1e-5)
    assert np.imag(matrix[0, 0]) == pytest.approx(0.0, abs=1e-5)
    assert np.real(matrix[2, 3]) == pytest.approx(0.0424574, abs=1e-5)


def test_fast_coefficients_are_correct_with_one_interval():
    number_of_cosines = 3
    pulse_duration = 8e-9
    weights_tuple = (1,)
    suppressed_freq_ranges_2d_tuple = ((150e6, 1000e6),)
    n_points_for_integration = 60

    coefficients = solve_fast_coefficients_for_given_weights_and_ranges(
        number_of_cosines,
        pulse_duration,
        weights_tuple,
        suppressed_freq_ranges_2d_tuple,
        n_points_for_integration,
    )
    assert coefficients.shape == (3,)
    assert_allclose(coefficients, [2.36448717, 0.45246246, 0.32464303], atol=1e-6)


def test_fast_coefficients_are_correct_with_three_intervals():
    number_of_cosines = 4
    pulse_duration = 20e-9
    weights_tuple = (
        2,
        15,
        1,
    )
    suppressed_freq_ranges_2d_tuple = ((30e6, 40e6), (190e6, 210e6), (150e6, 1000e6))
    n_points_for_integration = 60

    coefficients = solve_fast_coefficients_for_given_weights_and_ranges(
        number_of_cosines,
        pulse_duration,
        weights_tuple,
        suppressed_freq_ranges_2d_tuple,
        n_points_for_integration,
    )
    assert coefficients.shape == (4,)
    assert_allclose(coefficients, [-4.12176148, 7.98633869, -0.73946905, 0.01648449], atol=1e-6)


def test_evaluate_fast_drag_i_envelope():
    t_arr = np.linspace(-6, 6, 13)
    pulse_duration = 10.0
    coefficients = np.array([1.0, -1.0, 0.5])
    i_envelope_arr = evaluate_fast_drag_i_envelope(t_arr, pulse_duration, coefficients)
    assert i_envelope_arr.shape == (13,)
    assert i_envelope_arr[0] == pytest.approx(0.0, abs=1e-5)
    assert i_envelope_arr[1] == pytest.approx(0.0, abs=1e-5)
    assert i_envelope_arr[12] == pytest.approx(0.0, abs=1e-5)
    assert i_envelope_arr[6] == pytest.approx(3.0, abs=1e-5)


def test_evaluate_fast_drag_q_envelope():
    t_arr = np.linspace(-6, 6, 13)
    pulse_duration = 10.0
    coefficients = np.array([1.0, -1.0, 0.5])
    q_envelope_arr = evaluate_fast_drag_q_envelope(t_arr, pulse_duration, coefficients)
    assert q_envelope_arr.shape == (13,)
    assert q_envelope_arr[1] == pytest.approx(0.0, abs=1e-5)
    assert q_envelope_arr[12] == pytest.approx(0.0, abs=1e-5)
    assert q_envelope_arr[6] == pytest.approx(0.0, abs=1e-5)
    assert q_envelope_arr[5] == pytest.approx(3.91648306, abs=1e-5)


def test_fast_drag_i_samples_are_correct_using_frequencies():
    duration = 40e-9
    full_width = 20e-9
    suppressed_freq_ranges = (
        np.array([[40e6, 50e6], [190e6, 210e6], [150e6, 1000e6]]) * duration
    )  # normalize with duration
    weights = np.array([2, 15, 1])
    waveform = FastDragI(
        n_samples=21,
        full_width=full_width / duration,
        coefficients=np.array([1.0, 0.0]),
        compute_coefs_from_frequencies=True,
        suppressed_frequencies=(suppressed_freq_ranges[:, 0] + suppressed_freq_ranges[:, 1]) / 2,
        number_of_cos_terms=4,
        suppressed_interval_widths=(suppressed_freq_ranges[:, 1] - suppressed_freq_ranges[:, 0]),
        weights=weights,
    )
    samples = waveform.sample()
    assert waveform.n_samples == 21
    # Check that coefficients have been computed from frequencies
    assert_allclose(waveform.coefficients, [-0.91468943, 4.48141421, -0.43547698, 0.01034485], atol=1e-5)
    assert np.max(np.abs(samples)) == pytest.approx(1.0, abs=1e-5)
    assert samples[10] == pytest.approx(-0.3800726, abs=1e-3)


def test_fast_drag_i_samples_are_correct_using_coefficients():
    duration = 40e-9
    full_width = 20e-9
    suppressed_freq_ranges = (
        np.array([[40e6, 50e6], [190e6, 210e6], [150e6, 1000e6]]) * duration
    )  # normalize with duration
    weights = np.array([2, 15, 1])
    waveform = FastDragI(
        n_samples=11,
        full_width=full_width / duration,
        coefficients=np.array([1.0, 0.0]),
        compute_coefs_from_frequencies=False,
        suppressed_frequencies=(suppressed_freq_ranges[:, 0] + suppressed_freq_ranges[:, 1]) / 2,
        number_of_cos_terms=4,
        suppressed_interval_widths=(suppressed_freq_ranges[:, 1] - suppressed_freq_ranges[:, 0]),
        weights=weights,
    )
    samples = waveform.sample()
    assert waveform.n_samples == 11
    assert_allclose(waveform.coefficients, [1.0, 0.0], atol=1e-5)  # coefficients are set directly to provided value
    assert np.max(np.abs(samples)) == pytest.approx(1.0, abs=1e-5)
    assert samples[5] == pytest.approx(1.0, abs=1e-3)


def test_fast_drag_q_samples_are_correct_using_frequencies():
    duration = 40e-9
    full_width = 20e-9
    suppressed_freq_ranges = (
        np.array([[40e6, 50e6], [190e6, 210e6], [150e6, 1000e6]]) * duration
    )  # normalize with duration
    weights = np.array([2, 15, 1])
    waveform = FastDragQ(
        n_samples=21,
        full_width=full_width / duration,
        coefficients=np.array([1.0, 0.0]),
        compute_coefs_from_frequencies=True,
        suppressed_frequencies=(suppressed_freq_ranges[:, 0] + suppressed_freq_ranges[:, 1]) / 2,
        number_of_cos_terms=4,
        suppressed_interval_widths=(suppressed_freq_ranges[:, 1] - suppressed_freq_ranges[:, 0]),
        weights=weights,
    )
    samples = waveform.sample()
    assert waveform.n_samples == 21
    assert np.max(np.abs(samples)) == pytest.approx(1.0, abs=1e-5)
    assert samples[10] == pytest.approx(0.0, abs=1e-3)
