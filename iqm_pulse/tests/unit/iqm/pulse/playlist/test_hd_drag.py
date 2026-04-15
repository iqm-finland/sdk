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

from iqm.pulse.playlist.hd_drag import (
    HdDragI,
    HdDragQ,
    evaluate_hd_drag_i_envelope,
    evaluate_hd_drag_q_envelope,
    solve_cosine_coefs_for_hd_drag,
    solve_hd_drag_coefficients_from_suppressed_frequencies,
)


def test_cosine_coefficients_are_correct():
    number_of_suppressed_frequencies = 2
    cosine_coefficients = solve_cosine_coefs_for_hd_drag(number_of_suppressed_frequencies)
    assert_allclose(cosine_coefficients, [1.5, -0.6, 0.1], atol=1e-6)


def test_derivative_coefficients_are_correctly_solved_from_frequencies_with_one_frequency():
    pulse_duration = 10e-9
    suppressed_frequencies = (200e6,)
    coefficients = solve_hd_drag_coefficients_from_suppressed_frequencies(pulse_duration, suppressed_frequencies)
    assert len(coefficients) == 2
    assert_allclose(coefficients, [1.0, 0.25], atol=1e-6)


def test_derivative_coefficients_are_correctly_solved_from_frequencies_with_two_frequencies():
    pulse_duration = 20e-9
    suppressed_frequencies = (
        50e6,
        200e6,
    )
    coefficients = solve_hd_drag_coefficients_from_suppressed_frequencies(pulse_duration, suppressed_frequencies)
    assert len(coefficients) == 3
    assert_allclose(coefficients, [1.0, 1.0625, 0.0625], atol=1e-6)


def test_derivative_coefficients_are_correctly_solved_from_frequencies_with_three_frequencies():
    pulse_duration = 20e-9
    suppressed_frequencies = (
        50e6,
        150e6,
        200e6,
    )
    coefficients = solve_hd_drag_coefficients_from_suppressed_frequencies(pulse_duration, suppressed_frequencies)
    assert len(coefficients) == 4
    assert_allclose(coefficients, [1.0, 1.17361111, 0.18055556, 0.00694444], atol=1e-6)


def test_evaluate_hd_drag_i_envelope():
    pulse_duration = 20e-9
    t_arr = np.linspace(-pulse_duration / 2, pulse_duration / 2, 21)
    derivative_coefs_arr = np.array([1.0, -0.5, 0.5])
    cosine_coefs_arr = np.array([1.5, -0.6, 0.1])
    envelope = evaluate_hd_drag_i_envelope(t_arr, pulse_duration, derivative_coefs_arr, cosine_coefs_arr)
    assert envelope.shape == (21,)
    assert envelope[10] == pytest.approx(15.2, abs=1e-5)
    assert envelope[0] == pytest.approx(0.0, abs=1e-5)
    assert envelope[20] == pytest.approx(0.0, abs=1e-5)


def test_evaluate_hd_drag_q_envelope():
    pulse_duration = 20e-9
    t_arr = np.linspace(-pulse_duration / 2, pulse_duration / 2, 21)
    derivative_coefs_arr = np.array([1.0, -0.5, 0.5])
    cosine_coefs_arr = np.array([1.5, -0.6, 0.1])
    envelope = evaluate_hd_drag_q_envelope(t_arr, pulse_duration, derivative_coefs_arr, cosine_coefs_arr)
    assert envelope.shape == (21,)
    assert envelope[10] == pytest.approx(0.0, abs=1e-5)
    assert envelope[11] == pytest.approx(-19.85025083, abs=1e-5)
    assert envelope[0] == pytest.approx(0.0, abs=1e-5)
    assert envelope[20] == pytest.approx(0.0, abs=1e-5)


def test_hd_drag_i_samples_are_correct_using_frequencies():
    duration = 40e-9
    full_width = 20e-9
    suppressed_frequencies = np.array([45e6, 200e6]) * duration  # normalize with duration

    waveform = HdDragI(
        n_samples=21,
        full_width=full_width / duration,
        coefficients=np.array([1.0, 0.0]),
        compute_coefs_from_frequencies=True,
        suppressed_frequencies=suppressed_frequencies,
    )
    samples = waveform.sample()
    assert waveform.n_samples == 21
    assert_allclose(waveform.coefficients, [1.0, 1.2970679, 0.07716049], atol=1e-5)
    assert np.max(np.abs(samples)) == pytest.approx(1.0, abs=1e-5)
    assert samples[10] == pytest.approx(-0.5992786, abs=1e-3)


def test_hd_drag_i_samples_are_correct_using_coefficients():
    duration = 40e-9
    full_width = 20e-9
    suppressed_frequencies = np.array([45e6, 200e6]) * duration  # normalize with duration

    waveform = HdDragI(
        n_samples=21,
        full_width=full_width / duration,
        coefficients=np.array([1.0, 0.0]),
        compute_coefs_from_frequencies=False,
        suppressed_frequencies=suppressed_frequencies,
    )
    samples = waveform.sample()
    assert waveform.n_samples == 21
    assert_allclose(waveform.coefficients, [1.0, 0.0], atol=1e-5)
    assert np.max(np.abs(samples)) == pytest.approx(1.0, abs=1e-5)
    assert samples[10] == pytest.approx(1.0, abs=1e-3)


def test_hd_drag_q_samples_are_correct_using_frequencies():
    duration = 40e-9
    full_width = 20e-9
    suppressed_frequencies = np.array([45e6, 200e6]) * duration  # normalize with duration

    waveform = HdDragQ(
        n_samples=21,
        full_width=full_width / duration,
        coefficients=np.array([1.0, 0.0]),
        compute_coefs_from_frequencies=True,
        suppressed_frequencies=suppressed_frequencies,
    )
    samples = waveform.sample()
    assert waveform.n_samples == 21
    assert np.max(np.abs(samples)) == pytest.approx(1.0, abs=1e-5)
    assert samples[10] == pytest.approx(0.0, abs=1e-3)
