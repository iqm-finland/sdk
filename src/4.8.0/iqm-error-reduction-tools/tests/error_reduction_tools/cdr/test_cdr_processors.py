# Copyright 2022-2026 IQM
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

"""
Tests for the CDR linear-fit processors.
"""

from iqm.error_reduction_tools.cdr.cdr_processors import (
    _fit_linear,
    cdr_fit_and_predict,
    r_squared,
)
import numpy as np
import pytest

ATOL = 1e-10


@pytest.mark.parametrize("a_true, b_true", [(2.0, 3.0), (-1.5, 0.5), (0.0, 4.0)])
def test_fit_linear_recovers_a_perfect_line(a_true, b_true):
    """Test that _fit_linear recovers the exact slope and intercept of a perfect line."""
    noisy = np.array([0.0, 1.0, 2.0, 3.0])
    exact = a_true * noisy + b_true
    a, b = _fit_linear(noisy, exact)
    assert a == pytest.approx(a_true, abs=ATOL)
    assert b == pytest.approx(b_true, abs=ATOL)


def test_fit_linear_falls_back_to_the_mean_when_noisy_is_constant():
    """Test that _fit_linear falls back to slope 0 and the mean of exact when noisy is constant."""
    noisy = np.array([0.5, 0.5, 0.5, 0.5])
    exact = np.array([1.0, 2.0, 3.0, 4.0])
    a, b = _fit_linear(noisy, exact)
    assert a == 0.0
    assert b == pytest.approx(np.mean(exact), abs=ATOL)


@pytest.mark.parametrize("a_true, b_true", [(2.0, 3.0), (-1.5, 0.5), (0.5, -2.0)])
def test_r_squared_is_one_for_a_perfect_line(a_true, b_true):
    """Test that r_squared reports R^2 = 1 for a perfect line."""
    noisy = np.array([0.0, 1.0, 2.0, 3.0])
    exact = a_true * noisy + b_true
    slope, intercept, r2 = r_squared(noisy, exact)
    assert slope == pytest.approx(a_true, abs=ATOL)
    assert intercept == pytest.approx(b_true, abs=ATOL)
    assert r2 == pytest.approx(1.0, abs=ATOL)


def test_fit_linear_falls_back_to_the_mean_with_a_single_point():
    """Test that _fit_linear falls back to slope 0 and the mean of exact with a single training point."""
    noisy = np.array([1.5])
    exact = np.array([7.0])
    a, b = _fit_linear(noisy, exact)
    assert a == 0.0
    assert b == pytest.approx(7.0, abs=ATOL)


def test_r_squared_reports_partial_fit_quality_for_a_known_scatter():
    """Test that r_squared reports the exact R^2 for data with known scatter around a line."""
    noisy = np.array([0.0, 1.0, 2.0, 3.0])
    exact = np.array([1.0, 1.0, 3.0, 3.0])
    slope, intercept, r2 = r_squared(noisy, exact)
    assert slope == pytest.approx(0.8, abs=ATOL)
    assert intercept == pytest.approx(0.8, abs=ATOL)
    assert r2 == pytest.approx(0.8, abs=ATOL)


def test_r_squared_is_nan_without_variance_in_exact():
    """Test that r_squared returns NaN when exact has no variance."""
    noisy = np.array([0.0, 1.0, 2.0, 3.0])
    exact = np.full_like(noisy, 5.0)
    _slope, _intercept, r2 = r_squared(noisy, exact)
    assert np.isnan(r2)


@pytest.mark.parametrize(
    "a_true, b_true, noisy_target, expected",
    [
        (2.0, 3.0, 5.0, 13.0),
        (-1.0, 4.0, 2.0, 2.0),
        (0.5, -1.0, 10.0, 4.0),
    ],
)
def test_a_single_observable_recovers_the_target(a_true, b_true, noisy_target, expected):
    """Test that cdr_fit_and_predict recovers the exact target value for a single observable."""
    noisy = np.array([0.0, 1.0, 2.0, 3.0])
    exact = a_true * noisy + b_true
    mitigated = cdr_fit_and_predict(exact_train=exact, noisy_train=noisy, noisy_target=noisy_target)
    assert mitigated == pytest.approx(expected, abs=ATOL)


def test_returning_the_coefficients():
    """Test that cdr_fit_and_predict returns the fitted coefficients when requested."""
    noisy = np.array([0.0, 1.0, 2.0])
    exact = 2.0 * noisy + 3.0
    mitigated, coeffs = cdr_fit_and_predict(exact, noisy, 5.0, return_coeffs=True)
    assert mitigated == pytest.approx(13.0, abs=ATOL)
    assert len(coeffs) == 1
    a, b = coeffs[0]
    assert a == pytest.approx(2.0, abs=ATOL)
    assert b == pytest.approx(3.0, abs=ATOL)


def test_per_observable_mode_fits_each_column_independently():
    """Test that per_observable mode fits each observable column independently."""
    noisy = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    exact = np.array([[3.0, -3.0], [5.0, -4.0], [7.0, -5.0]])
    mitigated = cdr_fit_and_predict(exact, noisy, np.array([4.0, 4.0]), mode="per_observable")
    assert mitigated.shape == (2,)
    assert mitigated.dtype == np.float64
    assert np.allclose(mitigated, [11.0, -7.0], atol=ATOL)


def test_pooled_mode_fits_a_single_line_across_all_observables():
    """Test that pooled mode fits one line across all observables' pooled data points."""
    noisy = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    exact = np.array([[1.0, 2.0], [3.0, 2.0], [5.0, 2.0]])
    mitigated, coeffs = cdr_fit_and_predict(exact, noisy, np.array([3.0, 5.0]), mode="pooled", return_coeffs=True)
    assert mitigated.shape == (2,)
    assert mitigated.dtype == np.float64
    assert np.allclose(mitigated, [4.5, 6.5], atol=ATOL)
    assert len(coeffs) == 1
    a, b = coeffs[0]
    assert a == pytest.approx(1.0, abs=ATOL)
    assert b == pytest.approx(1.5, abs=ATOL)


def test_a_shape_mismatch_between_noisy_and_exact_is_refused():
    """Test that cdr_fit_and_predict rejects mismatched noisy_train and exact_train shapes."""
    with pytest.raises(ValueError, match="must match"):
        cdr_fit_and_predict(np.array([1.0, 2.0, 3.0]), np.array([0.0, 1.0]), 0.0)


def test_a_wrong_length_target_is_refused():
    """Test that cdr_fit_and_predict rejects a noisy_target with the wrong number of entries."""
    noisy = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    exact = 2.0 * noisy + 3.0
    with pytest.raises(ValueError, match="there are 2 columns"):
        cdr_fit_and_predict(exact, noisy, np.array([1.0, 2.0, 3.0]), mode="per_observable")


def test_an_unknown_mode_is_refused():
    """Test that cdr_fit_and_predict rejects an unknown mode."""
    noisy = np.array([0.0, 1.0, 2.0])
    exact = 2.0 * noisy + 3.0
    with pytest.raises(ValueError, match="Unknown mode"):
        cdr_fit_and_predict(exact, noisy, 5.0, mode="bogus")
