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

"""Tests for the ZNE numerics: fits evaluated at zero, their weights, and the curve they draw."""

from iqm.error_reduction_tools.zne.zne_processors import (
    extrapolate_at_zero,
    fit_curve,
    polynomial_extrapolation_weights,
)
import numpy as np
import pytest
from scipy import optimize

ATOL = 1e-10

POLYNOMIALS = [
    # Linear: y = 2x + 3
    ([1.0, 2.0, 3.0], [5.0, 7.0, 9.0], 1, [2.0, 3.0]),
    # Quadratic: y = x^2 + 2x + 1
    ([1.0, 2.0, 3.0], [4.0, 9.0, 16.0], 2, [1.0, 2.0, 1.0]),
    # Cubic: y = x^3 + 2x^2 + 3x + 4
    ([1.0, 2.0, 3.0, 4.0], [10.0, 26.0, 58.0, 112.0], 3, [1.0, 2.0, 3.0, 4.0]),
]
"""Exactly representable polynomials, as (x, y, degree, coefficients highest degree first)."""


class TestExtrapolateAtZero:
    @pytest.mark.parametrize("x_data,y_data,degree,coefficients", POLYNOMIALS)
    def test_a_polynomial_is_recovered_by_a_fit_of_its_own_degree(self, x_data, y_data, degree, coefficients):
        """Test that a fit of the same degree recovers every coefficient of a polynomial."""
        result, found = extrapolate_at_zero(x_data=x_data, y_data=y_data, extrapolation_type="Poly", degree=degree)

        assert isinstance(result, float)
        assert isinstance(found, list)
        assert len(found) == degree + 1
        assert all(isinstance(value, float) for value in found)
        assert np.allclose(found, coefficients, atol=ATOL)
        # Evaluating the polynomial at zero leaves its constant term, so the two must agree exactly.
        assert result == pytest.approx(coefficients[-1], abs=ATOL)

    def test_richardson_takes_the_degree_the_data_supports(self):
        """Test that without a degree the fit uses len(y_data) - 1 and recovers a cubic exactly."""
        x_data = [1.0, 2.0, 3.0, 4.0]
        y_data = [x**3 + 2 * x**2 + 3 * x + 4 for x in x_data]
        result, found = extrapolate_at_zero(x_data=x_data, y_data=y_data, extrapolation_type="Poly", degree=None)

        assert len(found) == len(y_data)
        assert np.allclose(found, [1.0, 2.0, 3.0, 4.0], atol=ATOL)
        assert np.isclose(result, 4.0, atol=ATOL)

    @pytest.mark.parametrize(
        "amplitude,rate,x_data",
        [
            (5.0, 0.5, [1.0, 2.0, 3.0, 4.0]),
            (0.9, -0.35, [1.0, 3.0, 5.0, 7.0]),
            (-2.0, 0.3, [1.0, 2.0, 3.0]),
        ],
    )
    def test_an_exponential_is_recovered_by_the_exponential_model(self, amplitude, rate, x_data):
        """Test that exponential data is fitted exactly by the exponential model."""
        y_data = [amplitude * np.exp(rate * x) for x in x_data]
        result, found = extrapolate_at_zero(x_data=x_data, y_data=y_data, extrapolation_type="Exp")

        assert len(found) == 2
        assert all(isinstance(value, float) for value in found)
        assert np.allclose(found, [amplitude, rate], atol=ATOL)
        # exp(0) is 1, so the value at zero is the amplitude parameter itself.
        assert result == pytest.approx(found[0], abs=ATOL)
        assert np.isclose(result, amplitude, atol=ATOL)

    def test_a_zero_value_does_not_break_the_exponential_model(self):
        """Test that a zero value gives a finite exponential extrapolation."""
        value, params = extrapolate_at_zero(
            x_data=[1.0, 3.0, 5.0, 7.0], y_data=[0.4, 0.2, 0.0, -0.2], extrapolation_type="Exp"
        )
        assert np.isfinite(value)
        assert np.all(np.isfinite(params))

    def test_an_exponential_fit_that_does_not_converge_is_reported_as_such(self, monkeypatch):
        """Test that an exponential fit that does not converge raises a ValueError."""

        def never_converges(*_args, **_kwargs):
            raise RuntimeError("Optimal parameters not found")

        monkeypatch.setattr(optimize, "curve_fit", never_converges)
        with pytest.raises(ValueError, match=r"did not converge on \[0.9, 0.5\] at \[1.0, 2.0\]"):
            extrapolate_at_zero(x_data=[1.0, 2.0], y_data=[0.9, 0.5], extrapolation_type="Exp")

    def test_a_length_mismatch_is_refused(self):
        """Test that x and y data of different lengths are rejected."""
        with pytest.raises(ValueError, match="Got 2 y values for 3 x values"):
            extrapolate_at_zero(x_data=[1.0, 2.0, 3.0], y_data=[5.0, 7.0])

    def test_an_unsupported_extrapolation_type_is_refused(self):
        """Test that an unknown extrapolation type is rejected."""
        with pytest.raises(ValueError, match="Unknown extrapolation_type 'Invalid'"):
            extrapolate_at_zero(x_data=[1.0, 2.0, 3.0], y_data=[5.0, 7.0, 9.0], extrapolation_type="Invalid")

    def test_a_degree_the_data_cannot_support_is_refused(self):
        """Test that a degree higher than the number of points allows is rejected."""
        with pytest.raises(ValueError, match="degree of the polynomial is too high"):
            extrapolate_at_zero(x_data=[1.0, 2.0, 3.0], y_data=[5.0, 7.0, 9.0], extrapolation_type="Poly", degree=5)


class TestPolynomialExtrapolationWeights:
    def test_richardson_weights_are_known(self):
        """Test the Richardson weights of the noise factors 1, 2 and 3."""
        weights = polynomial_extrapolation_weights([1.0, 2.0, 3.0])
        assert len(weights) == 3
        assert all(isinstance(weight, float) for weight in weights)
        assert np.allclose(weights, [3.0, -3.0, 1.0], atol=1e-8)

    def test_weights_sum_to_one(self):
        # A constant series must extrapolate to that constant, so the weights sum to 1.
        """Test that the extrapolation weights sum to 1."""
        for x_data in ([1.0, 1.5, 3.0], [1.0, 2.0, 3.0, 4.0]):
            assert np.isclose(sum(polynomial_extrapolation_weights(x_data)), 1.0, atol=1e-8)

    @pytest.mark.parametrize("x_data,y_data,degree,coefficients", POLYNOMIALS)
    def test_the_weights_extrapolate_to_the_same_constant_term(self, x_data, y_data, degree, coefficients):
        """Test that the extrapolation weights give the same value at zero as the fit."""
        weights = polynomial_extrapolation_weights(x_data, degree=degree)
        fitted, _found = extrapolate_at_zero(x_data=x_data, y_data=y_data, degree=degree)

        assert len(weights) == len(x_data)
        assert np.isclose(np.dot(weights, y_data), coefficients[-1], atol=1e-8)
        assert np.isclose(np.dot(weights, y_data), fitted, atol=1e-8)

    def test_the_weights_reproduce_a_fit_of_lower_degree_than_the_data(self):
        """Test that the weights match a least-squares fit of lower degree than the number of points."""
        x_data = [1.0, 1.5, 3.0]
        y_data = [0.6, 0.5, 0.3]
        weights = polynomial_extrapolation_weights(x_data, degree=1)
        expected, _found = extrapolate_at_zero(x_data=x_data, y_data=y_data, degree=1)
        assert np.isclose(np.dot(weights, y_data), expected, atol=1e-8)

    def test_a_degree_the_data_cannot_support_is_refused(self):
        """Test that a degree higher than the number of points allows is rejected."""
        with pytest.raises(ValueError, match="degree of the polynomial is too high"):
            polynomial_extrapolation_weights([1.0, 2.0], degree=5)


class TestFitCurve:
    """What the plots draw. It has to evaluate the same model the estimator reported."""

    def test_the_polynomial_model_takes_its_coefficients_highest_degree_first(self):
        """Test that the polynomial coefficients are read highest degree first."""
        x = np.array([0.0, 1.0, 2.0])
        curve = fit_curve([2.0, 3.0], x, "Poly")  # y = 2x + 3
        assert curve.shape == (3,)
        assert curve.dtype == np.float64
        assert np.allclose(curve, [3.0, 5.0, 7.0], atol=ATOL)

    def test_the_exponential_model_takes_amplitude_then_rate(self):
        """Test that the exponential parameters are read as amplitude, then rate."""
        x = np.array([0.0, 1.0, 2.0])
        curve = fit_curve([5.0, 0.5], x, "Exp")  # y = 5 * exp(0.5x)
        assert curve.shape == (3,)
        assert np.allclose(curve, [5.0, 5.0 * np.exp(0.5), 5.0 * np.exp(1.0)], atol=ATOL)

    @pytest.mark.parametrize(
        "extrapolation_type,x_data,y_data",
        [
            ("Poly", [1.0, 2.0, 3.0], [5.0, 7.0, 9.0]),
            ("Exp", [1.0, 2.0, 3.0], [5.0 * np.exp(0.5 * x) for x in (1.0, 2.0, 3.0)]),
        ],
    )
    def test_it_redraws_the_fit_the_extrapolation_reported(self, extrapolation_type, x_data, y_data):
        """Test that the fitted curve at zero equals the extrapolated value."""
        value, coefficients = extrapolate_at_zero(x_data=x_data, y_data=y_data, extrapolation_type=extrapolation_type)
        assert fit_curve(coefficients, np.array([0.0]), extrapolation_type)[0] == pytest.approx(value, abs=ATOL)

    def test_an_unknown_model_is_refused(self):
        """Test that an unknown extrapolation type is rejected."""
        with pytest.raises(ValueError, match="Unknown extrapolation_type 'Spline'"):
            fit_curve([1.0, 2.0], np.array([0.0]), "Spline")
