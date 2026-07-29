# Copyright 2024-2025 IQM
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
import pytest

from iqm.cpc.core.dataset import compute_1d_fit


def test_curve_fit_1d_linear():
    def model(time, coeff_a, coeff_b):
        return time * coeff_a + coeff_b

    x = np.linspace(0, 100, 100)
    opt, fit = compute_1d_fit(x, model(x, 2.5, -1.11), model, dict(coeff_a=5, coeff_b=10))

    assert opt["coeff_a"] == pytest.approx(2.5)
    assert opt["coeff_b"] == pytest.approx(-1.11)
    assert abs(opt["coeff_a_stddev"]) < 10**-10
    assert abs(model(x, 2.5, -1.11) - fit < 1e-5).all()


def test_curve_fit_1d_sine():
    def model(time, freq, offset):
        return np.sin(time * freq + offset)

    x = np.linspace(-2, 2, 10)
    opt, fit = compute_1d_fit(x, model(x, 0.4, -0.1), model, dict(freq=0.1, offset=0))

    assert opt["freq"] == pytest.approx(0.4)
    assert opt["offset"] == pytest.approx(-0.1)
    assert abs(opt["freq_stddev"]) < 10**-10
    assert model(x, 0.4, -0.1) == pytest.approx(fit)


def test_curve_fit_1d_fails_with_missing_guess():
    def model(time, freq, offset):
        return np.sin(time * freq + offset)

    x = np.linspace(-2, 2, 10)

    with pytest.raises(ValueError) as err:
        compute_1d_fit(x, model(x, 0.4, -0.1), model, dict(freq=0.1))

    assert "needs arguments ['freq', 'offset']" in str(err.value)
    assert "{'offset'} were missing" in str(err.value)


def test_curve_fit_1d_ignores_extra_guesses():
    def model(time, freq, offset):
        return np.sin(time * freq + offset)

    x = np.linspace(-2, 2, 10)

    compute_1d_fit(x, model(x, 0.4, -0.1), model, dict(freq=0.1, offset=0, extra=3456))
