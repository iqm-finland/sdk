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
"""Lmfit model for double exponential decay (sum of two exponential)."""

from typing import Any

import lmfit
import numpy as np

from .model_utils import BaseModel


def double_exp_decay(
    x: np.ndarray,
    a1: float,
    rate1: float,
    a2: float,
    rate2: float,
) -> np.ndarray:
    """Double exponential decay : a1 * rate1^(-x) + a2 * rate2^(-x).

    Args:
        x: Independent variable.
        a1: First amplitude of the decay rate.
        rate1: First decay constant.
        a2: Second amplitude of the decay rate.
        rate2: Second decay constant.

    Returns:
        np.ndarray: Evaluated double exponential decay .

    """
    return a1 * np.exp(-x * np.log(rate1)) + a2 * np.exp(-x * np.log(rate2))


class DoubleExponentialDecay(BaseModel):
    r"""Lmfit model for double-exponential decay w.

    This model describes data that decays as the sum of two exponential components

    The model function is:

        .. math:: f(x) = a1 * rate1^(-x) + a2 * rate2^(-x)

    Parameter hints:
        - ``a1``: Initial guess 1.0 (no bounds).
        - ``rate1``: Initial guess 2, lower bound 0.01, upper bound 5.0.
        - ``a2``: Initial guess 0.1 (no bounds).
        - ``rate2``: Initial guess 1.05, lower bound 0.01, upper bound 2.0.
    """

    def __init__(self, independent_vars: list[str] | None = None, **kwargs: Any) -> None:
        """Initializes the DoubleExponentialDecay model.

        Args:
            args: Passed to parent __init__.
            independent_vars: Independent variable names.
            kwargs: Passed to parent __init__.

        """
        if independent_vars is None:
            independent_vars = ["x"]
        super().__init__(double_exp_decay, independent_vars=independent_vars, **kwargs)
        self.set_param_hint(name="a1", value=1.0)
        self.set_param_hint(name="rate1", value=2, min=0.01, max=5)
        self.set_param_hint(name="a2", value=0.1)
        self.set_param_hint(name="rate2", value=1.05, min=0.01, max=2.0)

    def guess(self, data: np.ndarray, x: np.ndarray, **kws) -> lmfit.parameter.Parameters:
        """Estimate initial parameters for double-exponential decay with offset.

        For non-trivial data, it estimates the amplitude as the difference between the first and
        last data points, and the offset as the last data point.

        If the amplitude is close to zero, it uses the range of the data as the amplitude and the
        minimum value as the offset.

        The decay constant (tau) is estimated by finding the point where the data falls
        to 1/e of its initial amplitude above the offsets.

        If the initial x is 0 the amplitude a1 is the first data point.
        otherwise we set it to 1.0

        If this estimation fails, a default value is used. The method ensures rate1 is positive
        and not too small

        We set a2 to a1/10 and rate2 to rate1*10

        Args:
            data: Dependent variable data.
            x: Independent variable data.
            kws: Values for the model parameters.

        Returns:
            Estimated parameters for fitting.

        """
        data = np.asarray(data).flatten()
        x = np.asarray(x).flatten()
        # Handle edge cases: empty or mismatched arrays
        if data.size == 0 or x.size == 0 or data.size != x.size:
            params = self.make_params(a1=0.0, rate1=1.0, a2=0.0, rate2=1.0)
            return lmfit.models.update_param_vals(params, self.prefix, **kws)

        # Handle constant data
        if np.allclose(data, data[0]):
            params = self.make_params(a1=0.0, rate1=1.0, a2=float(data[0]), rate2=1.0)
            return lmfit.models.update_param_vals(params, self.prefix, **kws)

        a1 = float(data[0]) if x[0] == 0 else 1.0
        # Avoid division by zero or negative amplitude
        if np.isclose(a1, 0.0):
            a1 = float(np.max(data) - np.min(data))
        # Estimate tau: find where data falls to (a1) / e
        rate_default = 10.0  # 10 x to reach error probability 1/e
        rate1 = rate_default
        try:
            target = a1 / np.e
            # Find first index where data crosses below target
            idx = np.where((data - target) < 0)[0]
            if len(idx) > 0 and x[-1] != x[0]:
                tau_est = x[idx[0]] - x[0]
                if tau_est > 0:
                    rate1 = float(np.exp(1.0 / tau_est))
        except Exception:
            rate1 = rate_default

        # The second exponential component is usually much smaller than the first.
        # search when the data diverge from the single exponential.
        a2 = a1 / 10
        rate2 = 1 + (1 - rate1) / 10
        params = self.make_params(a1=a1, rate1=rate1, a2=a2, rate2=rate2)
        return lmfit.models.update_param_vals(params, self.prefix, **kws)
