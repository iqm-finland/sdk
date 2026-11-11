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
"""Lmfit model for exponential decay."""

import lmfit
import numpy as np

from .model_utils import BaseModel


def exp_decay_with_offset(x: np.ndarray, A: float, tau: float, y_0: float) -> np.ndarray:
    """Exponential decay with offset.

    Args:
        x: Independent variable.
        A: Amplitude of the decay at t=0.
        tau: Decay constant.
        y_0: Offset.

    Returns:
        np.ndarray: Evaluated exponential decay with offset.

    """
    return A * np.exp(-x / tau) + y_0


def biexponential_decay_with_offset(
    x: np.ndarray, A_bi: float, alpha: float, tau_a: float, tau_b: float, y_0_bi: float
) -> np.ndarray:
    """Biexpoential decay with offset.

    Args:
    x: Independent variable.
    A_bi: Amplitude of the decay at t=0.
    alpha: Mixing fraction of the two decay channels.
    tau_a: First decay constant.
    tau_b: Second decay constant.
    y_0_bi: Offset.

    Returns:
        np.ndarray: Evaluated biexponential decay with offset.

    """
    return A_bi * (alpha * np.exp(-x / tau_a) + (1 - alpha) * np.exp(-x / tau_b)) + y_0_bi


class ExponentialDecayWithOffset(BaseModel):
    r"""Lmfit model for single-exponential decay with constant offset.

    This model describes data that decays exponentially from an initial value toward a constant baseline offset.

    The model function is:

        .. math:: f(x) = A \exp(-x / \\tau) + y_0

    Parameter hints:
        - ``A``: Initial guess 1.0 (no bounds).
        - ``\\tau``: Initial guess 30e-6 (30 μs), lower bound 1e-9.
        - ``y_0``: Initial guess 0.0 (no bounds).
    """

    def __init__(self, independent_vars: list[str] | None = None, **kwargs) -> None:
        """Initializes the ExponentialDecayWithOffset model.

        Args:
            args: Passed to parent __init__.
            independent_vars: Independent variable names.
            kwargs: Passed to parent __init__.

        """
        if independent_vars is None:
            independent_vars = ["x"]
        super().__init__(exp_decay_with_offset, independent_vars=independent_vars, **kwargs)
        self.set_param_hint(name="A", value=1.0)
        self.set_param_hint(name="tau", value=30e-6, min=1e-9, max=10e-3)
        self.set_param_hint(name="y_0", value=0.0)

    def guess(self, data: np.ndarray, x: np.ndarray, **kws) -> lmfit.parameter.Parameters:
        """Estimate initial parameters for exponential decay with offset.

        For non-trivial data, it estimates the amplitude as the difference between the first and last data points,
        and the offset as the last data point.

        If the amplitude is close to zero, it uses the range of the data as the amplitude and the
        minimum value as the offset.

        The decay constant (tau) is estimated by finding the point where the data falls
        to 1/e of its initial amplitude above the offsets.

        If this estimation fails, a default value is used. The method ensures tau is positive and not too small

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
            params = self.make_params(A=1.0, tau=0.0, y_0=0.0)
            return lmfit.models.update_param_vals(params, self.prefix, **kws)

        # Handle constant data
        if np.allclose(data, data[0]):
            params = self.make_params(A=0.0, tau=0.0, y_0=float(data[0]))
            return lmfit.models.update_param_vals(params, self.prefix, **kws)

        y_0 = float(data[-1])
        A = float(data[0] - y_0)

        # Avoid division by zero or negative amplitude
        if np.isclose(A, 0.0):
            A = float(np.max(data) - np.min(data))
            y_0 = float(np.min(data))

        # Estimate tau: find where data falls to (A + y_0) / e
        tau_default = 30e-6  # 30 us
        tau = tau_default
        try:
            target = y_0 + A / np.e
            # Find first index where data crosses below target
            idx = np.where((data - target) < 0)[0]
            if len(idx) > 0 and x[-1] != x[0]:
                tau_est = x[idx[0]] - x[0]
                if tau_est > 0:
                    tau = float(tau_est)
        except Exception:
            tau = tau_default

        # Ensure tau is positive and not too small
        tau = max(abs(tau), 1e-9)

        params = self.make_params(A=A, tau=tau, y_0=y_0)
        return lmfit.models.update_param_vals(params, self.prefix, **kws)


class BiexponentialDecayWithOffset(BaseModel):
    r"""Lmfit model for biexponential decay with constant offset.

    This model describes data that decays biexponentially from an initial value toward a constant baseline
    offset. This happens when, for example, there is a TLS changing frequency on a timescale shorter than
    a T1 experiment, or there is hybridisation in the presence of dephasing.

    The model function is:

        .. math:: f(x) = A_bi (\\alpha \exp(-x / \\tau_a) + (1-\\alpha) \exp(-x / \\tau_b)+ y_0_bi

    Parameter hints:
        - ``A_bi``: Initial guess 1.0 (no bounds).
        - ``\\alpha``: Initial guess 0.7, bounded between 0 and 1.
        - ``\\tau_a``: Initial guess 30e-6 (30 μs), lower bound 1e-9.
        - ``\\tau_b``: Initial guess 5e-6 (5 μs), lower bound 1e-9.
        - ``y_0_bi``: Initial guess 0.0 (no bounds).
    """

    def __init__(self, *args, independent_vars: list[str] | None = None, **kwargs) -> None:
        """Initializes the BixponentialDecayWithOffset model.

        Args:
            args: Passed to parent __init__.
            independent_vars: Independent variable names.
            kwargs: Passed to parent __init__.

        """
        if independent_vars is None:
            independent_vars = ["x"]
        super().__init__(biexponential_decay_with_offset, *args, independent_vars=independent_vars, **kwargs)
        self.set_param_hint(name="A_bi", value=1.0, min=0.0, max=2)
        self.set_param_hint(name="alpha", value=0.7, min=0.1, max=0.90)
        self.set_param_hint(name="tau_a", value=30e-6, min=1e-9, max=10e-3)
        self.set_param_hint(name="tau_b", value=30e-6, min=1e-9, max=10e-3)
        self.set_param_hint(name="y_0_bi", value=0.0, min=-0.5, max=2)

    def guess(self, data: np.ndarray, x: np.ndarray, **kws) -> lmfit.parameter.Parameters:
        """Estimate initial parameters for the decay.

        For non-trivial data, it estimates the amplitude as the difference between the first and last data points,
        and the offset as the last data point.

        If the amplitude is close to zero, it uses the range of the data as the amplitude and the
        minimum value as the offset.

        Alpha is taken as default.

        The 'major' decay constant (tau_a) is estimated by finding the point where the data falls
        to 1/e of its initial amplitude above the offsets.

        The 'minor' decay constant (tau_b) is taken to be tau_a/5.

        If this estimation fails, a default value is used. The method ensures tau is positive and not too small

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
            params = self.make_params(A_bi=1.0, tau_a=0.0, tau_b=0.0, alpha=0.1, y_0_bi=0.0)
            return lmfit.models.update_param_vals(params, self.prefix, **kws)

        # Handle constant data
        if np.allclose(data, data[0]):
            params = self.make_params(A_bi=0.0, tau_a=0.0, tau_b=0.0, alpha=0.1, y_0_bi=float(data[0]))
            return lmfit.models.update_param_vals(params, self.prefix, **kws)

        y_0_bi = float(data[-1])
        A_bi = float(data[0] - y_0_bi)

        # Avoid division by zero or negative amplitude
        if np.isclose(A_bi, 0.0):
            A_bi = float(np.max(data) - np.min(data))
            y_0_bi = float(np.min(data))

        # Estimate tau: find where data falls to (A + y_0) / e
        tau_a_default = 30e-6  # 30 us
        tau_a = tau_a_default
        try:
            target = y_0_bi + A_bi / np.e
            # Find first index where data crosses below target
            idx = np.where((data - target) < 0)[0]
            if len(idx) > 0 and x[-1] != x[0]:
                tau_a_est = x[idx[0]] - x[0]
                if tau_a_est > 0:
                    tau_a = float(tau_a_est)
        except Exception:
            tau_a = tau_a_default

        # Ensure tau is positive and not too small
        tau_a = max(abs(tau_a), 1e-9)

        # Set tau_b to tau_a/5 to get something smaller but not miniscule
        tau_b = tau_a / 5

        params = self.make_params(A_bi=A_bi, alpha=0.7, tau_a=tau_a, tau_b=tau_b, y_0_bi=y_0_bi)

        return lmfit.models.update_param_vals(params, self.prefix, **kws)
