#  ********************************************************************************
#    _____  ____ _
#   / _ \ \/ / _` |  Framework for control
#  |  __/>  < (_| |  and measurement of
#   \___/_/\_\__,_|  superconducting qubits
#
#  Copyright 2021-2026 IQM Finland Oy
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
"""Functions for fitting models to data."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import partial
from typing import Any

import numpy as np
from scipy.fftpack import fft, fftfreq, fftshift
from scipy.optimize import curve_fit

import exa.core.util.dataset as data_helper
from iqm.cpc.core.dataset import compute_1d_fit

from .esprit import esprit

Fitter = Callable[[np.ndarray | Iterable, np.ndarray | Iterable], tuple[dict, np.ndarray]]


def _line_model(x: np.ndarray, slope: float, offset: float) -> np.ndarray:
    """First-degree polynomial function.

    Args:
        x: variable
        slope: slope of the line
        offset: constant offset

    Returns:
        model evaluated for each point in ``x``

    """
    return slope * x + offset


def _gaussian_model(x: float, x_0: float, sigma: float, y_0: float = 0, A: float = 1, b: float = 0):  # noqa: ANN202
    r"""General model of a Gaussian curve with an optional linear slope.

    .. math::
        f(x;x_0,\sigma,A,y_0,b) = A * \exp(-\frac{(x-x_0)^2}{2\sigma^2}) + y_0 + b * x

    Args:
        x: variable
        x_0: offset in the ``x`` coordinate, center of the curve (expected value in case of a probability density)
        sigma: corresponds to the width of the curve (standard deviation in case of a probability density).
        y_0: offset in the ``y`` coordinate
        A: Height of the peak (free normalization factor).
            Is a probability density function if :math:`A=1/\sigma\sqrt{2\pi}` (and :math:`b=0`)
        b: optional linear slope factor

    Returns:
        model evaluated for each point in ``x``

    """
    return A * np.exp(-((x - x_0) ** 2) / (2 * sigma**2)) + y_0 + b * x


def get_gaussian_fitter(  # noqa: ANN201
    x0_estimate: float | None = None,
    sigma_estimate: float | None = None,
    y0_estimate: float | None = None,
    A_estimate: float | None = None,
    b_estimate: float | None = None,
    **fit_kwargs,
):
    """Generate fitting function for a Gaussian curve (with optional linear slope `b`).

    This will try to generate sensible guesses based on the data ``(x, y)``.

    Args:
        x0_estimate: optional estimate for the center of the Gaussian
        sigma_estimate: optional estimate for the standard deviation of the Gaussian
        y0_estimate: optional estimate for the y-axis offset
        A_estimate: optional estimate for the height of the curve
        b_estimate:
            optional estimate for the linear slope added to the Gaussian.
            If not given, the model will be used without a linear slope instead of generating a default estimate.
        fit_kwargs: Additional arguments for the curve fit function.

    Returns:
        function for fitting data to a gaussian model

    """

    def _gaussian_fitter(x, y):  # noqa: ANN001, ANN202
        estimates = {
            "x_0": x0_estimate or x[np.argmax(np.abs(y - np.median(y)))],
            "sigma": sigma_estimate or (x[-1] - x[0]) / 5,  # peak prominence
            "y_0": y0_estimate or np.median(y),
            "A": A_estimate or y[np.argmax(np.abs(y - np.median(y)))],
        }
        bounds = (
            [np.min(x), 0, np.min(y), np.min(y)],
            [np.max(x), np.abs(np.max(x)), np.max(y), np.max(y)],
        )
        # if no estimate for linear slope is given, use simplified model (normal gaussian) without slope

        model = lambda x, x_0, sigma, y_0, A: _gaussian_model(x, x_0, sigma, y_0, A, b=0)  # noqa: E731
        # if linear slope estimate is given, use full model
        if b_estimate:
            estimates["b"] = b_estimate
            bounds[0].append(-np.finfo(np.float32).max)
            bounds[1].append(np.finfo(np.float32).max)
            model = _gaussian_model

        for i, (k, val) in enumerate(estimates.items()):
            if not bounds[0][i] <= val <= bounds[1][i]:
                raise ValueError(
                    f"Cannot run fit with estimate {k}={val}"
                    f" which is not within the bounds {(bounds[0][i], bounds[1][i])}."
                )

        return compute_1d_fit(
            x,
            y,
            model=model,
            estimates=estimates,
            bounds=bounds,
            maxfev=5000,
            **fit_kwargs,
        )

    return _gaussian_fitter


def get_line_fitter(slope_estimate: float = 1, offset_estimate: float = 0, **curve_fit_kwargs) -> Fitter:
    """Returns a 1d line fitter.

    The data is fitted to the model ``y(x) = slope * x + offset``.

    Args:
        slope_estimate: initial guess for ``slope``
        offset_estimate: initial guess for ``offset``
        curve_fit_kwargs: passed to ``curve_fit``.

    Returns:
        function for fitting data to a linear model

    """

    def _line_fitter(x, y):  # noqa: ANN001, ANN202
        """Helper function for fitting a line to (x, y) array_like data."""
        estimates = {
            "slope": slope_estimate,
            "offset": offset_estimate,
        }
        return compute_1d_fit(x, y, _line_model, estimates, **curve_fit_kwargs)

    return _line_fitter


def sine_with_decay_model(
    time: np.ndarray,
    pop_amplitude: float,
    T2: float,
    frequency: float,
    phase: float,
    offset: float,
) -> np.ndarray:
    """Exponentially decaying sine.

    Args:
        time: Time axis to evaluate the function on. unit: s
        pop_amplitude: Maximum of the oscillation.
        T2: Decay constant of the oscillation. unit: s
        frequency: Oscillation frequency. unit: Hz
        phase: Phase of the oscillation. unit: rad
        offset: Constant offset of the oscillation.

    Returns:
        Function evaluated for each point in `time`.

    """
    return pop_amplitude * np.exp(-time / T2) * np.sin(time * frequency * 2 * np.pi + phase) + offset


def get_oscillation_fitter(
    period_estimate: float | None = None,
    amplitude_estimate: float | None = None,
    offset_estimate: float | None = None,
    phase_value: float | None = -np.pi / 2,
) -> Fitter:
    r"""Generates a fitter function for sinusoidal oscillation.

    The fitter will do a least-squares fit to the model
    :math:`y(x) = A * \sin(1 / p * 2\pi * x + \phi) + B`,
    where :math:`A>0` is amplitude, :math:`p>0` is period, :math:`B` is offset, :math:`\phi` is phase.
    The phase can be fixed or a free parameter.
    The ESPRIT method will be used to get initial guesses for the least-squares fit, see :func:`.esprit`.
    The arguments can be used to override the guesses from ESPRIT method.

    Args:
        period_estimate: Initial guess for :math:`p`. By default given by :func:`.esprit`.
        amplitude_estimate: Initial guess for :math:`A`. By default given by :func:`.esprit`.
        offset_estimate: Initial guess for :math:`B`, defaults to mean of the data.
        phase_value: Fixed value for :math:`\phi`. If None, the phase will be a free parameter in the model.

    Returns:
        Function for fitting data to an oscillation model.

    """

    def _free_model(  # noqa: ANN202
        drive_amp: np.ndarray,
        pop_amplitude: float,
        period: float,
        offset: float,
        phase: float,
    ):
        return pop_amplitude * np.sin(1 / period * (2 * np.pi) * drive_amp + phase) + offset

    def _fixed_model(drive_amp: np.ndarray, pop_amplitude: float, period: float, offset: float):  # noqa: ANN202
        return pop_amplitude * np.sin(1 / period * (2 * np.pi) * drive_amp + phase_value) + offset  # type:ignore[operator]

    def _oscillation_fitter(x: np.ndarray | Iterable, y: np.ndarray | Iterable) -> tuple[dict, np.ndarray]:
        offset = offset_estimate or np.mean(np.asarray(y))
        data = np.asarray(y) - offset
        frequencies, amp_est, _ = esprit(data, model_size=2)

        x_scale = (np.max(x) - np.min(x)) / len(x)  # type:ignore[arg-type]

        period = x_scale / (abs(frequencies[1].real) + 1e-20) if period_estimate is None else period_estimate
        amplitude = 2 * abs(amp_est[1]) if amplitude_estimate is None else amplitude_estimate
        # The `min` prevents propagating frequency=0 ==> period=inf, which crashes the curve fitter in compute_1d_fit.
        estimates = {"pop_amplitude": amplitude, "period": min(period, 1e18), "offset": offset}
        lower_bounds = [-np.inf, 0, -np.inf]
        if phase_value is None:
            estimates["phase"] = np.angle(1j * amp_est[1])
            lower_bounds.append(-np.inf)
            return data_helper.compute_1d_fit(x, y, _free_model, estimates, bounds=(lower_bounds, np.inf), maxfev=2000)
        return data_helper.compute_1d_fit(x, y, _fixed_model, estimates, bounds=(lower_bounds, np.inf), maxfev=2000)

    return _oscillation_fitter


def get_decaying_oscillation_fitter(
    amp_guess: float | None = None,
    t2_guess: float | None = None,
    freq_guess: float | None = None,
    phase_guess: float | None = None,
    offset_guess: float | None = None,
) -> Callable:
    r"""Generates a fitter function for exponentially decaying sinusoidal oscillation.

    The fitter will do a least-squares fit to the model
    :math:`y(x) = A \exp(-x / T) \cos(2\pi f x + \phi) + B`,
    where :math:`A>0` is amplitude, :math:`f>0` is frequency, :math:`T>0` is exponential decay constant,
    and :math:`B` si offset.
    The ESPRIT method will be used to get initial guesses for the least-squares fit, see :func:`.esprit`.
    The arguments can be used to override the guesses from ESPRIT method.

    Args:
        amp_guess: Initial guess for :math:`A`.
        t2_guess: Initial guess for :math:`T`. Typical use case is the T2 Ramsey time.
        freq_guess: Initial guess for :math:`f`.
        phase_guess: Initial guess for :math:`phi`.
        offset_guess: Initial guess for :math:`B`, defaults to mean of the data.

    Returns:
        The fitting function.

    """

    def t2_model(
        time: np.ndarray,
        pop_amplitude: float,
        T2: float,
        frequency: float,
        phase: float,
        offset: float,
    ) -> np.ndarray:
        """Exponentially decaying cosine."""
        return pop_amplitude * np.exp(-time / T2) * np.cos(time * frequency * 2 * np.pi + phase) + offset

    def exponential_oscillation_fitter(x, y):  # noqa: ANN001, ANN202
        offset = np.mean(y) if offset_guess is None else offset_guess
        data = np.asarray(y) - offset
        frequencies, amp_est, _ = esprit(data, model_size=2)

        x_scale = (np.max(x) - np.min(x)) / len(x)
        estimates = {
            "pop_amplitude": 2 * abs(amp_est[1]) if amp_guess is None else amp_guess,
            "T2": x_scale / (2 * np.pi * abs(frequencies[1].imag)) if t2_guess is None else t2_guess,
            "frequency": abs(frequencies[1].real) / x_scale if freq_guess is None else freq_guess,
            "phase": np.angle(amp_est[1]) if phase_guess is None else phase_guess,
            "offset": offset,
        }
        lower_bounds = [0, 0, 0, -np.inf, -np.inf]
        fit = compute_1d_fit(x, y, t2_model, estimates, bounds=(lower_bounds, np.inf), maxfev=5_000)
        return fit

    return exponential_oscillation_fitter


def get_exponential_fitter(
    decay_estimate: float | None = None,
    amplitude_estimate: float | None = None,
    offset_estimate: float | None = None,
    decay_symbol: str = "",
) -> Callable:
    r"""Generates 1d exponential fitter function.

    Typically used in T1 or T2 Echo etc.

    The data :math:`y(x)` is fitted to the model :math:`y(x) = A * \exp(-x / d) + B`,
    where :math:`x` = independent variable (typically time), :math:`A` = amplitude,
    :math:`d>0` = decay and :math:`B` = offset.

    Args:
        decay_estimate: initial guess for `decay`, if not provided, then
            calculated with :func:`.get_estimated_exponent`.
        amplitude_estimate: initial guess for `amplitude`, defaults to y(x_min)-y(x_max).
        offset_estimate: initial guess for `offset`, defaults to y(x_max).
        decay_symbol: Optionally replace the parameter word "decay" with this symbol. E.g. T1, T2.

    Returns:
        function for fitting data to a exponential model

    """

    def _model(time: np.ndarray, pop_amplitude: float, decay: float, offset: float):  # noqa: ANN202
        return pop_amplitude * np.exp(-time / decay) + offset

    def _exponential_fitter(x, y):  # noqa: ANN001, ANN202
        estimates = {
            "pop_amplitude": amplitude_estimate or y[np.argmin(x)] - y[np.argmax(x)],
            "decay": decay_estimate or get_estimated_exponent(x, y),
            "offset": offset_estimate or y[np.argmax(x)],
        }
        lower_bounds = [-np.inf, 0, -np.inf]
        fit_parameters, fit = data_helper.compute_1d_fit(x, y, _model, estimates, bounds=(lower_bounds, np.inf))

        if decay_symbol:
            fit_parameters[decay_symbol] = fit_parameters["decay"]
            fit_parameters[f"{decay_symbol}_stddev"] = fit_parameters["decay_stddev"]
            fit_parameters.pop("decay")
            fit_parameters.pop("decay_stddev")
        return fit_parameters, fit

    return _exponential_fitter


def get_estimated_frequencies(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Estimates characteristic frequency in a given data set.

    This method uses fast Fourier transform (FFT) to extract the dominant frequency terms.
    Note that this method assumes equally based x-data in order to give reliable estimate in frequency.

    Args:
        x: independent variable, typically time.
        y: dependent variable, typically voltage signal.

    Returns:
        frequency components sorted according to their Fourier weights in the data.

    """
    if len(x) < 2:
        raise ValueError("Cannot estimate frequency if the length of x-data is less than 2")
    # compute step increment in x-data
    delta = np.abs(x[1] - x[0])
    ft_populations = np.abs(fftshift(fft(y)))
    frequencies = fftshift(fftfreq(len(y), delta))
    # Filter out negative and zero-frequency entries
    ft_populations = ft_populations[np.where(frequencies >= 0)]
    ft_populations = ft_populations[1:]
    frequencies = frequencies[np.where(frequencies >= 0)]
    frequencies = frequencies[1:]
    sorted_frequencies = frequencies[np.argsort(-1 * ft_populations)]
    return sorted_frequencies


def get_estimated_exponent(x: np.ndarray, y: np.ndarray) -> float:
    """Estimates decay constant in typical exponential decay data.

    Typically used in T1 or T2 Echo etc.

    Estimates the characteristic decay time x_guess where the following condition is fullfilled:
        y(x_guess) = (y(x_min) - y(x_max)) * np.exp(-1.0) + y(x_max)

    Args:
        x: independent variable, typically time.
        y: dependent variable.

    Returns:
        the absolute value of exponent

    """
    offset_guess = y[np.argmax(x)]
    amplitude_guess = y[np.argmin(x)] - y[np.argmax(x)]
    return x[np.argmin(np.abs(y - (amplitude_guess * np.exp(-1.0) + offset_guess)))]


def _cosine_model(angle: np.ndarray, amplitude: float, phase: float, offset: float) -> np.ndarray:
    """Cosine function with offset, as function of angle.

    Args:
        angle: Time axis to evaluate the function on.
        amplitude: Maximum of the oscillation.
        phase: phase of oscillation
        offset: Constant offset of the oscillation.

    Returns:
        Output of the cosine function for each point in `angle`.

    """
    return amplitude * np.cos(angle + phase) + offset


def fit_cos(phase: Iterable, signal: Iterable) -> dict[str, Any]:
    """Fits phase sweep with a cosine function.

    Args:
        phase: 1D array of the phase
        signal: 1D array of signal

    Returns:
        Dictionary containing fit parameters

    """
    phase = np.array(phase)
    signal = np.array(signal)

    # only exact if `phase` is equally distributed, i.e. sum(exp(1j*phase)) = 0
    #  (and usually it is), but also decent guess if only somewhat distributed
    approx_fourier_component = np.exp(-1j * phase).dot(signal)

    guess_phase_offset = np.angle(approx_fourier_component)
    guess_amp = np.max(signal) - np.min(signal)
    guess_offset = np.mean(signal)

    guess = np.array([guess_amp, guess_phase_offset, guess_offset])
    ((amplitude, phase_offset, signal_offset), pcov) = curve_fit(
        _cosine_model,
        xdata=phase,
        ydata=signal,
        p0=guess,
        bounds=([0, -np.pi, -np.inf], [np.inf, 2 * np.pi, np.inf]),
    )
    fitfunc = partial(_cosine_model, amplitude=amplitude, phase=phase_offset, offset=signal_offset)
    return {
        "amplitude": amplitude,
        "phase_offset": phase_offset,
        "signal_offset": signal_offset,
        "fitfunc": fitfunc,
        "maxcov": np.max(pcov),
        "rawres": (guess, (amplitude, phase_offset, signal_offset), pcov),
    }


def get_lorentzian_dip(
    amplitude_guess: float | None = None,
    x_offset_guess: float | None = None,
    sigma_guess: float | None = None,
    offset_guess: float | None = None,
) -> Fitter:
    """Generates 1d lorentzian fitter function that fits a lorenzian curve on the provided (x, y) values.

    Args:
        amplitude_guess: Guess for the amplitude in the Lorentzian model. The default value: the difference
            `max(y)-min(y)`
        x_offset_guess: Guess for the x-axis offset in the Lorentzian model. The default value: the x-value with which
            the function assumes its minimum.
        sigma_guess: Guess for the sigma in the Lorentzian model. The default value: the half point of the x-values
        offset_guess: Guess for the signal offset (y-axis) in the Lorentzian model. Tne default value: the minimum of
            the y-values

    Returns:
        function for fitting data to a Lorentzian model

    """

    def _lorentzian_model(x, amplitude: float, x_offset: float, sigma: float, offset: float):  # noqa: ANN001, ANN202
        return amplitude * np.exp(-((x - x_offset) ** 2) / (2 * sigma**2)) + offset

    def _lorentzian_dip_fitter(x, y):  # noqa: ANN001, ANN202
        guess = {
            "amplitude": amplitude_guess or (np.max(y) - np.min(y)),
            "x_offset": x_offset_guess or x[np.argmin(y)],
            "sigma": sigma_guess or (x[-1] - x[0]) / 5,
            "offset": offset_guess or np.max(y),
        }
        return data_helper.compute_1d_fit(x, y, _lorentzian_model, guess)

    return _lorentzian_dip_fitter


def get_square_pulse_response(
    square_pulse_length: float,
    decay_guess: float | None = None,
    amplitude_guess: float | None = None,
    offset_guess: float | None = None,
) -> Callable[[np.ndarray, np.ndarray], tuple[dict, np.ndarray]]:
    r"""Generates 1d fitter function for exponentially decaying response to a rectangular pulse.

    The data is fitted to the model :math:`y(x) = A * (\exp(-x / d) -\exp(-\frac{L + x}{d}) + B`,
    where :math:`A` = amplitude, :math:`d` = decay, :math:`L` is the length of the pulse, and :math:`B` = offset.

    Args:
        square_pulse_length: Legnth of the used pulse.
        decay_guess: Initial guess for `decay`, if not provided, then
            calculated with :func:`.get_estimated_exponent`. If
            there is no valid result from the function, then defaults
            to 1/3 of the time axis.
        amplitude_guess: Initial guess for `amplitude`, defaults to maximum of the data.
        offset_guess: Initial guess for `offset`, defaults to minimum of the data.

    Returns:
        Function for fitting data to a exponential model.

    """

    def _model(time: np.ndarray, amplitude: float, decay: float, offset: float):  # noqa: ANN202
        return amplitude * (np.exp(-time / decay) - np.exp(-(square_pulse_length + time) / decay)) + offset

    def _square_pulse_response_fitter(x, y) -> tuple[dict, np.ndarray]:  # noqa: ANN001
        estimates = {
            "amplitude": amplitude_guess or (y[0] - y[-1]),
            "decay": decay_guess or get_estimated_exponent(x, y) or np.max(x) / 3,
            "offset": offset_guess or y[-1],
        }
        return data_helper.compute_1d_fit(x, y, _model, estimates)

    return _square_pulse_response_fitter


def get_exponential_rb_fitter(
    clifford_fidelity_estimate: float, offset_estimate: float, amplitude_estimate: float
) -> Fitter:
    """Get a function for fitting an exponential RB model :math:`P_e = A p^{N_{Cl}} + B`
    for a randomized benchmarking (RB) experiment with the given initial parameters.
    Clifford fidelity estimate is given as: :math:`:math:`F_{Cl} = p + (1-p)/d`, where
    depolarization probability is `1-p`.

    Args:
        clifford_fidelity_estimate: guess for the clifford fidelity.
        offset_estimate: guess for the offset B
        amplitude_estimate: guess for amplitude A

    Returns:
        function for computing 1d fit of the exponential RB model
        with the given parameters

    """

    def _exponential_rb_fitter(x, y):  # noqa: ANN001, ANN202
        estimates = {
            "depolarization_probability": 2 - 2 * clifford_fidelity_estimate,
            "offset": offset_estimate,
            "amplitude": amplitude_estimate + offset_estimate,
        }
        return compute_1d_fit(x, y, _exponential_rb_model, estimates, maxfev=3000)

    return _exponential_rb_fitter


def get_exponential_interleaved_rb_fitter(
    clifford_fidelity_estimate: float, offset_estimate: float, amplitude_estimate: float
) -> Callable:
    """Get a function for fitting an exponential RB model :math:`P_e = A p^{N_{Cl}} + B`
    for a randomized benchmarking (RB) experiment with the given initial parameters.
    Clifford fidelity estimate is given as: :math:`:math:`F_{Cl} = p + (1-p)/d`, where
    depolarization probability is `1-p`.

    Args:
        clifford_fidelity_estimate: guess for the clifford fidelity.
        offset_estimate: guess for the offset B
        amplitude_estimate: guess for amplitude A

    Returns:
        function for computing 1d fit of the exponential RB model
        with the given parameters

    """

    def _exponential_interleaved_rb_model(
        number_of_cliffords: np.ndarray,
        y: np.ndarray,
        depolarization_probability: float,
        offset: float,
        amplitude: float,
    ) -> np.ndarray:
        return (amplitude - offset) * (1 - depolarization_probability) ** number_of_cliffords + offset

    def _exponential_interleaved_rb_fitter(x, y, z, cloned_arguments=None):  # noqa: ANN001, ANN202
        estimates = {
            "depolarization_probability": 2 - 2 * clifford_fidelity_estimate,
            "offset": offset_estimate,
            "amplitude": amplitude_estimate + offset_estimate,
        }
        return data_helper.compute_2d_fit(
            x, y, z, _exponential_interleaved_rb_model, estimates, cloned_arguments, only_index_y=True
        )

    return _exponential_interleaved_rb_fitter


def _exponential_rb_model(
    number_of_cliffords: np.ndarray,
    depolarization_probability: float,
    offset: float,
    amplitude: float,
) -> np.ndarray:
    """Exponential model :math:`P_e = A p^{N_{Cl}} + B` for a RB-type experiment,
    where :math:`p` = 1 - depolarization probability, :math:`A` = amplitude and :math:`B` = offset.

    Args:
        number_of_cliffords: array containing the lengths of the Clifford sequences.
        depolarization_probability: depolarization probability.
        offset: Constant offset of the exponential model
        amplitude: Amplitude multiplier of the exponential decay

    Returns:
        Function evaluated for each point in ``number_of_cliffords``.

    """
    return (amplitude - offset) * (1 - depolarization_probability) ** number_of_cliffords + offset


def get_parabola_fitter(
    a_guess: float | None = None,
    x_offset_guess: float | None = None,
    y_offset_guess: float | None = None,
) -> Callable:
    r"""Generates 1d parabola fitter function.

    The data is fitted to the model :math:`y(x) = A * (x - x_offset)^2 + y_offset`, where :math:`A` = amplitude,
    :math:`x_offset` = offset in the x-axis and :math:`y_offset` = offset in the y-axis.

    Args:
        a_guess: initial guess for the quadratic amplitude term `A`
        x_offset_guess: initial guess for the offset in the x-axis
        y_offset_guess: initial guess for the offset in the y-axis

    Returns:
        function for fitting data to an oscillation model

    """

    def _model(x: Iterable, a_parabola: float, x_offset: float, y_offset: float):  # noqa: ANN202
        return a_parabola * (np.array(x) - x_offset) ** 2 + y_offset

    def _parabola_fitter(x, y):  # noqa: ANN001, ANN202
        a_initial_guess = (y[0] - y[int(len(x) / 2)]) / (x[int(len(x) / 2)] - x[0]) ** 2 if a_guess is None else a_guess
        if y_offset_guess is None:
            y_offset_initial_guess = np.min(np.sign(a_initial_guess) * np.array(y))
        else:
            y_offset_initial_guess = y_offset_guess
        if x_offset_guess is None:
            x_offset_initial_guess = x[np.argmin(np.sign(a_initial_guess) * np.array(y))]
        else:
            x_offset_initial_guess = x_offset_guess

        estimates = {
            "a_parabola": a_initial_guess,
            "x_offset": x_offset_initial_guess,
            "y_offset": y_offset_initial_guess,
        }

        return data_helper.compute_1d_fit(x, y, _model, estimates)

    return _parabola_fitter


def get_leakage_rb_f_state_fitter(lambda1_estimate: float, offset_estimate: float, amplitude_estimate: float) -> Fitter:
    r"""Generates 1d leakage function fitter.

    Get a function for fitting the f-state probability :math:`P_f = A + B \lambda_1^{N_{Cl}}`
    in a leakage RB experiment with the given initial parameters. Here, :math:`A` = offset and :math:`B` = amplitude.

    Args:
        lambda1_estimate: Guess for the parameter :math:`\lambda_1`.
        offset_estimate: Guess for the offset A.
        amplitude_estimate: Guess for amplitude B.

    Returns:
        Function for computing 1d fit of the exponential RB model with the given parameters.

    """

    def _leakage_rb_f_state_model(
        number_of_cliffords: np.ndarray, lambda1: float, offset: float, amplitude: float
    ) -> np.ndarray:
        return amplitude * lambda1**number_of_cliffords + offset

    def _leakage_rb_f_state_fitter(x, y):  # noqa: ANN001, ANN202
        estimates = {"lambda1": lambda1_estimate, "offset": offset_estimate, "amplitude": amplitude_estimate}
        return compute_1d_fit(x, y, _leakage_rb_f_state_model, estimates)

    return _leakage_rb_f_state_fitter
