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
"""Utility functions used in models."""

from collections.abc import Callable
from functools import wraps
import re
from typing import Any
import warnings

import lmfit
import numpy as np
import scipy.signal


def filter_param_guesses(
    make_params: Callable[..., lmfit.parameter.Parameters],
) -> Callable[..., lmfit.parameter.Parameters]:
    """Decorates ``make_params`` to drop guesses for parameters a hint holds fixed or ties with ``expr``.

    ``lmfit.Model.make_params`` writes its keyword arguments into the parameters it has already built from
    the hints, and ``lmfit.Parameter.set`` clears the expression of any parameter it is given a value for.
    A guess for a parameter a hint holds fixed therefore moves it off the pinned value while leaving it
    fixed, and a guess for a parameter a hint ties to another with ``expr`` cuts that tie altogether. What
    a fit never moves on its own is the caller's to set, so the guesses for both kinds are dropped here and
    the hints stand.

    Args:
        make_params: The model's ``make_params`` method, wrapped so every guess it is given passes through
            this filter first.

    Returns:
        A ``make_params`` that silently drops the guesses for parameters its hints already settle.

    """

    @wraps(make_params)
    def wrapper(self: lmfit.Model, **kwargs: Any) -> lmfit.parameter.Parameters:
        def is_free(name: str) -> bool:
            hint = self.param_hints.get(name.removeprefix(self.prefix), {})
            return hint.get("vary", True) and not hint.get("expr")

        return make_params(self, **{name: value for name, value in kwargs.items() if is_free(name)})

    return wrapper


def frequency_guess_with_fft(
    data: np.ndarray, coordinate_vec: np.ndarray
) -> tuple[float, float, float, float, float, float]:
    """Makes a guess for the parameters of an oscillating sine function with decay coefficients.

    The parameters for the oscillation are found from a zero-padded fft.

    Args:
        data: A sequence that can be converted to 1-d numpy array. Has to be at least two elements long.
        coordinate_vec: The coordinate vector corresponding to the data.

    Returns:
        (f_guess, A_guess, phi_guess, offset_guess, T_exp_guess, T_gauss_guess).

    """
    data = np.asarray(data)
    if len(data) < 2:
        raise ValueError(f"Input data {data} requires at least two elements.")
    if len(data.shape) != 1:
        raise ValueError(f"Input data {data} must be one-dimensional, but has {len(data.shape)} dimensions.")
    if len(coordinate_vec) < 2:
        raise ValueError(f"Coordinate vector {coordinate_vec} requires at least two elements.")

    zero_padding_factor = 20
    dim = len(data) * zero_padding_factor
    fft_length = dim // 2
    Y = np.fft.fft(data - np.mean(data), n=dim)[:fft_length]
    Y_abs = 2 * 2 / dim * np.abs(Y)  # Assume real signals
    coord_subtract = coordinate_vec[1] - coordinate_vec[0]
    f = np.fft.fftfreq(dim, coord_subtract)[:fft_length]

    # find the index of maximum peak in frequency

    max_ind = np.argmax(Y_abs[int(zero_padding_factor / 2) :]) + int(zero_padding_factor / 2)  # Exclude DC
    with warnings.catch_warnings():
        # Suppress peak not found warnings. They are handled below.
        warnings.simplefilter("ignore")
        peak_width_out = scipy.signal.peak_widths(Y_abs, [max_ind], rel_height=0.5)
    peak_width = peak_width_out[0][0] * (f[1] - f[0])

    if peak_width == 0.0:  # Peak could not be found. This usually indicates DC signal.
        f_index = min(fft_length - 1, zero_padding_factor)
        f_guess = f[f_index]
        T_exp_guess = (coord_subtract) * 2 / 3
        T_gauss_guess = (coord_subtract) * 2 / 3
        offset_guess = np.mean(data[-np.max([1, int(0.3 * len(data))]) :])
        A_guess = np.max(np.abs(data - offset_guess)) * np.sign(
            data[np.argmax(np.abs(data - offset_guess))] - offset_guess
        )
        phi_guess = (np.angle(Y[f_index]) + np.sign(A_guess) * np.pi / 2 - f_guess * coordinate_vec[0] * 2 * np.pi) % (
            2 * np.pi
        )
        if A_guess < 0:
            phi_guess = phi_guess + np.pi
            A_guess = -A_guess
    else:
        f_guess = f[max_ind]
        T_exp_guess = 1 / peak_width * 2
        T_gauss_guess = 1 / peak_width * 2
        offset_guess = np.real(Y[0]) + np.mean(data)
        peak_start = peak_width_out[2][0] - (peak_width_out[3][0] - peak_width_out[2][0]) * 0.3
        peak_end = peak_width_out[3][0] + (peak_width_out[3][0] - peak_width_out[2][0]) * 0.3
        A_guess = (
            np.sum(Y_abs[int(np.max([0, np.round(peak_start)])) : int(np.min([len(Y_abs), np.round(peak_end) + 1]))])
            / 2
        )
        phi_guess = (np.angle(Y[max_ind]) + np.pi / 2 - f_guess * coordinate_vec[0] * 2 * np.pi) % (2 * np.pi)
    return f_guess, A_guess, phi_guess, offset_guess, T_exp_guess, T_gauss_guess


def prefix_expr(model: lmfit.Model, expr: str) -> str:
    """The expression with every parameter of ``model`` it names carrying the model prefix.

    ``lmfit`` prefixes the parameters a model owns but not the names inside the expressions its hints
    carry, so an expression written in bare names refers to nothing once the model has a prefix. Rewriting
    it here keeps one spelling of the expression working for every prefix.

    Used directly where a prefixed expression is needed outside of ``set_param_hint``, such as
    ``post_fit`` adding a parameter to an already-fitted ``Parameters`` object. Everywhere a hint is set,
    prefer decorating ``set_param_hint`` with :func:`prefixed_expr` instead.

    Args:
        model: Model whose prefix and parameter names the expression is rewritten for.
        expr: Expression in bare (prefix-free) parameter names.

    Returns:
        The expression with the model prefix on each name that is a parameter of the model, and every
        other name, such as a function or a constant, left as it was.

    """
    if not model.prefix:
        return expr
    names = {name.removeprefix(model.prefix) for name in model.param_names} | set(model.param_hints)
    return re.sub(
        r"\b[A-Za-z_]\w*\b",
        lambda match: f"{model.prefix}{match.group()}" if match.group() in names else match.group(),
        expr,
    )


def prefixed_expr(set_param_hint: Callable[..., None]) -> Callable[..., None]:
    """Decorates ``set_param_hint`` to prefix every parameter name inside an ``expr`` hint.

    ``lmfit`` prefixes the parameters a model owns but never rewrites the expression *text* of its hints,
    so an ``expr`` written in bare names refers to nothing once the model has a prefix. Rewriting the
    expression here, before it reaches ``lmfit``, means every hint that ties parameters together can be
    written once, in bare names, and keeps working regardless of the prefix the model is given.

    Args:
        set_param_hint: The model's ``set_param_hint`` method, wrapped so any ``expr`` it is given is
            prefixed first.

    Returns:
        A ``set_param_hint`` that prefixes the parameter names inside ``expr`` before ``lmfit`` sees it.

    """

    @wraps(set_param_hint)
    def wrapper(self: lmfit.Model, name: str, **kwargs: Any) -> None:
        if kwargs.get("expr") is not None:
            kwargs["expr"] = prefix_expr(self, kwargs["expr"])
        return set_param_hint(self, name, **kwargs)

    return wrapper


class BaseModel(lmfit.Model):
    """Base class for the model selection fitter's models.

    Every model in this package needs its ``make_params`` and ``set_param_hint`` decorated the same way,
    so this wires that up once rather than in each model.
    """

    @filter_param_guesses
    def make_params(self, **kwargs: Any) -> lmfit.parameter.Parameters:
        """Creates parameters, dropping guesses for parameters the hints already settle."""
        return super().make_params(**kwargs)

    @prefixed_expr
    def set_param_hint(self, name: str, **kwargs: Any) -> None:
        """Sets a parameter hint, prefixing the parameter names inside any ``expr``."""
        return super().set_param_hint(name, **kwargs)
