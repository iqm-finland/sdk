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
"""Lmfit model for Ramsey2f fitted in quadratures."""

from typing import ClassVar, Literal

import lmfit
import numpy as np

from .model_utils import BaseModel, frequency_guess_with_fft
from .ramsey2f_model import Ramsey2fModel

RAMSEY_2F_SEPARABLE_PARAMS = ["f_q", "tau", "T_exp", "c_cos", "c_sin", "offset_1", "offset_2"]

RAMSEY_2F_LINEAR_PARAMS = ("c_cos", "c_sin", "offset_1", "offset_2")


def ramsey_2f_basis(t: np.ndarray, f_d: np.ndarray, f_q: float, tau: float, T_exp: float) -> np.ndarray:
    r"""Basis matrix for the four linear parameters of the two-tone Ramsey response.

    With :math:`\Theta_j = 2\pi(f_q - f_{d,j})(t - t_0 + \tau)`, the shared oscillation expands as

    .. math::
        A\sin(\Theta_j + \phi) = (A\cos\phi)\sin\Theta_j + (A\sin\phi)\cos\Theta_j,

    so the first two columns are the sine and the cosine of that angle under the decay envelope, and the
    last two the offset of each tone.

    Args:
        t: The time coordinate values.
        f_d: Frequencies of the two drive tones.
        f_q: Frequency of the qubit.
        tau: Constant time offset, accounting for phase the time coordinate does not carry.
        T_exp: Exponential decay time, shared by both tones.

    Returns:
        Array of shape ``(len(t) * len(f_d), 4)``, rows in ``np.ravel`` order of the
        ``(len(t), len(f_d))`` response, so ``basis @ [c_cos, c_sin, offset_1, offset_2]`` is the flattened
        response.

    """
    t = np.asarray(t, dtype=float)
    f_d = np.asarray(f_d, dtype=float)
    elapsed = t - t[0]
    # Non-positive decay: +inf rather than overflow, since the solver reads a non-finite basis as a
    # direction not to go in.
    envelope = np.exp(-elapsed / T_exp) if T_exp > 0.0 else np.full(elapsed.shape, np.inf)
    angle = 2 * np.pi * (f_q - f_d)[None, :] * (elapsed[:, None] + tau)

    basis = np.zeros((t.size, f_d.size, len(RAMSEY_2F_LINEAR_PARAMS)))
    basis[:, :, 0] = np.sin(angle) * envelope[:, None]
    basis[:, :, 1] = np.cos(angle) * envelope[:, None]
    basis[:, 0, 2] = 1.0
    basis[:, 1, 3] = 1.0
    return basis.reshape(t.size * f_d.size, len(RAMSEY_2F_LINEAR_PARAMS))


def ramsey_2f_separable_response(  # noqa: PLR0913
    t: np.ndarray,
    f_d: np.ndarray,
    f_q: float,
    tau: float,
    T_exp: float,
    c_cos: float,
    c_sin: float,
    offset_1: float,
    offset_2: float,
) -> np.ndarray:
    """Response of a two-tone Ramsey experiment, in quadratures rather than an amplitude and a phase.

    Identical to :func:`~exa.core.model_selection_fitter.models.ramsey2f_model.ramsey_2f_response` with
    ``A = hypot(c_cos, c_sin)`` and ``phi = arctan2(c_sin, c_cos)``.

    Args:
        t: The time coordinate values.
        f_d: Frequencies of the two drive tones.
        f_q: Frequency of the qubit.
        tau: Constant time offset, accounting for phase the time coordinate does not carry.
        T_exp: Exponential decay time, shared by both tones.
        c_cos: In-phase quadrature, ``A * cos(phi)``.
        c_sin: Out-of-phase quadrature, ``A * sin(phi)``.
        offset_1: Offset of the first tone.
        offset_2: Offset of the second tone.

    Returns:
        Response values, of shape ``(len(t), len(f_d))``.

    """
    basis = ramsey_2f_basis(t, f_d, f_q, tau, T_exp)
    coefficients = np.array([c_cos, c_sin, offset_1, offset_2], dtype=float)
    return np.asarray(basis @ coefficients).reshape(t.size, f_d.size)


class Ramsey2fSeparableModel(BaseModel):
    r"""Two-tone Ramsey model fitted in quadratures rather than an amplitude and a phase.

    Same response as :class:`~exa.core.model_selection_fitter.models.ramsey2f_model.Ramsey2fModel`,
    reparameterised as in :func:`ramsey_2f_separable_response`.
    """

    __doc__ = lmfit.models.COMMON_INIT_DOC

    linear_parameter_names: ClassVar[tuple[str, ...]] = RAMSEY_2F_LINEAR_PARAMS
    """The parameters the response is linear in, in the column order of :func:`ramsey_2f_basis`."""

    def __init__(self, independent_vars: list[Literal["t", "f_d"]] | None = None, **kwargs) -> None:
        if independent_vars is None:
            independent_vars = ["t", "f_d"]
        super().__init__(
            ramsey_2f_separable_response,
            independent_vars=independent_vars,
            param_names=RAMSEY_2F_SEPARABLE_PARAMS,
            **kwargs,
        )
        self.set_param_hint(name="f_q", value=0)
        self.set_param_hint(name="tau", value=0)
        self.set_param_hint(name="T_exp", value=50e-6, min=1e-15, vary=True)
        self.set_param_hint(name="c_cos", value=0)
        self.set_param_hint(name="c_sin", value=1)
        self.set_param_hint(name="offset_1", value=0.5)
        self.set_param_hint(name="offset_2", value=0.5)

    def separable_basis(self, t: np.ndarray, f_d: np.ndarray, f_q: float, tau: float, T_exp: float) -> np.ndarray:
        """Basis the linear parameters multiply, for a solver that eliminates them.

        See :func:`ramsey_2f_basis` for the arguments and the layout of the returned matrix.
        """
        return ramsey_2f_basis(t, f_d, f_q, tau, T_exp)

    def guess(self, data: np.ndarray, t: np.ndarray, f_d: np.ndarray, **kws) -> lmfit.parameter.Parameters:
        """Initial guess for the model.

        Delegates to :class:`~exa.core.model_selection_fitter.models.ramsey2f_model.Ramsey2fModel`, whose
        guess is in an amplitude and a phase, and converts those to the two quadratures.

        Args:
            data: 2D array to fit.
            t: The time coordinate values.
            f_d: Frequencies of the two drive tones.
            kws: Values for the model parameters.

        """
        guessed = Ramsey2fModel().guess(data, t=t, f_d=f_d)
        decay = float(guessed["T_exp"].value)
        f_q, tau, linear = self._starting_point(
            np.asarray(data, dtype=float),
            np.asarray(t, dtype=float),
            np.asarray(f_d, dtype=float),
            float(guessed["f_q"].value),
            decay,
        )
        guesses = {
            "f_q": f_q,
            "tau": self._tau_guess(tau, f_d),
            "T_exp": decay,
            **dict(zip(RAMSEY_2F_LINEAR_PARAMS, linear, strict=True)),
        }
        params = self.make_params(**guesses)
        return lmfit.models.update_param_vals(params, self.prefix, **kws)

    @staticmethod
    def _starting_point(
        data: np.ndarray, t: np.ndarray, f_d: np.ndarray, f_q: float, decay: float
    ) -> tuple[float, float, np.ndarray]:
        """Initial guess for the qubit frequency, the timing offset, and the linear parameters.

        Args:
            data: 2D array to fit, one column per tone.
            t: The time coordinate values.
            f_d: Frequencies of the two drive tones.
            f_q: The frequency ``Ramsey2fModel`` guessed for the qubit.
            decay: The decay time guessed for both tones.

        Returns:
            The qubit frequency, the timing offset, and ``(c_cos, c_sin, offset_1, offset_2)``.

        """
        separation = float(np.ptp(f_d))
        frequencies = [f_q]
        for tone, drive in enumerate(f_d):
            peak, *_ = frequency_guess_with_fft(data[:, tone], t - t[0])
            frequencies += [drive + peak, drive - peak]
        # Endpoint left out: a closed period would offer the same tau twice.
        offsets = np.linspace(-0.5, 0.5, 33)[:-1] / separation if separation else np.zeros(1)

        observed = data.ravel()
        best = (np.inf, f_q, 0.0, np.array([0.0, 1.0, 0.5, 0.5]))
        for candidate in frequencies:
            for offset in offsets:
                basis = ramsey_2f_basis(t, f_d, float(candidate), float(offset), decay)
                if not np.isfinite(basis).all():
                    continue
                coefficients, *_ = np.linalg.lstsq(basis, observed, rcond=None)
                residual = float(np.linalg.norm(observed - basis @ coefficients))
                if residual < best[0]:
                    best = (residual, float(candidate), float(offset), coefficients)
        return best[1], best[2], best[3]

    def _tau_guess(self, guessed: float, f_d: np.ndarray) -> float | dict[str, float]:
        """The guess for ``tau``, wrapped into the window that makes it unique, with that window as bounds.

        Args:
            guessed: The value guessed for ``tau``.
            f_d: Frequencies of the two drive tones.

        Returns:
            The guess with its bounds, or the bare guess if the tones coincide or the caller has bounded
            ``tau``.

        """
        separation = float(np.ptp(np.asarray(f_d, dtype=float)))
        hint = self.param_hints.get("tau", {})
        if not separation or "min" in hint or "max" in hint:
            return guessed
        half_period = 0.5 / separation
        wrapped = (guessed + half_period) % (2 * half_period) - half_period
        return {"value": wrapped, "min": -half_period, "max": half_period}

    def post_fit(self, fitresult: lmfit.model.ModelResult) -> None:
        """Converts the quadratures back to amplitude and phase."""
        c_cos, c_sin = f"{self.prefix}c_cos", f"{self.prefix}c_sin"
        fitresult.params.add(f"{self.prefix}A", expr=f"sqrt({c_cos}**2 + {c_sin}**2)")
        fitresult.params.add(f"{self.prefix}phi", expr=f"arctan2({c_sin}, {c_cos})")
