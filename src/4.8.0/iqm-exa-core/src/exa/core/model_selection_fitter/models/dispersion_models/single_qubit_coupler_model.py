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
"""Lmfit model for single qubit coupler model."""

import logging
from typing import Literal
import warnings

import lmfit
import numpy as np
import scipy.signal
import scipy.stats
import xarray as xr

from exa.core import model_selection_fitter as msf
from exa.core.util import simple_transmon_ge_frequency

from ..model_utils import BaseModel

logger = logging.getLogger(__name__)


QUBIT_COUPLER_DISPERSION_PARAMS = ["f_sweet_spot", "v_offs", "v_period", "f_q", "g", "slope", "d", "quadratic"]


def qubit_coupler_dispersion(  # noqa: PLR0913
    v: np.ndarray,
    f_sweet_spot: float,
    v_offs: float,
    v_period: float,
    f_q: float,
    g: float,
    slope: float,
    d: float,
    quadratic: float = 0.0,
) -> np.ndarray:
    """Qubit dispersion model that assumes coupling with one coupler.

    Args:
        v: The coupler bias voltages.
        f_sweet_spot: The frequency of the coupler at the sweet spot, in GHz.
        v_offs: The coupler bias voltage offset.
        v_period: The coupler bias voltage period.
        f_q: The qubit frequency, in GHz.
        g: The interaction strength between the qubit and the coupler, in GHz.
        slope: Linear slope on the qubit frequency caused by flux crosstalk, GHz/V.
        d: SQUID asymmetry of the coupler.
        quadratic: Quadratic change on the qubit frequency caused by flux crosstalk, GHz/V**2.

    Returns:
        Frequency of the qubit-like mode, in GHz.

    """
    f_c = simple_transmon_ge_frequency(v, f_sweet_spot, v_offs, v_period, d)
    f_q_slope = f_q + slope * v + quadratic * v**2
    lower = 0.5 * (f_c + f_q_slope) - 0.5 * np.sqrt((f_c - f_q_slope) ** 2 + 4 * g**2)
    upper = 0.5 * (f_c + f_q_slope) + 0.5 * np.sqrt((f_c - f_q_slope) ** 2 + 4 * g**2)
    closest_index = np.argmin(np.vstack((np.abs(lower - f_q_slope), np.abs(upper - f_q_slope))), axis=0)
    f_q_coupled = np.array([lower[ii] if index == 0 else upper[ii] for ii, index in enumerate(closest_index)])
    return f_q_coupled


def qubit_coupler_dispersion_guess(
    data: np.ndarray, v: np.ndarray, f_sweet_spot: float, peak_prominence: float = 0.004
) -> tuple:
    """Guess for qubit coupler dispersion model.

    Tries to make a guess for the qubit-coupler dispersion model
    parameters by identifying the bias voltage values where the
    coupler crosses the qubits. If there are enough crossings,
    the bias voltage offset and period can be accurately determined.

    Args:
        data: Qubit frequencies in GHz.
        v: Coupler bias voltage in V.
        f_sweet_spot: Guess for coupler sweet spot frequency in GHz, in case it cannot be deduced from the data.
            The current version does not make a guess for this parameter but just passes the given value.
        peak_prominence: Which peaks to select.

    Returns:
        f_sweet_spot: Guess for coupler sweet spot frequency in GHz.
        v_offs: Voltage offset that displaces the sweet spot.
        v_period: Period of the repeating pattern.
        f_q: Median of the qubit frequency, in GHz.
        g: Guess for the coupling constant, in GHz.
        slope: 0.
        peak_voltages: Flux bias voltage values for each of the detected qubit-coupler crossings in the data.
        peak_prominence: Value for peak prominence used in peak detection.

    """
    g = 0.1
    not_nans = ~np.isnan(data)
    data_without_nans = data[not_nans]
    v_without_nans = v[not_nans]
    f_q = np.median(data_without_nans)
    linear_model = msf.models.LinearModel()
    fitter = msf.Fitter()
    fitter.add_model(linear_model)
    fit_result = fitter.fit(xr.DataArray(data=data_without_nans, dims=["v"], coords={"v": v_without_nans}), x="v")
    slope = fit_result.a.values[()]
    interpolation_factor = 5
    v_interp = np.interp(
        np.linspace(0, 1, len(v_without_nans) * interpolation_factor),
        np.linspace(0, 1, len(v_without_nans)),
        v_without_nans,
    )

    data_interp = np.interp(v_interp, v_without_nans, data_without_nans) - slope * v_interp

    diff_array = scipy.signal.savgol_filter(
        np.abs(np.diff(data_interp, prepend=data_interp[0])),
        np.max([5, 3 * interpolation_factor]),
        2,
    )
    with warnings.catch_warnings():
        # Suppress peak not found warnings. They are handled below.
        warnings.simplefilter("ignore")
        peaks, _ = scipy.signal.find_peaks(
            diff_array, prominence=peak_prominence / interpolation_factor, distance=2 * interpolation_factor
        )

    if len(peaks) == 0:
        logger.warning("No peaks found in the data. Cannot proceed with the guess.")
        return f_q - g, 0.0, np.abs(v_interp[-1] - v_interp[0]) / 3, f_q, g, slope, []

    nbr_of_intervals = len(peaks) + 1

    peak_sign_for_interval = []
    # -1 -> coupler below qubit
    # +1 -> coupler above qubit
    previous_peak_index = 0
    for index in range(1, nbr_of_intervals):
        average_first = np.median(data_interp[previous_peak_index : peaks[index - 1]])
        average_second = np.median(data_interp[peaks[index - 1] : peaks[index] if index < len(peaks) else None])
        peak_sign_for_interval.append(1 if average_first < average_second else -1)
        previous_peak_index = peaks[index - 1]
    peak_sign_for_interval.append(-peak_sign_for_interval[-1])
    center_point_values = []
    n_periods = []
    n_previous = -1.0
    for index in range(1, len(peaks)):
        if index == 1 and peak_sign_for_interval[index - 1] < 0:
            n_previous = n_previous + 0.5
        center_point = (v_interp[peaks[index - 1]] + v_interp[peaks[index]]) / 2
        n_period = 0.5
        center_point_values.append(center_point)
        n_periods.append(n_previous + n_period)
        n_previous = n_periods[-1]
    if len(center_point_values) == 1:
        v_offs = center_point_values[0]
        v_period = np.abs(v_interp[-1] - v_interp[0]) / 3
    elif len(center_point_values) == 0:
        v_offs = np.abs(v_interp[-1] + v_interp[0]) / 2
        v_period = np.abs(v_interp[-1] - v_interp[0]) / 3
    else:
        # Find period and offset from a linear fit
        [v_offs, v_period] = np.polynomial.polynomial.polyfit(n_periods, center_point_values, 1)

    v_offs_pos = v_offs % v_period
    v_offs_neg = v_offs % (-1 * v_period)
    v_offs = v_offs_pos if np.abs(v_offs_pos) < np.abs(v_offs_neg) else v_offs_neg

    return f_sweet_spot, v_offs, v_period, f_q, g, slope, v_interp[peaks]


class QubitCouplerDispersionModel(BaseModel):  # noqa: D101
    __doc__ = lmfit.models.COMMON_INIT_DOC

    def __init__(self, independent_vars: list[Literal["v"]] | None = None, **kwargs) -> None:
        """Model for qubit dispersion as a function of a coupler bias.

        Model parameters:

        * v: Flux bias voltage, independent parameter in the model.
        * f_sweet_spot: Coupler sweet spot frequency, in GHz.
        * v_offs: Coupler flux bias offset.
        * v_period: Coupler flux bias period.
        * f_q: Qubit frequency, in GHz.
        * g: Qubit-coupler coupling, in GHz.
        * slope:
            Additional slope in qubit frequency as a function of coupler bias voltage.
            Accounts for flux crosstalk, in GHz/V.
        * d: Asymmetry of the coupler squid loop.
        * Ec: Charging energy of the coupler, in GHz.
        """
        if independent_vars is None:
            independent_vars = ["v"]
        super().__init__(
            qubit_coupler_dispersion,
            independent_vars=independent_vars,
            param_names=QUBIT_COUPLER_DISPERSION_PARAMS,
            prefix="",
            **kwargs,
        )
        self.set_param_hint(name="f_sweet_spot", value=7.8, min=0, vary=True)
        self.set_param_hint(name="v_offs", value=0.0, min=-1, max=1, vary=True)
        self.set_param_hint(name="v_period", value=10.0, min=0.01, vary=True)
        self.set_param_hint(name="f_q", value=4.1, min=0, vary=True)
        self.set_param_hint(name="g", value=0.1, min=0, vary=True)
        self.set_param_hint(name="slope", value=0.0, vary=True)
        self.set_param_hint(name="quadratic", value=0.0, vary=True)
        self.set_param_hint(name="d", value=0.0, vary=False, min=0, max=1)
        self.set_param_hint(name="Ec", value=200e-3, vary=False)

        self.set_param_hint(name="Ejs", expr="(f_sweet_spot + Ec)**2/(8*Ec)", vary=False)

    def guess(self, data: np.ndarray, v: np.ndarray, *args, **kws) -> lmfit.parameter.Parameters:
        """Initial guess for the model."""
        _, v_offs, v_period, f_q, g, slope, _ = qubit_coupler_dispersion_guess(
            data, v, self.param_hints["f_sweet_spot"]["value"]
        )
        Ejs = eval(
            self.param_hints["Ejs"]["expr"],
            {},
            {
                "f_sweet_spot": self.param_hints["f_sweet_spot"]["value"],
                "Ec": self.param_hints["Ec"]["value"],
            },
        )
        params_map = {
            "v_offs": v_offs if self.param_hints["v_offs"]["vary"] else self.param_hints["v_offs"]["value"],
            "v_period": v_period if self.param_hints["v_period"]["vary"] else self.param_hints["v_period"]["value"],
            "g": g if self.param_hints["g"]["vary"] else self.param_hints["g"]["value"],
            "f_q": f_q if self.param_hints["f_q"]["vary"] else self.param_hints["f_q"]["value"],
            "slope": slope if self.param_hints["slope"]["vary"] else self.param_hints["slope"]["value"],
            "quadratic": 0.0 if self.param_hints["quadratic"]["vary"] else self.param_hints["quadratic"]["value"],
            "Ejs": Ejs,
        }
        params = self.make_params(**params_map)

        return lmfit.models.update_param_vals(params, self.prefix, **kws)

    def post_fit(self, fitresult: lmfit.model.ModelResult) -> None:
        fitresult.params.add("EJs", expr="(f_sweet_spot + Ec)**2/(8*Ec)")
