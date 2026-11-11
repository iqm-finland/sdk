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
"""Lmfit model for double qubit coupler model."""

from typing import Literal

import lmfit
import numpy as np
import xarray as xr

from .simple_transmon_model import SimpleTransmonDispersionModel
from .single_qubit_coupler_model import qubit_coupler_dispersion, qubit_coupler_dispersion_guess
from ..model_utils import BaseModel
from ...fitter import Fitter

DOUBLE_QUBIT_COUPLER_DISPERSION_PARAMS = [
    "f_sweet_spot",
    "v_offs",
    "v_period",
    "f_q1",
    "f_q2",
    "g1",
    "g_difference",
    "slope1",
    "slope2",
    "quadratic1",
    "quadratic2",
    "d",
]


def double_qubit_coupler_dispersion(  # noqa: PLR0913
    v: np.ndarray,
    qubits: list[str],
    f_sweet_spot: float,
    v_offs: float,
    v_period: float,
    f_q1: float,
    f_q2: float,
    g1: float,
    g_difference: float,
    slope1: float,
    slope2: float,
    d: float,
    quadratic1: float = 0.0,
    quadratic2: float = 0.0,
) -> np.ndarray:
    """Qubit dispersion model that assumes coupling with two couplers.

    Args:
        v: The coupler bias voltages.
        qubits: The qubits under consideration.
        f_sweet_spot: sweet spot of the coupler.
        v_offs: coupler bias voltage offset.
        v_period: coupler bias voltage period.
        f_q1: The frequency of the first qubit.
        f_q2: The frequency of the second qubit.
        g1: The interaction strength between the first qubit and the coupler.
        g_difference: interaction strength.
        slope1: linear slope on the first qubit frequency caused by flux crosstalk.
        slope2: linear slope on the second qubit frequency caused by flux crosstalk.
        d: SQUID asymmetry of the coupler.
        quadratic1: quadratic change on the first qubit frequency caused by flux crosstalk.
        quadratic2: quadratic change on the second qubit frequency caused by flux crosstalk.

    Returns:
        Frequencies of the qubit-like modes.

    """
    v = np.asarray(v)
    f_q1_coupled = qubit_coupler_dispersion(v, f_sweet_spot, v_offs, v_period, f_q1, g1, slope1, d, quadratic1)
    f_q2_coupled = qubit_coupler_dispersion(
        v, f_sweet_spot, v_offs, v_period, f_q2, g1 * g_difference, slope2, d, quadratic2
    )
    y = np.array([f_q1_coupled, f_q2_coupled]).T
    return y


class DoubleQubitCouplerDispersionModel(BaseModel):
    """Double Qubit Coupler Dispersion Model."""

    __doc__ = lmfit.models.COMMON_INIT_DOC

    def __init__(self, independent_vars: list[Literal["v", "qubits"]] | None = None, **kwargs) -> None:
        """Simultaneous fit of qubit spectra coupled to a shared coupler."""
        if independent_vars is None:
            independent_vars = ["v", "qubits"]
        super().__init__(
            double_qubit_coupler_dispersion,
            independent_vars=independent_vars,
            param_names=DOUBLE_QUBIT_COUPLER_DISPERSION_PARAMS,
            prefix="",
            **kwargs,
        )
        self.set_param_hint(name="f_sweet_spot", value=7.8, min=0, vary=True)
        self.set_param_hint(name="v_offs", value=0.0, min=-1, max=1, vary=True)
        self.set_param_hint(name="v_period", value=10, min=0.01, vary=True)
        self.set_param_hint(name="f_q1", value=4.3, min=1, max=10, vary=True)
        self.set_param_hint(name="f_q2", value=4.5, min=1, max=10, vary=True)
        self.set_param_hint(name="g1", value=0.07, min=0, max=0.3, vary=True)
        self.set_param_hint(name="g_difference", value=1, vary=True)
        self.set_param_hint(name="g2", value=1.0, expr="g1 * g_difference", vary=False)
        self.set_param_hint(name="slope1", value=0.0, min=-1, max=1, vary=True)
        self.set_param_hint(name="slope2", value=0.0, min=-1, max=1, vary=True)
        # Fixed at zero by default, so the model behaves as it did before they were added.
        self.set_param_hint(name="quadratic1", value=0.0, vary=False)
        self.set_param_hint(name="quadratic2", value=0.0, vary=False)
        self.set_param_hint(name="d", value=0.0, vary=False, min=0, max=1)
        self.set_param_hint(name="Ec", value=200e-3, vary=False)
        self.set_param_hint(name="Ejs", expr="(f_sweet_spot + Ec)**2/(8*Ec)", vary=False)

    def guess(self, data: np.ndarray, v: np.ndarray, qubits: list[str], *args, **kws) -> lmfit.parameter.Parameters:
        """Initial guess for the model."""
        if len(data.shape) < 2:
            raise ValueError(f"Data array must contain at least two dimensions, but only has {len(data.shape)}.")
        f_sweet_spot = self.param_hints["f_sweet_spot"]["value"]
        (
            _,
            v_offs,
            v_period,
            f_q1,
            g1,
            slope1,
            peak_voltages_1,
        ) = qubit_coupler_dispersion_guess(data[:, 0], v, f_sweet_spot)
        (
            _,
            v_offs,
            v_period,
            f_q2,
            g2,
            slope2,
            peak_voltages_2,
        ) = qubit_coupler_dispersion_guess(data[:, 1], v, f_sweet_spot)

        fitter = Fitter()
        model = SimpleTransmonDispersionModel()
        model.set_param_hint("v_period", value=v_period)
        model.set_param_hint("v_offs", value=v_offs)
        model.set_param_hint("f_sweet_spot", value=f_sweet_spot)
        fitter.add_model(model)

        peak_voltages = np.concatenate((peak_voltages_1, peak_voltages_2))
        peak_frequencies = np.array([f_q1] * len(peak_voltages_1) + [f_q2] * len(peak_voltages_2))
        data_array = xr.DataArray(peak_frequencies, dims=("voltage"), coords={"voltage": peak_voltages})
        fit_result = fitter.fit(data_array, plot_fit=False, v="voltage", override_model_initial_guess=True)
        f_sweet_spot = fit_result["f_sweet_spot"].values[()]
        v_period = fit_result["v_period"].values[()]
        v_offs = fit_result["v_offs"].values[()]

        Ejs = eval(
            self.param_hints["Ejs"]["expr"],
            {},
            {"f_sweet_spot": f_sweet_spot, "Ec": self.param_hints["Ec"]["value"]},
        )
        params_map = {
            "f_sweet_spot": f_sweet_spot
            if self.param_hints["f_sweet_spot"]["vary"]
            else self.param_hints["f_sweet_spot"]["value"],
            "v_offs": v_offs if self.param_hints["v_offs"]["vary"] else self.param_hints["v_offs"]["value"],
            "v_period": v_period if self.param_hints["v_period"]["vary"] else self.param_hints["v_period"]["value"],
            "g1": g1 if self.param_hints["g1"]["vary"] else self.param_hints["g1"]["value"],
            "g_difference": g2 / g1
            if self.param_hints["g_difference"]["vary"]
            else self.param_hints["g_difference"]["value"],
            "f_q1": f_q1 if self.param_hints["f_q1"]["vary"] else self.param_hints["f_q1"]["value"],
            "f_q2": f_q2 if self.param_hints["f_q2"]["vary"] else self.param_hints["f_q2"]["value"],
            "slope1": slope1 if self.param_hints["slope1"]["vary"] else self.param_hints["slope1"]["value"],
            "slope2": slope2 if self.param_hints["slope2"]["vary"] else self.param_hints["slope2"]["value"],
            "g2": g2,
            "Ejs": Ejs,
        }
        params = self.make_params(**params_map)
        return lmfit.models.update_param_vals(params, self.prefix, **kws)

    def post_fit(self, fitresult: lmfit.model.ModelResult) -> None:
        fitresult.params.add("g2", expr="g1 * g_difference")
        fitresult.params.add("Ejs", expr="(f_sweet_spot + Ec)**2/(8*Ec)")
