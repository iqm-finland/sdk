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
"""Lmfit model for simple transmon model."""

from typing import Literal

import lmfit
import numpy as np

from exa.core.util import simple_transmon_ge_frequency

from ..model_utils import BaseModel

SIMPLE_TRANSMON_GE_FREQUENCY_PARAMS = ["f_sweet_spot", "v_offs", "v_period", "d"]


class SimpleTransmonDispersionModel(BaseModel):
    r"""Simple Transmon Dispersion Model.

    Transmon spectrum fit:

    .. math::

       f_q(v) = f_{sweet_spot} * (\cos(\pi*(v-v_offs)/v_period)^2 + (d*\sin(\pi*(v-v_offs)/v_period))^2)^0.25

    For compatibility with the full model, also sets the values for Josephson
    energy Ejs and the charging energy Ec.
    """

    __doc__ = lmfit.models.COMMON_INIT_DOC

    def __init__(self, independent_vars: list[Literal["v"]] | None = None, **kwargs) -> None:
        if independent_vars is None:
            independent_vars = ["v"]
        super().__init__(
            simple_transmon_ge_frequency,
            independent_vars=independent_vars,
            param_names=SIMPLE_TRANSMON_GE_FREQUENCY_PARAMS,
            prefix="",
            **kwargs,
        )
        self.set_param_hint(name="f_sweet_spot", value=4.5, min=0, vary=True)
        self.set_param_hint(name="v_offs", value=0, min=-10, max=10, vary=True)
        self.set_param_hint(name="v_period", value=10, min=0.01, vary=True)
        self.set_param_hint(name="d", value=0, vary=False, min=0, max=1)
        self.set_param_hint(name="Ec", value=200e-3, vary=False)
        self.set_param_hint(name="Ejs", expr="(f_sweet_spot + Ec)**2/(8*Ec)", vary=False)

    def guess(self, data: np.ndarray, v: np.ndarray, **kws) -> lmfit.parameter.Parameters:
        """Initial guess for the model.

        Makes the guess by fitting a second order polynomial to the data
        around zero flux.

        If there is not enough data to make a guess for parameters using
        quadratic model, sweet spot frequency is assumed to be at v_offs=0.0
        and the current value for f_sweet_spot is used.

        Args:
            data: Qubit frequencies.
            v: Qubit bias voltage.
            kws: Values for the model parameters.

        """
        diff_data = np.abs(np.gradient(data))
        median_diff = np.median(diff_data)
        good_data = np.arange(len(data)) if len(data) < 4 else np.squeeze(np.argwhere(diff_data < median_diff * 6))

        if len(data[good_data]) > 2:
            x_data, y_data = v[good_data], data[good_data]
        elif len(data[good_data]) > 1:
            # Not enough data for polyfit, fix offset to zero.
            # a*v**2 + c
            # Add a new datapoint assumed to be the sweet spot.
            A = max([self.param_hints["f_sweet_spot"]["value"], 1.05 * np.max(data[good_data])])
            y_data = np.concatenate((data[good_data], [A]))
            x_data = np.concatenate((v[good_data], [0.0]))
        else:
            # Not enough data, just return the current values.
            return lmfit.models.update_param_vals(self.make_params(), self.prefix, **kws)

        coefficients = np.polynomial.polynomial.polyfit(x_data, y_data, 2)

        f_sweet_spot = (
            coefficients[1] ** 2 / 4 / coefficients[2] - coefficients[1] ** 2 / 2 / coefficients[2] + coefficients[0]
        )
        v_offs = -1 * coefficients[1] / 2 / coefficients[2]
        v_period = np.sqrt(np.abs(-(np.pi**2) / 2 / coefficients[2])) * np.sqrt(2)
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
            "d": 0 if self.param_hints["d"]["vary"] else self.param_hints["d"]["value"],
            "Ejs": Ejs,
        }
        params = self.make_params(**params_map)

        return lmfit.models.update_param_vals(params, self.prefix, **kws)

    def post_fit(self, fitresult: lmfit.model.ModelResult) -> None:
        fitresult.params.add("Ejs", expr="(f_sweet_spot + Ec)**2/(8*Ec)")
