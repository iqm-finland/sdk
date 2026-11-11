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
"""Convert raw fit arrays into the output xarray dataset."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import xarray as xr

from .model_selection import _select_best_models_values
from .structures import FitContext, FitRequest


@dataclass(frozen=True, slots=True)
class ModelArrays:
    """Raw per-model outputs produced by the fitting engine."""

    fit_result_all: np.ndarray
    fit_all: np.ndarray
    initial_guess_all: np.ndarray
    uncertainty_all: np.ndarray
    plot_defs_all: np.ndarray
    plots_all: np.ndarray | None
    outliers_all: np.ndarray | None


def _set_fit_data_attrs(fit_data: xr.DataArray, datavar: str | None) -> None:
    fit_data.attrs["parameter"] = SimpleNamespace(name="fit", label="Fit", unit="")
    fit_data.attrs["standard_name"] = "fit"
    fit_data.attrs["long_name"] = "Fit"
    fit_data.attrs["units"] = ""
    fit_data.attrs["target"] = {
        "parameter": SimpleNamespace(name=datavar, label=datavar, units=""),
        "standard_name": datavar,
        "long_name": datavar,
        "units": "",
    }


def _extract_fit_parameters(fit_result: xr.DataArray) -> dict:
    """Extract fit parameters from an object DataArray of lmfit ModelResult."""
    fit_parameters: dict[Any, dict[str, Any]] = {}
    result_values = fit_result.values
    if result_values.ndim > 0:
        for coordinate in np.ndindex(result_values.shape):
            model = result_values[coordinate]  # type: ignore[index]
            param_dict: dict[str, Any] = {}
            for param_name in model.params:  # type: ignore[attr-defined]
                param = model.params[param_name]  # type: ignore[attr-defined]
                param_dict[param_name] = param.value
                param_dict[param_name + "_stddev"] = param.stderr
            fit_parameters[coordinate] = param_dict
    else:
        model = result_values[()]  # type: ignore[index]
        param_dict = {}
        for param_name in model.params:  # type: ignore[attr-defined]
            param = model.params[param_name]  # type: ignore[attr-defined]
            param_dict[param_name] = param.value
            param_dict[param_name + "_stddev"] = param.stderr
        fit_parameters[()] = param_dict
    return fit_parameters


def add_exa_attrs(fit_data: xr.DataArray, datavar: str | None, fit_result: xr.DataArray) -> None:
    """Attach attrs in the format expected by exa tooling."""
    _set_fit_data_attrs(fit_data, datavar)
    fit_data.attrs["fit_parameters"] = _extract_fit_parameters(fit_result)


def assemble_results_dataset(
    *,
    models: list[Any],
    request: FitRequest,
    context: FitContext,
    arrays: ModelArrays,
) -> xr.Dataset:
    """Build the final output dataset from raw arrays."""
    remaining_dims = context.remaining_dims
    fit_data_coords_adjusted = context.fit_data_coords_adjusted

    model_names = [m.name for m in models]

    fit_result_dims = ("models",) + tuple(remaining_dims)
    fit_data_dims = ("models",) + tuple(fit_data_coords_adjusted)

    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
        "fit_result": (fit_result_dims, arrays.fit_result_all),
        "fit": (fit_data_dims, arrays.fit_all),
        "initial_guess": (fit_data_dims, arrays.initial_guess_all),
        "uncertainty": (fit_data_dims, arrays.uncertainty_all),
        "plot_definitions": (fit_result_dims, arrays.plot_defs_all),
    }
    if arrays.plots_all is not None:
        data_vars["plots"] = (fit_result_dims, arrays.plots_all)
    if arrays.outliers_all is not None:
        data_vars["outliers"] = (fit_data_dims, arrays.outliers_all)

    stats = _select_best_models_values(
        arrays.fit_result_all,
        request.settings.extra_parameter_penalty_factor,
        request.settings.select_by_bic,
    )

    parameters: set[str] = set()
    for model in models:
        parameters.update(model.param_names)

    best_model_values = stats.best_model
    param_vals_all = {p: np.empty(best_model_values.shape, dtype=float) for p in parameters}
    param_err_vals_all = {p: np.empty(best_model_values.shape, dtype=float) for p in parameters}
    for i, best_model in enumerate(best_model_values.flat):
        for parameter in parameters:
            param_vals_all[parameter].flat[i] = float(
                best_model.params[parameter].value if parameter in best_model.params else np.nan
            )
            param_err_vals_all[parameter].flat[i] = float(
                best_model.params[parameter].stderr
                if parameter in best_model.params and best_model.params[parameter].stderr is not None
                else 0.0
            )

    best_model_ind = stats.best_model_index
    n_models_dim, *fit_data_shape = arrays.fit_all.shape
    remaining_shape = best_model_ind.shape
    n_remaining = int(np.prod(remaining_shape)) if remaining_shape else 1
    indep_dims = fit_data_shape[len(remaining_shape) :] if len(fit_data_shape) > len(remaining_shape) else ()

    model_indices = stats.best_model_index.flatten()
    if indep_dims:
        extra_size = int(np.prod(indep_dims))
        out_shape = remaining_shape + tuple(indep_dims)
        fit_flat = arrays.fit_all.reshape((n_models_dim, n_remaining, *indep_dims))
        fit_flat = fit_flat.transpose(1, 0, *range(2, fit_flat.ndim))
        best_fit_stack = np.take_along_axis(
            fit_flat,
            model_indices[:, None, *(None,) * len(indep_dims)].repeat(extra_size, axis=1),
            axis=1,
        )
        best_fit = best_fit_stack[:, 0, ...].reshape(out_shape)
    else:
        best_fit = arrays.fit_all[model_indices, np.arange(n_remaining)].reshape(remaining_shape)

    coordinate_dict = dict(context.coordinate_dict)
    coordinate_dict["models"] = model_names

    data_vars_all: dict[str, tuple[tuple[str, ...], np.ndarray]] = dict(data_vars)
    data_vars_all["best_model"] = (tuple(remaining_dims), stats.best_model)
    data_vars_all["best_model_index"] = (tuple(remaining_dims), stats.best_model_index)
    data_vars_all["model_probabilities"] = (fit_result_dims, stats.model_probabilities)
    data_vars_all["AICs"] = (fit_result_dims, stats.AICs)
    data_vars_all["AICcs"] = (fit_result_dims, stats.AICcs)
    data_vars_all["BICs"] = (fit_result_dims, stats.BICs)
    data_vars_all["redchis"] = (fit_result_dims, stats.redchis)
    data_vars_all["r2s"] = (fit_result_dims, stats.r2s)
    for parameter in parameters:
        data_vars_all[parameter] = (tuple(remaining_dims), param_vals_all[parameter])
        data_vars_all[f"{parameter}_stderr"] = (tuple(remaining_dims), param_err_vals_all[parameter])
    data_vars_all["best_fit"] = (tuple(fit_data_coords_adjusted), best_fit)

    results_for_models = xr.Dataset(data_vars=data_vars_all, coords=coordinate_dict)
    add_exa_attrs(results_for_models["fit"], request.datavar, results_for_models["fit_result"])
    add_exa_attrs(results_for_models["best_fit"], request.datavar, results_for_models["best_model"])
    return results_for_models
