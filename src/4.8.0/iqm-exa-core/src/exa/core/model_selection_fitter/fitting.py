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
"""Per-model fitting loop and plot helpers for the model-selection fitter."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
import logging
from typing import Any, cast

import lmfit
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

from exa.core.analyze_result import PlotDefinition, wrap_mpl_plot

from .structures import FitContext, FitRequest
from .utils import (
    get_formatted_label,
    get_matching_indices,
    outlier_detection,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelFitOutputs:
    """Per-model fit outputs as NumPy arrays.
    This dataclass is used to avoid many xarray operations that are performance-heavy when done repeatedly in a loop.
    """

    fit_result: np.ndarray
    fit: np.ndarray
    initial_guess: np.ndarray
    uncertainty: np.ndarray
    plot_definitions: np.ndarray
    plots: np.ndarray | None
    outliers: np.ndarray | None


def _iter_remaining_slices(
    *,
    data_ndim: int,
    remaining_shape: tuple[int, ...],
    remaining_axes: list[int],
) -> Iterator[tuple[tuple[int, ...], tuple[int | slice, ...]]]:
    """Yield (remaining_ind, indexer_tuple) for slicing data arrays."""
    base_indexer: list[int | slice] = [slice(None)] * data_ndim
    if remaining_shape:
        remaining_multi_inds: Iterable[tuple[int, ...]] = np.ndindex(remaining_shape)
    else:
        remaining_multi_inds = [()]

    for remaining_ind in remaining_multi_inds:
        indexer: list[int | slice] = list(base_indexer)
        for axis, idx_val in zip(remaining_axes, remaining_ind):
            indexer[axis] = idx_val
        yield remaining_ind, tuple(indexer)


def _apply_nan_policy(
    *,
    sliced_data_values: np.ndarray,
    sliced_weights_values: np.ndarray | None,
    independent_vars_current: dict[str, Any],
    remove_nans: bool,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    """Apply NaN masking for 1D data when requested."""
    if not remove_nans:
        return sliced_data_values, sliced_weights_values, independent_vars_current

    if np.ndim(sliced_data_values) > 1:
        logger.warning("NaN removal is not supported for fits with more than 1 dimension.")
        return sliced_data_values, sliced_weights_values, independent_vars_current

    not_nans = ~np.isnan(sliced_data_values)
    if not np.any(~not_nans):
        # All-NaN slice; let lmfit handle (likely raise) with original arrays.
        return sliced_data_values, sliced_weights_values, independent_vars_current

    independent_vars_masked = {k: v[not_nans] for k, v in independent_vars_current.items()}
    y = sliced_data_values[not_nans]
    w = sliced_weights_values[not_nans] if sliced_weights_values is not None else None
    return y, w, independent_vars_masked


def _apply_outlier_detection(
    *,
    model: lmfit.Model,
    sliced_data_values: np.ndarray,
    sliced_weights_values: np.ndarray | None,
    independent_vars_current: dict[str, Any],
    request: FitRequest,
    context: FitContext,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any], np.ndarray | None]:
    """Optionally run outlier detection and map indices to original coordinates."""
    if not request.settings.use_outlier_detection:
        return sliced_data_values, sliced_weights_values, independent_vars_current, None

    # NOTE: outlier_detection only supports 1D fits.
    fitter_options_od = dict(request.fitter_options)
    fitter_options_od["calc_covar"] = False

    outlier_inds = outlier_detection(
        np.asarray(sliced_data_values),
        model,
        independent_vars=independent_vars_current,
        fitter_options=fitter_options_od,
        override_model_initial_guess=request.settings.override_model_initial_guess,
        **request.outlier_detection_options,
    )

    keep = np.invert(outlier_inds)
    y = np.asarray(sliced_data_values)[keep]
    w = None if sliced_weights_values is None else np.asarray(sliced_weights_values)[keep]
    independent_vars_filtered = {k: np.asarray(v)[keep] for k, v in independent_vars_current.items()}

    outlier_index_bool: np.ndarray | None = None
    for independent_value_name, filtered_values in independent_vars_filtered.items():
        outlier_inds_original_coordinates = get_matching_indices(
            context.coords[independent_value_name], filtered_values
        )
        outlier_index_bool = np.ones(len(context.coords[independent_value_name]), dtype=bool)
        np.put(outlier_index_bool, outlier_inds_original_coordinates, np.array([False]), mode="wrap")
        break

    return y, w, independent_vars_filtered, outlier_index_bool


def _make_params(model: lmfit.Model) -> lmfit.Parameters:
    """Workaround for lmfit make_params call to handle None in parameter hints."""
    none_params = {
        name: 0 for name in model.param_names if model.param_hints.get(name, {"value": None}).get("value") is None
    }
    params = model.make_params(**none_params)
    return params


def _fit_model_arrays(  # noqa: PLR0912, PLR0913, PLR0915
    model: lmfit.Model,
    request: FitRequest,
    context: FitContext,
) -> ModelFitOutputs:
    """Fit one model across all remaining coordinate slices.

    Returns raw NumPy arrays to avoid heavy xarray concat/merge operations.
    """
    data, _, _ = context.processed_dataset

    dim_to_axis = {dim: axis for axis, dim in enumerate(data.dims)}
    remaining_axes = [dim_to_axis[dim] for dim in context.remaining_dims]

    data_values = data.values
    weights_values = request.weights.values if request.weights is not None else None
    remove_nans = request.fitter_options.get("nan_policy", "omit") == "omit"

    fit_result = np.empty(context.remaining_shape, dtype=object)
    fit = np.empty(context.data_dimensions, dtype=float)
    initial_guess = np.empty(context.data_dimensions, dtype=float)
    uncertainty = np.empty(context.data_dimensions, dtype=float)
    plot_defs = np.empty(context.remaining_shape, dtype=object)
    plots = np.empty(context.remaining_shape, dtype=object) if request.settings.add_plot_to_dataset else None
    outliers = np.empty(context.data_dimensions, dtype=bool) if request.settings.use_outlier_detection else None

    # Precompute labels/scaling once for plotting.
    y_scaling, y_label = get_formatted_label(data)
    if len(context.independent_coord_names) >= 1:
        x_scaling, x_label = get_formatted_label(data.coords[context.independent_coord_names[0]])
    else:
        x_scaling, x_label = 1.0, ""

    # Interpolation coordinates are reused for all slices.
    coords_interpolated: dict[str, np.ndarray]
    try:
        coords_interpolated = {}
        for ii, (key, value) in enumerate(context.coords.items()):
            coords_interpolated[key] = (
                np.linspace(np.nanmin(value), np.nanmax(value), np.nanmax([501, len(value)])) if ii == 0 else value
            )
    except Exception:
        coords_interpolated = context.coords

    independent_vars_current_base: dict[str, Any] = dict(zip(context.coords_keys, context.coords_values))

    for remaining_ind, indexer_tuple in _iter_remaining_slices(
        data_ndim=data.ndim,
        remaining_shape=context.remaining_shape,
        remaining_axes=remaining_axes,
    ):
        sliced_data_values = data_values[indexer_tuple]
        sliced_weights_values = weights_values[indexer_tuple] if weights_values is not None else None

        independent_vars_current: dict[str, Any] = dict(independent_vars_current_base)

        sliced_data_values, sliced_weights_values, independent_vars_current = _apply_nan_policy(
            sliced_data_values=np.asarray(sliced_data_values),
            sliced_weights_values=None if sliced_weights_values is None else np.asarray(sliced_weights_values),
            independent_vars_current=independent_vars_current,
            remove_nans=remove_nans,
        )

        (
            sliced_data_values,
            sliced_weights_values,
            independent_vars_current,
            outlier_index_bool,
        ) = _apply_outlier_detection(
            model=model,
            sliced_data_values=np.asarray(sliced_data_values),
            sliced_weights_values=None if sliced_weights_values is None else np.asarray(sliced_weights_values),
            independent_vars_current=independent_vars_current,
            request=request,
            context=context,
        )
        if outliers is not None and outlier_index_bool is not None:
            outliers[remaining_ind] = outlier_index_bool

        try:
            if request.settings.override_model_initial_guess:
                pfit = model.make_params()
            else:
                pfit = model.guess(sliced_data_values, **independent_vars_current)
        except (NotImplementedError, TypeError):
            pfit = _make_params(model)
            logger.warning("Error in guess function", exc_info=True)

        weight_dict: dict[str, Any] = {"weights": sliced_weights_values} if sliced_weights_values is not None else {}
        # A bad fit can give least_squares a negative covariance, and lmfit takes its square
        # root. lmfit reports no error bars for such a fit, so only the warnings need hiding.
        with np.errstate(invalid="ignore"):
            result = model.fit(
                sliced_data_values,
                **independent_vars_current,
                params=pfit,
                **request.fitter_options,
                **weight_dict,
            )

        # Fill per-slice outputs.
        fit_result[remaining_ind] = result
        initial_guess[remaining_ind] = result.eval(params=result.init_params, **context.coords)
        fit[remaining_ind] = result.eval(**context.coords)

        if len(context.independent_coord_dims) > 1:
            uncertainty[remaining_ind] = np.zeros(context.independent_coord_dims)
            logger.debug("Uncertainty cannot be evaluated for fits with more than one dimension.")
        else:
            coords_for_uncertainty = {k: v for k, v in context.coords.items() if k not in {"sigma", "dscale"}}
            if len(coords_for_uncertainty) != len(context.coords):
                # eval_uncertainty has keyword-only args named sigma/dscale, so independent variables with those names
                # cannot be passed through **kwargs.
                uncertainty[remaining_ind] = np.zeros(context.independent_coord_dims)
                logger.debug(
                    "Uncertainty cannot be evaluated because an independent variable name conflicts with "
                    "eval_uncertainty kwargs (sigma/dscale)."
                )
            elif any(
                parameter.stderr is None or not np.isfinite(parameter.stderr) or parameter.stderr == 0.0
                for parameter in (result.params[name] for name in result.var_names)
            ):
                # eval_uncertainty divides by each parameter error, and only checks for None.
                uncertainty[remaining_ind] = np.zeros(context.independent_coord_dims)
                logger.debug("Uncertainty cannot be evaluated because the fit gave no parameter errors.")
            else:
                # lmfit stubs don't type this well.
                # lmfit typing stubs are incomplete for eval_uncertainty kwargs.
                uncertainty[remaining_ind] = result.eval_uncertainty(**cast(Any, coords_for_uncertainty))  # type: ignore[call-arg]

        # Build plot definition for this slice.
        plot_args: dict[str, Any] = {
            "coords": context.coords,
            "coords_interpolated": coords_interpolated,
            "independent_coord_names": context.independent_coord_names,
            "independent_vars_current": independent_vars_current,
            "result": result,
            "sliced_data_values": np.asarray(sliced_data_values),
            "y_scaling": float(y_scaling),
            "y_label": y_label,
            "x_scaling": float(x_scaling),
            "x_label": x_label,
        }
        if len(context.independent_coord_dims) == 1:
            plot_args |= {"fit": fit[remaining_ind], "uncertainty": uncertainty[remaining_ind]}
            plotter = _plot_1d_fast
        elif len(context.independent_coord_dims) == 2:
            plotter = _plot_2d_fast
        else:
            plotter = None

        if plotter is not None:
            plot_defs[remaining_ind] = PlotDefinition(wrap_mpl_plot(plotter), plot_args)

            if request.settings.plot_fit:
                for pd in plot_defs.flat:
                    if pd is None:
                        continue
                    pd.function(**pd.default_kwargs)
                    plt.show()

            if request.settings.add_plot_to_dataset and plots is not None:
                for plot_idx, pd in np.ndenumerate(plot_defs):
                    if pd is None:
                        continue
                    fig = pd.function(**pd.default_kwargs)
                    plots[plot_idx] = fig
                    plt.close()

    return ModelFitOutputs(
        fit_result=fit_result,
        fit=fit,
        initial_guess=initial_guess,
        uncertainty=uncertainty,
        plot_definitions=plot_defs,
        plots=plots,
        outliers=outliers,
    )


def _plot_1d_fast(  # noqa: PLR0913
    *args,
    coords: dict[str, np.ndarray],
    coords_interpolated: dict[str, np.ndarray],
    fit: np.ndarray,
    independent_coord_names: list[str],
    independent_vars_current: dict[str, np.ndarray],
    result: Any,
    sliced_data_values: np.ndarray,
    uncertainty: np.ndarray,
    y_scaling: float,
    y_label: str,
    x_scaling: float,
    x_label: str,
    title: str = "",
) -> Figure:
    fig, ax = plt.subplots()
    colorcycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    x_axis = result.userkws[list(result.userkws.keys())[0]]
    x_axis_truncated = list(independent_vars_current.values())[0]
    x_axis_orig = list(coords.values())[0]
    x_axis_interpolated = list(coords_interpolated.values())[0]
    data_interpolated = result.eval(**coords_interpolated)

    ax.plot(
        x_axis * x_scaling,
        result.init_fit * y_scaling,
        "--",
        label="init guess",
        color=colorcycle[1],
        alpha=0.5,
    )
    ax.plot(
        x_axis_truncated * x_scaling,
        sliced_data_values * y_scaling,
        "o",
        color=colorcycle[0],
        alpha=0.7,
        label="data",
    )
    ax.plot(
        x_axis_interpolated * x_scaling,
        data_interpolated * y_scaling,
        color=colorcycle[0],
        label="fit",
    )
    ylims = ax.get_ylim()
    ax.fill_between(
        x_axis_orig * x_scaling,
        (fit - uncertainty) * y_scaling,
        (fit + uncertainty) * y_scaling,
        alpha=0.2,
        color=colorcycle[0],
        label="uncertainty",
    )
    ax.set_ylim(ylims)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.legend()
    if title:
        ax.set_title(title)
    return fig


def _plot_2d_fast(  # noqa: PLR0913
    *args,
    coords: dict[str, np.ndarray],
    coords_interpolated: dict[str, np.ndarray],
    independent_coord_names: list[str],
    independent_vars_current: dict[str, np.ndarray],
    result: Any,
    sliced_data_values: np.ndarray,
    y_scaling: float,
    y_label: str,
    x_scaling: float,
    x_label: str,
    title: str = "",
    **kwargs,
) -> Figure:
    fig, ax = plt.subplots()
    colorcycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    x_axis_interpolated = list(coords_interpolated.values())[0]
    data_interpolated = result.eval(**coords_interpolated)

    x_axis = result.userkws[list(result.userkws.keys())[0]]
    x_axis_truncated = list(independent_vars_current.values())[0]
    y_axis = result.userkws[list(result.userkws.keys())[1]]
    if len(y_axis) > 4:
        try:
            ax.pcolor(x_axis * x_scaling, y_axis, result.eval(**coords).T)
            ax.set_xlabel(x_label)
        except Exception:
            logger.warning("Failed to make a 2d plot.", exc_info=True)
    else:
        for ii in range(len(y_axis)):
            ax.plot(
                x_axis * x_scaling,
                result.init_fit[:, ii] * y_scaling,
                "--",
                color=colorcycle[ii],
                alpha=0.5,
                label="init guess" if ii == 0 else None,
            )
            ax.plot(
                x_axis_truncated * x_scaling,
                sliced_data_values[:, ii] * y_scaling,
                "o",
                color=colorcycle[ii],
                alpha=0.7,
                label="data" if ii == 0 else None,
            )
            ax.plot(
                x_axis_interpolated * x_scaling,
                data_interpolated[:, ii] * y_scaling,
                "-",
                color=colorcycle[ii],
                label="fit" if ii == 0 else None,
            )
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
        ax.legend()
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig
