# Copyright 2019-2026 IQM Finland Oy
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility function for figures."""

from datetime import datetime
from typing import Any

from matplotlib.axes import Axes
import numpy as np
import xarray as xr

from exa.common.data import DataType
from exa.core.plotting import plot_helper
from exa.core.util.formatting import scientific_rounding
from iqm.station_control.interface.models import RunDefinition

from .figure import Figure
from .plot_options import FigureOptions, FitAnnotationOptions, LineOptions
from .plot_styles import LabelFormatter, LineStyle, MarkerFillStyle, MarkerStyle, PlotStyle


def build_results_figure(
    data: xr.DataArray,
    variables: list[str],
    fits: list[xr.DataArray] | None,
    title: str,
    figure_options: FigureOptions,
) -> list[Figure]:
    """Figure with multiline plots of observed/fits vs one or two variables."""
    legend = figure_options.plot_options.legend  # type: ignore[union-attr]
    figure_options = figure_options.copy(plot_options=figure_options.plot_options.copy(legend=legend))  # type: ignore[union-attr]
    parameter = data.attrs.get("parameter")
    data_type = parameter.data_type if parameter is not None else DataType.FLOAT
    if fits is None:
        fits = []
    figure, axs = _create_figure(data_type=data_type, title=title, figure_options=figure_options)
    figures = [figure]
    args = {
        "axs": axs,
        "data_type": data_type,
        "plot_options": figure_options.plot_options,
        "complex_options": figure_options.complex_options,
    }
    if figure_options.is_colormap or figure_options.is_contour:
        plot_function = plot_helper.plot_colormap if figure_options.is_colormap else plot_helper.plot_contour
        plot_function(data=data, variables=variables, **args)  # type: ignore[arg-type]  # type: ignore[arg-type]  # type: ignore[arg-type]  # type: ignore[arg-type]  # type: ignore[arg-type]
        for fit in fits:
            fit_figure, axs = _create_figure(data_type=data_type, title=title, figure_options=figure_options)
            figures.append(fit_figure)
            plot_function(data=fit, variables=variables, **args)  # type: ignore[arg-type]  # type: ignore[arg-type]  # type: ignore[arg-type]  # type: ignore[arg-type]  # type: ignore[arg-type]
        _draw_fit_annotations(figure, fits, figure_options.fit_annotations)  # type: ignore[arg-type]
    else:
        for fit in fits:
            if len(variables) == 1:
                plot_helper.plot_line(data=fit, variable=variables[0], **args)  # type: ignore[arg-type]  # type: ignore[arg-type]  # type: ignore[arg-type]  # type: ignore[arg-type]  # type: ignore[arg-type]
            else:
                plot_helper.plot_multiline(data=fit, variables=variables, **args)  # type: ignore[arg-type]  # type: ignore[arg-type]  # type: ignore[arg-type]  # type: ignore[arg-type]  # type: ignore[arg-type]
        for ax in figure.axs:
            _reset_color_cycle(ax[0], figure_options.style)

        lines_or_markers = figure_options.plot_options.line_options.for_data_points(len(fits) > 0)  # type: ignore[union-attr]  # type: ignore[union-attr]
        args["plot_options"] = figure_options.plot_options.copy(line_options=lines_or_markers)  # type: ignore[union-attr]

        if len(variables) == 1:
            plot_helper.plot_line(data=data, variable=variables[0], **args)  # type: ignore[arg-type]  # type: ignore[arg-type]  # type: ignore[arg-type]  # type: ignore[arg-type]  # type: ignore[arg-type]
        else:
            plot_helper.plot_multiline(data=data, variables=variables, **args)  # type: ignore[arg-type]  # type: ignore[arg-type]  # type: ignore[arg-type]  # type: ignore[arg-type]  # type: ignore[arg-type]
        _draw_fit_annotations(figure, fits, figure_options.fit_annotations)  # type: ignore[arg-type]
    return figures


def build_model_parameter_figure(
    fits: list[xr.DataArray],
    model_parameter: str,
    variables: list[str] | None = None,
    title: str | None = None,
    figure_options: FigureOptions | None = None,
) -> Figure:
    """Build a figure with model parameters plotted against fits data. This function supports up to 3D fits. With 1D fits,
    :func:`~exa.core.plotting.plot_helper.plot_bar` is used. Otherwise,
    :func:`~exa.core.plotting.plot_helper.plot_multiline` is used.

    Args:
        fits:
            List of fits. Each fit has to have same dimensions and coordinates.
        model_parameter:
            The key of the fit parameter to plot in each `fit`'s fit_parameters dictionary.
        variables:
            A list of strings corresponding to the variables to plot against. Number of variables must be N - 1, where
            N is the number of `fits` dimensions. Has no effect if N = 1. If None is given and N > 1, the model
            parameters are plotted against the stacked dimension variables in `fits`.
        title:
            Figure title. If None is given, one will be automatically created using `fits` and `model_parameter`.
        figure_options:
            Figure options to use. If None is given, a new instance of :class:`.FigureOptions` is created.

    Returns:
            A :class:`.Figure` showing a plot of the data provided.

    """  # noqa: E501
    if len(fits) == 0:
        raise ValueError("Length of fits cannot be 0")

    if figure_options is None:
        figure_options = FigureOptions()

    n_dims = len(fits[0].sizes)
    if n_dims == 1:
        return _build_model_parameter_vs_no_variable_figure(
            fits=fits,
            model_parameter=model_parameter,
            title=title,
            figure_options=figure_options,
        )

    if variables is None:
        variables = fits[0].attrs.get("stacked_dimensions")
        if variables is None:
            raise ValueError("Unable to extract stacked_dimensions from fits")

    if len(variables) != n_dims - 1:
        raise ValueError(
            f"Number of variables ({n_dims}) for the fit should be N - 1,where N is the number of data dimensions"
        )

    if n_dims == 2:
        return _build_model_parameter_vs_one_variable_figure(
            fits=fits,
            variable=variables[0],
            model_parameter=model_parameter,
            title=title,
            figure_options=figure_options,
        )
    if n_dims == 3:
        return _build_model_parameter_vs_two_variables_figure(
            fits=fits,
            variable1=variables[0],
            variable2=variables[1],
            model_parameter=model_parameter,
            title=title,
            figure_options=figure_options,
        )
    msg = "Building a model parameter figure with fits dimensions > 3 is not supported"
    raise NotImplementedError(msg)


def _build_model_parameter_vs_no_variable_figure(
    fits: list[xr.DataArray],
    model_parameter: str,
    title: str | None,
    figure_options: FigureOptions,
) -> Figure:
    """Figure with bar chart of model parameter vs fits."""
    if len(fits) == 0:
        raise ValueError("Length of fits cannot be 0")
    fig = Figure(1, 1, title, figure_options)
    plot_options = figure_options.plot_options
    plot_options = plot_options.copy(x=plot_options.x.copy(label_formatter=LabelFormatter.STRING))  # type: ignore[union-attr]  # type: ignore[union-attr]  # type: ignore[union-attr]
    figure_options = figure_options.copy(plot_options=plot_options)
    categories = []
    values = []
    for data_array in fits:
        categories.append(f"P{str(data_array.name).rsplit('::', maxsplit=1)[-1]}")
        fit_parameter = data_array.attrs.get("fit_parameters")
        if fit_parameter is None:
            raise ValueError("Unable to fetch fit_parameters from fit")
        values.append(fit_parameter.get(()).get(model_parameter))
    data_array = xr.DataArray(
        data=values,
        coords={"category": categories},
        dims=["category"],
        attrs={
            "standard_name": model_parameter,
            "long_name": model_parameter,
        },
    )
    data_array["category"].attrs = {
        "standard_name": "fit",
        "long_name": "fit",
    }

    plot_helper.plot_bar(
        axs=[fig.axs[0][0]],
        data=data_array,
        variable="category",
        plot_options=figure_options.plot_options,
    )
    return fig


def _build_model_parameter_vs_one_variable_figure(
    fits: list[xr.DataArray],
    variable: str,
    model_parameter: str,
    title: str | None,
    figure_options: FigureOptions,
) -> Figure:
    """Figure with multiline plot of model parameter vs variable and fits."""
    if len(fits) == 0:
        raise ValueError("Length of fits cannot be 0")
    fig = Figure(1, 1, title, figure_options)
    fit_x_points = fits[0].coords[variable].values
    lines = None
    fits_returned = []
    for fit in fits:
        fits_returned.append(f"fit {str(fit.name).rsplit('::', maxsplit=1)[-1]}")
        fit_parameters = fit.attrs.get("fit_parameters")
        if fit_parameters is None:
            raise ValueError("Unable to fetch fit_parameters from fit")
        values = [[fit_parameters.get((value,)).get(model_parameter)] for value in fit_x_points]
        lines = np.asarray(values) if lines is None else np.column_stack((lines, np.asarray(values)))
    data_array = xr.DataArray(
        data=lines,
        coords={variable: fit_x_points, "fit": fits_returned},
        dims=[variable, "fit"],
        attrs={
            "standard_name": model_parameter,
            "long_name": model_parameter,
        },
    )
    scatter_line_options = LineOptions(
        line_style=LineStyle.SOLID,
        marker_style=MarkerStyle.CIRCLE,
        marker_fill_style=MarkerFillStyle.NONE,
    )
    plot_options = figure_options.plot_options.copy()  # type: ignore[union-attr]
    plot_options = plot_options.copy(
        legend=plot_options.legend.copy(legend_format=""),  # type: ignore[union-attr]
        line_options=scatter_line_options,
    )
    plot_helper.plot_multiline(
        axs=[fig.axs[0][0]],
        data=data_array,
        variables=[variable, "fit"],
        plot_options=plot_options,
    )

    return fig


def _build_model_parameter_vs_two_variables_figure(
    fits: list[xr.DataArray],
    variable1: str,
    variable2: str,
    model_parameter: str,
    title: str | None,
    figure_options: FigureOptions,
) -> Figure:
    """Figure with multiline plot of model parameter vs two variables. Each fit creates a subplot on a different row."""
    if len(fits) == 0:
        raise ValueError("Length of fits cannot be 0")
    fig = Figure(len(fits), 1, title, figure_options)
    fit_x_points = fits[0].coords[variable1].values
    fit_y_points = fits[0].coords[variable2].values
    for i, fit in enumerate(fits):
        fit_parameters = fit.attrs.get("fit_parameters")
        if fit_parameters is None:
            raise ValueError("Unable to fetch fit_parameters from fit")
        lines = []
        for x in fit_x_points:
            fit_parameters_row = [
                fit_parameters.get(
                    (
                        x,
                        y,
                    )
                ).get(model_parameter)
                for y in fit_y_points
            ]
            lines.append(fit_parameters_row)

        data_array = xr.DataArray(
            data=lines,
            coords={variable1: fit_x_points, variable2: fit_y_points},
            dims=[variable1, variable2],
            attrs={
                "standard_name": model_parameter,
                "long_name": model_parameter,
            },
        )
        scatter_line_options = LineOptions(
            line_style=LineStyle.SOLID,
            marker_style=MarkerStyle.CIRCLE,
            marker_fill_style=MarkerFillStyle.NONE,
        )
        plot_options = figure_options.plot_options.copy()  # type: ignore[union-attr]
        plot_options = plot_options.copy(
            legend=plot_options.legend.copy(legend_format=""),  # type: ignore[union-attr]
            line_options=scatter_line_options,
        )
        plot_helper.plot_multiline(
            axs=[fig.axs[i][0]],
            data=data_array,
            variables=[variable1, variable2],
            extra_title_text=f"fit {str(fit.name).rsplit('::', maxsplit=1)[-1]}",
            plot_options=plot_options,
        )

    return fig


def _reset_color_cycle(axes: Axes, plot_style: PlotStyle):  # noqa: ANN202
    axes.set_prop_cycle(color=plot_style.colors())


def _draw_fit_annotations(fig: Figure, fits: list[xr.DataArray] | None, options: FitAnnotationOptions) -> None:
    if not fits:
        return
    if not options.visible or len(fits) == 0:
        return
    text = _generate_fits_texts(fits, options.params_to_show)
    if options.title:
        text = f"{options.title}\n{text}"

    axis = fig.figure.axes[0]
    existing_legend = axis.get_legend()
    if existing_legend:
        # Get existing legend position as data coordinates and use them to position the new legend under the old one.
        transf = axis.transAxes.inverted()
        bbox = existing_legend.get_window_extent(renderer=fig.figure.canvas.get_renderer())  # type: ignore[attr-defined]
        bb_datacoords = bbox.transformed(transf)
        bbox_to_anchor = (bb_datacoords.xmin, bb_datacoords.ymin)
    else:
        bbox_to_anchor = plot_helper.LEGEND_BBOX

    props = {"boxstyle": "square", "facecolor": "None", "edgecolor": "lightgray"}
    axis.text(*bbox_to_anchor, text, transform=axis.transAxes, verticalalignment="top", bbox=props)


def _generate_fits_texts(fits: list[xr.DataArray], params_to_show: list[str] | None) -> str:
    max_len = 40
    lines = []
    for fit in fits:
        name = fit.attrs.get("long_name", fit.attrs.get("name", ""))
        lines.append(f"{name}:")
        for param_set_key, param_set in fit.attrs.get("fit_parameters", {}).items():
            keys = set(param_set.keys()).intersection(params_to_show) if params_to_show else param_set.keys()
            if len(keys) > 0:
                if param_set_key:
                    lines.append(f"\n{_format_parameter_value(param_set_key)}:")
                for key in sorted(keys):
                    if key.endswith("_stddev"):
                        continue
                    uncertainty = param_set.get(f"{key}_stddev")
                    if uncertainty is not None:
                        lines.append(f"{key}: {scientific_rounding(param_set[key], uncertainty)}")
                    else:
                        lines.append(f"{key}: {_format_parameter_value(param_set[key])}")
                    if len(lines) >= max_len:
                        lines[-1] = "..."
                        return "\n".join(lines)
    return "\n".join(lines)


def _format_parameter_value(value: Any):  # noqa: ANN202
    if isinstance(value, float):
        return str(float(f"{value:.8g}"))
    return value


def generate_figure_title(run_definition: RunDefinition) -> str:
    """Generate a canonical title for a figure from a run.

    Args:
        run_definition: Metadata for the run.

    """
    if len(run_definition.components) < 5:
        components_text = ", ".join(run_definition.components)
    else:
        components_text = f"{len(run_definition.components)} components"
    name = run_definition.experiment_label or run_definition.experiment_name
    return f"{run_definition.sweep_definition.dut_label}; {components_text}; {name}; {str(run_definition.run_id)}"


def generate_figure_filename(
    run_definition: RunDefinition,
    prefix: str | None = None,
    suffix: str | None = None,
    *,
    timestamp: datetime | None = None,
) -> str:
    """Generate a canonical filename for a figure from a run.

    The return value will not contain any file type extension.

    Args:
        run_definition: Data related to the run.
        prefix: If specified, will be used as a replacement for experiment name.
        suffix: If specified, will be added as suffix for the filename.
        timestamp: Timestamp (normally execution begin) to include in the filename.

    """
    prefix = run_definition.experiment_name if prefix is None else prefix
    run_id = run_definition.run_id
    timestamp_str = timestamp.strftime("%Y_%m_%d_%H_%M_%S") if timestamp else "unknown"
    filename = f"{prefix}_{timestamp_str}_{run_id}"
    if suffix:
        filename += f"_{suffix}"
    return filename.replace("-", "_")


def _create_figure(data_type: DataType, title: str, figure_options: FigureOptions) -> tuple[Figure, list[Axes]]:
    """Create figure and list of x-axes needed for plotting functions."""
    if data_type == DataType.COMPLEX:
        fig = Figure(2, 1, title, figure_options)
        axs = [fig.axs[0][0], fig.axs[1][0]]
    else:
        fig = Figure(1, 1, title, figure_options)
        axs = [fig.axs[0][0]]
    return fig, axs
