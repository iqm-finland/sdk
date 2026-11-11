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
"""Helper functions for plotting."""

from textwrap import wrap

from matplotlib.axes import Axes
from matplotlib.axis import Axis
import numpy as np
import xarray as xr

from exa.common.data import DataType, Parameter

from .plot_options import AxisOptions, ComplexOptions, PlotOptions

LEGEND_BBOX = (1.01, 1.0)


def plot_line(
    axs: list[Axes],
    data: xr.DataArray,
    variable: str,
    data_type: DataType = DataType.FLOAT,
    title: str | None = None,
    extra_title_text: str | None = None,
    plot_options: PlotOptions | None = None,
    complex_options: ComplexOptions | None = None,
) -> None:
    """Create line plots for complex and real data.

    Args:
        axs: Plot axes.
        data: Y-axis data.
        variable: Key to x-axis data.
        data_type: Type of data (complex or number)
        title: Plot title. If None is given, the title will be automatically generated from `data` and `variables`.
        extra_title_text: Optional extra string appended after `title` and separated by '; '.
        plot_options: Plot options to use for the appearance of the plot.
            If None give a new instance of :class:`~exa.core.plotting.plot_options.PlotOptions` is created
        complex_options: Options to change the appearance for figures with complex data.
            If None give a new instance of :class:`~exa.core.plotting.plot_options.ComplexOptions` is created

    Notes:
        Function with a side effect - modifies axes.

    """
    if plot_options is None:
        plot_options = PlotOptions()
    if data_type == DataType.COMPLEX:
        if complex_options is None:
            complex_options = ComplexOptions()
        data1, data2 = split_complex_data(data, complex_options)
        _plot_real_line(
            ax=axs[0],
            data=data1,
            variable=variable,
            title=title,
            extra_title_text=extra_title_text,
            plot_options=plot_options,
        )
        _plot_real_line(
            ax=axs[1],
            data=data2,
            variable=variable,
            title=title,
            extra_title_text=extra_title_text,
            plot_options=plot_options,
        )
    else:
        _plot_real_line(
            ax=axs[0],
            data=data,
            variable=variable,
            title=title,
            extra_title_text=extra_title_text,
            plot_options=plot_options,
        )


def plot_multiline(
    axs: list[Axes],
    data: xr.DataArray,
    variables: list[str],
    data_type: DataType = DataType.FLOAT,
    title: str | None = None,
    extra_title_text: str | None = None,
    plot_options: PlotOptions | None = None,
    complex_options: ComplexOptions | None = None,
) -> None:
    """Create multiline plots of real or complex data.

    Args:
        axs: Plot axes.
        data: Z-axis data.
        variables: Keys to x-axis and y-axis data.
        data_type: Type of data (complex or number)
        title: Plot title. If None is given, the title will be automatically generated from `data` and `variables`.
        extra_title_text: Optional extra string appended after `title` and separated by '; '.
        plot_options: Plot options to use for the appearance of the plot.
            If None give a new instance of :class:`~exa.core.plotting.plot_options.PlotOptions` is created
        complex_options: Options to change the appearance for figures with complex data.
            If None give a new instance of :class:`~exa.core.plotting.plot_options.ComplexOptions` is created

    Notes:
        Function with a side effect - modifies axes.

    """
    if plot_options is None:
        plot_options = PlotOptions()
    if data_type == DataType.COMPLEX:
        if complex_options is None:
            complex_options = ComplexOptions()
        data1, data2 = split_complex_data(data, complex_options)
        phase_options = plot_options.copy(legend=plot_options.legend.copy(visible=False))  # type: ignore[union-attr]
        _plot_real_multiline(
            ax=axs[0],
            data=data1,
            variables=variables,
            title=title,
            extra_title_text=extra_title_text,
            plot_options=plot_options,
        )
        _plot_real_multiline(
            ax=axs[1],
            data=data2,
            variables=variables,
            title=title,
            extra_title_text=extra_title_text,
            plot_options=phase_options,
        )
    else:
        _plot_real_multiline(
            ax=axs[0],
            data=data,
            variables=variables,
            title=title,
            extra_title_text=extra_title_text,
            plot_options=plot_options,
        )


def plot_colormap(
    axs: list[Axes],
    data: xr.DataArray,
    variables: list[str],
    data_type: DataType = DataType.FLOAT,
    title: str | None = None,
    extra_title_text: str | None = None,
    plot_options: PlotOptions | None = None,
    complex_options: ComplexOptions | None = None,
    **colormap_kwargs,
) -> None:
    """Create colormap plots for complex and real data.

    Args:
        axs: Plot axes.
        data: Z-axis data.
        variables: Keys to x-axis and y-axis data.
        data_type: Type of data (complex or number).
        title: Plot title. If None is given, the title will be automatically generated from `data` and `variables`.
        extra_title_text: Optional extra string appended after `title` and separated by '; '.
        plot_options: Plot options to use for the appearance of the plot.
            If None give a new instance of :class:`~exa.core.plotting.plot_options.PlotOptions` is created
        complex_options: Options to change the appearance for figures with complex data.
            If None give a new instance of :class:`~exa.core.plotting.plot_options.ComplexOptions` is created
        **colormap_kwargs: extra arguments for :func:`matplotlib.pyplot.pcolormesh`

    Notes:
        Function with a side effect - modifies axes.

    """
    if plot_options is None:
        plot_options = PlotOptions()
    if data_type == DataType.COMPLEX:
        if complex_options is None:
            complex_options = ComplexOptions()
        data1, data2 = split_complex_data(data, complex_options)
        _plot_real_colormap(
            ax=axs[0],
            data=data1,
            variables=variables,
            title=title,
            extra_title_text=extra_title_text,
            plot_options=plot_options,
            **colormap_kwargs,
        )
        _plot_real_colormap(
            ax=axs[1],
            data=data2,
            variables=variables,
            title=title,
            extra_title_text=extra_title_text,
            plot_options=plot_options,
            **colormap_kwargs,
        )
    else:
        _plot_real_colormap(
            ax=axs[0],
            data=data,
            variables=variables,
            title=title,
            extra_title_text=extra_title_text,
            plot_options=plot_options,
            **colormap_kwargs,
        )


def plot_contour(
    axs: list[Axes],
    data: xr.DataArray,
    variables: list[str],
    data_type: DataType = DataType.FLOAT,
    title: str | None = None,
    extra_title_text: str | None = None,
    plot_options: PlotOptions | None = None,
    complex_options: ComplexOptions | None = None,
    **contour_kwargs,
) -> None:
    """Create contour plot for complex and real data.

    Args:
        axs: Plot axes.
        data: Z-axis data.
        variables: Keys to x-axis and y-axis data.
        data_type: Type of data (complex or number).
        title: Plot title. If None is given, the title will be automatically generated from `data` and `variables`.
        extra_title_text: Optional extra string appended after `title` and separated by '; '.
        plot_options: Plot options to use for the appearance of the plot.
            If None give a new instance of :class:`~exa.core.plotting.plot_options.PlotOptions` is created
        complex_options: Options to change the appearance for figures with complex data.
            If None give a new instance of :class:`~exa.core.plotting.plot_options.ComplexOptions` is created
        **contour_kwargs: extra arguments for :func:`matplotlib.pyplot.contourf`

    Notes:
        Function with a side effect - modifies axes.

    """
    if plot_options is None:
        plot_options = PlotOptions()
    if data_type == DataType.COMPLEX:
        if complex_options is None:
            complex_options = ComplexOptions()
        data1, data2 = split_complex_data(data, complex_options)
        _plot_real_contour(
            ax=axs[0],
            data=data1,
            variables=variables,
            title=title,
            extra_title_text=extra_title_text,
            plot_options=plot_options,
            **contour_kwargs,
        )
        _plot_real_contour(
            ax=axs[1],
            data=data2,
            variables=variables,
            title=title,
            extra_title_text=extra_title_text,
            plot_options=plot_options,
            **contour_kwargs,
        )
    else:
        _plot_real_contour(
            ax=axs[0],
            data=data,
            variables=variables,
            title=title,
            extra_title_text=extra_title_text,
            plot_options=plot_options,
            **contour_kwargs,
        )


def plot_bar(
    axs: list[Axes],
    data: xr.DataArray,
    variable: str,
    data_type: DataType = DataType.FLOAT,
    title: str | None = None,
    extra_title_text: str | None = None,
    plot_options: PlotOptions | None = None,
    complex_options: ComplexOptions | None = None,
) -> None:
    """Create bar plot of complex or real data.

    Args:
        axs: Plot axes.
        data: Y-axis data.
        variable: Key to x-axis data.
        data_type: Type of data (complex or number).
        title: Plot title. If None is given, the title will be automatically generated from `data` and `variables`.
        extra_title_text: Optional extra string appended after `title` and separated by '; '.
        plot_options: Plot options to use for the appearance of the plot.
            If None give a new instance of :class:`~exa.core.plotting.plot_options.PlotOptions` is created
        complex_options: Options to change the appearance for figures with complex data.
            If None give a new instance of :class:`~exa.core.plotting.plot_options.ComplexOptions` is created

    Notes:
        Function with a side effect - modifies axes.

    """
    if plot_options is None:
        plot_options = PlotOptions()
    if data_type == DataType.COMPLEX:
        if complex_options is None:
            complex_options = ComplexOptions()
        data1, data2 = split_complex_data(data, complex_options)
        _plot_real_bar(
            ax=axs[0],
            data=data1,
            variable=variable,
            title=title,
            extra_title_text=extra_title_text,
            plot_options=plot_options,
        )
        _plot_real_bar(
            ax=axs[1],
            data=data2,
            variable=variable,
            title=title,
            extra_title_text=extra_title_text,
            plot_options=plot_options,
        )
    else:
        _plot_real_bar(
            ax=axs[0],
            data=data,
            variable=variable,
            title=title,
            extra_title_text=extra_title_text,
            plot_options=plot_options,
        )


def _plot_real_line(
    ax: Axes,
    data: xr.DataArray,
    variable: str,
    title: str | None,
    extra_title_text: str | None,
    plot_options: PlotOptions,
) -> None:
    """Create a line plot for real data.

    Args:
        ax: Matplotlib Axes instance.
        data: Y-axis data.
        variable: Key to x-axis data.
        title: Plot title. If None is given, the title will be automatically generated from `data` and `variables`.
        extra_title_text: extra string appended after `title` and separated by '; '.
        plot_options: Plot options to use for the appearance of the plot.
            If None give a new instance of :class:`~exa.core.plotting.plot_options.PlotOptions` is created

    Notes:
        Function with a side effect - modifies axes.

    """
    line_options = plot_options.line_options
    if title is None:
        title = _build_title(data, [variable])
    if extra_title_text is not None:
        title += f"; {extra_title_text}"
    data.plot.line(
        ax=ax,
        x=variable,
        linestyle=line_options.line_style.value,  # type: ignore[union-attr]
        linewidth=line_options.line_width,  # type: ignore[union-attr]
        marker=line_options.marker_style.value,  # type: ignore[union-attr]
        markeredgewidth=line_options.line_width,  # type: ignore[union-attr]
        markersize=line_options.marker_size,  # type: ignore[union-attr]
        color=line_options.color,  # type: ignore[union-attr]
        alpha=line_options.alpha,  # type: ignore[union-attr]
        fillstyle=line_options.marker_fill_style.value,  # type: ignore[union-attr]
    )
    _configure_axes(ax, title, plot_options)


def _plot_real_multiline(
    ax: Axes,
    data: xr.DataArray,
    variables: list[str],
    title: str | None,
    extra_title_text: str | None,
    plot_options: PlotOptions,
) -> None:
    """Create a multiline plot for real data.

    Args:
        ax: Matplotlib Axes instance.
        data: Z-axis data.
        variables: Keys to x-axis and y-axis data.
        title: Plot title. If None is given, the title will be automatically generated from `data` and `variables`.
        extra_title_text: extra string appended after `title` and separated by '; '.
        plot_options: Plot options to use for the appearance of the plot.
            If None give a new instance of :class:`~exa.core.plotting.plot_options.PlotOptions` is created

    Raises:
        IndexError: Not enough variables were provided (needs at least 2)

    Notes:
        Function with a side effect - modifies axes.

    """
    line_options = plot_options.line_options
    if len(variables) < 2:
        raise IndexError("Not enough variables provided")
    if title is None:
        title = _build_title(data, variables)
    if extra_title_text is not None:
        title += f"; {extra_title_text}"
    long_name = data.attrs.get("long_name", data.name)
    units = data.attrs.get("units", None)
    label = None if "fit" in str(data.name) else f"{long_name} ({units})"
    stacked_data = _stack(data, plot_options)
    stacked_data.plot.line(
        ax=ax,
        x=variables[0],
        hue=variables[1],
        add_legend=False,
        alpha=line_options.alpha,  # type: ignore[union-attr]
        markersize=line_options.marker_size,  # type: ignore[union-attr]
        linestyle=line_options.line_style.value,  # type: ignore[union-attr]
        linewidth=line_options.line_width,  # type: ignore[union-attr]
        markeredgewidth=line_options.line_width,  # type: ignore[union-attr]
        marker=line_options.marker_style.value,  # type: ignore[union-attr]
        fillstyle=line_options.marker_fill_style.value,  # type: ignore[union-attr]
        label=label,
    )
    _configure_axes(ax, title, plot_options)
    if plot_options.legend.visible and "fit" not in str(data.name):  # type: ignore[union-attr]
        # By convention the last variable provided specifies legend metadata such as label name and units
        last_variable = data[variables[-1]]
        if last_variable is not None:
            long_name = last_variable.attrs.get("long_name", last_variable.name)
            units = last_variable.attrs.get("units", None)
        legend = ax.legend(
            bbox_to_anchor=LEGEND_BBOX,
            loc="upper left",
            borderaxespad=0.0,
            title=long_name,
        )
        for index, text in enumerate(legend.texts):
            value = data.coords[variables[-1]].values[index]
            if units:
                legend_format = "{" + plot_options.legend.legend_format + "} {}"  # type: ignore[operator,union-attr]
                text.set_text(legend_format.format(value, units))
            else:
                legend_format = "{" + plot_options.legend.legend_format + "}"  # type: ignore[operator,union-attr]
                text.set_text(legend_format.format(value))


def _plot_real_colormap(
    ax: Axes,
    data: xr.DataArray,
    variables: list[str],
    title: str | None,
    extra_title_text: str | None,
    plot_options: PlotOptions,
    **colormap_kwargs,
) -> None:
    """Create a colorbar plot for real data.

    Args:
        ax: Matplotlib Axes instance.
        data: Z-axis data.
        variables: Keys to x-axis and y-axis data.
        title: Plot title. If None is given, the title will be automatically generated from `data` and `variables`.
        extra_title_text: extra string appended after `title` and separated by '; '.
        plot_options: Plot options to use for the appearance of the plot.
            If None give a new instance of :class:`~exa.core.plotting.plot_options.PlotOptions` is created
        **colormap_kwargs: extra arguments for :func:`matplotlib.pyplot.pcolormesh`

    Notes:
        Function with a side effect - modifies axes.

    """
    if title is None:
        title = _build_title(data, variables)
    if extra_title_text is not None:
        title += f"; {extra_title_text}"
    args = _colorbar_args(data_parameter=data.attrs.get("parameter", ""), plot_options=plot_options)
    data.plot.pcolormesh(
        ax=ax,
        x=variables[0],
        y=variables[1],
        add_colorbar=True,
        cmap=plot_options.colorbar.colormap.get(),  # type: ignore[union-attr]
        cbar_kwargs=args,
        **colormap_kwargs,
    )
    _configure_axes(ax, title, plot_options)


def _plot_real_contour(
    ax: Axes,
    data: xr.DataArray,
    variables: list[str],
    title: str | None,
    extra_title_text: str | None,
    plot_options: PlotOptions,
    **contour_kwargs,
) -> None:
    """Create a contour plot for real data.

    Args:
        ax: Matplotlib Axes instance.
        data: Z-axis data.
        variables: Keys to x-axis and y-axis data.
        title: Plot title. If None is given, the title will be automatically generated from `data` and `variables`.
        extra_title_text: extra string appended after `title` and separated by '; '.
        plot_options:
            Plot options to use for the appearance of the plot.
            If None give a new instance of :class:`~exa.core.plotting.plot_options.PlotOptions` is created
        **contour_kwargs: extra arguments for :func:`matplotlib.pyplot.contourf`

    Notes:
        - Function with a side effect - modifies axes.

    """
    if title is None:
        title = _build_title(data, variables)
    if extra_title_text is not None:
        title += f"; {extra_title_text}"
    args = _colorbar_args(data_parameter=data.attrs.get("parameter", ""), plot_options=plot_options)
    data.plot.contourf(
        ax=ax,
        x=variables[0],
        y=variables[1],
        add_colorbar=True,
        cmap=plot_options.colorbar.colormap.get(),  # type: ignore[union-attr]
        cbar_kwargs=args,
        **contour_kwargs,
    )
    _configure_axes(ax, title, plot_options)


def _plot_real_bar(
    ax: Axes,
    data: xr.DataArray,
    variable: str,
    title: str | None,
    extra_title_text: str | None,
    plot_options: PlotOptions,
) -> None:
    """Create a bar plot for real data.

    Args:
        ax: Matplotlib Axes instance.
        data: Y-axis data.
        variable: Key to x-axis data.
        title: Plot title. If None is given, the title will be automatically generated from `data` and `variables`.
        extra_title_text: extra string appended after `title` and separated by '; '.
        plot_options:
            Plot options to use for the appearance of the plot.
            If None give a new instance of :class:`~exa.core.plotting.plot_options.PlotOptions` is created

    Notes:
        Function with a side effect - modifies axes.

    """
    if title is None:
        title = _build_title(data, [variable])
    if extra_title_text is not None:
        title += f"; {extra_title_text}"
    ax.bar(x=data.coords[variable].values, height=data.values)
    ax.set_xlabel(data.coords[variable].attrs.get("long_name"))
    ax.set_ylabel(data.attrs.get("long_name", ""))
    ax.set_xticks([i for i, _ in enumerate(data.coords[variable].values)])
    ax.set_xticklabels(data.coords[variable].values)
    _configure_axes(ax, title, plot_options)


def _colorbar_args(
    data_parameter: xr.DataArray,
    plot_options: PlotOptions,
) -> dict:
    colorbar = plot_options.colorbar
    return {
        "label": f"{data_parameter.label} ({data_parameter.unit})".removesuffix(" ()")
        if isinstance(data_parameter, Parameter)
        else "",
        "format": colorbar.label_formatter.formatter(  # type: ignore[union-attr]
            decimal_places=colorbar.decimal_places,  # type: ignore[union-attr]
            percent_scale=colorbar.percent_scale,  # type: ignore[union-attr]
            label_format=colorbar.label_format,  # type: ignore[union-attr]
        ),
    }


def _configure_axes(ax: Axes, title: str, plot_options: PlotOptions) -> None:
    """Configure axes.

    Note: Function with a side effect - modifies axes.

    Args:
        ax: Matplotlib Axes instance.
        title: Plot title.
        plot_options: Plot options.

    """
    if title is not None:
        ax.set_title("\n".join(wrap(title, plot_options.title_wrap)))
    _configure_axis(ax.xaxis, plot_options.x)  # type: ignore[arg-type]
    _configure_axis(ax.yaxis, plot_options.y)  # type: ignore[arg-type]
    if plot_options.x.lower_limit:  # type: ignore[union-attr]
        ax.set_xlim(plot_options.x.lower_limit, None)  # type: ignore[union-attr]
    if plot_options.x.upper_limit:  # type: ignore[union-attr]
        ax.set_xlim(None, plot_options.x.upper_limit)  # type: ignore[union-attr]
    if plot_options.y.lower_limit:  # type: ignore[union-attr]
        ax.set_ylim(plot_options.y.lower_limit, None)  # type: ignore[union-attr]
    if plot_options.y.upper_limit:  # type: ignore[union-attr]
        ax.set_ylim(None, plot_options.y.upper_limit)  # type: ignore[union-attr]
    ax.grid(visible=plot_options.x.use_grid, axis="x")  # type: ignore[union-attr]
    ax.grid(visible=plot_options.y.use_grid, axis="y")  # type: ignore[union-attr]
    if plot_options.x.log_scale:  # type: ignore[union-attr]
        ax.set_xscale("log")
    if plot_options.y.log_scale:  # type: ignore[union-attr]
        ax.set_yscale("log")
    if not plot_options.legend.visible:  # type: ignore[union-attr]
        legend = ax.get_legend()
        if legend:
            legend.remove()


def _configure_axis(axis: Axis, axis_options: AxisOptions = AxisOptions()) -> None:
    """Configure axis.

    Note: Function with a side effect - modifies axis.

    Args:
        axis: Matplotlib Axis instance.
        axis_options: Axis options.

    """
    axis.set_major_formatter(
        axis_options.label_formatter.formatter(
            decimal_places=axis_options.decimal_places,
            percent_scale=axis_options.percent_scale,
            label_format=axis_options.label_format,
        )
    )
    if axis_options.tick_count is not None:
        axis.set_major_locator(axis_options.tick_locator.locator(tick_count=axis_options.tick_count))


def split_complex_data(
    data: xr.DataArray, complex_options: ComplexOptions | None = None
) -> tuple[xr.DataArray, xr.DataArray]:
    """Convert dataset results from complex to real/imaginary.

    Args:
        data: Complex data set.
        complex_options: Options to change how the complex data is displayed. If None given, an new instance of
            :class:`~exa.core.plotting.plot_options.ComplexOptions` is created.

    """
    if complex_options is None:
        complex_options = ComplexOptions()
    if complex_options.use_magnitude_phase:
        suffix = ("magnitude", "phase")
        values = np.angle(data.values)
        if complex_options.unwrap:
            values = np.unwrap(values)
        if complex_options.in_degrees:
            values = np.rad2deg(values)
        converted_data = (np.abs(data.values), values)
    else:
        suffix = ("real", "imaginary")
        converted_data = (np.real(data.values), np.imag(data.values))
    parameter = data.attrs.get("parameter")
    if parameter is None:
        raise ValueError("The data array does not contain parameter as attribute")
    label = f"{parameter.label} fit" if data.attrs.get("target") else f"{parameter.label}"
    if parameter.data_type is not DataType.COMPLEX and not (
        isinstance(parameter.data_type, tuple) and DataType.COMPLEX in parameter.data_type
    ):
        raise ValueError(
            "Attempt to split complex into real parts from dataset which does not contain complex numbers."
        )
    coord_attributes = {}
    for dim in data.sizes:
        coord_attributes[dim] = data[dim].attrs
    da1 = Parameter(
        name=f"{suffix[0]}::{parameter.name}",
        label=f"{suffix[0].title()} {label}",
        unit=parameter.unit,
        data_type=DataType.FLOAT,
    ).build_data_array(
        data=converted_data[0],
        dimensions=list(data.sizes),
        coords=data.coords.variables.mapping,
    )
    unit = ("deg" if complex_options.in_degrees else "rad") if complex_options.use_magnitude_phase else parameter.unit
    da2 = Parameter(
        name=f"{suffix[1]}::{parameter.name}",
        label=f"{suffix[1].title()} {label}",
        unit=unit,
        data_type=DataType.FLOAT,
    ).build_data_array(
        data=converted_data[1],
        dimensions=list(data.sizes),
        coords=data.coords.variables.mapping,
    )
    return da1, da2


def _build_title(data: xr.DataArray, variables: str | list[str]) -> str:
    # standardize a list to iterate over
    variable_shortname_list = [variables] if isinstance(variables, str) else variables
    # getting more human-readable identifiers for the variables
    variable_list = []
    for variable in [str(v) for v in variable_shortname_list]:
        if variable in data.coords:
            variable_list.append(data[variable].attrs.get("long_name", variable))
        else:
            variable_list.append(variable)
    variable_str = ", ".join(variable_list)
    return f"{data.attrs.get('long_name', '')} vs {variable_str}"


def _stack(data: xr.DataArray, plot_options: PlotOptions) -> xr.DataArray:
    """Shift data vertically with respect to each other to create a stacked multiline plot.

    Args:
        data: Data to be plotted.
        plot_options: Plot-specific options.

    """
    stacked_data = []
    band = np.max(data.values) - np.min(data.values)
    gap = plot_options.stack * band / 100.0
    transposed_data = data.values.T
    for index, value in enumerate(transposed_data):
        stacked_data.append(value + gap * index)
    return data.copy(deep=True, data=np.asarray(stacked_data).T)
