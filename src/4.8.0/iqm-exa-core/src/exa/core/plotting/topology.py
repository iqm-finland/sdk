#  *****************************************************************************
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
#  *****************************************************************************
"""Functions to draw the topology plot."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
import csv
from datetime import UTC, datetime
from functools import cache
import os
import pathlib
from pathlib import Path
import re
from typing import Any, NamedTuple

import matplotlib as mpl
from matplotlib import patheffects
from matplotlib.colors import Colormap, ListedColormap
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from matplotlib.textpath import TextPath
import numpy as np
from typing_extensions import deprecated

from exa.common.helpers import format_deprecated
from exa.common.qcm_data import ChipLayout, ChipTopology
from exa.common.qcm_data.chad_model import CHAD
from exa.core.util.dut_field_decoder import decode_dut_field
from iqm.station_control.interface.models import ObservationData

from .plots import format_for_display
from .qpu_core.colormaps import IQM_COLORMAP, NormalizedColormap
from .qpu_core.component_styles import default_style
from .qpu_core.utils import (
    QPUPlotObservation,
    fix_unit_prefixes,
    get_exponent,
    get_unit,
)


class PlotData(NamedTuple):
    """Collection of values and units required to plot topology."""

    values: dict[str, int | float] | dict[str, int] | dict[str, float]
    units: dict[str, str]


@deprecated(format_deprecated(old="`topology_plot_for_chad`", new=None, since="2026-07-16"))
def topology_plot_for_chad(  # noqa: PLR0913
    chad: CHAD,
    chip_label: str,
    observations: list[ObservationData],
    observation_type: str,
    split_dut_field: Callable[[str], tuple[str, str]] = decode_dut_field,
    qb_colormap: ListedColormap | None = None,
    tc_colormap: ListedColormap | None = None,
    cr_colormap: ListedColormap | None = None,
    save_path: Path | None = None,
    include_component_names: bool = True,
    **kwargs,
) -> Figure:
    """Draw the topology plot for the given chip label.

    Args:
        chad: The chip architecture definition (CHAD).
        chip_label: This will label the figure too.
        observations: Data to plot.
        observation_type: Type of data to plot.
        split_dut_field: Function to extract component name and observation type from DUT field.
        qb_colormap: For qubits.
        tc_colormap: For tunable couplers.
        cr_colormap: For computational resonators.
        save_path: If given, the resulting plot is saved in the path.
        include_component_names: Include the names of the components.
        **kwargs: kwargs for create topology plot methods.

    Returns:
        The topology map as Matplotlib figure.

    Raises:
        ValueError if chip type not supported.

    """
    if any(
        isinstance(obs.value, str) or (isinstance(obs.value, np.ndarray) and np.issubdtype(obs.value.dtype, str))
        for obs in observations
    ):
        raise ValueError(f"Some values in {observations=} can't be plotted.")

    qpuplot_observations = []
    for obs in observations:
        comp_name, obs_type = split_dut_field(obs.dut_field)
        if obs_type == observation_type:
            qpuplot_observations.append(QPUPlotObservation(component=comp_name, metric=obs_type, observation=obs))

    if kwargs.get("chip_layout") is None and isinstance(csv_file := kwargs.get("csv_file"), str):
        kwargs["chip_layout"] = _chip_layout_from_csv(ChipTopology.from_chad(chad), csv_file)

    plot_data = _create_observation_plot_data(
        qpuplot_observations, kwargs["chip_layout"], only_rounding="vmin" in kwargs or "vmax" in kwargs
    )

    figure_name = (
        chip_label + " - " + format_for_display(qpuplot_observations[0].metric) if qpuplot_observations else "dummy"
    )
    kwargs.setdefault("figure_title", figure_name)
    return plot_topology(
        data=plot_data,
        layout=kwargs["chip_layout"],
        observation_type=observation_type,
        qb_colormap=qb_colormap,
        tc_colormap=tc_colormap,
        cr_colormap=cr_colormap,
        save_path=save_path,
        include_component_names=include_component_names,
        **kwargs,
    )


def _get_chip_layout_from_csv(chip_design_record: dict[str, Any], csv_file: str) -> ChipLayout:
    """Get chip layout from CSV file and checks compatibility with the chip design record."""
    chip_topology = ChipTopology.from_chip_design_record(chip_design_record)
    return _chip_layout_from_csv(chip_topology, csv_file)


def _chip_layout_from_csv(chip_topology: ChipTopology, csv_file: str) -> ChipLayout:
    """Build a chip layout from a CSV file, checking compatibility with the given chip topology."""
    csv_path = Path(csv_file)
    csv_plot_data = _sync_layout_with_chip(chip_topology, _create_topology_plot_data(_get_layout_from_file(csv_path)))
    chip_layout = ChipLayout(
        qubits={
            name: (csv_plot_data[name]["x"], csv_plot_data[name]["y"])
            for name in csv_plot_data
            if name.startswith("QB")
        },
        couplers={
            name: (csv_plot_data[name]["x"], csv_plot_data[name]["y"])
            for name in csv_plot_data
            if name.startswith("TC")
        },
        computational_resonators={
            name: (csv_plot_data[name]["x"], csv_plot_data[name]["y"])
            for name in csv_plot_data
            if name.startswith("COMP")
        },
    )

    chip_layout.mirror_yaxis()
    return chip_layout


def _create_observation_plot_data(
    observations: list[QPUPlotObservation],
    chip_layout: ChipLayout,
    only_rounding: bool = False,
) -> PlotData:
    """Prepares the observation data for the topology plot.

    Args:
        observations: The data to plot.
        chip_layout: The layout of the QPU the data belongs to.
        only_rounding: Formats the data values applying specific rounding.

    Returns:
        The formatted observation data.

    """
    chip_components = chip_layout.components
    data: dict[str, dict[str, Any]] = {comp: {} for comp in chip_components}
    obs_plot_data = {
        obs.component: {
            "value": float(np.squeeze(obs.observation.value)),
            "obs_param": obs.metric,
            "unit": obs.observation.unit,
        }
        for obs in observations
        if obs.component in chip_components
    }
    for component in chip_components:
        if component in obs_plot_data:
            data[component].update(**obs_plot_data[component])

    data = _fix_data_units(data, digit_count=3, only_rounding=only_rounding)

    plot_data = PlotData(
        values={component: data[component]["value"] for component in data if "value" in data[component]},
        units={component: data[component]["unit"] for component in data if "unit" in data[component]},
    )

    return plot_data


def plot_topology(  # noqa: PLR0912, PLR0915, PLR0913
    figure_title: str,
    layout: ChipLayout,
    data: PlotData | None = None,
    observation_type: str = "",
    qb_colormap: Colormap | None = IQM_COLORMAP,
    tc_colormap: Colormap | None = IQM_COLORMAP,
    cr_colormap: Colormap | None = IQM_COLORMAP,
    save_path: Path | None = None,
    include_component_names: bool = True,
    write_values: bool = True,
    show_colorbar: bool = True,
    background_color: str | None = None,
    resolution: int = 300,
    show_logo: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
    close_fig: bool = False,
    min_spacing: float | None = None,
    plot_style: dict | None = None,
) -> Figure:
    """Draw the topology showing a symbolic representation of the couplers and the qubits layout.

    Args:
        figure_title: The title of the figure.
        layout: The locations of the components.
        data: Contains component values and units.
        observation_type: Name of observation type, used for the colorbar label.
        qpu_type: Name of the QPU to plot. Necessary to select correct style format.
        qb_colormap: Colormap for qubits.
        tc_colormap: Colormap for couplers.
        cr_colormap: Colormap for computational resonators.
        save_path: Path object to save directory.
        include_component_names: Boolean value indicating whether components names should be explicitly shown.
        write_values: Boolean value indicating whether data values should be explicitly written.
        show_colorbar: Boolean value to determine whether colorbars should be added.
        background_color: Set background color.
        resolution: Dots-per-inch, only used if saved as a raster image.
        show_logo: If True, shows the IQM logo with timestamp. If False, logo is removed.
        vmin: Minimal value of the colormap.
        vmax: Maximal value of the colormap.
        close_fig: If True, closes plt figures.
        min_spacing: Distance between the closest components.
        plot_style: The plot style to use.

    Returns:
        The topology map.

    """
    use_colormap = False

    qb_norm_colormap = None
    tc_norm_colormap = None
    cr_norm_colormap = None

    if data is None:
        data = PlotData({}, {})

    style_to_use = plot_style if plot_style else default_style()

    compr_style_to_use = deepcopy(style_to_use)
    if len(layout.computational_resonators) == 1:
        width, height = style_to_use["comp_r"]["size"]
        compr_style_to_use["comp_r"]["size"] = (width * len(layout.qubits) / 1.3, height)

    if len(layout.qubits) == 150:
        style_to_use["figure_size"] = (30, 30)

    style_to_use["component_label"]["fontsize"] = (
        3
        / np.log(len(layout.qubits) + len(layout.computational_resonators))
        * style_to_use["component_label"]["fontsize"]
    )
    compr_style_to_use["component_label"]["fontsize"] = style_to_use["component_label"]["fontsize"]

    if min_spacing is not None:
        layout.normalize_coordinates(min_spacing)

    qb_values = [data.values[qb] for qb in layout.qubits if qb in data.values]
    tc_values = [data.values[tc] for tc in layout.couplers if tc in data.values]
    cr_values = [data.values[cr] for cr in layout.computational_resonators if cr in data.values]

    # If there are no data points at all, no values will be written and no colorbar will be shown.
    if len(list(data.values.keys())) == 0:
        write_values = False
        show_colorbar = False
    # If there are some data points and at least one colormap defined us it.
    elif any(cm is not None for cm in [qb_colormap, tc_colormap, cr_colormap]):
        use_colormap = True

    # If we have both qubit and coupler values to plot, topology plot creates two colorbars
    fig, ax = plt.subplots(figsize=style_to_use["figure_size"])
    n_colorbars = 0
    if show_colorbar and len(qb_values) > 0 and len(tc_values) > 0 and qb_colormap == tc_colormap:
        # coupler colormap is forced to be different from IQM colors
        tc_colormap = mpl.colormaps["binary"]
        if len(cr_values) > 0 and qb_colormap == cr_colormap:
            # computational resonator colormap is forced to be different from IQM colors and TC colormap
            cr_colormap = mpl.colormaps["Wistia"]

    # computational resonator colormap is forced to be different from IQM colors
    elif (show_colorbar and len(qb_values) > 0 and len(cr_values) > 0 and qb_colormap == cr_colormap) or (
        show_colorbar and len(tc_values) > 0 and len(cr_values) > 0 and tc_colormap == cr_colormap
    ):
        cr_colormap = mpl.colormaps["binary"]

    if qb_colormap is not None and len(qb_values) > 0:
        qb_min_max = _adjust_min_max(qb_values)
        qb_min_val = vmin if vmin is not None else qb_min_max[0]
        qb_max_val = vmax if vmax is not None else qb_min_max[1]
        qb_norm_colormap = NormalizedColormap(colormap=qb_colormap, min_val=qb_min_val, max_val=qb_max_val)  # type: ignore[arg-type]
        if show_colorbar:
            n_colorbars += 1
            qb_cbar_title = format_for_display(label=observation_type)
            qb_unit = get_unit(data.units, layout.qubits)
            if qb_unit:
                qb_cbar_title = f"{qb_cbar_title} ({qb_unit})"
            _add_colorbar(
                title=qb_cbar_title,
                min_val=qb_min_val,
                max_val=qb_max_val,
                figure_object=fig,
                axis_object=ax,
                style=style_to_use,
                cmap=qb_colormap,
                colorbar_count=n_colorbars,
            )

    if tc_colormap is not None and len(tc_values) > 0:
        tc_min_max = _adjust_min_max(tc_values)
        tc_min_val = vmin if vmin is not None else tc_min_max[0]
        tc_max_val = vmax if vmax is not None else tc_min_max[1]
        tc_norm_colormap = NormalizedColormap(colormap=tc_colormap, min_val=tc_min_val, max_val=tc_max_val)  # type: ignore[arg-type]

        if show_colorbar:
            n_colorbars += 1
            tc_cbar_title = format_for_display(label=observation_type)
            tc_unit = get_unit(data.units, layout.couplers)
            if tc_unit:
                tc_cbar_title = f"{tc_cbar_title} ({tc_unit})"
            _add_colorbar(
                title=tc_cbar_title,
                min_val=tc_min_val,
                max_val=tc_max_val,
                figure_object=fig,
                axis_object=ax,
                style=style_to_use,
                cmap=tc_colormap,
                colorbar_count=n_colorbars,
            )

    if cr_colormap is not None and len(cr_values) > 0:
        cr_min_max = _adjust_min_max(cr_values)
        cr_min_val = vmin if vmin is not None else cr_min_max[0]
        cr_max_val = vmax if vmax is not None else cr_min_max[1]
        cr_norm_colormap = NormalizedColormap(colormap=cr_colormap, min_val=cr_min_val, max_val=cr_max_val)  # type: ignore[arg-type]

        if show_colorbar:
            n_colorbars += 1
            cr_cbar_title = format_for_display(label=observation_type)
            cr_unit = get_unit(data.units, layout.computational_resonators)
            if cr_unit:
                cr_cbar_title = f"{cr_cbar_title} ({cr_unit})"
            _add_colorbar(
                title=cr_cbar_title,
                min_val=cr_min_val,
                max_val=cr_max_val,
                figure_object=fig,
                axis_object=ax,
                style=style_to_use,
                cmap=cr_colormap,
                colorbar_count=n_colorbars,
            )

    # Plot basic shapes
    for qubit in set(layout.qubits) - layout.qubits_without_probeline:
        _draw_qubit(
            ax,
            data.values.get(qubit),
            layout.get_coordinates(qubit),
            style_to_use,
            include_component_names,
            qubit,
            use_colormap,
            write_values,
            qb_norm_colormap,
        )
    for coupler in layout.couplers:
        _draw_coupler(
            ax,
            data.values.get(coupler),
            layout.get_coordinates(coupler),
            layout.get_angle(coupler),
            style_to_use,
            include_component_names,
            coupler,
            use_colormap,
            write_values,
            tc_norm_colormap,
        )
    for compr in layout.computational_resonators:
        _draw_comp_resonator(
            ax,
            data.values.get(compr),
            layout.get_coordinates(compr),
            layout.get_angle(compr),
            compr_style_to_use,
            include_component_names,
            compr,
            use_colormap,
            write_values,
            cr_norm_colormap,
        )
    central_qubit_style_to_use = deepcopy(style_to_use)
    central_qubit_style_to_use["qubit"]["color"] = style_to_use["comp_r"]["color"]
    for qubit in layout.qubits_without_probeline:
        _draw_qubit(
            ax,
            data.values.get(qubit),
            layout.get_coordinates(qubit),
            central_qubit_style_to_use,
            include_component_names,
            qubit,
            use_colormap,
            write_values,
            qb_norm_colormap,
        )
    ax.set_title(
        figure_title,
        fontsize=style_to_use["plot_title"]["fontsize"],
        fontfamily=style_to_use["plot_title"]["fontfamily"],
    )
    ax.autoscale()
    ax.set_aspect(1)
    ax.axis("off")

    if background_color is not None:
        fig.set_facecolor(background_color)

    if show_logo:
        _add_logo_with_timestamp(ax, style_to_use)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=resolution, bbox_inches="tight")

    if close_fig:
        plt.close(fig)

    return fig


def _adjust_min_max(values: list[float]) -> tuple[float, float]:
    """Extract minimum and maximum value and adjust if they are the same."""
    min_val = min(values)
    max_val = max(values)
    # In situations where there are only one component to be plotted, min and max values may be equal. If so,
    # force them to be different to avoid errors in colorbar
    if min_val == max_val and min_val:
        min_val -= 0.05 * abs(min_val)  # remove 5%
        max_val += 0.05 * abs(min_val)  # add 5%

    return min_val, max_val


@cache
def _reference_cap_height(fontweight: str) -> float:
    """Height of a capital "H" at unit font size, used as a text-independent scale reference.

    Cached because it only depends on the font weight (the same value is reused for every label),
    so the reference ``TextPath`` is built once per weight rather than once per component.
    """
    return TextPath((0, 0), "H", size=1.0, prop=FontProperties(weight=fontweight)).get_extents().height


def _write_component_label(
    axis: plt.Axes,
    style: dict[str, Any],
    label: str,
    x_coordinate: float,
    y_coordinate: float,
    fontweight: str,
) -> None:
    """Function to add labels to the topology plot in data coordinates."""
    axis.text(
        x_coordinate,
        y_coordinate,
        label,
        verticalalignment="center",
        horizontalalignment="center",
        color=style["color"],
        fontsize=style["fontsize"],
        fontweight=fontweight,
        path_effects=[
            patheffects.withStroke(
                linewidth=style["path_linewidth"],
                foreground=style["path_color"],
            )
        ],
    )


def _add_logo_with_timestamp(axis: plt.Axes, style_dict: dict[str, dict[str, Any]]) -> None:
    """Inserts the black IQM logo (imported as a jpg). with a time stamp of creation underneath."""
    parent_path = pathlib.Path(__file__).parent.resolve()
    filename = os.path.join(parent_path, "qpu_core/IQM-LOGO-BLACK.jpg")
    arr_image = plt.imread(filename, format="jpg")

    logo_position: tuple[float, float] = style_dict["logo"]["position"]
    logo_dimensions: tuple[float, float] = style_dict["logo"]["size"]

    time_stamp_position: tuple[float, float] = style_dict["time_stamp"]["position"]
    time_stamp_fontsize: int = style_dict["time_stamp"]["fontsize"]

    axin = axis.inset_axes((*logo_position, *logo_dimensions), transform=axis.transAxes)
    axin.imshow(arr_image)
    axin.axis("off")

    current_time = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    axin.text(*time_stamp_position, current_time, transform=axin.transAxes, fontsize=time_stamp_fontsize)


def _add_colorbar(
    title: str,
    min_val: float,
    max_val: float,
    style: dict[str, dict[str, Any]],
    cmap: Colormap | None,
    figure_object: mpl.figure.Figure,
    axis_object: plt.Axes,
    colorbar_count: int,
) -> None:
    """Add colorbar to plot with some customization."""
    axins1 = axis_object.inset_axes((1.0 + (colorbar_count - 1) * 0.17, 0, 0.04, 1))

    norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)
    mapped_colorscale = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = figure_object.colorbar(mapped_colorscale, cax=axins1, ticks=np.linspace(min_val, max_val, 7))
    fontsize = style["color_bar"]["fontsize"]
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.offsetText.set(size=fontsize)
    cbar.set_label(title, size=fontsize)


def _draw_qubit(  # noqa: PLR0913
    ax_object: plt.Axes,
    data_value: float | None,
    coordinates: tuple[float, float],
    style_to_use: dict[str, dict[str, Any]],
    write_qubit_label: bool,
    name: str,
    use_colormap: bool,
    write_values: bool,
    qb_norm_colormap: NormalizedColormap | None,
) -> None:
    """Draw qubits as circles."""
    qubit_size = style_to_use["qubit"]["radius"]
    x_position = coordinates[0]
    y_position = qubit_size + coordinates[1]

    if use_colormap:
        if data_value is not None and qb_norm_colormap is not None:
            facecolor = qb_norm_colormap.get_color(data_value)
            if write_values:
                _write_component_label(
                    ax_object,
                    style_to_use["component_label"],
                    str(data_value),
                    x_position,
                    y_position - qubit_size,
                    fontweight="black",
                )
        else:
            facecolor = "white"
    else:
        facecolor = style_to_use["qubit"]["color"]

    ax_object.add_patch(
        plt.Circle(
            (coordinates[0], coordinates[1]),
            qubit_size,
            facecolor=facecolor,
            edgecolor=style_to_use["qubit"]["edgecolor"],
            linewidth=style_to_use["qubit"]["linewidth"],
        )
    )

    if write_qubit_label:
        _write_component_label(
            ax_object,
            style_to_use["component_label"],
            name,
            x_position,
            y_position,
            fontweight=style_to_use["component_label"]["fontweight"],
        )


def _draw_coupler(  # noqa: PLR0913
    ax_object: plt.Axes,
    data_value: float | None,
    coordinates: tuple[float, float],
    angle: float,
    style_to_use: dict[str, dict[str, Any]],
    write_coupler_label: bool,
    name: str,
    use_colormap: bool,
    write_values: bool,
    tc_norm_colormap: NormalizedColormap | None,
) -> None:
    """Draw couplers as rotated squares."""
    coupler_size = style_to_use["coupler"]["size"]
    x_position = coordinates[0]
    y_position = coordinates[1] + coupler_size[1] / 2

    if use_colormap:
        if data_value is not None and tc_norm_colormap is not None:
            facecolor = tc_norm_colormap.get_color(data_value)
            if write_values:
                _write_component_label(
                    ax_object,
                    style_to_use["component_label"],
                    str(data_value),
                    x_position,
                    y_position - coupler_size[1] / 2,
                    fontweight="black",
                )
        else:
            facecolor = "white"
    else:
        facecolor = style_to_use["coupler"]["color"]

    rectangle_patch = plt.Rectangle(
        (coordinates[0] - coupler_size[0] / 2, coordinates[1] - coupler_size[1] / 2),
        *coupler_size,
        angle=angle,
        rotation_point="center",
        facecolor=facecolor,
        edgecolor=style_to_use["coupler"]["edgecolor"],
        linewidth=style_to_use["coupler"]["linewidth"],
    )
    ax_object.add_patch(rectangle_patch)

    if write_coupler_label:
        _write_component_label(
            ax_object,
            style_to_use["component_label"],
            name,
            x_position,
            y_position,
            fontweight=style_to_use["component_label"]["fontweight"],
        )


def _draw_comp_resonator(  # noqa: PLR0913
    ax_object: plt.Axes,
    data_value: float | None,
    coordinates: tuple[float, float],
    angle: float,
    style_to_use: dict[str, dict[str, Any]],
    write_coupler_label: bool,
    name: str,
    use_colormap: bool,
    write_values: bool,
    tc_norm_colormap: NormalizedColormap | None,
) -> None:
    """Draw computational resonators as rectangles."""
    resonator_size = style_to_use["comp_r"]["size"]
    x_position = coordinates[0]
    y_position = coordinates[1] + resonator_size[1] / 2

    if use_colormap:
        if data_value is not None and tc_norm_colormap is not None:
            facecolor = tc_norm_colormap.get_color(data_value)
            if write_values:
                _write_component_label(
                    ax_object,
                    style_to_use["component_label"],
                    str(data_value),
                    x_position,
                    y_position - resonator_size[1] / 2,
                    fontweight="black",
                )
        else:
            facecolor = "white"
    else:
        facecolor = style_to_use["comp_r"]["color"]

    rectangle_patch = plt.Rectangle(
        (coordinates[0] - resonator_size[0] / 2, coordinates[1] - resonator_size[1] / 2),
        *resonator_size,
        angle=angle,
        rotation_point="center",
        facecolor=facecolor,
        edgecolor=style_to_use["comp_r"]["edgecolor"],
        linewidth=style_to_use["comp_r"]["linewidth"],
    )
    ax_object.add_patch(rectangle_patch)

    if write_coupler_label:
        _write_component_label(
            ax_object,
            style_to_use["component_label"],
            name,
            x_position,
            y_position,
            fontweight=style_to_use["component_label"]["fontweight"],
        )


def _get_layout_from_file(csv_path: Path) -> list:
    with open(csv_path, encoding="utf-8-sig") as file:
        read = csv.DictReader(file, dialect="excel")
        csv_data = list(read)
    return csv_data


def _sync_layout_with_chip(chip_topology: ChipTopology, plot_data: dict[str, dict]) -> dict[str, dict[str, Any]]:
    """Data in CSV file could use different labels for couplers than chip design record, so this is fixed here."""
    # Any qubit with a non-positive number in the topology is considered a dummy
    qubits = [
        qubit
        for qubit in chip_topology.qubits_sorted
        if (match := re.search(r"\d+", qubit)) and int(match.group(0)) > 0
    ]
    qubits_in_data = [name for name in plot_data if name.startswith("QB")]
    # All qubits should be in plot data, otherwise something must have gone wrong
    if not all(qubit in qubits_in_data for qubit in qubits) or (len(qubits) != len(qubits_in_data)):
        raise ValueError("Qubits in CSV file differ from QPU.")

    new_plot_data = {}
    for qubit in qubits:
        new_plot_data[qubit] = plot_data[qubit]

    couplers = chip_topology.couplers_sorted
    couplers_in_data = [name for name in plot_data if name.startswith("TC")]
    if len(couplers) != len(couplers_in_data):
        raise ValueError("Couplers in CSV file differ from QPU.")
    if couplers[0] not in couplers_in_data:
        # Different naming convention, i.e., TC1 vs TC-1-R1 in star
        for component in couplers_in_data:
            if re.fullmatch("TC[0-9]+", component):
                new_component = re.sub("TC", "TC-", component)
                new_plot_data[next(iter(coupler for coupler in couplers if coupler.startswith(new_component)))] = (
                    plot_data[component]
                )

    else:
        for coupler in couplers:
            new_plot_data[coupler] = plot_data[coupler]

    computational_resonators = chip_topology.computational_resonators_sorted
    comp_in_data = [name for name in plot_data if name.startswith("COMP")]
    if len(computational_resonators) != len(comp_in_data):
        raise ValueError("Computational resonators in CSV file differ from QPU.")
    if computational_resonators:
        if not len(computational_resonators) == 1:
            raise ValueError("Found more than one computational resonator.")
        if computational_resonators[0] != comp_in_data[0]:
            new_plot_data[computational_resonators[0]] = plot_data[comp_in_data[0]]
        else:
            new_plot_data[computational_resonators[0]] = plot_data[computational_resonators[0]]

    return new_plot_data


def _create_topology_plot_data(csv_data: list) -> dict[str, dict[str, Any]]:
    """Creates a dictionary in the format needed for draw_topology_plot().

    This dict only contains the topology information--the observations will be added separately.
        Key:
            Component name (e.g. 'QB1', 'TC-1-2')
        Value dict:
            'x': X position
            'y': Y position
            'flux': (optional) Flux speed ('slow', 'fast')
            'probeline': (optional) Probeline
    """
    plot_data: dict[str, dict] = {}

    for data in csv_data:
        component = ""
        attr: dict[str, Any] = {}

        if re.fullmatch("FL-QB[0-9]+", data["device"]):
            component = data["device"].replace("FL-", "")  # make name in format e.g. 'QB1'
            if data["device type"] == "flux":  # fast flux qubit
                attr["flux"] = "fast"
            elif data["device type"] == "slow flux":  # slow flux qubit
                attr["flux"] = "slow"
            attr["probeline"] = data["probeline"]  # all qubits will have a probeline
        elif re.fullmatch("DL-QB[0-9]+", data["device"]):
            component = data["device"].replace("DL-", "")  # make name in format e.g. 'QB1'
            attr["flux"] = "slow"

        elif re.match("FL-TC", data["device"]):  # tunable coupler
            component = data["device"].replace("FL-", "")  # make name in format e.g. 'TC-1-2', 'TC1'

        elif re.match("COMP", data["device"]):  # computational resonator
            component = data["device"]  # make name in format e.g. 'COMPR1'
            # no flux, no probe line for couplers
        if component:
            attr["x"] = float(data["device x"])
            attr["y"] = float(data["device y"])

            plot_data.update({component: attr})

    return plot_data


def _fix_data_units(
    data: dict[str, dict[str, Any]], digit_count: int, only_rounding: bool = False
) -> dict[str, dict[str, Any]]:
    """Sets the units to be in scientific notation, such that the values have no fewer than 1 digit in front
    of the decimal point.

    The values have no fewer than 1 digit in front of the decimal point.
    Rounds the values to digit_count, and adjusts them according to new units.
    """
    plot_data = deepcopy(data)
    obs_types = [data[key]["obs_param"] for key in data if "obs_param" in data[key]]

    all_units_old: set[str] = {data[key]["unit"] for key in data if "unit" in data[key]}
    all_units_new = []

    for unit in all_units_old:
        # Minimum value with the unit
        data_values = np.asarray(
            [data[d]["value"] for d in data if ("value" in data[d] and "unit" in data[d] and data[d]["unit"] == unit)]
        )

        # If our only values are 0 get_exponent returns 0
        exponent = get_exponent(max(np.abs(data_values)))
        # exponent of max_val in scientific notation
        conversion_factor, new_unit = fix_unit_prefixes(exponent, unit, obs_types[0])
        if only_rounding:
            conversion_factor = 1

        all_units_new.append(new_unit)

        # Adjust values and units in data dict for this unit
        for key in data:  # noqa: PLC0206
            if "unit" in data[key] and data[key]["unit"] == unit:
                plot_data[key]["unit"] = new_unit  # e.g. "Hz" -> "kHz"
                if "value" in data[key]:
                    plot_data[key]["value"] = round(data[key]["value"] * conversion_factor, digit_count)

    return plot_data
