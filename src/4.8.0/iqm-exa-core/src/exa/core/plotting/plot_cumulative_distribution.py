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
"""Plot cumulative distributions."""

from collections.abc import Callable
import logging
from pathlib import Path
from typing import Any

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
import numpy as np
from scipy.optimize import minimize

from exa.common.errors import ValidationError
from exa.core.util.dut_field_decoder import decode_dut_field
from iqm.station_control.interface.models import ObservationData, ObservationDefinition

from .plots import (
    _check_invalid_observations,
    _construct_observations_list,
    _get_label_from_device,
    format_for_display,
)
from .qpu_core.utils import (
    QPUPlotObservation,
    change_unit_magnitude,
    check_same_unit,
    get_exponent,
    get_scaling_from_unit,
    logit_mean,
    shorten_label,
)

logger = logging.getLogger(__name__)


def plot_multi_cdf(  # noqa: PLR0913
    observations: list[ObservationDefinition],
    observation_types: list[str],
    color_scheme: ListedColormap,
    split_dut_field: Callable[[str], tuple[str, str]] = decode_dut_field,
    save_path: Path | None = None,
    resolution: int = 300,
    xscale: str = "linear",
    show_logit_mean: bool = False,
    fig_size: tuple[float, float] = (6.0, 4.0),
) -> Figure:
    """Draw the cumulative distribution of the observations for the given types.

    Multi-CDF - cumulative histogram plot with multiple lines.

    Args:
        observations: The data to plot.
        observation_types: One cumulative distribution for each type.
        color_scheme: colormap for the various lines in the figure.
        split_dut_field: How to extract component name and observation type from DUT field.
        save_path: Where to save the figure.
        resolution: In dpi for saving the figure.
        xscale: Accepts the values for set_xscale in matplotlib. Default is linear. Only with linear x-axis will the
            units be scaled automatically.
        show_logit_mean: If True, calculates the logit mean. If False, calculates the median as usual.
        fig_size: Size of figure in inches

    Returns:
        The Matplotlib figure containing the plot.

    Raises:
        ValueError if no observation types are provided or units are mismatching.

    """
    if len(observation_types) < 1:
        raise ValidationError(f"Please provide at least one observation type to be plotted. {observation_types=}")
    _check_invalid_observations(observations)

    qpuplot_observations = []
    for obs in observations:
        comp_name, obs_type = split_dut_field(obs.dut_field)
        qpuplot_observations.append(QPUPlotObservation(component=comp_name, metric=obs_type, observation=obs))  # type: ignore[arg-type]

    plot_data, plot_labels = _prepare_data_for_multi_cdf(
        qpuplot_observations, observation_types, xscale, show_logit_mean, color_scheme
    )

    # For x-axis label, use only the name of the unit, such as time or frequency
    return _plot_cdf(
        plot_data=plot_data,
        save_path=save_path,
        resolution=resolution,
        xscale=xscale,
        x_label=plot_labels["x_label"],
        show_logit_mean=show_logit_mean,
        color_scheme=color_scheme,
        fig_size=fig_size,
    )


def _plot_cdf(  # noqa: PLR0913
    plot_data: list[dict[str, Any]],
    save_path: Path | None,
    resolution: int,
    xscale: str,
    x_label: str,
    show_logit_mean: bool,
    color_scheme: ListedColormap,
    fig_size: tuple[float, float],
    color_label: bool = True,
) -> Figure:
    # Draw the plot
    fig, ax = plt.subplots(figsize=fig_size, dpi=resolution)

    # Create list to store median values and their positions
    store_median_values = []
    corresponding_units = []
    store_y_values = []
    store_median_coords = []

    for data in plot_data:
        line_label = format_for_display(f"{data['line_label']}")
        ax.hist(
            data["data"]["value"],
            1000,
            density=True,
            histtype="step",
            cumulative=True,
            label=line_label,
            linewidth=2.5,
            color=data["line_color"],
        )
        ax.vlines(
            data["median_coords"],
            linestyle="dashdot",
            alpha=0.7,
            ymin=data["y_min"],
            ymax=1.0,
            zorder=0,
            colors=data["line_color"],
            linewidth=1.5,
        )
        store_median_values.append(data["median"]["value"])
        store_median_coords.append(data["median_coords"])
        store_y_values.append(data["y_min"])
        corresponding_units.append(data["median"]["unit"])

    ax.set_ylim(0, 1)
    ax.set_xscale(xscale)

    ax.set_xlabel(x_label)
    ax.set_ylabel("Experimental CDF")
    plt.legend(loc="upper left", prop={"size": 8})
    plt.grid()

    # remove final vertical line in cdf plot
    axpolygons = [poly for poly in ax.get_children() if isinstance(poly, mpl.patches.Polygon)]
    for poly in axpolygons:
        poly.set_xy(poly.get_xy()[:-1])

    adjusted_text_positions = _label_placement_optimizer(store_median_coords, ax)

    # optimizer does not take into account any desired ordering
    adjusted_text_positions = _forced_sorting(adjusted_text_positions, store_median_coords)

    _adding_horizontal_lines_and_values(
        store_median_values,
        adjusted_text_positions,
        store_median_coords,
        store_y_values,
        corresponding_units,
        ax,
        show_logit_mean,
        color_scheme,
        color_label,
        linestyle="dashdot",
        linealpha=0.7,
        linewidth=1.5,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=resolution)

    return fig


def _label_weight_func(x_new: np.ndarray, x_old: np.ndarray, limit: float) -> float:
    """A function designed to help with optimizing the label spacing in the CDF plot. The weight is a function of the
    squared distance between x_old and x_new as well as the inverse of the squared distance between every element in
    x_new.

    Args:
        x_new: A 1D array of positions.
        x_old: A 1D array of the original/old positions
        limit: Value which sets the threshold for when the positions are considered to close to one another.

    """
    weight = np.sum((x_new - x_old) ** 2)
    distance_to_others = []
    for loop_index, current_x_value in enumerate(x_new):
        distance = np.abs(np.delete(x_new, loop_index) - current_x_value)
        if np.any(distance < limit):
            distance_to_others.append(1 / distance**2)
    total_weight = weight + np.sum(distance_to_others)
    return total_weight * 100


def _transform_func(axis_object: Any, point_in_data_coord: tuple) -> np.ndarray:
    """Helper function to transform some coordinates in a plot from data coordinates to axis coordinates."""
    new_point_loc = axis_object.transData.transform(point_in_data_coord)
    return axis_object.transAxes.inverted().transform(new_point_loc)


def _calculate_median_or_logit_mean(
    dataset: np.ndarray, use_logit_mean: bool, use_new_units: bool, observation_type: str, unit: str, n_bins: int
) -> tuple[float, str, float, float]:
    """Helper function for CDF plot. Calculates the median or logit mean of the data. Also provides what the expected
    y-value is. The median is calculated with respect to the original data and then the value is scaled to new units,
    if any.

    Args:
        dataset: Dataset to calculate median or mean for.
        use_logit_mean: Boolean value to indicate whether logit mean should be provided or not
        use_new_units: Boolean value to indicate whether we want to scale dataset and median to new units.
        observation_type: Name of the observation.
        unit: Unit of observation.
        n_bins: Number of bins for numpy histogram function.

    Returns:
        Returns either the logit mean or median, and the corresponding value of the measured CDF function.
        summary_number - Is the logit mena or median. Will always be scaled to new units.
        new_unit - provides the new unit of the summary number
        y_cdf_value - The CDF function value evaluated at the median or logit mean.
        median_axis_coordinates - The median value in position to the x-axis of the figure. It will be identical to the
            summary number if no unit scaling is provided.

    """
    summary_number = np.median(dataset)
    # for the median, we know the cdf value is 0.5 by definition
    y_cdf_value = 0.5

    if use_logit_mean:
        # if true, summary_number and y_cdf_value may be overwritten
        if not np.all((dataset < 1) & (dataset > 0)):
            logger.warning(
                "Logit is only defined for values between 0 and 1. Current dataset does not satisfy this."
                "Calculating median instead."
            )
        else:
            logit_mean_value = logit_mean(dataset)
            hist_output, bin_range = np.histogram(
                dataset,
                bins=n_bins,
                density=True,
            )
            step_width = bin_range[1] - bin_range[0]
            cdf_output = np.cumsum(hist_output) * step_width

            # identify where the new median cuts the cdf curve:
            y_cdf_value = float(np.interp(logit_mean_value, bin_range[1:], cdf_output))
            summary_number = logit_mean_value

    median_axis_coordinates = summary_number

    if use_new_units:
        unit, scaled_summary_number, _ = change_unit_magnitude(unit, [summary_number], observation_type)
        summary_number = float(scaled_summary_number[0])

    return summary_number, unit, y_cdf_value, median_axis_coordinates


def _label_placement_optimizer(
    original_label_coordinates: list | np.ndarray,
    axis_object: Any,
    closest_space: float = 0.03,
) -> np.ndarray:
    """Helper function to determine where to space labels. Not too close to each other and not too far away from original
    position.

    Args:
        original_label_coordinates: Assumes an 1D array of positions in data coordinates.
        axis_object: Matplotlib axis object
        closest_space: Distance that is considered to close, in axis coordinates.

    """  # noqa: E501
    # first make sure there are no exactly identical coordinates, as this might mess up the optimizer.
    unique_coords, counts = np.unique(original_label_coordinates, return_counts=True)
    if len(unique_coords) != len(original_label_coordinates):
        duplicates = unique_coords[counts > 1]
        for duplicate_value in duplicates:
            locations = np.where(original_label_coordinates == duplicate_value)[0]
            shift = closest_space / 10
            for index, loc in enumerate(locations):
                original_label_coordinates[loc] += (1 - (index % 2) * 2) * shift

    # convert data coordinates into axes coordinates:
    axis_median = []
    for median_value in original_label_coordinates:
        # convert into axes coord:
        ax_coord = _transform_func(axis_object, (median_value, 1.01))
        axis_median.append(ax_coord[0])

    initial_position = np.array(axis_median)
    bounds = tuple((0, 1) for _ in range(len(initial_position)))
    optimal_positions = minimize(
        _label_weight_func,
        initial_position,
        args=(initial_position, closest_space),
        bounds=bounds,
        method="Nelder-Mead",
    )
    adjusted_text_positions = optimal_positions.x

    return adjusted_text_positions


def _adding_horizontal_lines_and_values(  # noqa: PLR0913
    numbers_to_show: list | np.ndarray,
    text_positions: list | np.ndarray,
    median_coords: list | np.ndarray,
    y_positions: list | np.ndarray,
    unit: list[str] | str,
    axis_object: Any,
    show_logit_mean: bool,
    color_map: ListedColormap,
    color_label: bool,
    linestyle: str,
    linealpha: float,
    linewidth: float,
) -> None:
    """Helper function for CDF plot. Adds horizontal lines if one shows the logit mean."""
    unit_list = [unit] * len(numbers_to_show) if not isinstance(unit, list) else unit

    for index1, median_val in enumerate(numbers_to_show):
        value_label = f"{median_val:.2f} {unit_list[index1]}"

        axis_object.text(
            text_positions[index1],
            1.01,
            value_label,
            fontsize=8,
            verticalalignment="bottom",
            rotation=50,
            color=color_map(index1) if color_label else "black",
            transform=axis_object.transAxes,
        )
        if show_logit_mean:
            line_color = color_map(index1)
            # add horizontal lines to indicate probability value for logit mean
            # convert data points into coordinate points
            # prefer to use axhlines instead of hlines since it does not stretch the x-axis.
            point = (median_coords[index1], median_val)
            new_point = _transform_func(axis_object, point)
            axis_object.axhline(
                y_positions[index1],
                xmin=0,
                xmax=new_point[0],
                linestyle=linestyle,
                alpha=linealpha,
                zorder=0,
                color=line_color,
                linewidth=linewidth,
            )

    type_of_number = "logit mean" if show_logit_mean and not all(val == 0.5 for val in y_positions) else "median"

    axis_object.text(
        0.0,
        1.01,
        type_of_number,
        fontsize=8,
        verticalalignment="bottom",
        rotation=50,
        color="black",
        transform=axis_object.transAxes,
    )


def _forced_sorting(
    adjusted_text_positions: np.ndarray, store_median_values: np.ndarray | list, ascending_order: bool = True
) -> np.ndarray:
    """Helper function to sort the median/logit mean labels for CDF plot in ascending (descending) order.

    Quick optimization does not take ordering into account.
    """
    sorted_medians = np.sort(store_median_values)
    if not ascending_order:
        sorted_medians = np.flipud(sorted_medians)
    sorted_text_positions = np.sort(adjusted_text_positions)
    new_text_positions = np.zeros_like(sorted_text_positions)
    start_value = sorted_medians[0]
    helpful_index = 0
    for value, text_pos in zip(sorted_medians, sorted_text_positions):
        current_position = np.where(store_median_values == value)[0]
        new_position = current_position[0]
        if len(current_position) > 1:
            if start_value != value:
                start_value = value
                helpful_index = 0
            else:
                helpful_index += 1
                new_position = current_position[helpful_index]
        new_text_positions[new_position] = text_pos
    return new_text_positions


def _provide_xlabel(observation_type: str, unit: str | None):  # noqa: ANN202
    """Provide the correct x-label for the figure given observation type and its unit."""
    x_label = shorten_label(format_for_display(observation_type.split(".")[-1]))
    # Labels and settings
    if unit:
        x_label += f" ({unit})"

    return x_label


def _get_data_for_cdf_plot(
    observations: list[QPUPlotObservation],
    observation_types: list[str],
) -> tuple[dict[str, list[ObservationData]], dict[str, np.ndarray], dict[str, str], str]:
    """Collects data suitable for the CDF plot.

    Args:
        observations: The data to plot.
        observation_types: One cumulative distribution for each type.
        split_dut_field: How to extract component name and observation type from DUT field.

    Returns:
        Observations and data sorted by types.

    Raises:
        ValueError if no observation types are provided or units are mismatching.

    """
    obs_for_types: dict[str, list[ObservationData]] = {}  # an observation list for each obs type
    data_for_types: dict[str, np.ndarray] = {}  # the data values list for each obs type
    obs_for_types_list: list[ObservationData] = []  # combined list of all observations for all types, to check units
    device_for_types: dict[str, str] = {}  # device type for obs type

    for obs_type in observation_types:
        # list of observations that are of this observation type
        obs_list = _construct_observations_list(observations, obs_type)

        values = np.array([obs.observation.value for obs in obs_list])

        # fill out the dicts and lists
        obs_for_types[obs_type] = [obs.observation for obs in obs_list]  # type: ignore[misc]
        data_for_types[obs_type] = values
        obs_for_types_list += [obs.observation for obs in obs_list]  # type: ignore[misc]

        device_type = "Device"
        if "QB" in [obs.component for obs in obs_list]:
            device_type = "Qubit"
        elif "TC" in [obs.component for obs in obs_list]:
            device_type = "Coupler"
        device_for_types[obs_type] = device_type

    if not obs_for_types_list:
        raise ValueError(
            f"Could not find any of the observation types in the observations. {observations=}\n{observation_types=}"
        )

    # Check the units of the obs_types
    unit = check_same_unit(obs_for_types_list)  # type: ignore[arg-type]

    return obs_for_types, data_for_types, device_for_types, unit


def _prepare_data_for_multi_cdf(
    observations: list[QPUPlotObservation],
    observation_types: list[str],
    xscale: str,
    show_logit_mean: bool,
    color_scheme: ListedColormap,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    # Set up the data
    obs_for_types, data_for_types, device_for_types, old_unit = _get_data_for_cdf_plot(observations, observation_types)
    unit = old_unit
    plot_data: list[dict[str, Any]] = []

    use_new_units = xscale == "linear"
    for index, obs_type in enumerate(obs_for_types):
        data = data_for_types[obs_type]  # the data for the obs_type that we will plot
        # Calculate median value
        # we want the unscaled unit (old unit) and unscaled data (old data) to calculate the logit mean
        median_data, median_unit, y_min, median_coords = _calculate_median_or_logit_mean(
            data, show_logit_mean, use_new_units, obs_type, old_unit, n_bins=1000
        )
        # the summary numbers (median or logit mean) are always scaled to better new units. Done at the end.
        if use_new_units:
            unit, data, _ = change_unit_magnitude(old_unit, data, obs_type)  # type: ignore[assignment]
            scaling_factor = get_scaling_from_unit(unit)
            median_coords /= scaling_factor
            # above locks the coordinates of the label to the scaling of the data and hence the x-axis

        # Create line_label in the form e.g. 'Qubit drive frequency'
        line_label = _get_label_from_device(
            device_type=device_for_types[obs_type], observation_name=f"{obs_type.replace('.', ' ')}"
        )
        plot_data.append(
            {
                "data": {"value": data, "unit": unit},
                "line_label": line_label,
                "median": {"value": median_data, "unit": median_unit},
                "y_min": y_min,
                "median_coords": median_coords,
                "line_color": color_scheme(index),
            }
        )

    plot_data = _check_units_are_scaled_identically(plot_data)
    x_label_unit = plot_data[0]["data"]["unit"]  # previous line should have enforced all data to be of the same unit
    x_label = _provide_xlabel(list(obs_for_types.keys())[-1], x_label_unit)

    return plot_data, {"unit": unit, "x_label": x_label}


def _check_units_are_scaled_identically(plot_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Make sure the final data shown on the CDF plot all have the same scaling applied to their units.

    If they are different, the largest unit takes precedence. This will also affect the location of the
    labels for the median / logit mean values, since these must be tightly related to the x-axis scaling. This function
    ensures this is the case as well.
    """
    data_name = "data"
    units_in_data = [data[data_name]["unit"] for data in plot_data]

    if len(set(units_in_data)) != 1:
        # need to change them back to the same unit.
        # first symbol in the string should be the prefix
        scaling = []
        old_units = []
        for unit in units_in_data:
            scaling_number = get_scaling_from_unit(unit)
            old_units.append(unit)
            scaling.append(scaling_number)
        # largest number dictates the scaling for all others
        largest_scaling = np.max(scaling)
        new_common_unit = old_units[scaling.index(largest_scaling)]
        largest_exp = get_exponent(largest_scaling)
        rescale_exp = np.round([get_exponent(num) - largest_exp for num in scaling])
        for new_exp, data in zip(rescale_exp, plot_data):
            new_scale_factor = 10.0**new_exp
            data[data_name]["unit"] = new_common_unit
            data[data_name]["value"] = data[data_name]["value"] * new_scale_factor
            data["median_coords"] = data["median_coords"] * new_scale_factor
    return plot_data
