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
"""Methods to show observations over time."""

from collections.abc import Callable, Iterable
from datetime import datetime
from pathlib import Path
import re
from typing import Any

import matplotlib as mpl
import matplotlib.dates as md
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

from exa.core.observation import LocalObservation
from exa.core.util.dut_field_decoder import decode_dut_field
from iqm.station_control.interface.models import ObservationData

from .plots import _construct_observations_list, format_for_display
from .qpu_core.colormaps import IQM_COLORMAP
from .qpu_core.create_markers import PlotMarkers
from .qpu_core.utils import QPUPlotObservation, change_unit_magnitude, sort_components


def plot_time_series(
    observations: list[LocalObservation | ObservationData],
    dut_fields: list[str],
    split_dut_field: Callable[[str], tuple[str, str]] = decode_dut_field,
    moving_average_number: int = 3,
    save_path: None | Path = None,
    resolution: int = 300,
    show_uncertainty: bool = True,
    plot_difference: bool = False,
) -> Figure:
    """Draw the time series for multiple DUT fields.

    Time-series - a line plot shows the values of a specific observation over time.

    Args:
        observations: The data to plot. These objects must have a ``created_timestamp``.
        dut_fields: Determines number of graphs.
        split_dut_field: Method to extract the component and observation parameter names from the DUT field.
        moving_average_number: Number of points for the moving average.
        save_path: Where to save the figure.
        resolution: In dpi for saving the figure.
        show_uncertainty: Error bars will be shown.
        plot_difference: Rather than absolute values.

    Returns:
        The Matplolib figure of the time series.

    """
    if any(
        isinstance(obs.value, str) or (isinstance(obs.value, np.ndarray) and np.issubdtype(obs.value.dtype, str))
        for obs in observations
    ):
        raise ValueError(f"Some values in {observations=} can't be plotted.")
    qpuplot_observations = []
    for obs in observations:
        comp_name, obs_type = split_dut_field(obs.dut_field)
        qpuplot_observations.append(QPUPlotObservation(component=comp_name, metric=obs_type, observation=obs))

    components_and_metrics = {split_dut_field(dut_field) for dut_field in dut_fields}

    return _plot_time_series_from_observations(
        qpuplot_observations,
        components_and_metrics,
        moving_average_number,
        save_path,
        resolution,
        show_uncertainty,
        plot_difference,
    )


def _plot_time_series_from_observations(
    observations: list[QPUPlotObservation],
    components_and_metrics: Iterable[tuple[str, str]],
    moving_average_number: int = 3,
    save_path: None | Path = None,
    resolution: int = 300,
    show_uncertainty: bool = True,
    plot_difference: bool = False,
) -> Figure:
    sorted_observations = _sort_observations(
        observations,
        components_and_metrics,
    )

    collection, show_uncertainty = _calculate_data_and_error(
        sorted_observations=sorted_observations,
        plot_difference=plot_difference,
        show_uncertainty=show_uncertainty,
    )

    return _plot_time_series(
        collection=collection,
        dut_field_count=len(list(components_and_metrics)),
        moving_average_number=moving_average_number,
        save_path=save_path,
        resolution=resolution,
        show_uncertainty=show_uncertainty,
        plot_difference=plot_difference,
    )


def _plot_time_series(
    collection: dict,
    dut_field_count: int,
    moving_average_number: int,
    save_path: None | Path,
    resolution: int,
    show_uncertainty: bool,
    plot_difference: bool,
) -> Figure:
    # Set up the figure axis and plot colors
    unique_units = list(set(collection["units"]))
    unique_units_count = len(unique_units)
    if unique_units_count > 2:
        fig, ax = plt.subplots(figsize=(12, 9), nrows=unique_units_count, sharex=True)
        plt.xticks(rotation=45)
        qualitative_colormap = mpl.colormaps["tab10"]
        n_colors = qualitative_colormap.N
        color_array = [qualitative_colormap(i % n_colors / n_colors) for i in range(dut_field_count)]
        axes_colors: list[Any] = ["black"] * unique_units_count
        ax[-1].set_xlabel("Time")
        axis_list = ax
    elif unique_units_count == 2:
        fig, ax = plt.subplots(figsize=(12, 6))
        plt.xticks(rotation=45)
        ax_twin = ax.twinx()
        n1_units = collection["units"].count(unique_units[0])
        n2_units = collection["units"].count(unique_units[1])
        red_vals = np.linspace(0.9, 0.5, n1_units)
        blue_vals = np.linspace(0.9, 0.5, n2_units)
        g1_colors = [mpl.colormaps["Reds"](i) for i in red_vals]
        g2_colors = [mpl.colormaps["Blues"](i) for i in blue_vals]
        axes_colors = [mpl.colormaps["Reds"](0.99), mpl.colormaps["Blues"](0.99)]
        color_array = g1_colors + g2_colors if unique_units[0] == collection["units"][0] else g2_colors + g1_colors
        ax.set_xlabel("Time")
        axis_list = [ax, ax_twin]
    else:
        fig, ax = plt.subplots(figsize=(12, 6))
        plt.xticks(rotation=45)
        axes_colors = ["black"]
        if dut_field_count > 1:
            qualitative_colormap = mpl.colormaps["tab10"]
            n_colors = qualitative_colormap.N
            color_array = [qualitative_colormap(i % n_colors / n_colors) for i in range(dut_field_count)]
        else:
            color_array = [IQM_COLORMAP(0.99)]
        ax.set_xlabel("Time")
        axis_list = [ax]
    collection["color_array"] = color_array

    if show_uncertainty:
        axis_list, y_axis_limits, timespan = _plot_data_with_uncertainty(
            collection,
            unique_units=unique_units,
            axes=axis_list,
            axes_colors=axes_colors,
            y_axis_limits=dict.fromkeys(unique_units, (0, 0)),
            moving_average_number=moving_average_number,
            plt_marker=PlotMarkers(),  # type: ignore[arg-type]
            plot_difference=plot_difference,
        )
    else:
        axis_list, y_axis_limits, timespan = _plot_data(
            collection,
            unique_units=unique_units,
            axes=axis_list,
            axes_colors=axes_colors,
            y_axis_limits=dict.fromkeys(unique_units, (0, 0)),
            moving_average_number=moving_average_number,
            plt_marker=PlotMarkers(),  # type: ignore[arg-type]
            plot_difference=plot_difference,
        )

    # Set y limits based on data and not error bars
    _set_y_limits(axis_list, unique_units, y_axis_limits)

    # Select time interval more intelligently if there are multiple data points,
    # else this code will cause the plot to run infinitely.
    if timespan["days"] + timespan["seconds"] > 0:
        axis_list[-1].xaxis.set_major_locator(
            _calculate_timespan_locator(
                timespan=timespan,
                max_ticks_count=14,
            )
        )

    # Put a legend to the right of the current axis
    for idx, ax1 in enumerate(axis_list):
        if len(axis_list) == 2:
            number_of_labels = len(collection["data_labels"])
            axis_label_y_position = 0.85 - number_of_labels * idx * 0.055
            ax1.legend(loc="center left", bbox_to_anchor=(1.10, axis_label_y_position))
        else:
            ax1.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=resolution)

    return fig


def _sort_observations(
    observations: list[QPUPlotObservation],
    components_and_metrics: Iterable[tuple[str, str]],
) -> list[list[QPUPlotObservation]]:
    """Sort DUT fields according to their components and observations by timestamp."""
    unique_parameters = {cap[1] for cap in components_and_metrics}

    sorted_caps: list[tuple[str, str]] = []
    for unique_obs in unique_parameters:
        components_to_sort = [comp for (comp, obs) in components_and_metrics if obs == unique_obs]
        sorted_names = sort_components(components_to_sort)
        sorted_caps.extend([(name, unique_obs) for name in sorted_names])

    sorted_observations: list[list[QPUPlotObservation]] = []
    for comp_name, obs_type in sorted_caps:
        # Exclude LocalObservations because they don't have a timestamp.
        obs_list = _construct_observations_list(
            observations, obs_type, comp_name, one_per_dut_field=False, sort_names=False
        )
        new_obs_list = sorted([obs.observation for obs in obs_list], key=lambda x: str(x.created_timestamp))
        if len(new_obs_list) <= 0:
            raise ValueError("No observations with this dut_field to plot.")
        sorted_observations.append(
            [QPUPlotObservation(component=comp_name, metric=obs_type, observation=new_obs) for new_obs in new_obs_list]
        )

    return sorted_observations


def _moving_average(arr: list[float], num: int, a_error: list[float] | None = None) -> np.ndarray:
    num = max(1, num)
    window_length = len(arr) - num + 1
    if a_error is None:
        means = [np.mean(arr[i : num + i]) for i in range(window_length)]
        return np.asarray(means, dtype=float)

    # Check if there are any remaining zeros in the error array:
    weights = 1 / np.array(a_error) ** 2
    multiply_with_weights = np.array(arr) * weights
    weighted_means = [
        np.sum(multiply_with_weights[i : num + i]) / np.sum(weights[i : num + i]) for i in range(window_length)
    ]
    return np.asarray(weighted_means, dtype=float)


def _moving_std(arr: list[float], num: int, a_error: list[float] | None = None) -> np.ndarray:
    num = max(1, num)
    window_length = len(arr) - num + 1
    if a_error is None:
        # the estimated error in the mean
        std_list = [np.std(arr[i : num + i], ddof=1) / np.sqrt(num) for i in range(window_length)]
        return np.asarray(std_list, dtype=float)

    weights = 1 / np.array(a_error) ** 2
    # estimated error of the weighted mean
    std_list = [np.sqrt(1 / np.sum(weights[i : num + i])) for i in range(window_length)]
    return np.asarray(std_list, dtype=float)


def _calculate_difference(data: list[float] | np.ndarray) -> np.ndarray:
    return np.asarray(data) - np.mean(data)


def _calculate_data_and_error(
    sorted_observations: list[list[QPUPlotObservation]],
    plot_difference: bool,
    show_uncertainty: bool,
) -> tuple[dict, bool]:
    data_collection: list[list[float] | np.ndarray] = []
    data_labels: list[str] = []
    data_error_collection: list[list[float | None]] = []
    timestamp_collection: list[list[str]] = []
    obs_type_collection: list[str] = []
    units: list[str] = []

    qubits = [
        observations[0].component for observations in sorted_observations if observations[0].component.startswith("QB")
    ]
    couplers = [
        observations[0].component for observations in sorted_observations if observations[0].component.startswith("TC")
    ]

    for obs_list in sorted_observations:
        data = np.asarray([obs.observation.value for obs in obs_list])
        data_error = [getattr(obs.observation, "uncertainty", None) for obs in obs_list]
        timestamps = [obs.observation.created_timestamp for obs in obs_list]

        unit: str = obs_list[0].observation.unit

        # To check that every value comes with an uncertainty:
        if show_uncertainty and all(v == 0 or v is None for v in data_error):
            show_uncertainty = False

        if plot_difference:
            data = _calculate_difference(data)

        if show_uncertainty:
            new_data_error = _adjust_errors(data_error) if any(v == 0 or v is None for v in data_error) else data_error
            unit, data, data_error = change_unit_magnitude(unit, data, obs_list[0].metric, new_data_error)  # type: ignore[assignment,arg-type]
            data_error_collection.append(data_error)
        else:
            unit, data, _ = change_unit_magnitude(unit, data, obs_list[0].metric)  # type: ignore[assignment]
        data_collection.append(data)
        units.append(unit)
        obs_type_collection.append(obs_list[0].metric)
        timestamp_collection.append(timestamps)  # type: ignore[arg-type]

        device_type = "Device"
        if obs_list[0].component in qubits:
            device_type = "Qubit"
            if re.sub(r"[0-9a-zA-Z]", "", obs_list[0].component):
                device_type = ""
        elif obs_list[0].component in couplers:
            device_type = "Tunable Coupler"

        data_labels.append(format_for_display(f"{device_type} {obs_list[0].component} {obs_list[0].metric}"))

    return (
        {
            "data_collection": data_collection,
            "data_labels": data_labels,
            "data_error": data_error_collection,
            "timestamp_collection": timestamp_collection,
            "obs_type_collection": obs_type_collection,
            "units": units,
        },
        show_uncertainty,
    )


def _adjust_errors(data_error: list) -> list[float]:
    """If just a few errors are zero or None, replace them with a tiny value."""
    non_zeros = list(filter(lambda a: a != 0 and a is not None, data_error))
    new_uncertainty = min(np.abs(non_zeros)) * 1e-9
    new_data_error: list[float] = [new_uncertainty if value == 0.0 or value is None else value for value in data_error]

    return new_data_error


def _set_y_limits(axis_list: list[plt.Axes], unique_units: list[str], y_axis_limits: dict[str, list[float]]) -> None:
    """Set y limits for each axis in the list."""
    for ax_object, unit in zip(axis_list, unique_units):
        min_pt, max_pt = y_axis_limits[unit]
        y_axis_space = abs(max_pt - min_pt) / 10
        if y_axis_space == 0:
            y_axis_space = max_pt * 0.1
        ax_object.set_ylim((min_pt - y_axis_space, max_pt + y_axis_space))


def _plot_data(
    collection: dict[str, list],
    unique_units: list[str],
    axes: list[plt.Axes],
    axes_colors: list[str],
    y_axis_limits: dict[str, tuple[float, float]],
    moving_average_number: int,
    plt_marker: mpl.markers.MarkerStyle,
    *,
    plot_difference: bool,
) -> tuple:
    longest_span_seconds = 0
    longest_span_days = 0

    for y_values, time_label, unit, obs_type, line_label, color in zip(
        collection["data_collection"],
        collection["timestamp_collection"],
        collection["units"],
        collection["obs_type_collection"],
        collection["data_labels"],
        collection["color_array"],
    ):
        axis_index = unique_units.index(unit)
        y_axis_limits[unit] = _adjust_axis_limits(axis_limits=y_axis_limits[unit], data=y_values)

        # Plot the time_series
        data_moving_average = _moving_average(y_values, num=moving_average_number)
        # Convert time stamps into date time objects

        # Some observations are saved as string or as datetime objects and must be treated differently
        if isinstance(time_label[0], datetime):
            time_object = time_label.copy()
        elif isinstance(time_label[0], str):
            time_object = [datetime.strptime(ti, "%Y-%m-%d %H:%M:%S.%f%z") for ti in time_label]
        else:
            raise Warning("time label is neither a datetime object nor string!")

        plt1 = axes[axis_index].plot(
            time_object,
            y_values,
            ls="",
            marker=plt_marker.get_marker(),
            color=color,
            alpha=0.7,
            label=line_label,
        )
        if moving_average_number > 1:
            axes[axis_index].plot(
                time_object[moving_average_number - 1 :],
                data_moving_average,
                color=plt1[0].get_color(),
                ls="-",
                lw=2.5,
                alpha=0.5,
                label=f"{line_label}: {moving_average_number}-period moving mean",
            )
            moving_error = _moving_std(y_values, num=moving_average_number)
            axes[axis_index].fill_between(
                time_object[moving_average_number - 1 :],
                data_moving_average + moving_error,
                data_moving_average - moving_error,
                color=plt1[0].get_color(),
                alpha=0.2,
            )
        else:
            axes[axis_index].plot(
                time_object[:],
                data_moving_average,
                color=plt1[0].get_color(),
                ls="-",
                lw=2.5,
                alpha=0.5,
            )

        # Select time interval more intelligently
        if len(time_object) > 1:
            duration = time_object[-1] - time_object[0]
            longest_span_days = max(longest_span_days, duration.days)
            longest_span_seconds = max(longest_span_seconds, duration.seconds)

        # Assuming no measurements stretches over a year, we can neglect labelling the year
        axes[axis_index].xaxis.set_major_formatter(md.DateFormatter("%d/%m %H:%M"))
        axes[axis_index].tick_params(axis="y", labelcolor=axes_colors[axis_index], color=axes_colors[axis_index])

        # Shrink current axis by 10%
        _shrink_axis(axes[axis_index])

        _set_y_label(axes[axis_index], obs_type, plot_difference, unit, axes_colors[axis_index])

    timespan = {
        "days": longest_span_days,
        "hours": longest_span_seconds // (60 * 60),
        "minutes": longest_span_seconds // 60,
        "seconds": longest_span_seconds,
    }

    return axes, y_axis_limits, timespan


def _shrink_axis(axis: plt.Axes) -> None:
    box = axis.get_position()
    axis.set_position((box.x0, box.y0, box.width * 0.9, box.height))


def _set_y_label(axis: plt.Axes, obs_type: str, plot_difference: bool, unit: str, axis_color: str) -> None:
    y_label_text = format_for_display(obs_type)
    if plot_difference:
        y_label_text = f"{y_label_text} [difference from mean]"
    if unit:
        y_label_text = f"{y_label_text} ({unit})"

    axis.set_ylabel(y_label_text, color=axis_color)


def _plot_data_with_uncertainty(
    collection: dict[str, list],
    unique_units: list[str],
    axes: list[plt.Axes],
    axes_colors: list[str],
    y_axis_limits: dict[str, tuple[float, float]],
    moving_average_number: int,
    plt_marker: mpl.markers.MarkerStyle,
    *,
    plot_difference: bool,
) -> tuple:
    longest_span_seconds = 0
    longest_span_days = 0

    for index, (y_values, time_label, unit, obs_type, line_label, color) in enumerate(
        zip(
            collection["data_collection"],
            collection["timestamp_collection"],
            collection["units"],
            collection["obs_type_collection"],
            collection["data_labels"],
            collection["color_array"],
        )
    ):
        axis_index = unique_units.index(unit)
        y_axis_limits[unit] = _adjust_axis_limits(axis_limits=y_axis_limits[unit], data=y_values)

        # Plot the time_series
        data_moving_average = _moving_average(
            y_values, num=moving_average_number, a_error=collection["data_error"][index]
        )

        # Convert time stamps into date time objects

        # Some observations are saved as string or as datetime objects and must be treated differently
        if isinstance(time_label[0], datetime):
            time_object = time_label.copy()
        elif isinstance(time_label[0], str):
            time_object = [datetime.strptime(ti, "%Y-%m-%d %H:%M:%S.%f%z") for ti in time_label]
        else:
            raise Warning("time label is neither a datetime object nor string!")

        plt1 = axes[axis_index].errorbar(
            time_object,
            y_values,
            yerr=collection["data_error"][index],
            ls="",
            marker=plt_marker.get_marker(),
            color=color,
            alpha=0.7,
            label=line_label,
            capsize=3,
        )

        if moving_average_number > 1:
            axes[axis_index].plot(
                time_object[moving_average_number - 1 :],
                data_moving_average,
                color=plt1[0].get_color(),
                ls="-",
                lw=2.5,
                alpha=0.5,
                label=f"{line_label}: {moving_average_number}-period moving mean",
            )
            moving_error = _moving_std(y_values, num=moving_average_number, a_error=collection["data_error"][index])

            axes[axis_index].fill_between(
                time_object[moving_average_number - 1 :],
                data_moving_average + moving_error,
                data_moving_average - moving_error,
                color=plt1[0].get_color(),
                alpha=0.2,
            )
        else:
            axes[axis_index].plot(
                time_object[:],
                data_moving_average,
                color=plt1[0].get_color(),
                ls="-",
                lw=2.5,
                alpha=0.5,
            )

        # Select time interval more intelligently
        if len(time_object) > 1:
            duration = time_object[-1] - time_object[0]
            longest_span_days = max(longest_span_days, duration.days)
            longest_span_seconds = max(longest_span_seconds, duration.seconds)

        # Assuming no measurements stretches over a year, we can neglect labelling the year
        axes[axis_index].xaxis.set_major_formatter(md.DateFormatter("%d/%m %H:%M"))
        axes[axis_index].tick_params(axis="y", labelcolor=axes_colors[axis_index], color=axes_colors[axis_index])

        # Shrink current axis by 10%
        _shrink_axis(axes[axis_index])

        _set_y_label(axes[axis_index], obs_type, plot_difference, unit, axes_colors[axis_index])

    timespan = {
        "days": longest_span_days,
        "hours": longest_span_seconds // (60 * 60),
        "minutes": longest_span_seconds // 60,
        "seconds": longest_span_seconds,
    }

    return axes, y_axis_limits, timespan


def _adjust_axis_limits(axis_limits: tuple[float, float], data: list[float]) -> tuple[float, float]:
    """Adjust existing axis limits if zero or within data range."""
    minimum, maximum = axis_limits
    minimum = np.min(data) if minimum == 0 else min(minimum, np.min(data))
    maximum = np.max(data) if maximum == 0 else max(maximum, np.max(data))

    return (minimum, maximum)


def _calculate_timespan_locator(timespan: dict[str, float], max_ticks_count: int) -> md.RRuleLocator:
    if timespan["days"] > 2:
        day_interval = max(1, timespan["days"] // max_ticks_count)

        return md.DayLocator(interval=day_interval)

    if timespan["hours"] > 2:
        hour_interval = max(1, timespan["hours"] // max_ticks_count)

        return md.HourLocator(interval=hour_interval)

    if timespan["minutes"] > 2:
        if timespan["minutes"] // (max_ticks_count * 10) > 1:
            minute_interval = timespan["minutes"] // max_ticks_count
        else:
            minute_interval = 10

        return md.MinuteLocator(interval=minute_interval)

    if timespan["seconds"] // (max_ticks_count * 10) > 1:
        second_interval = timespan["seconds"] // max_ticks_count
    else:
        second_interval = 10

    return md.SecondLocator(interval=second_interval)
