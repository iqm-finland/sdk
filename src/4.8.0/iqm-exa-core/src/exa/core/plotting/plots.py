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
"""Various plotting methods for QPUPlot and standalone."""

from collections.abc import Callable
import logging
import math
from pathlib import Path
import re
from typing import Any
import warnings

from matplotlib import colormaps
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from exa.common.helpers import format_deprecated
from exa.common.qcm_data.chad_model import CHAD
from exa.core.observation import LocalObservation
from exa.core.util.dut_field_decoder import decode_dut_field
from iqm.station_control.interface.models import ObservationData, ObservationDefinition

from .qpu_core.colormaps import IQM_COLORMAP
from .qpu_core.component_styles import ComponentColors
from .qpu_core.utils import (
    QPUPlotObservation,
    change_unit_magnitude,
    check_same_unit,
    sort_components,
    split_qubits_into_low_high,
    split_qubits_into_low_high_frequency,
)

logger = logging.getLogger(__name__)

STANDARD_NAMES = [
    "CZ",
    "IRB",
    "PRX",
    "TGSS",
    "RB",
    "GHZ",
    "SSRO",
    "QB",
    "TC",
    "Clifford",
    "Gaussian",
    "DRAG",
    "VZ",
    "CPhase",
    "CRF",
    "RZ",
]


def _check_invalid_observations(observations):  # noqa: ANN001, ANN202
    if any(
        isinstance(obs.value, str) or (isinstance(obs.value, np.ndarray) and np.issubdtype(obs.value.dtype, str))
        for obs in observations
    ):
        raise ValueError(f"Some values in {observations=} can't be plotted.")


def _get_label_from_device(device_type: str, observation_name: str) -> str:
    if device_type[1:] in observation_name:
        return observation_name
    if "readout" in observation_name:
        return observation_name
    if "gate" in observation_name:
        return observation_name
    if "clifford" in observation_name:
        return observation_name

    return f"{device_type} {observation_name}"


def plot_bar_chart(
    observations: list[LocalObservation | ObservationData],
    observation_type: str,
    split_dut_field: Callable[[str], tuple[str, str]] = decode_dut_field,
    save_path: Path | None = None,
    one_per_dut_field: bool = True,
    resolution: int = 300,
) -> Figure:
    """Draw a bar chart for the latest observations of a given type.

    Bar chart - a bar shows an observation value for each component
    This has multiple functionalities depending on the number of values of the same type provided.
    - If the number of values is exactly one, a bar chart is plotted for the observations of the given observation_type.
    - If the number of values is larger than one a box-and-whiskers plot is drawn.

    Args:
        observations: The data to plot. The objects must have a ``created_timestamp``.
        observation_type: The type to plot.
        split_dut_field: How to extract component name and observation type from DUT field.
        save_path: Where to save the figure.
        one_per_dut_field: Only take the latest observation per DUT field.
        resolution: In dpi for saving the figure.

    Returns:
        The Matplotlib figure containing the plot.

    Raises:
        ValueError if observation type can't be found in observations.

    """
    _check_invalid_observations(observations)
    qpuplot_observations = []
    for obs in observations:
        comp_name, obs_type = split_dut_field(obs.dut_field)
        qpuplot_observations.append(QPUPlotObservation(component=comp_name, metric=obs_type, observation=obs))

    data_values, unique_components, plot_labels = _prepare_data_for_bar_chart(
        qpuplot_observations, observation_type, one_per_dut_field
    )

    return _plot_bar_chart(
        data_values=data_values,
        unique_components=unique_components,
        unit=plot_labels["unit"],
        line_label=plot_labels["line_label"],
        save_path=save_path,
        resolution=resolution,
    )


def _prepare_data_for_bar_chart(
    observations: list[QPUPlotObservation], observation_type: str, one_per_dut_field: bool = True
) -> tuple[list[list[float]], list[str], dict[str, str]]:
    obs_list = _construct_observations_list(observations, observation_type, one_per_dut_field=one_per_dut_field)
    if not obs_list:
        raise ValueError(
            f"Could not find the observation type in the observations. "
            f"{[obs.observation for obs in observations]}\n{observation_type=}"
        )
    data = [obs.observation.value for obs in obs_list]
    components = [obs.component for obs in obs_list]
    unit = obs_list[0].observation.unit
    obs_type = obs_list[0].metric
    unit, data, _ = change_unit_magnitude(unit, data, obs_type)  # type: ignore[assignment,arg-type]

    device_type = "Device"
    if "QB" in obs_list[0].component:
        device_type = "Qubit"
    elif "TC" in obs_list[0].component:
        device_type = "Coupler"

    # Create line_label in the form e.g. 'Qubit drive frequency'
    line_label = format_for_display(f"{device_type} {obs_type.replace('.', ' ')}")

    unique_components = sort_components(list(set(components)))
    unique_component_count = len(unique_components)

    data_values: list[list[float]] = [[] for _ in range(unique_component_count)]
    for value, component in zip(data, components):
        if component in unique_components:
            position = unique_components.index(component)
            data_values[position].append(value)  # type: ignore[arg-type]
    return data_values, unique_components, {"unit": unit, "line_label": line_label}


def _plot_bar_chart(
    data_values: list[list[float]],
    unique_components: list[str],
    unit: str,
    line_label: str,
    save_path: Path | None,
    resolution: int,
) -> Figure:
    unique_component_count = len(unique_components)

    # Check length of all sub lists in data_values
    number_of_values = [len(sublist) for sublist in data_values]

    color_array = [IQM_COLORMAP(0.9)]
    if unique_component_count != 1:
        color_array = [IQM_COLORMAP(i / (unique_component_count - 1)) for i in range(unique_component_count)]

    fig, ax = plt.subplots(figsize=(5 + int(unique_component_count / 2.5), 3 + int(unique_component_count / 5)))

    if max(number_of_values) == 1:
        # Plot the bar chart
        flatten_data_list = [item for sublist in data_values for item in sublist]
        ax.bar(unique_components, flatten_data_list, color=color_array)
    else:
        # Plotting box (and whiskers) plot, using the same definitions as Matplotlib
        bplot = ax.boxplot(
            data_values,
            label=unique_components,
            showmeans=True,
            vert=True,
            patch_artist=True,
            meanprops={"marker": "^", "mfc": "k", "mec": "k"},
        )
        # Fill with colors
        for patch, color in zip(bplot["boxes"], color_array):
            face_color = list(color)
            face_color[-1] = 0.5
            box_color = tuple(face_color)
            patch.set_facecolor(box_color)
            patch.set_edgecolor(color)
            patch.set_linewidth(1.8)

        plt.setp(bplot["medians"], color="black")

        for mean_pt in bplot["means"]:
            plt.setp(mean_pt, color="black")

        for index, (whisker, cap) in enumerate(zip(bplot["whiskers"], bplot["caps"])):
            plt.setp(whisker, color=color_array[index // 2])
            plt.setp(cap, color=color_array[index // 2])

        for flier in bplot["fliers"]:
            plt.setp(flier, markeredgecolor="grey")

    ax.set_xlabel("Component")
    if unit:
        line_label = f"{line_label} ({unit})"
    ax.set_ylabel(line_label)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=resolution)

    return fig


def plot_histogram(  # noqa: PLR0913
    observations: list[ObservationDefinition],
    observation_type: str,
    chad: CHAD | None = None,
    split_dut_field: Callable[[str], tuple[str, str]] = decode_dut_field,
    save_path: Path | None = None,
    split_qubits: bool = False,
    one_per_dut_field: bool = True,
    label_all_components: bool = False,
    resolution: int = 300,
    qpu_type: str | None = None,
    *,
    chip_design_record: dict[str, Any] | None = None,
) -> Figure:
    """Plot a histogram for observations of the given type.

    Adds a vertical line indicating the median of the distribution.

    Args:
        observations: The data to plot.
        observation_type: The type of data to plot.
        chad: The chip architecture definition (CHAD). Deprecated, use ``chip_design_record`` instead.
        split_dut_field: How to extract component name and observation type from DUT field.
        save_path: Where to save the figure.
        split_qubits: If high and low frequency qubits are present.
        one_per_dut_field: Only take the latest observation.
        label_all_components: Each component has its own label.
        resolution: In dpi for saving the figure.
        qpu_type: The type of the QPU that is used to split qubits into high and low frequency qubits, if applicable.
        chip_design_record: The chip design record.

    Returns:
        The Matplotlib figure containing the plot.

    Raises:
        ValueError: Observation type can't be found in ``observations``, units are mismatching, or both ``chad`` and
            ``chip_design_record`` are given.

    """
    if chad is not None:
        warnings.warn(
            format_deprecated(old="`chad`", new="`chip_design_record`", since="2026-09-17"),
            DeprecationWarning,
            stacklevel=2,
        )

    if chip_design_record is not None:
        if chad is not None:
            raise ValueError("Only one of `chad` or `chip_design_record` may be given.")
        return _plot_histogram_for_chad(
            observations=observations,
            observation_type=observation_type,
            chip_design_record=chip_design_record,
            split_dut_field=split_dut_field,
            save_path=save_path,
            split_qubits=split_qubits,
            one_per_dut_field=one_per_dut_field,
            label_all_components=label_all_components,
            resolution=resolution,
        )

    if chad is None:
        raise ValueError("Either `chad` or `chip_design_record` must be given.")

    return _plot_histogram_for_chad(
        observations=observations,
        observation_type=observation_type,
        split_dut_field=split_dut_field,
        save_path=save_path,
        split_qubits=split_qubits,
        one_per_dut_field=one_per_dut_field,
        label_all_components=label_all_components,
        resolution=resolution,
        chad=chad,
        qpu_type=qpu_type,
    )


def _plot_histogram_for_chad(  # noqa: PLR0913
    observations: list[ObservationDefinition],
    observation_type: str,
    split_dut_field: Callable[[str], tuple[str, str]],
    save_path: Path | None,
    split_qubits: bool,
    one_per_dut_field: bool,
    label_all_components: bool,
    resolution: int,
    chip_design_record: dict[str, Any] | None = None,
    chad: CHAD | None = None,
    qpu_type: str | None = None,
) -> Figure:
    """Shared implementation for plotting a histogram from an already parsed CHAD."""
    _check_invalid_observations(observations)
    qpuplot_observations = []
    for obs in observations:
        comp_name, obs_type = split_dut_field(obs.dut_field)
        qpuplot_observations.append(QPUPlotObservation(component=comp_name, metric=obs_type, observation=obs))  # type: ignore[arg-type]

    data_values, median_values, label_array, color_array, plot_labels = _prepare_data_for_histogram(
        observations=qpuplot_observations,
        observation_type=observation_type,
        one_per_dut_field=one_per_dut_field,
        label_all_components=label_all_components,
        split_qubits=split_qubits,
        chip_design_record=chip_design_record,
        chad=chad,
        qpu_type=qpu_type,
    )

    return _plot_histogram(
        data_values=data_values,
        median_values=median_values,
        label_array=label_array,
        color_array=color_array,
        unit=plot_labels["unit"],
        line_label=plot_labels["line_label"],
        save_path=save_path,
        resolution=resolution,
    )


def _prepare_data_for_histogram(
    observations: list[QPUPlotObservation],
    observation_type: str,
    one_per_dut_field: bool = True,
    label_all_components: bool = False,
    split_qubits: bool = False,
    chip_design_record: dict[str, Any] | None = None,
    chad: CHAD | None = None,
    qpu_type: str | None = None,
) -> tuple[list[list[float]], list, list[str], list[tuple], dict[str, str]]:
    """Format data so it is plotable as histogram.

    Args:
        observations: The data to plot.
        observation_type: The type of data to plot.
        one_per_dut_field: Only take the latest observation.
        label_all_components: Each component has its own label.
        split_qubits: If high and low frequency qubits are present.
        chip_design_record: Used to split qubits by design frequency.
        chad: The chip architecture definition (CHAD). Ignored if ``chip_design_record`` is given.
        qpu_type: The type of the QPU that is used to split qubits into high and low frequency qubits, if applicable.

    Returns:
        data values, median values, label array, color _array, and mapping for unit and line label.

    """
    obs_list = _construct_observations_list(observations, observation_type, one_per_dut_field=one_per_dut_field)
    if not obs_list:
        raise ValueError(
            f"Could not find the observation type in the observations. "
            f"{[obs.observation for obs in observations]}\n{observation_type=}"
        )

    # Check the units of the obs_types
    check_same_unit([obs.observation for obs in obs_list])

    data: list[float] = [float(obs.observation.value) for obs in obs_list]  # type: ignore[arg-type]
    components = [obs.component for obs in obs_list]

    unit = obs_list[0].observation.unit
    obs_type = obs_list[0].metric
    unit, data, _ = change_unit_magnitude(unit, data, obs_type)  # type: ignore[assignment,arg-type]

    # Create line_label in the form e.g. 'Qubit drive frequency'
    line_label = format_for_display(label=f"Device {obs_type.replace('.', ' ')}")

    # Sort components into couplers, high frequency qubits or low frequency qubits
    # splitting into high and low frequency qubits works only well for frequency where the distribution
    # is expected to be bimodal. For other values it may look messy
    data_values, median_values, label_array, color_array = _label_components(
        data=data,
        components=components,
        label_all_components=label_all_components,
        split_qubits=split_qubits,
        chip_design_record=chip_design_record,
        chad=chad,
        qpu_type=qpu_type,
    )
    return data_values, median_values, label_array, color_array, {"unit": unit, "line_label": line_label}


def _plot_histogram(
    data_values: list[list[float]],
    median_values: list,
    label_array: list,
    color_array: list,
    unit: str,
    line_label: str,
    save_path: Path | None,
    resolution: int,
) -> Figure:
    """Create histogram from formatted data.

    Args:
        data_values: The data to plot.
        median_values: Median to plot
        label_array: The labels for the data values.
        color_array: The colors for the dat values.
        unit: The unit of the values.
        line_label: The label for the vertical line.
        save_path: Where to save the figure.
        resolution: In dpi for saving the figure.

    Returns:
        The Matplotlib figure containing the plot.

    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    bins = _calculate_bins(data_values=data_values).tolist()

    for data_value, label, color in zip(data_values, label_array, color_array):
        if len(data_value) > 0:
            ax.hist(data_value, bins, alpha=0.3, label=label, color=color, stacked=True)
            ax.hist(
                data_value,
                bins,
                alpha=1.0,
                facecolor="None",
                stacked=True,
                edgecolor=color,
                linewidth=1.7,
                linestyle="-",
            )

    # Indicate median value on plot as vertical line
    for index, median_val in enumerate(median_values):
        ax.axvline(median_val, alpha=1, linestyle="-.", color=color_array[index])

    title_string = f"Median values ({unit})\n" if unit else "Median values\n"
    tracker = 1
    for label, value in zip(label_array, median_values):
        if not math.isnan(value):
            title_string += f"{label}: {value:4.2f}"
            if tracker < len(label_array):
                if tracker % 6 == 0:
                    title_string += "\n"
                else:
                    title_string += ", "
            tracker += 1

    ax.set_title(title_string)

    if unit:
        ax.set_xlabel(f"{line_label} ({unit})")
    else:
        ax.set_xlabel(line_label)
    ax.set_ylabel("Counts")
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=resolution)

    return fig


def _label_components(  # noqa: PLR0912
    data: list[float] | np.ndarray,
    components: list[str],
    label_all_components: bool,
    split_qubits: bool,
    chip_design_record: dict[str, Any] | None = None,
    chad: CHAD | None = None,
    qpu_type: str | None = None,
) -> tuple[list[list[float]], list, list[str], list[tuple]]:
    """Labels data for histogram plot.

    Uses either ``chip_design_record``, or ``chad`` and ``qpu_type`` to split qubits into high and low frequency.

    Args:
        data: The data to plot.
        components: The components to plot for.
        label_all_components: Each component has its own label.
        split_qubits: If high and low frequency qubits are present.
        chip_design_record: Used to split qubits by design frequency.
        chad: The chip architecture definition (CHAD).
        qpu_type: The type of the QPU that is used to split qubits into high and low frequency qubits.

    Returns:
        The data values, the median values, the label array, and the color array.

    """
    if label_all_components is False:
        coupler_values: list[float] = []

        if split_qubits is True:
            color_array = [ComponentColors.LOFREQ.value, ComponentColors.HIFREQ.value]
            label_array = ["low freq. qubits", "high freq. qubits"]
            if chip_design_record is not None:
                low_freq_qbs, high_freq_qbs = split_qubits_into_low_high_frequency(chip_design_record)
            else:
                if chad is None or qpu_type is None:
                    raise ValueError("Either `chip_design_record` or `chad` and `qpu_type` must be given.")
                low_freq_qbs, high_freq_qbs = split_qubits_into_low_high(chad, qpu_type)

            high_freq_values: list[float] = []
            low_freq_values: list[float] = []
            for value, component in zip(data, components):
                if "TC" in component:
                    coupler_values.append(value)
                elif component in high_freq_qbs:
                    high_freq_values.append(value)
                elif component in low_freq_qbs:
                    low_freq_values.append(value)

            qubit_values: list[list[float]] = [low_freq_values, high_freq_values]

        else:
            qubit_values = [[]]
            color_array = [ComponentColors.QUBIT.value]
            label_array = ["qubits"]
            for value, component in zip(data, components):
                if "TC" in component:
                    coupler_values.append(value)
                else:
                    qubit_values[0].append(value)

        median_qubit_values = [np.median(values) for values in qubit_values]
        data_values: list[list[float]] = qubit_values
        median_values = median_qubit_values

        if len(coupler_values) > 0:
            median_cpl_value = np.median(coupler_values)
            data_values = [*qubit_values, coupler_values]
            median_values = [*median_qubit_values, median_cpl_value]
            color_array += [ComponentColors.COUPLER.value]
            label_array += ["couplers"]

    else:
        nonsorted_unique_components = list(set(components))
        component_count = len(nonsorted_unique_components)
        # Attempt to sort components
        unique_components = sort_components(nonsorted_unique_components)

        if component_count == 1:
            color_array = [IQM_COLORMAP(0.9)]  # type: ignore[list-item]
        elif component_count < 4:
            color_array = [IQM_COLORMAP(i / (component_count - 1)) for i in range(component_count)]  # type: ignore[misc]
        else:
            color_array = [colormaps["turbo"](i / (component_count - 1)) for i in range(component_count)]  # type: ignore[misc]

        data_values = [[]] * component_count
        for value, component in zip(data, components):
            if component in unique_components:
                position = unique_components.index(component)
                data_values[position].append(value)

        median_values = [np.median(values) for values in data_values]
        label_array = unique_components

    return data_values, median_values, label_array, color_array


def _calculate_bins(data_values: list[list[float]]) -> np.ndarray:
    stacked_array: list[float] = []
    n_bars_array: list[int] = []
    measured_mean = [np.mean(list_to_stack) for list_to_stack in data_values]
    measured_width: list[np.ndarray] = []
    for list_to_stack in data_values:
        stacked_array += list_to_stack
        width = np.std(np.array(list_to_stack))
        measured_width.append(width)
        bin_width = 3.49 * width / (len(list_to_stack) ** (1 / 3))
        n_bars_selection = int(np.ceil((max(list_to_stack) - min(list_to_stack)) / bin_width)) if bin_width > 0 else 1
        n_bars_array.append(n_bars_selection)

    # Estimate data overlap
    lower_data_point = [mean_i - std_i for mean_i, std_i in zip(measured_mean, measured_width)]
    lowest_point = min(lower_data_point)
    upper_data_point = [mean_i + std_i for mean_i, std_i in zip(measured_mean, measured_width)]
    highest_point = min(upper_data_point)

    if lower_data_point.index(lowest_point) == upper_data_point.index(highest_point):
        n_bars = n_bars_array[upper_data_point.index(highest_point)]
    else:
        overlap = (
            upper_data_point[lower_data_point.index(lowest_point)]
            - lower_data_point[upper_data_point.index(highest_point)]
        )
        fractional_overlap = overlap / (2 * measured_width[lower_data_point.index(lowest_point)])
        n_bins_lower = n_bars_array[lower_data_point.index(lowest_point)]
        n_bins_higher = n_bars_array[upper_data_point.index(highest_point)]
        if 0 < fractional_overlap < 1:
            n_bins_lower = n_bins_lower * (1 - fractional_overlap)
            n_bars = int(np.ceil(n_bins_higher + n_bins_lower))
        elif fractional_overlap >= 1:
            n_bars = n_bins_higher
        elif fractional_overlap == 0:
            n_bars = int(np.ceil(n_bins_higher + n_bins_lower))
        else:
            n_bars = int(np.ceil(n_bins_higher + n_bins_lower * (1 + abs(fractional_overlap))))

    n_bars += 2
    return np.histogram(stacked_array, bins=n_bars)[1]  # get the bin edges


def _construct_observations_list(
    observations: list[QPUPlotObservation],
    observation_type: str | None = None,
    component_name: str | None = None,
    tag: str | None = None,
    one_per_dut_field: bool = True,
    sort_names: bool = True,
) -> list[QPUPlotObservation]:
    """Create a list of observations to be plotted.

    Args:
        observations: original list of QPUPlot observations. Our result will be a sublist of this.
        observation_type: only observations of this type will be included.
        component_name: only observations from this component will be included.
        tag: only observations with the given tag will be included.
        one_per_dut_field: if True, we take only the latest observation per dut field.
            Otherwise, we allow multiple observations per dut_field.
        sort_names: if True, the observations are sorted by component name.

    Returns:
        The list of component names, parameter names, and observations.

    """
    obs_list: list[QPUPlotObservation] = []
    qubits = [obs.component for obs in observations if obs.component.startswith("QB")]
    couplers = [obs.component for obs in observations if obs.component.startswith("TC")]
    for obs in observations:
        if observation_type is not None and obs.metric != observation_type:  # if not the right obs_type...
            continue  # don't add
        if component_name is not None and obs.component != component_name:  # if not the right component...
            continue  # don't add
        if (
            tag is not None and tag not in obs.tags  # type: ignore[attr-defined]
        ):  # if obs doesn't have the tag we want...
            continue  # don't add
        if one_per_dut_field:
            # We want the obs with the newer timestamp. If either of them is a LocalObservation, we can't compare,
            # so we take the local one. If both are local, we pick the new one.
            obs_with_dut_field = [
                ob.observation for ob in obs_list if ob.observation.dut_field == obs.observation.dut_field
            ]
            if len(obs_with_dut_field) > 0:  # if we already have an obs with this dut_field...
                obs_2 = obs_with_dut_field[0]  # this list should only have 1 element in the first place
                obs_timestamp = obs.observation.created_timestamp
                obs_2_timestamp = obs.observation.created_timestamp
                if obs_timestamp is not None and obs_2_timestamp is not None:
                    if str(obs_timestamp) > str(obs_2_timestamp):
                        obs_list.remove(obs_2)  # type: ignore[arg-type]
                        obs_list.append(obs)
                        continue
                    continue
                obs_list.remove(obs_2)
                obs_list.append(obs)
                continue

        obs_list.append(obs)

    if sort_names:
        qb_list = [obs for obs in obs_list if obs.component in qubits]
        qb_list = sorted(qb_list, key=lambda x: int(x.component.split("QB")[1]))

        tc_list = [obs for obs in obs_list if obs.component in couplers]
        tc_differentiator = "-"
        if len(tc_list) > 0 and "-" not in tc_list[0].component:
            tc_differentiator = "TC"
        tc_list = sorted(tc_list, key=lambda x: int(x.component.split(tc_differentiator)[1]))

        obs_list = qb_list + tc_list

    return obs_list


def format_for_display(label: str) -> str:
    """Capitalize first character and replace underscores and dots with spaces.

    Args:
        label: the unformatted label.

    Returns:
        The formatted label.

    """
    new_label = label.capitalize().replace("_", " ").replace(".", " ")
    for name in STANDARD_NAMES:
        new_label = re.sub(name, name, new_label, flags=re.IGNORECASE)
    return new_label
