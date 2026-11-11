# Copyright 2024 IQM Benchmarks developers
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

"""Plotting and elapsed-time helpers for the CLOPS benchmarks.

The pulse-level compilation and execution helpers live in :mod:`iqm.benchmarks.pulse_utils`.
"""

from datetime import datetime
from typing import Any

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

MAX_TAB10_SEGMENTS = 10
"""Largest number of waterfall segments that still fits in the 10-colour qualitative colormap."""


def plot_times(  # noqa: PLR0915
    clops_data: xr.Dataset, observations: dict[int, dict[str, dict[str, float]]]
) -> tuple[str, Figure]:
    """Generate a figure representing the different elapsed times in the CLOPS experiment.

    Args:
        clops_data (xr.Dataset): The dataset including elapsed time data from the CLOPS experiment
        observations (Dict[int, Dict[str, Dict[str, float]]]): The observations from the analysis of the
            CLOPS experiment.

    Returns:
        str: the name of the figure.
        Figure: the figure.

    """
    # Define the keys for different categories of times
    # Include validation and fetch_calibration totals only if they were computed
    job_keys = [
        k
        for k in [
            "validation_total",
            "fetch_calibration_total",
            "compile_total",
            "execution_total",
            "post_processing_total",
        ]
        if k in clops_data.attrs
    ]
    total_keys = ["job_total"]
    # Include the parameter assignment total only if the benchmark computed one
    user_keys = [
        k
        for k in [
            "user_retrieve_total",
            "user_submit_total",
            "assign_parameters_total",
            # Client-side pulse compilation, set only by the pulse-level benchmarks. Without it the
            # client-side bar omits what is usually their dominant cost.
            "time_compile",
            "time_transpile",
        ]
        if k in clops_data.attrs or k in observations[1]
    ]

    # Define variables for dataset values
    qubits = clops_data.attrs["qubits"]
    num_qubits = clops_data.attrs["num_qubits"]
    backend_name = clops_data.attrs["backend_name"]
    execution_timestamp = clops_data.attrs["execution_timestamp"]
    clops_time = float(clops_data.attrs["clops_time"])

    all_data = {}
    all_data.update(clops_data.attrs)
    all_data.update(observations[1])

    # Define colors
    cmap = plt.colormaps["winter"]
    colors = [cmap(i) for i in np.linspace(0, 1, len(job_keys) + len(user_keys) + len(total_keys) + 1)]

    # Plotting parameters
    fontsize = 6
    sep = 1
    barsize = 1
    alpha = 0.4

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    fig_name = f"{num_qubits}_qubits_{tuple(qubits)}"

    def annotate_bar(
        ax: plt.Axes,
        bar_bottom: float,
        bar_top: float,
        x_pos: float,
        label: str,
        arrow_annotations: list,
        min_bar_fraction: float = 0.03,
    ) -> None:
        """Place label in center of bar if tall enough, otherwise queue an arrow annotation.

        Args:
            ax: The matplotlib axes to annotate.
            bar_bottom (float): The bottom y-value of the bar segment.
            bar_top (float): The top y-value of the bar segment.
            x_pos (float): The x-position of the bar.
            label (str): The text label.
            arrow_annotations (list): List to collect (bar_mid_y, x_pos, label) for deferred arrow placement.
            min_bar_fraction (float): Minimum fraction of total axis height for label to fit inside bar.

        """
        bar_height = bar_top - bar_bottom
        bar_mid = bar_bottom + bar_height / 2.0
        y_min, y_max = ax.get_ylim()
        axis_range = y_max - y_min if y_max != y_min else 1.0
        if bar_height / axis_range >= min_bar_fraction:
            ax.text(x_pos, bar_mid, label, ha="center", va="center", fontsize=fontsize, color="black", zorder=10)
        else:
            arrow_annotations.append((bar_mid, x_pos, label))

    def place_arrow_annotations(ax: plt.Axes, arrow_annotations: list, x_offset: float = 1.2) -> None:
        """Place arrow annotations for bars that were too small to contain their label.

        Labels are spread vertically to avoid overlap.

        Args:
            ax: The matplotlib axes to annotate.
            arrow_annotations (list): List of (bar_mid_y, x_pos, label) tuples.
            x_offset (float): Horizontal offset from the bar center for the annotation text.

        """
        if not arrow_annotations:
            return
        y_min, y_max = ax.get_ylim()
        axis_range = y_max - y_min if y_max != y_min else 1.0
        min_spacing = axis_range * 0.05

        # Sort by bar_mid_y so we spread from bottom to top
        sorted_annotations = sorted(arrow_annotations, key=lambda t: t[0])

        # Spread text y-positions to avoid overlap
        text_ys = [sorted_annotations[0][0]]
        for idx in range(1, len(sorted_annotations)):
            prev_y = text_ys[-1]
            desired_y = sorted_annotations[idx][0]
            text_ys.append(max(desired_y, prev_y + min_spacing))

        for (bar_mid, x_pos, label), text_y in zip(sorted_annotations, text_ys, strict=True):
            ax.annotate(
                label,
                xy=(x_pos, bar_mid),
                xytext=(x_pos + x_offset, text_y),
                fontsize=fontsize,
                ha="left",
                va="center",
                arrowprops={"arrowstyle": "->", "color": "black", "lw": 0.8},
                zorder=10,
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "grey", "alpha": 0.6, "edgecolor": "none"},
            )

    # Plot user keys - stacked bars with labels inside or arrows outside
    user_arrow_annotations: list = []
    prev_cumulative = 0.0
    for i, (key, cumulative_value) in enumerate(
        zip(user_keys, np.cumsum([all_data[k] for k in user_keys]), strict=True)
    ):
        ax1.bar(
            3 * sep,
            cumulative_value,
            barsize,
            zorder=1 - i / 10,
            label=key,
            color=(colors[len(job_keys) + len(total_keys) + i], alpha),
            edgecolor="k",
        )
        label_text = f"{key.replace('_total', '').replace('_', ' ')}: {all_data[key]:.2f}"
        annotate_bar(ax1, prev_cumulative, float(cumulative_value), 3 * sep, label_text, user_arrow_annotations)
        prev_cumulative = float(cumulative_value)

    # Plot total CLOPS time
    ax1.bar(2 * sep, clops_time, barsize, zorder=0, color=(colors[-1], alpha), edgecolor="k")
    clops_arrow_annotations: list = []
    annotate_bar(ax1, 0.0, clops_time, 2 * sep, f"CLOPS time: {clops_time:.2f}", clops_arrow_annotations)
    place_arrow_annotations(ax1, clops_arrow_annotations, x_offset=0.7)

    # Plot job keys - stacked bars with labels inside or arrows outside
    job_arrow_annotations: list = []
    prev_cumulative = 0.0
    for i, (key, cumulative_value) in enumerate(zip(job_keys, np.cumsum([all_data[k] for k in job_keys]), strict=True)):
        ax1.bar(sep, cumulative_value, barsize, zorder=2 - i / 10, label=key, color=(colors[i], alpha), edgecolor="k")
        label_text = f"{key.replace('_total', '').replace('_', ' ')}: {all_data[key]:.2f}"
        annotate_bar(ax1, prev_cumulative, float(cumulative_value), sep, label_text, job_arrow_annotations)
        prev_cumulative = float(cumulative_value)

    # Place arrow annotations now that all bars are drawn and y-limits are stable
    place_arrow_annotations(ax1, job_arrow_annotations, x_offset=-0.7)
    place_arrow_annotations(ax1, user_arrow_annotations, x_offset=0.7)

    # Remove top and bottom spines
    ax1.spines["top"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)

    # Set axis labels and limits
    ax1.set_ylabel("Total CLOPS experiment time (seconds)")
    ax2.set_ylabel("Total CLOPS experiment time (%)")

    # Set x-ticks and labels
    time_types = ["Remote (components)", "Wall-time (CLOPS)", "Wall-time (all components)"]
    ax1.set_xticks([i * sep + 1 for i in range(3)], time_types, fontsize=fontsize)

    # Set plot title
    if "clops_h" not in all_data:
        plt.title(
            f"Total execution times for CLOPS experiment\n"
            f"{backend_name}, {execution_timestamp}\n"
            f"CLOPS_v {all_data['clops_v']['value']} on qubits {qubits}\n"
        )
    elif "clops_v" not in all_data:
        plt.title(
            f"Total execution times for CLOPS experiment\n"
            f"{backend_name}, {execution_timestamp}\n"
            f"CLOPS_h {all_data['clops_h']['value']} on qubits {qubits}\n"
        )
    else:
        plt.title(
            f"Total execution times for CLOPS experiment\n"
            f"{backend_name}, {execution_timestamp}\n"
            f"CLOPS_v {all_data['clops_v']['value']} / CLOPS_h {all_data['clops_h']['value']} on qubits {qubits}\n"
        )

    plt.gcf().set_dpi(250)
    plt.close()

    return fig_name, fig


def plot_timestamp_waterfall(job_meta: dict[str, dict[str, Any]], dataset: xr.Dataset) -> dict[str, Figure]:
    """Generate per-update waterfall plots showing remote job timestamps as horizontal separators on a vertical bar.

    For each update, all raw timestamps from every batch job are collected, sorted chronologically,
    and drawn as horizontal lines on a single vertical bar spanning from the earliest to the latest
    timestamp.  Each line is labelled with the timestamp name and its offset (in seconds) from the
    earliest event.

    Args:
        job_meta: The ``job_meta_per_update`` dictionary stored in the dataset.
            Structure: ``{update_key: {batch_key: {"timestamps": {...}, ...}, ...}, ...}``
        dataset: The xarray dataset with experiment attributes (used for titles).

    Returns:
        A dictionary mapping figure names to :class:`~matplotlib.figure.Figure` objects.

    """
    job_time_format = "%Y-%m-%dT%H:%M:%S.%f%z"

    qubits = dataset.attrs["qubits"]
    backend_name = dataset.attrs["backend_name"]
    execution_timestamp = dataset.attrs["execution_timestamp"]

    figures: dict[str, Figure] = {}

    for update_key, batches in job_meta.items():
        # Collect all (datetime, label) pairs across batches for this update
        all_timestamps: list[tuple[datetime, str]] = []

        for batch_key, batch_data in batches.items():
            ts = batch_data.get("timestamps")
            if ts is None:
                continue

            for ts_name, ts_value in ts.items():
                if ts_value is None:
                    continue
                if isinstance(ts_value, str):
                    try:
                        dt = datetime.strptime(ts_value, job_time_format)
                    except ValueError:
                        continue
                elif isinstance(ts_value, datetime):
                    dt = ts_value
                else:
                    continue
                label = f"{batch_key}: {ts_name}"
                all_timestamps.append((dt, label))

        if not all_timestamps:
            continue

        # Sort chronologically
        all_timestamps.sort(key=lambda t: t[0])
        t_origin = all_timestamps[0][0]
        t_end = all_timestamps[-1][0]
        total_seconds = (t_end - t_origin).total_seconds()

        # Build arrays for plotting
        offsets = [(dt - t_origin).total_seconds() for dt, _ in all_timestamps]
        labels = [label for _, label in all_timestamps]

        # --- Plotting ---
        fontsize = 7
        bar_x = 0.0
        bar_width = 0.6
        num_segments = len(offsets) - 1

        fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(all_timestamps))))

        # Color each segment between consecutive timestamps
        cmap = plt.colormaps["tab20"] if num_segments > MAX_TAB10_SEGMENTS else plt.colormaps["tab10"]
        for i in range(num_segments):
            seg_bottom = offsets[i]
            seg_height = offsets[i + 1] - offsets[i]
            seg_label = f"{labels[i]} → {labels[i + 1]} ({seg_height:.3f} s)"
            ax.bar(
                bar_x,
                seg_height,
                bar_width,
                bottom=seg_bottom,
                color=cmap(i % cmap.N / cmap.N),
                edgecolor="k",
                linewidth=0.5,
                label=seg_label,
            )

        # ax.set_xlim(-1, 1)
        y_lo = -total_seconds * 0.05 if total_seconds > 0 else -0.1
        y_hi = total_seconds * 1.05 if total_seconds > 0 else 0.1
        ax.set_ylim(y_lo, y_hi)

        # Legend
        ax.legend(fontsize=fontsize, loc="center left", bbox_to_anchor=(1.05, 0.5), borderaxespad=0.0)
        ax.set_ylabel("Time since first recorded event (seconds)")
        ax.set_xticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.set_title(
            f"Remote timestamps – {update_key}\n"
            f"Backend: {backend_name.capitalize()}, {execution_timestamp}\n"
            f"Qubits {qubits}  |  Total span: {total_seconds:.3f} s"
        )

        fig.set_dpi(250)
        fig.tight_layout()
        plt.close(fig)

        fig_name = f"timestamp_waterfall_{update_key}"
        figures[fig_name] = fig

    return figures


def retrieve_clops_elapsed_times(job_meta: dict[str, dict[str, Any]]) -> dict[str, float]:  # noqa: PLR0912
    """Retrieve the elapsed times from the CLOPS job metadata.

    Args:
        job_meta (Dict[Dict[str, Any]]): A dictionary of CLOPS jobs metadata
    Returns:
        Dict[str, float]: A dictionary of elapsed times of all CLOPS jobs

    """
    all_job_elapsed: dict = {}
    totals_keys: list[str] = []
    for u_index, update in enumerate(job_meta):
        all_job_elapsed[update] = {}
        for b_index, batch in enumerate(job_meta[update].keys()):
            # ["timestamps"] might be empty if backend is a simulator
            if job_meta[update][batch]["timestamps"] is not None:
                x = job_meta[update][batch]["timestamps"]

                # Optional timestamps; pulse-level jobs are compiled client-side and have no compilation ones
                has_compile = "compilation_started" in x and "compilation_ended" in x
                has_fetch_cal = "fetch_calibration_started" in x and "fetch_calibration_ended" in x
                has_validation = "validation_started" in x and "validation_ended" in x

                # Check if timestamps need parsing or are already datetime objects
                if isinstance(x["received"], str):
                    job_time_format = "%Y-%m-%dT%H:%M:%S.%f%z"
                    if has_compile:
                        compile_f = datetime.strptime(x["compilation_ended"], job_time_format)
                        compile_i = datetime.strptime(x["compilation_started"], job_time_format)
                    execution_f = datetime.strptime(x["execution_ended"], job_time_format)
                    execution_i = datetime.strptime(x["execution_started"], job_time_format)
                    job_f = datetime.strptime(x["ready"], job_time_format)
                    job_i = datetime.strptime(x["received"], job_time_format)
                    post_processing_i = datetime.strptime(x["post_processing_started"], job_time_format)
                    post_processing_f = job_f
                    if has_fetch_cal:
                        fetch_cal_i = datetime.strptime(x["fetch_calibration_started"], job_time_format)
                        fetch_cal_f = datetime.strptime(x["fetch_calibration_ended"], job_time_format)
                    if has_validation:
                        validation_i = datetime.strptime(x["validation_started"], job_time_format)
                        validation_f = datetime.strptime(x["validation_ended"], job_time_format)
                else:
                    if has_compile:
                        compile_f = x["compilation_ended"]
                        compile_i = x["compilation_started"]
                    execution_f = x["execution_ended"]
                    execution_i = x["execution_started"]
                    job_f = x["ready"]
                    job_i = x["received"]
                    post_processing_i = x["post_processing_started"]
                    post_processing_f = job_f
                    if has_fetch_cal:
                        fetch_cal_i = x["fetch_calibration_started"]
                        fetch_cal_f = x["fetch_calibration_ended"]
                    if has_validation:
                        validation_i = x["validation_started"]
                        validation_f = x["validation_ended"]

                all_job_elapsed[update][batch] = {
                    # Job total includes queueing time and thus depends on number of jobs submitted simultaneously
                    "job_total": (job_f - job_i).total_seconds(),
                    "execution_total": (execution_f - execution_i).total_seconds(),
                    "post_processing_total": (post_processing_f - post_processing_i).total_seconds(),
                    "_job_i": job_i,
                    "_job_f": job_f,
                }
                if has_compile:
                    all_job_elapsed[update][batch]["compile_total"] = (compile_f - compile_i).total_seconds()
                if has_fetch_cal:
                    all_job_elapsed[update][batch]["fetch_calibration_total"] = (
                        fetch_cal_f - fetch_cal_i
                    ).total_seconds()
                if has_validation:
                    all_job_elapsed[update][batch]["validation_total"] = (validation_f - validation_i).total_seconds()

                # Save the keys, will be needed later
                totals_keys = [k for k in all_job_elapsed[update][batch] if not k.startswith("_")]

                # Update metadata file - need to turn values ("timedelta" type) into str
                job_meta["update_" + str(u_index + 1)]["batch_job_" + str(b_index + 1)].update(
                    {k: str(all_job_elapsed[update][batch][k]) for k in totals_keys}
                )

    # Add up totals
    overall_elapsed: dict = {}
    # If backend is a simulator
    if not totals_keys:
        return overall_elapsed

    overall_elapsed = dict.fromkeys(totals_keys, 0)
    overall_elapsed_per_update: dict = {}
    for update, batches in job_meta.items():
        overall_elapsed_per_update[update] = dict.fromkeys(totals_keys, 0)
        for batch in batches:
            for k in totals_keys:
                overall_elapsed_per_update[update][k] += float(all_job_elapsed[update][batch][k])
        # The first update is treated as a warm-up and excluded, unless it is the only one
        for k in totals_keys:
            if update != "update_1" or len(job_meta) == 1:
                overall_elapsed[k] += float(overall_elapsed_per_update[update][k])
    return overall_elapsed
