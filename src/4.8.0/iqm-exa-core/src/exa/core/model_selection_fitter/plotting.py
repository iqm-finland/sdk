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
"""Plotting helpers for model-selection fitter results."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from matplotlib import lines
from matplotlib.axes import Axes
from matplotlib.figure import FigureBase
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from .utils import create_unit_string, get_formatted_label


def plot_result(
    *,
    result: xr.Dataset,
    quantity: str | list[str],
    indices: list[tuple[int] | int] | None = None,
    coordinate: str | list[str] | None = None,
    ax: Axes | None = None,
    plot_style: Literal["point"] = "point",
    uncertainty_style: Literal["errorbar", "fill"] = "errorbar",
) -> FigureBase | None:
    """Plot the results of a model selection fit.

    Args:
        result: The xarray dataset containing the fit results.
        quantity: The name(s) of the quantity(ies) to plot, e.g. "best_model" or "model_probabilities".
        indices: List of indices to plot for the remaining coordinates. Each index can be a single integer or a tuple of integers corresponding to the remaining coordinate dimensions. If None, all indices will be plotted.
        coordinate: The name(s) of the coordinate(s) to plot against. If None, the first coordinate of the best_model dimension will be used.
        ax: The matplotlib axes to plot on. If None, a new figure and axes will be created.
        plot_style: The style of the plot. Currently only "point" is supported.
        uncertainty_style: The style of the uncertainty representation. Can be "errorbar" for error bars or "fill" for filled uncertainty regions. Only applicable if the quantity has a corresponding "_stderr" variable in the dataset.

    Returns:
        A matplotlib figure containing the plot, or None if plotting failed.

    """  # noqa: E501
    linestyles = list(lines.lineStyles.keys())
    quantities = quantity if isinstance(quantity, list) else [quantity]

    if ax is None:
        plt.figure()
        ax = plt.gca()  # type: ignore[assignment]

    if coordinate is None:
        try:
            coordinates = [result.best_model.dims[0]]
        except Exception:
            return None
    else:
        coordinates = coordinate if isinstance(coordinate, list) else [coordinate]

    if isinstance(indices, tuple):
        indices = [indices]

    if plot_style == "point":
        _plot_points(
            quantities=quantities,
            result=result,
            coordinates=coordinates,
            indices=indices,
            uncertainty_style=uncertainty_style,
            ax=ax,
            linestyles=linestyles,
        )

    fig = ax.get_figure()  # type: ignore[union-attr]
    plt.close()
    return fig


def _plot_points(
    *,
    quantities: list[str],
    result: xr.Dataset,
    coordinates: list[str],
    indices: list | None,
    uncertainty_style: str,
    ax: Axes,
    linestyles: list,
) -> None:
    colorcycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for q_ind, q_val in enumerate(quantities):
        data = result[q_val]
        d_scaling, d_string = get_formatted_label(data)
        x_axis = result.coords[coordinates[0]]
        data = result[q_val].transpose(*coordinates, ...)
        try:
            data_stderr = result[q_val + "_stderr"].transpose(*coordinates, ...)
        except Exception:
            data_stderr = None
        remaining_coord_names = list(set(data.dims) - {coordinates[0]})
        remaining_dimensions = tuple(len(data.coords[coord].values) for coord in remaining_coord_names)
        total_length = int(np.prod(remaining_dimensions))
        if indices is not None:
            chosen_indices: list = []
            for index in indices:
                if not isinstance(index, Iterable):
                    flattened_index = np.unravel_index(index, remaining_dimensions)
                    chosen_indices.append(flattened_index)
                else:
                    chosen_indices.append(index)
        else:
            chosen_indices = list(range(total_length))
        x_scaling, x_string = get_formatted_label(data.coords[coordinates[0]])
        for ind in chosen_indices:
            if not isinstance(ind, Iterable):
                ind = np.unravel_index(ind, remaining_dimensions)  # noqa: PLW2901
            ind_tuple = tuple(ind)
            inds = (0,) if total_length == 1 else (int(np.ravel_multi_index(ind_tuple, remaining_dimensions)),)

            slice_dict = {value: slice(None) for value in [coordinates[0]]}
            label = "" if len(quantities) == 1 else q_val + ", "
            for ii, key in enumerate(remaining_coord_names):
                scaling, unit_string = create_unit_string(data.coords[key])
                slice_dict[f"{key}"] = ind_tuple[ii]
                try:
                    label = label + f"{key}: {data.coords[key].values[ind_tuple[ii]] * scaling:.3f} {unit_string} "
                except Exception:
                    label = label + f"{key}: {data.coords[key].values[ind_tuple[ii]]} "
            sliced_stderr = data_stderr[slice_dict] if data_stderr is not None else None
            sliced_data = data[slice_dict]
            if uncertainty_style == "errorbar":
                ax.errorbar(
                    x_axis * x_scaling,
                    sliced_data * d_scaling,
                    sliced_stderr * d_scaling if sliced_stderr is not None else None,
                    linestyle=linestyles[q_ind % len(linestyles)],
                    marker="o",
                    fmt="",
                    capsize=3,
                )
            elif uncertainty_style == "fill":
                ax.plot(
                    x_axis * x_scaling,
                    sliced_data * d_scaling,
                    linestyle=linestyles[q_ind % len(linestyles)],
                    label=label,
                    color=colorcycle[inds[0]],  # type: ignore[index]
                )
                try:  # noqa: SIM105
                    ax.fill_between(
                        x_axis * x_scaling,
                        (sliced_data - sliced_stderr) * d_scaling
                        if sliced_stderr is not None
                        else sliced_data * d_scaling,
                        (sliced_data + sliced_stderr) * d_scaling
                        if sliced_stderr is not None
                        else sliced_data * d_scaling,
                        alpha=0.2,
                        color=colorcycle[inds[0]],  # type: ignore[index]
                    )
                except Exception:
                    pass
        ax.set_ylabel(d_string)
        ax.set_xlabel(x_string)
        ax.set_ylim((float(data.min() * d_scaling), float(data.max() * d_scaling)))
        ax.legend()
