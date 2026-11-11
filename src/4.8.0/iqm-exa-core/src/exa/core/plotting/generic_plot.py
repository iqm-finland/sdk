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
"""Generic plot tool."""

import contextlib

import xarray as xr

from exa.core.plotting import figure_helper
from exa.core.run_result import RunResult

from .figure import Figure
from .plot_options import FigureOptions


def generic_plot(
    run_result: RunResult,
    data_variable: str | None = None,
    coords: str | list[str] | None = None,
    title: str | None = None,
    figure_options: FigureOptions | None = None,
    plot_fit: bool | None = True,
) -> Figure:
    """General purpose plotting tool to visualize a part of a dataset.

    Generates a single Figure based on `dataset`.
    Supports 1d and 2d data. Higher-dimensional data is not supported and should be sliced first.
    This function can be used for common basic plot types.
    Based on `data_variable` and `coords`, the function automatically figures out which data should be on which axis.
    Additionally, fitted curves to the data will be added, if found in the dataset.

    Note that possible customization is limited. Any plot that requires more customization, for example in
    Experiments, should use this and then modify the Matplotlib axes as needed, or be written from scratch.

    Args:
        run_result: The run result containing the full dataset and run data.
        data_variable: name of the raw data, should be found in `dataset`.
            The function also searches for arrays named ``<data_variable>_fit``. If such exists, it will be shown as a
            fit on top of the raw data.
        coords: String, or a list of strings describing list of coordinates to plot `data_variable` against.
            By default, tries to pick from the coordinates or dimensions of `data_variable`, but will fail if there
            are too many dimensions.
        title: Title for the figure.
        figure_options: Options to modify the appearance for the figure.
        plot_fit: If a fit is found in the dataset, we can still disable plotting it with this boolean.

    Returns:
        A Figure instance.

    """
    figure_options = figure_options or FigureOptions()
    title = title or figure_helper.generate_figure_title(run_result.run_definition)
    dataset = run_result.dataset

    data_variable = _resolve_data_variable(run_result, data_variable)
    coords = _resolve_coords(dataset[data_variable], coords)

    if plot_fit:
        fits = [dataset[name] for name in list(dataset.data_vars.keys()) if f"{data_variable}_fit" in str(name)]
    else:
        fits = []

    return figure_helper.build_results_figure(dataset[data_variable], coords, fits, title, figure_options)[0]


def _resolve_data_variable(run_result: RunResult, data_variable: str | None) -> str:
    if data_variable is not None:
        return data_variable

    dataset = run_result.dataset
    data_vars = [str(key) for key in dataset.data_vars]

    if not data_vars:
        raise ValueError(f"There is no data to plot: {dataset}")
    if data_vars == 1:
        return data_vars[0]
    with contextlib.suppress(AttributeError):
        data_vars = run_result.run_definition.default_data_parameters
    data_vars = [var for var in data_vars if not var.endswith("_fit")]
    if len(data_vars) == 1:
        return data_vars[0]

    raise ValueError(
        f"Could not resolve what to plot. Possible variables: {dataset.data_vars.keys()}. Given: {data_variable}."
    )


def _resolve_coords(data: xr.DataArray, coords: str | list[str] | None) -> list[str]:
    """Resolve which coordinates given data should be plotted against."""
    dims = [str(dim) for dim in data.sizes]
    if coords is None:
        coords = list(dims)
    if isinstance(coords, str):
        coords = [coords]

    if len(coords) == 1 and len(dims) > 1:
        # only one was specified, but data is multidimensional. Add second dimension as a line stack with legend.
        coords.append(next(key for key in dims if key != next(iter(data[coords[0]].sizes))))

    for coord in coords:
        # In the controllers, index coordinates are added to the result only as dimensions, because they simply
        # count from 0 to N.
        if coord not in data.coords and not ("_index" in coord and coord in dims):
            raise ValueError(
                f"{coord} is not a coordinate of given data. Possible coordinates: "
                f'{list(data.coords.keys())}. Also dims with "_index" suffix are possible.'
            )
    if len(coords) > 2:
        raise ValueError(
            f"Invalid number of coordinates: {coords}. Only one or two coords are supported. If the data has "
            f"more than 2 dimensions, slice the data first."
        )
    return coords
