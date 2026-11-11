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
"""Class for defining a plotting method and associated arguments, for postponing a plotting action."""

import base64
from collections.abc import Callable
from dataclasses import dataclass
import functools
from io import BytesIO
from typing import Any

from matplotlib.figure import Figure as MatplotlibFigure
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from exa.core.plotting import Figure, generic_plot
from exa.core.run_result import RunResult


def wrap_mpl_plot(function: Callable[..., MatplotlibFigure]) -> Callable[..., Figure]:
    """A decorator that changes a function to return an EXA figure instead of a matplotlib figure.

    Meant to be used if an original plotting function returns a matplotlib figure and should be made compatible with
    :class:`.PlotDefinition`.
    """

    @functools.wraps(function)
    def plot_with_mpl_and_convert_to_figure(*args, **kwargs) -> Figure:
        figure = function(*args, **kwargs)
        if previously_on := plt.isinteractive():
            plt.ioff()
        exa_figure = Figure()
        plt.close(exa_figure.figure)
        if previously_on:
            plt.ion()
        exa_figure.figure = figure
        exa_figure.axs = np.asarray(figure.axes)
        return exa_figure

    return plot_with_mpl_and_convert_to_figure


@dataclass
class PlotDefinition:
    """Defines a way to generate a Figure from an Xarray dataset."""

    function: Callable[..., Figure]
    """A callable that generates the figure. The first and only positional argument of `function` must be the run result
    where the data is read from. `function` can accept keyword arguments and must return a single Figure."""
    default_kwargs: dict[str, Any]
    """Default keyword arguments that will be passed to `function`, if not overridden."""

    def generate_from_run_result(self, run_result: RunResult, /, **kwargs) -> Figure:
        """Generate a figure from `RunResult` by calling :attr:`function`.

        Args:
            run_result: The run result containing the full dataset and run data.
            kwargs: Keyword arguments passed to :attr:`function`. These override :attr:`default_kwargs`.

        Returns:
            A new :class:`Figure`.

        """
        kwargs = self.default_kwargs | kwargs
        return self.function(run_result, **kwargs)

    def generate_from_dataset(self, dataset: xr.Dataset, /, **kwargs) -> Figure:
        """Generate a figure from `xarray.Dataset` by calling :attr:`function`.

        Args:
            dataset: dataset: The full dataset.
            kwargs: Keyword arguments passed to :attr:`function`. These override :attr:`default_kwargs`.

        Returns:
            A new :class:`Figure`.

        """
        kwargs = self.default_kwargs | kwargs
        return self.function(dataset, **kwargs)

    def embed_base64(self, data: RunResult | xr.Dataset) -> str:
        """Generate a figure and encode it in base64, so that it can be embedded to an ``img`` HTML attribute.

        Args:
            data: The run result containing the full dataset and run data, or the full dataset.

        Returns:
            A base64 representation of the figure. If rendering the figure fails for any reason, the error message
                will be returned as an alt text instead.

        """
        # We turn off interactive mode to prevent the figure from showing during self.generate:
        if previously_on := plt.isinteractive():
            plt.ioff()
        figure = None
        try:
            if isinstance(data, RunResult):
                figure = self.generate_from_run_result(data).figure
            else:
                figure = self.generate_from_dataset(data).figure
            plt.close(figure)  # close the latest figure so that it's not shown when entering interactive mode.
        except Exception as err:
            if figure is None:
                plt.close()
            return f';" alt="Error rendering image: {err!r}\n"'
        finally:
            if previously_on:
                plt.ion()

        image_bytes = BytesIO()
        figure.savefig(image_bytes, format="png")
        b64 = base64.b64encode(image_bytes.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"


def generic_plot_definition(data_variable: str, **kwargs) -> PlotDefinition:
    """Use :func:`generic_plot` to plot `data_variable`."""
    kwargs.setdefault("data_variable", data_variable)
    return PlotDefinition(generic_plot, kwargs)  # type: ignore[arg-type]
