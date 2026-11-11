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
"""Figure class to help plot figures."""

from pathlib import Path
from textwrap import wrap

import matplotlib.pyplot as plt
import numpy as np

from exa.core.plotting.plot_options import FigureOptions, SaveType


class Figure:
    """Plot figure.

    Acts as both a container and layout for plots.

    Attributes:
        rows: Number of plots to be placed vertically in figure.
        cols: Number of plots to be placed horizontally in figure.
        options: Plot-specific options.

    """

    def __init__(
        self, rows: int = 1, cols: int = 1, title: str | None = None, options: FigureOptions = FigureOptions()
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.options = options
        with plt.style.context(self.options.style.value):
            plt.rc("font", size=self.options.font_size)
            plt.rc("axes", labelsize=self.options.title_font_size, linewidth=self.options.border_width)
            self.figure, self.axs = plt.subplots(
                nrows=rows,
                ncols=cols,
                figsize=self.__adjust_size(),
                facecolor=self.options.background,
                tight_layout=True,
                squeeze=False,
            )
            should_be_interactive = plt.isinteractive()
            for axis in np.concatenate(self.axs):
                axis.set_facecolor(self.options.background)
                axis.xaxis.set_tick_params(width=self.options.border_width, length=self.options.border_width * 2)
                axis.yaxis.set_tick_params(width=self.options.border_width, length=self.options.border_width * 2)
            if title is not None:
                plt.suptitle("\n".join(wrap(title, self.options.title_wrap)), y=1.02, size=self.options.title_font_size)

        if should_be_interactive and not plt.isinteractive():
            # plt.style.context seems to be buggy in Matplotlib 3.6.3.
            # Interactive mode normally turns on the first call of subplots, but it turns off again after the context.
            # So we keep track manually and enable it if needed.
            plt.ion()
        plt.rc("axes", titlesize=self.options.title_font_size)

    def show(self) -> None:
        """Display figure."""
        self.figure.show(warn=False)

    def save(self, path: Path, save_as: SaveType | None = None) -> None:
        """Save figure."""
        if save_as is None:
            return self.figure.savefig(path, format=self.options.save_type.value, bbox_inches="tight")
        return self.figure.savefig(path, format=save_as.value, bbox_inches="tight")

    def __adjust_size(self) -> tuple[float, float]:
        """Calculate and set width and height for figure and its subplots."""
        default_width, _ = plt.rcParams["figure.figsize"]
        adjusted_width = self.options.magnification * default_width
        adjusted_height = self.options.magnification * self.options.aspect_ratio * default_width
        return adjusted_width, adjusted_height
