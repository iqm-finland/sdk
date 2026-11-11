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
"""Styles for multiple aspects of plots."""

from abc import abstractmethod
from enum import Enum, IntEnum

from matplotlib.colors import Colormap, LinearSegmentedColormap
import matplotlib.pyplot as plt
from matplotlib.ticker import (
    AutoLocator,
    EngFormatter,
    Formatter,
    LinearLocator,
    Locator,
    LogFormatter,
    LogLocator,
    NullFormatter,
    NullLocator,
    PercentFormatter,
    ScalarFormatter,
    StrMethodFormatter,
)
import numpy as np

from .extended_wilkinson_tick_locator import ExtendedWilkinsonTickLocator


class SaveType(Enum):
    """Plot save type enum."""

    EPS = "eps"
    JPG = "jpg"
    PDF = "pdf"
    PNG = "png"
    PS = "ps"
    SVG = "svg"
    TIFF = "tiff"


class PlotStyle(Enum):
    """Plot style enum for Matplotlib styles.

    See also https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html
    """

    BMH = "bmh"
    CLASSIC = "classic"
    COLORBLIND = "tableau-colorblind10"
    DARK = "dark_background"
    DEFAULT = "default"
    FAST = "fast"
    FIVETHIRTYEIGHT = "fivethirtyeight"
    GGPLOT = "ggplot"
    GRAYSCALE = "grayscale"
    SEABORN = "seaborn"
    SOLARIZED = "Solarize_Light2"

    def colors(self, count: int | None = None) -> list[str]:
        """Build a list of colors from plot style enum.

        See also https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html
        for available colors in Matplotlib style.

        Args:
            count: The numerator to decide how often color are repeated depending on the division by
                the number of colors.

        Returns:
            A list of colors.

        """
        if self == self.DEFAULT:
            # "default" is a special value that's not stored in the style library.
            style_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        else:
            style_colors = plt.style.library[self.value]["axes.prop_cycle"].by_key()["color"]
        if count is None:
            return style_colors
        return style_colors * (1 + count // len(style_colors))


class TickLocator(IntEnum):
    """Tick locator enum."""

    NONE = 0
    AUTO = 1
    LINEAR = 2
    LOG = 3
    EXTENDED_WILKINSON = 4

    def locator(self, tick_count: int = 8) -> Locator:
        """Set tick locator for specified axis.

        Args:
            tick_count: Number of ticks along an axis.

        """
        if self is TickLocator.NONE:
            return NullLocator()
        if self is TickLocator.LINEAR:
            return LinearLocator()
        if self is TickLocator.LOG:
            return LogLocator()
        if self is TickLocator.EXTENDED_WILKINSON:
            return ExtendedWilkinsonTickLocator(tick_count)
        return AutoLocator()


class LabelFormatter(IntEnum):
    """Label formatter enum."""

    NONE = 0
    STRING = 1
    SCALAR = 2
    LOG = 3
    PERCENT = 4
    ENGINEERING = 5

    def formatter(
        self,
        decimal_places: int | None = None,
        label_format: str | None = None,
        percent_scale: float | None = None,
    ) -> Formatter:
        """Set label formatter for specified axis.

        Args:
            decimal_places: Number of decimal places for numeric labels along an axis.
            label_format: String format for label.
            percent_scale: Maximum value used to scale label to percentage.

        """
        if self is LabelFormatter.NONE:
            return NullFormatter()
        if self is LabelFormatter.STRING:
            return StrMethodFormatter("{x}") if label_format is None else StrMethodFormatter(label_format)
        if self is LabelFormatter.LOG:
            return LogFormatter()
        if self is LabelFormatter.PERCENT:
            return PercentFormatter() if percent_scale is None else PercentFormatter(percent_scale)
        if self is LabelFormatter.ENGINEERING:
            return EngFormatter() if decimal_places is None else EngFormatter(places=decimal_places)
        return ScalarFormatter()


class LineStyle(Enum):
    """Line style enum."""

    NONE = ""
    SOLID = "solid"
    DASHED = "dashed"
    DASHDOT = "dashdot"
    DOTTED = "dotted"


class MarkerFillStyle(Enum):
    """Marker fill style enum."""

    NONE = "none"
    TOP = "top"
    BOTTOM = "bottom"
    RIGHT = "right"
    LEFT = "left"
    FULL = "full"


class MarkerStyle(Enum):
    """Marker style enum."""

    NONE = ""
    PIXEL = ","
    POINT = "."
    CIRCLE = "o"
    TRIANGLE_DOWN = "v"
    TRIANGLE_UP = "^"
    TRIANGLE_LEFT = "<"
    TRIANGLE_RIGHT = ">"
    SQUARE = "s"
    DIAMOND_THIN = "d"
    DIAMOND = "D"
    PENTAGON = "p"
    HEXAGON_VERTICAL = "h"
    HEXAGON_HORIZONTAL = "H"
    OCTAGON = "8"
    PLUS = "+"
    PLUS_FILLED = "P"
    CROSS = "x"
    CROSS_FILLED = "X"
    STAR = "*"
    TRI_DOWN = "1"
    TRI_UP = "2"
    TRI_LEFT = "3"
    TRI_RIGHT = "4"
    VERTICAL_LINE = "|"
    HORIZONTAL_LINE = "_"


SI_PREFIXES: dict[int, str] = {
    -30: "q",  # quecto
    -27: "r",  # ronto
    -24: "y",  # yocto
    -21: "z",  # zepto
    -18: "a",  # atto
    -15: "f",  # femto
    -12: "p",  # pico
    -9: "n",  # nano
    -6: "\u03bc",  # micro
    -3: "m",  # milli
    0: "",  # none
    3: "k",  # kilo
    6: "M",  # mega
    9: "G",  # giga
    12: "T",  # tera
    15: "P",  # peta
    18: "E",  # exa
    21: "Z",  # zetta
    24: "Y",  # yotta
    27: "R",  # ronna
    30: "Q",  # quetta
}


class ColorMap(Enum):
    """Perceptually uniform sequential colormap color map enum.

    See https://matplotlib.org/tutorials/colors/colormaps.html
    """

    # Perceptually uniform sequential color maps
    CIVIDIS = "cividis"
    INFERNO = "inferno"
    MAGMA = "magma"
    PLASMA = "plasma"
    VIRIDIS = "viridis"
    # Sequential color maps
    BLUES = "Blues"
    BUGN = "BuGn"
    BUPU = "BuPu"
    GNBU = "GnBu"
    GREENS = "Greens"
    GREYS = "Greys"
    ORANGES = "Oranges"
    ORRD = "OrRd"
    PUBU = "PuBu"
    PUBUGN = "PuBuGn"
    PURD = "PuRd"
    PURPLES = "Purples"
    REDS = "Reds"
    RDPU = "RdPu"
    YLGN = "YlGn"
    YLGNBU = "YlGnBu"
    YLORBR = "YlOrBr"
    YLORRD = "YlOrRd"
    # Sequential2 color maps
    AFMHOT = "afmhot"
    AUTUMN = "autumn"
    BINARY = "binary"
    BONE = "bone"
    COOL = "cool"
    COPPER = "copper"
    GIST_GRAY = "gist_gray"
    GIST_HEAT = "gist_heat"
    GIST_YARG = "gist_yarg"
    GRAY = "gray"
    HOT = "hot"
    PINK = "pink"
    SPRING = "spring"
    SUMMER = "summer"
    WINTER = "winter"
    WISTIA = "wistia"
    # Diverging color maps
    BRBG = "BrBG"
    BWR = "bwr"
    COOLWARM = "coolwarm"
    PIYG = "PiYG"
    PRGN = "PRGn"
    PUOR = "PuOr"
    RDBU = "RdBu"
    RDGY = "RdGy"
    RDYLBU = "RdYlBu"
    RDYLGN = "RdYlGn"
    SEISMIC = "seismic"
    SPECTRAL = "Spectral"
    # Cyclic color maps
    HSV = "hsv"
    TWILIGHT = "twilight"
    TWILIGHT_SHIFTED = "twilight_shifted"
    # Qualitative color maps
    ACCENT = "Accent"
    DARK2 = "Dark2"
    PAIRED = "Paired"
    PASTEL1 = "Pastel1"
    PASTEL2 = "Pastel2"
    SET1 = "Set1"
    SET2 = "Set2"
    SET3 = "Set3"
    TAB10 = "tab10"
    TAB20 = "tab20"
    TAB20B = "tab20b"
    TAB20C = "tab20c"
    # Miscellaneous color maps
    BRG = "brg"
    CMRMAP = "CMRmap"
    CUBEHELIX = "cubehelix"
    FLAG = "flag"
    GIST_EARTH = "gist_earth"
    GIST_NCAR = "gist_ncar"
    GIST_RAINBOW = "gist_rainbow"
    GIST_STERN = "gist_stern"
    GNUPLOT = "gnuplot"
    GNUPLOT2 = "gnuplot2"
    JET = "jet"
    NIPY_SPECTRAL = "nipy_spectral"
    OCEAN = "ocean"
    PRISM = "prism"
    RAINBOW = "rainbow"
    TERRAIN = "terrain"

    @abstractmethod
    def get(self) -> Colormap:
        """Get Matplotlib colormap."""
        return plt.get_cmap(self.value)

    def grayify(self) -> Colormap:
        """Convert colormap to grayscale.

        Based on https://jakevdp.github.io/blog/2014/10/16/how-bad-is-your-colormap/
        """
        # TODO: Need to consider other approaches as well.
        #  See https://matplotlib.org/tutorials/colors/colormaps.html#grayscale-conversion for further discussion.
        cmap = self.get()
        colors = cmap(np.arange(cmap.N))

        # Convert RGBA to perceived greyscale luminance
        # See http://alienryderflex.com/hsp.html
        rgb_weight = [0.299, 0.587, 0.114]
        luminance = np.sqrt(np.dot(colors[:, :3] ** 2, rgb_weight))
        colors[:, :3] = luminance[:, np.newaxis]

        return LinearSegmentedColormap.from_list(f"{cmap.name}_grayscale", colors, cmap.N)
