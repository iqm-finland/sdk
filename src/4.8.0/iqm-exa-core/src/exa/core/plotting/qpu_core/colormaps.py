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
"""Defines a normalized colormap."""

from matplotlib.colors import ListedColormap, to_hex
import matplotlib.pyplot as plt
import numpy as np

COLOR_SCALE = 255  # Colors are defined as floats in the range 0-1, so we need this to help define RGB
N_IQM_COLORS = 100  # Number of colors evenly distributed over the color linespace
iqm_colors = np.transpose(
    [
        [np.linspace(0, 2 / COLOR_SCALE, N_IQM_COLORS)],
        [np.linspace(226 / COLOR_SCALE, 71 / COLOR_SCALE, N_IQM_COLORS)],
        [np.linspace(151 / COLOR_SCALE, 149 / COLOR_SCALE, N_IQM_COLORS)],
    ]
)
IQM_COLORMAP = ListedColormap(iqm_colors)

IQM_MINOR_COLORS_5x3 = np.array(
    [
        ["#F2A251", "#F7B45F", "#FCCD6F"],
        ["#E36A45", "#EC7C51", "#F7915F"],
        ["#517CF2", "#5F8FF7", "#6FA7FC"],
        ["#5F51F2", "#6F5FF7", "#826FFC"],
        ["#7C725E", "#8F856E", "#A79B81"],
    ]
)


class NormalizedColormap:
    """Contains the information needed to use a colormap for a specific set of data.

    Main functionality is to use get_color(val) method to get the color for an individual data value.
    """

    def __init__(self, colormap: ListedColormap = IQM_COLORMAP, min_val: float = 0.0, max_val: float = 1.0) -> None:
        """Create a normalized colormap."""
        self.colormap = colormap
        self.min_val = min_val
        self.max_val = max_val

        self.__normalize()

    def __normalize(self) -> None:
        """Create normalized colormap."""
        self.norm = plt.Normalize(self.min_val, self.max_val)

    def get_color(self, val: float) -> str:
        """Returns a color for the given value."""
        return to_hex(self.colormap(self.norm(float(val))))

    def set_min_max(self, min_val: float, max_val: float) -> None:
        """Set min and max values of colormap."""
        self.min_val = min_val
        self.max_val = max_val

    def set_colormap(self, new_colormap: ListedColormap) -> None:
        """Change the colormap to the new one."""
        self.colormap = new_colormap
        self.__normalize()
