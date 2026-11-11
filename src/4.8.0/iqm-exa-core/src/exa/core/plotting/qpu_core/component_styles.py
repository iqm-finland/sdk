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
"""Styles for various chip layouts."""

from enum import Enum
from typing import Any


class ComponentColors(Enum):
    """Colors used for different components."""

    COUPLER = (179 / 255, 167 / 255, 9 / 255)
    HIFREQ = (0 / 255, 192 / 255, 255 / 255)
    LOFREQ = (255 / 255, 25 / 255, 85 / 255)
    QUBIT = (32 / 255, 0 / 255, 102 / 255)


def default_style() -> dict[str, Any]:
    """Create a dictionary that works as our default style.

    All chip specific functions will modify this dictionary.

    Returns:
        Values for Matplotlib formatting and style.

    """
    return {
        "figure_size": (20, 20),
        "qubit": {"radius": 0.3, "color": "#bb00bbff", "linewidth": 2, "edgecolor": "black"},
        "coupler": {"size": (0.45, 0.45), "color": "#09a199ff", "linewidth": 2, "edgecolor": "black"},
        "comp_r": {"size": (0.45, 0.45), "color": "#ee6b6e", "linewidth": 2, "angle": 0, "edgecolor": "black"},
        "component_label": {
            "fontsize": 18,
            "color": "black",
            "fontweight": "roman",
            "path_linewidth": 7,
            "path_color": "white",
        },
        "logo": {"position": (0.86, -0.13), "size": (0.14, 0.14)},
        "time_stamp": {"position": (-0.02, -0.6), "fontsize": 17},
        "color_bar": {"fontsize": 25},
        "plot_title": {"fontsize": 35, "fontfamily": "monospace"},
    }
