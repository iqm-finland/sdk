#  ********************************************************************************
#   _____  ____ _
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
"""Plotting tools."""

__all__ = [
    "IQM_COLORMAP",
    "ComplexOptions",
    "CrystalChipType",
    "Figure",
    "FigureOptions",
    "IQM_MINOR_COLORS_5x3",
    "PlotData",
    "PlotOptions",
    "QPUPlot",
    "SaveType",
    "change_unit_magnitude",
    "default_style",
    "display_all_parameters_for_components",
    "generate_table",
    "generic_plot",
    "parse_parameter_from_dut_field",
    "plot_bar_chart",
    "plot_histogram",
    "plot_multi_cdf",
    "plot_time_series",
    "plot_topology",
]

from .figure import Figure
from .generic_plot import generic_plot
from .plot_cumulative_distribution import plot_multi_cdf
from .plot_options import ComplexOptions, FigureOptions, PlotOptions
from .plot_styles import SaveType
from .plots import plot_bar_chart, plot_histogram
from .qpu_core.colormaps import IQM_COLORMAP, IQM_MINOR_COLORS_5x3
from .qpu_core.component_styles import default_style
from .qpu_core.utils import CrystalChipType, change_unit_magnitude
from .qpu_plot import QPUPlot
from .qpu_table import display_all_parameters_for_components, generate_table, parse_parameter_from_dut_field
from .time_series import plot_time_series
from .topology import (
    PlotData,
    plot_topology,
)
