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
"""Options to customize the appearance of multiple aspects of a plot."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .plot_styles import (
    ColorMap,
    LabelFormatter,
    LineStyle,
    MarkerFillStyle,
    MarkerStyle,
    PlotStyle,
    SaveType,
    TickLocator,
)


@dataclass()
class LineOptions:
    """Options to customize the appearance of a line or scatter data object."""

    line_style: LineStyle = LineStyle.SOLID
    """Line style."""
    marker_style: MarkerStyle = MarkerStyle.NONE
    """Marker style."""
    marker_size: float = 5
    """Marker size."""
    marker_fill_style: MarkerFillStyle = MarkerFillStyle.FULL
    """Marker fill."""
    line_width: float = 2
    """Linewidth of lines and markers, in pixels."""
    color: str | None = None
    """Color of the line and markers.
    Can be given as strings recognized by matplotlib, see https://matplotlib.org/3.1.0/tutorials/colors/colors.html.
    By default, uses the cycling colors set by :attr:`~exa.core.plotting.plot_options.FigureOptions.style`."""
    alpha: float = 1.0
    """Alpha value that describes visibility, from 0.0 (fully transparent) to 1.0 (fully visible).
    For an alternative way to set transparency, see :attr:`color`.
    """
    offset: float = 0.0
    """Offset value for shifting line on y-axis."""

    def copy(self, **attributes) -> LineOptions:
        """Returns a copy where attributes given in `attributes` are changed."""
        return LineOptions(**{**asdict(self), **attributes})

    def for_data_points(self, with_fits: bool) -> LineOptions:
        """Default look meant for datapoints.

        Args:
            with_fits: If True, will format markers so that they look nice with solid lines underneath.

        """
        style = MarkerStyle.CIRCLE if self.marker_style == MarkerStyle.NONE else self.marker_style
        if with_fits:
            return self.copy(marker_style=style, line_style=LineStyle.NONE, marker_fill_style=MarkerFillStyle.NONE)
        return self.copy(marker_style=style)


@dataclass()
class LegendOptions:
    """Options to customize the appearance of legend."""

    legend_format: str | None = ""
    """String format for legend entries, for example, ``':.3f'``."""
    visible: bool = True
    """Visibility of the legend."""

    def copy(self, **attributes) -> LegendOptions:
        """Returns a copy where attributes given in `attributes` are changed."""
        return LegendOptions(**{**asdict(self), **attributes})


@dataclass()
class FitAnnotationOptions:
    """Options to customize fit annotations."""

    visible: bool = True
    """Visibility of the annotations. Set to false to hide the annotations."""
    title: str | None = None
    """Can be used to add an additional title text to the annotations box."""
    params_to_show: list[str] | None = None
    """If this list is specified, only show annotations for parameters in this list.
      If empty or None, will show all parameters."""

    def copy(self, **attributes) -> FitAnnotationOptions:
        """Returns a copy where attributes given in `attributes` are changed."""
        return FitAnnotationOptions(**{**asdict(self), **attributes})


@dataclass()
class AxisOptions:
    """Options to customize the appearance of an axis."""

    log_scale: bool = False
    """Plot this axis on a logarithmic scale. Defaults to linear scale."""
    tick_locator: TickLocator = TickLocator.EXTENDED_WILKINSON
    """Option to select how to place tick labels."""
    label_formatter: LabelFormatter = LabelFormatter.SCALAR
    """Option to select how to format tick labels."""
    tick_count: int | None = None
    """How many ticks to show on the axis. Defaults to Matplotlib behavior."""
    decimal_places: int | None = None
    """If :attr:`label_formatter` is :attr:`exa.core.plotting.plot_styles.LabelFormatter.ENGINEERING`,
    the number of decimal places shown in tick labels. Defaults to Matplotlib behavior."""
    label_format: str | None = None
    """If :attr:`label_formatter` is :attr:`exa.core.plotting.plot_styles.LabelFormatter.STRING`,
    this is used to format each tick label. For example,
    ``'{x} ns'`` writes 'ns' after the number in each tick label. Default: no special formatting."""
    percent_scale: float | None = None
    """If :attr:`label_formatter` is :attr:`exa.core.plotting.plot_styles.LabelFormatter.PERCENT`,
    the numbers on an axis are divided by this number before displaying them as percentages. Default: 1."""
    use_grid: bool = False
    """Whether to use background grid for this axis."""
    lower_limit: Any | None = None
    """Lower limit for this axis."""
    upper_limit: Any | None = None
    """Upper limit for this axis."""

    def copy(self, **attributes) -> AxisOptions:
        """Returns a copy where attributes given in `attributes` are changed."""
        return AxisOptions(**{**asdict(self), **attributes})


@dataclass()
class ColorBarOptions:
    """Options to customize color bar appearance."""

    label_formatter: LabelFormatter = LabelFormatter.SCALAR
    """How to format tick labels."""
    decimal_places: int | None = None
    """Number of decimal places for numeric tick labels. Defaults to Matplotlib behavior."""
    label_format: str | None = None
    """If :attr:`label_formatter` is :attr:`exa.core.plotting.plot_styles.LabelFormatter.STRING`,
    this is used to format each tick label. For example,
    `'{x} ns'` writes 'ns' after the number in each tick label. Default: no special formatting."""
    percent_scale: float | None = None
    """If :attr:`label_formatter` is :attr:`exa.core.plotting.plot_styles.LabelFormatter.PERCENT`,
    the numbers on an axis are divided by this number
    before displaying them as percentages. Default: 1."""
    colormap: ColorMap = ColorMap.VIRIDIS
    """Only used when plotting colormaps. The color type of the map."""

    def copy(self, **attributes) -> ColorBarOptions:
        """Returns a copy where attributes given in `attributes` are changed."""
        return ColorBarOptions(**{**asdict(self), **attributes})


@dataclass()
class PlotOptions:
    """Options to change the appearance of a single plot (Matplotlib axes)."""

    x: AxisOptions | None = None
    """Options to change appearance of the horizontal axis.
    If None a new instance of :class:`~exa.core.plotting.plot_options.AxisOptions` is created."""
    y: AxisOptions | None = None
    """Options to change appearance of the vertical axis.
    If None a new instance of :class:`~exa.core.plotting.plot_options.AxisOptions` is created."""
    line_options: LineOptions | None = None
    """Line-specific options."""
    legend: LegendOptions | None = None
    """Options to change the appearance of the legend.
    If None a new instance of :class:`~exa.core.plotting.plot_options.LegendOptions` is created."""
    colorbar: ColorBarOptions | None = None
    """Options to change the appearance of a colobar, in case of colormap or contour plots.
    If None a new instance of :class:`~exa.core.plotting.plot_options.ColorBarOptions` is created."""
    title_wrap: int = 120
    """Wrap the title to another line after this many characters."""
    stack: float = 0.0  # as percentage
    """Stack offset defined as percentage of bandwidth formed by all plotted lines."""

    def __post_init__(self):
        if self.x is None:
            self.x = AxisOptions()
        if self.y is None:
            self.y = AxisOptions()
        if self.legend is None:
            self.legend = LegendOptions()
        if self.colorbar is None:
            self.colorbar = ColorBarOptions()
        if self.line_options is None:
            self.line_options = LineOptions()

    def copy(self, **attributes) -> PlotOptions:
        """Returns a copy where attributes given in `attributes` are changed."""
        # Following lines are just if someone for some reason sets the options to None (and to satisfy pylint)
        x = None if self.x is None else self.x.copy()
        y = None if self.y is None else self.y.copy()
        legend = None if self.legend is None else self.legend.copy()
        colorbar = None if self.colorbar is None else self.colorbar.copy()
        line_options = None if self.line_options is None else self.line_options.copy()
        old = dict(asdict(self), x=x, y=y, legend=legend, colorbar=colorbar, line_options=line_options)
        return PlotOptions(**{**old, **attributes})


@dataclass()
class ComplexOptions:
    """Options to change how the complex data is displayed."""

    use_magnitude_phase: bool = True
    """If True, plot magnitude and phase parts of a complex pulse as a pair of diagrams.
    If False, plot real and imaginary parts instead. Defaults to True."""
    in_degrees: bool = True
    """If this and use_magnitude_phase are both True, plot phase in degrees. If False, use radians.
    Defaults to True."""
    unwrap: bool = False
    """If this and use_magnitude_phase are both True, unwrap phase values. If False, wrap values.
    Defaults to False."""

    def copy(self, **attributes) -> ComplexOptions:
        """Returns a copy where attributes given in `attributes` are changed."""
        return ComplexOptions(**{**asdict(self), **attributes})


@dataclass()
class FigureOptions:
    """Options to change the appearance of a :class:`exa.core.plotting.figure.Figure`."""

    save_type: SaveType = SaveType.PNG
    """File format to use when saving the figure."""
    style: PlotStyle = PlotStyle.DEFAULT
    """General color theme to use when cycling line and marker colors."""
    aspect_ratio: float = 0.75
    """Aspect ratio of all the axes in the Figure. Ratio of width/height."""
    magnification: float = 2.0
    """Magnification factor."""
    dpi: int = 72
    """Dots-per-inch, used when scaling the figure."""
    background: str = "#fff"
    """Background color of the Figure (behond the individual axes). Formatted as hex color string, like ``'#ffffff'``
    for white, ``'#000000'`` for black."""
    title_wrap: int = 120
    """Wrap the title to another line after this many characters."""
    title_font_size: int = 10
    """Font size of the title."""
    font_size: int = 10
    """Font size of everything else."""
    border_width: int = 2
    """Width of the border boxes and tick marks, in pixels."""
    complex_options: ComplexOptions | None = None
    """Options to change the appearance for figures with complex data"""
    plot_options: PlotOptions | None = None
    """Options to change the appearance of all subplots in this figure."""
    fit_annotations: FitAnnotationOptions | None = None
    """Options for fit parameter annotations."""
    is_colormap: bool = False
    """Plot the data as a colormap. Can't be used with `is_contour`."""
    is_contour: bool = False
    """Plot the data as a contour plot. Can't be used with `is_colormap`."""

    def __post_init__(self):
        if self.complex_options is None:
            self.complex_options = ComplexOptions()
        if self.plot_options is None:
            self.plot_options = PlotOptions()
        if self.fit_annotations is None:
            self.fit_annotations = FitAnnotationOptions()

    def copy(self, **attributes) -> FigureOptions:
        """Returns a copy where attributes given in `attributes` are changed."""
        # Following lines are just if someone for some reason sets the options to None (and to satisfy pylint)
        complex_options = None if self.complex_options is None else self.complex_options.copy()
        plot_options = None if self.plot_options is None else self.plot_options.copy()
        fit_annotations = None if self.fit_annotations is None else self.fit_annotations.copy()
        old = dict(
            asdict(self),
            complex_options=complex_options,
            plot_options=plot_options,
            fit_annotations=fit_annotations,
        )
        return FigureOptions(**{**old, **attributes})
