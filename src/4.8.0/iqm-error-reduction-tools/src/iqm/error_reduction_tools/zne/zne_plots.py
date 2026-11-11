# Copyright 2022-2026 IQM
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

"""Plots of a ZNE fit: the measured points, the fitted curve and the zero-noise estimate."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any, cast

from iqm.error_reduction_tools.estimation import EstimationResult
from iqm.error_reduction_tools.zne.zne_processors import fit_curve
from iqm.error_reduction_tools.zne.zne_technique import ZNEDiagnostics
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.container import Container
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.typing import ColorType
import numpy as np

FIT_COLOUR = "#2a78d6"
"""Colour of the fit in a single-fit panel."""

COMPARISON_COLOURS: tuple[str, ...] = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7")
"""Colours of the fits compared on one panel, in order of use.

The order keeps adjacent colours distinguishable under common colour vision deficiencies.
"""

INK = "#0b0b0b"
"""Colour of titles."""

MUTED = "#52514e"
"""Colour of axes, legends and reference lines."""

SURFACE = "#ffffff"
"""Background colour, also used as the marker edge colour so that overlapping markers stay visible."""

PANELS_PER_ROW = 3
"""Number of panels per row of the grid."""

MEASURED_STYLE: dict[str, Any] = {
    "marker": "o",
    "linestyle": "none",
    "markersize": 6,
    "markeredgecolor": SURFACE,
    "markeredgewidth": 1.0,
    "elinewidth": 1.4,
    "capsize": 3,
}
"""Style of the measured points."""

ESTIMATE_STYLE: dict[str, Any] = {
    "marker": "*",
    "linestyle": "none",
    "markersize": 14,
    "markeredgecolor": SURFACE,
    "markeredgewidth": 1.0,
    "elinewidth": 1.6,
    "capsize": 5,
    # The marker sits on the left edge of the axes, at zero noise.
    "clip_on": False,
}
"""Style of the extrapolated value at zero noise."""

IDEAL_STYLE: dict[str, Any] = {"linestyle": ":", "linewidth": 1.6, "color": MUTED}
"""Style of the known ideal value."""


def plot_zne_fit(
    result: EstimationResult,
    target_index: int = 0,
    *,
    ideal: Sequence[float] | None = None,
) -> Figure:
    """Plot the measured points, the fit and the zero-noise estimate of every Pauli term.

    Args:
        result: What :meth:`~iqm.error_reduction_tools.workflow.ErrorReductionWorkflow.estimate` returned, with a
            :class:`~iqm.error_reduction_tools.zne.zne_technique.ZNEEstimation`.
        target_index: Which target to plot.
        ideal: Known ideal values, one per Pauli term, in the order of
            :attr:`~iqm.error_reduction_tools.observables.ObservableSet.pauli_terms`. Drawn as a reference line in
            each panel.

    Returns:
        The figure containing the plot. The figure is not shown.

    Raises:
        ValueError: If the diagnostics did not come from the ZNE estimator, or ``ideal`` does not
            have one value per Pauli term.

    """
    report = _report_of(result, target_index)
    measured_sigma = _measured_sigma(result, target_index, report)
    wanted = list(enumerate(report.pauli_terms))
    references = _references(ideal, report)

    columns = min(len(wanted), PANELS_PER_ROW)
    rows = math.ceil(len(wanted) / columns)
    figure, grid_of_axes = plt.subplots(
        rows, columns, figsize=(4.9 * columns, 3.9 * rows), squeeze=False, constrained_layout=True
    )
    figure.set_facecolor(SURFACE)
    panels = [panel for row in grid_of_axes for panel in row]
    for panel in panels[len(wanted) :]:
        panel.set_visible(False)

    for position, (column, label) in enumerate(wanted):
        panel = panels[position]
        _style(panel, report.noise_factors, xlabel=position >= len(wanted) - columns, ylabel=position % columns == 0)
        handles: list[Artist | Container] = [
            _draw_measured(panel, report, column, FIT_COLOUR, measured_sigma),
            _draw_fit(panel, report, column, FIT_COLOUR, label=_model_name(report)),
            _draw_estimate(panel, report, column, FIT_COLOUR),
        ]
        if references is not None:
            handles.append(
                panel.axhline(references[column], label=f"ideal   {references[column]:+.4f}", zorder=2, **IDEAL_STYLE)
            )
        _finish(panel, report.measured[:, column], handles)
        panel.set_title(f"$\\langle \\mathrm{{{label}}} \\rangle$", fontsize="large", color=INK)

    figure.suptitle(_title(result, target_index, report), fontsize="large", color=INK)
    return figure


def plot_zne_comparison(
    results: Mapping[str, EstimationResult],
    target_index: int = 0,
    pauli_term: int | str = 0,
    *,
    ideal: float | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Compare several estimation settings of one Pauli term on one panel.

    Args:
        results: Label to :class:`~iqm.error_reduction_tools.estimation.EstimationResult`, e.g.
            ``{"Poly deg 1": ..., "Exp": ...}``. All must come from one workflow.
        target_index: Which target to plot.
        pauli_term: Which Pauli term, by label or by position.
        ideal: Known ideal value of that Pauli term, drawn as a reference line.
        ax: Draw into this axes instead of making a figure.

    Returns:
        The figure containing the plot.

    Raises:
        ValueError: If ``results`` is empty, or holds more configurations than there are colours.

    """
    if not results:
        raise ValueError("Nothing to compare: `results` is empty.")
    if len(results) > len(COMPARISON_COLOURS):
        raise ValueError(
            f"{len(results)} configurations share one panel, but only {len(COMPARISON_COLOURS)} stay "
            "distinguishable. Compare them in batches."
        )

    if ax is None:
        figure, panel = plt.subplots(figsize=(7.0, 4.6), constrained_layout=True)
        figure.set_facecolor(SURFACE)
    else:
        figure, panel = cast(Figure, ax.get_figure()), ax

    first_result = next(iter(results.values()))
    first = _report_of(first_result, target_index)
    column = _resolve_pauli_term(first, pauli_term)

    _style(panel, first.noise_factors, xlabel=True, ylabel=True)
    handles: list[Artist | Container] = [
        _draw_measured(panel, first, column, INK, _measured_sigma(first_result, target_index, first))
    ]
    for position, (name, result) in enumerate(results.items()):
        report = _report_of(result, target_index)
        colour = COMPARISON_COLOURS[position]
        _draw_fit(panel, report, column, colour)
        handles.append(_draw_estimate(panel, report, column, colour, name=name))

    if ideal is not None:
        handles.append(panel.axhline(ideal, label=f"ideal   {ideal:+.4f}", zorder=2, **IDEAL_STYLE))

    _finish(panel, first.measured[:, column], handles)
    panel.set_title(
        f"$\\langle \\mathrm{{{first.pauli_terms[column]}}} \\rangle$: one dataset, several fits",
        fontsize="large",
        color=INK,
    )
    return figure


def _report_of(result: EstimationResult, target_index: int) -> ZNEDiagnostics:
    """Return the ZNE diagnostics of one target."""
    if not 0 <= target_index < len(result.targets):
        raise ValueError(f"There is no target {target_index}. The result has {len(result.targets)}.")
    diagnostics = result.targets[target_index].diagnostics
    if not isinstance(diagnostics, ZNEDiagnostics):
        raise ValueError(f"Target {target_index} was not estimated with ZNEEstimation.")
    return diagnostics


def _measured_sigma(result: EstimationResult, target_index: int, report: ZNEDiagnostics) -> np.ndarray | None:
    """Return the bootstrap spread of the fitted points, shape ``(num_fitted, num_terms)``, if there is one."""
    if result.term_sigma is None:
        return None
    return result.term_sigma[target_index][list(report.variant_indices)]


def _resolve_pauli_term(report: ZNEDiagnostics, pauli_term: int | str) -> int:
    """Return the column of a Pauli term given by label or by position."""
    labels = list(report.pauli_terms)
    if isinstance(pauli_term, str):
        if pauli_term not in labels:
            raise ValueError(f"No Pauli term {pauli_term!r} in this target. It has {labels}.")
        return labels.index(pauli_term)
    if not 0 <= pauli_term < len(labels):
        raise IndexError(f"Pauli term {pauli_term} is out of range. This target has {len(labels)}.")
    return int(pauli_term)


def _references(ideal: Sequence[float] | None, report: ZNEDiagnostics) -> list[float] | None:
    """Check that there is one ideal value per Pauli term."""
    if ideal is None:
        return None
    expected = len(report.pauli_terms)
    if len(ideal) != expected:
        raise ValueError(
            f"Got {len(ideal)} ideal values for {expected} Pauli terms {list(report.pauli_terms)}. "
            "Pass one per Pauli term, in that order."
        )
    return [float(value) for value in ideal]


def _model_name(report: ZNEDiagnostics) -> str:
    """Return the name of the fitted model."""
    if report.extrapolation_type == "Exp":
        return "exponential fit"
    if report.degree is None:
        return "Richardson extrapolation"
    return f"polynomial fit, degree {report.degree}"


def _title(result: EstimationResult, target_index: int, report: ZNEDiagnostics) -> str:
    """Return the figure title, noting when only some executed noise factors were fitted."""
    executed = report.num_executed_factors
    fitted = int(report.noise_factors.size)
    target = f", target {target_index}" if len(result.targets) > 1 else ""
    subset = f"  ({fitted} of {executed} executed noise factors fitted)" if executed and executed > fitted else ""
    return f"Zero-noise extrapolation{target}{subset}"


def _style(panel: Axes, factors: np.ndarray, *, xlabel: bool, ylabel: bool) -> None:
    """Set the axes limits, ticks and labels of one panel."""
    panel.set_xlim(0.0, float(factors.max()) * 1.05)
    panel.set_facecolor(SURFACE)
    panel.margins(y=0.15)
    for side in ("top", "right"):
        panel.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        panel.spines[side].set_color(MUTED)

    # Ticks at zero and at the executed noise factors.
    ticks = [0.0, *factors.tolist()]
    panel.set_xticks(ticks)
    panel.set_xticklabels([f"{value:g}" for value in ticks])
    panel.tick_params(colors=MUTED, labelsize="small", length=3)
    if xlabel:
        panel.set_xlabel("noise factor", color=MUTED, fontsize="small")
    if ylabel:
        panel.set_ylabel("expectation value", color=MUTED, fontsize="small")


def _draw_measured(
    panel: Axes, report: ZNEDiagnostics, column: int, colour: ColorType, sigma: np.ndarray | None
) -> Container:
    """Draw the measured points of one Pauli term, with error bars when available."""
    return panel.errorbar(
        report.noise_factors,
        report.measured[:, column],
        yerr=None if sigma is None else sigma[:, column],
        color=colour,
        label="measured",
        zorder=4,
        **MEASURED_STYLE,
    )


def _draw_fit(
    panel: Axes,
    report: ZNEDiagnostics,
    column: int,
    colour: ColorType,
    *,
    label: str | None = None,
) -> Artist:
    """Draw the fitted curve of one Pauli term, dashed below the smallest noise factor."""
    grid = np.linspace(0.0, float(report.noise_factors.max()) * 1.05, 200)
    curve = fit_curve(report.fit_parameters[column], grid, report.extrapolation_type)
    inside = grid >= report.noise_factors.min()
    (solid,) = panel.plot(grid[inside], curve[inside], color=colour, linewidth=2.0, label=label, zorder=3)
    panel.plot(grid[~inside], curve[~inside], color=colour, linewidth=2.0, linestyle="--", zorder=3)
    return solid


def _draw_estimate(
    panel: Axes,
    report: ZNEDiagnostics,
    column: int,
    colour: ColorType,
    *,
    name: str = "zero noise",
) -> Container:
    """Draw the zero-noise value of one Pauli term, with the value in the legend."""
    value = float(report.extrapolated[column])
    return panel.errorbar([0.0], [value], color=colour, label=f"{name}   {value:+.4f}", zorder=6, **ESTIMATE_STYLE)


def _finish(panel: Axes, series: np.ndarray, handles: Sequence[Artist | Container]) -> None:
    """Place the legend in the corner away from the data, and extend the y range to make room for it."""
    falling = bool(series[-1] <= series[0])

    bottom, top = panel.get_ylim()
    room = 0.4 * (top - bottom)
    panel.set_ylim(bottom, top + room) if falling else panel.set_ylim(bottom - room, top)

    panel.legend(
        handles=list(handles),
        loc="upper right" if falling else "lower right",
        frameon=True,
        facecolor=SURFACE,
        edgecolor="none",
        framealpha=1.0,
        fontsize="small",
        labelcolor=MUTED,
        handletextpad=0.6,
    )
