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

"""Plots of a CDR fit: the training data, the fitted line and the target."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from iqm.error_reduction_tools.cdr.cdr_technique import CDRDiagnostics
from iqm.error_reduction_tools.estimation import EstimationResult
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

TRAINING_STYLE: dict[str, Any] = {"marker": "o", "linestyle": "none", "markersize": 5, "alpha": 0.85}
"""Style of the training points."""

TARGET_STYLE: dict[str, Any] = {"marker": "*", "linestyle": "none", "markersize": 16}
"""Style of the target value."""


def _figure_of(ax: Axes) -> Figure:
    """Return the figure an axes belongs to, narrowed from ``Figure | SubFigure | None``."""
    figure = ax.get_figure(root=True)
    if not isinstance(figure, Figure):
        raise ValueError("The axes do not belong to a figure.")
    return figure


def _columns(report: CDRDiagnostics, columns: int | Sequence[int] | None) -> list[int]:
    """Choose the fitted columns to plot."""
    total = int(report.slopes.size)
    if columns is None:
        return list(range(total))
    wanted = [columns] if isinstance(columns, int) else list(columns)
    for column in wanted:
        if not 0 <= column < total:
            raise IndexError(f"Column {column} is out of range. This fit has {total}.")
    return [int(column) for column in wanted]


def _report_of(result: EstimationResult, target_index: int) -> CDRDiagnostics:
    """Return the CDR diagnostics of one target."""
    if not 0 <= target_index < len(result.targets):
        raise ValueError(f"There is no target {target_index}. The result has {len(result.targets)}.")
    diagnostics = result.targets[target_index].diagnostics
    if not isinstance(diagnostics, CDRDiagnostics):
        raise ValueError(f"Target {target_index} was not estimated with CDREstimation.")
    return diagnostics


def plot_cdr_fit(
    result: EstimationResult,
    target_index: int = 0,
    columns: int | Sequence[int] | None = None,
    *,
    ax: Axes | None = None,
    annotate: bool = True,
    max_panels: int = 6,
) -> Figure:
    """Plot the training data, the fitted line and the target of one target circuit.

    Args:
        result: What :meth:`~iqm.error_reduction_tools.workflow.ErrorReductionWorkflow.estimate` returned, with a
            :class:`~iqm.error_reduction_tools.cdr.cdr_technique.CDREstimation`.
        target_index: Which target to plot.
        columns: Index or indices of the columns to plot, Pauli terms with ``combine="per_term"`` and observables
            with ``combine="group"``. ``None`` plots all columns, up to ``max_panels``.
        ax: Draw into this axes instead of making a figure. All chosen columns share it.
        annotate: Annotate each fit with the number of training circuits, spread, R^2, and the target's measured
            and mitigated values.
        max_panels: Upper limit on the number of panels when ``columns`` is ``None``.

    Returns:
        The figure containing the plot.

    Raises:
        ValueError: If the target was not estimated with CDR.
        IndexError: If a requested column is out of range for this fit.

    """
    report = _report_of(result, target_index)
    chosen = _columns(report, columns)
    if columns is None and len(chosen) > max_panels:
        chosen = chosen[:max_panels]

    if ax is None:
        figure, axes_list = plt.subplots(
            1, len(chosen), figsize=(4.8 * len(chosen), 4.4), squeeze=False, constrained_layout=True
        )
        panels = list(axes_list[0])
    else:
        figure = _figure_of(ax)
        panels = [ax] * len(chosen)

    for panel, column in zip(panels, chosen):
        _plot_one(panel, report, column, shared=ax is not None, annotate=annotate)

    if ax is not None:
        ax.legend(fontsize="small")
    return figure


def _plot_one(panel: Axes, report: CDRDiagnostics, column: int, *, shared: bool, annotate: bool) -> None:
    """Draw the training points, the fitted line and the target of one column.

    Args:
        panel: Axes to draw into.
        report: Diagnostics of the fit.
        column: Index of the column to plot.
        shared: Whether several columns share the panel. If so, the column is named in the legend instead of the
            title.
        annotate: Whether to add a text box with the diagnostics.

    """
    kind = "term" if report.combine == "per_term" else "observable"
    noisy = report.noisy_train[:, column]
    exact = report.exact_train[:, column]
    target_noisy = float(report.noisy_target[column])
    slope = float(report.slopes[column])
    intercept = float(report.intercepts[column])
    r_squared = float(report.r_squared[column])
    suffix = f", {kind} {column}" if shared else ""
    points = panel.plot(noisy, exact, label=f"training circuits{suffix}", **TRAINING_STYLE)
    colour = points[0].get_color()

    low = min(float(noisy.min()), target_noisy)
    high = max(float(noisy.max()), target_noisy)
    pad = 0.08 * max(high - low, 1e-9)
    grid = np.linspace(low - pad, high + pad, 100)
    inside = (grid >= noisy.min()) & (grid <= noisy.max())

    # Solid over the range of the training points, dashed outside it.
    panel.plot(
        grid[inside],
        slope * grid[inside] + intercept,
        color=colour,
        linewidth=1.6,
        label=f"fit: {slope:+.3f} x {intercept:+.3f}{suffix}",
    )
    for segment in (grid[grid < noisy.min()], grid[grid > noisy.max()]):
        if segment.size:
            panel.plot(segment, slope * segment + intercept, color=colour, linewidth=1.6, linestyle="--")

    panel.plot(
        [target_noisy],
        [slope * target_noisy + intercept],
        color=colour,
        label=f"target{suffix}",
        **TARGET_STYLE,
    )
    panel.axvline(target_noisy, color="grey", linewidth=0.8, alpha=0.6)

    # Range of the training points.
    panel.axvspan(float(noisy.min()), float(noisy.max()), color=colour, alpha=0.07)

    panel.set_xlabel("measured (noisy) value")
    panel.set_ylabel("exact value")
    panel.margins(0.12)
    if not shared:
        panel.set_title(f"{kind} {column}", fontsize="medium")
        panel.legend(fontsize="small", loc="best")

    if annotate:
        lines = [f"{len(noisy)} training circuits", f"measured spread {float(np.ptp(noisy)):.3f}"]
        if np.isfinite(r_squared):
            lines.append(f"R^2 {r_squared:.3f}")
        bracketed = float(noisy.min()) <= target_noisy <= float(noisy.max())
        lines.append(f"target {target_noisy:+.3f}" + ("" if bracketed else ", outside the training range"))
        lines.append(f"mitigated {slope * target_noisy + intercept:+.4f}")
        panel.text(
            0.03,
            0.97,
            "\n".join(lines),
            transform=panel.transAxes,
            ha="left",
            va="top",
            fontsize="x-small",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "lightgrey"},
        )


def plot_cdr_summary(
    result: EstimationResult,
    observable_index: int = 0,
    *,
    ideals: Sequence[float] | Mapping[int, float] | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plot the mitigated and unmitigated value of one observable for every target.

    Args:
        result: What :meth:`~iqm.error_reduction_tools.workflow.ErrorReductionWorkflow.estimate` returned.
        observable_index: Which observable to plot.
        ideals: Known ideal value of each target.
        ax: Draw into this axes instead of making a figure.

    Returns:
        The figure containing the plot.

    """
    num_targets = len(result.targets)
    if ax is None:
        figure, panel = plt.subplots(figsize=(8.2, 1.4 + 0.6 * num_targets), constrained_layout=True)
    else:
        figure, panel = _figure_of(ax), ax

    positions = np.arange(num_targets)
    panel.plot(result.unmitigated[:, observable_index], positions, label="measured (unmitigated)", **TRAINING_STYLE)
    panel.plot(result.values[:, observable_index], positions, color="tab:green", label="CDR estimate", **TARGET_STYLE)
    if ideals is not None:
        reference = [
            float(ideals[index]) if not isinstance(ideals, Mapping) else float(ideals.get(index, np.nan))
            for index in range(num_targets)
        ]
        panel.plot(reference, positions, marker="|", linestyle="none", markersize=18, color="black", label="ideal")

    panel.set_yticks(positions)
    panel.set_yticklabels([f"target {index}" for index in positions])
    panel.set_xlabel("expectation value")
    panel.margins(x=0.08, y=0.35)
    panel.legend(fontsize="small", loc="center left", bbox_to_anchor=(1.01, 0.5))
    panel.grid(axis="x", alpha=0.25)
    return figure
