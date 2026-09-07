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

"""Visualization functions for readout characterization data."""

from typing import TypeAlias, cast
import warnings

from matplotlib.patches import ConnectionPatch
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

from .data_processing import (
    ErrorId,
    ErrorProbabilities,
    SingleCovarianceData,
    StateCovarianceData,
)
from .topologies import QPUTopology

CovarianceData: TypeAlias = SingleCovarianceData | StateCovarianceData


def visualize_time_stability(
    data: np.ndarray,
    std_shots: np.ndarray,
    threshold_err: float = 0.1,
    threshold_ratio: float = 3,
    upper_y_label: str | None = None,
) -> None:
    """Visualize temporal stability of readout characterization data across multiple runs.

    Creates a two-panel visualization to assess the temporal stability of readout
    measurements. The first panel shows the distribution of measurements across
    qubits with violin plots, individual data points, and error bars representing pure
    statistical uncertainty. The second panel displays the ratio of total standard
    deviation to shot-noise standard deviation for each qubit.

    Args:
        data: 2D array of shape (``num_runs``, ``num_qubits``) containing measurement data
            from multiple characterization runs.
        std_shots: Array of shot-noise standard deviations for each qubit, representing
            pure statistical uncertainty expected from finite sampling.
        threshold_err: Threshold value for acceptable error level, displayed as a
            horizontal line in the first panel. Default is 0.1 (10%).
        threshold_ratio: Threshold value for the ratio of standard deviations, displayed
            as a horizontal line in the second panel. Default is 3.
        upper_y_label: Label for the y-axis of the first panel. If ``None``, no label is set.

    Returns:
        ``None``. Displays a matplotlib figure with two subplots.

    .. note::

        * The first panel shows violin plots with overlaid scatter points (with jitter)
          and error bars representing shot-noise limited uncertainty.
        * The second panel uses a logarithmic y-scale to show the ratio of total
          standard deviation to shot-noise standard deviation. A ratio near 1 indicates
          that the measurement is shot-noise limited, while higher ratios suggest
          additional sources of variability (e.g., temporal drift).
        * Horizontal reference lines at y=1 and y=threshold_ratio help identify qubits
          with excessive time-dependent variability.

    """
    if data.ndim != 2:  # noqa: PLR2004
        raise ValueError(f"Data must be 2D array, got {data.ndim}D array.")

    if data.shape[1] != len(std_shots):
        raise ValueError(
            f"Number of qubits in data ({data.shape[1]}) does not match length of 'std_shots' ({len(std_shots)})."
        )

    _, ax = plt.subplots(2, 1, figsize=(12, 15), sharex=True)
    num_qubits = data.shape[1]
    positions = np.arange(num_qubits)

    # First panel: Violin plot with scatter and error bars
    ax[0].violinplot(data, positions=positions, points=100, showextrema=False)

    # Add swarmplot-like scatter plot to show individual data points
    rng = np.random.default_rng()
    for i in range(num_qubits):
        x_jitter = rng.normal(loc=positions[i], scale=0.04, size=len(data[:, i]))
        ax[0].scatter(x_jitter, data[:, i], alpha=0.6, s=10, color="black")

    ax[0].errorbar(
        positions,
        np.mean(data, axis=0),
        yerr=std_shots,
        fmt="none",
        ecolor="red",
        label="Pure statistical uncertainty",
        capsize=3,
    )
    ax[0].set_ylabel(upper_y_label)
    ax[0].legend()
    ax[0].yaxis.set_major_locator(plt.MultipleLocator(0.01))
    ax[0].grid(True, which="major", axis="y", linestyle="--", alpha=0.6)
    ax[0].set_title(f"Time stability: {len(data)} characterization runs")
    ax[0].axhline(threshold_err, color="k", ls="--", label=f"Threshold {threshold_err * 100:.1f}%")

    # Second panel: Ratio of standard deviations
    ax[1].plot(positions, np.std(data, axis=0) / np.array(std_shots), "o-", label="Ratio Std")
    ax[1].set_xlabel("Qubit index")
    ax[1].set_ylabel("Ratio of stds: total/shot-noise")
    ax[1].axhline(1, color="k", ls="--")
    ax[1].axhline(threshold_ratio, color="k", ls="--")
    ax[1].grid(True, axis="y", linestyle="--", alpha=0.6)
    ax[1].set_yscale("log")

    plt.xlim(left=-1)
    plt.tight_layout()
    plt.show()


def visualize_state_dep_matrix(
    matrix: np.ndarray,
    std: float,
    title: str = "",
    subset: list[int] | None = None,
) -> None:
    """Visualizes the state-dependent error matrix.

    Args:
        matrix: The matrix to plot.
        std: The standard deviation to indicate on the colorbar.
        title: The title for the plot.
        subset: A subset of qubit indices to plot.

    """
    # Create a copy to avoid modifying the original matrix
    plot_matrix = matrix.copy()
    # Set diagonal to NaN to be colored white
    np.fill_diagonal(plot_matrix, np.nan)

    # Plot the matrix
    fig, ax = plt.subplots(figsize=(12, 10))
    # Use nanmax to ignore the diagonal NaNs when finding the max value
    vmax = np.nanmax(np.abs(plot_matrix))

    # Get the colormap and set the color for bad values (NaNs) to white
    cmap = plt.get_cmap("coolwarm")
    cmap.set_bad(color="white")

    im = ax.imshow(plot_matrix, cmap=cmap, vmin=-vmax, vmax=vmax, origin="lower")
    cbar = fig.colorbar(im, ax=ax, label="Error probability difference")

    # Add horizontal lines at ±std to indicate statistical uncertainty
    cbar.ax.axhline(y=std, color="black", linestyle="--", linewidth=1)
    cbar.ax.axhline(y=-std, color="black", linestyle="--", linewidth=1)

    num_qubits = matrix.shape[0]
    if subset is None:
        qubit_indices = np.arange(num_qubits)
    else:
        qubit_indices = np.array(subset)

    # Set ticks and labels for both axes to show all qubit numbers
    ax.set_xticks(range(len(qubit_indices)))
    ax.set_xticklabels(qubit_indices, rotation=90, fontsize=8)
    ax.set_yticks(range(len(qubit_indices)))
    ax.set_yticklabels(qubit_indices, fontsize=8)

    # Set minor ticks to be between pixels for a fine grid
    ax.set_xticks(np.arange(-0.5, num_qubits, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, num_qubits, 1), minor=True)

    # Add a grid for the minor ticks, which will be between the data points
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.2)

    ax.set_title(title)
    ax.set_xlabel("Reference Qubit Index")
    ax.set_ylabel("Measured Qubit Index")
    plt.show()


def _validate_correlation_inputs(
    correlation_matrix: np.ndarray,
    topology: QPUTopology,
    thresholds: tuple[float, float] | None,
) -> tuple[float, float] | None:
    """Validate correlation visualization inputs and return processed thresholds."""
    if thresholds is not None:
        if len(thresholds) != 2:  # noqa: PLR2004
            raise ValueError("Thresholds must be a tuple of (upper, lower) values.")
        upper_threshold, lower_threshold = thresholds
        if upper_threshold < 0.01 or lower_threshold > -0.01:  # noqa: PLR2004
            warnings.warn("Thresholds is set very low; plot may be cluttered and may take a lot of time to plot.")

    if correlation_matrix.size == 0:
        raise ValueError("Correlation matrix is empty.")

    if correlation_matrix.ndim != 2:  # noqa: PLR2004
        raise ValueError(f"Correlation matrix must be 2D, got {correlation_matrix.ndim}D array.")

    if correlation_matrix.shape[0] != correlation_matrix.shape[1]:
        raise ValueError(f"Correlation matrix must be square, got shape {correlation_matrix.shape}.")

    return thresholds


def _setup_qubit_labels_and_positions(
    topology: QPUTopology,
    qubit_labels: list[str] | None,
) -> tuple[list[str], list[str], dict[str, tuple[int, int]], dict[str, int], bool]:
    """Set up qubit labels, positions, and determine if working with subset.

    Returns:
        Tuple of (formatted qubit labels, all topology qubits, position mapping, qubit index mapping, is_subset flag)

    """
    all_qubits = topology.get_qubit_labels()
    positions = topology.positions

    if qubit_labels is None:
        formatted_qubits = all_qubits
    else:
        formatted_qubits = qubit_labels

    is_subset = len(formatted_qubits) < len(all_qubits)

    # Set up positions for all qubits in the topology
    pos = {}
    for qb in all_qubits:
        qb_num = QPUTopology.parse_qubit_index(qb)
        if qb_num in positions:
            pos[qb] = positions[qb_num]
        else:
            print(f"Warning: Qubit {qb} not found in topology positions")
            pos[qb] = (0, 0)

    # Create indices mapping for the qubits in our dataset
    qubit_indices = {qb: idx for idx, qb in enumerate(formatted_qubits)}

    return formatted_qubits, all_qubits, pos, qubit_indices, is_subset


def _build_significant_edges(
    formatted_qubits: list[str],
    correlation_matrix: np.ndarray,
    thresholds: tuple[float, float] | None,
) -> list[tuple[str, str]]:
    """Build list of edges with significant correlations based on thresholds."""
    significant_couplers = []
    for i, _ in enumerate(formatted_qubits):
        for j, _ in enumerate(formatted_qubits):
            if i == j:
                continue
            if thresholds:
                upper_threshold, lower_threshold = thresholds
                if correlation_matrix[i, j] >= upper_threshold or correlation_matrix[i, j] <= lower_threshold:
                    significant_couplers.append((formatted_qubits[i], formatted_qubits[j]))
            else:
                significant_couplers.append((formatted_qubits[i], formatted_qubits[j]))
    return significant_couplers


def _draw_correlation_edges(
    g: nx.DiGraph,
    ax: plt.Axes,
    fig: plt.Figure,
    pos: dict[str, tuple[int, int]],
    qubit_indices: dict[str, int],
    correlation_matrix: np.ndarray,
    vmax: float | None,
    edge_curvature: float,
) -> None:
    """Draw correlation edges with colors representing correlation strength."""
    edges = list(g.edges())
    if not edges:
        return

    # Collect weights and add to graph
    all_weights = []
    for u, v in edges:
        i, j = qubit_indices[u], qubit_indices[v]
        weight = correlation_matrix[i, j]
        all_weights.append(weight)
        g[u][v]["weight"] = weight

    if vmax is None:
        vmax = max(abs(np.min(all_weights)), abs(np.max(all_weights)))

    cmap = plt.get_cmap("coolwarm")
    norm = plt.Normalize(vmin=-vmax, vmax=vmax)

    # Draw curved edges with colors
    for u, v in edges:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        connection_style = f"arc3,rad={edge_curvature}"
        weight = correlation_matrix[qubit_indices[u], qubit_indices[v]]

        edge = ConnectionPatch(
            (x1, y1),
            (x2, y2),
            "data",
            "data",
            arrowstyle="->",
            shrinkA=5,
            shrinkB=5,
            mutation_scale=20,
            connectionstyle=connection_style,
            color=cmap(norm(weight)),
            linewidth=2,
            zorder=0,
        )
        ax.add_patch(edge)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, label="Covariance", shrink=0.8)
    cbar.set_label("Covariance", size=14)


def _prepare_readout_line_colors(
    topology: QPUTopology,
) -> tuple[dict[str, int], np.ndarray]:
    """Prepare qubit-to-readout-line mapping and color array."""
    control_lines = topology.control_lines
    qubit_to_readout_line = {}
    readout_colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(len(control_lines), 1)))

    for i, control_line in enumerate(control_lines):
        for qb in control_line:
            qubit_to_readout_line[qb] = i

    return qubit_to_readout_line, readout_colors


def _draw_qubit_nodes(
    ax: plt.Axes,
    all_qubits: list[str],
    formatted_qubits: list[str],
    pos: dict[str, tuple[int, int]],
    is_subset: bool,
    qubit_to_readout_line: dict[str, int],
    readout_colors: np.ndarray,
    control_lines: list[tuple[str, ...]],
) -> None:
    """Draw qubit nodes with appropriate styling based on subset status."""
    all_qubits_graph: nx.Graph = nx.Graph()
    all_qubits_graph.add_nodes_from(all_qubits)

    if is_subset:
        _draw_subset_nodes(
            ax,
            all_qubits_graph,
            all_qubits,
            formatted_qubits,
            pos,
            qubit_to_readout_line,
            readout_colors,
            control_lines,
        )
    else:
        _draw_full_nodes(ax, all_qubits_graph, all_qubits, pos, qubit_to_readout_line, readout_colors, control_lines)


def _draw_subset_nodes(
    ax: plt.Axes,
    graph: nx.Graph,
    all_qubits: list[str],
    formatted_qubits: list[str],
    pos: dict[str, tuple[int, int]],
    qubit_to_readout_line: dict[str, int],
    readout_colors: np.ndarray,
    control_lines: list[tuple[str, ...]],
) -> None:
    """Draw nodes when working with a subset of qubits."""
    non_input_qubits = [qb for qb in all_qubits if qb not in formatted_qubits]
    if non_input_qubits:
        nx.draw_networkx_nodes(
            graph,
            pos,
            ax=ax,
            nodelist=non_input_qubits,
            node_size=700,
            node_color="lightgrey",
            alpha=0.3,
        )

    readout_line_qubits_subset: dict[int, list[str]] = {i: [] for i in range(len(control_lines))}
    other_qubits = []

    for qb in formatted_qubits:
        if qb in qubit_to_readout_line:
            line_idx = qubit_to_readout_line[qb]
            readout_line_qubits_subset[line_idx].append(qb)
        else:
            other_qubits.append(qb)

    for line_idx, qubits in readout_line_qubits_subset.items():
        if qubits:
            nx.draw_networkx_nodes(
                graph,
                pos,
                ax=ax,
                nodelist=qubits,
                node_size=700,
                node_color=[readout_colors[line_idx]],
                alpha=0.8,
            )

    if other_qubits:
        nx.draw_networkx_nodes(
            graph,
            pos,
            ax=ax,
            nodelist=other_qubits,
            node_size=700,
            node_color="lightblue",
            alpha=0.8,
        )

    non_input_labels = {qb: qb for qb in non_input_qubits}
    input_labels = {qb: qb for qb in formatted_qubits}

    nx.draw_networkx_labels(
        graph,
        pos,
        ax=ax,
        labels=non_input_labels,
        font_size=8,
        font_color="gray",
        alpha=0.4,
    )
    nx.draw_networkx_labels(
        graph,
        pos,
        ax=ax,
        labels=input_labels,
        font_size=10,
        font_color="black",
        font_weight="bold",
    )


def _draw_full_nodes(
    ax: plt.Axes,
    graph: nx.Graph,
    all_qubits: list[str],
    pos: dict[str, tuple[int, int]],
    qubit_to_readout_line: dict[str, int],
    readout_colors: np.ndarray,
    control_lines: list[tuple[str, ...]],
) -> None:
    """Draw nodes for the full qubit set."""
    readout_line_qubits_all: dict[int, list[str]] = {i: [] for i in range(len(control_lines))}
    other_qubits = []

    for qb in all_qubits:
        if qb in qubit_to_readout_line:
            line_idx = qubit_to_readout_line[qb]
            readout_line_qubits_all[line_idx].append(qb)
        else:
            other_qubits.append(qb)

    for line_idx, qubits in readout_line_qubits_all.items():
        if qubits:
            nx.draw_networkx_nodes(
                graph,
                pos,
                ax=ax,
                nodelist=qubits,
                node_size=700,
                node_color=[readout_colors[line_idx]],
                alpha=0.8,
            )

    if other_qubits:
        nx.draw_networkx_nodes(
            graph,
            pos,
            ax=ax,
            nodelist=other_qubits,
            node_size=700,
            node_color="lightblue",
            alpha=0.8,
        )

    nx.draw_networkx_labels(
        graph,
        pos,
        ax=ax,
        font_size=10,
        font_color="black",
        font_weight="bold",
    )


def _finalize_correlation_plot(
    fig: plt.Figure,
    ax: plt.Axes,
    title: str | tuple[int, str],
    control_lines: list[tuple[str, ...]],
    readout_colors: np.ndarray,
    show_plot: bool,
) -> plt.Figure | None:
    """Add title, legend, and show/return the plot."""
    if isinstance(title, tuple):
        title_text = title[1]
    else:
        title_text = title

    ax.set_title(title_text, fontsize=16)
    ax.axis("on")

    if control_lines:
        for i in range(len(control_lines)):
            ax.scatter([], [], c=[readout_colors[i]], label=f"{i + 1}")
        ax.legend(loc="best", title="Readout Lines")

    plt.tight_layout()

    if show_plot:
        plt.show()
        return None
    return fig


def visualize_qubit_correlations_on_grid(
    correlation_matrix: np.ndarray,
    topology: QPUTopology,
    thresholds: tuple[float, float] | None = None,
    vmax: float | None = None,
    qubit_labels: list[str] | None = None,
    title: str | tuple[int, str] = "Qubit Correlations",
    edge_curvature: float = 0.2,
    show_plot: bool = True,
) -> plt.Figure | None:
    """Visualize readout error correlations overlaid on QPU topology.

    Creates a network graph showing qubits positioned according to their physical
    layout, with directed edges representing correlation strength and sign.
    Qubits are colored by readout line grouping.

    Args:
        correlation_matrix: NxN array of correlation coefficients. Element [i,j]
            represents correlation from qubit i to qubit j.
        topology: :class:`QPUTopology` defining qubit positions and readout lines.
            Use :func:`~iqm.error_reduction_tools.utils.topology_utils.topology_from_qc`
            to obtain a fully-populated instance from a connected quantum computer.
        thresholds: Optional (upper, lower) thresholds for edge filtering.
            Only correlations ≥upper or ≤lower are displayed.
            If ``None``, all edges are shown.
        vmax: Colormap saturation value. Correlations are mapped to
             [-vmax, +vmax] range (red=negative, blue=positive).
        qubit_labels: List of qubit labels corresponding to the correlation matrix rows/columns.
            If ``None``, assumes matrix represents all topology qubits in order.
            Can be a subset to visualize only specific qubits (unlisted qubits appear faded).
        title: Plot title. String or tuple (error_id, title_string).
        edge_curvature: Curvature radius for edges (0=straight, 0.2=default curve).
        show_plot: If ``True``, displays plot and returns ``None``. If ``False``, returns ``Figure``.

    Returns:
        Matplotlib figure if ``show_plot=False``, otherwise ``None``.

    .. note::

        Arrows point from dimension 1 to dimension 2 of correlation_matrix.
        E.g., arrow QB0→QB1 shows ``correlation_matrix[QB0_idx, QB1_idx]``.

    """
    # Validate inputs
    _validate_correlation_inputs(correlation_matrix, topology, thresholds)

    # Setup qubit labels and positions
    formatted_qubits, all_qubits, pos, qubit_indices, is_subset = _setup_qubit_labels_and_positions(
        topology,
        qubit_labels,
    )

    # Build significant edges based on thresholds
    significant_couplers = _build_significant_edges(formatted_qubits, correlation_matrix, thresholds)

    # Create graph and add edges
    g: nx.DiGraph = nx.DiGraph()
    g.add_edges_from(significant_couplers)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(14, 14))

    # Draw correlation edges
    _draw_correlation_edges(g, ax, fig, pos, qubit_indices, correlation_matrix, vmax, edge_curvature)

    # Prepare readout line colors
    qubit_to_readout_line, readout_colors = _prepare_readout_line_colors(topology)

    # Draw qubit nodes
    _draw_qubit_nodes(
        ax,
        all_qubits,
        formatted_qubits,
        pos,
        is_subset,
        qubit_to_readout_line,
        readout_colors,
        topology.control_lines,
    )

    # Finalize plot with title, legend, and show/return
    return _finalize_correlation_plot(
        fig,
        ax,
        title,
        topology.control_lines,
        readout_colors,
        show_plot,
    )


def plot_error_probabilities(
    error_data: ErrorProbabilities,
    title: str = "Readout Error Probabilities",
    show_plot: bool = True,
) -> plt.Figure | None:
    """Plot readout error probabilities.

    Wraps :func:`~iqm.error_reduction_tools.readout_characterization.data_processing.compute_error_probabilities`
    output.

    Accepts the dictionary returned by ``compute_error_probabilities`` and renders
    a grouped bar chart of P(1|0) and P(0|1) for each measured qubit, with optional
    error bars from the standard-deviation matrices.

    Args:
        error_data: Obtained from ``compute_error_probabilities``.
        title: Title for the plot.
        show_plot: If ``True``, displays the plot. If ``False``, returns the ``Figure``.

    Returns:
        Matplotlib figure if ``show_plot=False``, otherwise ``None``.

    """
    charact_data = error_data["charact_data"]
    measured_qubits = list(charact_data.keys())
    charact_data_std = error_data.get("charact_data_std")

    # Extract P(1|0) and P(0|1) arrays from the per-qubit matrices
    p_0to1 = np.array([charact_data[q][1, 0] for q in measured_qubits])
    p_1to0 = np.array([charact_data[q][0, 1] for q in measured_qubits])

    std_0to1 = None
    std_1to0 = None
    if charact_data_std is not None:
        std_0to1 = np.array([charact_data_std[q][1, 0] for q in measured_qubits])
        std_1to0 = np.array([charact_data_std[q][0, 1] for q in measured_qubits])

    x = np.arange(len(measured_qubits))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot bars with optional error bars
    ax.bar(
        x - width / 2,
        p_0to1,
        width,
        yerr=std_0to1 if std_0to1 is not None else None,
        label="P(1|0)",
        color="coral",
        edgecolor="black",
        capsize=3,
        error_kw={"elinewidth": 1, "capthick": 1},
    )
    ax.bar(
        x + width / 2,
        p_1to0,
        width,
        yerr=std_1to0 if std_1to0 is not None else None,
        label="P(0|1)",
        color="steelblue",
        edgecolor="black",
        capsize=3,
        error_kw={"elinewidth": 1, "capthick": 1},
    )

    ax.set_xlabel("Qubit", fontsize=12)
    ax.set_ylabel("Error Probability", fontsize=12)
    ax.set_title(title, fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(measured_qubits, rotation=45, ha="right")
    ax.legend()

    plt.tight_layout()

    if show_plot:
        plt.show()
        return None
    return fig


def plot_covariance_in_topology(
    covariance_data: CovarianceData,
    topology: QPUTopology,
    qubits_to_plot: list[str] | None = None,
    thresholds: tuple[float, float] | None = None,
    vmax: float | None = None,
    show_plot: bool = True,
) -> list[plt.Figure] | None:
    """Generate topology-based correlation plots for all error types.

    Wrapper around :func:`visualize_qubit_correlations_on_grid` that creates one plot
    per error category from covariance analysis functions. Designed to accept
    direct outputs from ``compute_*_covariance`` functions.

    Args:
        covariance_data: Obtained from the ``compute_*_covariance`` functions.
        topology: QPU topology for qubit positioning.
            Use :func:`~iqm.error_reduction_tools.utils.topology_utils.topology_from_qc`
            to obtain a fully-populated instance from a connected quantum computer.
        qubits_to_plot: Optional list of qubit labels to visualize. If ``None``, uses
            ``measured_qubits`` from ``covariance_data``.
        thresholds: Optional (upper, lower) correlation thresholds for edge filtering.
        vmax: Colormap saturation value for correlation strength.
        show_plot: If ``True``, displays all plots. If ``False``, returns ``Figure`` list.

    Returns:
        Matplotlib figures if ``show_plot=False``, otherwise ``None``.

    Example:
        >>> corr, labels, qubits = compute_double_twirled_covariance(data)
        >>> topology = topology_from_qc(client)
        >>> plot_covariance_in_topology(corr, labels, qubits, topology=topology, vmax=0.005)
        # Displays 3 topology plots (one per syndrome type)

    """
    correlation_matrices = cast(dict[ErrorId, np.ndarray], covariance_data["covariance_matrices"])
    error_labels = cast(list[tuple[ErrorId, str]], covariance_data["error_labels"])

    if not correlation_matrices:
        raise ValueError("'correlation_matrices' dictionary is empty.")

    if not error_labels:
        raise ValueError("'error_labels' list is empty.")

    if qubits_to_plot is None:
        qubits_to_plot = covariance_data["measured_qubits"]

    if not qubits_to_plot:
        raise ValueError("'qubits_to_plot' is empty and no 'measured_qubits' found in 'covariance_data'.")

    if len(error_labels) != len(correlation_matrices):
        raise ValueError(f"Mismatch: {len(error_labels)} error labels but {len(correlation_matrices)} matrices.")

    figures = []
    for error_id, label in error_labels:
        matrix = correlation_matrices[error_id]

        fig = visualize_qubit_correlations_on_grid(
            correlation_matrix=matrix,
            topology=topology,
            thresholds=thresholds,
            vmax=vmax,
            qubit_labels=qubits_to_plot,
            title=label,
            show_plot=show_plot,
        )
        if fig is not None:
            figures.append(fig)

    if not show_plot:
        return figures
    return None


def plot_covariance_heatmaps(
    covariance_data: CovarianceData,
    vmax: float | None = None,
    show_plot: bool = True,
) -> list[plt.Figure] | None:
    """Generate heatmap visualizations for all correlation matrices.

    Creates one heatmap per error category from covariance analysis.
    Provides a matrix view complementary to topology-based visualization.
    Designed to accept direct outputs from ``compute_*_covariance`` functions.

    Args:
        covariance_data: Obtained from the ``compute_*_covariance`` functions.
        vmax: Colormap saturation value. If ``None``, uses max absolute covariance
            per matrix.
        show_plot: If ``True``, displays all heatmaps. If ``False``, returns ``Figure`` list.

    Returns:
        Matplotlib figures if ``show_plot=False``, otherwise ``None``.

    Example:
        >>> result = compute_state_dependent_covariance(data)
        >>> plot_covariance_heatmaps(result, vmax=0.01)
        # Displays 12 heatmaps (one per error pair combination)

    """
    if not covariance_data:
        raise ValueError("'covariance_data' dictionary is empty.")

    correlation_matrices = cast(dict[ErrorId, np.ndarray], covariance_data["covariance_matrices"])
    error_labels = cast(list[tuple[ErrorId, str]], covariance_data["error_labels"])
    measured_qubits = covariance_data["measured_qubits"]

    if not correlation_matrices:
        raise ValueError("'covariance_matrices' is empty in 'covariance_data'.")

    if not error_labels:
        raise ValueError("'error_labels' is empty in 'covariance_data'.")

    if not measured_qubits:
        raise ValueError("'measured_qubits' is empty in 'covariance_data'.")

    if len(error_labels) != len(correlation_matrices):
        raise ValueError(f"Mismatch: {len(error_labels)} error labels but {len(correlation_matrices)} matrices.")

    # Validate that correlation matrices match measured_qubits dimensions
    for error_id, matrix in correlation_matrices.items():
        if matrix.shape[0] != len(measured_qubits) or matrix.shape[1] != len(measured_qubits):
            raise ValueError(
                f"Correlation matrix for error {error_id} has shape {matrix.shape}, "
                f"but measured_qubits has {len(measured_qubits)} qubits. "
                f"Matrix dimensions must match the number of qubits to plot."
            )

    figures = []
    for error_id, label in error_labels:
        matrix = correlation_matrices[error_id]

        if matrix.size == 0:
            raise ValueError(f"Correlation matrix for error {error_id} is empty.")

        if matrix.ndim != 2:  # noqa: PLR2004
            raise ValueError(f"Correlation matrix for error {error_id} must be 2D, got {matrix.ndim}D array.")

        fig, ax = plt.subplots(figsize=(12, 10))

        # Compute vmax per matrix if not provided globally
        matrix_vmax = vmax if vmax is not None else np.abs(matrix).max()

        sns.heatmap(
            matrix,
            ax=ax,
            cmap="coolwarm",
            center=0,
            vmin=-matrix_vmax,
            vmax=matrix_vmax,
            xticklabels=measured_qubits,
            yticklabels=measured_qubits,
            square=True,
            cbar_kws={"label": "Correlation Coefficient"},
        )

        ax.set_title(label, fontsize=16)
        plt.tight_layout()

        if show_plot:
            plt.show()
        else:
            figures.append(fig)

    if not show_plot:
        return figures
    return None
