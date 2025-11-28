# Copyright (c) 2024-2025 IQM Quantum Computers
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted (subject to the
# limitations in the disclaimer below) provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this list of conditions and the following
#   disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
#   disclaimer in the documentation and/or other materials provided with the distribution.
# * Neither the name of IQM Quantum Computers nor the names of its contributors may be used to endorse or promote
#   products derived from this software without specific prior written permission.
#
# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY
# THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""Contains utility functions regarding graphs which are generic enough that they deserve having their own module."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Literal
import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

if TYPE_CHECKING:
    from iqm.applications.mis import ISInstance
    from iqm.applications.qubo import QUBOInstance

# Priority lists for node and edge attributes
NODE_ATTR_PRIORITY: tuple[str, ...] = ("bias", "weight", "field", "value", "h")
EDGE_ATTR_PRIORITY: tuple[str, ...] = ("bias", "weight", "coupling", "interaction", "J")


def _get_attr_with_priority(data: dict[str, int | float], priority_list: tuple[str, ...]) -> int | float | None:
    """Return the first attribute found in data according to priority_list, defaulting to 0.

    The input data is meant to be a dictionary of all attributes of a graph node or edge. In most realistic use cases it
    will have just one entry.

    Args:
        data: The dictionary containing the node / edge data to check.
        priority_list: The list of strings defining the priority in which the node / edge data should be retrieved.

    Returns:
        The value of the first found attribute.

    """
    for attr in priority_list:
        if attr in data:
            return data[attr]
    return None


def relabel_graph_nodes(graph: nx.Graph) -> tuple[nx.Graph, dict[Any, int], dict[int, Any]]:
    """Map original node labels of the :class:`~networkx.Graph` to consecutive integers starting from 0.

    Creates two dictionaries that keep track of the mapping between the original labels and the new labels.

    Args:
        graph: The graph whose nodes should be relabeled.

    Returns:
        A tuple containing the input graph with relabeled nodes and two dictionaries containing the mapping from the old
        labels to the new ones and vice versa.

    """
    orig_to_new_labels = {orig: new for new, orig in enumerate(graph.nodes())}
    new_to_orig_labels = dict(enumerate(graph.nodes()))
    if set(graph.nodes) != set(range(len(graph))):
        re_labeled_graph = nx.relabel_nodes(graph, orig_to_new_labels)
    else:
        re_labeled_graph = graph
    return re_labeled_graph, orig_to_new_labels, new_to_orig_labels


def _generate_desired_graph(
    graph_family: Literal["regular", "erdos-renyi"],
    n: int,
    p: float,
    d: int,
    seed: int | None | np.random.Generator = None,
    enforce_connected: bool = False,
    max_iterations: int = 1000,
) -> nx.Graph:
    """Wrapper helper function to encapsulate the graph generation logic.

    Args:
        graph_family: A string describing the random graph family to generate.
            Possible graph families include 'erdos-renyi' and 'regular'.
        n: The number of nodes of the graph.
        p: For the Erdős–Rényi graph, this is the edge probability. For other graph families, it's ignored.
        d: For the random regular graph, this is the degree of each node in the graph. For other graph families, it's
            ignored.
        seed: Optional random seed for the generation of the graphs.
        enforce_connected: True iff it is required that the random graphs are connected.
        max_iterations: In case ``enforce_connected`` is ``True``, the function generates random graphs in a ``while``
            loop until it finds a connected one. If it doesn't find a connected one after ``max_iterations``, it raises
            an error.

    Raises:
        ValueError: If incorrect / unknown `graph_family` is specified.
        RuntimeError: If ``enforce_connected`` is ``True`` and the algorithm doesn't find a connected graph in
            ``max_iterations`` iterations.

    Returns:
        Graph generated according to the provided parameters.

    """
    generators: dict[str, Callable[..., nx.Graph]] = {
        "erdos-renyi": lambda n, p, d, seed: nx.erdos_renyi_graph(n, p, seed=seed),
        "regular": lambda n, p, d, seed: nx.random_regular_graph(d, n, seed=seed),
    }

    if graph_family not in generators:
        raise ValueError("Invalid random graph type. Choose either 'regular' or 'erdos-renyi'.")

    rng = np.random.default_rng(seed=seed)
    for _ in range(max_iterations):
        g = generators[graph_family](n, p, d, rng)
        if not enforce_connected or nx.is_connected(g):
            return g

    raise RuntimeError(
        f"Failed to generate a connected graph after {max_iterations} attempts. "
        "Increase `max_iterations` or adjust graph parameters."
    )


# ============================================================
# Data structures
# ============================================================


@dataclass(frozen=True)
class ProblemData:
    """Normalized problem information extracted and validated before plotting."""

    graph: nx.Graph
    """The problem graph to be visualized."""
    orig_to_new_mapping: Mapping[Any, int]
    """Mapping of original variable names to integer node indices."""
    fixed_vars: frozenset[int]
    """Set of variables that were fixed in the optimization problem instance."""
    bitstring: str
    """Bitstring representing highlighted (active) nodes in the solution."""
    highlight_edge_by_node_count: frozenset[int]
    """Rule describing which edges to highlight based on how many connected nodes are highlighted."""


@dataclass(frozen=True)
class PlotData:
    """Precomputed visualization information used for drawing the graph."""

    graph: nx.Graph
    """The graph to be drawn."""
    pos: dict[Any, tuple[float, float]]
    """Layout positions of the nodes."""
    node_colors: list[str]
    """Colors for each node."""
    edge_colors: list[str]
    """Colors for each edge."""
    edge_widths: list[float]
    """Widths for each edge."""
    labels: dict[Any, str]
    """Labels for nodes."""
    edge_labels: dict[tuple[Any, Any], str]
    """Labels for edges (possibly empty)."""
    node_size: int
    """Scaled node size for plotting."""


# ============================================================
# Stage 1: Extract / Normalize Inputs
# ============================================================


def _extract_problem_info(
    problem_instance: QUBOInstance | ISInstance | None = None,
    *,
    graph_to_plot: nx.Graph | None = None,
    orig_to_new_mapping: Mapping[Any, int] | None = None,
    fixed_vars: frozenset[int] = frozenset(),
    bitstring: str | None = None,
    highlight_edge_by_node_count: frozenset[int] = frozenset({2}),
) -> ProblemData:
    """Extracts and validates problem-related data for graph visualization.

    This function handles both manually provided inputs and extraction from problem instances. It performs argument
    normalization and validation before the graph is prepared for plotting.

    See the docstring of :func:`draw_problem` for the documentation of the input args and the raised errors.

    Returns:
        A fully validated and normalized `ProblemData` object.

    """
    if problem_instance is not None:
        if bitstring is not None:
            for fixed_var in problem_instance._fixed_variables:
                if bitstring[fixed_var] != problem_instance._fixed_variables[fixed_var]:
                    warnings.warn(
                        f"The provided bitstring {bitstring} conflicts with one of the fixed variables of the problem"
                        f" instance {fixed_var}:{problem_instance._fixed_variables[fixed_var]}.",
                        stacklevel=2,
                    )
        if hasattr(problem_instance, "_graph"):
            graph_to_plot = graph_to_plot or problem_instance._graph  # ``or`` to avoid overwriting.
        elif hasattr(problem_instance, "qubo_graph"):
            # If using the QUBO graph, we relabel the nodes to their original names.
            # That's because QUBO graph comes from the internal BQM and that uses integers for variable names.
            graph_to_plot = graph_to_plot or nx.relabel_nodes(
                problem_instance.qubo_graph, problem_instance.new_to_orig_labels
            )
        else:  # This should never happen if ``problem_instance`` is the correct type.
            raise AttributeError(
                "The problem instance doesn't have a graph attached to it and ``graph_to_plot`` was not provided."
            )

        orig_to_new_mapping = orig_to_new_mapping or problem_instance.orig_to_new_labels  # ``or`` to avoid overwriting.
        fixed_vars = fixed_vars or frozenset(problem_instance._fixed_variables.keys())  # ``or`` to avoid overwriting.

        # This is ugly, but it avoids circular imports.
        if type(problem_instance).__name__ in {"MaxCutInstance", "WeightedMaxCutInstance"}:
            # This is the only choice that makes sense for maxcut -- highlight cut edges.
            highlight_edge_by_node_count = frozenset({1})

    if graph_to_plot is None or orig_to_new_mapping is None:
        raise ValueError(
            "The input variables `graph_to_plot` and `orig_to_new_mapping` must be provided or derived "
            "from provided problem instance."
        )

    if not highlight_edge_by_node_count <= {0, 1, 2}:
        raise ValueError(
            f"Invalid `highlight_edge_by_node_count`: {highlight_edge_by_node_count}. "
            "Must be a subset of `{0, 1, 2}`."
        )

    n_nodes = graph_to_plot.number_of_nodes()
    bitstring = bitstring or "0" * n_nodes
    if len(bitstring) != n_nodes:
        raise ValueError(f"Bitstring length {len(bitstring)} does not match number of nodes {n_nodes}.")

    return ProblemData(
        graph=graph_to_plot,
        orig_to_new_mapping=orig_to_new_mapping,
        fixed_vars=fixed_vars,
        bitstring=bitstring,
        highlight_edge_by_node_count=highlight_edge_by_node_count,
    )


# ============================================================
# Stage 2: Prepare Plot Data
# ============================================================


def prepare_plot_data(
    data: ProblemData,
    seed: int | None = None,
) -> PlotData:
    """Generates all layout and color data required for visualization.

    Args:
        data: Normalized problem data (output of :func:`_extract_problem_info`).
        seed: Optional seed to make the layout deterministic.

    Returns:
        A :class:`PlotData` object containing positions, colors, widths, and labels needed for rendering the graph.

    """
    graph = data.graph
    orig_to_new = data.orig_to_new_mapping
    fixed_vars = data.fixed_vars
    bitstring = data.bitstring
    highlight_edge_by_node_count = data.highlight_edge_by_node_count

    highlighted_nodes = {i for i, bit in enumerate(bitstring) if bit == "1"}

    # Compute layout.
    init_pos = nx.spring_layout(graph, seed=seed)
    # We don't want the layout to use the weight from the weighted problems, so we set ``weight`` to "None".
    pos = nx.kamada_kawai_layout(graph, pos=init_pos, weight="None")

    # Scale node size with number of nodes.
    base_size = 800
    n_nodes = graph.number_of_nodes()
    n_nodes_threshold: Final[int] = 20  # The number of nodes above which the node size for drawing is scaled down.
    node_size = base_size if n_nodes <= n_nodes_threshold else int(base_size * (n_nodes_threshold / n_nodes))

    # Node colors.
    color_map = {
        (True, True): "#F4834B",  # Highlighted & faint.
        (True, False): "#FF4500",  # Highlighted only.
        (False, True): "#E0E0E0",  # Faint only.
        (False, False): "#7bb2d6",  # Normal.
    }
    node_colors = [
        color_map[
            (
                orig_to_new[node] in highlighted_nodes,
                orig_to_new[node] in fixed_vars,
            )
        ]
        for node in graph.nodes()
    ]

    # Edge colors and widths.
    edge_colors: list[str] = []
    edge_widths: list[float] = []
    for u, v in graph.edges():
        n_highlighted = len({orig_to_new[u], orig_to_new[v]} & highlighted_nodes)
        is_highlight = n_highlighted in highlight_edge_by_node_count
        is_faint = bool({orig_to_new[u], orig_to_new[v]} & fixed_vars)

        if is_highlight and is_faint:
            edge_colors.append("#E18851")
            edge_widths.append(2.5)
        elif is_highlight:
            edge_colors.append("#FF4500")
            edge_widths.append(3)
        elif is_faint:
            edge_colors.append("#B0B0B0")
            edge_widths.append(2)
        else:
            edge_colors.append("#808080")
            edge_widths.append(2)

    # Node labels.
    labels: dict[Any, str] = {}
    for n, d in graph.nodes(data=True):
        val = _get_attr_with_priority(d, NODE_ATTR_PRIORITY)
        if val is not None:
            val_to_plot = f"{val:.2f}" if isinstance(val, float) else str(val)
            labels[n] = f"{n}\n({val_to_plot})"
        else:
            labels[n] = str(n)

    # Edge labels.
    edge_labels: dict[tuple[Any, Any], str] = {}
    for u, v, d in graph.edges(data=True):
        val = _get_attr_with_priority(d, EDGE_ATTR_PRIORITY)
        if val is not None:
            val_to_plot = f"{val:.2f}" if isinstance(val, float) else str(val)
            edge_labels[(u, v)] = val_to_plot

    return PlotData(
        graph=graph,
        pos=pos,
        node_colors=node_colors,
        edge_colors=edge_colors,
        edge_widths=edge_widths,
        labels=labels,
        edge_labels=edge_labels,
        node_size=node_size,
    )


# ============================================================
# Stage 3: Plotting
# ============================================================


def plot_graph(plot_data: PlotData) -> None:
    """Renders the graph using ``matplotlib`` and ``networkx``.

    Displays the resulting plot via ``plt.show()``.

    Args:
        plot_data: The prepared plot data, as returned by :func:`prepare_plot_data`.

    """
    # Networkx accepts lists of colors / widths, but it's not typed, so ``mypy`` gets confused.
    nx.draw_networkx_edges(
        plot_data.graph,
        plot_data.pos,
        edge_color=plot_data.edge_colors,  # type: ignore[arg-type]
        width=plot_data.edge_widths,  # type: ignore[arg-type]
    )
    nx.draw_networkx_nodes(
        plot_data.graph,
        plot_data.pos,
        node_color=plot_data.node_colors,
        node_size=plot_data.node_size,
        linewidths=0.5,
        edgecolors="black",
    )
    nx.draw_networkx_labels(plot_data.graph, plot_data.pos, labels=plot_data.labels, font_size=12)
    nx.draw_networkx_edge_labels(
        plot_data.graph,
        plot_data.pos,
        edge_labels=plot_data.edge_labels,
        font_size=10,
    )
    plt.axis("off")
    plt.show()
    plt.close()


# ============================================================
# Wrapper
# ============================================================


def draw_problem(  # noqa: D417
    problem_instance: QUBOInstance | ISInstance | None = None,
    *,
    graph_to_plot: nx.Graph | None = None,
    orig_to_new_mapping: Mapping[Any, int] | None = None,
    fixed_vars: frozenset[int] = frozenset(),
    bitstring: str | None = None,
    seed: int | None = None,
    highlight_edge_by_node_count: frozenset[int] = frozenset({2}),
) -> None:
    """High-level wrapper that prepares and visualizes a problem graph.

    This function orchestrates three steps:
      1. Extraction of problem data via :func:`_extract_problem_info`.
      2. Conversion into plotting data with :func:`prepare_plot_data`.
      3. Visualization using :func:`plot_graph`.

    It exists primarily as a convenience entry point — most argument validation and interpretation is handled by
    :func:`_extract_problem_info`.

    Args:
        problem_instance: Optional optimization problem instance from which graph and mapping data can be derived.
        graph_to_plot: Graph object to visualize. Overrides the graph from ``problem_instance``, if both are provided.
        orig_to_new_mapping: Mapping from original variable names to integer node indices. Overrides the mapping from
            ``problem_instance``, if both are provided.
        fixed_vars: Variables that have been fixed and should appear dimmed in the visualization. Overrides the fixed
            variables from ``problem_instance``, if both are provided.
        bitstring: Bitstring representing highlighted nodes in the graph.
        highlight_edge_by_node_count: Specifies which edges are highlighted based on the number of connected highlighted
            nodes (subset of ``{0, 1, 2}``).
        seed: Optional random seed for layout stability.

    Raises:
        AttributeError: If ``problem_instance`` is provided and it does not contain a valid graph (and a graph wasn't
            explicitly provided).
        ValueError: If ``graph_to_plot`` and ``orig_to_new_mapping`` are not both provided (or obtainable from provided
            ``problem_instance``).
        ValueError: If the provided ``highlight_edge_by_node_count`` is not a subset of ``{0, 1, 2}``.
        ValueError: If the provided ``bitstring`` has different length than the graph has nodes.

    """
    problem_data = _extract_problem_info(
        problem_instance=problem_instance,
        graph_to_plot=graph_to_plot,
        orig_to_new_mapping=orig_to_new_mapping,
        fixed_vars=fixed_vars,
        bitstring=bitstring,
        highlight_edge_by_node_count=highlight_edge_by_node_count,
    )
    plot_data = prepare_plot_data(problem_data, seed=seed)
    plot_graph(plot_data)


def residual_degree(graph: nx.Graph, node: Any, visited: set[Any]) -> int:
    """The degree of ``node`` in the graph, with regards to only non-visited nodes.

    Args:
        graph: The graph which we're working with.
        node: The node whose reduced degree we want to calculate.
        visited: The set of all visited nodes up to now.

    Returns:
        The residual degree of the node.

    """
    return sum(1 for nbr in graph.neighbors(node) if nbr not in visited)
