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

from collections.abc import Callable
from typing import Any, Literal

import networkx as nx
import numpy as np

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
    """Map original node labels of the :class:`~networkx.Graph` to new ones between 0 and `graph.number_of_nodes` - 1.

    Creates two dictionaries that keep track of the mapping between original labels and new labels numbered between 0
    and `graph.number_of_nodes` - 1.

    Args:
        graph: The graph whose nodes should be relabeled.

    Returns:
        A tuple containing the input graph with relabeled nodes and two dictionaries containing the mapping from old
        labels to new and vice versa.

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
