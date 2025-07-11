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
"""The module for the edge-coloring of the graph.

This file contains code to find the edge coloring of an arbitrary simple graph according to the program defined in
:cite:`Misra_1992`. Almost all functions in this file are helper functions. Only :func:`find_edge_coloring` is intended
for external use. The function :func:`plot_edge_coloring` may be used for debugging purposes.
"""

from __future__ import annotations

from typing import Any

from iqm.qaoa.transpiler.quantum_hardware import LogEdge
import networkx as nx

Color = int


def ec_is_valid(graph: nx.Graph) -> bool:
    """Check that a graph's edge coloring is valid.

    Takes a graph whose edges have an attribute ``color`` and checks that no two neighboring edges have the same color
    (i.e., the coloring is valid).

    Args:
        graph: A :class:`~networkx.Graph` whose edges have an attribute ``color``.

    Returns:
        ``True`` if the edge coloring is valid, ``False`` otherwise.

    """
    for _, nbrsdict in graph.adjacency():
        color_lst = []
        for edge_attr in nbrsdict.values():
            if "color" in edge_attr:
                if edge_attr["color"] not in color_lst:
                    color_lst.append(edge_attr["color"])
                else:
                    return False

    return True


def ec_is_complete(graph: nx.Graph) -> bool:
    """Check that a graph's edge coloring is complete.

    Args:
        graph: A :class:`~networkx.Graph` whose edge coloring we want to check.

    Returns:
        ``True`` if every edge of the graph has an attribute ``color``, ``False`` otherwise.

    """
    for _, _, eattr in graph.edges.data("color"):  # type: ignore[var-annotated]
        if eattr is None:
            return False

    return True


def plot_edge_coloring(
    graph: nx.Graph,
    pos: dict[int, tuple[float, float]] | None = None,
    color_palette: list[str] | None = None,
    default_color: str = "#00000000",
) -> None:
    """A method that plots the edge coloring of a graph.

    Careful! Here the variable ``color_palette`` is a list of strings, to convey the colors for plotting the graph.
    It maps the colors as integers (i.e., indices of the list) into strings which actually describe the color to plot,
    e.g., 'r' for red. This is different from other function in this module where ``color_palette`` is a set of
    integers.

    Args:
        graph: A :class:`~networkx.Graph` whose edge coloring we want to plot.
        pos: A dictionary of positions for drawing ``graph`` (i.e., the layout). If not provided, it defaults to
            the circular layout.
        color_palette: A list of strings describing the colors to use in the plot.
        default_color: The color to use for edges that don't have a color assigned to them yet.

    """
    if pos is None:
        pos = nx.circular_layout(graph)

    if color_palette is None:
        color_palette = ["b", "g", "r", "c", "m", "y"]

    edge_color_arr = []
    for edge in graph.edges():
        if "color" in graph[edge[0]][edge[1]]:
            edge_color_arr.append(color_palette[graph[edge[0]][edge[1]]["color"]])
        else:
            edge_color_arr.append(default_color)

    nx.draw_networkx(graph, pos, edge_color=edge_color_arr)


def _colors_on_node(graph: nx.Graph, node: int) -> dict[Color, int]:
    """Function that lists the colors of edges neighboring a given node.

    Looks at all edges between ``node`` and its neighbors and collects their colors as keys of a dictionary. The values
    of the dictionary are the other end nodes of the edges.

    Args:
        graph: A :class:`~networkx.Graph` with an edge coloring.
        node: A node of the graph.

    Returns:
        A dictionary whose keys are colors and values the neighbors of ``node``.

    """
    color_arr = {}

    for neighbor in graph.neighbors(node):
        if "color" in graph.edges[node, neighbor]:
            color_arr[graph.edges[node, neighbor]["color"]] = neighbor

    return color_arr


def _color_is_free_on(graph: nx.Graph, node: int, color: Color) -> bool:
    """Checks if a color is 'free' on a given node.

    Looks at all the edges incident on ``node`` and checks if any of them is colored by ``color``.

    Args:
        graph: A :class:`~networkx.Graph` with an edge coloring.
        node: A node of ``graph``.
        color: A color.

    Returns:
        False if one of the edges incident on ``node`` has the color ``color``. True otherwise.

    """
    for neighbor in graph.neighbors(node):
        if "color" in graph.edges[node, neighbor]:
            if color is graph.edges[node, neighbor]["color"]:
                return False

    return True


def _free_colors(graph: nx.Graph, node: int, color_palette: set[Color]) -> set[Color]:
    """Checks which colors are 'free' on a given node.

    Looks at the colors of the edges incident on ``node`` and subtracts them from ``color_palette``, leaving only
    the colors from ``color_palette`` which are available to use at node ``node``.

    Args:
        graph: A :class:`~networkx.Graph` with an edge coloring.
        node: A node of ``graph``.
        color_palette: A set of all the available colors.

    Returns:
        A set of colors available to use at node ``node``.

    """
    return color_palette - _colors_on_node(graph, node).keys()


def _find_maximal_fan(graph: nx.Graph, node1: int, node2: int, color_palette: set[Color]) -> list[int]:
    """Finds the maximal fan corresponding to an ordered pair of nodes ``node1`` and ``node2``.

    A fan is a list of distinct neighbors of ``node1`` such that:
    1. The edge between ``node1`` and ``node2`` is uncolored.
    2. ``node2`` is ``fan[0]``.
    2. The color of the edge between ``node1`` and and ``fan[i]`` is free on ``fan[i-1]``.

    Args:
        graph: A :class:`~networkx.Graph` with an edge coloring.
        node1: The central node of the fan.
        node2: The starting node of the fan.
        color_palette: The set of available colors (as integers).

    Returns:
        The fan as a list of nodes.

    """
    if not graph.has_edge(node1, node2):
        raise ValueError(f"The nodes {node1} and {node2} aren't connected by an edge.")

    fan = [node2]

    found = True

    while found is True:
        found = False
        for color in _free_colors(graph, fan[-1], color_palette):
            if color in _colors_on_node(graph, node1):
                if _colors_on_node(graph, node1)[color] not in fan:
                    fan.append(_colors_on_node(graph, node1)[color])
                    found = True
                    break

    return fan


def _find_and_invert_cdpath(graph: nx.Graph, node: int, c: Color, d: Color) -> None:
    """Finds and inverts the cd-path associated with node ``node`` and the colors ``c`` and ``d``.

    It is assumed that the color ``c`` is free on ``node``. A cd-path is a path that includes the node ``node``,
    has edges colored only ``c`` or ``d`` and is maximal (cannot be extended). Inverting it swaps the colors ``c`` and
    ``d`` along the path, in order to make ``d`` free on ``node``. Note that the cd-path may be 'degenerate', i.e., it
    contains no other node beside ``node``. In that case inverting does nothing (and the color ``d`` is already free on
    ``node`` to begin with). The function potentially modifies the ``color`` of some of the graph edges.

    Args:
        graph: A :class:`~networkx.Graph` with an edge coloring.
        node: The starting point of the cd-path.
        c: One of the two colors in the cd-path.
        d: The other color in the cd-path.

    """
    current_color = d
    other_color = c
    path = [node]

    # Find
    while current_color in _colors_on_node(graph, path[-1]):
        path.append(_colors_on_node(graph, path[-1])[current_color])
        current_color, other_color = other_color, current_color

    # Invert
    for current_node, next_node in zip(path[:-1], path[1:]):
        if graph.edges[current_node, next_node]["color"] is c:
            graph.edges[current_node, next_node]["color"] = d
        else:
            graph.edges[current_node, next_node]["color"] = c


def _rotate_fan(graph: nx.Graph, node: int, fan: list[int], color: Color) -> None:
    """Rotates the colors of the ``fan`` and colors the last edge with ``color``.

    Takes a ``fan`` (i.e., a list of some neighbors of ``node``) and changes the color of the edge between ``node``
    and ``fan[i]`` to the color of the edge between ``node`` and ``fan[i+1]``, leaving the edge between ``node`` and
    ``fan[-1]`` uncolored. However, since the color ``color`` is free on ``node`` by assumption, we can now assign it
    to the edge between ``fan[-1]`` and ``node``. The function modifies the ``color`` of some of the graph edges.

    Args:
        graph: A :class:`~networkx.Graph` with an edge coloring.
        node: A node in ``graph``.
        fan: A fan around the node ``node``.
        color: A color to use on the edge between ``node`` and ``fan[-1]``.

    """
    for node1, node2 in zip(fan[:-1], fan[1:]):
        graph.edges[node, node1]["color"] = graph.edges[node, node2]["color"]

    graph.edges[node, fan[-1]]["color"] = color


def _color_edge(graph: nx.Graph, node1: int, node2: int, color_palette: set[Color]) -> None:
    """Colors the edge between ``node1`` and ``node2``.

    Combines the helper functions above to achieve the end result of coloring the edge between ``node1`` and ``node2``.
    In the process it also potentially re-colors a lot of the other already-colored edges of the graph.

    Args:
        graph: A :class:`~networkx.Graph` with an edge coloring.
        node1: An endpoint of the edge to be colored.
        node2: Another endpoint of the edge to be colored.
        color_palette: A set of colors to be used in the coloring.

    """
    fan = _find_maximal_fan(graph, node1, node2, color_palette)
    c_color = _free_colors(graph, node1, color_palette).pop()
    d_color = _free_colors(graph, fan[-1], color_palette).pop()
    _find_and_invert_cdpath(graph, node1, c_color, d_color)

    reduced_fan = []
    for node in fan:
        if d_color not in _free_colors(graph, node, color_palette):
            reduced_fan.append(node)
        else:
            reduced_fan.append(node)
            break

    _rotate_fan(graph, node1, reduced_fan, d_color)


def find_edge_coloring(input_graph: nx.Graph) -> tuple[list[set[LogEdge]], nx.Graph]:
    """This function finds an edge coloring for the given graph.

    It iterates over the graph edges and colors each one using elaborate helper functions. It modifies ``input_graph``
    by adding an attribute ``color`` to every edge, given by an integer number.

    Args:
        input_graph: The :class:`~networkx.Graph` to be edge-colored.

    Returns:
        A tuple containing:

        1. A list whose i-th entry contains the set of edges colored by the i-th color.
        2. A colored copy of the input ``input_graph``.

    """
    # Create a copy of the input graph, so that it's not modified in-place.
    graph = input_graph.copy()
    graph_degree = max(graph.degree(node) for node in graph)  # type: ignore[type-var]
    color_palette: set[Color] = set(range(graph_degree + 1))  # type: ignore[operator]
    for edge in graph.edges():
        _color_edge(graph, edge[0], edge[1], color_palette)

    color_sets: list[set[frozenset[Any]]] = [set() for _ in range(graph_degree + 1)]  # type: ignore[operator]
    for u, v, color in graph.edges.data("color"):  # type: ignore[var-annotated]
        color_sets[color].add(frozenset((u, v)))

    return color_sets, graph
