#  ********************************************************************************
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
#  ********************************************************************************
"""Edge coloring functions and algorithms."""

from networkx import Graph, MultiGraph, connected_components, set_edge_attributes  # noqa: F401
from networkx.algorithms import bipartite


def color_qubit_pairs(qubit_pairs: list[tuple[str, str]]) -> list[list[tuple[str, str]]]:
    """Create an edge-coloring of the given graph with minimum number of colors:
    any edges connected with a vertex do not share the same color.

    If graph is bipartite, we use a simple edge-coloring algorithm which gives a coloring using d number of colors,
    where d = maximum degree of a vertex in the graph.
    If graph is not bipartite, we use the Misra-Gries algorithm, which gives a coloring with d+1 number of colors.

    Works for both simple and multi-graphs (parallel edges), for connected and disconnected graphs,
    for bipartite and non-bipartite graphs.

    Args:
        qubit_pairs: list of tuples of two qubits

    Returns:
        List of sublists of tuples of two qubits. All pairs in a given sublist correspond to a given color.

    """
    # create a networkx graph G
    G = MultiGraph()  # type: ignore[var-annotated]
    G.add_edges_from(qubit_pairs)  # construct it from the input edges
    # initialize G to set all colors to 0
    set_edge_attributes(G, 0, "color")  # type: ignore[call-overload]

    # d = degree of graph G
    # Since G.degree gives individual degrees as a list in the form (node, degree), we want the
    #  maximum over the elements in index 1. Since this gives us the tuple, we want index 1 of that.
    d = max(list(G.degree), key=lambda x: x[1])[1]

    if bipartite.is_bipartite(G):  # use networkx function to check if graph is bipartite
        # Construct a set of colors [1, 2, ..., d]
        colors = list(range(1, d + 1))  # type: ignore[call-overload]
        G = bipartite_alg(G, colors)

    else:
        # Construct a set of colors [1, 2, ..., d, d+1]
        colors = list(range(1, d + 2))  # type: ignore[call-overload]
        G = misra_gries_alg(G, colors)

    # Make a list of sublists of tuples, each sublist corresponding to a different color
    color_sorted_edges = []
    for c in colors:
        # list of edges of color c
        c_edges_color = [(u, v, i, x) for u, v, i, x in G.edges(data=True, keys=True) if (x["color"] == c)]
        # make sure we write edge nodes in same order as original
        c_edges = [(u, v) if (u, v) in qubit_pairs else (v, u) for u, v, _, _ in list(c_edges_color)]

        color_sorted_edges.append(c_edges)  # add this list to the total list

    # Post-process the edges we show to make sure they're in the same form as the given qubit pairs.
    # This is necessary because networkx sometimes swaps the order of the nodes in an edge, and we want to keep them
    # same as the original list of edges.
    for edge in qubit_pairs:  # for every edge in the original list of qubit pairs...
        if not any(edge in sublist for sublist in color_sorted_edges):  # if it doesn't appear in total list...
            wrong_edge = (edge[1], edge[0])  # since original edge isn't there, the edge with wrong node order is there
            # list shows which sublist contains wrong edge, formatted [False, False, ..., True, ..,]
            is_edge_here = [wrong_edge in sublist for sublist in color_sorted_edges]
            sublist_index = is_edge_here.index(True)  # index of the sublist in total list where wrong edge exists

            # replace the wrong edge with the right edge in that sublist
            color_sorted_edges[sublist_index].append(edge)
            color_sorted_edges[sublist_index].remove(wrong_edge)

    return color_sorted_edges


"""---------------- Bipartite case ------------------"""


def bipartite_alg(G: MultiGraph, colors: list[int]) -> MultiGraph:
    """Algorithm to create edge-coloring for a bipartite graph.
    Bottom of this page for reference: https://codeforces.com/blog/entry/21827.
    """
    for e in list(G.edges(keys=True)):  # for every edge in the graph...
        u = e[0]  # first vertex of edge e
        v = e[1]  # second vertex of edge e
        c = _free_color_in_e(G, u, v, colors)  # find a color c that is free in both vertices
        if c != 0:  # if we found a free color...
            G.edges[e]["color"] = c  # set the edge to that color

        else:  # if we did not find a free color for both vertices...
            c_1, c_2 = _find_c_1_and_c_2(G, u, v, colors)  # find c_1 and c_2 (see function for details)
            _free_color_chain(G, e, c_1, c_2)  # recursively swap alternating colors until free

            # In the end, we set e to c_2 and the connecting edge that previously had c_2 is now c_1,
            # prev connecting edge that had c_1 is now c_2, and so on until we find a free spot for current color.

    return G


def _free_color_in_e(G: MultiGraph, u: str, v: str, colors: list[int]) -> int:
    """Returns a color that is free in both vertices u and v. Else returns 0.
    Used in bipartite alg.
    """
    for c in colors:  # for each color...
        # list all edges with color c incident to vertices u and v, respectively
        u_edges = [
            (a, b, i, x)
            for a, b, i, x in G.edges(data=True, keys=True)
            if (x["color"] == c and (u in (a, b)))  # noqa: PLR1714
        ]
        v_edges = [
            (a, b, i, x)
            for a, b, i, x in G.edges(data=True, keys=True)
            if (x["color"] == c and (v in (a, b)))  # noqa: PLR1714
        ]

        if not u_edges and not v_edges:  # if both lists are empty...
            return c  # c is a color that is free in both u and v

    return 0  # otherwise, there are no colors free in both u and v


def _find_c_1_and_c_2(G: MultiGraph, u: str, v: str, colors: list[int]) -> tuple[int, int]:
    """Returns a tuple of two colors (c_1, c_2) such that c_1 is free in v and not free in u,
    and c_2 is free in u and not free in v.
    Used in bipartite alg.
    """
    c_1 = 0
    c_2 = 0

    for c in colors:  # for each color...
        # list all edges with color c incident to vertices u and v, respectively
        u_edges = [
            (a, b, i, x)
            for a, b, i, x in G.edges(data=True, keys=True)
            if (x["color"] == c and (u in (a, b)))  # noqa: PLR1714
        ]
        v_edges = [
            (a, b, i, x)
            for a, b, i, x in G.edges(data=True, keys=True)
            if (x["color"] == c and (v in (a, b)))  # noqa: PLR1714
        ]

        if u_edges and not v_edges:  # if this color is free in v and not free in u...
            c_1 = c
        if not u_edges and v_edges:  # if this color is free in u and not free in v...
            c_2 = c

        if c_1 != 0 and c_2 != 0:  # if we found both colors already...
            break  # we're done here

    return c_1, c_2


def _is_free(G: MultiGraph, v: str, c: int) -> bool:
    """Checks if color c is free on vertex v in graph G."""
    # list all edges with color c incident to vertex v
    v_edges = [(a, b, i, x) for a, b, i, x in G.edges(data=True, keys=True) if (x["color"] == c and (v in (a, b)))]  # noqa: PLR1714

    if not v_edges:  # if there are no edges with color c incident to v...  # noqa: SIM103
        return True  # v is free of c!

    return False


def _free_color_chain(G: MultiGraph, e: tuple[str, str, int], c_1: int, c_2: int):  # noqa: ANN202
    """Recursively swap colors alternating c_1 and c_2 in a chain starting at the edge between vertices u and v,
    until we reach a point where the current color c_2 is free in current vertex v.
    Used in bipartite alg.
    """
    v = e[1]

    if _is_free(G, v, c_2):  # if c_2 is free in v
        G.edges[e]["color"] = c_2  # set the current edge to c_2
        return 0

    # while c_2 is not free in v...
    # next_edge is the edge between v and next_vertex
    next_edge = [
        (a, b, i)
        for a, b, i, x in G.edges(data=True, keys=True)
        if (x["color"] == c_2 and (v in (a, b)))  # noqa: PLR1714
    ][0]
    if next_edge[1] == v:  # if v is the second listed node...
        next_edge = (next_edge[1], next_edge[0], next_edge[2])  # swap the order of the nodes to make v first

    _free_color_chain(G, next_edge, c_2, c_1)  # notice that we swapped the order of c_1 and c_2

    G.edges[e]["color"] = c_2  # set the current edge to c_2 when we backtrack


"""-------------- Non-bipartite case ----------------"""


def misra_gries_alg(G: MultiGraph, colors: list[int]) -> MultiGraph:
    """Algorithm to create edge-coloring for a non-bipartite graph.
    Link for reference: https://en.wikipedia.org/wiki/Misra_%26_Gries_edge_coloring_algorithm.
    """
    for e in list(G.edges(keys=True)):  # for every edge in the graph...
        u = e[0]

        fan = _max_fan(G, u, e)  # maximal fan (list of edges) of u with fan[0] = e

        c = _free_color_in_v(G, u, colors)  # c is a color that is free on u
        d = _free_color_in_v(G, fan[-1][1], colors)  # d is a color that is free on the fan's last edge's vertex

        _invert_c_d_u_path(G, u, c, d)

        # f is an edge in fan, fan' = [fan[0]...f] is a fan (given), and d is free on other vertex in f (f[1])
        f = None
        fan_prime = []
        for edge in fan:
            fan_prime.append(edge)  # fan' is a subfan of fan, we construct it on the way
            if _is_free(G, edge[1], d):
                f = edge
                break  # we found our f and constructed fan', so we're done here

        # rotate fan' and set color of edge f to d
        _rotate_fan(G, fan_prime)
        G.edges[f]["color"] = d  # type: ignore[index]

    return G


def _free_color_in_v(G: MultiGraph, v: str, colors: list[int]) -> int:
    """Returns a color that is free in v. Else returns 0.
    Used in Misra-Gries alg.
    """
    for c in colors:  # for each color...
        # list all edges with color c incident to vertex v
        v_edges = [
            (a, b, i, x)
            for a, b, i, x in G.edges(data=True, keys=True)
            if (x["color"] == c and (v in (a, b)))  # noqa: PLR1714
        ]
        if not v_edges:  # if there are no edges with that color
            return c  # this is a color free in v

    return 0  # there are no colors free in v (I hope there are though)


def _max_fan(G: MultiGraph, u: str, e: tuple[str, str, int]) -> list[tuple[str, str, int]]:
    """Find the maximal fan (list of edges) of vertex u.
    Criteria:
        - Starting edge fan[0] = e = (u, v).
        - Edge fan[0] = (u, v) is uncolored. (given)
        - The color of edge fan[i+1] is free on fan[i][1] for all 0 <= i < len(fan).
    Used in Misra-Gries alg.
    """
    # we want all colored edges of u
    u_edges = [
        (a, b, i) if (a == u) else (b, a, i)  # we want to make sure the edges all start with vertex u
        for a, b, i, x in G.edges(data=True, keys=True)
        if (x["color"] != 0 and (u in (a, b)))  # noqa: PLR1714
    ]
    fan = [e]  # the fan is a list of edges starting with edge (v, u).

    # We need to add as many edges as we can such that the next edge is free on the current edge's vertex v.

    found_free = True  # if inner loop does not find an edge with color free in the previous vertex, we stop
    while len(u_edges) > 0 and found_free:
        found_free = False  # we will set it to True once we find the edge
        for u_edge in u_edges:
            if _is_free(
                G, fan[-1][1], G.edges[u_edge]["color"]
            ):  # if current edge color is free on prev edge's vertex...
                fan.append(u_edge)  # add the edge
                u_edges.remove(u_edge)  # remove from list so that we don't go over it again

                found_free = True
                break  # we can go on to find the next vertex in fan
        # If we did not find any edges with a color that's free in prev vertex,
        # found_free will be False, and we end the while loop.

    return fan


def _invert_c_d_u_path(G: MultiGraph, u: str, c: int, d: int):  # noqa: ANN202
    """Find the longest path through vertex u that has only edges colored c or d.
    Then recolor all c-colored edges on the path into color d, and all d-colored edges on the path into color c.
    Used in Misra-Gries alg.
    """
    # First, we find the max length c d path through vertex u.

    # all edges colored c or d
    c_d_edges = [(a, b, i, x) for a, b, i, x in G.edges(data=True, keys=True) if (x["color"] == c or x["color"] == d)]
    # make a graph out of all edges colored c or d, so we can use connected_components to find a path
    c_d_graph = MultiGraph()  # type: ignore[var-annotated]
    c_d_graph.add_edges_from(c_d_edges)  # type: ignore[arg-type]
    all_max_c_d_paths = connected_components(c_d_graph)  # list of sets of connected vertices with edges color c or d

    for node_set in [path for path in all_max_c_d_paths if u not in path]:  # remove paths that aren't through u
        c_d_graph.remove_nodes_from(node_set)

    # At this point, c_d_graph contains only the max length c d path through u.

    # Now we invert the path in the original graph G.

    for e in list(c_d_graph.edges):
        G.edges[e]["color"] = d if G.edges[e]["color"] == c else c  # swap the colors


def _rotate_fan(G: MultiGraph, fan: list[tuple[str, str, int]]):  # noqa: ANN202
    """Rotates the fan as such:
        - Set the color of edge fan[i] to the color of fan[i+1]
        - Uncolor edge fan[-1]
    Used in Misra-Gries alg.
    """
    for i in range(0, len(fan) - 1):
        G.edges[fan[i]]["color"] = G.edges[fan[i + 1]]["color"]  # set color of edge to the color of the next one

    G.edges[fan[-1]]["color"] = 0  # uncolor the last edge
