# Copyright 2024 IQM Benchmarks developers
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

"""Plotting and visualization utility functions."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, cast

from iqm.benchmarks.entanglement.ghz_optimal import TreeSearchOptions, generate_ghz_tree_optimal
from iqm.benchmarks.logging_config import qcvv_logger
from iqm.benchmarks.utils import (
    extract_fidelities_unified,
    process_backend,
)
from iqm.qiskit_iqm.iqm_backend import IQMBackendBase
from iqm.qiskit_iqm.iqm_provider import IQMBackend
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyBboxPatch
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from qiskit.transpiler import CouplingMap
from rustworkx import (
    PyGraph,
    spring_layout,
    visualization,
)


@dataclass
class GraphPositions:
    """A class to store and generate graph positions for different chip layouts.

    This class contains predefined node positions for various quantum chip topologies and
    provides methods to generate positions for different layout types.

    Attributes:
        r5_positions:
            Mapping of node indices to (x,y) positions for the 5-qubit crystal chip.
        r20_positions:
            Mapping of node indices to (x,y) positions for the 20-qubit crystal chip.
        r54_positions:
            Mapping of node indices to (x,y) positions for the 54-qubit crystal chip.
        r150_positions:
            Mapping of node indices to (x,y) positions for the 150-qubit crystal chip.
        n24_positions:
            Mapping of node indices to (x,y) positions for the 24-qubit star (resonator) chip.
        n6_positions:
            Mapping of node indices to (x,y) positions for the 6-qubit Deneb (resonator) chip.
        predefined_stations:
            Mapping of station names to their position dictionaries.

    """

    r5_positions = {
        0: (0.0, 0.0),
        1: (2.0, 0.0),
        2: (1.0, 1.0),
        3: (0.0, 2.0),
        4: (2.0, 2.0),
    }

    r20_positions = {
        0: (5.0, 7.0),
        1: (6.0, 6.0),
        2: (3.0, 7.0),
        3: (4.0, 6.0),
        4: (5.0, 5.0),
        5: (6.0, 4.0),
        6: (7.0, 3.0),
        7: (2.0, 6.0),
        8: (3.0, 5.0),
        9: (4.0, 4.0),
        10: (5.0, 3.0),
        11: (6.0, 2.0),
        12: (1.0, 5.0),
        13: (2.0, 4.0),
        14: (3.0, 3.0),
        15: (4.0, 2.0),
        16: (5.0, 1.0),
        17: (1.0, 3.0),
        18: (2.0, 2.0),
        19: (3.0, 1.0),
    }

    r54_positions = {
        0: (10, 10),
        1: (11, 9),
        2: (7, 11),
        3: (8, 10),
        4: (9, 9),
        5: (10, 8),
        6: (11, 7),
        7: (5, 11),
        8: (6, 10),
        9: (7, 9),
        10: (8, 8),
        11: (9, 7),
        12: (10, 6),
        13: (11, 5),
        14: (3, 11),
        15: (4, 10),
        16: (5, 9),
        17: (6, 8),
        18: (7, 7),
        19: (8, 6),
        20: (9, 5),
        21: (10, 4),
        22: (2, 10),
        23: (3, 9),
        24: (4, 8),
        25: (5, 7),
        26: (6, 6),
        27: (7, 5),
        28: (8, 4),
        29: (9, 3),
        30: (10, 2),
        31: (2, 8),
        32: (3, 7),
        33: (4, 6),
        34: (5, 5),
        35: (6, 4),
        36: (7, 3),
        37: (8, 2),
        38: (9, 1),
        39: (1, 7),
        40: (2, 6),
        41: (3, 5),
        42: (4, 4),
        43: (5, 3),
        44: (6, 2),
        45: (7, 1),
        46: (1, 5),
        47: (2, 4),
        48: (3, 3),
        49: (4, 2),
        50: (5, 1),
        51: (1, 3),
        52: (2, 2),
        53: (3, 1),
    }

    r150_positions = {
        # Group QB1-QB15 (indices 0-14): column pair x=0 (odd QBs) / x=1 (even QBs)
        0: (0.0, 14.0),
        1: (1.0, 13.0),
        2: (0.0, 12.0),
        3: (1.0, 11.0),
        4: (0.0, 10.0),
        5: (1.0, 9.0),
        6: (0.0, 8.0),
        7: (1.0, 7.0),
        8: (0.0, 6.0),
        9: (1.0, 5.0),
        10: (0.0, 4.0),
        11: (1.0, 3.0),
        12: (0.0, 2.0),
        13: (1.0, 1.0),
        14: (0.0, 0.0),
        # Group QB16-QB30 (indices 15-29): column pair x=2 / x=3
        15: (2.0, 14.0),
        16: (3.0, 13.0),
        17: (2.0, 12.0),
        18: (3.0, 11.0),
        19: (2.0, 10.0),
        20: (3.0, 9.0),
        21: (2.0, 8.0),
        22: (3.0, 7.0),
        23: (2.0, 6.0),
        24: (3.0, 5.0),
        25: (2.0, 4.0),
        26: (3.0, 3.0),
        27: (2.0, 2.0),
        28: (3.0, 1.0),
        29: (2.0, 0.0),
        # Group QB31-QB45 (indices 30-44): column pair x=4 / x=5
        30: (4.0, 14.0),
        31: (5.0, 13.0),
        32: (4.0, 12.0),
        33: (5.0, 11.0),
        34: (4.0, 10.0),
        35: (5.0, 9.0),
        36: (4.0, 8.0),
        37: (5.0, 7.0),
        38: (4.0, 6.0),
        39: (5.0, 5.0),
        40: (4.0, 4.0),
        41: (5.0, 3.0),
        42: (4.0, 2.0),
        43: (5.0, 1.0),
        44: (4.0, 0.0),
        # Group QB46-QB60 (indices 45-59): column pair x=6 / x=7
        45: (6.0, 14.0),
        46: (7.0, 13.0),
        47: (6.0, 12.0),
        48: (7.0, 11.0),
        49: (6.0, 10.0),
        50: (7.0, 9.0),
        51: (6.0, 8.0),
        52: (7.0, 7.0),
        53: (6.0, 6.0),
        54: (7.0, 5.0),
        55: (6.0, 4.0),
        56: (7.0, 3.0),
        57: (6.0, 2.0),
        58: (7.0, 1.0),
        59: (6.0, 0.0),
        # Group QB61-QB75 (indices 60-74): column pair x=8 / x=9
        60: (8.0, 14.0),
        61: (9.0, 13.0),
        62: (8.0, 12.0),
        63: (9.0, 11.0),
        64: (8.0, 10.0),
        65: (9.0, 9.0),
        66: (8.0, 8.0),
        67: (9.0, 7.0),
        68: (8.0, 6.0),
        69: (9.0, 5.0),
        70: (8.0, 4.0),
        71: (9.0, 3.0),
        72: (8.0, 2.0),
        73: (9.0, 1.0),
        74: (8.0, 0.0),
        # Group QB76-QB90 (indices 75-89): column pair x=10 / x=11
        75: (10.0, 14.0),
        76: (11.0, 13.0),
        77: (10.0, 12.0),
        78: (11.0, 11.0),
        79: (10.0, 10.0),
        80: (11.0, 9.0),
        81: (10.0, 8.0),
        82: (11.0, 7.0),
        83: (10.0, 6.0),
        84: (11.0, 5.0),
        85: (10.0, 4.0),
        86: (11.0, 3.0),
        87: (10.0, 2.0),
        88: (11.0, 1.0),
        89: (10.0, 0.0),
        # Group QB91-QB105 (indices 90-104): column pair x=12 / x=13
        90: (12.0, 14.0),
        91: (13.0, 13.0),
        92: (12.0, 12.0),
        93: (13.0, 11.0),
        94: (12.0, 10.0),
        95: (13.0, 9.0),
        96: (12.0, 8.0),
        97: (13.0, 7.0),
        98: (12.0, 6.0),
        99: (13.0, 5.0),
        100: (12.0, 4.0),
        101: (13.0, 3.0),
        102: (12.0, 2.0),
        103: (13.0, 1.0),
        104: (12.0, 0.0),
        # Group QB106-QB120 (indices 105-119): column pair x=14 / x=15
        105: (14.0, 14.0),
        106: (15.0, 13.0),
        107: (14.0, 12.0),
        108: (15.0, 11.0),
        109: (14.0, 10.0),
        110: (15.0, 9.0),
        111: (14.0, 8.0),
        112: (15.0, 7.0),
        113: (14.0, 6.0),
        114: (15.0, 5.0),
        115: (14.0, 4.0),
        116: (15.0, 3.0),
        117: (14.0, 2.0),
        118: (15.0, 1.0),
        119: (14.0, 0.0),
        # Group QB121-QB135 (indices 120-134): column pair x=16 / x=17
        120: (16.0, 14.0),
        121: (17.0, 13.0),
        122: (16.0, 12.0),
        123: (17.0, 11.0),
        124: (16.0, 10.0),
        125: (17.0, 9.0),
        126: (16.0, 8.0),
        127: (17.0, 7.0),
        128: (16.0, 6.0),
        129: (17.0, 5.0),
        130: (16.0, 4.0),
        131: (17.0, 3.0),
        132: (16.0, 2.0),
        133: (17.0, 1.0),
        134: (16.0, 0.0),
        # Group QB136-QB150 (indices 135-149): column pair x=18 / x=19
        135: (18.0, 14.0),
        136: (19.0, 13.0),
        137: (18.0, 12.0),
        138: (19.0, 11.0),
        139: (18.0, 10.0),
        140: (19.0, 9.0),
        141: (18.0, 8.0),
        142: (19.0, 7.0),
        143: (18.0, 6.0),
        144: (19.0, 5.0),
        145: (18.0, 4.0),
        146: (19.0, 3.0),
        147: (18.0, 2.0),
        148: (19.0, 1.0),
        149: (18.0, 0.0),
    }

    n24_positions = {
        # Even nodes on the bottom
        2: (1.0, 5.0),
        4: (3.0, 5.0),
        6: (5.0, 5.0),
        8: (7.0, 5.0),
        10: (9.0, 5.0),
        12: (11.0, 5.0),
        14: (13.0, 5.0),
        16: (15.0, 5.0),
        18: (17.0, 5.0),
        20: (19.0, 5.0),
        22: (21.0, 5.0),
        24: (23.0, 5.0),
        # Odd nodes on the top
        1: (1.0, 1.0),
        3: (3.0, 1.0),
        5: (5.0, 1.0),
        7: (7.0, 1.0),
        9: (9.0, 1.0),
        11: (11.0, 1.0),
        13: (13.0, 1.0),
        15: (15.0, 1.0),
        17: (17.0, 1.0),
        19: (19.0, 1.0),
        21: (21.0, 1.0),
        23: (23.0, 1.0),
    }
    # Add dummy indices for the resonator
    max_qubit_number = np.max(list(n24_positions.keys()))
    previous_nodes = list(n24_positions.keys())
    for node in previous_nodes:
        n24_positions.update({node + max_qubit_number: (n24_positions[node][0], 3)})
    # Node 0 in the middle
    n24_positions.update({0: (16.5, 3.0)})

    n6_positions = {
        # Even nodes on the bottom
        2: (1.0, 5.0),
        4: (3.0, 5.0),
        6: (5.0, 5.0),
        # Odd nodes on the top
        1: (1.0, 1.0),
        3: (3.0, 1.0),
        5: (5.0, 1.0),
    }
    # Add dummy indices for the resonator
    max_qubit_number = np.max(list(n6_positions.keys()))
    previous_nodes = list(n6_positions.keys())
    for node in previous_nodes:
        n6_positions.update({node + max_qubit_number: (n6_positions[node][0], 3)})
    # Node 0 in the middle
    n6_positions.update({0: (2.5, 3.0)})

    predefined_stations = {
        "r5": r5_positions,
        "n6": n6_positions,
        "r20": r20_positions,
        "n24": n24_positions,
        "r54": r54_positions,
        "r150": r150_positions,
    }

    @staticmethod
    def create_positions(
        graph: PyGraph, topology: Literal["star", "crystal"] | None = None
    ) -> dict[int, tuple[float, float]]:
        """Generate node positions for a given graph and topology.

        Args:
            graph: The graph to generate positions for.
            topology: The type of layout to generate. Must be either "star" or "crystal".

        Returns:
            A dictionary mapping node indices to (x,y) coordinates.

        """
        n_nodes = len(graph.node_indices())

        if topology == "star":
            # Place center node at (0,0)
            pos = {0: (0.0, 0.0)}

            if n_nodes > 1:
                # Place other nodes in a circle around the center
                angles = np.linspace(0, 2 * np.pi, n_nodes - 1, endpoint=False)
                radius = 1.0

                for i, angle in enumerate(angles, start=1):
                    x = radius * np.cos(angle)
                    y = radius * np.sin(angle)
                    pos[i] = (x, y)

        # Crystal and other topologies
        else:
            # Fix first node position in bottom right
            fixed_pos = {0: (1.0, 1.0)}  # For more consistent layouts

            # Get spring layout with one fixed position
            pos = {
                int(k): (float(v[0]), float(v[1]))
                for k, v in spring_layout(graph, scale=2, pos=fixed_pos, num_iter=300, fixed={0}).items()
            }
        return pos

    @staticmethod
    def get_positions(
        graph: PyGraph | None = None,
        num_qubits: int | None = None,
    ) -> dict[int, tuple[float, float]]:
        """Get predefined positions for a specific station or generate positions for a custom graph.

        Args:
            graph: The graph to generate positions for if no predefined positions exist.
                Used only when num_qubits doesn't match any predefined layout.
            num_qubits: The number of qubits to get a layout for.
                If matches a known system, predefined positions will be used.

        Returns:
            A dictionary mapping node indices to (x,y) coordinates.

        Raises:
            ValueError: If none of graph, or num_qubits are provided, or if num_qubits doesn't
                match any predefined layout and graph is None.

        """
        qubit_station_dict = {
            5: "r5",
            6: "n6",
            20: "r20",
            24: "n24",
            54: "r54",
            150: "r150",
        }
        if num_qubits is not None and num_qubits in qubit_station_dict:
            station = qubit_station_dict[num_qubits]
            qubit_positions = cast(
                dict[int, tuple[float, float]],
                GraphPositions.predefined_stations[station],
            )

        elif graph is not None:
            qubit_positions = GraphPositions.create_positions(graph)
        else:
            raise ValueError("Either a station name, a graph, or a qubit count must be provided to get positions.")
        return qubit_positions


def get_static_connectivity(backend: IQMBackendBase) -> list[list[str]] | None:
    """Returns the connectivity of the static quantum architecture, if available from the backend.

    Args:
        backend: The backend to get the static connectivity of.

    Returns:
        The static connectivity as a list of loci of qubit names, or None if the backend has no client.

    """
    if not isinstance(backend, IQMBackend):
        return None
    try:
        connectivity = backend.client.get_static_quantum_architecture().connectivity
    except Exception as exc:  # noqa: BLE001
        qcvv_logger.warning(f"Could not retrieve the static quantum architecture connectivity: {exc}")
        return None
    return [list(locus) for locus in connectivity]


def chip_figure(qubit_positions_static: dict[int, tuple[float, float]]) -> tuple[Figure, Axes]:
    """Creates a figure for a chip graph, scaled so that larger chips keep the qubit spacing of the 54-qubit layout.

    Args:
        qubit_positions_static: The positions of all qubits of the static architecture.

    Returns:
        The figure and its axes.

    """
    xs, ys = zip(*qubit_positions_static.values(), strict=True)
    scale = max(1.0, max(max(xs) - min(xs), max(ys) - min(ys)) / 10)
    width, height = plt.rcParams["figure.figsize"]
    fig = plt.figure(figsize=(width * scale, height * scale))
    return fig, plt.axes()


def draw_broken_couplers(
    ax: Axes,
    static_connectivity: Sequence[Sequence[str]],
    backend_coupling_map: CouplingMap,
    qubit_names: dict[int, str],
    qubit_positions_static: dict[int, tuple[float, float]],
) -> None:
    """Draws the static couplers missing from the backend in black and the static qubits missing from it in gray.

    Args:
        ax: The axes to draw on.
        static_connectivity: The connectivity of the static quantum architecture, as loci of qubit names.
        backend_coupling_map: The coupling map of the backend.
        qubit_names: The names of the backend qubits, keyed by backend index.
        qubit_positions_static: The positions of all qubits of the static architecture.

    """
    # Qubits missing from the backend are keyed by their name
    name_to_idx: dict[str, int | str] = {v: k for k, v in qubit_names.items()}
    loci = [locus for locus in static_connectivity if all(q[:2] == "QB" for q in locus)]
    static_graph: nx.Graph = nx.Graph([(name_to_idx.get(a, a), name_to_idx.get(b, b)) for a, b in loci])
    missing = [q for q in static_graph if isinstance(q, str)]
    static_pos = {
        q: qubit_positions_static[int((qubit_names[q] if isinstance(q, int) else q)[2:]) - 1] for q in static_graph
    }
    working = {frozenset(edge) for edge in backend_coupling_map}
    broken = [edge for edge in static_graph.edges if frozenset(edge) not in working]
    nx.draw_networkx_edges(static_graph, pos=static_pos, edgelist=broken, width=1.5, edge_color="k", ax=ax)
    nx.draw_networkx_nodes(static_graph, static_pos, nodelist=missing, node_color="darkgray", node_size=200, ax=ax)
    labels_missing = {q: q for q in missing}
    nx.draw_networkx_labels(static_graph, static_pos, labels_missing, font_size=4.5, font_color="w", ax=ax)  # type: ignore[arg-type]


def draw_graph_edges(  # noqa: PLR0913
    backend_coupling_map: CouplingMap,
    num_qubits_static: int,
    edge_list: Sequence[tuple[int, int]],
    timestamp: str,
    disjoint_layers: Sequence[Sequence[tuple[int, int]]] | None = None,
    station: str | None = None,
    qubit_names: dict[int, str] | None = None,
    is_eplg: bool | None = False,
    static_connectivity: Sequence[Sequence[str]] | None = None,
) -> tuple[str, Figure]:
    """Draw given edges on a graph within the given backend.

    Args:
        backend_coupling_map: The coupling map to draw the graph from.
        num_qubits_static: The number of qubits of the respectve backend.
        edge_list: The edge list of the linear chain.
        timestamp: The timestamp to include in the figure name.
        disjoint_layers:
            Sequences of edges defining disjoint layers to draw.
        station: The name of the station.
        qubit_names: A dictionary mapping qubit indices to their names.
        is_eplg: A flag indicating if the graph refers to an EPLG experiment.
        static_connectivity: The connectivity of the static quantum architecture, drawn in the background if
            given together with qubit_names; qubits not in the backend are shown in gray.

    Returns:
         Tuple[str, Figure]: The figure name and the figure object.

    """
    disjoint = "_disjoint" if disjoint_layers is not None else ""
    fig_name_station = f"_{station.lower()}" if station is not None else ""
    fig_name = f"edges_graph{disjoint}{fig_name_station}_{timestamp}"

    fig = plt.figure()
    ax = plt.axes()

    if qubit_names:
        qubit_positions_static = GraphPositions.get_positions(graph=None, num_qubits=num_qubits_static)
        static_indices = {k: int(v[2:]) - 1 for k, v in qubit_names.items()}
        qubit_positions = {k: qubit_positions_static[v] for k, v in static_indices.items()}
    else:
        qubit_positions = GraphPositions.get_positions(graph=None, num_qubits=num_qubits_static)

    if static_connectivity is not None and qubit_names:
        # Draw the static architecture in the background; qubits missing from the backend are keyed by their name
        name_to_idx: dict[str, int | str] = {v: k for k, v in qubit_names.items()}
        loci = [locus for locus in static_connectivity if all(q[:2] == "QB" for q in locus)]
        static_graph: nx.Graph = nx.Graph([(name_to_idx.get(a, a), name_to_idx.get(b, b)) for a, b in loci])
        missing = [q for q in static_graph if isinstance(q, str)]
        static_pos: dict = {**qubit_positions, **{q: qubit_positions_static[int(q[2:]) - 1] for q in missing}}
        nx.draw_networkx_edges(static_graph, pos=static_pos, width=1.5, edge_color="lightgray", ax=ax)
        nx.draw_networkx_nodes(static_graph, static_pos, nodelist=missing, node_color="darkgray", node_size=200, ax=ax)
        labels_missing = {q: q for q in missing}
        nx.draw_networkx_labels(static_graph, static_pos, labels_missing, font_size=4.5, font_color="w", ax=ax)  # type: ignore[arg-type]

    label_station = station if station is not None else f"{num_qubits_static}-qubit IQM Backend"
    if disjoint_layers is None:
        nx.draw_networkx(
            rx_to_nx_graph(backend_coupling_map),
            pos=qubit_positions,
            edgelist=edge_list,
            width=4.0,
            edge_color="k",
            node_color="k",
            node_size=200,
            font_color="w",
            ax=ax,
        )

        plt.title(f"Selected edges in {label_station}\n\n{timestamp}")

    else:
        num_disjoint_layers = len(disjoint_layers)
        colors = plt.colormaps["rainbow"](np.linspace(0, 1, num_disjoint_layers))
        all_edge_colors = [[colors[i]] * len(layer) for i, layer in enumerate(disjoint_layers)]  # Flatten below
        backend_nx_graph = rx_to_nx_graph(backend_coupling_map)
        nx.draw_networkx(
            backend_nx_graph,
            pos=qubit_positions,
            labels=(dict(qubit_names) if qubit_names else {x: x for x in backend_nx_graph.nodes()}),
            font_size=4.5 if qubit_names else 10,
            edgelist=[x for y in disjoint_layers for x in y],
            width=4.0,
            edge_color=[x for y in all_edge_colors for x in y],
            node_color="k",
            node_size=200,
            font_color="w",
            ax=ax,
        )

        is_eplg_string = " for EPLG experiment" if is_eplg else ""
        plt.title(
            f"Selected edges in {label_station.capitalize()}{is_eplg_string}\n"
            f"{len(disjoint_layers)} groups of disjoint layers"
            f"\n{timestamp}"
        )
    ax.set_aspect(0.925)
    plt.gca().invert_yaxis()
    plt.close()

    return fig_name, fig


def calculate_node_radii(
    metric_dict: dict[str, dict[int | tuple[int, int], float]], qubit_nodes: list[int], sq_metric: str
) -> np.ndarray:
    """Calculate node radii based on the specified single qubit metric.

    For the coherence metric, the fidelity is calculated as the idling fidelity of a single qubit gate duration.

    Args:
        metric_dict: Dictionary containing various qubit metrics.
        qubit_nodes: List of qubits to calculate the radius for.
        sq_metric: Metric to use for radius calculation.
        Options: "fidelity", "coherence", or "readout".

    Returns:
        Array of radii values for each qubit node.

    Raises:
        ValueError: If an unsupported metric type is provided.

    """
    if sq_metric == "fidelity":
        radii = -np.log(np.array([metric_dict["fidelity_1qb_gates_averaged"][node] for node in qubit_nodes]))
        if "fidelity_1qb_gates_averaged" not in metric_dict:
            raise ValueError("The metric 'fidelity_1qb_gates_averaged' is not available in the backend metrics.")
    elif sq_metric == "coherence":
        if "t1_time" not in metric_dict or "t2_time" not in metric_dict:
            raise ValueError(
                "At least one of the metrics 't1_time' and 't2_time' is not available in the backend metrics."
            )
        sqg_time = 32e-9
        t1_times = [metric_dict["t1_time"][node] for node in qubit_nodes]
        try:
            t2_times = [metric_dict["t2_time"][node] for node in qubit_nodes]
        except KeyError:
            try:
                t2_times = [metric_dict["t2_echo_time"][node] for node in qubit_nodes]
                qcvv_logger.warning(
                    "Not all T2 times are present in the calibration data, "
                    "using T2 echo times instead for idling error calculations."
                )
            except KeyError:
                t2_times = [np.inf for _ in qubit_nodes]
                qcvv_logger.warning(
                    "Neither T2 nor T2-echo times were fully available in the calibration data, "
                    "disregarding T2 errors for idling error calculations."
                )

        idle_fidelities = (3 + np.exp(-sqg_time / np.array(t1_times)) + 2 * np.exp(-sqg_time / np.array(t2_times))) / 6
        radii = -np.log(idle_fidelities)
    elif sq_metric == "readout":
        if "single_shot_readout_fidelity" not in metric_dict:
            raise ValueError("The metric 'single_shot_readout_fidelity' is both available in the backend metrics.")
        readout_fidelities = [metric_dict["single_shot_readout_fidelity"][node] for node in qubit_nodes]
        radii = -np.log(readout_fidelities)
    else:
        raise ValueError(
            f"Unsupported single qubit metric: {sq_metric}, supported metrics are: fidelity, coherence, readout"
        )
    return radii


def _plot_edges(
    ax: plt.Axes,
    graph: PyGraph,
    qubit_positions: dict[int, tuple[float, float]],
    ghz_edges: set[frozenset[int]] | None = None,
) -> None:
    """Plot edges on the graph with custom styles and colors.

    Args:
        ax: Matplotlib axes to plot on.
        graph: PyGraph containing the quantum chip topology.
        qubit_positions: Dictionary mapping qubit indices to (x, y) positions.
        ghz_edges: Edges of the GHZ tree, as node pairs of ``graph``, to highlight in red. Computed by
            the caller so that it is the tree the benchmark would actually run.

    """
    # draw edges manually with custom styles and colors
    graph_edge_pairs = [list(edge) for edge in graph.edge_list()]
    for _, (edge, weight) in enumerate(zip(graph_edge_pairs, graph.edges(), strict=True)):
        x1, y1 = qubit_positions[edge[0]]
        x2, y2 = qubit_positions[edge[1]]

        # define edge properties (customize these as needed)
        edge_width = weight / np.max(list(graph.edges())) * 10 + 0.1
        edge_color = "black"
        edge_style = "solid"

        # highlight edges with zero weight (wrong calibration)
        if weight == 0:
            edge_width = 1
            edge_style = "dashed"
        # highlight edges that are part of the ghz path
        if ghz_edges and frozenset(edge) in ghz_edges:
            edge_color = "red"

        ax.plot([x1, x2], [y1, y2], color=edge_color, linewidth=edge_width, linestyle=edge_style, zorder=1)


def _ghz_tree_edges(
    backend: IQMBackendBase,
    qubit_layout: list[int],
    metric_dict: dict[str, dict[int | tuple[int, int], float]],
    idx_to_qubit: dict[int, int],
    tree_options: TreeSearchOptions | None,
) -> set[frozenset[int]]:
    """The edges of the GHZ tree a benchmark run would build, as node pairs of the topology plot.

    The tree is built by the same search the "tree" state generation routine uses, from the same
    calibration, so the highlighted path is the one that would actually be played rather than an
    approximation of it.

    Args:
        backend: The backend being plotted.
        qubit_layout: The layout to build the tree over, indexed from 0.
        metric_dict: Metrics from :func:`~iqm.benchmarks.utils.extract_fidelities_unified`.
        idx_to_qubit: Mapping from metric/physical qubit index to the node index used by the plot.
        tree_options: What the search minimizes; ``None`` uses the same defaults as
            :class:`~iqm.benchmarks.entanglement.ghz.GHZConfiguration`.

    Returns:
        The tree edges as frozensets of plot node indices; empty if no tree could be built.

    """
    _circuit, _order, candidate = generate_ghz_tree_optimal(
        qubit_layout, backend, metric_dict, len(qubit_layout), tree_options
    )
    if candidate is None:
        return set()
    qcvv_logger.info(
        f"GHZ path highlighted on the topology plot: {candidate.pairs} "
        f"({int(candidate.stats['layers'])} CZ layers). Non-overlapping CZs are parallelized."
    )
    # The plot's nodes are indexed by idx_to_qubit, while the tree is in physical qubit indices, which
    # are the metric key space. Translating here keeps _plot_edges free of index-space juggling.
    return {
        frozenset((idx_to_qubit[control], idx_to_qubit[target]))
        for control, target in candidate.pairs
        if control in idx_to_qubit and target in idx_to_qubit
    }


def _extract_edges_and_fidelities(
    metric_dict: dict[str, dict[int | tuple[int, int], float]],
) -> tuple[list[list[int]], list[float]]:
    """Normalize edges to have only one ordering per pair.

    Args:
        metric_dict: A dictionary containing qubit and gate metrics.

    Returns:
        A tuple containing a list of unique edges and a corresponding list of fidelities.

    """
    edges_cal: list[list[int]] = []
    fidelities_cal = []
    seen_edges = set()
    for edge, fidelity in metric_dict["cz_gate_fidelity"].items():
        edge = cast(tuple[int, int], edge)
        normalized_edge = tuple(sorted(edge))
        if normalized_edge not in seen_edges:
            edges_cal.append(list(edge))
            fidelities_cal.append(fidelity)
            seen_edges.add(normalized_edge)
    return edges_cal, fidelities_cal


# ruff: noqa: PLR0912, PLR0915
def plot_layout_fidelity_graph(
    backend: str | IQMBackendBase,
    qubit_layouts: list[list[int]] | None = None,
    sq_metric: str = "coherence",
    show_ghz_path: bool = False,
    tree_options: TreeSearchOptions | None = None,
) -> Figure:
    """Plot a graph showing the quantum chip layout with fidelity information.

    Creates a visualization of the quantum chip topology where nodes represent qubits
    and edges represent connections between qubits. Edge thickness indicates gate errors
    (thinner edges mean better fidelity) and selected qubits are highlighted in orange.

    Args:
        backend: The backend to visualize, either as a string name or an IQMBackendBase instance.
        qubit_layouts: List of qubit layouts where each layout is a list of qubit indices
        sq_metric: Optional single qubit metric to use for the visualization, can be either "fidelity", "coherence",
                or "readout".
        show_ghz_path: Whether to highlight the edges that are part of the GHZ state creation tree path.
            The path shown is the one the "tree" state generation routine would build for the first
            layout, from the same calibration, so it is what a run would actually play.
        tree_options: What the GHZ tree search minimizes when show_ghz_path is set. None uses the same
            defaults as GHZConfiguration, so the drawn path matches a default run; pass a
            TreeSearchOptions built from the tree_* options of a non-default configuration.

    Returns:
        matplotlib.figure.Figure: The generated figure object containing the graph visualization

    """
    backend, _ = process_backend(backend)
    num_qubits_static = len(backend.client.get_static_quantum_architecture().qubits)
    coupling_map = backend.coupling_map
    qubit_mapping, metric_dict = extract_fidelities_unified(backend)
    edges_cal, fidelities_cal = _extract_edges_and_fidelities(metric_dict)

    if backend.has_resonators():
        idx_to_qubit = {idx: qubit for qubit, idx in qubit_mapping.items()}
        qubit_to_idx = qubit_mapping
        qubit_nodes = list(idx_to_qubit.keys())[1:]
        fig, ax = plt.subplots(figsize=(len(qubit_nodes), 3))
    else:
        # For other topologies, qubits are indexed starting from 0 as per the Qiskit convention.
        # Shift by the lowest key rather than by a hardcoded 1: the keys are QB numbers (1-based) on a
        # real backend but already 0-based on a fake one, where subtracting 1 produced node -1 and
        # rustworkx rejected it outright. On a real backend the lowest key is 1, so this is unchanged.
        qubit_offset = min(qubit_mapping)
        idx_to_qubit = {idx: qubit - qubit_offset for qubit, idx in qubit_mapping.items()}
        qubit_to_idx = {qubit - qubit_offset: idx for qubit, idx in qubit_mapping.items()}
        qubit_nodes = list(idx_to_qubit.keys())
        fig, ax = plt.subplots(figsize=(1.5 * np.sqrt(len(qubit_nodes)), 1.5 * np.sqrt(len(qubit_nodes))))

    # Filter out any edges that are not in the backend's coupling map
    fidelities_cal = [
        fidelity
        for edge, fidelity in zip(edges_cal, fidelities_cal, strict=True)
        if (edge[0], edge[1]) in coupling_map or (edge[1], edge[0]) in coupling_map
    ]
    edges_cal = [edge for edge in edges_cal if (edge[0], edge[1]) in coupling_map or (edge[1], edge[0]) in coupling_map]

    weights = -np.log(np.array(fidelities_cal))

    # Define qubit positions in plot
    qubit_positions = GraphPositions.get_positions(graph=None, num_qubits=num_qubits_static)

    graph: PyGraph = PyGraph()
    nodes = list(set(qubit_positions.keys()))
    graph.add_nodes_from(nodes)
    for edge, weight in zip(edges_cal, weights, strict=True):
        if backend.has_resonators():
            max_qubit_number = (np.max(list(qubit_positions.keys())) + 1) // 2
            graph.add_edge(idx_to_qubit[edge[0]], idx_to_qubit[edge[0]] + max_qubit_number, weight)
        else:
            graph.add_edge(idx_to_qubit[edge[0]], idx_to_qubit[edge[1]], weight)

    # Draw the main graph
    visualization.mpl_draw(
        graph,
        ax=ax,
        with_labels=True,
        node_color="none",  # No node color since we're using circles
        pos=qubit_positions,
        labels=lambda node: str(qubit_to_idx[node]) if node in qubit_to_idx else "",
        font_color="white",
        edge_color="white",
    )

    ghz_edges: set[frozenset[int]] | None = None
    min_layout_size = 2
    if show_ghz_path and qubit_layouts and len(qubit_layouts[0]) >= min_layout_size:
        ghz_edges = _ghz_tree_edges(backend, list(qubit_layouts[0]), metric_dict, idx_to_qubit, tree_options)

    _plot_edges(ax, graph, qubit_positions, ghz_edges)

    # Draw nodes as circles with varying radii given by the single qubit metric
    radii = calculate_node_radii(metric_dict, qubit_nodes, sq_metric)
    node_colors = ["darkgray" for _ in range(len(nodes))]
    if qubit_layouts is not None:
        for qb in {qb for layout in qubit_layouts for qb in layout}:
            node_colors[idx_to_qubit[qb]] = "orange"
    max_radius = 0.12 + np.max(radii) / np.max(radii) / 2.5

    for idx, node in enumerate(qubit_nodes):
        position = qubit_positions[idx_to_qubit[node]]
        radius = 0.12 + radii[idx] / np.max(radii) / 2.5
        circle = Circle(
            position,
            radius=radius,
            color=node_colors[idx_to_qubit[node]],
            fill=True,
            alpha=1,
        )
        ax.add_patch(circle)

    graph_edge_pairs = [list(edge) for edge in graph.edge_list()]
    # Add edge labels using matplotlib's annotate
    for edge, weight in zip(graph_edge_pairs, list(graph.edges()), strict=True):
        x1, y1 = qubit_positions[edge[0]]
        x2, y2 = qubit_positions[edge[1]]
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        plt.annotate(
            f"{weight:.1e}",
            xy=(x, y),
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.2",
                "fc": "white",
                "ec": "none",
                "alpha": 0.6,
            },
        )

    # Add horizontal bar representing resonator
    if backend.has_resonators():
        resonator_height = 3
        resonator_thickness = 0.8
        x_min = 0.5
        x_max = qubit_positions[idx_to_qubit[qubit_nodes[-1]]][0] + 0.5
        resonator_width = x_max - x_min

        # Create rectangle with rounded corners
        resonator = FancyBboxPatch(
            (x_min, resonator_height - resonator_thickness / 2),
            resonator_width,
            resonator_thickness,
            boxstyle="round,pad=0.01,rounding_size=0.3",
            color="lightsteelblue",
            zorder=10,
        )
        ax.add_patch(resonator)

        # Add "Resonator" label in the center
        plt.annotate(
            "Resonator",
            xy=((x_min + x_max) / 2, resonator_height),
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            va="center",
            color="black",
            fontsize=10,
            zorder=11,
            bbox={
                "boxstyle": "round,pad=0.2",
                "fc": "white",
                "ec": "none",
                "alpha": 0.8,
            },
        )

    # Calculate axis limits to ensure all circles are visible
    all_x = [pos[0] for pos in qubit_positions.values()]
    all_y = [pos[1] for pos in qubit_positions.values()]
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)

    # Add padding for circles
    padding = max_radius * 1.5
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)

    # Adjust layout first
    plt.tight_layout(pad=2.0)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    plt.figtext(
        0.5,
        0.99,  # x=0.5 (center), y=0.01 (bottom)
        f"Qubit connectivity with selected qubits in orange\n"
        f"CZ errors -log(F) indicated by edge thickness (thinner is better)\n"
        f"Single qubit errors -log(F) shown as node size with F computed from {sq_metric} metrics",
        fontsize=10,
        ha="center",
        wrap=True,
    )

    plt.show()
    return fig


def rx_to_nx_graph(backend_coupling_map: CouplingMap) -> nx.Graph:
    """Convert the Rustworkx graph returned by a backend to a Networkx graph.

    Args:
        backend_coupling_map: The coupling map of the backend.

    Returns:
        The Networkx Graph corresponding to the backend graph.

    """
    # Generate a Networkx graph
    graph_backend = backend_coupling_map.graph.to_undirected(multigraph=False)
    backend_egdes, backend_nodes = (
        list(graph_backend.edge_list()),
        list(graph_backend.node_indices()),
    )
    backend_nx_graph: nx.Graph = nx.Graph()  # Type annotation added
    backend_nx_graph.add_nodes_from(backend_nodes)
    backend_nx_graph.add_edges_from(backend_egdes)

    return backend_nx_graph
