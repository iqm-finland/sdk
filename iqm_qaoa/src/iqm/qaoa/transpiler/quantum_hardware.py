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
"""The module for classes representing various QPU architectures.

The module also contains four type aliases, which are imported by other modules for more clear type hinting.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, TypeAlias, cast

from iqm.qaoa.transpiler.rx_to_nx import rustworkx_to_networkx
from iqm.qiskit_iqm.iqm_backend import IQMBackendBase
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import networkx as nx

if TYPE_CHECKING:
    from iqm.qaoa.transpiler.routing import Mapping


# Defining type aliases for integers and frozensets thereof
LogQubit: TypeAlias = int
"""
A custom type alias for :class:`int` to refer to logical / problem qubits.
"""
HardQubit: TypeAlias = int
"""
A custom type alias for :class:`int` to refer to hardware qubits.
"""
LogEdge: TypeAlias = frozenset[LogQubit]
"""
A custom type alias for :class:`frozenset` of :class:`LogQubit` to refer to interactions between logical qubits.
"""
HardEdge: TypeAlias = frozenset[HardQubit]
"""
A custom type alias of :class:`frozenset` of :class:`HardQubit` to refer to interactions between hardware qubits.
"""


# pylint: disable=anomalous-backslash-in-string
class QPU:
    r"""A parent class for all QPU architectures.

    The main purpose of the QPU class is to store the :attr:`hardware_graph` and the :attr:`shortest_path`/s in there.
    The method :meth:`draw` can be used independently to plot the graph (using the :attr:`hardware_layout`), but it's
    meant to be used by the :meth:`~iqm.qaoa.transpiler.routing.Layer.draw` method of the class
    :class:`~iqm.qaoa.transpiler.routing.Layer`.

    Args:
        hardware_graph: A :class:`~networkx.Graph` representing the topology of the QPU, i.e., the connections between
            the :class:`HardQubit`\s.
        hardware_layout: A layout of the QPU, i.e., the coordinates of the qubits in the 2D plane.

    """

    def __init__(
        self, hardware_graph: nx.Graph, hardware_layout: dict[HardQubit, tuple[Any, Any]] | None = None
    ) -> None:
        self._hardware_graph = hardware_graph
        if hardware_layout is None:
            # Assuming that the QPU topology is planar, this is likely going to get the nicest result.
            self._hardware_layout = nx.planar_layout(self._hardware_graph)
        else:
            self._hardware_layout = hardware_layout
        self._shortest_path = dict(nx.shortest_path(self._hardware_graph))

    # pylint: disable=anomalous-backslash-in-string
    @property
    def qubits(self) -> set[HardQubit]:
        r"""The set of all :class:`HardQubit`\s of the QPU."""
        return set(self._hardware_graph.nodes())

    # pylint: disable=anomalous-backslash-in-string
    def has_edge(self, gate: HardEdge) -> bool:
        r"""Is there an edge between the qubits involved in ``gate``?

        Args:
            gate: A :class:`HardEdge` between two :class:`HardQubit`\s.

        Returns:
            True if there is an edge between the two :class:`HardQubit`\s on the QPU graph and False otherwise.

        """
        return self._hardware_graph.has_edge(*gate)

    @property
    def hardware_graph(self) -> nx.Graph:
        """The connectivity graph of the QPU."""
        return self._hardware_graph

    @property
    def hardware_layout(self) -> dict[HardQubit, tuple[float, float]]:
        """The layout of the hardware qubits (in the 2D plane)."""
        return self._hardware_layout

    @property
    def shortest_path(self) -> dict[HardQubit, dict[HardQubit, list[HardQubit]]]:
        """The dictionary of dictionaries of shortest paths.

        It's defined so that ``shortest_path[source][target]`` is the list of nodes lying on the/a shortest path
        between the ``source`` and ``target`` nodes.
        """
        return self._shortest_path

    def draw(
        self,
        mapping: Mapping | None = None,
        ax: Axes | None = None,
        gate_lists: dict[str, list[tuple[HardQubit]]] | None = None,
        show: bool = True,
        **kwargs: Any,
    ) -> None:
        """A method for drawing the QPU.

        It displays the picture of the QPU in a pop-up window, with edges colored based on ``gate_lists``.

        Args:
            mapping: The mapping between the logical and hardware qubits, for labels of the graph nodes.
            ax: An instance of :class:`matplotlib.axes.Axes` object, to define the plotting area.
            gate_lists: A dictionary whose keys are colors (as single-letter strings) and values are lists of edges
                which should be colored that color.
            show: Boolean which decides if the graph will be shown in a pop-up window.
            **kwargs: Arbitrary keyword arguments.

        """
        nx.draw_networkx_edges(self._hardware_graph, ax=ax, pos=self._hardware_layout, **kwargs)
        if gate_lists is not None:
            for color, gates in gate_lists.items():
                edge_list = list(gates)
                nx.draw_networkx_edges(
                    self._hardware_graph,
                    ax=ax,
                    pos=self._hardware_layout,
                    edgelist=edge_list,
                    width=6.0,
                    edge_color=color,
                    alpha=0.5,
                )
        if mapping is not None:
            labels = {hard_qb: mapping.hard2log[hard_qb] for hard_qb in mapping.hard2log}
            nx.draw_networkx_labels(self._hardware_graph, ax=ax, pos=self._hardware_layout, labels=labels)
            nx.draw_networkx_nodes(self._hardware_graph, ax=ax, pos=self._hardware_layout)
        else:
            nx.draw_networkx_nodes(self._hardware_graph, ax=ax, pos=self._hardware_layout)
        if show:
            plt.show()


class CrystalQPUFromBackend(QPU):
    """Class for a QPU with square lattice topology, initialited from an
    :class:`~iqm.qiskit_iqm.iqm_provider.IQMBackend` object.

    Since the topology is square lattice, the qubits can be identified with 2D integer coordinates (up to a global
    shift). However, it appears difficult to calculate these coordinates just from the topology graph, so instead
    a helper function is used with hard-coded sets of coordinates for IQM's public QPU designs.

    Args:
        backend: The backend containing information about the QPU.

    """

    def __init__(self, backend: IQMBackendBase) -> None:
        hw_graph = rustworkx_to_networkx(backend.coupling_map.graph)

        # The coupling map may be a directed graph, so we make it un-directed.
        if isinstance(hw_graph, nx.DiGraph):
            hw_graph = hw_graph.to_undirected()
        # For Crystal QPUs we get the layout here.
        hardware_layout = _layout_of_crystal(hw_graph.number_of_nodes())
        super().__init__(hw_graph, hardware_layout)


class Grid2DQPU(QPU):
    """Class for 2D rectangular QPU.

    Contains variables for number of rows and columns, which determine the hardware graph and layout. Also contains
    a simple :meth:`embedded_chain` method to embed a chain in the hardware graph.

    Args:
        num_rows: The number of rows in the grid.
        num_columns: The number of columns in the grid.

    """

    def __init__(self, num_rows: int, num_columns: int) -> None:
        self._num_rows = num_rows
        self._num_columns = num_columns
        self._hardware_graph_2d = nx.grid_2d_graph(num_rows, num_columns)
        # For compatibility with other functions, we need the nodes in the graph to be integers (i.e., HardQubit)
        self._nodes_2d_sorted = sorted(self._hardware_graph_2d.nodes())
        hardware_layout = {node: self._nodes_2d_sorted[node] for node in range(self._num_rows * self._num_columns)}
        hardware_graph = nx.convert_node_labels_to_integers(self._hardware_graph_2d, ordering="sorted")
        super().__init__(hardware_graph, hardware_layout)

    def embedded_chain(self) -> Iterator[HardQubit]:
        """Embeds a chain in the grid QPU (by going around like a snake)::

            -----------------╷
            ╷----------------╵
            ╵----------------╷
            ╷----------------╵
            ╵----------------╷
            ╷----------------╵
            ╵-----------------

        Yields:
            Integer index of the next qubit in the chain.

        """
        for row_ind in range(self._num_rows):
            if row_ind % 2 == 0:
                for column_ind in range(self._num_columns):
                    yield self._nodes_2d_sorted.index((row_ind, column_ind))
            else:
                for column_ind in range(self._num_columns - 1, -1, -1):
                    yield self._nodes_2d_sorted.index((row_ind, column_ind))


class LineQPU(QPU):
    """A linear QPU (qubits on a line).

    Nothing fancy here, just a special case of a qubit hardware connectivity graph, which is a line. Given a ``length``,
    creates a path ``hardware_graph`` and the corresponding ``hardware_layout`` which are then passed to :class:`QPU`
    class initialization.

    Args:
        length: The length of the QPU (as number of qubits).

    """

    def __init__(self, length: int) -> None:
        hardware_graph = nx.path_graph(length)
        hardware_layout = {cast(HardQubit, node): (0.0, cast(float, node)) for node in hardware_graph.nodes()}
        super().__init__(hardware_graph, hardware_layout)

    def embedded_chain(self) -> Iterator[HardQubit]:
        """Embeds a chain in the line QPU (which is just a line).

        Yields:
            Integer index of the next qubit in the chain.

        """
        yield from range(self._hardware_graph.number_of_qubits())  # type: ignore[attr-defined]


def _layout_of_crystal(n: int) -> dict[HardQubit, tuple[int, int]]:
    """A helper function that reads the imported global variable ``LAYOUTS`` and outputs the correct QPU layout.

    The main task of this function is just changing the data type, from a list of lists of strings (the type of
    ``LAYOUTS``) to a dictionary whose keys are :class:`HardQubit` and whose values are 2D integer coordinates.

    Args:
        n: The size of the QPU in qubits.

    Returns:
        The layour of the QPU with integer 2D coodrinates.

    """
    layout_of_qpu: dict[HardQubit, tuple[int, int]] = {}
    num_qb_to_name: dict[int, str] = {20: "crystal_20", 54: "crystal_54", 150: "crystal_150"}

    qpu_graphs: dict[str, nx.Graph] = {}
    # Garnet graph topology with coordinates.
    qpu_graphs["crystal_20"] = nx.grid_2d_graph(5, 5)
    qpu_graphs["crystal_20"].remove_nodes_from({(0, 0), (0, 3), (0, 4), (4, 0), (4, 4)})
    # Emerald graph topology with coordinates.
    qpu_graphs["crystal_54"] = nx.grid_2d_graph(9, 9)
    qpu_graphs["crystal_54"].remove_nodes_from(
        {
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 6),
            (0, 7),
            (0, 8),
            (1, 0),
            (1, 1),
            (1, 7),
            (1, 8),
            (2, 0),
            (2, 8),
            (3, 8),
            (5, 0),
            (6, 0),
            (6, 8),
            (7, 0),
            (7, 1),
            (7, 7),
            (7, 8),
            (8, 0),
            (8, 1),
            (8, 2),
            (8, 6),
            (8, 7),
            (8, 8),
        }
    )

    for i, coords in enumerate(sorted(qpu_graphs[num_qb_to_name[n]].nodes)):
        layout_of_qpu[i] = coords

    return layout_of_qpu


class StarQPU(QPU):
    """A star-shaped QPU (Daneb, Sirius, ...).

    Importantly, the central resonator always has label 0 in the QPU graph. This is used in circuits built on the star.

    Args:
        n: The number of the spokes of the star graph, so that the graph as a whole has ``n+1`` vertices, including
            the central vertex.

    """

    def __init__(self, n: int) -> None:
        graph = nx.star_graph(n)
        layout = nx.shell_layout(graph, nlist=[[0], range(1, n + 1)])
        super().__init__(graph, layout)
