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

from collections import deque
from collections.abc import Iterator, Sequence
import itertools
import math
from typing import TYPE_CHECKING, Any, TypeAlias, cast
import warnings

from dimod.typing import Variable
from iqm.iqm_client import StaticQuantumArchitecture
from iqm.qaoa.transpiler.rx_to_nx import rustworkx_to_networkx
from iqm.qiskit_iqm.iqm_backend import IQMBackendBase
from iqm.qiskit_iqm.iqm_provider import IQMBackend, IQMFakeBackend
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import networkx as nx

if TYPE_CHECKING:
    from iqm.qaoa.transpiler.routing import Mapping


# Defining type aliases for integers and frozensets thereof
LogQubit: TypeAlias = Variable
"""
A custom type alias for :data:`~dimod.typing.Variable` to refer to logical / problem qubits.
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


class NonCrystalSQAError(ValueError):
    """Custom error class for errors coming from the QPU topology not matching the Crystal architecture."""


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

    @property
    def qubits(self) -> set[HardQubit]:
        r"""The set of all :class:`HardQubit`\s of the QPU."""
        return set(self._hardware_graph.nodes())

    def has_edge(self, gate: HardEdge) -> bool:
        r"""True iff there is an edge between the qubits involved in ``gate``.

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
        gate_lists: dict[str, list[tuple[HardQubit] | tuple[HardQubit, HardQubit]]] | None = None,
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
            **kwargs: Arbitrary keyword arguments for :func:`~networkx.draw_networkx_edges`.

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
    """Class for a QPU with square lattice topology, initialited from a Qiskit backend object.

    Since the topology is square lattice, the qubits can be identified with 2D integer coordinates (up to a global
    shift). The 2D coordinates are calculated from the connectivity topology (provided as the
    :class:`~iqm.iqm_client.StaticQuantumArchitecture` of the input ``backend``).

    If the provided ``backend`` is an instance of :class:`~iqm.qiskit_iqm.iqm_provider.IQMBackend` and it was created
    by ``IQMProvider.get_backend(use_metrics=True)``, then it contains the calibration metrics. The CZ gate errors from
    these are converted to gate fidelities and saved as edge attributes of the hardware graph.

    Args:
        backend: The backend containing information about the QPU.

    Raises:
        ValueError: If the ``backend`` is :class:`~iqm.qiskit_iqm.iqm_provider.IQMBackend` and it contains multiple
            different CZ gate fidelity entries for a pair of coupled qubits.
        TypeError: If an unknown type of backend is provided (neither :class:`~iqm.qiskit_iqm.iqm_provider.IQMBackend`
            nor :class:`~iqm.qiskit_iqm.iqm_provider.IQMFakeBackend`).

    """

    def __init__(self, backend: IQMBackendBase) -> None:
        # ``backend.coupling_map`` is a graph of couplings. Qubits with no connections don't appear in it.
        # We compute qubits missing from ``hw_graph`` and add them as isolated (unconnected) nodes.
        hw_graph = rustworkx_to_networkx(backend.coupling_map.graph)
        set_of_all_qubits = {backend.qubit_name_to_index(qb_name) for qb_name in backend.physical_qubits}
        qubits_missing_from_hw_graph = set_of_all_qubits - set(hw_graph.nodes)
        hw_graph.add_nodes_from(qubits_missing_from_hw_graph)

        # The coupling map may be a directed graph, so we make it un-directed.
        if isinstance(hw_graph, nx.DiGraph):
            hw_graph = hw_graph.to_undirected()

        # If it's an ``IQMBackend`` and it has metrics available.
        if isinstance(backend, IQMBackend) and (backend.metrics is not None):
            # The index in the ``Target.instructions`` list is needed to use ``instruction_properties``.
            for indx, (instr, qbts) in enumerate(backend.target.instructions):
                if instr.name == "cz":
                    gate_error = backend.target.instruction_properties(indx).error
                    # It's possible that the instruction is in the instruction list, but it doesn't have a fidelity.
                    if gate_error is None:
                        warnings.warn(
                            f"No CZ fidelity found for qubit pair {qbts}. Using a value of 0 instead.", stacklevel=2
                        )
                        fidelity = 0.0
                    else:
                        fidelity = 1 - gate_error

                    existing_fidelity = hw_graph[qbts[0]][qbts[1]].get("fidelity")
                    if existing_fidelity is not None and not math.isclose(
                        existing_fidelity, fidelity, abs_tol=1e-5, rel_tol=0
                    ):
                        raise ValueError(
                            f"Conflicting CZ fidelities for qubit pair {(qbts[0], qbts[1])}: "
                            f"{fidelity}, {existing_fidelity}. This is a crucial problem with the backend's metrics."
                        )
                    hw_graph[qbts[0]][qbts[1]]["fidelity"] = fidelity

        if isinstance(backend, IQMBackend):
            sqa = backend.client.get_static_quantum_architecture()
        elif isinstance(backend, IQMFakeBackend):
            sqa = backend._IQMFakeBackend__sqa  # Access the internal ``__sqa`` attribute (name-mangled by Python).
        else:
            raise TypeError(f"Unknown type of backend provided: {type(backend)}.")

        layout_with_qb_names = CrystalQPUFromBackend._get_layout_from_sqa(sqa)
        hw_layout = {backend.qubit_name_to_index(qb_name): coords for qb_name, coords in layout_with_qb_names.items()}
        super().__init__(hw_graph, hw_layout)

    @staticmethod
    def _get_layout_from_sqa(sqa: StaticQuantumArchitecture) -> dict[str, tuple[int, int]]:
        """Compute a square grid layout for the qubits of Crystal QPUs.

        The coordinate origin position and the X and Y axis directions are not
        guaranteed to be deterministic.

        Args:
            sqa: The static quantum architecture (SQA) from which the layout is to be calculated. The SQA describes the
                connections on the QPU.

        Returns:
            The layout as a dictionary whose keys are the names of the qubits and values their 2D coordinates.

        Raises:
            NonCrystalSQAError: If the SQA is for a QPU with computational resonators.
            NonCrystalSQAError: If the SQA does not have any qubits with 2 neighbors (those in a 'corner') of the QPU.
            NonCrystalSQAError: If the calculation leads to ambiguous coordinates for any of the qubits. This is
                typically caused by the SQA not being a square lattice.
            NonCrystalSQAError: If the SQA has a qubit with 2 neighbors which is not a part of a plaquette.

        """
        if sqa.computational_resonators:
            raise NonCrystalSQAError("Only works for Crystal QPUs.")

        # Get the neighbor dict for the qubits.
        neighbors: dict[str, set[str]] = {}
        for pair in sqa.connectivity:
            neighbors.setdefault(pair[0], set()).add(pair[1])
            neighbors.setdefault(pair[1], set()).add(pair[0])

        layout: dict[str, tuple[int, int]] = {}

        def assign_coords(plaquette: Sequence[str], x: int, y: int) -> None:
            """Assign coordinates for the plaquette qubits.

            Assumes ``plaquette[0]`` is the lower left corner and the plaquette continues in the positive direction.

            Args:
                plaquette: A plaquette of qubits to which we want to assign coordinates.
                x: The x-coordinate of the qubit ``plaquette[0]``.
                y: The y-coordinate of the qubit ``plaquette[0]``.

            """

            def set_coords(q: str, coords: tuple[int, int]) -> None:
                """Set the coordinates for ``q``; raise ``RuntimeError`` if ``q`` already has different coordinates."""
                if (old_coords := layout.get(q)) is not None:
                    if coords != old_coords:
                        raise NonCrystalSQAError(
                            f"Ambiguous coordinates for qubit {q}. Likely caused by "
                            "the static quantum architecture (SQA) not being a square grid."
                        )
                    return
                layout[q] = coords

            set_coords(plaquette[0], (x, y))
            set_coords(plaquette[1], (x + 1, y))
            set_coords(plaquette[2], (x + 1, y + 1))
            set_coords(plaquette[3], (x, y + 1))

        def create_plaquette(q1: str, q2: str, taken: set[str]) -> deque[str] | None:
            """Try to construct a new 4-qubit plaquette.

            ``(q1, q2)`` forms the first edge and the new plaquette continues in the positive direction.

            Args:
                q1: First qubit.
                q2: Second qubit.
                taken: Qubits that belong to the plaquette on the other side of the ``(q1, q2)`` edge.

            Returns:
                Qubits of the new plaquette in the positive direction, or ``None`` if no such plaquette could be
                constructed, or ``None`` if all qubits in the newly-constructed plaquette already have coordinates in
                ``layout``.

            Raises:
                ValueError if the topology of the QPU (obtained from the SQA) is not Crystal, i.e., not a square grid.

            """
            n1 = neighbors[q1] - taken
            n2 = neighbors[q2] - taken
            # At most two neighbors left for each, since the other plaquette of ``q1`` and ``q2`` is already done.
            max_number_unassigned_neighbors = 2
            if not max(len(n1), len(n2)) <= max_number_unassigned_neighbors:
                raise NonCrystalSQAError(
                    f"The topology of the chip is not Crystal, i.e., square grid. Namely, some qubits ({q1} or {q2}) "
                    "seem to have more than 4 neighbors."
                )
            for a, b in itertools.product(n1, n2):
                if b in neighbors[a]:  # Plaquette found.
                    break  # Breaking away from the ``for`` loop keeps the latest assignment of ``a`` and ``b``.
            else:  # No plaquette found.
                return None
            plaq = [q1, q2, b, a]
            if all(q in layout for q in plaq):
                # All qubits in the plaquette already have layout coordinates.
                # Without this we may loop back to already handled plaquettes in the recursion.
                return None
            return deque(plaq)

        def handle_plaquette(plaquette: Sequence[str], x: int, y: int) -> None:
            """Recursively adds the given plaquette and its neighbor plaquettes to the layout.

            Args:
                plaquette: 4-qubit plaquette where ``plaquette[0]`` is the lower left corner and the qubits continue in
                    the positive direction.
                x: The x-coordinate of the qubit ``plaquette[0]``.
                y: The y-coordinate of the qubit ``plaquette[0]``.

            """
            assign_coords(plaquette, x, y)

            # Create neighboring plaquettes.
            p = deque(plaquette)
            taken = set(plaquette)
            for k, (dx, dy) in enumerate([(0, 1), (1, 0), (0, -1), (-1, 0)]):
                new_plaquette = create_plaquette(p[3], p[2], taken)
                if new_plaquette is not None:
                    new_plaquette.rotate(-k)
                    handle_plaquette(new_plaquette, x + dx, y + dy)
                p.rotate(1)

        # Find the first plaquette.
        # Find qubits with exactly 2 neighbors.
        number_of_neighbors_of_corner_qubit = 2
        corners = [q for q, n in neighbors.items() if len(n) == number_of_neighbors_of_corner_qubit]
        if not corners:
            raise NonCrystalSQAError(
                "Invalid static quantum architecture (SQA): "
                "The connectivity graph has no qubits with degree 2 (corner qubits)."
            )
        # Pick one at random, this is the origin and start of first 4-qubit plaquette.
        # Then pick one of its two neighbors at random, this is the positive X axis direction.
        # The other neighbor defines the positive Y axis direction.
        q1 = corners[0]
        n = list(neighbors[q1])
        nn = neighbors[n[0]] & neighbors[n[1]]
        nn.remove(q1)
        if not nn:
            raise NonCrystalSQAError(
                f"Invalid static quantum architecture (SQA): The connectivity graph has a qubit with degree 2 ({q1}), "
                "which is not part of a 4-qubit plaquette."
            )
        fourth = next(iter(nn))
        plaquette = [q1, n[0], fourth, n[1]]

        # Then find the rest recursively.
        handle_plaquette(plaquette, 0, 0)

        return layout


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
        """Embeds a chain in the grid QPU (by going around like a snake).

        .. code-block:: text

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
        yield from range(self._hardware_graph.number_of_nodes())


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
