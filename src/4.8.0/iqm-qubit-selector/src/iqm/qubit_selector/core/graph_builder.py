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

"""Build coupling graphs from :class:`~iqm.station_control.interface.models.dynamic_quantum_architecture.DynamicQuantumArchitecture`."""  # noqa: E501

from iqm.iqm_client import simplify_architecture
import networkx as nx

from iqm.station_control.interface.models import DynamicQuantumArchitecture

_COUPLING_LOCUS_SIZE = 2
"""Number of qubits in a 2-qubit gate locus, i.e. an edge of the coupling graph."""


def build_coupling_graph(arch: DynamicQuantumArchitecture) -> nx.Graph:
    """Return the qubit coupling graph of ``arch``.

    Nodes are qubits and edges are 2-qubit gate loci; computational resonators are left out.

    A Star QPU has only qubit-resonator loci, which would give a graph without edges, so it is
    first collapsed by :func:`~iqm.iqm_client.transpile.simplify_architecture` into the qubit-qubit
    CZs that ``MOVE`` + ``CZ`` can realise. Crystal QPUs pass through unchanged.

    Args:
        arch: The dynamic quantum architecture describing available gates.

    Returns:
        The coupling graph. Its nodes are in architecture order, which the enumerators use as the
        qubit order.

    """
    arch = simplify_architecture(arch)

    qubits = list(arch.qubits)
    qubit_set = set(qubits)

    graph: nx.Graph = nx.Graph()
    # Added before any edge so that the node order is the architecture order
    graph.add_nodes_from(qubits)

    for gate_info in arch.gates.values():
        for impl_info in gate_info.implementations.values():
            for locus in impl_info.loci:
                if len(locus) != _COUPLING_LOCUS_SIZE:
                    continue
                # Only consider edges between qubits (skip resonator-qubit edges). ``nx.Graph`` is
                # undirected and simple, so repeated or reversed loci collapse to one edge on their own.
                q0, q1 = locus
                if q0 in qubit_set and q1 in qubit_set:
                    graph.add_edge(q0, q1)

    return graph


def remove_qubits_from_graph(graph: nx.Graph, remove: list[str]) -> nx.Graph:
    """Return a copy of ``graph`` without the qubits in ``remove``, keeping the node order.

    Args:
        graph: The coupling graph.
        remove: Qubits to remove; names not in the graph are ignored.

    Returns:
        The coupling graph without the removed qubits.

    """
    remove_set = set(remove)
    return graph.subgraph([node for node in graph if node not in remove_set]).copy()
