# Copyright 2026 IQM
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

"""VF2++ subgraph isomorphism-based layout pass."""

from __future__ import annotations

from collections.abc import Callable
import itertools

from iqm.qrisp_iqm.passes.routing.core.graph_processing_tools import connectivity_to_topology
from iqm.qrisp_iqm.passes.routing.core.qubit_padding import pad_qubits
from iqm.qrisp_iqm.passes.routing.core.vf2pp import (
    qc_to_interaction_graph,
    vf2pp_subgraph_isomorphism,
)
from qrisp import QuantumCircuit
from qrisp.circuit.pass_management.circuit_pass import CircuitPass


def vf2pp_layout(connectivity: list[tuple[int, int]]) -> Callable[[QuantumCircuit], QuantumCircuit]:
    """Create a pass that finds a qubit layout using VF2++ subgraph isomorphism.

    This pass uses the VF2++ algorithm to find a subgraph isomorphism between
    the circuit's interaction graph and the hardware topology. If a valid
    mapping exists, the circuit can be executed without any SWAP gates.

    If the circuit's *current* qubit order is already a valid placement on the
    topology, the pass is a no-op and leaves every qubit where it is — a
    layout produced by an earlier pass (e.g. ``manual_layout``) therefore
    survives this one unchanged.

    Parameters
    ----------
    connectivity : list[tuple[int]]
        The list of edges representing the hardware topology.

    Returns
    -------
    Callable[[QuantumCircuit], QuantumCircuit]
        A pass function that transforms the circuit.

    Raises
    ------
    ValueError
        If no valid subgraph isomorphism can be found (the circuit's
        interaction graph is not embeddable in the topology).

    Example
    -------
    >>> from qrisp import QuantumCircuit, PassManager
    >>> from iqm.qrisp_iqm import vf2pp_layout
    >>> qc = QuantumCircuit(2); qc.cx(0, 1); qc.measure(qc.qubits)
    >>> pm = PassManager()
    >>> pm += vf2pp_layout(connectivity=[(0,1), (1,2)])
    >>> transpiled_qc = pm.run(qc)

    """
    if not connectivity:
        raise ValueError(
            "vf2pp_layout requires a non-empty connectivity list. "
            "The connectivity must contain at least one edge "
            "describing available two-qubit gate loci on the device."
        )

    _EDGE_SIZE = 2
    for edge in connectivity:
        if len(edge) != _EDGE_SIZE:
            raise ValueError("Each connectivity entry must be a 2-tuple")

    topology_qubits = set(itertools.chain.from_iterable(connectivity))

    if any(qb < 0 for qb in topology_qubits):
        raise ValueError("vf2pp_layout requires non-negative integer qubit labels in connectivity")

    @CircuitPass
    def _vf2pp_layout(qc: QuantumCircuit) -> QuantumCircuit:
        # Convert circuit and topology to sparse representations
        qc_indptr, qc_indices = qc_to_interaction_graph(qc)
        topology = connectivity_to_topology(connectivity)

        # Find subgraph isomorphism
        mapping = vf2pp_subgraph_isomorphism(qc_indices, qc_indptr, topology.indices, topology.indptr)

        if mapping[0] == -1:
            raise ValueError(
                "VF2++ could not find a matching qubit set for the circuit. "
                "The circuit's connectivity graph is not a subgraph of the topology. "
                "Consider using 'plasma_layout' and 'plasma_route' for circuits "
                "requiring swap insertion."
            )

        # Apply the mapping to create a new circuit by copying the original
        # (preserving qubit identities) and reordering qubits.
        qubit_amount = len(topology.indptr) - 1  # QPUTopology adjacency matrix dimension
        new_qc = qc.copy()

        pad_qubits(new_qc, qubit_amount)

        # mapping[logical_idx] == physical_idx, one entry per original circuit qubit
        # (len(mapping) == qc.num_qubits()); padding qubits are not in it.
        # connectivity_to_topology labels topology node i as physical qubit i, so the
        # indices are those of the connectivity argument and need no translation.
        # Build the inverse mapping: physical_idx -> qubit index in new_qc.
        inv_map: list[int] = [-1] * qubit_amount
        for logical_idx in range(qc.num_qubits()):
            physical_pos = int(mapping[logical_idx])
            inv_map[physical_pos] = logical_idx

        # Assign padding qubits to unmapped physical positions
        padding_idx = qc.num_qubits()
        for pos in range(qubit_amount):
            if inv_map[pos] == -1:
                inv_map[pos] = padding_idx
                padding_idx += 1

        # sort the logical qubits in the physical qubit index order
        new_qc.qubits = [new_qc.qubits[logical_idx] for logical_idx in inv_map]

        return new_qc

    _vf2pp_layout.__name__ = "vf2pp_layout"
    return _vf2pp_layout
