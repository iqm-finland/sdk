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

"""Default plasma-sabre PassManager for IQM-targeted transpilation.

This module provides :func:`create_iqm_pass_manager`, a factory that builds a
fully configured :class:`~qrisp.PassManager` implementing the complete
plasma-sabre transpilation pipeline (predicates, decomposition, layout,
routing, SWAP optimization, and gate conversion to the CZ + PRX native gate
set).  The high-level convenience function :func:`transpile_to_iqm` calls the
factory and immediately runs the resulting PassManager on a circuit.
"""

from __future__ import annotations

from iqm.qrisp_iqm.passes.commute_phases import commute_phases
from iqm.qrisp_iqm.passes.routing import plasma_layout, plasma_route
from qrisp import (
    CircuitPass,
    PassManager,
    QuantumCircuit,
    Qubit,
)
from qrisp.circuit.operation import Operation
from qrisp.circuit.pass_management.passes import (
    arrange_swaps,
    cancel_zero_controls,
    combine_single_qubit_gates,
    commute_swaps,
    convert_to_cz,
    convert_to_prx,
    decompose,
    fuse_adjacents,
    gray_synth_toffoli,
    is_toffoli,
    resolve_swaps,
    reverse_parallelize,
)

# ---------------------------------------------------------------------------
# Predicate helpers for selective decomposition
# ---------------------------------------------------------------------------


def _leave_toffoli_predicate(op: Operation) -> bool:
    """Decompose everything except swaps, Toffoli gates, and permable 2-qubit gates."""
    if op.name == "swap":
        return False
    if is_toffoli(op):
        return False
    if op.num_qubits > 2:  # noqa: PLR2004
        return True
    if sum(val is None for val in op.permeability.values()) < 2:  # noqa: PLR2004
        return False
    return True


def _dissolve_toffoli_predicate(op: Operation) -> bool:
    """Decompose everything except swaps and permable 2-qubit gates (Toffoli gets decomposed)."""
    if op.name == "swap":
        return False
    if op.num_qubits > 2:  # noqa: PLR2004
        return True
    if sum(val is None for val in op.permeability.values()) < 2:  # noqa: PLR2004
        return False
    return True


def _leave_swap_predicate(op: Operation) -> bool:
    """Decompose everything except swaps."""
    return op.name != "swap"


# ---------------------------------------------------------------------------
# Utility pass: re-insert ignored qubits
# ---------------------------------------------------------------------------


def _reinsert_ignored_qubits(ignore_nodes: list[int]) -> CircuitPass:
    """Create a pass that re-inserts ignored qubits into the circuit.

    Ignored qubit indices are sorted in descending order and a fresh
    :class:`~qrisp.Qubit` is inserted at each position.  This pass is a
    no-op when *ignore_nodes* is empty.
    """
    sorted_nodes = sorted(ignore_nodes, reverse=True)

    @CircuitPass
    def _pass(qc: QuantumCircuit) -> QuantumCircuit:
        for i in sorted_nodes:
            qc.qubits.insert(i, Qubit("ignored_qubit." + str(i)))
        return qc

    return _pass


# ---------------------------------------------------------------------------
# PassManager factory
# ---------------------------------------------------------------------------


def create_iqm_pass_manager(
    connectivity: list[tuple[int, int]],
    effort: int = 100,
    depth_weight: float = 0.0,
    ignore_nodes: list[int] | None = None,
    ignore_edges: list[tuple[int, int]] | None = None,
) -> PassManager:
    """Create a :class:`~qrisp.PassManager` pre-configured for IQM transpilation.

    This pass applies the full transpilation pipeline including:
    - Layout and routing (SABRE-based)
    - SWAP optimization
    - Gate conversions (to CZ + PRX)
    - Gate cancellation and optimization

    Parameters
    ----------
    connectivity : list[tuple[int, int]]
        The connectivity describing the device topology.
    effort : int, optional
        Effort level for layout and routing. Default is 100.
    depth_weight : float, optional
        Tradeoff between gate count and circuit depth (-1.0 to 1.0).

        - ``-1``: Circuit optimized purely for swap count.
        - ``0`` (default): Balanced.
        - ``+1``: Circuit optimized purely for circuit depth.
    ignore_nodes : list[int], optional
        Qubit indices to ignore (e.g., faulty qubits).  Ignored nodes are
        removed from the effective connectivity via
        :func:`~iqm.qrisp_iqm.misc.shift_graph_edges` so that layout and
        routing never place logical qubits on them.
    ignore_edges : list[tuple[int, int]], optional
        Edges to ignore in the connectivity.  Like *ignore_nodes*, these
        are stripped from the effective topology before the passes are
        configured.

    Returns
    -------
    PassManager
        A configured PassManager with all IQM transpilation passes.
        Call ``pm.run(qc)`` to apply all passes to a circuit.

    Examples
    --------
    >>> from qrisp import QuantumCircuit
    >>> from iqm.qrisp_iqm import create_iqm_pass_manager
    >>> qc = QuantumCircuit(2); qc.h(0); qc.cx(0, 1); qc.measure(qc.qubits)
    >>> connectivity = [(0, 1), (1, 2), (2, 3)]
    >>> pm = create_iqm_pass_manager(connectivity)
    >>> transpiled_qc = pm.run(qc)

    Excluding a faulty qubit:

    >>> pm = create_iqm_pass_manager(
    ...     connectivity, ignore_nodes=[2]
    ... )
    >>> transpiled_qc = pm.run(circuit)

    """
    if ignore_nodes is None:
        ignore_nodes = []
    if ignore_edges is None:
        ignore_edges = []

    # Build the effective connectivity by removing ignored nodes/edges
    effective_connectivity = shift_graph_edges(connectivity, ignore_nodes, ignore_edges)

    if not effective_connectivity:
        raise ValueError(
            "create_iqm_pass_manager requires a non-empty effective connectivity. "
            "After removing ignored nodes and edges, no two-qubit gate loci remain. "
            "Check that at least one edge survives after applying "
            f"ignore_nodes={ignore_nodes} and ignore_edges={ignore_edges}."
        )

    device_qubits = len(set(sum([list(tp) for tp in effective_connectivity], [])))

    # -------------------------------------------------------------------
    # Pre-flight validation pass
    # -------------------------------------------------------------------
    @CircuitPass
    def _validate_qc(qc: QuantumCircuit) -> QuantumCircuit:
        if qc.num_qubits() > device_qubits:
            raise Exception(
                f"Tried to transpile QuantumCircuit with {qc.num_qubits()} "
                f"qubits onto device with {device_qubits} qubits"
            )
        return qc

    pm = PassManager()

    # Pre-flight validation
    pm += _validate_qc

    # Initial gate combination and decomposition
    pm += combine_single_qubit_gates
    pm += decompose(decompose_predicate=_leave_toffoli_predicate)
    pm += gray_synth_toffoli
    pm += decompose(decompose_predicate=_dissolve_toffoli_predicate)

    # Cleanup (barriers are now natively handled by the routing pipeline)
    pm += resolve_swaps
    pm += fuse_adjacents

    # Layout and routing
    pm += plasma_layout(effective_connectivity, effort=effort, depth_weight=depth_weight)
    pm += plasma_route(effective_connectivity, effort=effort, depth_weight=depth_weight)

    # SWAP optimization
    pm += commute_swaps

    # Reverse-order parallelization:
    # running the scheduler on reversed instruction order tends to pull
    # commuting CZ/SWAP-related structures together in the original order.
    # This exposes adjacent self-inverse pairs for the following cancellation.
    pm += reverse_parallelize

    # Decomposes composite two-qubit interactions but leaves swaps alive.
    # The subsequent commute swaps pass moves the swaps towards the atomic
    # two-qubit interactions (CZ, CX, CY).
    pm += decompose(decompose_predicate=_leave_swap_predicate)
    pm += commute_swaps

    # Cancel adjacent inverses
    pm += fuse_adjacents

    # Swap the implementation circuit of swap gates that act on one fresh
    # qubit, such that the control is canceled.
    pm += arrange_swaps

    # Final gate conversions
    pm += decompose()
    pm += convert_to_cz()
    pm += combine_single_qubit_gates
    pm += cancel_zero_controls
    pm += commute_phases(False)
    pm += convert_to_prx

    # Re-insert qubits that were excluded from the connectivity so the
    # output circuit has the full expected qubit set.
    if ignore_nodes:
        pm += _reinsert_ignored_qubits(ignore_nodes)

    return pm


# ---------------------------------------------------------------------------
# High-level transpilation entry point
# ---------------------------------------------------------------------------


def transpile_to_iqm(
    qc: QuantumCircuit,
    connectivity: list[tuple[int, int]],
    ignore_nodes: list[int] | None = None,
    ignore_edges: list[tuple[int, int]] | None = None,
    effort: int = 100,
    depth_weight: float = 0.4,
) -> QuantumCircuit:
    """Transpile a Qrisp QuantumCircuit for execution on IQM hardware.

    This function applies the full transpilation pipeline including:
    - Layout and routing (SABRE-based)
    - SWAP optimization
    - Gate conversions (to CZ + PRX)
    - Gate cancellation and optimization

    Internally delegates to :func:`create_iqm_pass_manager` and applies the
    resulting :class:`~qrisp.PassManager` to the circuit.

    Parameters
    ----------
    qc : QuantumCircuit
        The Qrisp QuantumCircuit to transpile.
    connectivity : list[list[int]]
        The connectivity describing the device topology.
    ignore_nodes : list[int], optional
        Qubit indices to ignore (e.g., faulty qubits).
    ignore_edges : list[tuple[int, int]], optional
        Edges to ignore in the connectivity.
    effort : int, optional
        Effort level for layout and routing.  Default is 100.
    depth_weight : float, optional
        Tradeoff between gate count and circuit depth (-1.0 to 1.0).

        - ``-1``: Circuit optimized purely for swap count.
        - ``0`` (default): Balanced.
        - ``+1``: Circuit optimized purely for circuit depth.

    Returns
    -------
    QuantumCircuit
        The transpiled circuit suitable for IQM hardware.

    Raises
    ------
    Exception
        If the circuit has more qubits than the device.

    Examples
    --------

    .. code-block:: python

        from qrisp import QuantumCircuit
        from iqm.qrisp_iqm import transpile_to_iqm

        # Circuit with a non-adjacent CX gate
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 2)
        qc.measure(range(3))

        # Transpile for a 4-qubit line topology
        connectivity = [(0, 1), (1, 2), (2, 3)]
        result = transpile_to_iqm(qc, connectivity)

        print(result)

    .. code-block:: text

                      ┌──────────────┐   ┌─┐
               qb_58: ┤ U3(π/2,-π,π) ├─■─┤M├──────────────────
                      ├──────────────┤ │ └╥┘┌─────────────┐┌─┐
               qb_60: ┤ U3(π/2,-π,π) ├─■──╫─┤ U3(π/2,0,0) ├┤M├
                      └──────────────┘    ║ └─────┬─┬─────┘└╥┘
               qb_59: ────────────────────╫───────┤M├───────╫─
                                          ║       └╥┘       ║
        amended_qb_0: ────────────────────╫────────╫────────╫─
                                          ║        ║        ║
                cb_3: ════════════════════╩════════╩════════╩═


    The non-adjacent CX(0,2) is decomposed and routed onto the line
    topology, resulting in a mixture of CZ and native U3 gates.

    """
    # Build and run the PassManager (handles connectivity shifting,
    # qubit validation, and re-insertion of ignored qubits).
    pm = create_iqm_pass_manager(
        connectivity,
        effort=effort,
        depth_weight=depth_weight,
        ignore_nodes=ignore_nodes,
        ignore_edges=ignore_edges,
    )
    return pm.run(qc)


def shift_graph_edges(
    edges: list[tuple[int, int]],
    removed_nodes: list[int],
    removed_edges: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    """Removes specified edges and nodes from a graph, then shifts the indices of the remaining nodes.

    Parameters
    ----------
    edges : list of tuple of int
        List of edges in the graph, where each edge is represented as a tuple (a, b) with node indices.
    removed_nodes : list of int
        List of node indices to be removed from the graph.
    removed_edges : list of tuple of int or None, optional
        List of edges to be removed from the graph before processing node removals
        and shifts. Each edge should be a tuple (a, b), using the initial node
        indices. Default is None (no additional edges removed).

    Returns
    -------
    new_edges : list of tuple of int
        List of edges for the updated graph after specified edges and nodes have been removed, with all
        node indices shifted so that remaining indices are contiguous from 0 to n - k - 1, where k is the
        number of removed nodes.

    Notes
    -----
    - All edges in ``removed_edges`` are eliminated from the initial edge list before node removal and index shifting.
    - Edges connected to any removed node are omitted.
    - After removal, all node indices greater than the removed nodes are decreased by the number of removed
      nodes with lower index, to ensure contiguous numbering from 0 upwards.

    """
    if removed_edges is not None:
        # Remove specified edges before node removal
        removed_edge_set: set[tuple[int, int]] = set(removed_edges)
        edges = [e for e in edges if (e[0], e[1]) not in removed_edge_set]

    if not edges:
        return []

    removed_set = set(removed_nodes)
    idx_map = {}
    next_shift = 0
    for i in range(max(max(a, b) for a, b in edges) + 1):
        if i in removed_set:
            next_shift += 1
            continue
        idx_map[i] = i - next_shift

    new_edges = [(idx_map[a], idx_map[b]) for a, b in edges if a not in removed_set and b not in removed_set]
    return new_edges
