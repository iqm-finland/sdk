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

"""Routing-only pass using SABRE algorithm."""

from __future__ import annotations

from collections.abc import Callable

from iqm.qrisp_iqm.passes.routing.core.graph_processing_tools import connectivity_to_topology
from iqm.qrisp_iqm.passes.routing.core.parameter_selection import compute_parameters
from iqm.qrisp_iqm.passes.routing.core.sabre_meta_functions import (
    instruction_tuples_to_qc,
    sectionized_sabre,
)
from iqm.qrisp_iqm.passes.routing.core.sectionalization import prepare_dag_and_sections
import psutil
from qrisp import QuantumCircuit
from qrisp.circuit.pass_management.circuit_pass import CircuitPass


def plasma_route(  # noqa: PLR0913
    connectivity: list[tuple[int, int]],
    effort: int = 30,
    C: int | None = None,
    sections: int = 0,
    depth_weight: float = 0.0,
    tempering_range: int = 3,
    seed: int = 0,
) -> Callable[[QuantumCircuit], QuantumCircuit]:
    """Create a pass that performs routing (SWAP insertion) on the existing layout.

    This pass assumes the circuit already has a valid layout and only
    performs SWAP insertion to make the circuit executable on the given
    connectivity.

    Parameters
    ----------
    connectivity : list[tuple[int]]
        The list of edges representing the hardware topology.
    effort : int, optional
        Single knob controlling classical compute investment. Higher values
        explore more routing variants, improving circuit quality at the cost
        of longer compilation time.  Internally this derives the routing
        diversity multiplier *C* via :func:`compute_parameters` using the
        circuit's 2-qubit gate count.  Ignored when *C* is given
        explicitly.  Default 30.
    C : int or None, optional
        Override: routing diversity multiplier.  Total routing threads =
        ``C * cpu_count``.  When *None* (default), derived automatically
        from *effort* and the circuit's 2-qubit gate count.
    sections : int, optional
        Number of sections for sectionalized routing.

        - ``0`` (default): Automatic. Section boundaries are placed at
          structurally meaningful positions (TerminatorNodes in the
          PermeabilityGraph) with the count derived from circuit depth.
        - ``1``: No sectionalization — compile the circuit in one pass.
        - ``N > 1``: Exactly *N* sections with boundaries at
          TerminatorNode positions.

        Default is 0.
    depth_weight : float, optional
        Tradeoff between gate count and circuit depth (-1.0 to 1.0).

        - ``-1``: Optimize purely for gate count (selection_exponent=0,
          congestion_penalty=0).
        - ``0`` (default): Balanced sweet-spot (selection_exponent=0.5,
          congestion_penalty=0.1).
        - ``+1``: Optimize purely for circuit depth (selection_exponent=1.0,
          congestion_penalty=0.2).

        Internally derives two parameters via linear interpolation:

        - *selection_exponent* — geometric-mean exponent for trial selection:
          ``score = swaps^(1-e) * depth^e``.
        - *congestion_penalty* — swap-scoring penalty for congested qubits.
    tempering_range : int, optional
        Controls parallel tempering. When > 0, different parallel threads use
        different greediness values (exploration rates). Default is 3.
    seed : int, optional
        Seed for the random number generation used during routing. Each
        parallel thread and section derives its own seed from this base
        value, ensuring reproducible results. Default is 0.

    Returns
    -------
    Callable[[QuantumCircuit], QuantumCircuit]
        A pass function that transforms the circuit.

    Example
    -------
    >>> from qrisp import QuantumCircuit, PassManager
    >>> from iqm.qrisp_iqm import plasma_route
    >>> qc = QuantumCircuit(2); qc.cx(0, 1); qc.measure(qc.qubits)
    >>> pm = PassManager()
    >>> pm += plasma_route(connectivity=[(0,1), (1,2)])
    >>> transpiled_qc = pm.run(qc)

    """
    if not connectivity:
        raise ValueError(
            "plasma_route requires a non-empty connectivity list. "
            "The connectivity must contain at least one edge "
            "describing available two-qubit gate loci on the device."
        )

    _effort = max(1, effort)
    _C_override = C
    _w = max(-1.0, min(1.0, float(depth_weight)))
    _selection_exponent = 0.5 * (1.0 + _w)
    _congestion_penalty = 0.1 * (1.0 + _w)
    _tempering_range = tempering_range
    _sections = sections
    _TWO = 2
    _seed = seed

    @CircuitPass
    def _route(qc: QuantumCircuit) -> QuantumCircuit:
        cpu_count = psutil.cpu_count() or 1

        topology = connectivity_to_topology(connectivity)
        n_physical = topology.dist_matrix.shape[0]

        # Build CSR DAGs and compute section boundaries
        forward_sparse_dag, forward_instruction_list, section_lengths = prepare_dag_and_sections(
            qc,
            sections=_sections,
        )

        # Derive C from effort + circuit size, or use explicit override
        if _C_override is not None:
            local_C = max(1, _C_override)
        else:
            num_2qb_gates = sum(1 for instr in qc.data if instr.op.num_qubits == _TWO)
            params = compute_parameters(_effort, num_2qb_gates, cpu_count)
            local_C = params["C"]

        greediness = 5

        res_swaps, final_mapping = sectionized_sabre(
            topology,
            forward_sparse_dag,
            greediness,
            threads=local_C * cpu_count,
            section_lengths=section_lengths,
            tempering_range=_tempering_range,
            congestion_penalty=_congestion_penalty,
            selection_exponent=_selection_exponent,
            seed=_seed,
        )

        result_qc = instruction_tuples_to_qc(
            res_swaps,
            forward_instruction_list,
            qc,
            num_physical_qubits=n_physical,
        )
        return result_qc

    _route.__name__ = "plasma_route"
    return _route
