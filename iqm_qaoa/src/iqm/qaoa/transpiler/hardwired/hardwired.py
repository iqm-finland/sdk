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
"""This module contains "hard-wired" routing.

For problem sizes from 4 to 15, this creates the optimal :class:`~iqm.qaoa.transpiler.routing.Routing` so that all
2-qubit interactions can be executed and the number of swaps is minimized. We believe this is the optimal routing
strategy for fully / densely connected problems.
"""

from itertools import combinations

from dimod import BinaryQuadraticModel
from iqm.qaoa.transpiler.quantum_hardware import CrystalQPUFromBackend
from iqm.qaoa.transpiler.routing import Mapping, Routing
import networkx as nx


# pylint: disable=too-many-statements
# pylint: disable=too-many-branches
# pylint: disable=anomalous-backslash-in-string
def hardwired_router(problem_bqm: BinaryQuadraticModel, qpu: CrystalQPUFromBackend) -> Routing:  # noqa: PLR0915
    """The function that creates an optimal routing for all-to-all connected problems, designed by hand.

    The original code was written for hand-picked qubits from the Apollo QPU. When this was expanded to be used on
    any QPU, the algorithm had to be adjusted. Here is how it works now:

    1. First, the :class:`~dimod.BinaryQuadraticModel` representation of the problem is padded with extra
       interactions (of strength 0), to make it trully all-to-all connected.
    2. Then, based on the number of variables of the problem, we construct a dummy graph ``underlying_graph`` which
       represents the part of the QPU on which the circuit acts.
    3. We find a suitable mapping between ``underlying_graph`` and the QPU graph
       :attr:`~iqm.qaoa.transpiler.quantum_hardware.QPU.hardware_graph`. This mapping ``inverse_iso``
       is used in the :meth:`~iqm.qaoa.transpiler.routing.Routing.apply_int` and
       :meth:`~iqm.qaoa.transpiler.routing.Routing.apply_swap` method calls.
    4. Using the :meth:`~iqm.qaoa.transpiler.routing.Routing.apply_int` and
       :meth:`~iqm.qaoa.transpiler.routing.Routing.apply_swap` methods, the hardwired routing is constructed.

    Args:
        problem_bqm: The :class:`~dimod.BinaryQuadraticModel` representation of the problem, needed only to
            instantiate the :class:`~iqm.qaoa.transpiler.routing.Routing`.
        qpu: The :class:`~iqm.qaoa.transpiler.quantum_hardware.CrystalQPUFromBackend` object.

    Returns:
        A :class:`~iqm.qaoa.transpiler.routing.Routing` object containing the entire routing schedule.

    Raises:
        ValueError: If the number of variables of the problem is not between 4 and 15 (inclusive).

    """
    bqm_to_be_used = problem_bqm.copy()
    # Padding the BQM problem with 0-strength interactions.
    for v1, v2 in combinations(bqm_to_be_used.variables, 2):
        if (v1, v2) not in bqm_to_be_used.quadratic and (v2, v1) not in bqm_to_be_used.quadratic:
            bqm_to_be_used.add_quadratic(v1, v2, 0)

    # This is the graph of interactions of the hardwired transpiler.
    underlying_graph = nx.Graph()  # type: ignore[var-annotated]
    # Based on the size of the problem, the underlying graph will be different.
    dict_of_sets_of_edges = {
        4: {(13, 8), (13, 14), (14, 9), (9, 8)},
        5: {(13, 8), (13, 14), (8, 3), (14, 9), (9, 8)},
        6: {(13, 8), (13, 14), (3, 8), (4, 3), (14, 9), (9, 8), (9, 4)},
        7: {(13, 8), (13, 14), (3, 8), (4, 3), (8, 7), (14, 9), (9, 8), (9, 4)},
        8: {(9, 10), (13, 14), (3, 8), (13, 8), (4, 3), (8, 7), (14, 9), (9, 8), (9, 4)},
        9: {(9, 10), (13, 14), (3, 8), (13, 8), (10, 15), (4, 3), (8, 7), (14, 9), (9, 8), (14, 15), (9, 4)},
        10: {
            (9, 10),
            (13, 14),
            (3, 8),
            (13, 8),
            (10, 15),
            (10, 5),
            (4, 3),
            (8, 7),
            (4, 5),
            (14, 9),
            (9, 8),
            (14, 15),
            (9, 4),
        },
        11: {
            (9, 10),
            (13, 14),
            (3, 8),
            (13, 8),
            (10, 15),
            (10, 5),
            (10, 11),
            (4, 3),
            (8, 7),
            (4, 5),
            (14, 9),
            (9, 8),
            (14, 15),
            (9, 4),
        },
        12: {
            (9, 10),
            (13, 14),
            (3, 8),
            (13, 8),
            (10, 15),
            (10, 5),
            (10, 11),
            (4, 3),
            (8, 7),
            (11, 16),
            (15, 16),
            (4, 5),
            (14, 9),
            (9, 8),
            (14, 15),
            (9, 4),
        },
        13: {
            (9, 10),
            (13, 14),
            (3, 8),
            (13, 8),
            (10, 15),
            (10, 5),
            (10, 11),
            (4, 3),
            (8, 7),
            (11, 16),
            (11, 6),
            (15, 16),
            (4, 5),
            (14, 9),
            (9, 8),
            (5, 6),
            (14, 15),
            (9, 4),
        },
        14: {
            (4, 3),
            (9, 8),
            (13, 8),
            (13, 14),
            (10, 15),
            (4, 5),
            (14, 9),
            (5, 6),
            (14, 15),
            (9, 4),
            (9, 10),
            (10, 5),
            (10, 11),
            (11, 16),
            (15, 16),
            (7, 12),
            (3, 8),
            (8, 7),
            (11, 6),
            (13, 12),
        },
        15: {
            (4, 3),
            (9, 8),
            (13, 8),
            (13, 14),
            (10, 15),
            (4, 5),
            (14, 9),
            (5, 6),
            (14, 15),
            (9, 4),
            (9, 10),
            (10, 5),
            (10, 11),
            (11, 16),
            (15, 16),
            (7, 12),
            (3, 2),
            (3, 8),
            (8, 7),
            (11, 6),
            (13, 12),
            (7, 2),
        },
    }

    underlying_graph.add_edges_from(dict_of_sets_of_edges[bqm_to_be_used.num_variables])

    # For embedding the underlying graph into the QPU graph, we need to create a new ``GraphMatcher`` object.
    gm = nx.isomorphism.GraphMatcher(qpu.hardware_graph, underlying_graph)

    # Gets the subgraph isomorphism (and calculates its inverse).
    isomorphism = next(gm.subgraph_monomorphisms_iter())
    inverse_iso = {q2: q1 for q1, q2 in isomorphism.items()}
    # Use the calculated isomorphism to construct a ``Mapping``.
    mapping = Mapping(qpu, bqm_to_be_used, partial_initial_mapping={hq: i for i, hq in enumerate(isomorphism)})
    # Use ``mapping`` to construct the initial ``Routing``.
    route = Routing(bqm_to_be_used, qpu, initial_mapping=mapping)

    if bqm_to_be_used.num_variables == 4:
        route.apply_int((inverse_iso[13], inverse_iso[14]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[13], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[14], inverse_iso[9]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[13], inverse_iso[14]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[8]))  # type: ignore[arg-type]

    elif bqm_to_be_used.num_variables == 5:
        route.apply_int((inverse_iso[13], inverse_iso[14]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[13], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[14], inverse_iso[9]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[8], inverse_iso[3]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[13], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[13], inverse_iso[14]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[8]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[14], inverse_iso[9]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[8], inverse_iso[3]))  # type: ignore[arg-type]

    elif bqm_to_be_used.num_variables == 6:
        route.apply_int((inverse_iso[13], inverse_iso[14]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[4], inverse_iso[3]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[14], inverse_iso[9]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[3], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[13], inverse_iso[8]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[4]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[13], inverse_iso[14]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[4], inverse_iso[3]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[14], inverse_iso[9]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[3], inverse_iso[8]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[13], inverse_iso[14]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[4], inverse_iso[3]))  # type: ignore[arg-type]

    elif bqm_to_be_used.num_variables == 7:
        route.apply_int((inverse_iso[13], inverse_iso[14]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[4], inverse_iso[3]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[14], inverse_iso[9]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[3], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[8], inverse_iso[7]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[13], inverse_iso[8]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[4]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[8], inverse_iso[7]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[13], inverse_iso[14]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[4], inverse_iso[3]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[14], inverse_iso[9]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[3], inverse_iso[8]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[13], inverse_iso[14]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[4], inverse_iso[3]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[8], inverse_iso[7]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[8]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[14], inverse_iso[9]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[4]))  # type: ignore[arg-type]

    elif bqm_to_be_used.num_variables == 8:
        route.apply_int((inverse_iso[13], inverse_iso[14]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[4], inverse_iso[3]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[14], inverse_iso[9]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[3], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[8], inverse_iso[7]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[10]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[13], inverse_iso[8]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[4]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[8], inverse_iso[7]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[10]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[13], inverse_iso[14]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[4], inverse_iso[3]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[14], inverse_iso[9]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[3], inverse_iso[8]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[13], inverse_iso[14]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[4], inverse_iso[3]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[8], inverse_iso[7]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[10]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[8]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[13], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[4]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[14], inverse_iso[9]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[3], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[10]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[8], inverse_iso[7]))  # type: ignore[arg-type]

    elif bqm_to_be_used.num_variables == 9:
        route.apply_int((inverse_iso[13], inverse_iso[14]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[4], inverse_iso[3]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[14], inverse_iso[9]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[3], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[8], inverse_iso[7]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[10]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[13], inverse_iso[8]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[4]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[8], inverse_iso[7]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[10]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[13], inverse_iso[14]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[4], inverse_iso[3]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[14], inverse_iso[9]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[3], inverse_iso[8]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[13], inverse_iso[14]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[4], inverse_iso[3]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[8], inverse_iso[7]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[10]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[8]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[14], inverse_iso[9]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[4]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[13], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[3], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[8], inverse_iso[7]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[10]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[10], inverse_iso[15]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[14], inverse_iso[15]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[13], inverse_iso[14]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[14], inverse_iso[9]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[4]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[8]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[3], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[8], inverse_iso[7]))  # type: ignore[arg-type]

    elif bqm_to_be_used.num_variables == 10:
        route.apply_int((inverse_iso[13], inverse_iso[14]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[4], inverse_iso[3]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[14], inverse_iso[9]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[3], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[8], inverse_iso[7]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[10]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[13], inverse_iso[8]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[4]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[8], inverse_iso[7]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[10]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[13], inverse_iso[14]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[4], inverse_iso[3]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[14], inverse_iso[9]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[3], inverse_iso[8]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[13], inverse_iso[14]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[4], inverse_iso[3]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[8], inverse_iso[7]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[10]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[8]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[13], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[4]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[14], inverse_iso[9]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[3], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[10]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[8], inverse_iso[7]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[10], inverse_iso[15]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[4], inverse_iso[5]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[14], inverse_iso[15]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[10], inverse_iso[5]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[13], inverse_iso[14]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[10], inverse_iso[15]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[10]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[14], inverse_iso[9]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[4]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[8]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[3], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[8], inverse_iso[7]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[10]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[8]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[13], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[8], inverse_iso[7]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[3], inverse_iso[8]))  # type: ignore[arg-type]

    elif bqm_to_be_used.num_variables == 11:
        route.apply_int((inverse_iso[13], inverse_iso[14]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[4], inverse_iso[3]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[14], inverse_iso[9]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[3], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[8], inverse_iso[7]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[10]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[13], inverse_iso[8]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[4]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[8], inverse_iso[7]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[10]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[13], inverse_iso[14]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[4], inverse_iso[3]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[14], inverse_iso[9]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[3], inverse_iso[8]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[13], inverse_iso[14]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[4], inverse_iso[3]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[8], inverse_iso[7]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[10]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[8]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[13], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[4]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[14], inverse_iso[9]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[3], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[10]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[8], inverse_iso[7]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[10], inverse_iso[15]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[4], inverse_iso[5]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[14], inverse_iso[15]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[10], inverse_iso[5]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[13], inverse_iso[14]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[10], inverse_iso[15]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[10]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[14], inverse_iso[9]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[4]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[8]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[3], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[8], inverse_iso[7]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[10]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[8]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[13], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[8], inverse_iso[7]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[3], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[10], inverse_iso[11]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[10], inverse_iso[15]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[10], inverse_iso[5]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[10]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[14], inverse_iso[9]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[4]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[8]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[13], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[3], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[8], inverse_iso[7]))  # type: ignore[arg-type]

    elif bqm_to_be_used.num_variables == 12:
        route.apply_int((inverse_iso[13], inverse_iso[14]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[4], inverse_iso[3]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[14], inverse_iso[9]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[3], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[8], inverse_iso[7]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[10]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[13], inverse_iso[8]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[4]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[8], inverse_iso[7]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[10]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[13], inverse_iso[14]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[4], inverse_iso[3]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[14], inverse_iso[9]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[3], inverse_iso[8]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[13], inverse_iso[14]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[4], inverse_iso[3]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[8], inverse_iso[7]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[10]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[8]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[13], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[4]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[14], inverse_iso[9]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[3], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[10]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[8], inverse_iso[7]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[10], inverse_iso[15]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[4], inverse_iso[5]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[14], inverse_iso[15]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[10], inverse_iso[5]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[13], inverse_iso[14]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[10], inverse_iso[15]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[10]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[14], inverse_iso[9]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[4]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[8]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[3], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[8], inverse_iso[7]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[10]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[8]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[13], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[8], inverse_iso[7]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[3], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[10], inverse_iso[11]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[10], inverse_iso[15]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[10], inverse_iso[5]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[10]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[14], inverse_iso[9]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[4]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[8]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[13], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[3], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[8], inverse_iso[7]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[11], inverse_iso[16]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[15], inverse_iso[16]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[14], inverse_iso[15]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[10], inverse_iso[15]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[10], inverse_iso[5]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[10]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[4]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[8]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[13], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[3], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[8], inverse_iso[7]))  # type: ignore[arg-type]

    elif bqm_to_be_used.num_variables == 13:
        route.apply_int((inverse_iso[13], inverse_iso[14]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[4], inverse_iso[3]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[14], inverse_iso[9]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[3], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[8], inverse_iso[7]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[10]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[13], inverse_iso[8]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[4]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[8], inverse_iso[7]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[10]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[13], inverse_iso[14]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[4], inverse_iso[3]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[14], inverse_iso[9]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[3], inverse_iso[8]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[13], inverse_iso[14]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[4], inverse_iso[3]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[8], inverse_iso[7]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[10]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[8]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[13], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[4]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[14], inverse_iso[9]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[3], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[10]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[8], inverse_iso[7]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[10], inverse_iso[15]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[4], inverse_iso[5]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[14], inverse_iso[15]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[10], inverse_iso[5]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[13], inverse_iso[14]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[10], inverse_iso[15]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[10]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[14], inverse_iso[9]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[4]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[8]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[3], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[8], inverse_iso[7]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[10]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[8]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[13], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[8], inverse_iso[7]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[3], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[10], inverse_iso[11]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[10], inverse_iso[15]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[10], inverse_iso[5]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[10]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[14], inverse_iso[9]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[4]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[8]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[13], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[3], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[8], inverse_iso[7]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[11], inverse_iso[16]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[15], inverse_iso[16]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[14], inverse_iso[15]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[10], inverse_iso[15]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[10], inverse_iso[5]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[10]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[4]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[8]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[13], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[3], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[8], inverse_iso[7]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[5], inverse_iso[6]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[11], inverse_iso[6]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[11], inverse_iso[16]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[10], inverse_iso[11]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[10], inverse_iso[15]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[10]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[14], inverse_iso[9]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[4]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[8]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[13], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[3], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[8], inverse_iso[7]))  # type: ignore[arg-type]

    elif bqm_to_be_used.num_variables == 14:
        route.apply_int((inverse_iso[13], inverse_iso[14]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[4], inverse_iso[3]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[14], inverse_iso[9]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[3], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[8], inverse_iso[7]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[10]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[13], inverse_iso[8]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[4]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[8], inverse_iso[7]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[10]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[13], inverse_iso[14]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[4], inverse_iso[3]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[14], inverse_iso[9]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[3], inverse_iso[8]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[13], inverse_iso[14]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[4], inverse_iso[3]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[8], inverse_iso[7]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[10]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[8]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[13], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[4]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[14], inverse_iso[9]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[3], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[10]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[8], inverse_iso[7]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[10], inverse_iso[15]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[4], inverse_iso[5]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[14], inverse_iso[15]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[10], inverse_iso[5]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[13], inverse_iso[14]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[10], inverse_iso[15]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[10]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[14], inverse_iso[9]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[4]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[8]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[3], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[8], inverse_iso[7]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[10]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[8]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[13], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[8], inverse_iso[7]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[3], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[10], inverse_iso[11]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[10], inverse_iso[15]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[10], inverse_iso[5]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[10]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[14], inverse_iso[9]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[4]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[8]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[13], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[3], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[8], inverse_iso[7]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[11], inverse_iso[16]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[15], inverse_iso[16]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[14], inverse_iso[15]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[10], inverse_iso[15]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[10], inverse_iso[5]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[10]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[4]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[8]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[13], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[3], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[8], inverse_iso[7]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[5], inverse_iso[6]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[11], inverse_iso[6]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[11], inverse_iso[16]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[10], inverse_iso[11]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[10], inverse_iso[15]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[10]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[14], inverse_iso[9]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[4]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[8]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[13], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[3], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[8], inverse_iso[7]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[13], inverse_iso[12]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[7], inverse_iso[12]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[8], inverse_iso[7]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[3], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[8]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[14], inverse_iso[9]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[4]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[10]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[10], inverse_iso[15]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[10], inverse_iso[5]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[10], inverse_iso[11]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[11], inverse_iso[16]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[11], inverse_iso[6]))  # type: ignore[arg-type]

    elif bqm_to_be_used.num_variables == 15:
        route.apply_int((inverse_iso[13], inverse_iso[14]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[4], inverse_iso[3]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[14], inverse_iso[9]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[3], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[8], inverse_iso[7]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[10]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[13], inverse_iso[8]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[4]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[8], inverse_iso[7]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[10]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[13], inverse_iso[14]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[4], inverse_iso[3]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[14], inverse_iso[9]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[3], inverse_iso[8]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[13], inverse_iso[14]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[4], inverse_iso[3]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[8], inverse_iso[7]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[10]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[8]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[13], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[4]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[14], inverse_iso[9]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[3], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[10]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[8], inverse_iso[7]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[10], inverse_iso[15]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[4], inverse_iso[5]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[14], inverse_iso[15]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[10], inverse_iso[5]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[13], inverse_iso[14]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[10], inverse_iso[15]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[10]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[14], inverse_iso[9]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[4]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[8]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[3], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[8], inverse_iso[7]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[10]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[8]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[13], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[8], inverse_iso[7]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[3], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[10], inverse_iso[11]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[10], inverse_iso[15]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[10], inverse_iso[5]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[10]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[14], inverse_iso[9]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[4]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[8]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[13], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[3], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[8], inverse_iso[7]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[11], inverse_iso[16]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[15], inverse_iso[16]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[14], inverse_iso[15]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[10], inverse_iso[15]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[10], inverse_iso[5]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[10]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[4]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[8]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[13], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[3], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[8], inverse_iso[7]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[5], inverse_iso[6]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[11], inverse_iso[6]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[11], inverse_iso[16]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[10], inverse_iso[11]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[10], inverse_iso[15]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[10]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[14], inverse_iso[9]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[4]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[8]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[13], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[3], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[8], inverse_iso[7]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[13], inverse_iso[12]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[7], inverse_iso[12]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[8], inverse_iso[7]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[3], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[8]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[14], inverse_iso[9]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[4]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[10]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[10], inverse_iso[15]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[10], inverse_iso[5]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[10], inverse_iso[11]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[11], inverse_iso[16]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[11], inverse_iso[6]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[3], inverse_iso[2]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[7], inverse_iso[2]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[7], inverse_iso[12]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[8], inverse_iso[7]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[13], inverse_iso[8]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[8]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[14], inverse_iso[9]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[9], inverse_iso[4]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[9], inverse_iso[10]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[10], inverse_iso[15]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[10], inverse_iso[5]))  # type: ignore[arg-type]
        route.apply_swap((inverse_iso[10], inverse_iso[11]), attempt_int=True)  # type: ignore[arg-type]
        route.apply_int((inverse_iso[11], inverse_iso[16]))  # type: ignore[arg-type]
        route.apply_int((inverse_iso[11], inverse_iso[6]))  # type: ignore[arg-type]

    else:
        raise ValueError("The number of qubits needs to be between 4 and 15 (inclusive).")

    return route
