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
"""Module for testing that the greedy router executes all interactions."""

from dimod import BinaryQuadraticModel, to_networkx_graph
from iqm.applications.mis import MISInstance
from iqm.qaoa.transpiler.quantum_hardware import QPU, Grid2DQPU
from iqm.qaoa.transpiler.sparse.greedy_router import Routing, _greedy_pair_mapper, greedy_router


def test_greedy_pair_mapper_basics_work() -> None:
    """Tests the greedy pair mapper on 4-variable all-to-all connected problem, fixing the "buffer interactions"."""
    qpu = Grid2DQPU(2, 2)

    # Create a 4-variable all-to-all BQM
    linear = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
    quadratic = {(0, 1): 1, (1, 2): 1, (2, 3): 1, (3, 0): 1, (0, 2): 1, (1, 3): 1}
    bqm = BinaryQuadraticModel(linear, quadratic, 0.0, "SPIN")

    custom_buffer_interactions = {frozenset((0, 2)), frozenset((1, 3))}
    route = Routing(bqm, qpu)
    problem_graph = to_networkx_graph(bqm)
    _greedy_pair_mapper(route, custom_buffer_interactions, {0, 1, 2, 3}, problem_graph)
    for pair in custom_buffer_interactions:
        assert route.remaining_interactions.has_edge(*pair), (
            "The interactions given in ``custom_buffer_interactions`` weren't executed."
        )


def test_greedy_router_gets_through_all_interactions(bqms: list[BinaryQuadraticModel], square_qpu: QPU) -> None:
    """Tests if the greedy router gets through all interactions."""
    for bqm in bqms:
        route = greedy_router(bqm, square_qpu)
        assert 0 == route.remaining_interactions.size(), "Router did not finish all interactions."

        number_of_ints = 0
        for layer in route.layers:
            for i in layer.gates.edges(data=True):
                if i[2]["int"]:
                    number_of_ints += 1

        assert number_of_ints == bqm.num_interactions


def test_more_color_pairs_improves_results(sparse_mis_instance: MISInstance, square_qpu: QPU) -> None:
    """Test that the more pairs of colors we consider for the routing, the better the results becomes."""
    # The list of the post-selected routings
    best_routings: list[Routing] = [
        greedy_router(sparse_mis_instance.bqm, square_qpu, num_of_pairs)
        for num_of_pairs in range(1, 11)  # This should be the max number of pairs.
    ]

    assert all(
        len(item.layers) >= len(next_item.layers)
        for item, next_item in zip(best_routings[:-1], best_routings[1:], strict=True)
    )
