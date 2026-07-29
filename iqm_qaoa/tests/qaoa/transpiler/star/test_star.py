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
"""Test(s) for "routing" on the star QPU."""

from dimod import from_networkx_graph
from iqm.qaoa.transpiler.quantum_hardware import StarQPU
from iqm.qaoa.transpiler.star.star import star_router
import networkx as nx
import pytest


# A graph (which represents a problem to solve) and the size of its minimum vertex cover.
@pytest.mark.parametrize(
    "graph, size_of_mvc",
    [
        (nx.complete_graph(16), 15),
        (nx.cycle_graph(16), 8),
        (nx.star_graph(15), 1),  # This graph has 15 spokes and 1 center, so 16 nodes total.
        (nx.grid_2d_graph(4, 4), 8),
    ],
)
def test_star_routing_for_known_cases(graph: nx.Graph, size_of_mvc: int) -> None:
    """Checks "routing" on star QPU for a bunch of special case problem graphs.

    Specifically, checks if the number of move gates corresponds to the known size of the minimum vertex cover on
    the selected problem graphs.
    """
    bqm = from_networkx_graph(graph, vartype="BINARY")  # Create BQM from the input graph.
    for v1, v2 in bqm.quadratic:
        bqm.set_quadratic(v1, v2, 1)  # Set its interactions (by default they're zero)
    star_qpu = StarQPU(16)
    route = star_router(bqm, star_qpu)

    assert route.count_move_gates() == 2 * size_of_mvc

    number_of_ints = 0
    for layer in route.layers:
        if layer[0] == "int":
            number_of_ints += 1

    assert number_of_ints == graph.number_of_edges()
