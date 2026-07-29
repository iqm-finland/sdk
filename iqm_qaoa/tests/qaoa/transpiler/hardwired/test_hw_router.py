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
"""Tests for the hardwired router."""

import math

from dimod.generators import uniform
from iqm.qaoa.transpiler.hardwired.hardwired import hardwired_router
from iqm.qaoa.transpiler.quantum_hardware import QPU, CrystalQPUFromBackend
from iqm.qiskit_iqm.iqm_provider import IQMBackend
import networkx as nx


def test_hw_routing_number_of_cnots(apollo_backend: IQMBackend) -> None:
    """Tests that the hw router creates circuits with the correct number of cnots."""
    expected_cnot_numbers = {
        4: 13,
        5: 23,
        6: 34,
        7: 48,
        8: 63,
        9: 82,
        10: 103,
        11: 126,
        12: 152,
        13: 180,
        14: 211,
        15: 244,
    }

    for num_qubits in range(4, 16):
        random_bqm = uniform(num_qubits, "SPIN", low=-1, high=1, seed=1337)

        apollo = CrystalQPUFromBackend(apollo_backend)

        routing = hardwired_router(problem_bqm=random_bqm, qpu=apollo)

        number_of_ints = 0
        number_of_cnots = 0
        for layer in routing.layers:
            for i in layer.gates.edges(data=True):
                if i[2]["swap"]:
                    number_of_cnots += 3
                    if i[2]["int"]:
                        number_of_ints += 1
                elif i[2]["int"]:
                    number_of_cnots += 2
                    number_of_ints += 1

        assert number_of_cnots == expected_cnot_numbers[num_qubits]
        assert number_of_ints == math.comb(num_qubits, 2)


def test_hw_routing_gets_through_all_interactions(apollo_backend: IQMBackend) -> None:
    """Tests that the hw router gets through all interactions."""
    for num_qubits in range(4, 16):
        random_bqm = uniform(num_qubits, "SPIN", low=-1, high=1, seed=1337)
        apollo = CrystalQPUFromBackend(apollo_backend)

        route = hardwired_router(problem_bqm=random_bqm, qpu=apollo)

        assert 0 == route.remaining_interactions.size(), "Router did not finish all interactions."


def test_hw_router_finds_subgraph_in_arbitrary_qpu_graph() -> None:
    """Tests that the ``hardwired_router`` can find the correct subgraph in a specific QPU graph.

    12--0---1---2---3
        |       |   |
        9  10--11---4
      / | /     | / |
    13--8---7---6---5
    """
    test_graph = nx.cycle_graph(10)
    test_graph.add_edges_from({(12, 0), (13, 8), (13, 9), (8, 10), (10, 11), (11, 4), (11, 2), (11, 6), (4, 6)})
    test_qpu = QPU(test_graph)
    test_bqm = uniform(7, "SPIN", low=-1, high=1, seed=1337)

    route = hardwired_router(test_bqm, test_qpu)
    # Check that the problem is mapped on the correct hardware qubits.
    assert set(route.mapping.log2hard.values()) == {2, 3, 4, 5, 6, 11, 10}
