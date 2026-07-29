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
"""Module for testing mostly the properties of the class :class:`~iqm.qaoa.transpiler.routing.Layer`."""

from collections.abc import Callable
import math

from dimod.generators import uniform
from iqm.applications.maxcut import MaxCutInstance
from iqm.applications.sk import SherringtonKirkpatrick
from iqm.qaoa.transpiler.hardwired.hardwired import hardwired_router
from iqm.qaoa.transpiler.quantum_hardware import QPU, CrystalQPUFromBackend, HardEdge
from iqm.qaoa.transpiler.routing import Layer
from iqm.qaoa.transpiler.sn.sn import sn_router
from iqm.qaoa.transpiler.sparse.greedy_router import greedy_router
from iqm.qiskit_iqm.iqm_backend import IQMBackendBase
import networkx as nx
from qiskit.quantum_info import Statevector


def test_applying_gates_keeps_layer_valid(
    matching_gates: set[HardEdge], layer: Layer, is_layer_valid: Callable[[Layer], bool]
) -> None:
    """Applies interaction, swap and combined swap-interaction gates into the layer and checks its validity."""
    size = len(matching_gates) // 3  # We'll apply gates on third of the ``matching_gates`` interactions
    for _ in range(size):
        layer.apply_int_gate(matching_gates.pop())
        assert is_layer_valid(layer), "After adding interaction gate the layer was not valid any more."
    for _ in range(size):
        layer.apply_swap_gate(matching_gates.pop())
        assert is_layer_valid(layer), "After adding swap gate the layer was not valid any more."
    for gate in matching_gates:
        layer.apply_swap_gate(gate)
        layer.apply_int_gate(gate)
        assert is_layer_valid(layer), "After adding a combined swap-interaction gate the layer was not valid any more."


def test_padding_hw_works() -> None:
    """Testing that padding problems work for the hardwired transpiler.

    Creates an almost-fully-connected BQM and inputs it to ``hardwired_router``. Then checks that the routing is
    performed, i.e., the remaining interactions get to 0 and the number of interaction gates is equal to the number of
    qubit pairs. The routing is oblivious to the interaction strengths, so it adds interaction gates even if
    the interaction strength is 0.
    """
    # 10 node graph
    qpu_graph = nx.grid_2d_graph(3, 3)
    qpu_graph.add_edge((1, 2), (1, 3))
    layout: dict[int, tuple[int, int]] = {}
    for i, coords in enumerate(qpu_graph.nodes):
        layout[i] = coords

    nx.relabel_nodes(qpu_graph, {coords: ind for ind, coords in layout.items()})

    local_qpu = QPU(qpu_graph, layout)
    n = qpu_graph.number_of_nodes()
    # Start with a fully-connected BQM instance, with quasi-random interaction strengths (they're not important anyway)
    almost_fully_connected_bqm = uniform(n, "BINARY", seed=1337)
    # Arbitrarily remove a few interactions
    almost_fully_connected_bqm.remove_interaction(1, 4)
    almost_fully_connected_bqm.remove_interaction(1, 3)
    almost_fully_connected_bqm.remove_interaction(2, 5)

    local_routing = hardwired_router(almost_fully_connected_bqm, local_qpu)
    assert local_routing.remaining_interactions.number_of_edges() == 0

    number_of_ints = 0
    for layer in local_routing.layers:
        for _, _, edge_data in layer.gates.edges(data=True):  # We ignore the information about the nodes.
            if edge_data["int"]:
                number_of_ints += 1

    assert number_of_ints == math.comb(n, 2) - 3  # The number of realized interactions, equal to nC2 - 3


def test_padding_sn_works(square_qpu: QPU) -> None:
    """Testing that padding problems work for the swap-network transpiler.

    Creates an almost-fully-connected BQM and inputs it to ``sn_router``. Then checks that the routing is performed,
    i.e., the remaining interactions get to 0 and the number of interaction gates is equal to the number of qubit
    pairs. The routing is oblivious to the interaction strengths, so it adds interaction gates even if the interaction
    strength is 0.
    """
    n = len(square_qpu.qubits)
    # Start with a fully-connected BQM instance, with quasi-random interaction strengths (they're not important anyway)
    almost_fully_connected_bqm = uniform(n, "BINARY", seed=1337)
    # Arbitrarily remove a few interactions
    almost_fully_connected_bqm.remove_interaction(1, 4)
    almost_fully_connected_bqm.remove_interaction(1, 3)
    almost_fully_connected_bqm.remove_interaction(2, 5)

    local_routing = sn_router(almost_fully_connected_bqm, square_qpu)
    assert local_routing.remaining_interactions.number_of_edges() == 0

    number_of_ints = 0
    for layer in local_routing.layers:
        for _, _, edge_data in layer.gates.edges(data=True):  # We ignore the information about the nodes.
            if edge_data["int"]:
                number_of_ints += 1

    assert number_of_ints == math.comb(len(square_qpu.qubits), 2) - 3


def test_circuit_with_cancelled_cnots_is_equivalent(
    sparse_maxcut_instance: MaxCutInstance, apollo_backend: IQMBackendBase, small_sk_instance: SherringtonKirkpatrick
) -> None:
    """Test that if we cancel CNOTs in the :mod:`qiskit` circuit, it remains equivalent to the full circuit."""
    garnet_qpu = CrystalQPUFromBackend(apollo_backend)

    rt_greedy = greedy_router(sparse_maxcut_instance.bqm, garnet_qpu)
    rt_hardwired = hardwired_router(small_sk_instance.bqm, garnet_qpu)
    rt_sn = sn_router(small_sk_instance.bqm, garnet_qpu)

    qc_greedy_1 = rt_greedy.build_qiskit(betas=[0.1, 0.2], gammas=[0.3, 0.4], cancel_cnots=False, measurement=False)
    qc_greedy_2 = rt_greedy.build_qiskit(betas=[0.1, 0.2], gammas=[0.3, 0.4], cancel_cnots=True, measurement=False)

    assert Statevector.from_instruction(qc_greedy_1).equiv(Statevector.from_instruction(qc_greedy_2))

    qc_hw_1 = rt_hardwired.build_qiskit(betas=[0.1, 0.2], gammas=[0.3, 0.4], cancel_cnots=False, measurement=False)
    qc_hw_2 = rt_hardwired.build_qiskit(betas=[0.1, 0.2], gammas=[0.3, 0.4], cancel_cnots=True, measurement=False)

    assert Statevector.from_instruction(qc_hw_1).equiv(Statevector.from_instruction(qc_hw_2))

    qc_sn_1 = rt_sn.build_qiskit(betas=[0.1, 0.2], gammas=[0.3, 0.4], cancel_cnots=False, measurement=False)
    qc_sn_2 = rt_sn.build_qiskit(betas=[0.1, 0.2], gammas=[0.3, 0.4], cancel_cnots=True, measurement=False)

    assert Statevector.from_instruction(qc_sn_1).equiv(Statevector.from_instruction(qc_sn_2))
