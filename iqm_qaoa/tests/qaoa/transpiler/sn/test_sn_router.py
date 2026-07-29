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
"""Tests for the SN router."""

import itertools
import math

from dimod.generators import uniform
from iqm.applications.sk import sk_generator
from iqm.qaoa.transpiler.quantum_hardware import QPU, CrystalQPUFromBackend, Grid2DQPU
from iqm.qaoa.transpiler.sn.sn import _find_rectangular_subgraph, _get_s_grid, sn_router
from iqm.qiskit_iqm import IQMFakeApollo
import numpy as np
import pytest


def test_sn_routing_number_of_swaps_and_ints() -> None:
    """Tests that the SN router creates circuits with the correct number of swaps and interactions.

    The number of interactions simply needs to match the number of edges in a fully-connected graph.
    The expected number of swaps is calculated from examining how many the SN algorithm should add.
    """
    for x, y in [(2, 2), (2, 4), (4, 2), (3, 3), (3, 4), (5, 3), (3, 5)]:
        num_qubits = x * y
        random_bqm = uniform(num_qubits, "SPIN", low=-1, high=1, seed=1337)
        grid_qpu = Grid2DQPU(x, y)

        routing = sn_router(problem_bqm=random_bqm, qpu=grid_qpu)

        number_of_swaps = 0
        number_of_ints = 0
        for layer in routing.layers:
            for i in layer.gates.edges(data=True):
                if i[2]["swap"]:
                    number_of_swaps += 1
                if i[2]["int"]:
                    number_of_ints += 1

        assert number_of_ints == math.comb(num_qubits, 2)

        # The number of total SN cycles.
        n_cycles = math.ceil(x / 2)
        # The number of horizontal swaps per full SN cycle, coming from applying S_0 and S_1.
        # S_0 is applied ``math.floor((y-1)/2)`` times. S_1 is applied ``math.ceil((y-1)/2)`` times.
        # S_0 contains ``math.floor(x*(y-1)/2)`` swaps. S_1 contains ``math.ceil(x*(y-1)/2)`` swaps.
        n_h_per_cycle = math.floor((y - 1) / 2) * math.floor(x * (y - 1) / 2) + math.ceil((y - 1) / 2) * math.ceil(
            x * (y - 1) / 2
        )
        # The number of vetical swaps per cycle, from S_2 and S_3. Vertical swaps aren't applied in the last cycle.
        n_v_per_cycle = y * (x - 1)
        # Total number of swaps.
        n_swaps = n_cycles * n_h_per_cycle + (n_cycles - 1) * n_v_per_cycle

        assert number_of_swaps == n_swaps


def test_find_rectangular_subgraph(qpu_with_hole: QPU) -> None:
    """Tests the helper function ``_find_rectangular_graph``."""
    # There should be no square of size 4-by-4 in the ``qpu_with_hole``.
    assert _find_rectangular_subgraph(qpu_with_hole, 4, 4) is None

    # Find a rectangle of dimensions 3-by-5 and check that it really is there
    hw_qb, height, width = _find_rectangular_subgraph(qpu_with_hole, 5, 3)
    hw_qb_coord = qpu_with_hole.hardware_layout[hw_qb]

    for h_i in range(height):
        for w_i in range(width):
            assert (hw_qb_coord[0] + h_i, hw_qb_coord[1] + w_i) in qpu_with_hole.hardware_layout.values()


def test_sn_finds_rectangle_in_qpu(qpu_with_hole: QPU) -> None:
    """Tests that when given the QPU with a hole, ``sn_router`` manages to find a suitable rectangle in it."""
    ns = [21, 18, 15]

    for n in ns:
        test_bqm = uniform(n, "SPIN", low=-1, high=1, seed=1337)
        route = sn_router(test_bqm, qpu_with_hole, do_line_swapping=False)
        # Get the coordinates of the selected qubits.
        coords = {qpu_with_hole.hardware_layout[qb_id] for qb_id in route.mapping.log2hard.values()}

        # Check that there's ``n`` of qubits selected.
        assert len(coords) == n

        x_coords = {x for x, _ in coords}
        y_coords = {y for _, y in coords}

        # The lengths of ``x_coords`` and ``y_coords`` should be the sides of the rectangle containing ``n`` qubits.
        assert len(x_coords) * len(y_coords) == n


def test_sn_does_not_find_rectangle_in_qpu(qpu_with_hole: QPU) -> None:
    """Tests that ``sn_router`` raises the correct error when it can't find a rectangular patch on the QPU."""
    test_prob = next(sk_generator(n=16, n_instances=1, distribution="gaussian"))

    with pytest.raises(RuntimeError, match="No rectangle of total size"):
        _ = sn_router(test_prob.bqm, qpu_with_hole, do_line_swapping=False)


def test_sn_line_swapping_n_layers_and_all_ints(qpu_with_hole: QPU) -> None:
    """Tests that the line swapping strategy executes all interactions, has the expected number of layers and swaps."""
    # Make the problem quite big to force it to look for a bended line in the QPU.
    n = 30
    test_prob = next(sk_generator(n=n, n_instances=1, distribution="gaussian"))
    routing = sn_router(test_prob.bqm, qpu_with_hole, do_line_swapping=True)
    assert routing.remaining_interactions.number_of_edges() == 0

    # ``n-2`` swap layers (combined with interactions) + 1 layer of interactions before and 1 after.
    assert len(routing.layers) == n

    assert routing.count_swap_gates() == (n - 1) * (n - 2) / 2


def test_get_s_does_not_accept_non_rectangular_layout(qpu_with_hole: QPU) -> None:
    """Tests that ``_get_s`` raises the correct error when given a non-rectangular layout of qubits."""
    with pytest.raises(ValueError, match="The provided layout is not a rectangular arrangement of qubits."):
        _ = _get_s_grid(qpu_with_hole.hardware_layout)


@pytest.mark.parametrize(
    "offset, dims",
    [
        ((5, -7), (6, 2)),  # 6-by-2 rectangle shifted by 5, -7 (``s[3]`` is expected to be empty here).
        ((0, 0), (6, 1)),  # 6-by-1 rectangle shifted by 0, 0  (``s[2]`` and ``s[3]`` are expected to be empty here).
        ((9, 9), (1, 9)),  # 1-by-9 rectangle shifted by 9, 9 (``s[0]`` and ``s[1]`` are expected to be empty here).
        ((1337, 420), (15, 15)),  # 15-by-15 square shifted by 1337, 420.
        ((10, -10), (15, 10)),
    ],
)
def test_s_has_correct_length_and_gates_are_between_neighbors(offset: tuple[int, int], dims: tuple[int, int]) -> None:
    """Tests that the gates in the four components of ``s`` behave as expected when given various input layouts.

    That means:
    1. They only connect neighboring qubits.
    2. Their number is as expected from examining carefully the drawings in the reference.
    """
    layout = dict(enumerate((x + offset[0], y + offset[1]) for x in range(dims[0]) for y in range(dims[1])))
    s = _get_s_grid(layout)

    for gate in itertools.chain(*s):  # Iterate over all gates in ``s[0]``, ``s[1]``, ``s[2]``, ``s[3]`` at once.
        # Using ``max`` and ``min`` to uniquely identify the qubits because ``gate`` is a frozenset.
        qb1 = layout[max(gate)]
        qb2 = layout[min(gate)]
        manhattan_distance_between_qubits = np.sum(np.abs(np.array(qb1) - np.array(qb2)))
        # Check that gates are only between neighboring qubits (i.e., Manhattan distance equal to 1).
        assert manhattan_distance_between_qubits == 1

    # Check that the number of gates in each ``s`` element is correct.
    assert len(s[0]) == np.floor(dims[0] * (dims[1] - 1) / 2)
    assert len(s[1]) == np.ceil(dims[0] * (dims[1] - 1) / 2)
    assert len(s[2]) == np.ceil((dims[0] - 1) / 2) * dims[1]
    assert len(s[3]) == np.ceil((dims[0] - 2) / 2) * dims[1]


def test_sn_works_with_fake_backends() -> None:
    """Tests that `sn_router` works with fake backends, i.e., it performs the routing without an error."""
    sk_instance = next(sk_generator(n=15, n_instances=1, distribution="gaussian"))
    crystal_qpu = CrystalQPUFromBackend(IQMFakeApollo())

    # We don't care about the result, we just want to check if it runs.
    sn_router(problem_bqm=sk_instance.bqm, qpu=crystal_qpu)
