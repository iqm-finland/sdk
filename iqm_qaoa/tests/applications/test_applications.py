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
"""Module for testing the functionality of the ``ProblemInstance`` class."""

from iqm.applications.maxcut import MaxCutInstance
from iqm.applications.mis import MISInstance
from iqm.applications.qubo import QUBOInstance
import networkx as nx
import numpy as np
import pytest


@pytest.mark.parametrize(
    "bit_str, correct_bit_str",
    [
        ("1111111", "0111111"),  # The first position has the most energy, so it should be flipped first
        ("0000000", "0000001"),  # The last position has the most energy, so it should be flipped first
        ("0101011", "0001011"),  # The 2nd bit has more energy than the 3rd from the end bit
        ("0001111", "0001111"),  # This is the lowest-energy solution, so it shouldn't be changed
        ("0000111", "0000111"),  # This is another lowest-energy solution (middle bit has 0 energy either way)
    ],
)
def test_local_bitflip_bitstring(bit_str: str, correct_bit_str: str) -> None:
    """Testing that the local_bitflip_bitstring method works.

    The problem rewards having 0's on the first `n/2` positions and 1's on the last `n/2` positions, whereas the closer
    the position is to the end/beginning of the bitstring, the higher weight it has.
    """
    n = 6  # We consider a problem of this many variables + 1

    # In the next few lines, we create a matrix to instantiate a suitable instance of ``QUBOInstance`` for this test.
    matrix = np.zeros((n + 1, n + 1))
    diagonal_values = np.arange(n // 2, -(n // 2) - 1, -1)
    np.fill_diagonal(matrix, diagonal_values)
    test_object = QUBOInstance(matrix)

    new_bit_str = test_object.local_bitflip_bitstring(bit_str)

    assert new_bit_str == correct_bit_str


def test_local_bitflip_postprocessing() -> None:
    """Testing that the `local_bitflip_postprocessing` method works.

    Tested on a problem that wants to have 1's on odd positions and 0's on even positions.
    """
    n = 6  # We consider a problem of this many variables

    # In the next few lines, we create a matrix to instantiate a suitable instance of ``QUBOInstance`` for this test.
    matrix = np.zeros((n, n))
    diagonal = np.tile([-1, 0.5], n // 2 + 1)[:n]
    np.fill_diagonal(matrix, diagonal)
    test_object = QUBOInstance(matrix)

    # All of the keys will get mapped to one single bitstring "101010"
    test_dict = {"101000": 10, "111010": 5, "001010": 3, "100010": 2, "101011": 10}

    new_dict = test_object.local_bitflip_postprocessing(test_dict)

    assert new_dict["101010"] == 30  # 30 is the sum of the values in `test_dict`


def test_percentile_selects_correct_number_of_counts(
    sparse_mis_instance: MISInstance, samples_dict: dict[str, int]
) -> None:
    """Tests that ``percentile_counts`` selects the correct number of counts.

    Runs over quintiles from [0, 0.2, 0.4, 0.6, 0.8, 1] and checks that the correct percentage of the input dictionary
    of counts has been selected.
    """
    n = 5
    total_samples = sum(samples_dict.values())
    for i in range(n + 1):
        filtered_counts = sparse_mis_instance.percentile_counts(samples_dict, i / n)
        assert sum(filtered_counts.values()) == i / n * total_samples


def test_percentile_separates_good_and_bad_bitstrings(
    sparse_mis_instance: MISInstance, samples_dict: dict[str, int]
) -> None:
    """Tests that ``percentile_counts`` separates the bad and the good bitstrings.

    Runs over quintiles from [0.2, 0.4, 0.6, 0.8]. For each quintile, separates the counts into the good ones and
    the bad ones. Checks that the maximum energy of the good ones is less or equal to the minimum energy of the bad
    ones.
    """
    n = 5
    for i in range(1, n):
        filtered_counts_best = sparse_mis_instance.percentile_counts(samples_dict, i / n, True)
        filtered_counts_worst = sparse_mis_instance.percentile_counts(samples_dict, 1 - i / n, False)

        worst_of_the_best = max(sparse_mis_instance.quality(b) for b in filtered_counts_best)
        best_of_the_worst = min(sparse_mis_instance.quality(b) for b in filtered_counts_worst)

        assert worst_of_the_best <= best_of_the_worst


def test_percentile_edge_cases(sparse_mis_instance: MISInstance, samples_dict: dict[str, int]) -> None:
    """Tests whether ``percentile_counts`` works correctly with edge cases."""
    # Choosing quantile 1 shouldn't change the counts at all.
    assert samples_dict == sparse_mis_instance.percentile_counts(samples_dict, 1)

    # Choosing quantile 0 should return empty dictionary.
    assert len(sparse_mis_instance.percentile_counts(samples_dict, 0.0)) == 0


def test_cvar_is_monotonic(sparse_mis_instance: MISInstance, samples_dict: dict[str, int]) -> None:
    """Tests whether ``cvar`` is monotonic (increasing) with respect to the threshold."""
    quantiles = np.linspace(0, 1, num=101)[1:]  # 100 evenly spaced values from 0.01 to 1.00.
    cvars = [sparse_mis_instance.cvar(samples_dict, quant) for quant in quantiles]

    assert np.all(cvars[:-1] <= cvars[1:])  # Check that the ``cvar`` list is non-decreasing.


def test_initialize_properties_special_g(special_g: nx.Graph) -> None:
    """Tests that `initialize_properties` works properly on a small test graph."""
    special_mc = MaxCutInstance(special_g)

    assert special_mc.lower_bound == -12  # Largest cut size is 12.
    assert special_mc.lowest_quality_bitstrings == {"0111000", "1000111"}  # This is the best solution (Z2 symmetric).
    assert special_mc.highest_quality_bitstrings == {"0000000", "1111111"}  # No cut.

    special_mis = MISInstance(special_g)

    assert special_mis.lower_bound == -3  # Largest MIS has size 3.
    assert special_mis.lowest_quality_bitstrings == {"0111000"}  # The MIS solution.
    assert special_mis.highest_quality_bitstrings == {"0000000"}  # The worst valid bitstring, selecting no nodes.


def test_initialize_properties_cycle() -> None:
    """Tests that `initialize_properties` works properly on a cycle graph.

    Relies on the properties of MIS on a cycle graph. On an even-length cycle, the solution is 2-degenerate. On
    an odd-length cycle, the degeneracy is the same as the number of nodes.
    """
    even_size = 12
    cycle_g_even = nx.cycle_graph(even_size)
    cycle_g_odd = nx.cycle_graph(even_size + 1)
    mis_even = MISInstance(cycle_g_even)
    mis_odd = MISInstance(cycle_g_odd)

    assert mis_even.lower_bound == -even_size / 2  # Half of nodes selected
    assert len(mis_even.lowest_quality_bitstrings) == 2

    assert mis_odd.lower_bound == -even_size / 2  # Can't select the extra node because of its neighbors.
    assert len(mis_odd.lowest_quality_bitstrings) == even_size + 1
