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
"""Tests for functions inside of the ``generate_basis.py`` file."""

from iqm.qaoa.tree_calculation.generate_basis import calculate_t, get_z_basis_m, get_z_basis_m_t
import numpy as np
from numpy.typing import NDArray
import pytest


@pytest.mark.parametrize(
    "input_vector, expected_t",
    [
        (np.array([1, 1]), 0),  # Fully symmetric -> t = 0
        (
            np.array([1, -1, -1, -1, 1, 1]),
            2,
        ),  # Removing 2 elements from left and right leaves us with [-1, -1], which is symmetric
        (np.array([-1, 1, -1, 1, 1, 1]), 3),  # Not symmetric at all, so t = half of the length = 3
        (
            np.array([-1, 1, 1, -1, -1, -1, -1, 1, 1, 1]),
            1,
        ),  # Except for the last / first element, this is symmetric -> t = 1
    ],
)
def test_calculate_t(input_vector: NDArray[np.int8], expected_t: int) -> None:
    """Tests whether the function `calculate_t` works as expected for a set of given input vectors."""
    assert calculate_t(input_vector) == expected_t


def test_get_z_basis_m() -> None:
    """Tests some basis properties of the function generating the Z basis.

    Specifically tests that the array only contains +1's and -1's. And that summing it along the axis labelling
    the basis elements yields a vector of all zeros (since there should be the same number of +1's and -1's).
    """
    my_basis = get_z_basis_m(4)
    assert set(np.unique(my_basis)) == {-1, 1}  # Check that the array only contains +1 and -1
    sum_all_basis_vectors = np.sum(my_basis, axis=0)
    assert np.all(sum_all_basis_vectors == 0)  # Check that the sum of all basis elements adds up to a vector of 0's.


def test_get_z_basis_m_t(basis_6: NDArray[np.int8]) -> None:
    """Tests two basic properties of `get_z_basis_m_t`.

    The `get_z_basis_m_t` takes an input basis and sorts it based on its symmetry sectors. One sub-test is about
    the size of the symmetry sectors. The other test is checking whether the vectors in each symmetry sector indeed have
    the correct symmetry.
    """
    m = basis_6.shape[1] // 2

    basis_sorted = get_z_basis_m_t(basis_6)

    expected_dimensions = [2**m] + [2 ** (m - 2 + T) for T in range(1, m + 1)]
    actual_dimensions = [sym_sec.shape[0] for sym_sec in basis_sorted]

    assert actual_dimensions == expected_dimensions

    for t in range(1, m):
        for vector in basis_sorted[t]:
            assert vector[t - 1] == -vector[-t]  # Check that the `t`th element is anti-symmetric.
            assert np.all(
                vector[t:m] == vector[-t - 1 : m - 1 : -1]
            )  # Check that all elements in the middle are symmetric.
