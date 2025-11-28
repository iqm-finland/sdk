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
"""Contains some auxiliary functions to help with tree angle calculations."""

import itertools

from numba import jit  # type: ignore[attr-defined]
from numba.typed import List
import numpy as np
from numpy.typing import NDArray


# This function is not used anywhere.
def get_z_basis(p: int) -> NDArray[np.int8]:
    r"""Generate an array containing the Z basis on a qubit.

    The Z basis basically means "all possible combinations of +1 and -1", multiple times. In the calcultion, this comes
    from resolutions of identity between the QAOA layers, which is why the Z basis has :math:`2^{2p+1}` elements.

    Args:
        p: The number of QAOA layers.

    Returns:
        A :mod:`numpy` array of size :math:`2^{2p+1}` containing the basis on the RCC.

    """
    basis = [np.array(b) for b in itertools.product(np.array([-1, 1]), repeat=2 * p + 1)]
    return np.array(basis, dtype=np.int8)


@jit(nopython=True)
def get_z_basis_m(m: int) -> NDArray[np.int8]:
    r"""Generate an array containing the Z basis on a qubit.

    Basically, this function is the same as :func:`get_z_basis`, but instead of specifying `p`, here one specifies
    the length of the basis elements directly. Effectively, the function implements the following line of code, but in
    a `numba`-compatible way.
    .. code-block:: python
        return np.array(list(itertools.product([-1, 1], repeat=2 * m)), dtype=np.int8)

    Args:
        m: The length of the basis elements to be generated. There will be :math:`4^m` total elements generated.

    Returns:
        A :mod:`numpy` array of size :math:`4^m \times 2m` containing the Z basis.

    """
    n = 2 * m
    size = 1 << n  # This is basically `2**n`, but apparently faster.
    result = np.empty((size, n), dtype=np.int8)
    for i in range(size):  # `i` is an integer whose binary representation corresponds to one basis element.
        for j in range(n):
            bit = (i >> j) & 1  # Check `j`-th bit of `i`.
            result[i, n - j - 1] = 2 * bit - 1  # Map it to +1 / -1.
    return result


@jit(nopython=True)
def calculate_t(v1: np.ndarray) -> np.int64:
    """Given a vector `v1`, calculate its symmetry sector.

    For an input vector `v1` of length :math:`2m`, the symmetry sector :math:`t` is the smallest integer, such that
    removing :math:`t` elements from the beginning and the end of `v1` makes the array symmetric with respect to flip,
    i.e., a palindrome. In other words, it's the smallest integer, such that starting from the middle of `v1`, the first
    :math:`m-t` elements to the left equal the first :math:`m-t` elements to the right. By definition :math:`t` lies
    between 0 and :math:`m` (included). Some examples:

    +-------------------------------+--------------------------------+---------------------+
    | Input array                   | Half of input length :math:`m` | Output :math:`t`    |
    +===============================+================================+=====================+
    | [-1, **1, 1, 1, 1**, 1]       | 3                              | 1                   |
    +-------------------------------+--------------------------------+---------------------+
    | [-1, 1, **-1, -1**, -1, 1]    | 3                              | 2                   |
    +-------------------------------+--------------------------------+---------------------+
    | [**-1, 1, -1, -1, 1, -1**]    | 3                              | 0                   |
    +-------------------------------+--------------------------------+---------------------+
    | [-1, 1, -1, 1, 1, 1]          | 3                              | 3                   |
    +-------------------------------+--------------------------------+---------------------+
    | [-1, **1, 1**, 1]             | 2                              | 1                   |
    +-------------------------------+--------------------------------+---------------------+
    | [-1, **1, 1, 1, 1, 1, 1**, 1] | 4                              | 1                   |
    +-------------------------------+--------------------------------+---------------------+

    Args:
        v1: The array whose symmetry sector we want to calculate.

    Returns:
        The symmetry sector of the input array.

    Raises:
        ValueError: If `v1` does not have the proper shape (1D array of even length).
        ValueError: If `v1` contains other entries than +1 and -1

    """
    if v1.ndim != 1 or len(v1) % 2 != 0:
        raise ValueError(f"Input must be a 1D array of even length. It has shape {v1.shape} and length {len(v1)}.")

    if not np.all(np.isin(v1, [-1, 1])):
        raise ValueError(f"The input array may only contain elements -1 or +1. It contains: {set(v1.flat)}.")

    m2 = len(v1)
    m = m2 // 2
    keep_iterating = True
    if np.sum(v1 * np.flip(v1)) == m2:
        # The input bitstring is completely reflection symmetric, t = 0.
        return np.int64(0)
    else:
        index = 0  # Index for iterating through the input array, STARTING FROM THE CENTER
        while keep_iterating:
            if v1[m - 1 - index] == v1[m + index]:
                # If the `index`th element to the left from the center is equal to the `index`th element to the right.
                index += 1
            else:
                keep_iterating = False
        return np.int64(m - index)


@jit(nopython=True)
def get_z_basis_m_t(basis: np.ndarray) -> list[NDArray[np.int8]]:
    """Takes an array containing (all) basis vectors and sorts them into a list based on their symmetry sector.

    The input is assumed to be a (full) array of basis vectors.

    Args:
        basis: The array containing the basis vectors to be sorted.

    Returns:
        A list of :mod:`numpy` arrays, each of which contains the basis vectors belonging to the corresponding symmetry
        sector.

    """
    # Create lists of arrays representing different basis sorting according to T symmetry.
    # Start from full basis (input) and seperate it according to T.
    m = basis.shape[1] // 2
    basis_all_ts = List()
    for t in range(m + 1):
        size = 2**m if t == 0 else 2 ** (m - 2 + t)
        basis_one_t = np.zeros((size, 2 * m), dtype=np.int8)
        basis_all_ts.append(basis_one_t)

    # Reset index trackers
    idxs = np.zeros(m + 1, dtype=np.int64)

    # Second pass: fill in the arrays
    for vector in basis:
        t_vector = calculate_t(vector)
        if t_vector == 0:
            idx = idxs[0]
            basis_all_ts[0][idx, :] = vector
            idxs[0] += 1
        elif vector[t_vector - 1] == 1:  # IMPORTANT: The `T_vector - 1`th element is fixed for nonzero `T_vector`.
            idx = idxs[t_vector]
            basis_all_ts[t_vector][idx, :] = vector
            idxs[t_vector] += 1

    return basis_all_ts
