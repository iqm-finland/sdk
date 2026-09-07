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

"""Permutation utilities for plasma-sabre routing passes."""

from __future__ import annotations

from numba import njit
import numpy as np


@njit(cache=True)
def invert_permutation(permutation: np.ndarray) -> np.ndarray:
    """Return the inverse permutation.

    Parameters
    ----------
    permutation : np.ndarray
        Input permutation array.

    Returns
    -------
    np.ndarray
        The inverse permutation.

    """
    dt = permutation.dtype
    res = np.zeros(len(permutation), dtype=dt)
    res[permutation] = np.arange(len(permutation), dtype=dt)
    return res


@njit(cache=True)
def mul_perm(perm_a: np.ndarray, perm_b: np.ndarray) -> np.ndarray:
    """Concatenate two permutations.

    Parameters
    ----------
    perm_a : np.ndarray
        First permutation.
    perm_b : np.ndarray
        Second permutation.

    Returns
    -------
    np.ndarray
        The composed permutation.

    """
    return perm_a[perm_b]


@njit(cache=True)
def permute_array(address_to_citizen: np.ndarray, path: np.ndarray) -> np.ndarray:
    """Permute an address-to-citizen array along a given swap path.

    Parameters
    ----------
    address_to_citizen : np.ndarray
        Address-to-citizen mapping.
    path : np.ndarray
        Swap path.

    Returns
    -------
    np.ndarray
        The permuted array.

    """
    address_to_citizen[path[:-1]] = np.roll(address_to_citizen[path[:-1]], -1)
    return address_to_citizen
