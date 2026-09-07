# Copyright 2022-2026 IQM
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

# This code has been translated to plain Python from the original code drafted in Cython provided by IBM under the
# following License:

# This code is part of Mthree.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Hadamard array generator, from :cite:`Bravyi_2021`."""

from __future__ import annotations

import math

import numpy as np


class HadamardGenerator:
    """Hadamard calibration generator."""

    def __init__(self, num_qubits: int) -> None:
        """Hadamard calibration generator.

        Generates a set of bit-arrays that evenly
        sample all independent and pair-wise correlated
        measurement errors.

        Args:
            num_qubits: The number of qubits.

        """
        self.name = "hadamard"
        self.num_qubits = num_qubits
        self.p = int(math.floor(math.log2(num_qubits) + 1))
        self.length = 2**self.p
        self.integer_bits = np.zeros(self.p, dtype=np.uint8)
        self.out_bits = np.zeros(num_qubits, dtype=np.uint8)
        self._iter_index = 0

    def __iter__(self) -> HadamardGenerator:
        """Return an iterator over the bit arrays (the class itself is also the iterator)."""
        self._iter_index = 0
        return self

    def __next__(self) -> np.ndarray:
        """Next bit array in the iteration."""
        if self._iter_index < self.length:
            self._iter_index += 1
            return self._generate_array(self._iter_index - 1)
        raise StopIteration

    def _generate_array(self, index: int) -> np.ndarray:
        """Return the bit array corresponding to ``index``."""
        if index > self.length - 1:
            raise ValueError(f"Index must within generator length {self.length}")  # M3Error > ValueError

        # Set the bitstrings for the integer_bits
        for kk in range(self.p):
            self.integer_bits[self.p - kk - 1] = (index >> kk) & 1

        for kk in range(self.num_qubits):
            tot = 0
            for jj in range(self.p):
                tot += self.integer_bits[self.p - jj - 1] and ((kk + 1) >> jj) & 1
            self.out_bits[kk] = tot % 2

        # Return a copy since the underlying memory will be reused
        return self.out_bits.copy()
