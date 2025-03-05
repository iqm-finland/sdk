# Copyright 2024 IQM
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
"""Enumerations for representing a set of important one- and two-qubit gates."""

from __future__ import annotations

from enum import Enum, IntEnum, unique

import numpy as np


@unique
class XYGate(IntEnum):
    """Enumerate all single-qubit Clifford gates whose rotation axis is in the XY plane.

    Members of this enum can be mapped to the corresponding unitary propagator using
    :data:`XYGATE_UNITARIES`.

    Only used in the tomography experiments.
    """

    IDENTITY = 0
    X_90 = 1
    X_180 = 2
    X_M90 = 3
    Y_90 = 4
    Y_180 = 5
    Y_M90 = 6


XYGATE_UNITARIES = {
    XYGate.IDENTITY: np.eye(2, dtype=complex),
    XYGate.X_90: np.array([[1, -1j], [-1j, 1]], dtype=complex) / np.sqrt(2),
    XYGate.X_180: np.array([[0, -1j], [-1j, 0]], dtype=complex),
    XYGate.X_M90: np.array([[1, 1j], [1j, 1]], dtype=complex) / np.sqrt(2),
    XYGate.Y_90: np.array([[1, -1], [1, 1]], dtype=complex) / np.sqrt(2),
    XYGate.Y_180: np.array([[0, -1], [1, 0]], dtype=complex),
    XYGate.Y_M90: np.array([[1, 1], [-1, 1]], dtype=complex) / np.sqrt(2),
}
"""Mapping of XYGates to the corresponding SU(2) matrices"""


@unique
class TwoQubitGate(Enum):
    """Enumerates a subset of two-qubit gates.

    Members of this enum can be mapped to the corresponding unitary propagator using
    the dictionary returned by :data:`TWO_QUBIT_UNITARIES`.
    """

    CZ = 0
    """Controlled-Z gate."""
    ISWAP = 1
    """iSWAP gate."""
    SQRT_ISWAP = 2
    """Square root of the iSWAP gate."""


TWO_QUBIT_UNITARIES = {
    TwoQubitGate.CZ: np.array(np.diag([1, 1, 1, -1]), dtype=float),
    TwoQubitGate.ISWAP: np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]], dtype=complex),
    TwoQubitGate.SQRT_ISWAP: np.array(
        [[1, 0, 0, 0], [0, 1 / np.sqrt(2), 1j / np.sqrt(2), 0], [0, 1j / np.sqrt(2), 1 / np.sqrt(2), 0], [0, 0, 0, 1]],
        dtype=complex,
    ),
}
"""Mapping of TwoQubitGates to the corresponding U(4) matrices"""
