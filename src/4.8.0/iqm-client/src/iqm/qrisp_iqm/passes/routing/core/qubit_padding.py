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

"""Helpers for padding a circuit out to the device qubit count.

Layout passes map the circuit onto a device topology that usually holds more
qubits than the circuit uses.  The missing qubits are appended as *padding*
qubits named ``f"padding_qb_{counter}"``.

Because several passes can run one after another (e.g. ``manual_layout``
followed by ``plasma_layout``), a pass must not restart the counter at zero:
the incoming circuit may already carry padding qubits from an earlier pass,
and reusing a name raises a ValueError. :func:`pad_qubits` therefore
continues the numbering where the incoming circuit left off.
"""

from __future__ import annotations

import re

from qrisp import QuantumCircuit, Qubit

PADDING_QUBIT_PREFIX = "padding_qb_"
"""Name prefix of the qubits the layout passes append as padding."""

_PADDING_QUBIT_PATTERN = re.compile(rf"^{re.escape(PADDING_QUBIT_PREFIX)}(\d+)$")


def next_padding_qubit_index(qc: QuantumCircuit) -> int:
    """First padding-qubit counter value that is free in ``qc``.

    Args:
        qc: The circuit to inspect.

    Returns:
        ``0`` if ``qc`` carries no padding qubits, otherwise one above the
        largest counter found among its qubit identifiers.

    """
    return (
        max(
            (
                int(match.group(1))
                for qubit in qc.qubits
                if (match := _PADDING_QUBIT_PATTERN.match(qubit.identifier)) is not None
            ),
            default=-1,
        )
        + 1
    )


def pad_qubits(qc: QuantumCircuit, n_qubits: int) -> list[Qubit]:
    """Pad a quantum circuit in-place until it holds the given number of qubits.

    The padding counter continues from the padding qubits already present in ``qc``,
    so chaining layout passes never produces duplicate qubit names.

    Args:
        qc: The circuit to pad.  Modified in-place.
        n_qubits: The desired qubit count.  If ``qc`` already holds at
            least that many qubits, nothing is added.

    Returns:
        The freshly created qubits, in the order they were appended.

    """
    n_qubits_current = qc.num_qubits()
    first_index = next_padding_qubit_index(qc)
    padding_qubits: list[Qubit] = [
        Qubit(f"{PADDING_QUBIT_PREFIX}{first_index + k}") for k in range(n_qubits - n_qubits_current)
    ]

    for qubit in padding_qubits:
        qc.add_qubit(qubit)

    return padding_qubits
