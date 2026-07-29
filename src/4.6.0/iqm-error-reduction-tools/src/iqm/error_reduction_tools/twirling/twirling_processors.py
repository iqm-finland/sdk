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

"""Shared utility functions for processing measurement counts.

These primitives are used by both readout_characterization and error_reduction_tools
for merging and un-twirling measurement count dictionaries.
"""

from collections.abc import Mapping, Sequence
from typing import TypeAlias

from iqm.error_reduction_tools.utils.general_utils import remove_whitespace_from_bitstrings

CountValue: TypeAlias = int | float


def sum_counts(list_counts: Sequence[Mapping[str, CountValue]]) -> dict[str, float]:
    """Sum a list of measurement counts dictionaries.

    Args:
        list_counts: List of dictionaries containing measurement counts.

    Returns:
        Dictionary with summed measurement counts.

    """
    summed_counts: dict[str, float] = {}
    for counts in list_counts:
        for bitstring, count in counts.items():
            if bitstring in summed_counts:
                summed_counts[bitstring] += count
            else:
                summed_counts[bitstring] = count
    return summed_counts


def untwirl_counts(counts: dict[str, CountValue], rot_string: list[str] | str | None) -> dict[str, float]:
    """Apply untwirling to measurement counts.

    Flips bits in measurement bitstrings according to a readout twirling string.
    For each qubit where the twirling operation was 'X', the corresponding
    measurement bit is flipped (0 ↔ 1).

    Args:
        counts: Dictionary of measurement counts.
        rot_string: List of readout twirling Pauli strings or a string
            (BIG endian, so the opposite of Qiskit's little endian).

    Returns:
        Dictionary of untwirled counts.

    Raises:
        ValueError: If rot_string length doesn't match number of qubits.

    Example:
        >>> counts = {'000': 10, '001': 5, '010': 3}
        >>> rot_string = "XII"
        >>> untwirl_counts(counts, rot_string)
        {'001': 10, '000': 5, '011': 3}

    """
    untwirled_counts: dict[str, float] = {}

    counts = remove_whitespace_from_bitstrings(counts)

    num_qubits = len(list(counts.keys())[0])
    rot_string_str: str = "I"
    if isinstance(rot_string, list):
        rot_string_str = "".join(rot_string)
    elif isinstance(rot_string, str):
        rot_string_str = rot_string

    if rot_string_str == "I":
        rot_string_str = "I" * num_qubits
    elif len(rot_string_str) != num_qubits:
        raise ValueError(f"Length of rot_string ({len(rot_string_str)}) does not match number of qubits ({num_qubits})")

    # Build XOR mask: rot_string_str is big-endian so position j (left) maps to
    # integer bit weight 2^j (counting from the right of the string).
    mask = sum(1 << j for j, c in enumerate(rot_string_str) if c == "X")

    for bitstring, count in counts.items():
        new_int = int(bitstring, 2) ^ mask
        untwirled_counts[format(new_int, f"0{num_qubits}b")] = count

    return untwirled_counts


def untwirl_and_sum_counts(
    raw_counts_list: list[dict[str, CountValue]], rot_string_list: list[str]
) -> dict[str, float]:
    """Untwirl and sum measurement counts from multiple circuits.

    This function takes a list of dictionaries containing measurement counts
    and a list of readout twirling strings, applies untwirling based on the
    provided strings and then sums the counts.

    Args:
        raw_counts_list: List of dictionaries containing measurement counts.
        rot_string_list: List of readout twirling strings (BIG endian).

    Returns:
        A dictionary with summed and untwirled measurement counts.

    """
    untwirled = [untwirl_counts(counts, rot_string) for counts, rot_string in zip(raw_counts_list, rot_string_list)]
    return sum_counts(untwirled)
