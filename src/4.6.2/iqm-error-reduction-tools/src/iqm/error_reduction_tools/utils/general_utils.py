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

"""Utils for circuit analysis."""

from collections import Counter
from collections.abc import Mapping
import re
from typing import TypeAlias, TypeVar

import numpy as np

CountValue: TypeAlias = int | float
_V = TypeVar("_V", int, float)


def remove_whitespace_from_bitstrings(counts: Mapping[str, _V]) -> dict[str, _V]:
    """Remove whitespace from bitstring keys in a counts dictionary.

    Args:
        counts: Dictionary mapping bitstrings (possibly with spaces) to counts.

    Returns:
        Dictionary with whitespace removed from all bitstring keys.

    Example:
        >>> remove_whitespace_from_bitstrings({"0 1": 10, "1 0": 5})
        {'01': 10, '10': 5}

    """
    return {k.replace(" ", ""): v for k, v in counts.items()}


def expectation_from_counts(counts: dict[str, int] | dict[str, float], pauli_string: str | None = None) -> float:
    """Calculate expectation value of a Pauli string from measurement counts.

    Args:
        counts: Dictionary mapping bitstrings to their counts (``int`` or ``float`` after mitigation).
        pauli_string: String of Pauli operators ('I' or 'Z') for each qubit.
            If ``None``, defaults to all 'Z' operators.

    .. note::

        Bitstrings and Pauli strings both follow the same ordering (typically little-endian).
        This is consistent with ``qiskit.result.sampled_expectation_value``.

    Example:
        >>> counts = {"01": 10}
        >>> pauli_string = "ZI"
        >>> expectation_from_counts(counts, pauli_string)
        1.0

    Returns:
        Expectation value.

    """
    counts = remove_whitespace_from_bitstrings(counts)
    total_counts = sum(counts.values())
    num_qubits = len(next(iter(counts.keys())))
    expectation = 0.0

    if pauli_string is None:
        pauli_string = "Z" * num_qubits
    elif len(pauli_string) != num_qubits:
        raise ValueError("Length of 'pauli_string' must match number of qubits in counts.")

    if not re.fullmatch(r"[ZI]+", pauli_string):
        raise ValueError(f"'pauli_string' '{pauli_string}' contains characters other than 'I' and 'Z'.")

    for bitstring, count in counts.items():
        parity = 1
        for qubit_index, pauli in enumerate(pauli_string):
            if pauli == "Z":
                parity *= (-1) ** int(bitstring[qubit_index])

        expectation += parity * count
    return expectation / total_counts


def marginalize_counts(counts: dict[str, int], qubit_indices: list[int]) -> dict[str, int]:
    """Marginalize counts with respect to a subset of qubits.

    Args:
        counts: A dictionary of counts with bitstrings as keys.
        qubit_indices: A list of qubit indices to keep.

    .. note::

        The qubit indexes relates to the bitstring in little-endian format
        (consistent with ``qiskit.result.marginal_counts``).

    Example:
        >>> counts = {"000": 10, "100": 20, "110": 10}
        >>> qubit_indices = [2]
        >>> marginalize_counts(counts, qubit_indices)
        {'0': 10, '1': 30}

    Returns:
        A dictionary of marginalized counts.

    Raises:
        ValueError: If qubit indices are outside of valid range.

    """
    counts = remove_whitespace_from_bitstrings(counts)
    num_qubits = len(next(iter(counts.keys())))
    if any(not isinstance(i, int) for i in qubit_indices):
        raise ValueError("Qubit indices must be integers.")
    if any(i < 0 or i >= num_qubits for i in qubit_indices):
        raise ValueError("Qubit indices must be within the valid range.")

    marginalized_counts: Counter[str] = Counter()
    for bitstring, count in counts.items():
        # Extract the bits corresponding to the specified qubit indices
        marginalized_bitstring = "".join(bitstring[-i - 1] for i in sorted(qubit_indices, reverse=True))
        marginalized_counts[marginalized_bitstring] += count
    return dict(marginalized_counts)


def total_variational_distance(dist_a: dict[str, int | float], dist_b: dict[str, int | float]) -> float:
    r"""Total Variation Distance between two probability distributions.

    The Total Variation Distance between the probability distributions P and Q is defined as

    .. math:: \text{TVD}(P, Q) = \frac{1}{2} \sum_x |P(x) - Q(x)|.

    TVD is always in :math:`[0, 1]`.

    This function accepts both:

    - Proper probability distributions (``dict[str, float]`` with values summing to 1)
    - Samplings of distributions (``dict[str, int]`` with arbitrary sums)

    The distributions are automatically normalized before computing the distance,
    so the input values do not actually need to sum to 1.

    Args:
        dist_a: First probability distribution.
        dist_b: Second probability distribution.

    Returns:
        Total Variation Distance between the two distributions.

    """
    dist_a = remove_whitespace_from_bitstrings(dist_a)
    dist_b = remove_whitespace_from_bitstrings(dist_b)
    total_a = np.sum(list(dist_a.values()))
    total_b = np.sum(list(dist_b.values()))
    all_keys = set(dist_a.keys()).union(set(dist_b.keys()))

    distance = 0
    for key in all_keys:
        p_a = dist_a.get(key, 0) / total_a
        p_b = dist_b.get(key, 0) / total_b
        distance += abs(p_a - p_b)

    return distance / 2
