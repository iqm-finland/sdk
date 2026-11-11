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

"""Parsing of diagonal Pauli observables, and their expectation values from counts."""

from __future__ import annotations

from typing import Any

import numpy as np


def pauli_label_to_z_qubits(label: str) -> list[int]:
    """Convert a diagonal Pauli label to the qubit indices where Z acts.

    Args:
        label: Pauli label, e.g. ``"IZZ"``.

    Returns:
        Qubit indices on which Z acts.

    Raises:
        ValueError: If the label is empty, contains X or Y, or contains
            unsupported characters.

    """
    if len(label) == 0:
        raise ValueError("Pauli label cannot be empty.")

    z_qubits = []
    for q, char in enumerate(reversed(label)):
        if char == "I":
            continue

        if char == "Z":
            z_qubits.append(q)
            continue

        if char in {"X", "Y"}:
            raise ValueError(
                f"Observable term {label!r} contains the non-diagonal Pauli {char!r}. Only I and Z are supported."
            )

        raise ValueError(f"Unsupported character {char!r} in label {label!r}. Allowed characters are 'I' and 'Z'.")

    return z_qubits


def _is_number(x: Any) -> bool:
    """Check whether a value is a Python or NumPy real or complex scalar."""
    return isinstance(x, (int, float, complex, np.integer, np.floating, np.complexfloating))


def to_pauli_terms(observable: Any) -> list:
    """Convert an observable into a list of (coefficient, label) terms.

    Args:
        observable: ``"IIIZ"``, ``[0.5, "IIIZ"]``, ``["IIIZ", "IIZI"]`` or ``[[0.5, "IIIZ"], [-1.0, "IIZI"]]``.

    Returns:
        The (coefficient, label) terms.

    Raises:
        TypeError: If the observable type is unsupported.
        ValueError: If the observable is empty or its format is unrecognized.

    """
    if isinstance(observable, str):
        return [(1.0 + 0.0j, observable)]

    if isinstance(observable, (list, tuple)):
        seq = list(observable)
        if len(seq) == 0:
            raise ValueError("Empty observable.")
        # [coef, "label"]: one weighted term.
        if len(seq) == 2 and _is_number(seq[0]) and isinstance(seq[1], str):  # noqa: PLR2004
            return [(complex(seq[0]), seq[1])]
        # ["label", "label", ...]: sum with unit coefficients.
        if all(isinstance(x, str) for x in seq):
            return [(1.0 + 0.0j, x) for x in seq]
        # [[coef, "label"], ...]: weighted sum.
        if all(
            isinstance(x, (list, tuple)) and len(x) == 2 and _is_number(x[0]) and isinstance(x[1], str)  # noqa: PLR2004
            for x in seq
        ):
            return [(complex(x[0]), x[1]) for x in seq]
        raise ValueError(
            "Unrecognized observable list format. Use 'IIIZ', [coef, 'IIIZ'], ['IIIZ', 'IIZI', ...] or "
            "[[coef, 'IIIZ'], ...]."
        )
    raise TypeError(f"Unsupported observable type: {type(observable)!r}.")


def diagonal_observable_to_terms(observable: Any) -> list[tuple[complex, list[int]]]:
    """Convert a diagonal observable into weighted Z-product terms.

    Labels are little-endian: the rightmost character acts on qubit 0.

    Args:
        observable: Diagonal observable in a format accepted by :func:`to_pauli_terms`.

    Returns:
        The (coefficient, z_qubits) terms, where z_qubits are the indices of the qubits Z acts on.

    Raises:
        TypeError: If the observable type is unsupported.
        ValueError: If a label is empty, contains X or Y, or contains other
            unsupported characters.

    """
    terms = []
    for coeff, label in to_pauli_terms(observable):
        z_qubits = pauli_label_to_z_qubits(label)
        terms.append((complex(coeff), z_qubits))
    return terms


def get_observable_num_qubits(observable: Any) -> int:
    """Return the length of the Pauli labels of the observable.

    Args:
        observable: Observable in a format accepted by :func:`to_pauli_terms`.

    Returns:
        The length of the Pauli labels.

    Raises:
        TypeError: If the observable type is unsupported.
        ValueError: If the observable is empty, its format is unrecognized, or
            its terms have inconsistent label lengths.

    """
    lengths = {len(label) for _, label in to_pauli_terms(observable)}
    if len(lengths) > 1:
        raise ValueError(f"Observable terms have inconsistent lengths: {sorted(lengths)}.")
    return lengths.pop()
