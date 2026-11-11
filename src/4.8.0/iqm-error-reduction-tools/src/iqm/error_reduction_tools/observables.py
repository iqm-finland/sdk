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

"""Diagonal Pauli observables, and their expectation values from measurement counts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Number
from typing import TypeAlias

import numpy as np
from qiskit.quantum_info import SparsePauliOp

ObservableLike: TypeAlias = str | SparsePauliOp | Sequence[tuple[float, str]]
"""A Pauli label such as ``"ZZI"``, a :class:`~qiskit.quantum_info.SparsePauliOp`, or ``[(coefficient, label)]``."""

_IMAG_ATOL = 1e-12
"""Largest imaginary part of a coefficient that is treated as zero."""

_MAX_CLBITS = 64
"""Widest register the bit-parity computation handles."""


@dataclass(frozen=True)
class ObservableSet:
    """Observables that are sums of Z-type Pauli terms, sharing one measurement of every classical bit.

    Labels are little-endian: the rightmost character acts on classical bit 0. The distinct non-identity terms of all
    observables form :attr:`pauli_terms`, which is the axis expectation values are computed and mitigated on.
    """

    observables: tuple[SparsePauliOp, ...]
    """The observables, in the order given."""

    pauli_terms: tuple[str, ...]
    """Distinct non-identity Pauli labels over all observables, in order of first appearance."""

    coefficients: np.ndarray
    """Real coefficient of each Pauli term in each observable, shape ``(num_observables, num_terms)``."""

    offsets: np.ndarray
    """Coefficient of the identity in each observable, shape ``(num_observables,)``."""

    @classmethod
    def build(cls, observables: ObservableLike | Sequence[ObservableLike]) -> ObservableSet:
        """Build the set from one observable or a sequence of observables.

        A string or a :class:`~qiskit.quantum_info.SparsePauliOp` is one observable. Any other sequence is a sequence
        of observables, so a single weighted sum must be wrapped in a list: ``[[(1.0, "ZZ"), (0.5, "ZI")]]``.

        Args:
            observables: The observables.

        Returns:
            The observable set.

        Raises:
            ValueError: If there are no observables, a label contains anything but ``I`` and ``Z``, labels differ in
                length, or a coefficient is not real.

        """
        items = [observables] if isinstance(observables, (str, SparsePauliOp)) else list(observables)
        if not items:
            raise ValueError("At least one observable is required.")
        operators = tuple(_to_sparse_pauli_op(item) for item in items)

        widths = {operator.num_qubits for operator in operators}
        if len(widths) != 1:
            raise ValueError(f"All observables must act on the same number of classical bits, got {sorted(widths)}.")
        identity = "I" * widths.pop()

        terms: list[str] = []
        for operator in operators:
            for label in operator.paulis.to_labels():
                if set(label) - {"I", "Z"}:
                    raise ValueError(f"Pauli term {label!r} is not diagonal. Only 'I' and 'Z' can be measured.")
                if label != identity and label not in terms:
                    terms.append(label)

        coefficients = np.zeros((len(operators), len(terms)), dtype=float)
        offsets = np.zeros(len(operators), dtype=float)
        for row, operator in enumerate(operators):
            for label, coefficient in zip(operator.paulis.to_labels(), operator.coeffs):
                # An observable estimated from counts must be Hermitian, i.e. every coefficient real.
                if abs(coefficient.imag) > _IMAG_ATOL:
                    raise ValueError(f"The coefficient of {label!r} is {coefficient}, but it must be real.")
                if label == identity:
                    offsets[row] += coefficient.real
                else:
                    coefficients[row, terms.index(label)] += coefficient.real
        return cls(observables=operators, pauli_terms=tuple(terms), coefficients=coefficients, offsets=offsets)

    @property
    def num_clbits(self) -> int:
        """Return the number of classical bits every label acts on."""
        return int(self.observables[0].num_qubits)

    @property
    def num_observables(self) -> int:
        """Return the number of observables."""
        return len(self.observables)

    @property
    def num_terms(self) -> int:
        """Return the number of distinct non-identity Pauli terms."""
        return len(self.pauli_terms)

    def combine(self, term_values: np.ndarray) -> np.ndarray:
        """Combine Pauli term expectation values into observable expectation values.

        Args:
            term_values: Values on the last axis, one per entry of :attr:`pauli_terms`, shape ``(..., num_terms)``.

        Returns:
            Shape ``(..., num_observables)``.

        """
        return np.asarray(term_values, dtype=float) @ self.coefficients.T + self.offsets

    def term_values_from_counts(self, counts: Mapping[str, float]) -> np.ndarray:
        """Compute the expectation value of every Pauli term from measurement counts.

        Args:
            counts: Bitstring to count. The rightmost character of a bitstring is classical bit 0.

        Returns:
            Shape ``(num_terms,)``, each value in ``[-1, 1]``.

        Raises:
            ValueError: If the counts are empty or a bitstring has the wrong length.

        """
        cleaned = {bitstring.replace(" ", ""): float(count) for bitstring, count in counts.items()}
        total = sum(cleaned.values())
        if total <= 0:
            raise ValueError("Cannot compute expectation values from counts with no shots.")
        if any(len(bitstring) != self.num_clbits for bitstring in cleaned):
            raise ValueError(f"Every bitstring must have {self.num_clbits} bits.")
        if self.num_clbits > _MAX_CLBITS:
            raise ValueError(f"At most {_MAX_CLBITS} classical bits are supported, got {self.num_clbits}.")
        outcomes = np.array([int(bitstring, 2) for bitstring in cleaned], dtype=np.uint64)
        weights = np.array(list(cleaned.values()), dtype=float)
        masks = np.array(
            [int(label.replace("I", "0").replace("Z", "1"), 2) for label in self.pauli_terms], dtype=np.uint64
        )
        signs = 1.0 - 2.0 * (np.bitwise_count(outcomes[:, np.newaxis] & masks[np.newaxis, :]) % 2)
        return (weights @ signs) / total


def _to_sparse_pauli_op(observable: ObservableLike) -> SparsePauliOp:
    """Convert one observable to a :class:`~qiskit.quantum_info.SparsePauliOp`, with repeated labels merged."""
    if isinstance(observable, SparsePauliOp):
        return observable.simplify()
    if isinstance(observable, str):
        return SparsePauliOp(observable)
    terms = list(observable)
    if _is_weighted_term(terms) and isinstance(terms[0], Number):
        raise ValueError(
            f"{tuple(terms)!r} is a single weighted term, not an observable. Write it as [{tuple(terms)!r}]."
        )
    if not terms or not all(_is_weighted_term(term) for term in terms):
        raise ValueError(f"An observable given as a sequence must be [(coefficient, label), ...], got {observable!r}.")
    # complex() so that a complex coefficient reaches the realness check in build() instead of failing here.
    operator = SparsePauliOp([label for _, label in terms], coeffs=[complex(coefficient) for coefficient, _ in terms])
    return operator.simplify()


def _is_weighted_term(item: object) -> bool:
    """Return whether ``item`` is a ``(coefficient, label)`` pair, given as a tuple or a list."""
    match item:
        case [_, str()]:
            return True
        case _:
            return False
