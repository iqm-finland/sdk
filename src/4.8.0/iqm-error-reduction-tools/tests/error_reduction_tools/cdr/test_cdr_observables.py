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

"""
Tests for the CDR counts-to-expectation utilities.
"""

from iqm.error_reduction_tools.cdr.cdr_observables import (
    diagonal_observable_to_terms,
    get_observable_num_qubits,
    pauli_label_to_z_qubits,
    to_pauli_terms,
)
import numpy as np
import pytest

ATOL = 1e-12


@pytest.mark.parametrize(
    "label,expected",
    [
        ("III", []),
        ("IIZ", [0]),
        ("ZII", [2]),
        ("ZIZ", [0, 2]),
    ],
)
def test_pauli_label_extracts_the_z_qubits(label, expected):
    """Test that pauli_label_to_z_qubits extracts the qubit indices where Z acts."""
    assert pauli_label_to_z_qubits(label) == expected


def test_pauli_label_rejects_x_or_y():
    """Test that pauli_label_to_z_qubits rejects a label containing X or Y."""
    with pytest.raises(ValueError, match="non-diagonal Pauli"):
        pauli_label_to_z_qubits("IIX")


def test_pauli_label_rejects_an_empty_label():
    """Test that pauli_label_to_z_qubits rejects an empty label."""
    with pytest.raises(ValueError, match="cannot be empty"):
        pauli_label_to_z_qubits("")


def test_pauli_label_rejects_an_unsupported_character():
    """Test that pauli_label_to_z_qubits rejects a label with an unsupported character."""
    with pytest.raises(ValueError, match="Unsupported character"):
        pauli_label_to_z_qubits("IIQ")


def test_to_pauli_terms_from_a_string():
    """Test that to_pauli_terms normalizes a bare Pauli string into a unit-coefficient term."""
    terms = to_pauli_terms("IIIZ")
    assert len(terms) == 1
    assert isinstance(terms[0][0], complex)
    assert terms == [(1.0 + 0.0j, "IIIZ")]


def test_to_pauli_terms_from_a_weighted_term():
    """Test that to_pauli_terms normalizes a [coeff, label] pair into a weighted term."""
    terms = to_pauli_terms([0.5, "IIIZ"])
    assert len(terms) == 1
    assert isinstance(terms[0][0], complex)
    assert terms == [(0.5 + 0.0j, "IIIZ")]


def test_to_pauli_terms_from_a_list_of_strings():
    """Test that to_pauli_terms normalizes a list of labels into unit-coefficient terms."""
    terms = to_pauli_terms(["IIZ", "ZII"])
    assert len(terms) == 2
    assert all(isinstance(coeff, complex) for coeff, _ in terms)
    assert terms == [(1.0 + 0.0j, "IIZ"), (1.0 + 0.0j, "ZII")]


def test_to_pauli_terms_from_a_list_of_weighted_terms():
    """Test that to_pauli_terms normalizes a list of [coeff, label] pairs into weighted terms."""
    terms = to_pauli_terms([[0.5, "IIZ"], [-0.5, "ZII"]])
    assert len(terms) == 2
    assert all(isinstance(coeff, complex) for coeff, _ in terms)
    assert terms == [(0.5 + 0.0j, "IIZ"), (-0.5 + 0.0j, "ZII")]


@pytest.mark.parametrize(
    "coeff,expected_complex",
    [
        (2, 2.0 + 0.0j),
        (np.float64(0.5), 0.5 + 0.0j),
        (1 + 2j, 1.0 + 2.0j),
    ],
)
def test_to_pauli_terms_accepts_various_numeric_coefficient_types(coeff, expected_complex):
    """Test that to_pauli_terms accepts int, NumPy scalar, and complex-literal coefficients."""
    terms = to_pauli_terms([coeff, "IIZ"])
    assert len(terms) == 1
    assert isinstance(terms[0][0], complex)
    assert terms == [(expected_complex, "IIZ")]


def test_to_pauli_terms_rejects_an_unrecognized_list():
    """Test that to_pauli_terms rejects a list that matches none of the supported formats."""
    with pytest.raises(ValueError, match="Unrecognized observable list format"):
        to_pauli_terms([1, 2, 3])


def test_to_pauli_terms_rejects_an_empty_observable():
    """Test that to_pauli_terms rejects an empty observable list."""
    with pytest.raises(ValueError, match="Empty observable"):
        to_pauli_terms([])


def test_to_pauli_terms_rejects_an_unsupported_type():
    """Test that to_pauli_terms rejects an observable of an unsupported type."""
    with pytest.raises(TypeError, match="Unsupported observable type"):
        to_pauli_terms(3.14)


def test_get_observable_num_qubits():
    """Test that get_observable_num_qubits returns the label length for single and weighted-sum observables."""
    assert get_observable_num_qubits("IIIZ") == 4
    assert get_observable_num_qubits([[0.5, "IZ"], [-0.5, "ZI"]]) == 2


def test_get_observable_num_qubits_rejects_inconsistent_lengths():
    """Test that get_observable_num_qubits rejects terms with inconsistent label lengths."""
    with pytest.raises(ValueError, match="inconsistent lengths"):
        get_observable_num_qubits([[0.5, "IIZ"], [-0.5, "ZI"]])


@pytest.mark.parametrize(
    "observable,expected",
    [
        ([[0.5, "IIZ"], [-0.5, "ZII"]], [(0.5 + 0.0j, [0]), (-0.5 + 0.0j, [2])]),
        ("ZIZ", [(1.0 + 0.0j, [0, 2])]),
        ([[2.0, "III"]], [(2.0 + 0.0j, [])]),
    ],
)
def test_diagonal_observable_to_terms_returns_coeff_and_z_qubits(observable, expected):
    """Test that diagonal_observable_to_terms returns the exact coefficient and z_qubits pairs."""
    terms = diagonal_observable_to_terms(observable)
    assert len(terms) == len(expected)
    assert all(isinstance(coeff, complex) for coeff, _ in terms)
    assert terms == expected
