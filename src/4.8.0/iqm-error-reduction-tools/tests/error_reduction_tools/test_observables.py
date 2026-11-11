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

from iqm.error_reduction_tools.observables import ObservableSet
import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp


class TestBuild:
    def test_repeated_terms_share_one_column(self):
        """Test that a Pauli term shared by several observables gets one column, and identities go to the offsets."""
        observables = ObservableSet.build(["ZZ", [(1.0, "ZI"), (2.0, "ZZ")], [(0.5, "ZI"), (1.5, "II")]])
        assert observables.pauli_terms == ("ZZ", "ZI")
        assert observables.coefficients.shape == (3, 2)
        assert observables.coefficients.dtype == np.float64
        np.testing.assert_array_equal(observables.coefficients, [[1.0, 0.0], [2.0, 1.0], [0.0, 0.5]])
        np.testing.assert_array_equal(observables.offsets, [0.0, 0.0, 1.5])

    @pytest.mark.parametrize(
        "single",
        ["ZIZ", SparsePauliOp(["ZIZ"])],
    )
    def test_string_and_sparse_pauli_op_are_one_observable(self, single):
        """Test that a label or a SparsePauliOp is read as one observable."""
        observables = ObservableSet.build(single)
        assert observables.num_observables == 1
        assert observables.pauli_terms == ("ZIZ",)
        assert observables.num_clbits == 3

    @pytest.mark.parametrize(
        "cancelling",
        [SparsePauliOp(["ZZ", "ZZ", "IZ"], coeffs=[1j, -1j, 2.0]), [(1j, "ZZ"), (-1j, "ZZ"), (2.0, "IZ")]],
    )
    def test_imaginary_parts_that_cancel_are_accepted(self, cancelling):
        """Test that imaginary coefficients that cancel out are accepted."""
        observables = ObservableSet.build([cancelling])
        assert observables.pauli_terms == ("IZ",)
        np.testing.assert_array_equal(observables.coefficients, [[2.0]])

    @pytest.mark.parametrize("imaginary", [[SparsePauliOp(["ZZ"], coeffs=[0.5 + 1j])], [[(0.5j, "ZZ")]]])
    def test_imaginary_coefficients_are_rejected_in_every_form(self, imaginary):
        """Test that an imaginary coefficient is rejected in every input form."""
        with pytest.raises(ValueError, match="must be real"):
            ObservableSet.build(imaginary)

    def test_bare_weighted_term_is_rejected(self):
        """Test that a (coefficient, label) pair not wrapped in a list is rejected."""
        with pytest.raises(ValueError, match="single weighted term"):
            ObservableSet.build([(1.0, "ZZ")])

    @pytest.mark.parametrize(
        ("observables", "message"),
        [
            (["XZ"], "not diagonal"),
            (["ZZ", "ZZZ"], "same number of classical bits"),
            ([], "At least one"),
            ([["ZZ"]], r"must be \[\(coefficient, label\), \.\.\.\]"),
        ],
    )
    def test_invalid_observables_are_rejected(self, observables, message):
        """Test that non-diagonal, mixed-width, empty and malformed observables are rejected."""
        with pytest.raises(ValueError, match=message):
            ObservableSet.build(observables)


class TestValues:
    @pytest.mark.parametrize(
        ("counts", "expected"),
        [
            # Rightmost character is clbit 0: "01" means clbit 0 is 1.
            ({"00": 1}, [1.0, 1.0, 1.0]),
            ({"01": 3, "10": 1}, [-1.0, -0.5, 0.5]),
            ({"00": 1, "11": 1}, [1.0, 0.0, 0.0]),
        ],
    )
    def test_parities_on_known_distributions(self, counts, expected):
        """Test the Pauli term values of distributions with known parities."""
        observables = ObservableSet.build(["ZZ", "IZ", "ZI"])
        values = observables.term_values_from_counts(counts)
        assert values.shape == (3,)
        assert values.dtype == np.float64
        np.testing.assert_allclose(values, expected)

    def test_combine_adds_the_identity_offset(self):
        """Test that combine() weights the term values and adds the identity offset."""
        observables = ObservableSet.build([[(2.0, "ZZ"), (-1.0, "IZ"), (0.25, "II")], "IZ"])
        combined = observables.combine(np.array([[0.5, 1.0], [-1.0, 0.0]]))
        assert combined.shape == (2, 2)
        np.testing.assert_allclose(combined, [[0.25, 1.0], [-1.75, 0.0]])

    def test_spaces_between_registers_are_ignored(self):
        """Test that spaces in a bitstring are ignored."""
        observables = ObservableSet.build("ZZ")
        np.testing.assert_allclose(observables.term_values_from_counts({"0 1": 1}), [-1.0])

    def test_wrong_width_and_empty_counts_are_rejected(self):
        """Test that bitstrings of the wrong width and counts with no shots are rejected."""
        observables = ObservableSet.build("ZZ")
        with pytest.raises(ValueError, match="2 bits"):
            observables.term_values_from_counts({"000": 1})
        with pytest.raises(ValueError, match="no shots"):
            observables.term_values_from_counts({"00": 0})

    def test_registers_wider_than_64_bits_are_rejected(self):
        """Test that more than 64 classical bits are rejected."""
        observables = ObservableSet.build("Z" * 65)
        with pytest.raises(ValueError, match="At most 64 classical bits"):
            observables.term_values_from_counts({"0" * 65: 1})
