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

"""Tests for CDR Clifford / native prx(theta, phi) gate utilities."""

import warnings

from iqm.error_reduction_tools.cdr.cdr_clifford_utils import (
    _warn_if_identity_delay_unused,
    append_stim_sequence,
    build_stim_single_qubit_clifford_table,
    get_clifford_r_library,
    is_logically_single_qubit_clifford_r,
    is_signed_permutation_matrix,
    normalize_angle_2pi,
    r_gate_so3_matrix,
    signed_permutation_key_from_so3,
    so3_h_matrix,
    stim_sequence_for_r_gate,
)
import numpy as np
import pytest
import stim

ATOL = 1e-10


class TestNormalizeAngle2Pi:
    """Test the normalize_angle_2pi function."""

    @pytest.mark.parametrize(
        "angle,expected",
        [
            (0.0, 0.0),
            (2.0 * np.pi, 0.0),
            (-np.pi / 2, 3 * np.pi / 2),
            (np.pi, np.pi),
            (5 * np.pi, np.pi),
        ],
    )
    def test_normalize(self, angle, expected):
        """Test that normalize_angle_2pi maps angles into [0, 2*pi)."""
        assert np.isclose(normalize_angle_2pi(angle), expected, atol=ATOL)


class TestCliffordRLibrary:
    """Tests for the restricted single-R-gate Clifford library."""

    def test_default_library_size_and_keys(self):
        """Test that the default Clifford R library has 7 entries with the expected keys."""
        library = get_clifford_r_library()
        assert len(library) == 7
        for entry in library:
            assert set(entry.keys()) == {"label", "theta", "phi", "is_delay"}

    def test_delay_flag_toggles(self):
        """Test that identity_as_delay toggles only the identity entry's is_delay flag."""
        with_delay = get_clifford_r_library(identity_as_delay=True)
        without_delay = get_clifford_r_library(identity_as_delay=False)
        assert with_delay[0]["is_delay"] is True
        assert without_delay[0]["is_delay"] is False
        assert with_delay[1:] == without_delay[1:]


class TestWarnIfIdentityDelayUnused:
    """Tests for :func:`_warn_if_identity_delay_unused`."""

    def test_warns_when_identity_as_delay_false_and_delay_is_non_default(self):
        """Test that a non-default identity_delay warns when identity_as_delay is False."""
        with pytest.warns(UserWarning, match="identity_delay has no effect"):
            _warn_if_identity_delay_unused(identity_as_delay=False, identity_delay=50e-9)

    def test_no_warning_when_identity_as_delay_is_true(self):
        """Test that no warning fires when identity_as_delay is True, regardless of identity_delay."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            _warn_if_identity_delay_unused(identity_as_delay=True, identity_delay=50e-9)

    def test_no_warning_when_identity_delay_is_left_at_default(self):
        """Test that no warning fires when identity_delay is left at its default value."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            _warn_if_identity_delay_unused(identity_as_delay=False, identity_delay=20e-9)


class TestIsLogicallyCliffordR:
    """Tests for the SO(3)-based Clifford verification."""

    @pytest.mark.parametrize(
        "theta,phi",
        [
            (0.0, 0.0),
            (np.pi, 0.0),
            (np.pi / 2, 0.0),
            (np.pi / 2, np.pi / 2),
        ],
    )
    def test_clifford_r_gates(self, theta, phi):
        """Test that known Clifford prx(theta, phi) gates are recognized as Clifford."""
        assert is_logically_single_qubit_clifford_r(theta, phi)

    @pytest.mark.parametrize(
        "theta,phi",
        [
            (0.3, 0.0),
            (np.pi / 3, np.pi / 5),
        ],
    )
    def test_non_clifford_r_gates(self, theta, phi):
        """Test that known non-Clifford prx(theta, phi) gates are rejected as non-Clifford."""
        assert not is_logically_single_qubit_clifford_r(theta, phi)

    def test_so3_matrix_of_identity(self):
        """Test that r_gate_so3_matrix returns the identity rotation for R(0, 0)."""
        so3 = r_gate_so3_matrix(0.0, 0.0)
        assert so3.shape == (3, 3)
        assert so3.dtype == np.float64
        assert np.allclose(so3, np.eye(3), atol=ATOL)

    @pytest.mark.parametrize(
        "theta,phi,expected",
        [
            (np.pi / 2, 0.0, np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])),
            (np.pi, np.pi / 2, np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])),
        ],
    )
    def test_so3_matrix_known_rotations(self, theta, phi, expected):
        """Test that r_gate_so3_matrix returns the exact rotation matrix for known R gates."""
        so3 = r_gate_so3_matrix(theta, phi)
        assert so3.shape == (3, 3)
        assert np.allclose(so3, expected, atol=ATOL)

    def test_signed_permutation_of_identity(self):
        """Test that the identity matrix is recognized as a signed permutation matrix."""
        assert is_signed_permutation_matrix(np.eye(3))

    def test_signed_permutation_rejects_generic_rotation(self):
        """Test that a generic (non-Clifford) rotation is rejected as a signed permutation matrix."""
        assert not is_signed_permutation_matrix(r_gate_so3_matrix(0.3, 0.1))

    def test_signed_permutation_rejects_reflection(self):
        """Test that a determinant -1 reflection is rejected as a signed permutation matrix."""
        assert not is_signed_permutation_matrix(np.diag([1.0, 1.0, -1.0]))


class TestSignedPermutationKeyFromSo3:
    """Test for converting SO(3) Clifford matrices into hashable keys."""

    def test_identity_key(self):
        """Test that the identity matrix maps to the expected flattened key."""
        key = signed_permutation_key_from_so3(np.eye(3))
        assert key == (1, 0, 0, 0, 1, 0, 0, 0, 1)

    def test_key_of_a_genuine_non_identity_permutation(self):
        """Test that a genuine non-identity signed permutation matrix produces the matching flattened key."""
        key = signed_permutation_key_from_so3(so3_h_matrix())
        assert key == (0, 0, 1, 0, -1, 0, 1, 0, 0)

    def test_raises_on_non_permutation_matrix(self):
        """Test that a generic rotation matrix is rejected as not a signed permutation."""
        with pytest.raises(ValueError, match="not close to a signed permutation matrix"):
            signed_permutation_key_from_so3(r_gate_so3_matrix(0.3, 0.5))

    @pytest.mark.parametrize(
        ("matrix", "message"),
        [
            ([[1, 1, 0], [0, 0, 0], [0, 0, 1]], "row condition failed"),
            ([[1, 0, 0], [1, 0, 0], [0, 0, 1]], "column condition failed"),
        ],
    )
    def test_raises_when_a_row_or_column_has_not_exactly_one_entry(self, matrix, message):
        """Test that a matrix of -1, 0, 1 entries without one nonzero per row and column is rejected."""
        with pytest.raises(ValueError, match=message):
            signed_permutation_key_from_so3(np.array(matrix, dtype=float))

    def test_raises_on_reflection(self):
        """Test that a determinant -1 reflection is rejected due to its determinant."""
        with pytest.raises(ValueError, match=r"determinant is not \+1"):
            signed_permutation_key_from_so3(np.diag([1.0, 1.0, -1.0]))


class TestStimTable:
    """Test the Qiskit->Stim single-qubit Clifford translation table."""

    def test_table_has_24_cliffords(self):
        """Test that the built Stim Clifford table has all 24 single-qubit Cliffords."""
        table = build_stim_single_qubit_clifford_table()
        assert len(table) == 24

    def test_table_values_are_lists_of_stim_gate_names(self):
        """Test that every table entry is a list of "H"/"S" Stim gate names."""
        table = build_stim_single_qubit_clifford_table()
        for sequence in table.values():
            assert isinstance(sequence, list)
            assert all(isinstance(gate, str) and gate in {"H", "S"} for gate in sequence)

    def test_stim_sequence_for_identity_is_empty(self):
        """Test that the Stim gate sequence for the identity R gate is empty."""
        assert stim_sequence_for_r_gate(0.0, 0.0) == []

    def test_stim_sequence_for_non_clifford_raises(self):
        """Test that requesting a Stim sequence for a non-Clifford R gate raises a ValueError."""
        with pytest.raises(ValueError):
            stim_sequence_for_r_gate(0.3, 0.0)

    @pytest.mark.parametrize(
        "theta,phi",
        [
            (np.pi / 2, 0.0),  # RX(pi/2)
            (np.pi, np.pi / 2),  # RY(pi)
            (np.pi / 2, np.pi / 2),  # RY(pi/2)
        ],
    )
    def test_stim_sequence_reproduces_the_r_gate_bloch_rotation(self, theta, phi):
        """Test that the Stim gate sequence for a Clifford prx gate reproduces its Bloch-vector action on |0>."""
        sequence = stim_sequence_for_r_gate(theta, phi)

        # prx(theta, phi)|0>
        psi = np.array([np.cos(theta / 2), -1j * np.exp(1j * phi) * np.sin(theta / 2)])
        pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
        pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
        expected = [
            float(np.real(psi.conj() @ pauli_x @ psi)),
            float(np.real(psi.conj() @ pauli_y @ psi)),
            float(np.real(psi.conj() @ pauli_z @ psi)),
        ]

        circuit = stim.Circuit()
        append_stim_sequence(circuit, sequence, 0)
        sim = stim.TableauSimulator()
        sim.do_circuit(circuit)
        actual = [
            sim.peek_observable_expectation(stim.PauliString("X")),
            sim.peek_observable_expectation(stim.PauliString("Y")),
            sim.peek_observable_expectation(stim.PauliString("Z")),
        ]

        assert np.allclose(actual, expected, atol=ATOL)


class TestAppendStimSequence:
    """Test appending a Stim gate sequence to a circuit."""

    def test_appends_each_gate_in_order_to_the_target_qubit(self):
        """Test that append_stim_sequence appends each gate in order to the given qubit."""
        circuit = stim.Circuit()
        append_stim_sequence(circuit, ["H", "S", "H"], 2)
        assert [instruction.name for instruction in circuit] == ["H", "S", "H"]
        assert all(instruction.targets_copy()[0].value == 2 for instruction in circuit)

    def test_empty_sequence_appends_nothing(self):
        """Test that an empty sequence appends no instructions."""
        circuit = stim.Circuit()
        append_stim_sequence(circuit, [], 0)
        assert len(circuit) == 0
