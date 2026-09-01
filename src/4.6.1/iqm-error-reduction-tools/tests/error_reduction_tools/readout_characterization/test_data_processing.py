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

"""Pytest tests for data_processing.py module."""

from iqm.error_reduction_tools.readout_characterization.data_processing import (
    ERROR_TO_PREP,
    _compute_prep_counts,
    _compute_prep_joint_counts,
    _get_error_ids_for_shot,
    compute_double_twirled_covariance,
    compute_error_probabilities,
    compute_single_twirled_covariance,
    compute_state_dependent_covariance,
)
import numpy as np
import pytest

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_test_data():
    """Simple test data with 2 qubits, with different error rates for each qubit."""
    # Qubit 0 (rightmost, little-endian): 5/9 errors (0->1), 3/10 errors (1->0)
    # Qubit 1 (leftmost, little-endian): 4/10 errors (0->1), 2/9 errors (1->0)
    return {
        "counts_by_prep": {
            # Prep: II (Q0=0, Q1=0):
            #   Q0: 0 1 0 1 0   (2 errors 0-> 1)
            #   Q1: 0 0 0 0 1   (1 error 0-> 1)
            "II": {"00": 2, "01": 2, "10": 1},
            # Prep: IX (Q0=0, Q1=1):
            #   Q0: 1 1 0 1   (3 errors 0-> 1)
            #   Q1: 0 0 0 1   (1 error 1-> 0)
            "IX": {"11": 2, "10": 1, "01": 1},
            # Prep: XI (Q0=1, Q1=0):
            #   Q0: 1 0 0 1 0   (2 errors 1 -> 0)
            #   Q1: 1 0 1 0 1   (3 errors 0 -> 1)
            "XI": {"10": 1, "01": 1, "11": 2, "00": 1},
            # Prep: XX (Q0=1, Q1=1):
            #   Q0: 0 0 0 1 0   (1 error 1 -> 0)
            #   Q1: 0 0 1 0 0   (1 error 1 -> 0)
            "XX": {"11": 3, "01": 1, "10": 1},
        },
        "measured_qubits": ["QB1", "QB2"],
    }


@pytest.fixture
def perfect_data():
    """Test data with no errors.

    Bit ordering: prep_str is big-endian (prep_str[0] = first qubit),
    bitstring is little-endian (bitstring[-1] = first qubit).

    So for prep_str="IX" (Q0 prepared |0>, Q1 prepared |1>),
    the correct measurement is "10" (bit0=0, bit1=1 in little endian).
    """
    return {
        "counts_by_prep": {
            "II": {"00": 100},  # Q0=|0>, Q1=|0> -> measure "00"
            "IX": {"10": 100},  # Q0=|0>, Q1=|1> -> measure "10" (little-endian: bit0=0, bit1=1)
            "XI": {"01": 100},  # Q0=|1>, Q1=|0> -> measure "01" (little-endian: bit0=1, bit1=0)
            "XX": {"11": 100},  # Q0=|1>, Q1=|1> -> measure "11"
        },
        "measured_qubits": ["QB1", "QB2"],
    }


@pytest.fixture
def large_test_data():
    """Larger test data for robust testing."""
    rgen = np.random.default_rng(42)
    num_qubits = 3
    shots_per_prep = 100
    counts_by_prep = {}

    # Generate all possible prep strings
    for i in range(2**num_qubits):
        prep_str = "".join(["X" if (i >> j) & 1 else "I" for j in range(num_qubits)])
        counts_dict = {}
        for _ in range(shots_per_prep):
            # Generate mostly correct measurements with ~5% error rate
            bitstring = ""
            for j, p in enumerate(prep_str):
                expected = "1" if p == "X" else "0"
                if rgen.random() < 0.95:
                    bitstring = expected + bitstring
                else:
                    bitstring = ("0" if expected == "1" else "1") + bitstring
            # Add to counts
            if bitstring in counts_dict:
                counts_dict[bitstring] += 1
            else:
                counts_dict[bitstring] = 1
        counts_by_prep[prep_str] = counts_dict

    return {
        "counts_by_prep": counts_by_prep,
        "measured_qubits": [f"QB{i + 1}" for i in range(num_qubits)],
    }


# =============================================================================
# Tests for _get_error_ids_for_shot
# =============================================================================


class TestGetErrorIdsForShot:
    """Tests for _get_error_ids_for_shot function."""

    def test_no_error_prep_0_meas_0(self):
        """Test error ID when prepared |0>, measured |0> -> 0."""
        error_ids = _get_error_ids_for_shot("00", "II")
        assert error_ids == [0, 0]

    def test_no_error_prep_1_meas_1(self):
        """Test error ID when prepared |1>, measured |1> -> 1."""
        error_ids = _get_error_ids_for_shot("11", "XX")
        assert error_ids == [1, 1]

    def test_error_prep_0_meas_1(self):
        """Test error ID when prepared |0>, measured |1> -> 2 (bit flip)."""
        error_ids = _get_error_ids_for_shot("11", "II")
        assert error_ids == [2, 2]

    def test_error_prep_1_meas_0(self):
        """Test error ID when prepared |1>, measured |0> -> 3 (bit flip)."""
        error_ids = _get_error_ids_for_shot("00", "XX")
        assert error_ids == [3, 3]

    def test_three_qubits(self):
        """Test with three qubits."""
        error_ids = _get_error_ids_for_shot("010", "III")
        assert error_ids == [0, 2, 0]

        error_ids = _get_error_ids_for_shot("110", "XXI")
        assert error_ids == [3, 1, 2]


# =============================================================================
# Tests for compute_error_probabilities
# =============================================================================


class TestComputeErrorProbabilities:
    """Tests for compute_error_probabilities function."""

    def test_returns_correct_structure(self, simple_test_data):
        """Test that function returns a dict with the expected keys."""
        result = compute_error_probabilities(simple_test_data)

        assert isinstance(result, dict)
        assert "charact_data" in result
        assert "charact_data_std" in result
        assert set(result["charact_data"].keys()) == {"QB1", "QB2"}
        assert set(result["charact_data_std"].keys()) == {"QB1", "QB2"}

    def test_matrices_are_2x2(self, simple_test_data):
        """Test that each qubit has a 2x2 assignment matrix."""
        result = compute_error_probabilities(simple_test_data)
        for qubit in result["charact_data"]:
            assert result["charact_data"][qubit].shape == (2, 2)
            assert result["charact_data_std"][qubit].shape == (2, 2)

    def test_matrices_are_column_stochastic(self, simple_test_data):
        """Test that assignment matrices are column-stochastic (columns sum to 1)."""
        result = compute_error_probabilities(simple_test_data)
        for qubit in result["charact_data"]:
            matrix = result["charact_data"][qubit]
            np.testing.assert_allclose(matrix[:, 0].sum(), 1.0, atol=1e-10)
            np.testing.assert_allclose(matrix[:, 1].sum(), 1.0, atol=1e-10)

    def test_perfect_data_gives_identity(self, perfect_data):
        """Test that perfect data gives identity assignment matrices."""
        result = compute_error_probabilities(perfect_data)
        for qubit in result["charact_data"]:
            np.testing.assert_allclose(result["charact_data"][qubit], np.eye(2), atol=1e-8)
            np.testing.assert_allclose(result["charact_data_std"][qubit], np.zeros((2, 2)), atol=1e-8)

    def test_simple_data_exact_probabilities(self, simple_test_data):
        """Test that error probabilities are exactly as expected for simple_test_data, with little-endian convention."""
        result = compute_error_probabilities(simple_test_data)
        charact = result["charact_data"]
        charact_std = result["charact_data_std"]

        # QB1: p_0to1 = 5/9, p_1to0 = 3/10
        np.testing.assert_allclose(charact["QB1"][1, 0], 5 / 9, atol=1e-8)  # P(1|0)
        np.testing.assert_allclose(charact["QB1"][0, 1], 3 / 10, atol=1e-8)  # P(0|1)
        np.testing.assert_allclose(charact["QB1"][0, 0], 1 - 5 / 9, atol=1e-8)  # P(0|0)
        np.testing.assert_allclose(charact["QB1"][1, 1], 1 - 3 / 10, atol=1e-8)  # P(1|1)

        # QB2: p_0to1 = 4/10, p_1to0 = 2/9
        np.testing.assert_allclose(charact["QB2"][1, 0], 4 / 10, atol=1e-8)
        np.testing.assert_allclose(charact["QB2"][0, 1], 2 / 9, atol=1e-8)

        # Standard deviations
        np.testing.assert_allclose(charact_std["QB1"][1, 0], np.sqrt(5 / 9 * (1 - 5 / 9) / 9), atol=1e-8)
        np.testing.assert_allclose(charact_std["QB1"][0, 1], np.sqrt(3 / 10 * (1 - 3 / 10) / 10), atol=1e-8)
        np.testing.assert_allclose(charact_std["QB2"][1, 0], np.sqrt(4 / 10 * (1 - 4 / 10) / 10), atol=1e-8)
        np.testing.assert_allclose(charact_std["QB2"][0, 1], np.sqrt(2 / 9 * (1 - 2 / 9) / 9), atol=1e-8)

    def test_only_measured_qubits_in_output(self, simple_test_data):
        """Test that only measured qubits appear in the output dictionary."""
        result = compute_error_probabilities(simple_test_data)
        assert list(result["charact_data"].keys()) == simple_test_data["measured_qubits"]
        assert list(result["charact_data_std"].keys()) == simple_test_data["measured_qubits"]


# =============================================================================
# Tests for _compute_prep_counts
# =============================================================================


class TestComputePrepCounts:
    """Tests for _compute_prep_counts function."""

    def test_correct_shape(self, simple_test_data):
        """Test that prep counts have correct shape."""
        prep_counts = _compute_prep_counts(
            simple_test_data["counts_by_prep"],
            num_qubits=2,
        )

        assert prep_counts.shape == (2, 2)

    def test_counts_sum_correctly(self, simple_test_data):
        """Test that counts sum to total shots."""
        prep_counts = _compute_prep_counts(
            simple_test_data["counts_by_prep"],
            num_qubits=2,
        )

        total_shots = sum(sum(counts_dict.values()) for counts_dict in simple_test_data["counts_by_prep"].values())
        # Each qubit has prep_counts[0] + prep_counts[1] = total_shots
        for q in range(2):
            assert prep_counts[0, q] + prep_counts[1, q] == total_shots

    def test_specific_counts(self, simple_test_data):
        """Test that specific counts match expected values."""
        prep_counts = _compute_prep_counts(
            simple_test_data["counts_by_prep"],
            num_qubits=2,
        )

        assert prep_counts[0, 0] == 9
        assert prep_counts[1, 0] == 10

        assert prep_counts[0, 1] == 10
        assert prep_counts[1, 1] == 9


# =============================================================================
# Tests for _compute_prep_joint_counts
# =============================================================================


class TestComputePrepJointCounts:
    """Tests for _compute_prep_joint_counts function."""

    def test_correct_shape(self, simple_test_data):
        """Test that joint prep counts have correct shape."""
        num_qubits = 2
        joint_counts = _compute_prep_joint_counts(
            simple_test_data["counts_by_prep"],
            num_qubits=num_qubits,
        )

        assert joint_counts.shape == (2, 2, num_qubits, num_qubits)

    def test_counts_sum_correctly(self, simple_test_data):
        """Test that joint counts sum to total shots."""
        num_qubits = 2
        joint_counts = _compute_prep_joint_counts(
            simple_test_data["counts_by_prep"],
            num_qubits=num_qubits,
        )

        total_shots = sum(sum(counts_dict.values()) for counts_dict in simple_test_data["counts_by_prep"].values())
        assert np.all(
            np.sum(joint_counts[:, :, i, j]) for i in range(num_qubits) for j in range(num_qubits) == total_shots
        )

    def test_specific_joint_counts(self, simple_test_data):
        """Test that specific joint counts match expected values."""
        num_qubits = 2
        joint_counts = _compute_prep_joint_counts(
            simple_test_data["counts_by_prep"],
            num_qubits=num_qubits,
        )

        # For Qubit 0 (rightmost, little-endian) and Qubit 1 (leftmost, little-endian):
        # Prep II:
        assert joint_counts[0, 0, 0, 0] == 0
        assert joint_counts[0, 0, 0, 1] == 5
        assert joint_counts[0, 0, 1, 0] == 5
        assert joint_counts[0, 0, 1, 1] == 0

        # Prep IX:
        assert joint_counts[0, 1, 0, 0] == 0
        assert joint_counts[0, 1, 0, 1] == 4
        assert joint_counts[0, 1, 1, 1] == 0
        assert joint_counts[0, 1, 1, 0] == 5

        # Prep XI:
        assert joint_counts[1, 0, 0, 0] == 0
        assert joint_counts[1, 0, 1, 0] == 4
        assert joint_counts[1, 0, 0, 1] == 5
        assert joint_counts[1, 0, 1, 1] == 0

        # Prep XX:
        assert joint_counts[1, 1, 0, 0] == 0
        assert joint_counts[1, 1, 1, 0] == 5
        assert joint_counts[1, 1, 1, 1] == 0
        assert joint_counts[1, 1, 0, 1] == 5


# =============================================================================
# Tests for compute_single_twirled_covariance
# =============================================================================


class TestComputeSingleTwirledCovariance:
    """Tests for compute_single_twirled_covariance function."""

    def test_returns_correct_structure(self, simple_test_data):
        """Test that function returns correct structure."""
        result = compute_single_twirled_covariance(simple_test_data)

        assert isinstance(result, dict)
        assert "covariance_matrices" in result
        assert "error_labels" in result
        assert "measured_qubits" in result

        corr_matrices = result["covariance_matrices"]
        error_labels = result["error_labels"]
        measured_qubits = result["measured_qubits"]

        assert isinstance(corr_matrices, dict)
        assert len(corr_matrices) == 4  # 4 error IDs
        assert len(error_labels) == 4
        assert measured_qubits == ["QB1", "QB2"]

        # Verify error_labels have expected error IDs
        error_ids = [eid for eid, _ in error_labels]
        assert sorted(error_ids) == [0, 1, 2, 3]

    def test_matrix_shape(self, simple_test_data):
        """Test that each matrix has correct shape."""
        result = compute_single_twirled_covariance(simple_test_data)
        corr_matrices = result["covariance_matrices"]

        for eid, matrix in corr_matrices.items():
            assert matrix.shape == (2, 2)

    def test_diagonal_is_zero(self, simple_test_data):
        """Test that diagonal elements are zero."""
        result = compute_single_twirled_covariance(simple_test_data)
        corr_matrices = result["covariance_matrices"]

        for matrix in corr_matrices.values():
            np.testing.assert_array_equal(np.diag(matrix), [0, 0])

    def test_error_labels_format(self, simple_test_data):
        """Test that error labels are properly formatted."""
        result = compute_single_twirled_covariance(simple_test_data)
        error_labels = result["error_labels"]
        measured_qubits = result["measured_qubits"]

        # Verify measured_qubits match input
        assert measured_qubits == simple_test_data["measured_qubits"]

        # Expected labels from data_processing.py label_map
        expected_labels = {
            0: "P(01|00) + P(00|01)",
            1: "P(10|11) + P(11|10)",
            2: "P(11|00) + P(10|01)",
            3: "P(01|10) + P(00|11)",
        }

        for eid, label in error_labels:
            assert eid in [0, 1, 2, 3]
            assert label == expected_labels[eid]

    def test_with_large_data(self, large_test_data):
        """Test with larger dataset."""
        result = compute_single_twirled_covariance(large_test_data)
        corr_matrices = result["covariance_matrices"]

        # Correlation values should be small for independent errors
        for matrix in corr_matrices.values():
            assert np.all(np.abs(matrix) < 0.1)


# =============================================================================
# Tests for compute_state_dependent_covariance
# =============================================================================


class TestComputeStateDependentCovariance:
    """Tests for compute_state_dependent_covariance function."""

    def test_returns_correct_structure(self, simple_test_data):
        """Test that function returns correct structure."""
        result = compute_state_dependent_covariance(simple_test_data)

        assert isinstance(result, dict)
        assert "covariance_matrices" in result
        assert "error_labels" in result
        assert "measured_qubits" in result

        corr_matrices = result["covariance_matrices"]
        error_labels = result["error_labels"]
        measured_qubits = result["measured_qubits"]

        assert isinstance(corr_matrices, dict)
        assert len(corr_matrices) == 12  # 12 error ID pairs
        assert len(error_labels) == 12
        assert measured_qubits == ["QB1", "QB2"]

        # Verify expected error ID pairs
        expected_pairs = [
            (2, 0),
            (3, 0),
            (2, 1),
            (3, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (2, 2),
            (3, 3),
            (2, 3),
            (3, 2),
        ]
        error_id_pairs = [eid_pair for eid_pair, _ in error_labels]
        assert sorted(error_id_pairs) == sorted(expected_pairs)

    def test_keys_are_tuples(self, simple_test_data):
        """Test that keys are tuples of length 2."""
        result = compute_state_dependent_covariance(simple_test_data)
        corr_matrices = result["covariance_matrices"]

        for key in corr_matrices.keys():
            assert isinstance(key, tuple)
            assert len(key) == 2

    def test_error_labels_format(self, simple_test_data):
        """Test that error labels are properly formatted."""
        result = compute_state_dependent_covariance(simple_test_data)
        error_labels = result["error_labels"]
        measured_qubits = result["measured_qubits"]

        # Verify measured_qubits match input
        assert measured_qubits == simple_test_data["measured_qubits"]

        # Expected labels from data_processing.py label_map
        expected_labels = {
            (0, 2): "P(01|00)",
            (0, 3): "P(00|01)",
            (1, 2): "P(11|10)",
            (1, 3): "P(10|11)",
            (2, 0): "P(10|00)",
            (3, 0): "P(00|10)",
            (2, 1): "P(11|01)",
            (3, 1): "P(01|11)",
            (2, 2): "P(11|00)",
            (3, 3): "P(00|11)",
            (2, 3): "P(10|01)",
            (3, 2): "P(01|10)",
        }

        for eid_pair, label in error_labels:
            assert isinstance(eid_pair, tuple)
            assert label == expected_labels[eid_pair]


# =============================================================================
# Tests for compute_double_twirled_covariance
# =============================================================================


class TestComputeDoubleTwirledCovariance:
    """Tests for compute_double_twirled_covariance function."""

    def test_returns_correct_structure(self, simple_test_data):
        """Test that function returns correct structure."""
        result = compute_double_twirled_covariance(simple_test_data)

        assert isinstance(result, dict)
        assert "covariance_matrices" in result
        assert "error_labels" in result
        assert "measured_qubits" in result

        corr_matrices = result["covariance_matrices"]
        error_labels = result["error_labels"]
        measured_qubits = result["measured_qubits"]

        assert isinstance(corr_matrices, dict)
        assert len(corr_matrices) == 3  # 3 syndrome types
        assert len(error_labels) == 3
        assert measured_qubits == ["QB1", "QB2"]

        # Verify syndrome IDs are exactly 0, 1, 2
        syndrome_ids = [sid for sid, _ in error_labels]
        assert sorted(syndrome_ids) == [0, 1, 2]

    def test_syndrome_ids(self, simple_test_data):
        """Test that syndrome IDs are 0, 1, 2."""
        result = compute_double_twirled_covariance(simple_test_data)
        corr_matrices = result["covariance_matrices"]

        assert 0 in corr_matrices
        assert 1 in corr_matrices
        assert 2 in corr_matrices

    def test_error_labels_format(self, simple_test_data):
        """Test that error labels are properly formatted."""
        result = compute_double_twirled_covariance(simple_test_data)
        error_labels = result["error_labels"]
        measured_qubits = result["measured_qubits"]

        # Verify measured_qubits match input
        assert measured_qubits == simple_test_data["measured_qubits"]

        # Expected labels from data_processing.py label_map
        expected_labels = {
            0: "P(j0|j1) + P(j1|j0)",
            1: "P(0j|1j) + P(1j|0j)",
            2: "P(11|00) + P(00|11)",
        }

        for syndrome_id, label in error_labels:
            assert syndrome_id in [0, 1, 2]
            assert label == expected_labels[syndrome_id]


# =============================================================================
# Tests for ERROR_TO_PREP mapping
# =============================================================================


class TestErrorToPrep:
    """Tests for ERROR_TO_PREP constant."""

    def test_mapping_values(self):
        """Test that ERROR_TO_PREP has correct mappings."""
        assert ERROR_TO_PREP[0] == 0  # No error from |0> prep
        assert ERROR_TO_PREP[1] == 1  # No error from |1> prep
        assert ERROR_TO_PREP[2] == 0  # Bit flip from |0> prep
        assert ERROR_TO_PREP[3] == 1  # Bit flip from |1> prep
