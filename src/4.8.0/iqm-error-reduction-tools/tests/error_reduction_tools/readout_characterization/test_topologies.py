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

"""Pytest tests for topologies.py module."""

from iqm.error_reduction_tools.readout_characterization.topologies import (
    QPUTopology,
    get_qubits_by_control_line,
    get_subset_positions,
)
from iqm.error_reduction_tools.utils.topology_utils import compute_crystal_layout
import pytest


class TestQPUTopology:
    """Tests for QPUTopology class."""

    def test_parse_qubit_index(self):
        """Test extracting numeric index from qubit labels."""
        assert QPUTopology.parse_qubit_index("QB1") == 1
        assert QPUTopology.parse_qubit_index("QB12") == 12
        assert QPUTopology.parse_qubit_index("QB54") == 54

    def test_sort_qubit_labels(self):
        """Test natural numeric sorting of qubit labels."""
        labels = ["QB12", "QB4", "QB1", "QB20", "QB3"]
        sorted_labels = QPUTopology.sort_qubit_labels(labels)
        assert sorted_labels == ["QB1", "QB3", "QB4", "QB12", "QB20"]

    def test_sort_qubit_labels_empty(self):
        """Test sorting an empty list of qubit labels."""
        assert QPUTopology.sort_qubit_labels([]) == []

    def test_create_custom_topology(self):
        """Test creating a custom topology."""
        topology = QPUTopology(
            name="test",
            num_qubits=3,
            positions={1: (0, 0), 2: (1, 0), 3: (0, 1)},
            couplers=[("QB1", "QB2"), ("QB2", "QB3")],
        )

        assert topology.num_qubits == 3
        assert len(topology.couplers) == 2

    def test_compute_adjacent_qubits(self):
        """Test automatic computation of adjacent qubits from couplers."""
        topology = QPUTopology(
            name="test",
            num_qubits=3,
            positions={1: (0, 0), 2: (1, 0), 3: (0, 1)},
            couplers=[("QB1", "QB2"), ("QB2", "QB3")],
        )

        assert "QB2" in topology.adjacent_qubits["QB1"]
        assert "QB1" in topology.adjacent_qubits["QB2"]
        assert "QB3" in topology.adjacent_qubits["QB2"]

    def test_get_qubits_in_control_line_no_lines_raises(self):
        """Test that getting control line raises error when none defined."""
        topology = QPUTopology(
            name="no_lines",
            num_qubits=2,
            positions={1: (0, 0), 2: (1, 0)},
            couplers=[("QB1", "QB2")],
            control_lines=None,
        )

        with pytest.raises(ValueError):
            topology.get_qubits_in_control_line(0)

    def test_get_coupler_pairs_returns_copy(self):
        """Test getting coupler pairs returns a copy."""
        topology = QPUTopology(
            name="test_couplers",
            num_qubits=3,
            couplers=[("QB1", "QB2"), ("QB2", "QB3")],
        )
        pairs = topology.get_coupler_pairs()
        assert isinstance(pairs, list)
        assert len(pairs) == len(topology.couplers)
        # Should be a copy - modifying shouldn't affect original
        original_len = len(topology.couplers)
        pairs.append(("QB999", "QB1000"))
        assert len(topology.couplers) == original_len


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_get_subset_positions(self):
        """Test getting positions for a subset of qubits."""
        topology = QPUTopology(
            name="test_positions",
            num_qubits=5,
            positions={1: (0, 0), 2: (1, 0), 3: (2, 0), 4: (0, 1), 5: (1, 1)},
        )
        subset = ["QB1", "QB2", "QB5"]
        positions = get_subset_positions(topology, subset)

        assert len(positions) == 3
        assert 1 in positions
        assert 2 in positions
        assert 5 in positions

    def test_get_qubits_by_control_line_none_returns_all(self):
        """Test that None returns all qubits."""
        topology = QPUTopology(
            name="test_lines",
            num_qubits=4,
            positions={},
            control_lines=[("QB1", "QB2"), ("QB3", "QB4")],
        )
        qubits = get_qubits_by_control_line(topology, None)

        assert len(qubits) == 4


class TestComputeCrystalLayout:
    """Tests for compute_crystal_layout function."""

    def test_2x2_grid(self):
        """Test layout computation for a simple 2x2 grid (4 qubits)."""
        couplers = [("QB1", "QB2"), ("QB2", "QB3"), ("QB3", "QB4"), ("QB4", "QB1")]
        qubits = ["QB1", "QB2", "QB3", "QB4"]
        layout = compute_crystal_layout(couplers, qubits)

        assert len(layout) == 4
        # All positions should be unique
        assert len(set(layout.values())) == 4

    def test_star_topology_returns_empty(self):
        """Test that a star topology (no corners) returns empty layout."""
        couplers = [("QB1", "QB2"), ("QB1", "QB3"), ("QB1", "QB4"), ("QB1", "QB5")]
        qubits = ["QB1", "QB2", "QB3", "QB4", "QB5"]
        layout = compute_crystal_layout(couplers, qubits)

        assert layout == {}

    def test_garnet_like_grid(self):
        """Test layout for a Garnet-like 4x5 grid topology."""
        # Build a 4x5 grid connectivity (20 qubits)
        qubits = [f"QB{i}" for i in range(1, 21)]
        couplers = []
        # Arrange as 4 columns x 5 rows
        for row in range(5):
            for col in range(4):
                idx = row * 4 + col + 1
                # Right neighbor
                if col < 3:
                    couplers.append((f"QB{idx}", f"QB{idx + 1}"))
                # Up neighbor
                if row < 4:
                    couplers.append((f"QB{idx}", f"QB{idx + 4}"))
        layout = compute_crystal_layout(couplers, qubits)

        assert len(layout) == 20
        assert len(set(layout.values())) == 20

    def test_empty_couplers_returns_empty(self):
        """Test that no couplers returns empty layout."""
        layout = compute_crystal_layout([], ["QB1", "QB2"])
        assert layout == {}
