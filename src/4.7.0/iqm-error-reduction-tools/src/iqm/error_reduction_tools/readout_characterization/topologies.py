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

"""QPU topology utilities for readout error characterization.

This module provides topology data for IQM quantum processing units (QPUs).

For production use, coupler connectivity and probe-line groupings are fetched
directly from the quantum computer via
:func:`~iqm.error_reduction_tools.utils.topology_utils.topology_from_qc`.

Qubit grid positions are computed from the connectivity using
:func:`~iqm.error_reduction_tools.utils.topology_utils.compute_crystal_layout`.

Usage with data_collection and data_processing:
    >>> from iqm.error_reduction_tools.utils.topology_utils import topology_from_qc
    >>> topology = topology_from_qc(client)
    >>> qubit_list = topology.get_qubit_labels()
    >>> # Use qubit_list with run_calibration_circuits or create_calibration_circuits
"""


# =============================================================================
# Topology Registry and Utility Functions
# =============================================================================


class QPUTopology:
    """Container for QPU topology information.

    Attributes:
        name: Name of the quantum computer (e.g., "emerald", "garnet")
        num_qubits: Total number of qubits
        positions: Dictionary mapping qubit number to (x, y) position for visualization.
            Populated when the topology is built via
            :func:`~iqm.error_reduction_tools.utils.topology_utils.topology_from_qc`;
            empty by default.
        couplers: List of tuples representing connected qubit pairs
        control_lines: List of tuples grouping qubits by control/readout lines
        adjacent_qubits: Optional dictionary mapping each qubit to its neighbors

    """

    def __init__(
        self,
        name: str,
        num_qubits: int,
        positions: dict[int, tuple[int, int]] | None = None,
        couplers: list[tuple[str, str]] | None = None,
        control_lines: list[tuple[str, ...]] | None = None,
        adjacent_qubits: dict[str, list[str]] | None = None,
    ) -> None:
        self.name = name
        self.num_qubits = num_qubits
        self.positions = positions or {}
        self.couplers = couplers or []
        self.control_lines = control_lines or []
        self.adjacent_qubits = adjacent_qubits or self._compute_adjacent_qubits()

    def _compute_adjacent_qubits(self) -> dict[str, list[str]]:
        """Compute adjacent qubits from coupler list.

        Returns:
            Mapping from qubit name to a list of its neighbors (including itself).

        """
        adjacent: dict[str, list[str]] = {}
        for qb1, qb2 in self.couplers:
            if qb1 not in adjacent:
                adjacent[qb1] = [qb1]
            if qb2 not in adjacent:
                adjacent[qb2] = [qb2]
            if qb2 not in adjacent[qb1]:
                adjacent[qb1].append(qb2)
            if qb1 not in adjacent[qb2]:
                adjacent[qb2].append(qb1)
        return adjacent

    def get_qubit_labels(self) -> list[str]:
        """Return list of all qubit names (e.g., ['QB1', 'QB2', ...])."""
        return [f"QB{i}" for i in range(1, self.num_qubits + 1)]

    @staticmethod
    def parse_qubit_index(label: str) -> int:
        """Extract the numeric index from a qubit name.

        Parses names of the form ``"QBn"`` and returns the integer *n*.

        Args:
            label: Qubit name (e.g., ``"QB12"``).

        Returns:
            The integer index embedded in the name.

        Example:
            >>> QPUTopology.parse_qubit_index("QB12")
            12

        """
        return int(label[2:])

    @staticmethod
    def sort_qubit_labels(labels: list[str]) -> list[str]:
        """Sort qubit labels by their numeric index.

        Example:
            >>> QPUTopology.sort_qubit_labels(["QB12", "QB4", "QB1"])
            ['QB1', 'QB4', 'QB12']

        """
        return sorted(labels, key=QPUTopology.parse_qubit_index)

    def get_neighbors(self, qubit: str) -> list[str]:
        """Get neighboring qubits for a given qubit, excluding itself.

        Args:
            qubit: Qubit label (e.g., 'QB1').

        Returns:
            List of neighboring qubit labels, **not** including `qubit`.

        Example:
            >>> topology.get_neighbors("QB5")
            ['QB1', 'QB4', 'QB6', 'QB11']

        """
        return [q for q in self.adjacent_qubits.get(qubit, []) if q != qubit]

    def get_qubits_in_control_line(self, line_index: int) -> tuple[str, ...]:
        """Get qubits belonging to a specific control/readout line.

        Args:
            line_index: Index of the control line (0-based).

        Returns:
            Tuple of qubit labels in that control line.

        Raises:
            ValueError: If no control lines are defined or ``line_index`` is out of range.

        """
        if not self.control_lines:
            raise ValueError(f"No control lines defined for topology '{self.name}'")
        if line_index < 0 or line_index >= len(self.control_lines):
            raise ValueError(
                f"line_index {line_index} is out of range for topology '{self.name}' "
                f"with {len(self.control_lines)} control line(s)."
            )
        return self.control_lines[line_index]

    def get_control_line_for_qubit(self, qubit: str) -> int | None:
        """Get the control line index for a given qubit.

        Args:
            qubit: Qubit label (e.g., 'QB1').

        Returns:
            Index of the control line containing the ``qubit``, or ``None`` if not found.

        """
        for i, line in enumerate(self.control_lines):
            if qubit in line:
                return i
        return None

    def get_coupler_pairs(self) -> list[tuple[str, str]]:
        """Return list of all coupler pairs (connected qubits)."""
        return self.couplers.copy()


def get_subset_positions(
    topology: QPUTopology,
    qubit_labels: list[str],
) -> dict[int, tuple[int, int]]:
    """Get positions for a subset of qubits from a topology.

    Useful when running characterization on a subset of qubits but still
    wanting to visualize them on the full QPU layout.

    Args:
        topology: :class:`QPUTopology` instance.
        qubit_labels: List of qubit labels to get positions for.

    Returns:
        Dictionary mapping qubit indices to (x, y) positions.

    Example:
        >>> topology = topology_from_qc(client)  # obtain from a connected quantum computer
        >>> subset = ["QB1", "QB2", "QB5", "QB6"]
        >>> positions = get_subset_positions(topology, subset)

    """
    positions = {}
    for qb in qubit_labels:
        qb_idx = QPUTopology.parse_qubit_index(qb)
        if qb_idx in topology.positions:
            positions[qb_idx] = topology.positions[qb_idx]
    return positions


def get_qubits_by_control_line(
    topology: QPUTopology,
    line_indices: list[int] | None = None,
) -> list[str]:
    """Get qubit labels grouped by control lines.

    Useful for selecting qubits that share readout infrastructure.

    Args:
        topology: :class:`QPUTopology` instance.
        line_indices: Optional list of control line indices to include.
            If ``None``, returns all qubits.

    Returns:
        List of qubit labels from the specified control lines.

    Example:
        >>> topology = topology_from_qc(client)  # obtain from a connected quantum computer
        >>> # Get qubits from first two control lines
        >>> qubits = get_qubits_by_control_line(topology, [0, 1])

    """
    if line_indices is None:
        return topology.get_qubit_labels()

    qubits: list[str] = []
    for idx in line_indices:
        if idx < len(topology.control_lines):
            qubits.extend(topology.control_lines[idx])
    return qubits
