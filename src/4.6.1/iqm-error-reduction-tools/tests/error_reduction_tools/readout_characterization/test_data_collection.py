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

"""Pytest tests for data_collection.py module."""

from unittest.mock import MagicMock

from iqm.error_reduction_tools.readout_characterization.data_collection import (
    create_calibration_circuits,
    generate_strings,
    run_calibration_circuits,
)
from iqm.error_reduction_tools.utils.topology_utils import operational_qubits_from_qc
import numpy as np
import pytest


class TestGenerateStrings:
    """Tests for generate_strings function."""

    def test_correct_length(self):
        """Test that generated strings have correct length."""
        rgen = np.random.default_rng(42)
        strings = generate_strings(number_of_qubits=5, number_of_circuits=10, rgen=rgen)

        assert len(strings) == 10
        for s in strings:
            assert len(s) == 5

    def test_only_i_and_x(self):
        """Test that generated strings only contain 'I' and 'X'."""
        rgen = np.random.default_rng(42)
        strings = generate_strings(number_of_qubits=10, number_of_circuits=100, rgen=rgen)

        for s in strings:
            for char in s:
                assert char in ["I", "X"]

    def test_symmetrize_creates_pairs(self):
        """Test that symmetrize creates matching pairs."""
        rgen = np.random.default_rng(42)
        strings = generate_strings(number_of_qubits=5, number_of_circuits=10, rgen=rgen, symmetrize=True)

        # Check pairs
        for i in range(0, len(strings) - 1, 2):
            s1, s2 = strings[i], strings[i + 1]
            # s2 should be the bit-flipped version of s1
            for c1, c2 in zip(s1, s2):
                if c1 == "I":
                    assert c2 == "X"
                else:
                    assert c2 == "I"

    def test_no_symmetrize(self):
        """Test that without symmetrize, correct number of strings generated."""
        rgen = np.random.default_rng(42)
        strings = generate_strings(number_of_qubits=5, number_of_circuits=10, rgen=rgen, symmetrize=False)

        assert len(strings) == 10

    def test_odd_number_of_circuits(self):
        """Test handling of odd number of circuits."""
        rgen = np.random.default_rng(42)
        strings = generate_strings(number_of_qubits=5, number_of_circuits=11, rgen=rgen, symmetrize=True)

        assert len(strings) == 11

    def test_single_circuit(self):
        """Test generating a single circuit."""
        rgen = np.random.default_rng(42)
        strings = generate_strings(number_of_qubits=3, number_of_circuits=1, rgen=rgen)

        assert len(strings) == 1
        assert len(strings[0]) == 3

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same strings."""
        rgen1 = np.random.default_rng(42)
        rgen2 = np.random.default_rng(42)

        strings1 = generate_strings(number_of_qubits=5, number_of_circuits=10, rgen=rgen1)
        strings2 = generate_strings(number_of_qubits=5, number_of_circuits=10, rgen=rgen2)

        assert strings1 == strings2


# =============================================================================
# Tests for create_calibration_circuits
# =============================================================================


class TestCreateCalibrationCircuits:
    @pytest.mark.parametrize(
        "qubit_list,num_circuits,symmetrize,equatorial_randomization",
        [
            (["QB1", "QB2"], 6, False, False),
            (["QB1", "QB2", "QB3"], 8, True, False),
            (["QB1", "QB2", "QB3", "QB4"], 7, False, True),
            (["QB1"], 5, True, True),
        ],
    )
    def test_gate_application_and_symmetry(self, qubit_list, num_circuits, symmetrize, equatorial_randomization):
        """Test prx/I gate application for various qubit lists, symmetrize, and equatorial randomization."""
        seed = 123
        circuits, prep_strings = create_calibration_circuits(
            qubits=qubit_list,
            number_of_circuits=num_circuits,
            symmetrize=symmetrize,
            seed=seed,
            equatorial_randomization=equatorial_randomization,
        )

        # Check prx/I gate application matches prep_string
        for circ, prep in zip(circuits, prep_strings):
            prx_ops = [op for op in circ.instructions if op.name == "prx"]
            for idx, (qname, pchar) in enumerate(zip(qubit_list, prep)):
                has_prx = any(qname in op.locus for op in prx_ops)
                if pchar == "X":
                    assert has_prx, f"Expected prx on {qname} for prep '{prep}'"
                else:
                    assert not has_prx or all(qname not in op.locus for op in prx_ops), (
                        f"Unexpected prx on {qname} for prep '{prep}'"
                    )

        # If symmetrize, check that every pair is bit-flipped
        if symmetrize:
            for i in range(0, len(prep_strings) - 1, 2):
                s1, s2 = prep_strings[i], prep_strings[i + 1]
                for c1, c2 in zip(s1, s2):
                    assert (c1 == "I" and c2 == "X") or (c1 == "X" and c2 == "I"), f"Symmetrize failed: {s1}, {s2}"

        # If equatorial_randomization, check phase is not always zero for prx gates
        # (support both "phase" and legacy "phase_t" argument names)
        if equatorial_randomization:
            found_nonzero = False
            for circ in circuits:
                for op in circ.instructions:
                    if op.name == "prx" and hasattr(op, "args"):
                        phase = op.args.get("phase", op.args.get("phase_t", 0.0))
                        if not np.isclose(phase, 0):
                            found_nonzero = True
            assert found_nonzero, "Equatorial randomization did not set any nonzero phase."

    def test_empty_qubit_list_raises(self):
        """Test that empty qubit list raises ValueError."""

        with pytest.raises(ValueError) as exc_info:
            create_calibration_circuits(qubits=[], number_of_circuits=10)

        assert "empty" in str(exc_info.value).lower()

    def test_zero_circuits_raises(self):
        """Test that zero circuits raises ValueError."""

        with pytest.raises(ValueError) as exc_info:
            create_calibration_circuits(qubits=["QB1", "QB2"], number_of_circuits=0)

        assert "at least 1" in str(exc_info.value).lower()

    def test_correct_count(self):
        """Test that correct number of circuits is created."""

        circuits, prep_strings = create_calibration_circuits(
            qubits=["QB1", "QB2", "QB3"],
            number_of_circuits=10,
            seed=42,
        )

        assert len(circuits) == 10
        assert len(prep_strings) == 10

    def test_prep_strings_match_qubit_count(self):
        """Test that prep strings have same length as qubit list."""

        qubit_list = ["QB1", "QB2", "QB3", "QB4"]
        circuits, prep_strings = create_calibration_circuits(
            qubits=qubit_list,
            number_of_circuits=5,
            seed=42,
        )

        for prep_str in prep_strings:
            assert len(prep_str) == len(qubit_list)

    def test_deterministic_with_seed(self):
        """Test that same seed produces same circuits."""

        circuits1, prep1 = create_calibration_circuits(
            qubits=["QB1", "QB2"],
            number_of_circuits=5,
            seed=42,
        )
        circuits2, prep2 = create_calibration_circuits(
            qubits=["QB1", "QB2"],
            number_of_circuits=5,
            seed=42,
        )

        assert prep1 == prep2


# =============================================================================
# Tests for operational-qubit resolution
# =============================================================================


def _make_dqa(measure_loci: list[tuple[str, ...]] | None, qubits: list[str] | None = None) -> MagicMock:
    """Build a mock DynamicQuantumArchitecture with the given measure-gate loci."""
    dqa = MagicMock()
    gates: dict[str, MagicMock] = {}
    if measure_loci is not None:
        measure_gate = MagicMock()
        measure_gate.loci = measure_loci
        gates["measure"] = measure_gate
    dqa.gates = gates
    dqa.qubits = qubits if qubits is not None else []
    return dqa


def _make_client(measure_loci: list[tuple[str, ...]] | None, chip_qubits: list[str]) -> MagicMock:
    """Build a mock Pulla client exposing a DQA and a chip topology."""
    client = MagicMock()
    client._iqm_server_client.get_dynamic_quantum_architecture.return_value = _make_dqa(measure_loci)
    chip_topo = MagicMock()
    chip_topo.qubits_sorted = chip_qubits
    client.get_chip_topology.return_value = chip_topo
    return client


class TestOperationalQubitsFromQc:
    """Tests for operational_qubits_from_qc."""

    def test_returns_measure_gate_qubits(self):
        """Operational qubits come from the measure-gate loci, preserving order."""
        client = _make_client([("QB1",), ("QB2",), ("QB5",)], chip_qubits=["QB1", "QB2", "QB3", "QB4", "QB5"])
        assert operational_qubits_from_qc(client) == ["QB1", "QB2", "QB5"]

    def test_falls_back_to_dqa_qubits_without_measure_gate(self):
        """When no measure gate exists, fall back to DQA qubits."""
        client = MagicMock()
        client._iqm_server_client.get_dynamic_quantum_architecture.return_value = _make_dqa(
            measure_loci=None, qubits=["QB1", "QB3"]
        )
        assert operational_qubits_from_qc(client) == ["QB1", "QB3"]

    def test_returns_none_on_error(self):
        """Return None when the DQA cannot be retrieved."""
        client = MagicMock()
        client._iqm_server_client.get_dynamic_quantum_architecture.side_effect = RuntimeError("no dqa")
        assert operational_qubits_from_qc(client) is None


# =============================================================================
# Tests for run_calibration_circuits qubit resolution
# =============================================================================


class TestRunCalibrationCircuitsQubitResolution:
    """Tests that run_calibration_circuits scopes characterization to operational qubits."""

    @staticmethod
    def _wire_compiler(client: MagicMock) -> None:
        compiler = MagicMock()
        compiler.compile.return_value = (MagicMock(), {"shots": 0})
        client.get_standard_compiler.return_value = compiler
        client.submit_playlist.return_value = MagicMock()

    def test_none_resolves_to_operational_qubits(self):
        """``qubits=None`` characterizes operational qubits."""
        client = _make_client([("QB1",), ("QB2",)], chip_qubits=["QB1", "QB2", "QB3", "QB4"])
        self._wire_compiler(client)

        _, job_info = run_calibration_circuits(client, qubits=None, number_of_circuits=4, shots=400)

        assert job_info.qubits == ["QB1", "QB2"]

    def test_explicit_list_drops_non_operational_qubits(self):
        """Non-operational qubits in an explicit list are dropped with a warning."""
        client = _make_client([("QB1",), ("QB2",)], chip_qubits=["QB1", "QB2", "QB3", "QB4"])
        self._wire_compiler(client)

        with pytest.warns(UserWarning, match="non-operational"):
            _, job_info = run_calibration_circuits(client, qubits=["QB1", "QB4"], number_of_circuits=4, shots=400)

        assert job_info.qubits == ["QB1"]

    def test_no_operational_qubits_raises(self):
        """Requesting only non-operational qubits raises ValueError."""
        client = _make_client([("QB1",), ("QB2",)], chip_qubits=["QB1", "QB2", "QB3", "QB4"])
        self._wire_compiler(client)

        with pytest.warns(UserWarning, match="non-operational"):
            with pytest.raises(ValueError, match="No operational qubits"):
                run_calibration_circuits(client, qubits=["QB4"], number_of_circuits=4, shots=400)

    def test_falls_back_to_chip_topology_when_dqa_unavailable(self):
        """When the DQA cannot be retrieved, fall back to the chip topology with a warning."""
        client = MagicMock()
        client._iqm_server_client.get_dynamic_quantum_architecture.side_effect = RuntimeError("no dqa")
        chip_topo = MagicMock()
        chip_topo.qubits_sorted = ["QB1", "QB2"]
        client.get_chip_topology.return_value = chip_topo
        self._wire_compiler(client)

        with pytest.warns(UserWarning, match="Could not determine operational qubits"):
            _, job_info = run_calibration_circuits(client, qubits=None, number_of_circuits=4, shots=400)

        assert job_info.qubits == ["QB1", "QB2"]
