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

"""Unit tests for the CircuitTwirler twirling API.

Tests cover:
- TwirlingConfiguration defaults and construction
- CircuitTwirler lifecycle (twirl → submit → retrieve) via mocks
- Error handling for out-of-order lifecycle calls
- Strategy validation and fallback behaviour
- Star topology detection fallback
- Persistence (to_dict / from_dict / save / load)
- Retry logic in retrieve_counts
- Optional client: construction without client, late-binding at submit()
- get_twirled_circuits() and get_twirled_circuits_flat() accessors (with qiskit flag)
- submit(client=) late-binding and LOCAL strategy client-swap rejection
"""

from __future__ import annotations

import json
import math
from unittest.mock import MagicMock, patch
import warnings

from iqm.error_reduction_tools.twirling.twirling_api import (
    CircuitTwirler,
    TwirlingConfiguration,
)
from iqm.error_reduction_tools.utils.circuit_utils import TwirledCircuit
import pytest
from qiskit.circuit import QuantumCircuit as QiskitQuantumCircuit
from qrisp import QuantumCircuit as QrispQuantumCircuit

from iqm.pulse import Circuit as PulseCircuit
from iqm.pulse.builder import CircuitOperation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pulse_circuit(qubits: list[str]) -> TwirledCircuit:
    """Build a minimal TwirledCircuit that measures *qubits*."""
    ops: list[CircuitOperation] = []
    for i, qb in enumerate(qubits):
        ops.append(CircuitOperation(name="prx", locus=(qb,), args={"angle": 0.5, "phase": 0.0}))
    for i, qb in enumerate(qubits):
        ops.append(CircuitOperation(name="measure", locus=(qb,), args={"key": f"m_{i}"}))
    return TwirledCircuit(ops)


@pytest.fixture()
def mock_client() -> MagicMock:
    """Create a mock Pulla client that exposes a lattice topology."""
    client = MagicMock()
    chip_topo = MagicMock()
    chip_topo.qubits_sorted = ["QB1", "QB2", "QB3", "QB4"]
    chip_topo.qubits = frozenset(["QB1", "QB2", "QB3", "QB4"])
    chip_topo.computational_resonators = frozenset()
    # coupler_to_components is the API used by topology_from_qc()
    chip_topo.coupler_to_components = {
        "TC-1-2": ("QB1", "QB2"),
        "TC-1-3": ("QB1", "QB3"),
        "TC-2-4": ("QB2", "QB4"),
        "TC-3-4": ("QB3", "QB4"),
    }
    chip_topo.probe_line_to_components = {}
    client.get_chip_topology.return_value = chip_topo
    return client


@pytest.fixture()
def mock_client_no_topology() -> MagicMock:
    """Create a mock Pulla client that raises on topology look-up."""
    client = MagicMock()
    client.get_chip_topology.side_effect = RuntimeError("no topology")
    return client


@pytest.fixture()
def mock_client_star() -> MagicMock:
    """Create a mock Pulla client that exposes an IQM MOVE-gate topology.

    Includes computational resonators so that ``uses_move_gates`` returns ``True``.
    """
    client = MagicMock()
    chip_topo = MagicMock()
    chip_topo.qubits_sorted = ["QB1", "QB2", "QB3", "QB4", "QB5"]
    chip_topo.qubits = frozenset(["QB1", "QB2", "QB3", "QB4", "QB5"])
    # Computational resonators are the authoritative signal for MOVE-gate QPUs.
    chip_topo.computational_resonators = frozenset(["CR1", "CR2", "CR3", "CR4"])
    # coupler_to_components is the API used by topology_from_qc()
    chip_topo.coupler_to_components = {
        "TC-1-3": ("QB1", "QB3"),
        "TC-2-3": ("QB2", "QB3"),
        "TC-3-4": ("QB3", "QB4"),
        "TC-3-5": ("QB3", "QB5"),
    }
    chip_topo.probe_line_to_components = {}
    client.get_chip_topology.return_value = chip_topo
    return client


# ===================================================================
# TwirlingConfiguration
# ===================================================================


class TestTwirlingConfiguration:
    """Tests for TwirlingConfiguration dataclass."""

    def test_defaults(self) -> None:
        """Default config: LOCAL readout strategy, no seed, circuit twirling on, 20 instances."""
        cfg = TwirlingConfiguration()
        assert cfg.readout_twirl_strategy == "LOCAL"
        assert cfg.seed is None
        assert cfg.circuit_twirling is True
        assert cfg.num_twirling_instances == 20

    def test_custom_values(self) -> None:
        """Custom values are preserved."""
        cfg = TwirlingConfiguration(
            readout_twirl_strategy="NONE",
            seed=42,
            circuit_twirling=False,
        )
        assert cfg.readout_twirl_strategy == "NONE"
        assert cfg.seed == 42
        assert cfg.circuit_twirling is False


# ===================================================================
# CircuitTwirler — lifecycle errors (before twirl)
# ===================================================================


class TestCircuitTwirlerLifecycleErrors:
    """Calling lifecycle methods out of order must raise RuntimeError."""

    def test_submit_before_twirl(self, mock_client: MagicMock) -> None:
        """submit() without twirl() raises RuntimeError."""
        twirler = CircuitTwirler(mock_client)
        with pytest.raises(RuntimeError, match="Call twirl"):
            twirler.submit()

    def test_retrieve_before_submit(self, mock_client: MagicMock) -> None:
        """retrieve_counts() without submit() raises RuntimeError."""
        twirler = CircuitTwirler(mock_client)
        with pytest.raises(RuntimeError, match="Call submit"):
            twirler.retrieve_counts()

    def test_get_rot_strings_before_twirl(self, mock_client: MagicMock) -> None:
        """get_rot_strings() without twirl() raises RuntimeError."""
        twirler = CircuitTwirler(mock_client)
        with pytest.raises(RuntimeError, match="Call twirl"):
            twirler.get_rot_strings()

    def test_get_qubit_to_bit_mapping_before_twirl(self, mock_client: MagicMock) -> None:
        """get_qubit_to_bit_mapping() without twirl() raises RuntimeError."""
        twirler = CircuitTwirler(mock_client)
        with pytest.raises(RuntimeError, match="Call twirl"):
            twirler.get_qubit_to_bit_mapping()

    def test_get_job_before_submit(self, mock_client: MagicMock) -> None:
        """get_job() without submit() raises RuntimeError."""
        twirler = CircuitTwirler(mock_client)
        with pytest.raises(RuntimeError, match="Call submit"):
            twirler.get_job()


# ===================================================================
# CircuitTwirler — input validation
# ===================================================================


class TestCircuitTwirlerValidation:
    """Input validation for twirl()."""

    def test_empty_circuits_raises(self, mock_client: MagicMock) -> None:
        """Passing an empty list to twirl() raises ValueError."""
        twirler = CircuitTwirler(mock_client)
        with pytest.raises(ValueError, match="At least one circuit"):
            twirler.twirl([])

    def test_unsupported_strategy_raises(self, mock_client: MagicMock) -> None:
        """An invalid strategy name raises ValueError."""
        cfg = TwirlingConfiguration(readout_twirl_strategy="BOGUS")
        with pytest.raises(ValueError, match="Unsupported twirling strategy"):
            _ = CircuitTwirler(mock_client, config=cfg)

    def test_unsupported_circuit_type_raises(self, mock_client: MagicMock) -> None:
        """Passing an unsupported object to twirl() raises TypeError."""
        twirler = CircuitTwirler(mock_client)
        with pytest.raises(TypeError, match="Unsupported circuit type"):
            twirler.twirl(["not_a_circuit"])


# ===================================================================
# CircuitTwirler — twirl with different strategies
# ===================================================================


class TestCircuitTwirlerTwirl:
    """Tests for the twirl() method with various strategies and topologies."""

    def test_local_strategy_with_topology(self, mock_client: MagicMock) -> None:
        """LOCAL strategy with a lattice topology produces 4 rot strings (one per circuit-twirling
        instance is 1 when circuit_twirling=False)."""
        qubits = ["QB1", "QB2"]
        pc = _make_pulse_circuit(qubits)
        cfg = TwirlingConfiguration(readout_twirl_strategy="LOCAL", seed=0, circuit_twirling=False)
        twirler = CircuitTwirler(mock_client, config=cfg)
        result = twirler.twirl([pc])

        assert result is twirler  # method chaining
        rot_strings = twirler.get_rot_strings()
        assert len(rot_strings) == 1  # one input circuit
        assert len(rot_strings[0]) == 4  # LOCAL produces 4 strings

        for rs in rot_strings[0]:
            # Each string must have exactly one char per measured qubit
            assert len(rs) == len(qubits), f"Expected length {len(qubits)}, got {len(rs)}: {rs!r}"
            # Only valid characters
            assert set(rs) <= {"I", "X"}, f"Invalid chars in rot string: {rs!r}"

        # LOCAL guarantees balanced I/X per qubit across the 4 strings
        for qubit_idx in range(len(qubits)):
            chars = [rot_strings[0][s][qubit_idx] for s in range(4)]
            assert chars.count("I") == 2 and chars.count("X") == 2, (
                f"Qubit {qubit_idx}: expected 2×I and 2×X across 4 LOCAL strings, got {chars}"
            )

    def test_minimal_strategy(self, mock_client: MagicMock) -> None:
        """MINIMAL strategy produces 2 complementary rot strings (circuit_twirling=False)."""
        qubits = ["QB1", "QB2"]
        pc = _make_pulse_circuit(qubits)
        cfg = TwirlingConfiguration(readout_twirl_strategy="MINIMAL", seed=0, circuit_twirling=False)
        twirler = CircuitTwirler(mock_client, config=cfg)
        twirler.twirl([pc])

        rot_strings = twirler.get_rot_strings()
        assert len(rot_strings) == 1
        assert len(rot_strings[0]) == 2

        s0, s1 = rot_strings[0]
        # Each string must have one char per qubit, only I/X
        for s in (s0, s1):
            assert len(s) == len(qubits)
            assert set(s) <= {"I", "X"}

        # MINIMAL produces alternating patterns — the two strings are complementary
        assert s0 != s1, "MINIMAL strings must differ"
        for i in range(len(qubits)):
            assert {s0[i], s1[i]} == {
                "I",
                "X",
            }, f"Qubit {i}: MINIMAL strings must be complementary, got {s0[i]!r} and {s1[i]!r}"

    def test_hadamard_strategy(self, mock_client: MagicMock) -> None:
        """HADAMARD strategy produces 2^p rot strings with balanced I/X (circuit_twirling=False)."""
        qubits = ["QB1", "QB2", "QB3"]
        n = len(qubits)
        pc = _make_pulse_circuit(qubits)
        cfg = TwirlingConfiguration(readout_twirl_strategy="HADAMARD", seed=0, circuit_twirling=False)
        twirler = CircuitTwirler(mock_client, config=cfg)
        twirler.twirl([pc])

        rot_strings = twirler.get_rot_strings()
        assert len(rot_strings) == 1
        strings = rot_strings[0]

        # HadamardGenerator produces 2^p strings where p = floor(log2(n)) + 1
        p = int(math.floor(math.log2(n)) + 1)
        expected_count = 2**p
        assert len(strings) == expected_count, (
            f"Expected {expected_count} Hadamard strings for {n} qubits, got {len(strings)}"
        )

        for rs in strings:
            assert len(rs) == n, f"Expected length {n}, got {len(rs)}: {rs!r}"
            assert set(rs) <= {"I", "X"}, f"Invalid chars in rot string: {rs!r}"

        # Each qubit must have equal I and X counts (balanced)
        for qubit_idx in range(n):
            chars = [strings[s][qubit_idx] for s in range(len(strings))]
            assert chars.count("I") == chars.count("X") == expected_count // 2, (
                f"Qubit {qubit_idx}: expected {expected_count // 2}×I and {expected_count // 2}×X, "
                f"got {chars.count('I')}×I and {chars.count('X')}×X"
            )

    def test_multiple_input_circuits(self, mock_client: MagicMock) -> None:
        """Multiple input circuits each get independent randomized groups (circuit_twirling=False)."""
        qubits1 = ["QB1", "QB2"]
        qubits2 = ["QB3", "QB4"]
        pc1 = _make_pulse_circuit(qubits1)
        pc2 = _make_pulse_circuit(qubits2)
        cfg = TwirlingConfiguration(readout_twirl_strategy="MINIMAL", seed=0, circuit_twirling=False)
        twirler = CircuitTwirler(mock_client, config=cfg)
        twirler.twirl([pc1, pc2])

        rot_strings = twirler.get_rot_strings()
        assert len(rot_strings) == 2  # two input circuits

        # Each group gets 2 MINIMAL strings matching its own qubit count
        for group_idx, qubits in enumerate([qubits1, qubits2]):
            group = rot_strings[group_idx]
            assert len(group) == 2, f"Group {group_idx}: expected 2 strings, got {len(group)}"
            for rs in group:
                assert len(rs) == len(qubits), f"Group {group_idx}: expected length {len(qubits)}, got {len(rs)}"
                assert set(rs) <= {"I", "X"}
            # Complementary
            for i in range(len(qubits)):
                assert {group[0][i], group[1][i]} == {"I", "X"}

        # Mappings must reflect each circuit's qubits independently
        mappings = twirler.get_qubit_to_bit_mapping()
        assert len(mappings) == 2
        assert mappings[0] == {"QB1": 0, "QB2": 1}
        assert mappings[1] == {"QB3": 0, "QB4": 1}

    def test_circuit_twirling_multiplies_circuits(self, mock_client: MagicMock) -> None:
        """circuit_twirling=True multiplies output by default num_twirling_instances (split across rot strings).

        MINIMAL has 2 rot strings.  With default num_twirling_instances=20 the budget is
        split evenly: 10 Pauli randomizations × 2 rot strings = 20 total circuits.
        """
        pc = _make_pulse_circuit(["QB1", "QB2"])
        cfg = TwirlingConfiguration(
            readout_twirl_strategy="MINIMAL",
            circuit_twirling=True,
            seed=0,
        )
        twirler = CircuitTwirler(mock_client, config=cfg)
        twirler.twirl([pc])

        rot_strings = twirler.get_rot_strings()
        assert len(rot_strings[0]) == 20  # 10 per rot-string × 2 rot-strings

        # Exactly half of the 20 strings should be each MINIMAL pattern
        counts = {}
        for rs in rot_strings[0]:
            counts[rs] = counts.get(rs, 0) + 1
        assert len(counts) == 2, "MINIMAL produces exactly 2 distinct rot strings"
        assert list(counts.values()) == [
            10,
            10,
        ], "Each rot string should appear 10 times"

    def test_seed_reproducibility(self, mock_client: MagicMock) -> None:
        """Same seed produces identical rotation strings."""
        pc = _make_pulse_circuit(["QB1", "QB2"])
        cfg = TwirlingConfiguration(readout_twirl_strategy="MINIMAL", seed=123)

        twirler1 = CircuitTwirler(mock_client, config=cfg)
        twirler1.twirl([pc])
        rot1 = twirler1.get_rot_strings()

        twirler2 = CircuitTwirler(mock_client, config=cfg)
        twirler2.twirl([pc])
        rot2 = twirler2.get_rot_strings()

        assert rot1 == rot2

    def test_readout_twirling_disabled(self, mock_client: MagicMock) -> None:
        """With readout_twirl_strategy='NONE', every circuit has an all-I rotation string.

        NONE + circuit_twirling=True produces num_twirling_instances circuits, all
        with the trivial (all-I) readout-twirling pattern.
        """
        qubits = ["QB1", "QB2"]
        pc = _make_pulse_circuit(qubits)
        cfg = TwirlingConfiguration(
            readout_twirl_strategy="NONE",
            seed=0,
            circuit_twirling=True,
            num_twirling_instances=22,
        )
        twirler = CircuitTwirler(mock_client, config=cfg)
        twirler.twirl([pc])

        rot_strings = twirler.get_rot_strings()
        assert len(rot_strings) == 1
        # NONE has one trivial rot string; with circuit_twirling=True and 22 instances → 22 circuits
        assert len(rot_strings[0]) == 22

        # Every circuit must carry the all-I readout string (no readout flips)
        for rs in rot_strings[0]:
            assert rs == "I" * len(qubits), f"Expected all-I string, got {rs!r}"

        # Verify directly on the randomized circuits
        for group in twirler._randomized_circuits_per_input:
            for circ in group:
                if circ.rot_dict:
                    assert all(v == "I" for v in circ.rot_dict.values()), (
                        f"rot_dict should be all-I, got {circ.rot_dict}"
                    )


# ===================================================================
# CircuitTwirler — topology fallbacks
# ===================================================================


class TestCircuitTwirlerTopologyFallback:
    """Tests for topology-related fallback behaviour."""

    def test_local_without_topology_warns_and_skips(self, mock_client_no_topology: MagicMock) -> None:
        """LOCAL strategy without topology emits warning and keeps original circuit unchanged."""
        qubits = ["QB1", "QB2"]
        pc = _make_pulse_circuit(qubits)
        cfg = TwirlingConfiguration(readout_twirl_strategy="LOCAL", seed=0)
        twirler = CircuitTwirler(mock_client_no_topology, config=cfg)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            twirler.twirl([pc])
            assert any("Cannot determine QPU topology" in str(warning.message) for warning in w)

        # Should have one group with exactly the original circuit (no randomization)
        rot_strings = twirler.get_rot_strings()
        assert len(rot_strings) == 1
        assert len(rot_strings[0]) == 1

        # The single circuit in the group must be the unmodified original
        assert twirler._randomized_circuits_per_input[0][0] is pc
        assert twirler._twirling_skipped is True

    def test_move_gate_qpu_warns_and_skips(self, mock_client_star: MagicMock) -> None:
        """MOVE-gate QPU emits warning and keeps original circuit unchanged."""
        qubits = ["QB1", "QB2", "QB3"]
        pc = _make_pulse_circuit(qubits)
        cfg = TwirlingConfiguration(readout_twirl_strategy="LOCAL", seed=0)
        twirler = CircuitTwirler(mock_client_star, config=cfg)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            twirler.twirl([pc])
            assert any("MOVE gates" in str(warning.message) for warning in w)

        rot_strings = twirler.get_rot_strings()
        assert len(rot_strings) == 1
        assert len(rot_strings[0]) == 1  # original circuit only

        # Verify the original circuit is preserved and twirling was skipped
        assert twirler._randomized_circuits_per_input[0][0] is pc
        assert twirler._twirling_skipped is True

    def test_minimal_without_topology_still_works(self, mock_client_no_topology: MagicMock) -> None:
        """MINIMAL strategy works without topology and produces 2 rot strings (circuit_twirling=False)."""
        qubits = ["QB1", "QB2"]
        pc = _make_pulse_circuit(qubits)
        cfg = TwirlingConfiguration(readout_twirl_strategy="MINIMAL", seed=0, circuit_twirling=False)
        twirler = CircuitTwirler(mock_client_no_topology, config=cfg)
        twirler.twirl([pc])

        rot_strings = twirler.get_rot_strings()
        assert len(rot_strings) == 1
        assert len(rot_strings[0]) == 2

        s0, s1 = rot_strings[0]
        for s in (s0, s1):
            assert len(s) == len(qubits)
            assert set(s) <= {"I", "X"}
        # Complementary
        for i in range(len(qubits)):
            assert {s0[i], s1[i]} == {"I", "X"}

        assert twirler._twirling_skipped is False

    def test_hadamard_without_topology_still_works(self, mock_client_no_topology: MagicMock) -> None:
        """HADAMARD strategy works without topology and produces 2^p rot strings (circuit_twirling=False)."""
        qubits = ["QB1", "QB2"]
        n = len(qubits)
        pc = _make_pulse_circuit(qubits)
        cfg = TwirlingConfiguration(readout_twirl_strategy="HADAMARD", seed=0, circuit_twirling=False)
        twirler = CircuitTwirler(mock_client_no_topology, config=cfg)
        twirler.twirl([pc])

        rot_strings = twirler.get_rot_strings()
        assert len(rot_strings) == 1
        strings = rot_strings[0]

        p = int(math.floor(math.log2(n)) + 1)
        expected_count = 2**p
        assert len(strings) == expected_count

        for rs in strings:
            assert len(rs) == n
            assert set(rs) <= {"I", "X"}

        # Per-qubit I/X balance
        for qubit_idx in range(n):
            chars = [strings[s][qubit_idx] for s in range(len(strings))]
            assert chars.count("I") == chars.count("X") == expected_count // 2

        assert twirler._twirling_skipped is False


# ===================================================================
# CircuitTwirler — qubit-to-bit mapping
# ===================================================================


class TestQubitToBitMapping:
    """Tests for qubit-to-bit mapping extraction."""

    def test_mapping_matches_measured_qubits(self, mock_client: MagicMock) -> None:
        """Mapping is an exact dict of qubit → sequential index."""
        qubits = ["QB1", "QB3", "QB4"]
        pc = _make_pulse_circuit(qubits)
        cfg = TwirlingConfiguration(readout_twirl_strategy="MINIMAL", seed=0)
        twirler = CircuitTwirler(mock_client, config=cfg)
        twirler.twirl([pc])

        mappings = twirler.get_qubit_to_bit_mapping()
        assert len(mappings) == 1
        assert mappings[0] == {"QB1": 0, "QB3": 1, "QB4": 2}


# ===================================================================
# CircuitTwirler — persistence
# ===================================================================


class TestCircuitTwirlerPersistence:
    """Tests for to_dict / from_dict / save / load round-trip."""

    def test_to_dict_contains_exact_state(self, mock_client: MagicMock) -> None:
        """to_dict output contains exact rot_strings, mappings, and config."""
        qubits = ["QB1", "QB2"]
        pc = _make_pulse_circuit(qubits)
        cfg = TwirlingConfiguration(readout_twirl_strategy="MINIMAL", seed=7, circuit_twirling=False)
        twirler = CircuitTwirler(mock_client, config=cfg)
        twirler.twirl([pc])

        data = twirler.to_dict()

        # Config must match exactly
        assert data["config"] == {
            "readout_twirl_strategy": "MINIMAL",
            "seed": 7,
            "circuit_twirling": False,
            "num_twirling_instances": 20,
        }

        # Mappings must match the live state
        assert data["qubit_to_bit_mappings"] == [{"QB1": 0, "QB2": 1}]

        # Rot strings must match get_rot_strings() exactly
        assert data["rot_strings"] == twirler.get_rot_strings()
        # Verify actual content
        assert len(data["rot_strings"]) == 1
        assert len(data["rot_strings"][0]) == 2
        for rs in data["rot_strings"][0]:
            assert len(rs) == len(qubits)
            assert set(rs) <= {"I", "X"}

    def test_from_dict_restores_exact_state(self, mock_client: MagicMock) -> None:
        """from_dict restores config and qubit_to_bit_mappings exactly."""
        qubits = ["QB1", "QB2"]
        pc = _make_pulse_circuit(qubits)
        cfg = TwirlingConfiguration(
            readout_twirl_strategy="HADAMARD",
            seed=99,
            circuit_twirling=False,
        )
        twirler = CircuitTwirler(mock_client, config=cfg)
        twirler.twirl([pc])

        data = twirler.to_dict()
        restored = CircuitTwirler.from_dict(data, client=mock_client)

        # Config round-trip must be exact
        assert restored._config.readout_twirl_strategy == "HADAMARD"
        assert restored._config.seed == 99
        assert restored._config.circuit_twirling is False

        # Mappings must be identical
        assert restored._qubit_to_bit_mappings == [{"QB1": 0, "QB2": 1}]

    def test_save_and_load_round_trip(self, mock_client: MagicMock, tmp_path) -> None:
        """save_twirling_info / load_twirling_info round-trip preserves exact state."""
        qubits = ["QB1", "QB2"]
        pc = _make_pulse_circuit(qubits)
        cfg = TwirlingConfiguration(readout_twirl_strategy="MINIMAL", seed=42)
        twirler = CircuitTwirler(mock_client, config=cfg)
        twirler.twirl([pc])

        original_data = twirler.to_dict()
        filepath = str(tmp_path / "twirl_state.json")
        twirler.save_twirling_info(filepath)

        # Verify the file is valid JSON matching to_dict() exactly
        with open(filepath) as f:
            raw = json.load(f)
        assert raw == original_data

        # Load and verify full round-trip
        loaded = CircuitTwirler.load_twirling_info(filepath, client=mock_client)
        assert loaded._config.readout_twirl_strategy == "MINIMAL"
        assert loaded._config.seed == 42
        assert loaded._config.circuit_twirling is True
        assert loaded._qubit_to_bit_mappings == [{"QB1": 0, "QB2": 1}]


# ===================================================================
# CircuitTwirler — submit and retrieve (mocked client)
# ===================================================================


class TestCircuitTwirlerSubmitRetrieve:
    """Tests for submit() and retrieve_counts() with a fully mocked client."""

    def _setup_submit_mocks(self, mock_client: MagicMock) -> MagicMock:
        """Wire up compiler, job, and results mocks on the client."""
        compiler = MagicMock()
        job_definition = MagicMock()
        context = {"shots": 0}
        compiler.compile.return_value = (job_definition, context)
        # get_settings returns a mock SettingNode with a set_shots method
        settings_mock = MagicMock()
        compiler.get_settings.return_value = settings_mock
        mock_client.get_standard_compiler.return_value = compiler

        job = MagicMock()
        mock_client.submit_playlist.return_value = job
        return job

    def test_submit_sets_job(self, mock_client: MagicMock) -> None:
        """submit() stores a job handle retrievable via get_job()."""
        job = self._setup_submit_mocks(mock_client)

        pc = _make_pulse_circuit(["QB1", "QB2"])
        cfg = TwirlingConfiguration(readout_twirl_strategy="MINIMAL", seed=0)
        twirler = CircuitTwirler(mock_client, config=cfg)
        result = twirler.twirl([pc]).submit(shots=1000)

        assert result is twirler  # method chaining
        assert twirler.get_job() is job
        mock_client.get_standard_compiler.assert_called_once()

    def test_submit_sets_shots_in_compiler_settings(self, mock_client: MagicMock) -> None:
        """submit() injects shots_per_circuit into the compiler settings before compilation.

        This ensures that playlist_repeats in the RunDefinition matches the
        intended per-circuit shot count, rather than falling back to the
        compiler default of 1000.
        """
        self._setup_submit_mocks(mock_client)
        compiler = mock_client.get_standard_compiler.return_value
        settings_mock = compiler.get_settings.return_value

        pc = _make_pulse_circuit(["QB1", "QB2"])
        # MINIMAL with circuit_twirling=False → 2 randomized circuits
        cfg = TwirlingConfiguration(readout_twirl_strategy="MINIMAL", seed=0, circuit_twirling=False)
        twirler = CircuitTwirler(mock_client, config=cfg)
        twirler.twirl([pc]).submit(shots=1000)

        # 1000 total / 2 circuits = 500 shots per circuit
        settings_mock.set_shots.assert_called_once_with(500)

        # The settings object must be passed to compile()
        compile_call_kwargs = compiler.compile.call_args.kwargs
        assert compile_call_kwargs.get("settings") is settings_mock

    def test_submit_distributes_shots_across_many_circuits(self, mock_client: MagicMock) -> None:
        """submit() correctly distributes a small shot budget across many twirled circuits."""
        self._setup_submit_mocks(mock_client)
        compiler = mock_client.get_standard_compiler.return_value
        settings_mock = compiler.get_settings.return_value

        pc = _make_pulse_circuit(["QB1", "QB2"])
        # circuit_twirling=True with 40 instances → 40 circuits (NONE strategy)
        cfg = TwirlingConfiguration(
            readout_twirl_strategy="NONE", seed=0, circuit_twirling=True, num_twirling_instances=40
        )
        twirler = CircuitTwirler(mock_client, config=cfg)
        twirler.twirl([pc]).submit(shots=1000)

        # 1000 total / 40 circuits = 25 shots per circuit
        settings_mock.set_shots.assert_called_once_with(25)

    def test_submit_shots_are_per_input_circuit(self, mock_client: MagicMock) -> None:
        """``shots`` is per input circuit, split across that circuit's twirled instances.

        With multiple input circuits, the per-instance shot count must not shrink as more
        input circuits are added: each input circuit independently receives ~``shots`` shots.
        """
        self._setup_submit_mocks(mock_client)
        compiler = mock_client.get_standard_compiler.return_value
        settings_mock = compiler.get_settings.return_value

        # Two input circuits, MINIMAL with circuit_twirling=False → 2 instances each (4 total).
        pcs = [_make_pulse_circuit(["QB1", "QB2"]), _make_pulse_circuit(["QB3", "QB4"])]
        cfg = TwirlingConfiguration(readout_twirl_strategy="MINIMAL", seed=0, circuit_twirling=False)
        twirler = CircuitTwirler(mock_client, config=cfg)
        twirler.twirl(pcs).submit(shots=1000)

        # 1000 per input circuit / 2 instances per circuit = 500 shots per circuit
        # (independent of the number of input circuits).
        settings_mock.set_shots.assert_called_once_with(500)

    @patch("iqm.error_reduction_tools.twirling.twirling_api.sweep_job_to_qiskit")
    @patch("iqm.error_reduction_tools.twirling.twirling_api.untwirl_and_sum_counts")
    def test_retrieve_counts_returns_one_dict_per_input(
        self,
        mock_untwirl: MagicMock,
        mock_sweep: MagicMock,
        mock_client: MagicMock,
    ) -> None:
        """retrieve_counts() returns exact untwirled dicts per input circuit."""
        job = self._setup_submit_mocks(mock_client)
        job.wait_for_completion.return_value = None

        raw_counts = {"00": 500, "11": 500}
        results_obj = MagicMock()
        results_obj.get_counts.return_value = raw_counts
        mock_sweep.return_value = results_obj

        untwirled_result = {"00": 0.5, "11": 0.5}
        mock_untwirl.return_value = untwirled_result

        pc1 = _make_pulse_circuit(["QB1", "QB2"])
        pc2 = _make_pulse_circuit(["QB3", "QB4"])
        cfg = TwirlingConfiguration(readout_twirl_strategy="MINIMAL", seed=0, circuit_twirling=False)
        twirler = CircuitTwirler(mock_client, config=cfg)
        counts = twirler.twirl([pc1, pc2]).submit(shots=2000).retrieve_counts()

        # One result per input circuit, each matching the untwirl return value
        assert len(counts) == 2
        assert counts[0] == untwirled_result
        assert counts[1] == untwirled_result

        # untwirl_and_sum_counts was called once per input circuit
        assert mock_untwirl.call_count == 2

        # Each call received 2 raw-count dicts (MINIMAL has 2 rot strings, circuit_twirling=False → 2 circuits)
        for call_args in mock_untwirl.call_args_list:
            raw_counts_list, rot_string_list = call_args.args
            assert len(raw_counts_list) == 2
            assert len(rot_string_list) == 2
            # Each raw count dict must be the one returned by results_obj
            for rc in raw_counts_list:
                assert rc == raw_counts
            # Each rot string must be a valid I/X string or None
            for rs in rot_string_list:
                assert rs is None or set(rs) <= {"I", "X"}

    @patch("iqm.error_reduction_tools.twirling.twirling_api.sleep")
    @patch("iqm.error_reduction_tools.twirling.twirling_api.sweep_job_to_qiskit")
    def test_retrieve_retries_on_waiting(
        self,
        mock_sweep: MagicMock,
        mock_sleep: MagicMock,
        mock_client: MagicMock,
    ) -> None:
        """retrieve_counts() retries when sweep raises ValueError with 'WAITING'."""
        job = self._setup_submit_mocks(mock_client)
        job.wait_for_completion.return_value = None

        # First call raises WAITING, second succeeds.
        results_obj = MagicMock()
        results_obj.get_counts.return_value = {"0": 1000}
        mock_sweep.side_effect = [ValueError("status is WAITING"), results_obj]

        pc = _make_pulse_circuit(["QB1"])
        cfg = TwirlingConfiguration(readout_twirl_strategy="MINIMAL", seed=0)
        twirler = CircuitTwirler(mock_client, config=cfg)

        with patch(
            "iqm.error_reduction_tools.twirling.twirling_api.untwirl_and_sum_counts",
            return_value={"0": 1.0},
        ):
            counts = twirler.twirl([pc]).submit(shots=1000).retrieve_counts()

        assert len(counts) == 1
        mock_sleep.assert_called_once_with(30.0)

    @patch("iqm.error_reduction_tools.twirling.twirling_api.sleep")
    @patch("iqm.error_reduction_tools.twirling.twirling_api.sweep_job_to_qiskit")
    def test_retrieve_raises_after_max_retries(
        self,
        mock_sweep: MagicMock,
        mock_sleep: MagicMock,
        mock_client: MagicMock,
    ) -> None:
        """retrieve_counts() re-raises ValueError on the final attempt."""
        job = self._setup_submit_mocks(mock_client)
        job.wait_for_completion.return_value = None
        mock_sweep.side_effect = ValueError("status is WAITING")

        pc = _make_pulse_circuit(["QB1"])
        cfg = TwirlingConfiguration(readout_twirl_strategy="MINIMAL", seed=0)
        twirler = CircuitTwirler(mock_client, config=cfg)
        twirler.twirl([pc]).submit(shots=1000)

        # On the 10th (last) attempt the WAITING ValueError is re-raised
        # because attempt == max_retries - 1.
        with pytest.raises(ValueError, match="WAITING"):
            twirler.retrieve_counts()

        # sleep is called for attempts 0..8 (9 times), not on the last one
        assert mock_sleep.call_count == 9

    @patch("iqm.error_reduction_tools.twirling.twirling_api.sweep_job_to_qiskit")
    def test_retrieve_raises_non_waiting_error_immediately(
        self,
        mock_sweep: MagicMock,
        mock_client: MagicMock,
    ) -> None:
        """retrieve_counts() re-raises ValueError that is not about WAITING."""
        job = self._setup_submit_mocks(mock_client)
        job.wait_for_completion.return_value = None
        mock_sweep.side_effect = ValueError("something else went wrong")

        pc = _make_pulse_circuit(["QB1"])
        cfg = TwirlingConfiguration(readout_twirl_strategy="MINIMAL", seed=0)
        twirler = CircuitTwirler(mock_client, config=cfg)
        twirler.twirl([pc]).submit(shots=1000)

        with pytest.raises(ValueError, match="something else"):
            twirler.retrieve_counts()


# ===================================================================
# CircuitTwirler — multi-frontend circuit conversion
# ===================================================================


class TestCircuitTwirlerMultiFrontend:
    """Tests for duck-typed to_qiskit() support in _convert_circuits."""

    def test_to_qiskit_object_is_accepted(self, mock_client: MagicMock) -> None:
        """A real Qrisp QuantumCircuit (which has to_qiskit()) is accepted."""
        qrisp = pytest.importorskip("qrisp", reason="qrisp is not installed")
        QrispQuantumCircuit = qrisp.QuantumCircuit

        qc = QrispQuantumCircuit(2)
        qc.rx(0.5, 0)
        qc.rx(0.5, 1)
        qc.measure(qc.qubits)

        cfg = TwirlingConfiguration(readout_twirl_strategy="MINIMAL", seed=0, circuit_twirling=False)
        twirler = CircuitTwirler(mock_client, config=cfg)
        twirler.twirl([qc])

        # The circuit was successfully twirled — rotation strings are available.
        rot_strings = twirler.get_rot_strings()
        assert len(rot_strings) == 1
        assert len(rot_strings[0]) == 2  # MINIMAL, circuit_twirling=False → 2 circuits

    def test_unsupported_circuit_type_raises(self, mock_client: MagicMock) -> None:
        """An unsupported circuit type raises TypeError."""
        mock_circuit = MagicMock()
        type(mock_circuit).__name__ = "BadFrontend"

        twirler = CircuitTwirler(mock_client)
        with pytest.raises(TypeError, match="Unsupported circuit type"):
            twirler.twirl([mock_circuit])

    def test_mixed_frontend_types(self, mock_client: MagicMock) -> None:
        """A mix of TwirledCircuit and a real Qrisp circuit works."""
        qrisp = pytest.importorskip("qrisp", reason="qrisp is not installed")
        QrispQuantumCircuit = qrisp.QuantumCircuit

        # A native pulse circuit.
        pc = _make_pulse_circuit(["QB1", "QB2"])

        # A real Qrisp circuit.
        qrisp_qc = QrispQuantumCircuit(2)
        qrisp_qc.rx(0.5, 0)
        qrisp_qc.rx(0.5, 1)
        qrisp_qc.measure(qrisp_qc.qubits)

        cfg = TwirlingConfiguration(readout_twirl_strategy="MINIMAL", seed=0, circuit_twirling=False)
        twirler = CircuitTwirler(mock_client, config=cfg)
        twirler.twirl([pc, qrisp_qc])

        rot_strings = twirler.get_rot_strings()
        assert len(rot_strings) == 2  # two input circuits


# ===================================================================
# CircuitTwirler — compilation_options (dict) passthrough
# ===================================================================


class TestCircuitTwirlerCompilationOptions:
    """Tests for compilation_options dict passthrough in CircuitTwirler."""

    def _setup_submit_mocks(self, mock_client: MagicMock) -> tuple[MagicMock, MagicMock]:
        """Wire up compiler and job mocks, return (compiler, job)."""
        compiler = MagicMock()
        job_definition = MagicMock()
        context = {"shots": 0}
        compiler.compile.return_value = (job_definition, context)
        mock_client.get_standard_compiler.return_value = compiler

        job = MagicMock()
        mock_client.submit_playlist.return_value = job
        return compiler, job

    def test_submit_without_options_passes_none_context(self, mock_client: MagicMock) -> None:
        """Without compilation_options, compile() is called with context=None."""
        compiler, _ = self._setup_submit_mocks(mock_client)
        pc = _make_pulse_circuit(["QB1", "QB2"])
        cfg = TwirlingConfiguration(readout_twirl_strategy="MINIMAL", seed=0)
        twirler = CircuitTwirler(mock_client, config=cfg)
        twirler.twirl([pc]).submit(shots=1000)

        call_kwargs = compiler.compile.call_args
        assert call_kwargs.kwargs.get("context") is None

    def test_submit_with_options_passes_context(self, mock_client: MagicMock) -> None:
        """compilation_options dict is forwarded as the compiler context."""
        compiler, _ = self._setup_submit_mocks(mock_client)
        dd_strategy = MagicMock(name="DDStrategy")
        pc = _make_pulse_circuit(["QB1", "QB2"])
        cfg = TwirlingConfiguration(readout_twirl_strategy="MINIMAL", seed=0)
        twirler = CircuitTwirler(
            mock_client,
            config=cfg,
            compilation_options={"DDStrategy": dd_strategy, "extra": 42},
        )
        twirler.twirl([pc]).submit(shots=1000)

        call_kwargs = compiler.compile.call_args
        compile_context = call_kwargs.kwargs.get("context")
        assert compile_context is not None
        assert compile_context["DDStrategy"] is dd_strategy
        assert compile_context["extra"] == 42


# ===================================================================
# CircuitTwirler — optional client
# ===================================================================


class TestCircuitTwirlerOptionalClient:
    """client is now optional; it is only required for LOCAL strategy and submit()."""

    def test_construction_without_client(self) -> None:
        """CircuitTwirler can be constructed without a client."""
        twirler = CircuitTwirler(config=TwirlingConfiguration(readout_twirl_strategy="MINIMAL"))
        assert twirler._client is None

    def test_minimal_twirl_without_client(self) -> None:
        """MINIMAL strategy does not need a client; twirl() succeeds."""
        pc = _make_pulse_circuit(["QB1", "QB2"])
        twirler = CircuitTwirler(
            config=TwirlingConfiguration(readout_twirl_strategy="MINIMAL", circuit_twirling=False, seed=0)
        )
        twirler.twirl([pc])
        assert len(twirler.get_rot_strings()[0]) == 2

    def test_hadamard_twirl_without_client(self) -> None:
        """HADAMARD strategy does not need a client; twirl() succeeds."""
        pc = _make_pulse_circuit(["QB1", "QB2"])
        twirler = CircuitTwirler(
            config=TwirlingConfiguration(readout_twirl_strategy="HADAMARD", circuit_twirling=False, seed=0)
        )
        twirler.twirl([pc])
        assert len(twirler.get_rot_strings()[0]) == 4

    def test_none_twirl_without_client(self) -> None:
        """NONE readout strategy with circuit_twirling=True does not need a client."""
        pc = _make_pulse_circuit(["QB1", "QB2"])
        twirler = CircuitTwirler(
            config=TwirlingConfiguration(readout_twirl_strategy="NONE", circuit_twirling=True, seed=0)
        )
        twirler.twirl([pc])
        assert len(twirler.get_rot_strings()[0]) > 0

    def test_local_without_client_raises_on_twirl(self) -> None:
        """LOCAL strategy requires a client; twirl() raises ValueError if none given."""
        pc = _make_pulse_circuit(["QB1", "QB2"])
        twirler = CircuitTwirler(config=TwirlingConfiguration(readout_twirl_strategy="LOCAL"))
        with pytest.raises(ValueError, match="'LOCAL' requires a client"):
            twirler.twirl([pc])

    def test_submit_without_client_raises(self) -> None:
        """submit() raises RuntimeError when no client is available."""
        pc = _make_pulse_circuit(["QB1", "QB2"])
        twirler = CircuitTwirler(
            config=TwirlingConfiguration(readout_twirl_strategy="MINIMAL", circuit_twirling=False, seed=0)
        )
        twirler.twirl([pc])
        with pytest.raises(RuntimeError, match="requires a client"):
            twirler.submit()

    def test_late_bind_client_at_submit(self, mock_client: MagicMock) -> None:
        """Passing client= to submit() when none was given at construction works."""
        compiler = MagicMock()
        compiler.compile.return_value = (MagicMock(), {"shots": 0})
        mock_client.get_standard_compiler.return_value = compiler
        mock_client.submit_playlist.return_value = MagicMock()

        pc = _make_pulse_circuit(["QB1", "QB2"])
        twirler = CircuitTwirler(
            config=TwirlingConfiguration(readout_twirl_strategy="MINIMAL", circuit_twirling=False, seed=0)
        )
        twirler.twirl([pc])
        twirler.submit(shots=100, client=mock_client)

        assert twirler._client is mock_client
        assert twirler.get_job() is not None


# ===================================================================
# CircuitTwirler — get_twirled_circuits() and get_twirled_circuits_flat()
# ===================================================================


class TestGetTwirledCircuits:
    """Tests for the get_twirled_circuits() and get_twirled_circuits_flat() accessors (with qiskit flag)."""

    def test_raises_before_twirl(self) -> None:
        """get_twirled_circuits() raises RuntimeError if twirl() has not been called."""
        twirler = CircuitTwirler(config=TwirlingConfiguration(readout_twirl_strategy="MINIMAL"))
        with pytest.raises(RuntimeError, match="Call twirl"):
            twirler.get_twirled_circuits()

    def test_nested_structure(self) -> None:
        """get_twirled_circuits() returns one inner list per input circuit."""
        pc1 = _make_pulse_circuit(["QB1", "QB2"])
        pc2 = _make_pulse_circuit(["QB3", "QB4"])
        twirler = CircuitTwirler(
            config=TwirlingConfiguration(readout_twirl_strategy="MINIMAL", circuit_twirling=False, seed=0)
        )
        twirler.twirl([pc1, pc2])

        nested = twirler.get_twirled_circuits()
        assert len(nested) == 2  # one group per input circuit
        assert len(nested[0]) == 2  # MINIMAL → 2 variants
        assert len(nested[1]) == 2

    def test_flat_structure(self) -> None:
        """get_twirled_circuits_flat() returns a single list in submit() order."""
        pc1 = _make_pulse_circuit(["QB1", "QB2"])
        pc2 = _make_pulse_circuit(["QB3", "QB4"])
        twirler = CircuitTwirler(
            config=TwirlingConfiguration(readout_twirl_strategy="MINIMAL", circuit_twirling=False, seed=0)
        )
        twirler.twirl([pc1, pc2])

        flat = twirler.get_twirled_circuits_flat()
        # 2 circuits × 2 MINIMAL variants = 4
        assert len(flat) == 4
        assert all(isinstance(c, PulseCircuit) for c in flat)

    def test_nested_length_matches_rot_strings(self) -> None:
        """Each inner group has the same length as the corresponding rot-strings group."""
        pc = _make_pulse_circuit(["QB1", "QB2", "QB3"])
        twirler = CircuitTwirler(
            config=TwirlingConfiguration(readout_twirl_strategy="HADAMARD", circuit_twirling=False, seed=0)
        )
        twirler.twirl([pc])

        nested = twirler.get_twirled_circuits()
        rot_strings = twirler.get_rot_strings()
        assert len(nested[0]) == len(rot_strings[0])

    def test_returns_shallow_copies(self) -> None:
        """Modifying the returned list does not mutate internal state."""
        pc = _make_pulse_circuit(["QB1", "QB2"])
        twirler = CircuitTwirler(
            config=TwirlingConfiguration(readout_twirl_strategy="MINIMAL", circuit_twirling=False, seed=0)
        )
        twirler.twirl([pc])

        variants = twirler.get_twirled_circuits()
        original_len = len(variants[0])
        variants[0].clear()  # mutate the returned copy

        # Internal state must be intact
        assert len(twirler._randomized_circuits_per_input[0]) == original_len

    def test_qiskit_flag_raises_before_twirl(self) -> None:
        """get_twirled_circuits(return_qiskit=True) raises RuntimeError if twirl() has not been called."""
        twirler = CircuitTwirler(config=TwirlingConfiguration(readout_twirl_strategy="MINIMAL"))
        with pytest.raises(RuntimeError, match="Call twirl"):
            twirler.get_twirled_circuits(return_qiskit=True)

    def test_qiskit_flag_nested_structure(self) -> None:
        """get_twirled_circuits(return_qiskit=True) returns Qiskit circuits with correct grouping."""
        qc1 = _make_qiskit_circuit(["QB1", "QB2"])
        qc2 = _make_qiskit_circuit(["QB3", "QB4"])
        twirler = CircuitTwirler(
            config=TwirlingConfiguration(readout_twirl_strategy="MINIMAL", circuit_twirling=False, seed=0)
        )
        twirler.twirl([qc1, qc2])

        nested = twirler.get_twirled_circuits(return_qiskit=True)
        assert len(nested) == 2
        assert len(nested[0]) == 2
        assert all(isinstance(c, QiskitQuantumCircuit) for group in nested for c in group)

    def test_qiskit_flag_flat_structure(self) -> None:
        """get_twirled_circuits_flat(return_qiskit=True) returns all variants as Qiskit circuits."""
        qc1 = _make_qiskit_circuit(["QB1", "QB2"])
        qc2 = _make_qiskit_circuit(["QB3", "QB4"])
        twirler = CircuitTwirler(
            config=TwirlingConfiguration(readout_twirl_strategy="MINIMAL", circuit_twirling=False, seed=0)
        )
        twirler.twirl([qc1, qc2])

        flat = twirler.get_twirled_circuits_flat(return_qiskit=True)
        assert len(flat) == 4
        assert all(isinstance(c, QiskitQuantumCircuit) for c in flat)


# ===================================================================
# CircuitTwirler — submit(client=) swap rules
# ===================================================================


class TestSubmitClientSwap:
    """Tests for the client-swap rules in submit()."""

    def _setup_client(self, client: MagicMock) -> None:
        """Wire up a minimal compiler + job mock on *client*."""
        compiler = MagicMock()
        compiler.compile.return_value = (MagicMock(), {"shots": 0})
        client.get_standard_compiler.return_value = compiler
        client.submit_playlist.return_value = MagicMock()

    def test_same_client_rebound_is_ok(self, mock_client: MagicMock) -> None:
        """Passing the same client object to submit() is always allowed."""
        self._setup_client(mock_client)
        pc = _make_pulse_circuit(["QB1", "QB2"])
        twirler = CircuitTwirler(
            mock_client,
            config=TwirlingConfiguration(readout_twirl_strategy="MINIMAL", circuit_twirling=False, seed=0),
        )
        twirler.twirl([pc])
        twirler.submit(shots=10, client=mock_client)

    def test_local_strategy_swap_raises(self, mock_client: MagicMock) -> None:
        """Swapping the client after LOCAL twirl() raises ValueError."""
        self._setup_client(mock_client)
        other_client = MagicMock()
        self._setup_client(other_client)

        topo = MagicMock()
        topo.name = "apollo"
        pc = _make_pulse_circuit(["QB1", "QB2"])

        with (
            patch(
                "iqm.error_reduction_tools.twirling.twirling_api.topology_from_qc",
                return_value=topo,
            ),
            patch(
                "iqm.error_reduction_tools.twirling.twirling_api.uses_move_gates",
                return_value=False,
            ),
        ):
            twirler = CircuitTwirler(
                mock_client,
                config=TwirlingConfiguration(readout_twirl_strategy="LOCAL", circuit_twirling=False, seed=0),
            )
            twirler.twirl([pc])

            with pytest.raises(ValueError, match="'LOCAL' strategy"):
                twirler.submit(shots=10, client=other_client)

    def test_minimal_strategy_swap_is_allowed(self, mock_client: MagicMock) -> None:
        """Swapping the client after MINIMAL twirl() is permitted (no topology dependency)."""
        self._setup_client(mock_client)
        other_client = MagicMock()
        self._setup_client(other_client)

        pc = _make_pulse_circuit(["QB1", "QB2"])
        twirler = CircuitTwirler(
            mock_client,
            config=TwirlingConfiguration(readout_twirl_strategy="MINIMAL", circuit_twirling=False, seed=0),
        )
        twirler.twirl([pc])
        twirler.submit(shots=10, client=other_client)

        assert twirler._client is other_client

    def test_twirl_strategy_recorded(self, mock_client: MagicMock) -> None:
        """_twirl_strategy is set to the strategy used at twirl() time."""
        topo = MagicMock()
        topo.name = "apollo"
        pc = _make_pulse_circuit(["QB1", "QB2"])

        with (
            patch(
                "iqm.error_reduction_tools.twirling.twirling_api.topology_from_qc",
                return_value=topo,
            ),
            patch(
                "iqm.error_reduction_tools.twirling.twirling_api.uses_move_gates",
                return_value=False,
            ),
        ):
            for strategy in ("LOCAL", "MINIMAL", "HADAMARD"):
                t = CircuitTwirler(
                    mock_client,
                    config=TwirlingConfiguration(readout_twirl_strategy=strategy, circuit_twirling=False, seed=0),
                )
                t.twirl([pc])
                assert t._twirl_strategy == strategy, f"Expected '{strategy}', got '{t._twirl_strategy}'"


# ===================================================================
# WorkflowConfiguration — defaults
# ===================================================================


class TestWorkflowConfigurationDefaults:
    """WorkflowConfiguration defaults match the intended REM-only behaviour."""

    def test_default_circuit_twirling_is_off(self) -> None:
        """Default WorkflowConfiguration has circuit_twirling=False (REM only)."""

    def test_default_readout_strategy_is_local(self) -> None:
        """Default WorkflowConfiguration still uses LOCAL readout twirling."""

    def test_twirling_configuration_own_default_unchanged(self) -> None:
        """TwirlingConfiguration's own default still has circuit_twirling=True."""
        assert TwirlingConfiguration().circuit_twirling is True


# ===================================================================
# Helpers — circuit factories for each frontend
# ===================================================================


def _make_qiskit_circuit(qubits: list[str]) -> QiskitQuantumCircuit:
    """Build a Qiskit QuantumCircuit with PRX-equivalent gates and measurements."""
    n = len(qubits)
    qc = QiskitQuantumCircuit(n, n)
    for i in range(n):
        qc.rx(0.5, i)
    for i in range(n):
        qc.measure(i, i)
    return qc


def _make_immutable_pulse_circuit(qubits: list[str]) -> PulseCircuit:
    """Build an immutable iqm.pulse.Circuit (the Pulla-native type)."""
    ops: list[CircuitOperation] = []
    for i, qb in enumerate(qubits):
        ops.append(CircuitOperation(name="prx", locus=(qb,), args={"angle": 0.5, "phase": 0.0}))
    for i, qb in enumerate(qubits):
        ops.append(CircuitOperation(name="measure", locus=(qb,), args={"key": f"m_{i}"}))
    return PulseCircuit(name="test", instructions=tuple(ops))


def _make_qrisp_circuit(num_qubits: int) -> object:
    """Build a real Qrisp QuantumCircuit with gates and measurements."""

    qc = QrispQuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.rx(0.5, i)
    qc.measure(qc.qubits)
    return qc


# ===================================================================
# CircuitTwirler — acceptance of all three frontend types
# ===================================================================


class TestCircuitTwirlerAcceptsAllFrontends:
    """Verify that CircuitTwirler.twirl() works with Qiskit, Pulla, and Qrisp circuits."""

    def test_qiskit_circuit_accepted(self, mock_client: MagicMock) -> None:
        """A plain Qiskit QuantumCircuit is accepted and twirled."""
        qc = _make_qiskit_circuit(["QB1", "QB2"])
        cfg = TwirlingConfiguration(readout_twirl_strategy="MINIMAL", seed=0, circuit_twirling=False)
        twirler = CircuitTwirler(mock_client, config=cfg)
        twirler.twirl([qc])

        rot_strings = twirler.get_rot_strings()
        assert len(rot_strings) == 1
        assert len(rot_strings[0]) == 2  # MINIMAL, circuit_twirling=False → 2 circuits

    def test_pulse_circuit_accepted(self, mock_client: MagicMock) -> None:
        """An immutable iqm.pulse.Circuit (Pulla native) is accepted and twirled."""
        pc = _make_immutable_pulse_circuit(["QB1", "QB2"])
        cfg = TwirlingConfiguration(readout_twirl_strategy="MINIMAL", seed=0, circuit_twirling=False)
        twirler = CircuitTwirler(mock_client, config=cfg)
        twirler.twirl([pc])

        rot_strings = twirler.get_rot_strings()
        assert len(rot_strings) == 1
        assert len(rot_strings[0]) == 2

    def test_qrisp_circuit_accepted(self, mock_client: MagicMock) -> None:
        """A real Qrisp QuantumCircuit is accepted and twirled."""
        qrisp_circ = _make_qrisp_circuit(2)
        cfg = TwirlingConfiguration(readout_twirl_strategy="MINIMAL", seed=0, circuit_twirling=False)
        twirler = CircuitTwirler(mock_client, config=cfg)
        twirler.twirl([qrisp_circ])

        rot_strings = twirler.get_rot_strings()
        assert len(rot_strings) == 1
        assert len(rot_strings[0]) == 2

    def test_mixed_all_three_frontends(self, mock_client: MagicMock) -> None:
        """A heterogeneous list of Qiskit + Pulla + Qrisp circuits is accepted."""
        qiskit_circ = _make_qiskit_circuit(["QB1", "QB2"])
        pulse_circ = _make_immutable_pulse_circuit(["QB3", "QB4"])
        qrisp_circ = _make_qrisp_circuit(2)

        cfg = TwirlingConfiguration(readout_twirl_strategy="MINIMAL", seed=0, circuit_twirling=False)
        twirler = CircuitTwirler(mock_client, config=cfg)
        twirler.twirl([qiskit_circ, pulse_circ, qrisp_circ])

        rot_strings = twirler.get_rot_strings()
        assert len(rot_strings) == 3  # one group per input circuit
        for group in rot_strings:
            assert len(group) == 2  # MINIMAL, circuit_twirling=False → 2 circuits each

    def test_twirled_pulse_circuit_still_accepted(self, mock_client: MagicMock) -> None:
        """The internal TwirledCircuit type is still accepted (regression check)."""
        pc = _make_pulse_circuit(["QB1", "QB2"])
        cfg = TwirlingConfiguration(readout_twirl_strategy="MINIMAL", seed=0, circuit_twirling=False)
        twirler = CircuitTwirler(mock_client, config=cfg)
        twirler.twirl([pc])

        rot_strings = twirler.get_rot_strings()
        assert len(rot_strings) == 1
        assert len(rot_strings[0]) == 2
