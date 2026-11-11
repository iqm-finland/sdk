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
Tests for CDR (pulse-native) training-set construction and anti-clustering.
"""

from iqm.error_reduction_tools.cdr.cdr_clifford_utils import get_clifford_r_library
from iqm.error_reduction_tools.cdr.cdr_training_circuits import validate_training_circuit_is_clifford
from iqm.error_reduction_tools.cdr.cdr_training_set import (
    TrainingSet,
    build_clifford_circuit_from_assignment,
    farthest_point_selection,
    generate_clifford_training_set,
    load_training_set,
    non_clifford_slot_indices,
    save_training_set,
    training_set_spread,
)
import numpy as np
import pytest

from iqm.pulse import Circuit, CircuitOperation


def target_with_slots():
    """Construct a two-qubit circuit with one Clifford prx and three non-Clifford prx slots."""
    ops = (
        CircuitOperation(name="prx", locus=("QB1",), args={"angle": np.pi / 2, "phase": 0.0}),  # Clifford
        CircuitOperation(name="prx", locus=("QB2",), args={"angle": 0.3, "phase": 0.0}),  # slot 0
        CircuitOperation(name="cz", locus=("QB1", "QB2")),
        CircuitOperation(name="prx", locus=("QB1",), args={"angle": 0.7, "phase": 0.1}),  # slot 1
        CircuitOperation(name="prx", locus=("QB2",), args={"angle": 1.1, "phase": 0.2}),  # slot 2
        CircuitOperation(name="measure", locus=("QB1",), args={"key": "m0"}),
        CircuitOperation(name="measure", locus=("QB2",), args={"key": "m1"}),
    )
    return Circuit(name="native_target_with_slots", instructions=ops)


def clifford_target():
    """Construct a two-qubit circuit where every prx gate is already Clifford (no slots)."""
    ops = (
        CircuitOperation(name="prx", locus=("QB1",), args={"angle": np.pi / 2, "phase": 0.0}),
        CircuitOperation(name="cz", locus=("QB1", "QB2")),
        CircuitOperation(name="prx", locus=("QB2",), args={"angle": np.pi, "phase": 0.0}),
        CircuitOperation(name="measure", locus=("QB1",), args={"key": "m0"}),
        CircuitOperation(name="measure", locus=("QB2",), args={"key": "m1"}),
    )
    return Circuit(name="native_clifford_target", instructions=ops)


def generate(circuit, observables, num=3, **overrides):
    """generate_clifford_training_set with the run-on-hardware options turned off."""
    options = {
        "num_training_circuits": num,
        "pool_size": 40,
        "rng_seed": 0,
        "progress": False,
        "build_stim_circuits": False,
        "validate_selected": False,
    }
    options.update(overrides)
    return generate_clifford_training_set(circuit, observables=observables, **options)


def test_only_the_non_clifford_prx_gates_are_slots():
    """Test that only non-Clifford prx gates are identified as slots."""
    circuit = target_with_slots()
    slots = non_clifford_slot_indices(circuit)
    assert slots == [1, 3, 4]


def test_a_fully_clifford_circuit_has_no_slots():
    """Test that a fully Clifford circuit has no slots."""
    assert non_clifford_slot_indices(clifford_target()) == []


def test_built_circuit_is_clifford():
    """Test that a circuit built from a Clifford assignment is itself Clifford."""
    circuit = target_with_slots()
    slots = non_clifford_slot_indices(circuit)
    library = get_clifford_r_library(identity_as_delay=False)
    training_circuit = build_clifford_circuit_from_assignment(circuit, slots, tuple(0 for _ in slots), library)
    assert validate_training_circuit_is_clifford(training_circuit)


def test_the_identity_library_member_becomes_a_delay():
    """Test that the identity library member becomes a delay operation."""
    circuit = target_with_slots()
    slots = non_clifford_slot_indices(circuit)
    library = get_clifford_r_library(identity_as_delay=True)
    training_circuit = build_clifford_circuit_from_assignment(circuit, slots, tuple(0 for _ in slots), library)
    assert any(op.name == "delay" for op in training_circuit.instructions)


def test_assignment_length_must_match_the_slots():
    """Test that a mismatched assignment length raises ValueError."""
    circuit = target_with_slots()
    slots = non_clifford_slot_indices(circuit)
    library = get_clifford_r_library(identity_as_delay=False)
    with pytest.raises(ValueError, match="equal length"):
        build_clifford_circuit_from_assignment(circuit, slots, assignment=(0,), clifford_library=library)


def test_it_selects_the_requested_number_of_distinct_rows():
    """Test that farthest_point_selection selects the requested number of distinct rows."""
    vectors = np.random.default_rng(0).normal(size=(50, 3))
    chosen = farthest_point_selection(vectors, num_select=10)
    assert len(chosen) == len(set(chosen)) == 10
    assert all(0 <= i < 50 for i in chosen)


def test_asking_for_everything_returns_every_row_once():
    """Test that requesting num_select == n returns all n rows, with no duplicates."""
    vectors = np.arange(10, dtype=float).reshape(-1, 1)
    assert farthest_point_selection(vectors, num_select=10) == list(range(10))


def test_asking_for_more_than_available_returns_every_row_once():
    """Test that requesting num_select > n still returns only the n available rows, with no duplicates."""
    vectors = np.arange(10, dtype=float).reshape(-1, 1)
    assert farthest_point_selection(vectors, num_select=20) == list(range(10))


def test_it_reaches_for_the_outliers_more_than_random_sampling_does():
    """Test that farthest_point_selection favors outliers more than random sampling does."""
    rng = np.random.default_rng(0)
    cluster = rng.normal(scale=0.01, size=(100, 1))
    outliers = np.array([[-5.0], [-3.0], [3.0], [5.0], [7.0]])
    vectors = np.vstack([cluster, outliers])
    fps_std = np.std(vectors[farthest_point_selection(vectors, num_select=5)])
    random_std = np.std(vectors[rng.choice(len(vectors), size=5, replace=False)])
    assert fps_std > random_std


def test_weights_increase_the_spread_of_the_combined_value():
    """Test that weights increase the spread of the combined weighted value."""
    vectors = np.array([[1.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [0.0, 0.0]])
    chosen = farthest_point_selection(vectors, num_select=2, weights=[1.0, 1.0])
    combined = vectors @ np.array([1.0, 1.0])
    assert set(chosen) == {int(np.argmax(combined)), int(np.argmin(combined))}


def test_a_random_start_still_selects_the_requested_count():
    """Test that a random start still selects the requested count."""
    vectors = np.random.default_rng(1).normal(size=(20, 2))
    chosen = farthest_point_selection(vectors, num_select=5, rng=np.random.default_rng(1), seed="random")
    assert len(chosen) == 5


def test_duplicate_noiseless_values_are_repeated_evenly():
    """Test that rows with repeated exact values fill the selection, spread evenly over the distinct values."""
    vectors = np.random.default_rng(0).choice([-1.0, 0.0, 1.0], size=(60, 1), p=[0.2, 0.6, 0.2])
    chosen = farthest_point_selection(vectors, num_select=10)
    assert len(chosen) == len(set(chosen)) == 10
    _, counts = np.unique(vectors[chosen], return_counts=True)
    assert sorted(counts) == [3, 3, 4]
    assert chosen[:3] == farthest_point_selection(vectors, num_select=10, allow_duplicate_noiseless_values=False)


def test_spread_reports_min_max_and_std():
    """Test that training_set_spread reports min, max, and std per observable."""
    spread = training_set_spread(np.array([[-1.0], [-0.5], [0.0], [0.5], [1.0]]))
    assert spread["num_circuits"] == 5
    assert spread["num_observables"] == 1
    assert np.isclose(spread["per_observable"]["min"][0], -1.0)
    assert np.isclose(spread["per_observable"]["max"][0], 1.0)
    assert spread["per_observable"]["std"][0] > 0


def test_training_set_spread_matches_the_free_function():
    """Test that TrainingSet.spread() matches training_set_spread on its exact_values."""
    ts = generate(target_with_slots(), ["IZ"], num=5)
    assert ts.spread() == training_set_spread(ts.exact_values)


def test_pool_fps_returns_clifford_circuits_with_exact_values():
    """Test that pool-based farthest-point selection returns Clifford circuits with exact values."""
    ts = generate(target_with_slots(), ["IZ"], num=3)
    assert isinstance(ts, TrainingSet)
    assert ts.num_training == 3
    assert ts.exact_values.shape == (3, 1)
    assert ts.exact_values.dtype == np.float64
    assert ts.num_observables == 1
    assert all(validate_training_circuit_is_clifford(c) for c in ts.pulse_circuits)


def test_clifford_exact_values_are_minus_one_zero_or_one():
    """Test that Clifford exact values are exactly -1, 0, or 1."""
    ts = generate(target_with_slots(), ["IZ"], num=3)
    assert np.all(np.isin(np.round(ts.exact_values, 9), [-1.0, 0.0, 1.0]))


def test_the_same_seed_gives_the_same_training_set():
    """Test that the same seed gives the same training set."""
    a = generate(target_with_slots(), ["IZ"], num=3, rng_seed=42)
    b = generate(target_with_slots(), ["IZ"], num=3, rng_seed=42)
    assert np.allclose(a.exact_values, b.exact_values)


def test_a_summed_observable_gets_a_readable_label():
    """Test that a summed observable gets a readable label."""
    ts = generate(target_with_slots(), [[[0.5, "IZ"], [0.5, "ZI"]]], num=2)
    assert "+" in ts.observable_labels[0]


def test_optimize_increases_the_spread_of_the_primary_term():
    """Test that the optimize strategy increases the spread of the primary observable's expectation values."""
    ts = generate(target_with_slots(), ["IZ"], num=5, strategy="optimize")
    assert ts.num_training == 5
    assert all(validate_training_circuit_is_clifford(c) for c in ts.pulse_circuits)
    assert len(np.unique(np.round(ts.exact_values[:, 0], 6))) >= 2


def test_optimize_with_coefficients_increases_the_spread_of_the_combined_value():
    """Test that the optimize strategy with coefficients increases the spread of the combined value."""
    ts = generate(target_with_slots(), ["IZ", "ZI"], num=5, strategy="optimize", coefficients=[0.7, 0.3])
    combined = ts.exact_values @ np.array([0.7, 0.3])
    assert combined.std() > 0
    assert len(np.unique(np.round(combined, 6))) >= 2


def test_a_clifford_target_warns_and_returns_a_single_circuit():
    """Test that a Clifford target warns and returns a single circuit."""
    with pytest.warns(UserWarning, match="already Clifford"):
        ts = generate(clifford_target(), ["ZZ"], num=4)
    assert ts.num_training == 1


def test_identity_delay_warns_when_identity_as_delay_is_false():
    """Test that generate_clifford_training_set warns when identity_delay has no effect."""
    with pytest.warns(UserWarning, match="identity_delay has no effect"):
        generate(target_with_slots(), ["IZ"], num=2, identity_as_delay=False, identity_delay=50e-9)


def test_an_unknown_strategy_is_rejected():
    """Test that an unknown strategy raises ValueError."""
    with pytest.raises(ValueError, match="Unknown strategy"):
        generate(target_with_slots(), ["IZ"], num=2, strategy="not_a_strategy")


def test_an_unsupported_instruction_is_rejected():
    """Test that an unsupported instruction is rejected."""
    circuit = Circuit(
        name="unsupported",
        instructions=(
            CircuitOperation(name="prx", locus=("QB1",), args={"angle": 0.3, "phase": 0.0}),  # a slot
            CircuitOperation(name="not_a_gate", locus=("QB1",), args={}),
            CircuitOperation(name="measure", locus=("QB1",), args={"key": "m0"}),
        ),
    )
    with pytest.raises(ValueError, match="not supported"):
        generate(circuit, ["Z"], num=2)


def test_it_round_trips_through_json(tmp_path):
    """Test that a TrainingSet round-trips through JSON, reproducing the same numbers."""
    ts = generate(target_with_slots(), ["IZ", "ZI"], num=3, build_stim_circuits=True)
    paths = save_training_set(ts, str(tmp_path / "training_set"))
    loaded = load_training_set(str(tmp_path / "training_set"))

    assert paths["json"] == str(tmp_path / "training_set.json")
    assert np.allclose(loaded.exact_values, ts.exact_values)
    assert loaded.observable_labels == ts.observable_labels
    assert loaded.metadata == ts.metadata
    assert [c.name for c in loaded.pulse_circuits] == [c.name for c in ts.pulse_circuits]
    assert [tuple(op.name for op in c.instructions) for c in loaded.pulse_circuits] == [
        tuple(op.name for op in c.instructions) for c in ts.pulse_circuits
    ]
    assert [str(s) for s in loaded.stim_circuits] == [str(s) for s in ts.stim_circuits]
    assert loaded.target_circuit.name == ts.target_circuit.name


def test_load_can_skip_the_target_circuit(tmp_path):
    """Test that load_training_set with load_target=False omits the target circuit."""
    ts = generate(target_with_slots(), ["IZ"], num=2)
    save_training_set(ts, str(tmp_path / "training_set"))
    loaded = load_training_set(str(tmp_path / "training_set"), load_target=False)
    assert loaded.target_circuit is None
