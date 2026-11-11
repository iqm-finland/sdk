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

from dataclasses import dataclass, replace
from itertools import pairwise
import warnings

from iqm.error_reduction_tools.circuits import CircuitVariant, TargetCircuit
from iqm.error_reduction_tools.observables import ObservableSet
from iqm.error_reduction_tools.plan import DEFAULT_SHOTS, build_plan
from iqm.error_reduction_tools.readout_characterization.readout_error_characterization import RECConfiguration
from iqm.error_reduction_tools.readout_characterization.topologies import QPUTopology
from iqm.error_reduction_tools.twirling.twirling_api import TwirlingConfiguration
import numpy as np
import pytest

from .builders import bell_circuit, circuit, prx


@dataclass(frozen=True)
class Copies:
    """Technique that repeats the target. ``None`` leaves a variant's budget to the workflow."""

    shots: tuple[int | None, ...]
    num_instances: int | None = None

    def build_variants(self, target, observables):
        return tuple(
            CircuitVariant(
                target_index=target.target_index,
                variant_index=index,
                circuit=target.circuit,
                shots=variant_shots,
                num_instances=self.num_instances,
            )
            for index, variant_shots in enumerate(self.shots)
        )


@dataclass(frozen=True)
class Broken:
    """Technique whose variants break one of the rules the plan checks."""

    fault: str

    def build_variants(self, target, observables):
        other = circuit([prx("QB1", 0.1)], ["QB2", "QB1"])
        first = CircuitVariant(
            target_index=target.target_index,
            variant_index=0,
            circuit=other if self.fault == "variant 0" else target.circuit,
        )
        second = replace(first, variant_index=5 if self.fault == "index" else 1)
        if self.fault == "measurement":
            second = replace(second, circuit=other)
        return (first, second)


def _targets(count=2):
    return [TargetCircuit(target_index=index, circuit=bell_circuit()) for index in range(count)]


OBSERVABLES = ObservableSet.build("ZZ")


def _plan(technique, *, targets=None, twirling=None, rem=None, topology=None, shots=None):
    return build_plan(
        targets or _targets(),
        OBSERVABLES,
        technique,
        shots=shots,
        twirling=twirling,
        rem=rem,
        topology=topology,
    )


class TestBatches:
    def test_every_instance_is_in_exactly_one_batch_grouped_by_shots(self):
        """Test that every instance is in exactly one batch, and batches are grouped by shots per instance."""
        twirling = TwirlingConfiguration(readout_twirl_strategy="MINIMAL", num_twirling_instances=4)
        plan = _plan(Copies(shots=(400, 40, 40)), twirling=twirling)
        keys = [key for batch in plan.batches for key in batch.keys]
        tree = [instance.key for target in plan.targets for instance in target.instances()]
        assert sorted(keys) == sorted(tree)
        assert len(set(keys)) == len(keys) == 2 * 3 * 4
        assert [(batch.shots, len(batch.keys)) for batch in plan.batches] == [(100, 8), (10, 16)]

    def test_cost_counts_calibration(self):
        """Test that the number of circuits, the total shots and the summary include the calibration job."""
        rem = RECConfiguration(num_circuits=10, shots=1000, seed=0)
        plan = _plan(Copies(shots=(30,)), rem=rem)
        assert plan.num_circuits == 2 + 10
        assert plan.total_shots == 2 * 30 + 10 * 100
        assert "readout calibration: 10 circuits x 100 shots on 2 qubits" in plan.summary()


class TestTwirling:
    def test_without_twirling_each_variant_runs_once_unflipped(self):
        """Test that without twirling each variant runs once, unchanged and without readout flips."""
        plan = _plan(Copies(shots=(10, 20)))
        instances = list(plan.targets[1].instances())
        assert [instance.rot_string for instance in instances] == ["II", "II"]
        assert [instance.circuit.instructions for instance in instances] == [bell_circuit().instructions] * 2
        assert not plan.readout_twirled

    @pytest.mark.parametrize(("requested", "expected"), [(2, 2), (6, 6), (5, 6)])
    def test_minimal_strategy_balances_both_rotations(self, requested, expected):
        """Test that the MINIMAL strategy flips every classical bit in half of the instances."""
        twirling = TwirlingConfiguration(readout_twirl_strategy="MINIMAL", num_twirling_instances=requested)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            plan = _plan(Copies(shots=(60,)), targets=_targets(1), twirling=twirling)
        assert len(caught) == requested % 2
        rot_strings = [instance.rot_string for instance in plan.targets[0].instances()]
        assert len(rot_strings) == expected
        # Two complementary rotations, so every classical bit is flipped in exactly half of the instances.
        assert len(set(rot_strings)) == 2
        for clbit in range(2):
            assert sum(rot[clbit] == "X" for rot in rot_strings) == expected // 2
        assert plan.readout_twirled

    def test_local_strategy_flips_every_clbit_in_half_of_the_instances(self):
        """Test that the LOCAL strategy flips every classical bit in half of the instances."""
        twirling = TwirlingConfiguration(readout_twirl_strategy="LOCAL", num_twirling_instances=8, seed=0)
        topology = QPUTopology("line", 2, couplers=[("QB1", "QB2")])
        plan = _plan(Copies(shots=(80,)), targets=_targets(1), twirling=twirling, topology=topology)
        rot_strings = [instance.rot_string for instance in plan.targets[0].instances()]
        assert len(rot_strings) == 8
        assert sorted(set(rot_strings)) == ["II", "IX", "XI", "XX"]
        for clbit in range(2):
            assert sum(rot[clbit] == "X" for rot in rot_strings) == 4
        assert plan.readout_twirled

    def test_same_seed_gives_identical_instances(self):
        """Test that the same seed gives the same twirling instances."""
        twirling = TwirlingConfiguration(readout_twirl_strategy="HADAMARD", num_twirling_instances=4, seed=3)
        first, second = (_plan(Copies(shots=(10,)), twirling=twirling) for _ in range(2))
        assert first.targets == second.targets

    def test_local_strategy_without_topology_is_rejected(self):
        """Test that the LOCAL strategy without a QPU topology is rejected."""
        with pytest.raises(ValueError, match="needs the QPU topology"):
            _plan(Copies(shots=(10,)), twirling=TwirlingConfiguration(readout_twirl_strategy="LOCAL"))

    def test_instances_requested_without_twirling_warn(self):
        """Test that requesting instances without twirling warns and runs the variant once."""
        with pytest.warns(UserWarning, match="no twirling is configured"):
            plan = _plan(Copies(shots=(10,), num_instances=3), targets=_targets(1))
        assert len(plan.targets[0].variants()[0].instances()) == 1


class TestBudgets:
    @pytest.mark.parametrize(("shots", "expected"), [(None, DEFAULT_SHOTS), (500, 500), (7, 7)])
    def test_unset_budgets_are_filled_in_from_the_workflow(self, shots, expected):
        """Test that unset shots and instances are filled in from the workflow, without warnings."""
        twirling = TwirlingConfiguration(readout_twirl_strategy="NONE", num_twirling_instances=3)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            plan = _plan(Copies(shots=(None, None)), twirling=twirling, shots=shots)
        variants = [variant for target in plan.targets for variant in target.variants()]
        assert [(variant.shots, variant.num_instances) for variant in variants] == [(expected, 3)] * 4
        assert all(len(variant.instances()) == 3 for variant in variants)

    def test_without_twirling_a_variant_gets_one_instance(self):
        """Test that without twirling a variant gets one instance."""
        plan = _plan(Copies(shots=(None,)), targets=_targets(1))
        assert plan.targets[0].variants()[0].num_instances == 1

    @pytest.mark.parametrize(("workflow_shots", "warns"), [(None, False), (900, False), (500, True)])
    def test_own_shots_warn_only_when_overriding_an_explicit_workflow_value(self, workflow_shots, warns):
        """Test that a variant's own shots warn only when they override shots set by the user."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            plan = _plan(Copies(shots=(900, None)), targets=_targets(1), shots=workflow_shots)
        messages = [str(warning.message) for warning in caught]
        assert len(messages) == int(warns)
        if warns:
            assert "1 variant(s) set their own shots [900], which override shots=500" in messages[0]
        assert [variant.shots for variant in plan.targets[0].variants()] == [900, workflow_shots or DEFAULT_SHOTS]

    def test_own_instances_warn_once_when_overriding_the_twirling_configuration(self):
        """Test that variants overriding the twirling instances give one warning."""
        twirling = TwirlingConfiguration(readout_twirl_strategy="NONE", num_twirling_instances=4)
        with pytest.warns(UserWarning, match=r"2 variant\(s\) set their own num_twirling_instances \[2\]") as record:
            plan = _plan(Copies(shots=(None,), num_instances=2), twirling=twirling)
        assert len(record) == 1
        assert all(len(target.variants()[0].instances()) == 2 for target in plan.targets)


class TestChecks:
    @pytest.mark.parametrize(
        ("fault", "message"),
        [("variant 0", "must be the target circuit"), ("index", "is labelled"), ("measurement", "measures")],
    )
    def test_inconsistent_variants_are_rejected(self, fault, message):
        """Test that variants with a wrong variant 0, index or measurement map are rejected."""
        with pytest.raises(ValueError, match=message):
            _plan(Broken(fault))

    def test_calibration_must_cover_the_measured_qubits(self):
        """Test that a calibration missing a measured qubit is rejected."""
        with pytest.raises(ValueError, match=r"\['QB2'\]"):
            _plan(Copies(shots=(10,)), rem=RECConfiguration(qubits=["QB1"], num_circuits=2, shots=10))

    def test_calibration_shots_must_cover_every_circuit(self):
        """Test that fewer calibration shots than circuits are rejected."""
        with pytest.raises(ValueError, match="5 calibration shots cannot be split over 10 circuits"):
            _plan(Copies(shots=(10,)), rem=RECConfiguration(num_circuits=10, shots=5))

    def test_calibration_prepares_both_states_on_every_qubit(self):
        """Test that the calibration prepares each string followed by its complement."""
        plan = _plan(Copies(shots=(10,)), rem=RECConfiguration(num_circuits=6, shots=60, seed=3))
        calibration = plan.calibration
        assert calibration.qubits == ("QB1", "QB2")
        assert len(calibration.circuits) == len(calibration.prep_strings) == 6
        # Symmetrized preparation: every string is followed by its complement.
        for first, second in list(pairwise(calibration.prep_strings))[::2]:
            assert {a + b for a, b in zip(first, second)} <= {"IX", "XI"}
        assert np.all([len(prep) == 2 for prep in calibration.prep_strings])
