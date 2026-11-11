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

from dataclasses import dataclass

from iqm.error_reduction_tools.circuits import CircuitVariant, TargetCircuit
from iqm.error_reduction_tools.estimation import measure_term_values, resample_counts, variant_counts
from iqm.error_reduction_tools.observables import ObservableSet
from iqm.error_reduction_tools.plan import build_plan
from iqm.error_reduction_tools.readout_characterization.topologies import QPUTopology
from iqm.error_reduction_tools.rem.rem_processors import ReadoutErrorMitigation
from iqm.error_reduction_tools.twirling.twirling_api import TwirlingConfiguration
import numpy as np
import pytest

from .builders import circuit, prx, rotated_circuit
from .exact import exact_counts


@dataclass(frozen=True)
class Identity:
    def build_variants(self, target, observables):
        return (CircuitVariant(target_index=target.target_index, variant_index=0, circuit=target.circuit),)


def _planned(target_circuit, observables, strategy, instances=4, shots=4000, topology=None):
    twirling = TwirlingConfiguration(readout_twirl_strategy=strategy, num_twirling_instances=instances, seed=7)
    plan = build_plan(
        [TargetCircuit(target_index=0, circuit=target_circuit)],
        observables,
        Identity(),
        shots=shots,
        twirling=twirling,
        rem=None,
        topology=topology,
    )
    return plan


def _simulate(plan, readout=0.0):
    shots = plan.targets[0].variants()[0].shots_per_instance
    return {
        instance.key: exact_counts(instance.circuit, shots, readout=readout) for instance in plan.targets[0].instances()
    }


class TestUntwirling:
    @pytest.mark.parametrize(
        ("strategy", "topology"),
        [("HADAMARD", None), ("LOCAL", QPUTopology("line", 2, couplers=[("QB1", "QB2")]))],
    )
    @pytest.mark.parametrize(
        ("flipped", "expected"),
        [(["QB1"], "01"), (["QB2"], "10"), (["QB1", "QB2"], "11")],
    )
    def test_readout_twirled_deterministic_state_is_restored(self, flipped, expected, strategy, topology):
        """Test that untwirling the readout-twirled instances restores the prepared basis state."""
        target = circuit([prx(qubit, np.pi) for qubit in flipped], ["QB1", "QB2"])
        plan = _planned(target, ObservableSet.build("ZZ"), strategy, topology=topology)
        assert plan.readout_twirled
        counts = _simulate(plan)
        # Each instance alone reads a different outcome, the untwirled sum reads the prepared one.
        assert len({next(iter(c)) for c in counts.values()}) > 1
        assert variant_counts(plan.targets[0].variants()[0], counts) == {expected: 4000}

    def test_missing_instance_counts_are_reported(self):
        """Test that missing counts of an instance raise."""
        plan = _planned(circuit([], ["QB1"]), ObservableSet.build("Z"), "MINIMAL")
        counts = _simulate(plan)
        dropped = next(iter(counts))
        del counts[dropped]
        with pytest.raises(KeyError, match="No counts"):
            variant_counts(plan.targets[0].variants()[0], counts)


class TestReadoutMitigation:
    @pytest.mark.parametrize("theta", [0.4, 1.0, 2.2])
    def test_mitigation_of_twirled_counts_recovers_cos_theta(self, theta):
        """Test that readout mitigation of twirled counts recovers cos(theta) exactly."""
        flip = 0.1
        observables = ObservableSet.build("Z")
        plan = _planned(rotated_circuit(theta), observables, "MINIMAL", instances=8, shots=8000)
        counts = _simulate(plan, readout=flip)
        target = plan.targets[0]
        mitigation = ReadoutErrorMitigation({"QB1": np.array([[1 - flip, flip], [flip, 1 - flip]])})

        raw = measure_term_values(target, counts, observables)
        mitigated = measure_term_values(target, counts, observables, mitigation, readout_twirled=True)
        assert raw.shape == mitigated.shape == (1, 1)
        assert raw[0, 0] == pytest.approx((1 - 2 * flip) * np.cos(theta), abs=1e-9)
        assert mitigated[0, 0] == pytest.approx(np.cos(theta), abs=1e-6)

    # HADAMARD balances every pair of clbits, which the symmetrized correction of ZZ needs.
    @pytest.mark.parametrize("strategy", ["NONE", "HADAMARD"])
    def test_each_clbit_is_mitigated_with_the_readout_errors_of_its_qubit(self, strategy):
        # QB1 lands in clbit 1 and QB2 in clbit 0, with different asymmetric errors (P(0 -> 1), P(1 -> 0)).
        """Test that each classical bit is mitigated with the readout errors of the qubit measured into it."""
        readout = {"QB1": (0.02, 0.1), "QB2": (0.15, 0.05)}
        observables = ObservableSet.build(["ZI", "IZ", "ZZ"])
        target = circuit([prx("QB1", 0.6), prx("QB2", 2.0)], ["QB1", "QB2"], clbits=[1, 0])
        plan = _planned(target, observables, strategy)
        mitigation = ReadoutErrorMitigation({q: np.array([[1 - a, b], [a, 1 - b]]) for q, (a, b) in readout.items()})
        values = measure_term_values(
            plan.targets[0], _simulate(plan, readout=readout), observables, mitigation, plan.readout_twirled
        )
        assert plan.readout_twirled == (strategy == "HADAMARD")
        assert values.shape == (1, 3)
        assert values.dtype == np.float64
        np.testing.assert_allclose(values[0], [np.cos(0.6), np.cos(2.0), np.cos(0.6) * np.cos(2.0)], atol=1e-6)

    def test_identity_only_observables_have_no_terms_to_measure(self):
        """Test that an identity observable has no terms and combines to its offset."""
        observables = ObservableSet.build("II")
        plan = _planned(circuit([], ["QB1", "QB2"]), observables, "NONE", instances=1)
        values = measure_term_values(plan.targets[0], _simulate(plan), observables)
        assert values.shape == (1, 0)
        np.testing.assert_allclose(observables.combine(values), [[1.0]])


class TestResampling:
    def test_totals_and_support_are_kept_and_seed_reproduces(self):
        """Test that resampling keeps the shot totals and outcomes, and the same seed gives the same counts."""
        counts = {(0, 0, 0): {"00": 70, "11": 30}, (0, 0, 1): {"01": 5, "10": 5, "11": 90}}
        first = resample_counts(counts, np.random.default_rng(3))
        second = resample_counts(counts, np.random.default_rng(3))
        assert first == second
        for key, distribution in counts.items():
            assert sum(first[key].values()) == sum(distribution.values())
            assert set(first[key]) <= set(distribution)
        assert first != {key: {k: float(v) for k, v in d.items()} for key, d in counts.items()}

    def test_empty_distribution_cannot_be_resampled(self):
        """Test that counts with no shots cannot be resampled."""
        with pytest.raises(ValueError, match="no shots"):
            resample_counts({(0, 0, 0): {"0": 0}}, np.random.default_rng(0))
