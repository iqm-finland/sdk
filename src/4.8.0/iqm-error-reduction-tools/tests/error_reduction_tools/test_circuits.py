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

from dataclasses import replace

from iqm.error_reduction_tools.circuits import (
    CircuitKey,
    CircuitVariant,
    TargetCircuit,
    TwirlingInstance,
    measurement_map,
)
import numpy as np
import pytest

from iqm.pulse import Circuit

from .builders import circuit, cz, measurements, prx


class TestMeasurementMap:
    @pytest.mark.parametrize(
        ("qubits", "clbits", "expected"),
        [
            (["QB1", "QB2"], [0, 1], {"QB1": 0, "QB2": 1}),
            (["QB3", "QB1", "QB2"], [2, 0, 1], {"QB3": 2, "QB1": 0, "QB2": 1}),
        ],
    )
    def test_reads_clbits_from_measurement_keys(self, qubits, clbits, expected):
        """Test that each qubit is mapped to the classical bit in its measurement key."""
        assert measurement_map(circuit([prx("QB1", 1.0)], qubits, clbits=clbits)) == expected

    def test_rejects_gap_in_clbits(self):
        """Test that classical bits that are not 0 to n - 1 are rejected."""
        with pytest.raises(ValueError, match="classical bits 0 to 1"):
            measurement_map(circuit([prx("QB1", 1.0)], ["QB1", "QB2"], clbits=[0, 2]))

    @pytest.mark.parametrize(
        "late",
        [prx("QB1", 1.0), cz("QB1", "QB2"), *measurements(["QB1"])],
    )
    def test_rejects_acting_on_a_qubit_after_measuring_it(self, late):
        """Test that an operation on a qubit after its measurement is rejected."""
        ops = [prx("QB1", 1.0), *measurements(["QB1"]), late, *measurements(["QB2"], clbits=[1])]
        with pytest.raises(ValueError, match=r"after measuring them"):
            measurement_map(Circuit(name="mid", instructions=tuple(ops)))

    def test_gate_on_another_qubit_after_a_measurement_is_allowed(self):
        # Readout twirling produces this: an X on QB2 inserted after QB1 is measured.
        """Test that a gate on a qubit not yet measured may follow a measurement of another qubit."""
        ops = [prx("QB1", 1.0), *measurements(["QB1"]), prx("QB2", np.pi), *measurements(["QB2"], clbits=[1])]
        assert measurement_map(Circuit(name="twirled", instructions=tuple(ops))) == {"QB1": 0, "QB2": 1}

    def test_rejects_circuit_without_measurements(self):
        """Test that a circuit without measurements is rejected."""
        with pytest.raises(ValueError, match="no measurements"):
            measurement_map(Circuit(name="none", instructions=(prx("QB1", 1.0),)))

    def test_barrier_after_measurements_is_allowed(self):
        """Test that a barrier after the measurements is accepted."""
        barrier = type(prx("QB1", 0.0))(name="barrier", locus=("QB1", "QB2"), args={})
        ops = [cz("QB1", "QB2"), *measurements(["QB1", "QB2"]), barrier]
        assert measurement_map(Circuit(name="b", instructions=tuple(ops))) == {"QB1": 0, "QB2": 1}


class TestTargetCircuit:
    def test_qubit_to_clbit_cannot_be_set_by_hand(self):
        """Test that qubit_to_clbit is read from the circuit and cannot be passed in."""
        with pytest.raises(TypeError):
            TargetCircuit(target_index=0, circuit=circuit([], ["QB1"]), qubit_to_clbit={"QB1": 5})

    def test_instances_walk_the_tree_in_order_and_keys_are_unique(self):
        """Test that instances() yields every instance in variant order, each with a unique key."""
        base = circuit([prx("QB1", 0.3)], ["QB1"])
        variants = tuple(
            CircuitVariant(
                target_index=1,
                variant_index=v,
                circuit=base,
                shots=10,
                num_instances=2,
                twirling_instances=tuple(
                    TwirlingInstance(target_index=1, variant_index=v, instance_index=i, circuit=base, rot_string="I")
                    for i in range(v + 1)
                ),
            )
            for v in range(3)
        )
        target = replace(TargetCircuit(target_index=1, circuit=base), circuit_variants=variants)
        keys = [instance.key for instance in target.instances()]
        assert keys == [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 2, 0), (1, 2, 1), (1, 2, 2)]
        assert all(isinstance(key, CircuitKey) for key in keys)
        assert target.qubit_to_clbit == {"QB1": 0}

    def test_unplanned_target_has_no_variants(self):
        """Test that asking an unplanned target for its variants raises."""
        with pytest.raises(RuntimeError, match="Plan the workflow first"):
            TargetCircuit(target_index=0, circuit=circuit([], ["QB1"])).variants()


class TestCircuitVariant:
    @pytest.mark.parametrize(("shots", "instances", "expected"), [(100, 4, 25), (100, 3, 34), (5, 8, 1)])
    def test_shots_per_instance_rounds_up(self, shots, instances, expected):
        """Test that the shots of a variant are split over its instances, rounded up."""
        base = circuit([], ["QB1"])
        variant = CircuitVariant(
            target_index=0,
            variant_index=0,
            circuit=base,
            shots=shots,
            num_instances=instances,
            twirling_instances=tuple(
                TwirlingInstance(target_index=0, variant_index=0, instance_index=i, circuit=base, rot_string="I")
                for i in range(instances)
            ),
        )
        assert variant.shots_per_instance == expected
        assert variant.shots_per_instance * instances >= shots

    @pytest.mark.parametrize(("shots", "instances"), [(0, 1), (10, 0)])
    def test_rejects_empty_budget(self, shots, instances):
        """Test that zero shots or zero twirling instances are rejected."""
        with pytest.raises(ValueError, match="at least 1"):
            CircuitVariant(
                target_index=0, variant_index=0, circuit=circuit([], ["QB1"]), shots=shots, num_instances=instances
            )

    def test_unbudgeted_variant_has_no_shots_per_instance(self):
        """Test that shots_per_instance raises when the variant has no shots."""
        base = circuit([], ["QB1"])
        instance = TwirlingInstance(target_index=0, variant_index=0, instance_index=0, circuit=base, rot_string="I")
        variant = CircuitVariant(target_index=0, variant_index=0, circuit=base, twirling_instances=(instance,))
        with pytest.raises(RuntimeError, match="has no shots yet"):
            _ = variant.shots_per_instance

    def test_untwirled_variant_has_no_instances(self):
        """Test that asking an untwirled variant for its instances raises."""
        variant = CircuitVariant(
            target_index=0, variant_index=0, circuit=circuit([], ["QB1"]), shots=1, num_instances=1
        )
        with pytest.raises(RuntimeError, match="has not been twirled"):
            variant.instances()
        assert np.isclose(variant.shots, 1)
