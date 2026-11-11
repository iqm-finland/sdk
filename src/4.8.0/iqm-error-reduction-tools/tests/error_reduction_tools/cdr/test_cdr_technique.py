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

from iqm.error_reduction_tools.cdr import CDR, CDRDiagnostics, CDREstimation, TrainingVariant
from iqm.error_reduction_tools.circuits import CircuitVariant, TargetCircuit
from iqm.error_reduction_tools.observables import ObservableSet
from iqm.error_reduction_tools.zne import ZNE
import numpy as np
import pytest

from ..builders import circuit, cz, prx
from ..exact import exact_distribution

OBSERVABLES = ObservableSet.build([[(1.0, "ZZ"), (0.5, "IZ")], "ZI"])


def _non_clifford_target():
    gates = [prx("QB1", 0.7, 0.2), prx("QB2", np.pi / 2, np.pi / 2), cz("QB1", "QB2"), prx("QB2", 1.3, 0.4)]
    return TargetCircuit(target_index=0, circuit=circuit(gates, ["QB1", "QB2"]))


def _with_training(exact_rows):
    """Build a planned target whose training circuits carry the given exact values, without generating any."""
    target = _non_clifford_target()
    common = {"target_index": 0, "circuit": target.circuit, "shots": 10, "num_instances": 1}
    variants = [CircuitVariant(variant_index=0, **common)]
    variants += [
        TrainingVariant(variant_index=index, exact_term_values=tuple(row), **common)
        for index, row in enumerate(exact_rows, start=1)
    ]
    return replace(target, circuit_variants=tuple(variants))


class TestCDR:
    def test_training_circuits_carry_their_noiseless_values(self):
        """Test that each training circuit carries the values a noiseless run of it measures."""
        target = _non_clifford_target()
        variants = CDR(num_training_circuits=4, seed=3).build_variants(target, OBSERVABLES)
        assert type(variants[0]) is CircuitVariant
        assert variants[0].circuit == target.circuit
        training = variants[1:]
        assert 2 <= len(training) <= 4
        for variant in training:
            assert isinstance(variant, TrainingVariant)
            assert len(variant.exact_term_values) == OBSERVABLES.num_terms
            # The exact values must be what a noiseless run of the training circuit measures.
            measured = OBSERVABLES.term_values_from_counts(exact_distribution(variant.circuit))
            np.testing.assert_allclose(measured, variant.exact_term_values, atol=1e-9)

    def test_budget_overrides_apply_to_target_and_training_separately(self):
        """Test that the target and the training circuits get their own shots and instances."""
        technique = CDR(
            num_training_circuits=3, num_target_instances=8, num_training_instances=2, target_shots=900, seed=0
        )
        variants = technique.build_variants(_non_clifford_target(), OBSERVABLES)
        assert (variants[0].shots, variants[0].num_instances) == (900, 8)
        # Unset training shots are left to the workflow.
        assert {(v.shots, v.num_instances) for v in variants[1:]} == {(None, 2)}

    @pytest.mark.parametrize("field", ["num_training_circuits", "target_shots", "num_training_instances"])
    def test_invalid_budgets_are_rejected(self, field):
        """Test that fewer than 2 training circuits and zero shots or instances are rejected."""
        with pytest.raises(ValueError, match=field):
            CDR(**{field: 0 if field != "num_training_circuits" else 1})

    def test_unknown_strategy_is_rejected(self):
        """Test that an unknown strategy is rejected."""
        with pytest.raises(ValueError, match="Unknown strategy 'bogus'"):
            CDR(strategy="bogus")

    def test_identity_only_observables_are_rejected(self):
        """Test that observables with only the identity are rejected."""
        with pytest.raises(ValueError, match="no Pauli terms besides the identity"):
            CDR().build_variants(_non_clifford_target(), ObservableSet.build("II"))

    def test_repeated_exact_values_still_give_every_requested_training_circuit(self, recwarn):
        """Test that every requested training circuit is built when there are fewer distinct exact values."""
        # A single Pauli term has only three exact values on a Clifford circuit, so they must repeat.
        variants = CDR(num_training_circuits=10, seed=0).build_variants(
            _non_clifford_target(), ObservableSet.build("ZI")
        )
        assert len(variants) == 11
        assert not [w for w in recwarn if "training circuits could be built" in str(w.message)]
        # The training circuits share exact values, but are distinct circuits.
        assert len({repr(variant.circuit.instructions) for variant in variants[1:]}) == 10


class TestCDREstimation:
    EXACT = np.array([[1.0, 0.0, -1.0], [-1.0, 1.0, 0.0], [0.0, -1.0, 1.0], [1.0, 1.0, 1.0]])

    @pytest.mark.parametrize(("slope", "intercept"), [(0.5, 0.1), (0.8, -0.05), (0.3, 0.0)])
    @pytest.mark.parametrize("combine", ["per_term", "group"])
    def test_linear_noise_is_inverted_exactly(self, slope, intercept, combine):
        """Test that a linear noise map is inverted exactly, per term and per observable."""
        ideal_target = np.array([0.4, -0.2, 0.7])
        measured = slope * np.vstack([ideal_target, self.EXACT]) + intercept
        estimate = CDREstimation(combine=combine).estimate(_with_training(self.EXACT), measured, OBSERVABLES)
        assert estimate.values.shape == (2,)
        # A linear map of every term is also a linear map of every combination, so both orders are exact.
        expected = OBSERVABLES.combine(ideal_target)
        np.testing.assert_allclose(estimate.values, expected, atol=1e-9)
        diagnostics = estimate.diagnostics
        assert isinstance(diagnostics, CDRDiagnostics)
        assert diagnostics.num_training == 4
        np.testing.assert_allclose(diagnostics.r_squared, 1.0)

    def test_fewer_than_two_training_circuits_are_rejected(self):
        """Test that fewer than 2 training circuits are rejected."""
        with pytest.raises(ValueError, match="At least 2 training circuits"):
            CDREstimation().estimate(_with_training(self.EXACT[:1]), np.zeros((2, 3)), OBSERVABLES)

    def test_flat_training_data_and_extrapolation_warn(self):
        """Test that flat training values and a target outside their range warn."""
        measured = np.vstack([np.full(3, 0.9), np.full((4, 3), 0.1)])
        with pytest.warns(UserWarning) as record:
            CDREstimation().estimate(_with_training(self.EXACT), measured, OBSERVABLES)
        messages = " ".join(str(warning.message) for warning in record)
        assert "ill-determined" in messages
        assert "extrapolation" in messages

    @pytest.mark.parametrize(
        ("settings", "message"),
        [({"combine": "bogus"}, "Unknown combine 'bogus'"), ({"min_training_spread": -0.1}, "at least 0")],
    )
    def test_invalid_settings_are_rejected(self, settings, message):
        """Test that an unknown combine and a negative minimum spread are rejected."""
        with pytest.raises(ValueError, match=message):
            CDREstimation(**settings)

    def test_variants_not_built_by_cdr_are_rejected(self):
        """Test that variants not built by CDR are rejected."""
        target = _non_clifford_target()
        target = replace(target, circuit_variants=ZNE().build_variants(target, OBSERVABLES))
        with pytest.raises(TypeError, match="not a TrainingVariant"):
            CDREstimation().estimate(target, np.zeros((3, 3)), OBSERVABLES)

    def test_accepts_only_cdr(self):
        """Test that CDREstimation accepts CDR and not ZNE."""
        assert CDREstimation().accepts(CDR())
        assert not CDREstimation().accepts(ZNE())
