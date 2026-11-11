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

from iqm.error_reduction_tools.cdr import CDR
from iqm.error_reduction_tools.circuits import CircuitVariant, TargetCircuit
from iqm.error_reduction_tools.observables import ObservableSet
from iqm.error_reduction_tools.zne import ZNE, NoiseAmplifiedVariant, ZNEDiagnostics, ZNEEstimation
import numpy as np
import pytest

from ..builders import bell_circuit

OBSERVABLES = ObservableSet.build(["ZZ", [(0.5, "ZI"), (0.5, "IZ"), (1.0, "II")]])


def _target(noise_factors):
    target = TargetCircuit(target_index=0, circuit=bell_circuit())
    variants = ZNE(noise_factors=noise_factors).build_variants(target, OBSERVABLES)
    return replace(target, circuit_variants=variants)


def _gates(circuit):
    return [operation for operation in circuit.instructions if operation.name != "measure"]


class TestZNE:
    @pytest.mark.parametrize(
        ("factors", "message"),
        [((1,), "At least 2"), ((1, 2), "odd integer"), ((3, 5), "start at 1"), ((1, 5, 3), "strictly increase")],
    )
    def test_invalid_noise_factors_are_rejected(self, factors, message):
        """Test that too few, even, unordered or not starting at 1 noise factors are rejected."""
        with pytest.raises(ValueError, match=message):
            ZNE(noise_factors=factors)

    def test_each_variant_repeats_every_gate_noise_factor_times(self):
        """Test that each variant executes every gate of the target noise factor times."""
        target = _target((1, 3, 7))
        body = len(_gates(bell_circuit()))
        variants = target.variants()
        assert all(isinstance(variant, NoiseAmplifiedVariant) for variant in variants)
        assert [variant.noise_factor for variant in variants] == [1, 3, 7]
        assert [len(_gates(variant.circuit)) for variant in variants] == [body, 3 * body, 7 * body]
        assert variants[0].circuit == target.circuit
        # Budgets are left to the workflow.
        assert all(variant.shots is None and variant.num_instances is None for variant in variants)


class TestZNEEstimation:
    @pytest.mark.parametrize(
        ("estimation", "curve"),
        [
            (ZNEEstimation(degree=1), lambda x, a: a - 0.05 * x),
            (ZNEEstimation(), lambda x, a: a - 0.1 * x + 0.01 * x**2),
            (ZNEEstimation(extrapolation_type="Exp"), lambda x, a: a * np.exp(-0.2 * x)),
        ],
    )
    @pytest.mark.parametrize("at_zero", [(0.9, -0.3, 0.6), (0.4, 0.2, -0.8)])
    def test_data_on_the_model_extrapolates_to_its_value_at_zero(self, estimation, curve, at_zero):
        """Test that data lying on the fitted model extrapolates to its value at zero."""
        target = _target((1, 3, 5))
        x = np.array([1.0, 3.0, 5.0])
        measured = np.stack([curve(x, a) for a in at_zero], axis=1)
        estimate = estimation.estimate(target, measured, OBSERVABLES)
        assert estimate.values.shape == (2,)
        assert estimate.values.dtype == np.float64
        np.testing.assert_allclose(estimate.diagnostics.extrapolated, at_zero, atol=1e-6)
        expected = OBSERVABLES.combine(np.array(at_zero))
        np.testing.assert_allclose(estimate.values, expected, atol=1e-6)

    def test_selected_noise_factors_ignore_the_others(self):
        """Test that only the selected noise factors are fitted."""
        target = _target((1, 3, 5, 7))
        measured = np.array([[0.8, 0.8, 0.8], [0.6, 0.6, 0.6], [0.4, 0.4, 0.4], [-5.0, -5.0, -5.0]])
        estimate = ZNEEstimation(degree=1, noise_factors=(1, 3, 5)).estimate(target, measured, OBSERVABLES)
        diagnostics = estimate.diagnostics
        assert isinstance(diagnostics, ZNEDiagnostics)
        assert diagnostics.variant_indices == (0, 1, 2)
        assert diagnostics.num_executed_factors == 4
        np.testing.assert_allclose(diagnostics.extrapolated, [0.9, 0.9, 0.9])

    def test_unexecuted_noise_factor_and_excess_degree_are_rejected(self):
        """Test that an unexecuted noise factor and a too high degree are rejected."""
        target = _target((1, 3))
        measured = np.zeros((2, 3))
        with pytest.raises(ValueError, match=r"\[5\] were not executed"):
            ZNEEstimation(noise_factors=(1, 5)).estimate(target, measured, OBSERVABLES)
        with pytest.raises(ValueError, match="degree of the polynomial is too high"):
            ZNEEstimation(degree=2).estimate(target, measured, OBSERVABLES)

    def test_unphysical_extrapolation_warns(self):
        """Test that a Pauli term extrapolated outside [-1, 1] warns."""
        measured = np.array([[0.9, 0.0, 0.0], [0.5, 0.0, 0.0]])
        with pytest.warns(UserWarning, match="'ZZ' = \\+1.1"):
            ZNEEstimation(degree=1).estimate(_target((1, 3)), measured, OBSERVABLES)

    @pytest.mark.parametrize(
        ("settings", "message"),
        [
            ({"extrapolation_type": "Exp", "degree": 1}, "no degree"),
            ({"noise_factors": (1, 1)}, "distinct noise factors"),
            ({"extrapolation_type": "Linear"}, "Unknown extrapolation_type 'Linear'"),
            ({"degree": 0}, "degree must be at least 1, got 0"),
        ],
    )
    def test_invalid_settings_are_rejected(self, settings, message):
        """Test that contradictory or unknown settings are rejected."""
        with pytest.raises(ValueError, match=message):
            ZNEEstimation(**settings)

    def test_variants_not_built_by_zne_are_rejected(self):
        """Test that variants not built by ZNE are rejected."""
        variants = tuple(CircuitVariant(target_index=0, variant_index=i, circuit=bell_circuit()) for i in range(2))
        target = TargetCircuit(target_index=0, circuit=bell_circuit(), circuit_variants=variants)
        with pytest.raises(TypeError, match="not a NoiseAmplifiedVariant"):
            ZNEEstimation().estimate(target, np.zeros((2, 3)), OBSERVABLES)

    def test_accepts_only_zne(self):
        """Test that ZNEEstimation accepts ZNE and not CDR."""
        assert ZNEEstimation().accepts(ZNE())
        assert not ZNEEstimation().accepts(CDR())
