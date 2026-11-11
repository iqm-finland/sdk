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

import numpy as np
import pytest

from .builders import bell_circuit, circuit, cz, prx, rotated_circuit
from .exact import exact_distribution, num_gates


class TestExactDistribution:
    """The other tests trust this helper for their known answers, so it is checked against analytic ones."""

    def test_bell_state(self):
        """Test the distribution of a Bell state."""
        distribution = exact_distribution(bell_circuit())
        assert set(distribution) == {"00", "11"}
        np.testing.assert_allclose(list(distribution.values()), [0.5, 0.5])

    @pytest.mark.parametrize("theta", [0.3, 1.2, 2.5])
    def test_rotation_gives_cos_theta(self, theta):
        """Test that an X rotation by theta gives <Z> = cos(theta)."""
        distribution = exact_distribution(rotated_circuit(theta))
        assert distribution["0"] - distribution.get("1", 0.0) == pytest.approx(np.cos(theta), abs=1e-12)

    def test_outcome_lands_on_its_clbit_not_its_measurement_position(self):
        """Test that an outcome is written to its classical bit, not to the position of its measurement."""
        flipped = circuit([prx("QB1", np.pi)], ["QB1", "QB2"], clbits=[1, 0])
        assert exact_distribution(flipped) == pytest.approx({"10": 1.0})

    @pytest.mark.parametrize("flip", [0.05, 0.2])
    def test_readout_flip_scales_z_by_one_minus_twice_flip(self, flip):
        """Test that a symmetric readout flip p scales <Z> by 1 - 2p."""
        distribution = exact_distribution(circuit([], ["QB1"]), readout=flip)
        assert distribution["0"] - distribution["1"] == pytest.approx(1 - 2 * flip)

    def test_asymmetric_readout_flips_each_qubit_with_its_own_probabilities(self):
        # QB1 is prepared in 1 and lands in clbit 1, QB2 stays in 0 and lands in clbit 0.
        """Test that asymmetric readout errors are applied with the probabilities of each qubit."""
        flipped = circuit([prx("QB1", np.pi)], ["QB1", "QB2"], clbits=[1, 0])
        distribution = exact_distribution(flipped, readout={"QB1": (0.4, 0.1), "QB2": (0.2, 0.3)})
        assert distribution == pytest.approx({"10": 0.9 * 0.8, "11": 0.9 * 0.2, "00": 0.1 * 0.8, "01": 0.1 * 0.2})

    @pytest.mark.parametrize("error", [0.01, 0.1])
    def test_depolarizing_scales_z_by_survival_per_gate(self, error):
        """Test that depolarizing noise scales <Z> by the survival probability of every gate."""
        target = circuit([prx("QB1", 0.4), cz("QB1", "QB2"), prx("QB2", 0.0)], ["QB1", "QB2"])
        ideal = exact_distribution(target)
        noisy = exact_distribution(target, depolarizing=error)
        z_first = {key: 1 - 2 * int(key[-1]) for key in noisy}
        assert num_gates(target) == 3
        assert sum(p * z_first[k] for k, p in noisy.items()) == pytest.approx(
            (1 - error) ** 3 * sum(p * z_first[k] for k, p in ideal.items())
        )
