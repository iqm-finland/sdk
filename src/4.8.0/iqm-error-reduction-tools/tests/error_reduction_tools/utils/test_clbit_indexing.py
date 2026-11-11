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

"""Tests for the qubit-to-classical-bit mapping every tool shares."""

from iqm.error_reduction_tools.utils.circuit_utils import TwirledCircuit, clbit_index_from_key
import pytest

from iqm.pulse import CircuitOperation


def measure(qubit: str, key: str) -> CircuitOperation:
    return CircuitOperation(name="measure", locus=(qubit,), args={"key": key})


def prx(qubit: str) -> CircuitOperation:
    return CircuitOperation(name="prx", locus=(qubit,), args={"angle": 1.0, "phase": 0.0})


@pytest.mark.parametrize(
    "key, expected",
    [
        ("m_3_0_0", 0),
        ("m_3_0_2", 2),
        ("m_16_0_12", 12),
        ("c_reg_2_0_1", 1),
    ],
)
def test_structured_keys_give_their_clbit_index(key, expected):
    """Test that a structured measurement key gives the clbit index it ends with."""
    index = clbit_index_from_key(key)
    assert isinstance(index, int)
    assert index == expected


@pytest.mark.parametrize("key", ["m0", "meas_a_b_c", "m_3_0", "m_3_0_x"])
def test_keys_without_a_clbit_index_give_none(key):
    """Test that a key without a clbit index gives None."""
    assert clbit_index_from_key(key) is None


class TestQubitToClbit:
    def test_emission_order_does_not_decide_the_mapping(self):
        """Test that the mapping does not depend on the order in which the measurements appear."""
        ordered = TwirledCircuit(
            [prx("QB5"), measure("QB5", "m_3_0_0"), measure("QB2", "m_3_0_1"), measure("QB7", "m_3_0_2")]
        )
        scrambled = TwirledCircuit(
            [prx("QB5"), measure("QB7", "m_3_0_2"), measure("QB5", "m_3_0_0"), measure("QB2", "m_3_0_1")]
        )
        assert ordered.qubit_to_clbit() == {"QB5": 0, "QB2": 1, "QB7": 2}
        assert scrambled.qubit_to_clbit() == ordered.qubit_to_clbit()

    def test_unstructured_keys_fall_back_to_measurement_order(self):
        """Test that keys without a clbit index are numbered in measurement order."""
        circuit = TwirledCircuit([measure("QB7", "m0"), measure("QB2", "m1")])
        assert circuit.qubit_to_clbit() == {"QB7": 0, "QB2": 1}

    def test_values_are_ints_covering_every_measured_qubit(self):
        """Test that every measured qubit is mapped to an int clbit."""
        circuit = TwirledCircuit([measure("QB3", "m_2_0_1"), measure("QB1", "m_2_0_0")])
        mapping = circuit.qubit_to_clbit()
        assert len(mapping) == len(circuit.measured_qubits)
        assert all(isinstance(bit, int) for bit in mapping.values())
        assert sorted(mapping.values()) == [0, 1]
