# Copyright 2024-2025 IQM
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
"""Multiplexing unit tests"""

from iqm.cpc.compiler.standard_stages import merge_multiplexed_timeboxes
from iqm.cpc.interface.compiler import Circuit
from iqm.pulse import CircuitOperation as I

_circuit = Circuit(
    name="meas gates",
    instructions=(
        I(name="measure", locus=("QB1",), args={"key": "a"}),
        I(name="measure", locus=("QB2",), args={"key": "b"}),
        I(name="measure", locus=("QB3", "QB4"), args={"key": "c"}),  # Should merge a+b+c
        I(name="prx", locus=("QB2",), args={"angle": 0.2, "phase": 0.1}),
        I(name="measure", locus=("QB1",), args={"key": "d"}),
        I(name="measure", locus=("QB4",), args={"key": "e"}),  # Should merge d+e
        I(name="measure", locus=("QB1",), args={"key": "f"}),
        I(name="prx", locus=("QB3",), args={"angle": 0.3, "phase": 0.1}),
        I(name="measure", locus=("QB2",), args={"key": "g"}),  # Should merge f+g
    ),
)


def test_multiplexing_merges_correct_boxes(schedule_builder):
    circuit_box = schedule_builder.circuit_to_timebox(_circuit.instructions, name=_circuit.name)
    result = merge_multiplexed_timeboxes(circuit_box)
    assert len(result.children) == 5

    # a+b+c
    assert result[0].locus_components == {"QB1", "QB2", "QB3", "QB4"}
    assert len(result[0].atom["PL_1__readout"][0].probe_pulse.entries) == 2
    assert len(result[0].atom["PL_2__readout"][0].probe_pulse.entries) == 2

    assert result[1] is circuit_box[3]  # first PRX, unchanged

    # d+e
    assert result[2].locus_components == {"QB1", "QB4"}
    assert len(result[2].atom["PL_1__readout"][0].probe_pulse.entries) == 2
    assert "PL_2__readout" not in result[2].atom

    assert result[3] is circuit_box[7]  # second PRX, unchanged

    # f+g
    assert result[4].locus_components == {"QB1", "QB2"}
    assert len(result[4].atom["PL_1__readout"][0].probe_pulse.entries) == 1
    assert len(result[4].atom["PL_2__readout"][0].probe_pulse.entries) == 1
