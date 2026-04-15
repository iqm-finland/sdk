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
from dataclasses import dataclass

import pytest

from iqm.pulse.gates.prx import PRX_CustomWaveforms
from iqm.pulse.playlist.waveforms import Waveform


@dataclass(frozen=True)
class RaisedCosine(Waveform):
    width: float
    center_offset: float

    def _sample(self, _):
        pass


@dataclass(frozen=True)
class RaisedCosineDerivative(Waveform):
    width: float
    center_offset: float

    def _sample(self, _):
        pass


class PRX_RaisedCosine(PRX_CustomWaveforms, wave_i=RaisedCosine, wave_q=RaisedCosineDerivative):
    """Implementation of PRX using a raised cosine pulse instead of a gaussian."""

    center_offset: float


CUSTOM_CAL_DATA = {
    "duration": 40e-09,
    "amplitude_i": 0.1662,
    "amplitude_q": -0.00802,
    "width": 20e-9,
    "center_offset": 0.0,
}


def test_compiler_amend_raises_unregistered_gate(pulla_on_spark):
    """
    Test that trying to amend calibration for an unregistered gate raises a ValueError.
    """
    compiler = pulla_on_spark.get_standard_compiler()

    with pytest.raises(
        ValueError,
        match="bad_gate is not a registered gate",
    ):
        compiler.amend_calibration_for_gate_implementation("bad_gate", "raised_cosine", "QB1", CUSTOM_CAL_DATA)


def test_compiler_amend_raises_unregistered_implementation(pulla_on_spark):
    """
    Test that trying to amend calibration for an unregistered gate implementation raises a ValueError.
    """
    compiler = pulla_on_spark.get_standard_compiler()

    with pytest.raises(
        ValueError,
        match="raised_cosine is not a registered gate implementation of prx",
    ):
        compiler.amend_calibration_for_gate_implementation("prx", "raised_cosine", "QB1", CUSTOM_CAL_DATA)


def test_compiler_add_implementation(pulla_on_spark):
    """
    Test that adding a gate implementation to the compiler works as expected.
    """
    compiler = pulla_on_spark.get_standard_compiler()

    assert "raised_cosine" not in compiler.gates["prx"].implementations

    compiler.add_implementation("prx", "raised_cosine", PRX_RaisedCosine)
    assert compiler.gates["prx"].implementations["raised_cosine"] == PRX_RaisedCosine


def test_compiler_amend_calibration_for_gate_implementation(pulla_on_spark):
    """
    Test that amending calibration for a gate implementation works as expected.
    """
    compiler = pulla_on_spark.get_standard_compiler()

    qubits = pulla_on_spark._chip_topology.qubits_sorted
    compiler.component_mapping = {str(index): value for index, value in enumerate(qubits)}

    compiler.add_implementation("prx", "raised_cosine", PRX_RaisedCosine)

    for qubit in list(compiler.component_mapping.values()):
        compiler.amend_calibration_for_gate_implementation("prx", "raised_cosine", (qubit,), CUSTOM_CAL_DATA)

    for qubit in list(compiler.component_mapping.values()):
        for param, value in CUSTOM_CAL_DATA.items():
            assert compiler.get_calibration_set_values()[f"gates.prx.raised_cosine.{qubit}.{param}"] == value
            assert compiler.builder.has_calibration("prx", "raised_cosine", (qubit,))  # builder loci are tuples
