# Copyright 2024 IQM
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
import pytest

from iqm.pulse.playlist.instructions import ConditionalInstruction, Instruction, IQPulse, RealPulse, VirtualRZ, Wait
from iqm.pulse.playlist.waveforms import Constant, Gaussian


def test_instruction():
    """Test base instruction"""
    assert Instruction(1).validate() is None
    with pytest.raises(ValueError, match="Instruction.duration .* is negative."):
        Instruction(-1).validate()


def test_wait():
    """Test Wait instruction"""
    wait = Wait(64)
    assert wait.validate() is None
    assert wait.copy(duration=48) == Wait(48)


def test_virtual_rz():
    """Test VirtualRZ instruction"""
    rz = VirtualRZ(64, 0.5)
    assert rz.validate() is None
    assert rz.copy(phase_increment=0.25) == VirtualRZ(rz.duration, phase_increment=0.25)


def test_real_pulse():
    """Test RealPulse instruction"""
    waveform = Constant(10)
    real_pulse = RealPulse(64, waveform, 0.6)
    assert real_pulse.validate() is None
    assert real_pulse.get_waveforms() == (waveform,)
    with pytest.raises(ValueError, match="RealPulse.scale 1.1 not in \\[-1, 1\\]."):
        real_pulse.copy(scale=1.1).validate()


def test_iq_pulse():
    """Test IQPulse"""
    wave_i = Constant(10)
    wave_q = Constant(20)
    iq_pulse = IQPulse(64, wave_i, wave_q, scale_i=0.6, scale_q=0.1, phase=1.0)
    assert iq_pulse.validate() is None
    with pytest.raises(ValueError, match="IQPulse.scale_i 1.5 not in \\[-1, 1\\]."):
        iq_pulse.copy(scale_i=1.5).validate()
    with pytest.raises(ValueError, match="IQPulse.scale_q -1.5 not in \\[-1, 1\\]."):
        iq_pulse.copy(scale_q=-1.5).validate()
    assert iq_pulse.get_waveforms() == (wave_i, wave_q)


def test_conditional_instruction():
    outcome_1 = RealPulse(10, Constant(10), 0.3)
    outcome_2 = RealPulse(10, Constant(10), 0.5)
    cond = ConditionalInstruction(10, condition="condition", outcomes=(outcome_1, outcome_2))
    assert cond.validate() is None
    with pytest.raises(ValueError, match="All the conditional instructions must have the same duration .*"):
        cond.copy(outcomes=(outcome_1, outcome_2.copy(duration=12))).validate()
    with pytest.raises(ValueError, match="There must be at least one outcome."):
        cond.copy(outcomes=()).validate()
    assert cond.get_child_instructions() == (outcome_1, outcome_2)


def test_instructions_have_same_hash():
    instruction1 = RealPulse(24, Gaussian(24, 7, 0), 0.8)
    instruction2 = RealPulse(24, Gaussian(24, 7, 0), 0.8)
    assert hash(instruction1) == hash(instruction2)


def test_instruction_different_waveforms_have_different_hash():
    instruction1 = RealPulse(24, Gaussian(24, 7, 0), 0.8)
    instruction2 = RealPulse(24, Gaussian(24, 7, 2), 0.8)
    assert hash(instruction1) != hash(instruction2)


def test_instruction_different_durations_have_different_hash():
    instruction1 = RealPulse(24, Gaussian(24, 7, 0), 0.8)
    instruction2 = RealPulse(30, Gaussian(24, 7, 0), 0.8)
    assert hash(instruction1) != hash(instruction2)


def test_instruction_different_scales_have_different_hash():
    instruction1 = RealPulse(24, Gaussian(24, 7, 0), 0.8)
    instruction2 = RealPulse(24, Gaussian(24, 7, 0), 0.7)
    assert hash(instruction1) != hash(instruction2)
