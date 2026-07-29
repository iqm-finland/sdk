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

from iqm.pulse.playlist.channel import (
    ChannelDescription,
    ChannelProperties,
    IQChannelConfig,
    round_duration_to_granularity_samples,
)
from iqm.pulse.playlist.instructions import ConditionalInstruction, IQPulse, RealPulse, VirtualRZ, Wait
from iqm.pulse.playlist.waveforms import Constant, GaussianSmoothedSquare


@pytest.fixture
def channel_properties() -> ChannelProperties:
    return ChannelProperties(2.0e9, 16, 64, (Wait, IQPulse, VirtualRZ), True)


@pytest.fixture
def channel_description() -> ChannelDescription:
    config = IQChannelConfig(sample_rate=2.4e9, frequency=4.2e9)
    return ChannelDescription("test_controller", config)


def test_channel_properties_duration_calculation(channel_properties):
    """Test duration calculation in ChannelProperties"""
    assert channel_properties.duration_to_samples(1.0) == 2.0e9
    assert channel_properties.duration_to_seconds(2.0e9) == 1.0
    assert channel_properties.duration_to_int_samples(3.2e-8) == 64
    regexp = r"Given duration \([^\)]*\) is 64.5 samples, which is not an integer"
    with pytest.raises(ValueError, match=regexp):
        channel_properties.duration_to_int_samples(3.225e-8)
    regexp = r"Given duration \([^\)]*\) is 72 samples, which is not an integer multiple of 16 samples."
    with pytest.raises(ValueError, match=regexp):
        channel_properties.duration_to_int_samples(3.6e-8)
    regexp = r"Given duration \([^\)]*\) is 48 samples, which is less than 64 samples."
    with pytest.raises(ValueError, match=regexp):
        channel_properties.duration_to_int_samples(2.4e-8)
    assert channel_properties.round_duration_to_granularity(3.225e-8) == 3.2e-8
    assert channel_properties.round_duration_to_granularity(3.225e-8, round_up=True) == 4.0e-8
    assert channel_properties.round_duration_to_granularity(1e-9, force_min_duration=True) == 3.2e-8
    assert channel_properties.round_duration_to_granularity(1e-9, round_up=True, force_min_duration=True) == 3.2e-8


def test_round_duration_to_granularity_samples():
    """Test the multi-channel to-granularity rounding."""
    channels = [
        ChannelProperties(1.0e9, 16, 80, (Wait, IQPulse, VirtualRZ), True),
        ChannelProperties(1.0e9, 32, 64, (Wait, IQPulse, VirtualRZ), True),
    ]
    assert round_duration_to_granularity_samples(channels, 17e-9) == 32
    assert round_duration_to_granularity_samples(channels, 32.2e-9) == 32
    assert round_duration_to_granularity_samples(channels, 30e-9, round_up=True) == 32
    assert round_duration_to_granularity_samples(channels, 32.3e-9, round_up=True) == 64
    assert round_duration_to_granularity_samples(channels, 33e-9, force_min_duration=True) == 96  # == 3*32 >= 80
    assert round_duration_to_granularity_samples(channels, 100e-9, force_min_duration=True) == 96
    assert round_duration_to_granularity_samples(channels, 33e-9, round_up=True, force_min_duration=True) == 96
    assert round_duration_to_granularity_samples(channels, 100e-9, round_up=True, force_min_duration=True) == 128


def test_waveform_and_instructions_are_unique(channel_description):
    """Test that only unique instructions and waveforms are stored into channel description's tables"""
    waveform = Constant(10)
    waveform2 = GaussianSmoothedSquare(20, 2, 2, 2)
    instructions = [
        RealPulse(25, waveform, 0.6),
        RealPulse(25, waveform, 0.9),
        IQPulse(25, waveform2, waveform, 0.6, 0.1, 3.1),
    ]
    for i, expected_instructions, expected_waveforms in [(0, 1, 1), (2, 2, 2), (1, 3, 2), (0, 3, 2), (2, 3, 2)]:
        channel_description.add_instruction(instructions[i])
        assert len(channel_description.instruction_table) == expected_instructions
        assert len(channel_description.waveform_table) == expected_waveforms


def test_add_child_instruction(channel_description):
    instruction = ConditionalInstruction(
        duration=24.0 / channel_description.config.sample_rate,
        condition="some-condition",
        outcomes=(Wait(24), RealPulse(24, Constant(10), 0.6)),
    )
    channel_description.add_instruction(instruction)
    assert len(channel_description.instruction_table) == 3  # Wait, RealPulse, ConditionalInstruction
    assert len(channel_description.waveform_table) == 1  # Constant(10) waveform of RealPulse
