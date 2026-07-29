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
from iqm.models.playlist.channel_descriptions import ChannelDescription, IQChannelConfig
import iqm.models.playlist.instructions as sc_instructions
import pytest

from iqm.pulse.playlist.channel import ChannelConfiguration, ChannelProperties
from iqm.pulse.playlist.instructions import ConditionalInstruction, IQPulse, Wait
from iqm.pulse.playlist.playlist import Playlist
from iqm.pulse.playlist.schedule import Segment
from iqm.pulse.playlist.waveforms import Gaussian
from iqm.pulse.validation import (
    InvalidInstructionError,
    PlaylistValidationError,
    validate_channel,
    validate_instruction_and_wf_length,
    validate_playlist_compatibility,
)


@pytest.fixture
def channel_properties() -> ChannelProperties:
    return ChannelProperties(
        sample_rate=2.4e9,
        instruction_duration_granularity=16,
        instruction_duration_min=32,
        compatible_instructions=(Wait, ConditionalInstruction, IQPulse),
    )


channel_name = "drive_2.awg"
test_cases = [
    (
        2.2e9,
        None,
        f"Sample rates do not match. Device {channel_name} expects 2400000000.0 but got from playlist 2200000000.0",
    ),
    (
        2.4e9,
        sc_instructions.Instruction(61, sc_instructions.Wait()),
        f"Instruction length of 61 doesn't match the {channel_name} granularity of the device",
    ),
    (
        2.4e9,
        sc_instructions.Instruction(16, sc_instructions.Wait()),
        f"Instruction length of 16 is less than the minimum number of samples 32 for {channel_name}",
    ),
    (
        2.4e9,
        sc_instructions.Instruction(61, sc_instructions.VirtualRZ(2.7)),
        f"{channel_name}: Incompatible instruction VirtualRZ",
    ),
    (2.4e9, sc_instructions.Instruction(32, sc_instructions.Wait()), None),
]


@pytest.mark.parametrize("sampling_rate, instruction, expected_error", test_cases)
def test_validate_channel(sampling_rate, instruction, expected_error, channel_properties):
    """Test that channel validation raises error when necessary"""
    channel = ChannelDescription(IQChannelConfig(sampling_rate), channel_name)
    if instruction is not None:
        channel.add_instruction(instruction)
    if expected_error is None:
        assert validate_channel(channel, channel_properties) is None
    else:
        with pytest.raises(PlaylistValidationError, match=expected_error):
            validate_channel(channel, channel_properties)


@pytest.mark.parametrize("sampling_rate, instruction, expected_error", test_cases)
def test_validate_playlist(sampling_rate, instruction, expected_error, channel_properties):
    """Test that playlist validation raises error when necessary"""
    channel_descriptions = {
        channel_name: ChannelDescription(IQChannelConfig(sampling_rate), channel_name),
        "other_channel": ChannelDescription(IQChannelConfig(sampling_rate), "other_channel"),
    }
    if instruction is not None:
        channel_descriptions[channel_name].add_instruction(instruction)
    channel_descriptions["other_channel"].add_instruction(sc_instructions.Instruction(32, sc_instructions.Wait()))
    playlist = Playlist(channel_descriptions=channel_descriptions, segments=[Segment({})])
    device_constraints = {
        channel_name: channel_properties,
        "other_channel": channel_properties,
    }

    if expected_error is None:
        assert validate_playlist_compatibility(playlist, device_constraints) is None
    else:
        with pytest.raises(PlaylistValidationError, match=expected_error):
            validate_playlist_compatibility(playlist, device_constraints)


def test_invalid_channel_configuration(channel_properties):
    """Test that validate_channel raises error if channel configuration does not have sampling rate"""
    channel = ChannelDescription(ChannelConfiguration(), channel_name)
    with pytest.raises(PlaylistValidationError, match="Channel configuration does not have sampling rate"):
        validate_channel(channel, channel_properties)


@pytest.mark.parametrize(
    "instruction, expected_message",
    [
        (sc_instructions.Instruction(24, sc_instructions.RealPulse(Gaussian(24, 2, 4), 0.5)), None),
        (
            sc_instructions.Instruction(25, sc_instructions.RealPulse(Gaussian(24, 2, 4), 1.0)),
            "Instruction duration != waveform length",
        ),
        (sc_instructions.Instruction(24, sc_instructions.RealPulse(Gaussian(24, 2, 4), 2.0)), "scale not in -1..1"),
        (sc_instructions.Instruction(24, sc_instructions.RealPulse(Gaussian(24, 2, 4), -1.5)), "scale not in -1..1"),
        (
            sc_instructions.Instruction(
                24, sc_instructions.IQPulse(Gaussian(24, 2, 4), Gaussian(24, 2, 4), 0.6, 0.7, 0.0)
            ),
            None,
        ),
        (
            sc_instructions.Instruction(
                25, sc_instructions.IQPulse(Gaussian(24, 2, 4), Gaussian(25, 2, 4), 0.6, 0.7, 0.0)
            ),
            "Instruction duration != waveform_i length",
        ),
        (
            sc_instructions.Instruction(
                25, sc_instructions.IQPulse(Gaussian(25, 2, 4), Gaussian(24, 2, 4), 0.6, 0.7, 0.0)
            ),
            "Instruction duration != waveform_q length",
        ),
        (
            sc_instructions.Instruction(
                24, sc_instructions.IQPulse(Gaussian(24, 2, 4), Gaussian(24, 2, 4), 1.1, 0.7, 0.0)
            ),
            "scale not in -1..1",
        ),
        (
            sc_instructions.Instruction(
                24, sc_instructions.IQPulse(Gaussian(24, 2, 4), Gaussian(24, 2, 4), 0.5, -1.7, 0.0)
            ),
            "scale not in -1..1",
        ),
    ],
)
def test_validate_instruction_and_wf_length(instruction, expected_message):
    """Test that validation raises error if instruction and waveform lengths do not match"""
    if expected_message is None:
        assert validate_instruction_and_wf_length(instruction) is None  # expect validation to pass
    else:
        with pytest.raises(InvalidInstructionError, match=expected_message):
            validate_instruction_and_wf_length(instruction)
