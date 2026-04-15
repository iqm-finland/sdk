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
from iqm.models.playlist.channel_descriptions import ChannelConfiguration, ChannelDescription, IQChannelConfig
from iqm.models.playlist.instructions import ConditionalInstruction, Instruction, IQPulse, RealPulse, VirtualRZ, Wait
import pytest

from iqm.pulse.playlist.waveforms import Constant
from iqm.pulse.validation import (
    AWGScheduleValidationData,
    AWGScheduleValidationError,
    InvalidInstructionError,
    validate_awg_and_schedule_compatibility,
)

mock_validation = AWGScheduleValidationData(
    sampling_rate=2.4e9,
    granularity=16,
    min_number_of_samples=32,
    compatible_instructions=(Wait, RealPulse, ConditionalInstruction, IQPulse),
)


@pytest.mark.parametrize(
    "channel, expected_message",
    [
        (None, "Controller was not found in the playlist"),
        (
            ChannelDescription(ChannelConfiguration(), "test_controller"),
            "Channel configuration does not have sampling rate",
        ),
    ],
)
def test_invalid_channel_description_raises_error(channel, expected_message):
    """Test that validation raises error if channel description is None or does not have sampling rate"""
    with pytest.raises(AWGScheduleValidationError, match=expected_message):
        validate_awg_and_schedule_compatibility(channel, mock_validation)


def test_sampling_rate_raises_error():
    controller_name = "drive_2.awg"
    error_string = (
        f"Sampling rates do not match. "
        f"Device {controller_name} expects {mock_validation.sampling_rate} "
        f"but got from playlist {2.2e9}"
    )
    channel_description = ChannelDescription(IQChannelConfig(2.2e9), controller_name)
    with pytest.raises(AWGScheduleValidationError, match=error_string):
        validate_awg_and_schedule_compatibility(channel_description, mock_validation)


def test_granularity_raises_error():
    controller_name = "drive_2.awg"
    channel_description = ChannelDescription(IQChannelConfig(2.4e9), controller_name)
    mock_instruction = Instruction(61, Wait())
    channel_description.add_instruction(mock_instruction)
    error_string = (
        f"Instruction length of {mock_instruction.duration_samples} doesn't match the "
        f"{controller_name} granularity of the device"
    )
    with pytest.raises(AWGScheduleValidationError, match=error_string):
        validate_awg_and_schedule_compatibility(channel_description, mock_validation)


def test_not_enough_samples_raises_error():
    controller_name = "drive_2.awg"
    channel_description = ChannelDescription(IQChannelConfig(2.4e9), controller_name)
    mock_instruction = Instruction(16, Wait())
    channel_description.add_instruction(mock_instruction)
    error_string = (
        f"Instruction length of {mock_instruction.duration_samples} is less than the minimum number "
        f"of samples {mock_validation.min_number_of_samples} for {controller_name}"
    )
    with pytest.raises(AWGScheduleValidationError, match=error_string):
        validate_awg_and_schedule_compatibility(channel_description, mock_validation)


def test_incompatible_instruction_raises_error():
    controller_name = "drive_2.awg"
    channel_description = ChannelDescription(IQChannelConfig(2.4e9), controller_name)
    mock_instruction = Instruction(61, VirtualRZ(2.7))
    channel_description.add_instruction(mock_instruction)
    error_string = (
        f"Incompatible instruction {type(mock_instruction.operation).__name__} not compatible with "
        f"{channel_description.controller_name}"
    )
    with pytest.raises(AWGScheduleValidationError, match=error_string):
        validate_awg_and_schedule_compatibility(channel_description, mock_validation)


def test_incompatible_conditional_instruction_lengths():
    controller_name = "drive_2.awg"
    channel_description = ChannelDescription(IQChannelConfig(2.4e9), controller_name)
    instruction_wait = Instruction(64, Wait())
    mock_instruction = Instruction(32, ConditionalInstruction(1, instruction_wait, instruction_wait))
    channel_description.add_instruction(mock_instruction)
    error_string = (
        "Conditional Instruction's instructions must have the same duration\n"
        f"Duration of Conditional instruction: {mock_instruction.duration_samples} \n"
        f"Duration of if_false instruction: {instruction_wait.duration_samples}\n"
        f"Duration of if_true instruction: {instruction_wait.duration_samples}\n"
    )
    with pytest.raises(AWGScheduleValidationError, match=error_string):
        validate_awg_and_schedule_compatibility(channel_description, mock_validation)


def test_compatible_instruction():
    controller_name = "drive_2.awg"
    channel_description = ChannelDescription(IQChannelConfig(2.4e9), controller_name)
    mock_instruction = Instruction(32, Wait())
    channel_description.add_instruction(mock_instruction)
    validate_awg_and_schedule_compatibility(channel_description, mock_validation)


def test_invalid_instruction_scale_larger_than_one():
    controller_name = "drive_2.awg"
    channel_description = ChannelDescription(IQChannelConfig(2.4e9), controller_name)
    mock_instruction = Instruction(64, RealPulse(Constant(n_samples=64), 1.2))
    channel_description.add_instruction(mock_instruction)
    error_string = "scale not in -1..1"
    with pytest.raises(InvalidInstructionError, match=error_string):
        validate_awg_and_schedule_compatibility(channel_description, mock_validation)


def test_invalid_instruction_scale_larger_than_one_iq():
    controller_name = "drive_2.awg"
    channel_description = ChannelDescription(IQChannelConfig(2.4e9), controller_name)

    wave = Constant(n_samples=64)
    mock_instruction = Instruction(64, IQPulse(wave, wave, scale_i=1.2, scale_q=0.1, phase=1))
    channel_description.add_instruction(mock_instruction)
    error_string = "scale not in -1..1"
    with pytest.raises(InvalidInstructionError, match=error_string):
        validate_awg_and_schedule_compatibility(channel_description, mock_validation)
