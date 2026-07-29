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
"""Validation of playlists and instructions schedules."""

from __future__ import annotations

from dataclasses import dataclass

from iqm.models.playlist.channel_descriptions import ChannelDescription
import iqm.models.playlist.instructions as sc
from iqm.models.playlist.instructions import (
    ConditionalInstruction,
    Instruction,
    IQPulse,
    Operation,
    ReadoutTrigger,
    RealPulse,
)

from iqm.pulse.playlist.channel import ChannelProperties
import iqm.pulse.playlist.instructions as front
from iqm.pulse.playlist.playlist import Playlist


class PlaylistValidationError(Exception):
    """Error raised when Playlist validation fails."""


def validate_playlist_compatibility(playlist: Playlist, device_constraints: dict[str, ChannelProperties]) -> None:
    """Validates that the given playlist is compatible with the provided AWG data.

    The following requirements are validated as they are the same for all controllers.

    1. Playlist sample rate vs. Actual controller sample rate
    2. Instruction granularity with respect to the controllers granularity requirements
    3. Checks that no other than supported instructions are used.
    4. Checks that instruction length matches waveform length in IQ and RealPulse
    5. Checks that all instructions are at least the length of minimum number of samples
    6. Checks that Conditional instruction has the same duration in every child instruction

    Args:
        playlist: instructions used on each channel, as well as the channel configurations
        device_constraints: actual hardware limitations of the channels

    """
    for channel_name, channel_description in playlist.channel_descriptions.items():
        validate_channel(channel_description, device_constraints[channel_name])


def validate_channel(channel_description: ChannelDescription, device_constraints: ChannelProperties) -> None:
    """Validate a single channel."""
    channel_name = channel_description.controller_name
    # TODO remove once we use the iqm.pulse Instruction classes on both frontend and SC
    _instruction_map = {
        sc.Wait: front.Wait,
        sc.RealPulse: front.RealPulse,
        sc.IQPulse: front.IQPulse,
        sc.VirtualRZ: front.VirtualRZ,
        sc.ConditionalInstruction: front.ConditionalInstruction,
    }

    if not hasattr(channel_description.channel_config, "sampling_rate"):
        raise PlaylistValidationError("Channel configuration does not have sampling rate")

    # validate the channel config
    if channel_description.channel_config.sampling_rate != device_constraints.sample_rate:
        raise PlaylistValidationError(
            f"Sample rates do not match. "
            f"Device {channel_name} expects {device_constraints.sample_rate} "
            f"but got from playlist {channel_description.channel_config.sampling_rate}"
        )
    # validate the instructions
    allowed_names = tuple(map(lambda x: x.__name__, device_constraints.compatible_instructions))

    for instruction in channel_description.instruction_table:
        instruction_type = type(instruction.operation)
        mapped_type = _instruction_map[instruction_type]
        if mapped_type not in device_constraints.compatible_instructions:
            raise PlaylistValidationError(
                f"{channel_name}: Incompatible instruction {mapped_type.__name__} (allowed: {allowed_names})"
            )
        try:
            pass
            # TODO turn back on and remove pragma once we use the iqm.pulse Instruction classes on both frontend and SC
            # instruction.validate()
        except ValueError as ex:  # pragma: no cover
            raise PlaylistValidationError(ex) from ex

        if (instruction.duration_samples % device_constraints.instruction_duration_granularity) != 0:
            raise PlaylistValidationError(
                f"Instruction length of {instruction.duration_samples} doesn't match the "
                f"{channel_name} granularity of the device"
            )
        if instruction.duration_samples < device_constraints.instruction_duration_min:
            raise PlaylistValidationError(
                f"Instruction length of {instruction.duration_samples} is less than the minimum number "
                f"of samples {device_constraints.instruction_duration_min} for {channel_name}"
            )


@dataclass()
class AWGScheduleValidationData:
    """Controller specific validation data"""

    sampling_rate: float
    granularity: int
    min_number_of_samples: int
    compatible_instructions: tuple[type[Operation], ...]  # type: ignore[valid-type]


class AWGScheduleValidationError(Exception):
    """Error raised when schedule validation for an AWG fails."""


class InvalidInstructionError(Exception):
    """Error raised when encountering an invalid instruction."""

    def __init__(self, instruction, issue_string="unknown reason"):  # noqa: ANN001
        self.issue_string = issue_string
        self.instruction = instruction
        super().__init__(issue_string)

    def __str__(self):
        return f"{self.issue_string} (in {self.instruction})"


def validate_instruction_and_wf_length(instruction: Instruction):  # noqa: ANN201
    """Validate that instruction and waveform lengths match

    Args:
        instruction: The IQPulse or RealPulse to be validated

    """
    if isinstance(instruction.operation, RealPulse):
        if instruction.duration_samples != instruction.operation.wave.n_samples:
            raise InvalidInstructionError(instruction, "Instruction duration != waveform length")
        if abs(instruction.operation.scale) > 1.0:
            raise InvalidInstructionError(instruction, "scale not in -1..1")
    if isinstance(instruction.operation, IQPulse):
        if instruction.duration_samples != instruction.operation.wave_i.n_samples:
            raise InvalidInstructionError(instruction, "Instruction duration != waveform_i length")
        if instruction.duration_samples != instruction.operation.wave_q.n_samples:
            raise InvalidInstructionError(instruction, "Instruction duration != waveform_q length")
        if abs(instruction.operation.scale_i) > 1 or abs(instruction.operation.scale_q) > 1:
            raise InvalidInstructionError(instruction, "scale not in -1..1")


def validate_awg_and_schedule_compatibility(  # noqa: ANN201
    channel_description: ChannelDescription, device_constraints: AWGScheduleValidationData
):
    """Validates that the given playlist is compatible with the provided AWG data.
    The following requirements are validated as they are the same for all controllers.

    1. Playlist sampling rate vs. Actual controller sampling rate
    2. Instruction granularity with respect to the controllers granularity requirements
    3. Checks that no other than supported instructions are used.
    4. Checks that instruction length matches waveform length in IQ and RealPulse
    5. Checks that all instructions are at least the length of minimum number of samples
    6. Checks that Conditional instruction has the same duration in every child instruction

    Args:
        channel_description: Contains instructions used as well as the channel specific configuration from playlist
        device_constraints: Contains the actual hardware limitations

    """
    if channel_description is None:
        raise AWGScheduleValidationError("Controller was not found in the playlist")

    if not hasattr(channel_description.channel_config, "sampling_rate"):
        raise AWGScheduleValidationError("Channel configuration does not have sampling rate")

    if channel_description.channel_config.sampling_rate != device_constraints.sampling_rate:
        raise AWGScheduleValidationError(
            f"Sampling rates do not match. "
            f"Device {channel_description.controller_name} expects {device_constraints.sampling_rate} "
            f"but got from playlist {channel_description.channel_config.sampling_rate}"
        )
    instruction_table = channel_description.instruction_table
    for instruction in instruction_table:
        instruction_type = type(instruction.operation)
        if instruction_type not in device_constraints.compatible_instructions:
            raise AWGScheduleValidationError(
                f"Incompatible instruction {instruction_type.__name__} not compatible with "
                f"{channel_description.controller_name}"
            )
        if instruction_type == ConditionalInstruction:
            if_false_duration = instruction.operation.if_false.duration_samples
            if_true_duration = instruction.operation.if_true.duration_samples
            if if_false_duration != instruction.duration_samples or if_true_duration != instruction.duration_samples:
                raise AWGScheduleValidationError(
                    "Conditional Instruction's instructions must have the same duration\n"
                    f"Duration of Conditional instruction: {instruction.duration_samples} \n"
                    f"Duration of if_false instruction: {if_false_duration}\n"
                    f"Duration of if_true instruction: {if_true_duration}\n"
                )
        elif instruction_type == ReadoutTrigger:
            if instruction.duration_samples <= instruction.operation.probe_pulse.duration_samples:
                raise AWGScheduleValidationError(
                    "Duration of ReadoutTrigger must be longer than the probe pulse inside it."
                )

        if (instruction.duration_samples % device_constraints.granularity) != 0:
            raise AWGScheduleValidationError(
                f"Instruction length of {instruction.duration_samples} doesn't match the "
                f"{channel_description.controller_name} granularity of the device"
            )
        if instruction.duration_samples < device_constraints.min_number_of_samples:
            raise AWGScheduleValidationError(
                f"Instruction length of {instruction.duration_samples} is less than the minimum number "
                f"of samples {device_constraints.min_number_of_samples} for {channel_description.controller_name}"
            )
        validate_instruction_and_wf_length(instruction)
