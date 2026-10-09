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

from typing import TypeAlias

from iqm.models.channel_properties import ChannelProperties as sc_ChannelProperties
from iqm.models.playlist.channel_descriptions import ChannelDescription
import iqm.models.playlist.instructions as sc
from iqm.models.playlist.instructions import (
    ConditionalInstruction,
    Instruction,
    IQPulse,
    MultiplexedIQPulse,
    ReadoutTrigger,
    RealPulse,
    VirtualRZ,
    Wait,
)

from exa.common.errors.iqm_error import ValidationError
from iqm.pulse.playlist.channel import ChannelProperties
import iqm.pulse.playlist.instructions as front
from iqm.pulse.playlist.playlist import Playlist

OperationClass: TypeAlias = (
    type[Wait]
    | type[IQPulse]
    | type[RealPulse]
    | type[VirtualRZ]
    | type[ConditionalInstruction]
    | type[MultiplexedIQPulse]
    | type[ReadoutTrigger]
)
"""Union of operation classes station-control actually uses, across AWG and readout controllers.

Local alias because `iqm.models.playlist.instructions.Operation` is a constrained TypeVar meant only to
parametrize `Instruction[Operation]`, not a general-purpose union type for typing sets/tuples of classes.
"""


class PlaylistValidationError(ValidationError):
    """Raised when Playlist validation fails.

    Args:
        channel: Name of the offending control channel.
        issue_string: Explanation why it is invalid.

    """

    def __init__(self, channel: str, issue_string: str = "unknown reason") -> None:
        self.channel = channel
        self.issue_string = issue_string
        message = f"{channel}: {issue_string}"
        super().__init__(message=message, error_code="playlist_validation_error")


class InvalidInstructionError(ValidationError):
    """Raised when an instruction is malformed or not supported by the target instrument.

    Args:
        instruction: Offending instruction.
        channel: Name of the control channel it is on.
        issue_string: Explanation why it is invalid.

    """

    def __init__(self, instruction: Instruction, channel: str, issue_string: str = "unknown reason") -> None:
        self.instruction = instruction
        self.channel = channel
        self.issue_string = issue_string
        message = f"{channel}: {issue_string}: {instruction}"
        super().__init__(message=message, error_code="invalid_instruction_error")


def validate_playlist_compatibility(playlist: Playlist, device_constraints: dict[str, ChannelProperties]) -> None:
    """Validate that the given playlist is compatible with the control instrument constraints.

    Checks the channel configuration and its contents against the controller/instrument hardware constraints.

    The following requirements are validated:

    1. Channel sampling rate vs. instrument sampling rate.
    2. Only supported instructions are used.
    3. :meth:`Instruction.validate` passes.
    4. Instruction durations match instrument granularity.
    5. Instructions durations are at least the minimum number of samples for the instrument.

    Args:
        playlist: Instructions used on each channel, as well as the channel configurations.
        device_constraints: Hardware limitations of the control instruments.

    Raises:
        ValidationError: Not compatible.

    """
    for channel_name, channel_description in playlist.channel_descriptions.items():
        _validate_channel(channel_description, device_constraints[channel_name])


def _validate_channel(channel_description: ChannelDescription, device_constraints: ChannelProperties) -> None:
    """Validate a single channel."""
    # TODO merge with validate_channel_and_instrument_compatibility once we use the
    # iqm.pulse Instruction classes on both frontend and SC

    channel = channel_description.controller_name
    _instruction_map = {
        sc.Wait: front.Wait,
        sc.RealPulse: front.RealPulse,
        sc.IQPulse: front.IQPulse,
        sc.VirtualRZ: front.VirtualRZ,
        sc.ConditionalInstruction: front.ConditionalInstruction,
    }

    if not hasattr(channel_description.channel_config, "sampling_rate"):
        raise PlaylistValidationError(channel, "Channel configuration does not have sampling rate")

    # validate the channel config
    if channel_description.channel_config.sampling_rate != device_constraints.sample_rate:
        raise PlaylistValidationError(
            channel,
            f"Sample rates do not match. "
            f"Device expects {device_constraints.sample_rate} "
            f"but got from playlist {channel_description.channel_config.sampling_rate}",
        )
    # validate the instructions
    for instruction in channel_description.instruction_table:
        instruction_type = type(instruction.operation)
        mapped_type = _instruction_map[instruction_type]
        if mapped_type not in device_constraints.compatible_instructions:
            raise InvalidInstructionError(instruction, channel, "Device does not support instruction type")
        try:
            pass
            # TODO turn back on and remove pragma once we use the iqm.pulse Instruction classes on both frontend and SC
            # instruction.validate()
        except ValueError as ex:  # pragma: no cover
            raise InvalidInstructionError(instruction, channel, str(ex)) from ex

        granularity = device_constraints.instruction_duration_granularity
        if (instruction.duration_samples % granularity) != 0:
            raise InvalidInstructionError(
                instruction, channel, f"Duration doesn't match the granularity {granularity} of the device"
            )
        min_duration = device_constraints.instruction_duration_min
        if instruction.duration_samples < min_duration:
            raise InvalidInstructionError(
                instruction, channel, f"Duration is less than the minimum {min_duration} for the device"
            )


def _validate_instruction_and_wf_duration(instruction: Instruction, channel: str) -> None:
    """Validate that instruction and waveform durations match.

    Args:
        instruction: The IQPulse or RealPulse to be validated.
        channel: Name of the control channel ``instruction`` is on.

    """
    if isinstance(instruction.operation, RealPulse):
        if instruction.duration_samples != instruction.operation.wave.n_samples:
            raise InvalidInstructionError(instruction, channel, "duration != waveform length")
        if abs(instruction.operation.scale) > 1.0:
            raise InvalidInstructionError(instruction, channel, "scale not in -1..1")
    if isinstance(instruction.operation, IQPulse):
        if instruction.duration_samples != instruction.operation.wave_i.n_samples:
            raise InvalidInstructionError(instruction, channel, "duration != waveform_i length")
        if instruction.duration_samples != instruction.operation.wave_q.n_samples:
            raise InvalidInstructionError(instruction, channel, "duration != waveform_q length")
        if abs(instruction.operation.scale_i) > 1 or abs(instruction.operation.scale_q) > 1:
            raise InvalidInstructionError(instruction, channel, "scale not in -1..1")


def validate_channel_and_instrument_compatibility(
    channel_description: ChannelDescription,
    device_constraints: sc_ChannelProperties,
) -> None:
    """Validate that the given control channel and its contents are compatible with the instrument.

    Checks the channel configuration and its contents against the controller/instrument hardware constraints.

    The following requirements are validated:

    1. Channel sampling rate vs. instrument sampling rate.
    2. Only supported instructions are used.
    3. ConditionalInstruction has the same duration in every outcome.
    4. ReadoutTrigger duration is longer than the probe pulse in it.
    5. Instruction durations match instrument granularity.
    6. Instructions durations are at least the minimum number of samples for the instrument.
    7. Instruction durations match waveform lengths in IQ and RealPulses.

    Args:
        channel_description: Channel config and its instruction table from a Playlist.
        device_constraints: Hardware limitations of the control instrument.

    Raises:
        ValidationError: Not compatible.

    """
    if not hasattr(channel_description.channel_config, "sampling_rate"):
        raise PlaylistValidationError("Channel configuration does not have sampling rate")

    channel = channel_description.controller_name

    if channel_description.channel_config.sampling_rate != device_constraints.sampling_rate:
        raise PlaylistValidationError(
            f"Sampling rates do not match. "
            f"Device {channel} expects {device_constraints.sampling_rate} "
            f"but got from playlist {channel_description.channel_config.sampling_rate}"
        )
    for instruction in channel_description.instruction_table:
        instruction_type = type(instruction.operation)
        if instruction_type not in device_constraints.compatible_instructions:
            raise InvalidInstructionError(instruction, channel, "Device does not support instruction type")
        if instruction_type == ConditionalInstruction:
            if_false_duration = instruction.operation.if_false.duration_samples
            if_true_duration = instruction.operation.if_true.duration_samples
            if if_false_duration != instruction.duration_samples or if_true_duration != instruction.duration_samples:
                raise InvalidInstructionError(
                    instruction,
                    channel,
                    "ConditionalInstruction outcomes must have the same duration as the parent",
                )
        elif instruction_type == ReadoutTrigger:
            if instruction.duration_samples <= instruction.operation.probe_pulse.duration_samples:
                raise InvalidInstructionError(
                    instruction, channel, "Duration of ReadoutTrigger must be longer than the probe pulse in it"
                )

        granularity = device_constraints.instruction_duration_granularity
        if (instruction.duration_samples % granularity) != 0:
            raise InvalidInstructionError(
                instruction, channel, f"Duration doesn't match the granularity {granularity} of the device"
            )
        min_duration = device_constraints.instruction_duration_min
        if instruction.duration_samples < min_duration:
            raise InvalidInstructionError(
                instruction, channel, f"Duration is less than the minimum {min_duration} for the device"
            )
        _validate_instruction_and_wf_duration(instruction, channel)
