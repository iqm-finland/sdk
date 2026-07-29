# Copyright 2025 IQM
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
"""Serializers and deserializers for :class:`~iqm.models.channel_properties.channel_properties.ChannelProperties`"""

from collections.abc import Iterable

import iqm.data_definitions.common.v1.channel_properties_pb2 as pb
from iqm.models.channel_properties import AWGProperties, ChannelProperties, ReadoutProperties
from iqm.models.playlist.instructions import (
    ConditionalInstruction,
    IQPulse,
    MultiplexedIQPulse,
    Operation,
    ReadoutTrigger,
    RealPulse,
    VirtualRZ,
    Wait,
)


def serialize_channel_properties(
    channel_property_dictionary: dict[str, ChannelProperties],
) -> pb.ChannelPropertyDictionary:
    """Pack the given dictionary of channel properties into a protobuf format for further serialization.

    Args:
        channel_properties: channel properties to pack

    Returns:
        ``ChannelPropertyDictionary``

    """
    pb_channel_prop_dict = pb.ChannelPropertyDictionary()
    for name, channel_properties in channel_property_dictionary.items():
        string_types = [instr_type.__name__ for instr_type in channel_properties.compatible_instructions]
        pb_channel_props = pb.ChannelProperties(
            sampling_rate=channel_properties.sampling_rate,
            instruction_duration_granularity=channel_properties.instruction_duration_granularity,
            instruction_duration_min=channel_properties.instruction_duration_min,
            compatible_instructions=string_types,
        )
        if isinstance(channel_properties, AWGProperties):
            channel_props = pb.AWGProperties(
                channel_properties=pb_channel_props,
                fast_feedback_sources=channel_properties.fast_feedback_sources,
                local_oscillator=channel_properties.local_oscillator,
                mixer_correction=channel_properties.mixer_correction,
            )
            pb_channel_prop_dict.channel_property_mapping[name].CopyFrom(pb.ChannelPropertyEntry(awg=channel_props))
        elif isinstance(channel_properties, ReadoutProperties):
            channel_props = pb.ReadoutProperties(
                channel_properties=pb_channel_props,
                integration_start_dead_time=channel_properties.integration_start_dead_time,
                integration_stop_dead_time=channel_properties.integration_stop_dead_time,
            )
            pb_channel_prop_dict.channel_property_mapping[name].CopyFrom(pb.ChannelPropertyEntry(ro=channel_props))
    return pb_channel_prop_dict


def deserialize_instructions(instructions: Iterable[str]) -> tuple[Operation, ...]:
    """Convert a repeated scalar container of instruction type strings into a tuple of python types."""
    instruction_types = []
    for instr_type in instructions:
        match instr_type:
            case "VirtualRZ":
                instruction_types.append(VirtualRZ)
            case "ConditionalInstruction":
                instruction_types.append(ConditionalInstruction)
            case "Wait":
                instruction_types.append(Wait)
            case "RealPulse":
                instruction_types.append(RealPulse)
            case "IQPulse":
                instruction_types.append(IQPulse)
            case "ReadoutTrigger":
                instruction_types.append(ReadoutTrigger)
            case "MultiplexedIQPulse":
                instruction_types.append(MultiplexedIQPulse)
            case _:
                raise ValueError(f"Unknown instruction type {instr_type}")
    return tuple(instruction_types)


def deserialize_channel_properties(
    channel_property_dictionary: pb.ChannelPropertyDictionary, convert_instructions: bool = True
) -> dict[str, ChannelProperties]:
    """Convert the given protobuf dictionary of channel properties into a dictionary of ``ChannelProperties``.

    Args:
        channel_properties_dictionary: channel property dictionary in protobuf format
        convert_instructions: whether to convert string representation of instruction types to actual
            python types.

    Returns:
        dictionary of channel properties

    """
    channel_prop_dict = {}
    for name, props in channel_property_dictionary.channel_property_mapping.items():
        match props.WhichOneof("value"):
            case "awg":
                instructions = (
                    deserialize_instructions(props.awg.channel_properties.compatible_instructions)
                    if convert_instructions
                    else list(props.awg.channel_properties.compatible_instructions)
                )
                channel_prop_dict[name] = AWGProperties(
                    sampling_rate=props.awg.channel_properties.sampling_rate,
                    instruction_duration_granularity=props.awg.channel_properties.instruction_duration_granularity,
                    instruction_duration_min=props.awg.channel_properties.instruction_duration_min,
                    fast_feedback_sources=list(props.awg.fast_feedback_sources),
                    compatible_instructions=instructions,
                    local_oscillator=props.awg.local_oscillator,
                    mixer_correction=props.awg.mixer_correction,
                )
            case "ro":
                instructions = (
                    deserialize_instructions(props.ro.channel_properties.compatible_instructions)
                    if convert_instructions
                    else list(props.ro.channel_properties.compatible_instructions)
                )
                channel_prop_dict[name] = ReadoutProperties(
                    sampling_rate=props.ro.channel_properties.sampling_rate,
                    instruction_duration_granularity=props.ro.channel_properties.instruction_duration_granularity,
                    instruction_duration_min=props.ro.channel_properties.instruction_duration_min,
                    compatible_instructions=instructions,
                    integration_start_dead_time=props.ro.integration_start_dead_time,
                    integration_stop_dead_time=props.ro.integration_stop_dead_time,
                )
    return channel_prop_dict


def unpack_channel_properties(payload: bytes, convert_instructions: bool = True) -> dict[str, ChannelProperties]:
    """Parse the Channel Property Dictionary from a string serialised protobuf payload.

    Args:
        payload: protobuf serialised payload of channel property dictionary.
        convert_instructions: whether to convert string representation of instruction types to actual
            python types.

    """
    pb_channel_properties = pb.ChannelPropertyDictionary()
    pb_channel_properties.ParseFromString(payload)
    return deserialize_channel_properties(pb_channel_properties, convert_instructions)
