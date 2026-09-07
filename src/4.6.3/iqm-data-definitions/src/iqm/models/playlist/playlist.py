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
"""Implements the new data structure of a playlist."""

from dataclasses import dataclass, field
from functools import reduce
from typing import Any

from iqm.models.playlist.channel_descriptions import ChannelDescription
from iqm.models.playlist.segment import Segment


@dataclass()
class Playlist:
    """Information required to build a batch of programs for AWGs and readout instruments.

    Consists of a number of :class:`.Segment` s, executed in a sequence, and
    information about the properties of the control and readout channels used
    in the Segments, as well as :class:`.Instruction` and :class:`.Waveform` tables.

    This class implements the new data structure that contains all the data necessary
    for an experiment to execute. Playlist contains information of all the channels used as
    well as the instruction execution schedule.

    Args:
        channel_descriptions: Controller name mapped to channel and channel specific instruction
            and waveform data.
        segments: Contains all the segments in the order of execution.

    """

    channel_descriptions: dict[str, ChannelDescription] = field(default_factory=dict)
    segments: list[Segment] = field(default_factory=list)

    def add_channel(self, new_channel: ChannelDescription) -> None:
        """Adds a new channel to the Playlist.

        Args:
            new_channel: channel to add
        Raises:
            ValueError: channel with that name already exists, and has different properties

        """
        name = new_channel.controller_name
        if (old_channel := self.channel_descriptions.get(name)) and old_channel != new_channel:
            raise ValueError(
                f"Controller '{name}' already has a channel with different properties: {old_channel} -> {new_channel}"
            )
        self.channel_descriptions[name] = new_channel

    def __str__(self):
        """Helper function for debugging"""
        unique_waveforms = (
            "- "
            + str(
                reduce(
                    lambda a, b: a + b,
                    [len(channel.waveform_table) for channel in self.channel_descriptions.values()],
                )
            )
            + " unique waveforms"
        )
        unique_instructions = (
            "- "
            + str(
                reduce(
                    lambda a, b: a + b,
                    [len(channel.instruction_table) for channel in self.channel_descriptions.values()],
                )
            )
            + " unique instructions"
        )
        segments = f"- {len(self.segments)} segments"
        channel_descriptions = f"- {len(self.channel_descriptions)} channels"
        rest = ""
        for name, channel in self.channel_descriptions.items():
            rest += f" {name}:\n"
            for inst in channel.instruction_table:
                rest += f"    {inst}\n"
            rest += "\n"
            for wave in channel.waveform_table:
                rest += f"    {wave}\n"
            rest += "\n"

        return (
            f"Schedule info:\n {channel_descriptions}\n {segments}\n {unique_waveforms}"
            f"\n {unique_instructions}\n\n{rest}"
        )

    def view(
        self,
        *,
        filter_segments: list[int] | None = None,
        filter_channels: list[str] | None = None,
        kind: str = "operations",
    ) -> dict[int, dict[str, Any]]:
        """Return operations of the playlist grouped by segments and channel names for easy inspection.
        This method is not meant to be used for integration purposes, but rather for hands-on inspection and debugging.

        Args:
            filter_segments: List of segment indexes to filter by. If None, all segments are included.
            filter_channels: List of channel names to filter by. If None, all channels are included.
            kind: Type of data to return in the inner dict, either "strings", "operations" or "instructions".

        Returns:
            Dict of segments, where values are dicts: keys are channels, values are lists of
            operations or instructions, or string representations of the operations.

        Raises:
            ValueError: If kind is not "strings", "operations" or "instructions".

        """
        if kind not in ("operations", "instructions", "strings"):
            raise ValueError(f'kind must be either, "strings", "operations" or "instructions", got "{kind}"')

        result: dict[int, dict[str, Any]] = {}
        segment_indexes = filter_segments if filter_segments is not None else list(range(len(self.segments)))
        channel_names = filter_channels if filter_channels is not None else list(self.channel_descriptions.keys())

        for segment_index in segment_indexes:
            result[segment_index] = {}
            segment = self.segments[segment_index]
            for channel_name in channel_names:
                result[segment_index][channel_name] = []
                for instruction_idx in segment.instructions.get(channel_name, []):
                    instruction = self.channel_descriptions[channel_name].instruction_table[instruction_idx]
                    if kind == "operations":
                        item = instruction.operation
                    elif kind == "instructions":
                        item = instruction
                    elif kind == "strings":
                        item = (
                            f"{instruction_idx:<5}{type(instruction.operation).__name__}"
                            f"({instruction.duration_samples})"
                        )
                    else:
                        raise ValueError(kind)
                    result[segment_index][channel_name].append(item)

        return result
