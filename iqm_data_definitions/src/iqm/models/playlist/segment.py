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
""" Segment class definitions
"""
from dataclasses import dataclass, field

from iqm.models.playlist.channel_descriptions import ChannelDescription
from iqm.models.playlist.instructions import Instruction


@dataclass(frozen=True)
class Segment:
    """Contains the instructions to be executed in one segment for each channel.

    Args:
        instructions: dict containing controller name as the key and the list of Instruction indices to be executed
        as the value.
    """

    instructions: dict[str, list[int]] = field(default_factory=dict)

    def add_to_segment(self, channel_description: ChannelDescription, instruction: Instruction):
        """Adds an instruction to the segment for a specific channel. Also calls the add_instruction of the channel
        for adding the instruction to the channels waveform map.

        Args:
            channel_description: The target :class:`.ChannelDescription` object
            instruction: The instruction to be added
        """
        idx = channel_description.add_instruction(instruction)
        self.instructions.setdefault(channel_description.controller_name, []).append(idx)
