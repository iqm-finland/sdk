# Copyright 2019-2025 IQM
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
""" Channel Description and Channel Configuration definitions."""
from dataclasses import dataclass, field

from iqm.models.playlist.instructions import (
    AcquisitionMethod,
    ComplexIntegration,
    ConditionalInstruction,
    Instruction,
    IQPulse,
    MultiplexedIQPulse,
    MultiplexedRealPulse,
    ReadoutTrigger,
    RealPulse,
    ThresholdStateDiscrimination,
)
from iqm.models.playlist.waveforms import CanonicalWaveform

# pylint: disable=too-few-public-methods,too-many-instance-attributes


@dataclass(frozen=True)
class ChannelConfiguration:
    """Base type for all channel configurations."""


@dataclass(frozen=True)
class IQChannelConfig(ChannelConfiguration):
    """Placeholder configuration for Complex Channels"""

    sampling_rate: float


@dataclass(frozen=True)
class RealChannelConfig(ChannelConfiguration):
    """Placeholder configuration for Real Channels"""

    sampling_rate: float


@dataclass(frozen=True)
class ReadoutChannelConfig(ChannelConfiguration):
    """Requested configuration of a readout channel."""

    sampling_rate: float


@dataclass
class ChannelDescription:
    """ChannelDescription class contains all channel specific data and the suitable instructions and waveforms
        for a channel.

    Args:
        channel_config: ChannelConfiguration object which contains data related to the channel
        controller_name: name of the controller
        instruction_table: Contains mapping of the instructions to be executed on this channel. Each
            instruction should be unique.
        waveform_table: Contains mapping of the waveforms to be executed on this channel. Each
            waveform should be unique.
        acquisition_table: Table of acquisition configs.
            Each ReadoutTrigger instruction may ask to perform an arbitrary combination of these.
            In practice, possible combinations are limited by device capabilities.
    """

    channel_config: ChannelConfiguration
    controller_name: str
    instruction_table: list[Instruction] = field(repr=False, init=False, default_factory=list)
    waveform_table: list[CanonicalWaveform] = field(repr=False, init=False, default_factory=list)
    acquisition_table: list[AcquisitionMethod] = field(repr=False, init=False, default_factory=list)

    _reverse_waveform_index: dict[CanonicalWaveform, int] = field(repr=False, init=False, default_factory=dict)
    _reverse_instruction_index: dict[Instruction, int] = field(repr=False, init=False, default_factory=dict)
    _reverse_acquisition_index: dict[AcquisitionMethod, int] = field(repr=False, init=False, default_factory=dict)

    def add_instruction(self, instruction: Instruction) -> int:
        """Add an instruction to the instruction table if the instruction is unique.

        If the instruction contains other instructions, those are also added if they are unique.
        If any of the instructions contain unique waveforms, those are also added to the waveform table.

        Args:
            instruction: Instruction to be added
        Returns:
            corresponding index to the instruction table
        """
        match instruction.operation:
            case IQPulse(wave_i=wave_i, wave_q=wave_q):
                self._lookup_or_insert_waveform(wave_i)
                self._lookup_or_insert_waveform(wave_q)
            case RealPulse(wave=wave):
                self._lookup_or_insert_waveform(wave)
            case ConditionalInstruction(if_true=if_true, if_false=if_false):
                self.add_instruction(if_true)
                self.add_instruction(if_false)
            case MultiplexedIQPulse(entries=entries) | MultiplexedRealPulse(entries=entries):
                for entry in entries:
                    self.add_instruction(entry[0])
            case ReadoutTrigger(probe_pulse=probe_pulse, acquisitions=acquisitions):
                self.add_instruction(probe_pulse)
                for acq in acquisitions:
                    self.add_acquisition(acq)

        return self._lookup_or_insert_instruction(instruction)

    def add_acquisition(self, acquisition: AcquisitionMethod) -> int:
        """Add an acquisition method to the table map if the configuration is unique.

        If it contains unique waveforms, those are added to the waveform table.

        Args:
            acquisition: Configuration to be added.
        Returns:
            Corresponding index to the acquisition table.
        """
        if not isinstance(self.channel_config, ReadoutChannelConfig):
            raise ValueError(f"Channel {self.controller_name} is not a readout channel, cannot add {acquisition}.")
        if isinstance(acquisition, ComplexIntegration | ThresholdStateDiscrimination):
            self._lookup_or_insert_waveform(acquisition.weights.wave_i)
            self._lookup_or_insert_waveform(acquisition.weights.wave_q)
        return self._lookup_or_insert_acquisition(acquisition)

    def _lookup_or_insert_waveform(self, waveform: CanonicalWaveform) -> int:
        new_idx = len(self.waveform_table)
        idx = self._reverse_waveform_index.setdefault(waveform, new_idx)
        if idx == new_idx:
            self.waveform_table.append(waveform)
        return idx

    def _lookup_or_insert_instruction(self, instruction: Instruction) -> int:
        new_idx = len(self.instruction_table)
        idx = self._reverse_instruction_index.setdefault(instruction, new_idx)
        if idx == new_idx:
            self.instruction_table.append(instruction)
        return idx

    def _lookup_or_insert_acquisition(self, acquisition: AcquisitionMethod) -> int:
        new_idx = len(self.acquisition_table)
        idx = self._reverse_acquisition_index.setdefault(acquisition, new_idx)
        if idx == new_idx:
            self.acquisition_table.append(acquisition)
        return idx
