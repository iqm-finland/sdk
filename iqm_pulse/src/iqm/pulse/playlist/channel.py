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
"""Control channel properties."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from math import ceil, lcm

from exa.common.data.setting_node import SettingNode
from exa.common.qcm_data.chip_topology import ChipTopology
from iqm.pulse.playlist.instructions import Instruction
from iqm.pulse.playlist.waveforms import Waveform


@dataclass(frozen=True)
class ChannelProperties:
    """Defines the properties of a control or measurement channel.

    All ZI instruments (HDAWG, UHFQA, SHFSG, SHFQA) can handle control pulses and waits where the
    number of samples is ``32 + n * 16``, where ``n in {0, 1, 2, ...}``.
    """

    sample_rate: float
    """sample rate of the instrument responsible for the channel (in Hz)"""

    instruction_duration_granularity: int
    """all instruction durations on this channel must be multiples of this granularity (in samples)"""

    instruction_duration_min: int
    """all instruction durations on this channel must at least this long (in samples)"""

    compatible_instructions: tuple[type[Instruction], ...] = field(default_factory=tuple)
    """instruction types that are allowed on this channel"""

    is_iq: bool = False
    """HACK, True iff this is an IQ channel. TODO do better"""

    is_virtual: bool = False
    """Virtual channels are only used on the frontend side during compilation and scheduling.
    They are removed from the :class:`~iqm.pulse.playlist.Schedule` before it is sent to Station
    Control. For example, virtual drive channels of computational resonators."""

    blocks_component: bool = True
    """Whether content in this channel should block the entire component that it is associated with in the scheduling.
    Typically all physical channels should block their components, but certain virtual channels might not
    require this."""

    def duration_to_samples(self, duration: float) -> float:
        """Convert a time duration to number of samples at the channel sample rate.

        Args:
            duration: time duration in s

        Returns:
            ``duration`` in samples

        """
        return duration * self.sample_rate

    def duration_to_seconds(self, duration: float) -> float:
        """Convert a time duration in samples at the channel sample rate to seconds.

        Args:
            duration: time duration in samples

        Returns:
            ``duration`` in seconds

        """
        return duration / self.sample_rate

    def duration_to_int_samples(
        self, duration: float, message: str = "Given duration", check_min_samples: bool = True
    ) -> int:
        """Convert a time duration to an integer number of samples at the channel sample rate.

        ``duration`` must be sufficiently close to an integer number of samples, and
        that number must be something the channel can handle.

        Args:
            duration: time duration in s
            message: message identifying the duration we are testing
            check_min_samples: If True, check that the output is at least :attr:`instruction_duration_min`.

        Returns:
            ``duration`` as an integer number of samples

        Raises:
            ValueError: ``duration`` is not close to an integer number of samples, or is
                otherwise unacceptable to the channel

        """
        message += f" ({duration} s at {self.sample_rate} Hz sample rate)"
        # Do rounding to account for floating point issues, so we only need to specify a reasonable number of decimals.
        # If the number of samples is within 0.005 samples of an integer number, we assume that's what the user meant.
        samples = round(duration * self.sample_rate, ndigits=2)
        if not samples.is_integer():
            raise ValueError(message + f" is {samples} samples, which is not an integer.")

        samples = int(samples)
        message += f" is {samples} samples, which is "
        if samples % self.instruction_duration_granularity != 0:
            raise ValueError(message + f"not an integer multiple of {self.instruction_duration_granularity} samples.")
        if check_min_samples and samples < self.instruction_duration_min:
            raise ValueError(message + f"less than {self.instruction_duration_min} samples.")
        return samples

    def round_duration_to_granularity(
        self,
        duration: float,
        round_up: bool = False,
        force_min_duration: bool = False,
    ) -> float:
        """Round a time duration to the channel granularity.

        Args:
            duration: time duration in s
            round_up: whether to round the durations up, or to the closest granularity
            force_min_duration: whether to force the duration to be at least :attr:`instruction_duration_min`

        Returns:
            ``duration`` rounded to channel granularity, in s

        """
        granularity = self.instruction_duration_granularity / self.sample_rate
        n_granularity = ceil(duration / granularity) if round_up else round(duration / granularity)
        rounded = n_granularity * granularity
        if force_min_duration:
            min_possible_duration = self.duration_to_seconds(self.instruction_duration_min)
            return max(min_possible_duration, rounded)
        return rounded


def round_duration_to_granularity_samples(
    channels: Iterable[ChannelProperties],
    duration: float,
    round_up: bool = False,
    force_min_duration: bool = False,
) -> int:
    """Round a time duration to the least common multiple of the granularities of the given channels.

    .. note:: Assumes that all the given control channels have the same sample rate.

    Args:
        channels: all these channels must be able to handle the rounded duration
        duration: time duration in s
        round_up: whether to round the duration up, or to the closest granularity
        force_min_duration: whether to force the duration to be at least the largest ``instruction_duration_min``
            of ``channels``

    Returns:
        ``duration`` rounded to common channel granularity, in samples

    """
    granularity: int = lcm(*(c.instruction_duration_granularity for c in channels))
    min_duration: int = max((c.instruction_duration_min for c in channels), default=0)

    # NOTE we assume that all the control channels here have the same sample rate!
    channel = next(iter(channels))  # arbitrary channel

    def round_to_granularity(duration_in_samples: float, round_up: bool) -> int:
        """Rounded duration in samples."""
        duration_in_granularity: float = duration_in_samples / granularity
        rounded_duration_in_granularity: int = (
            ceil(duration_in_granularity) if round_up else round(duration_in_granularity)
        )
        return rounded_duration_in_granularity * granularity

    rounded_duration_in_samples = round_to_granularity(channel.duration_to_samples(duration), round_up)
    if force_min_duration:
        # also the min duration needs to be rounded up to granularity
        min_duration = round_to_granularity(min_duration, round_up=True)
        return max(min_duration, rounded_duration_in_samples)
    return rounded_duration_in_samples


@dataclass(frozen=True)
class ProbeChannelProperties(ChannelProperties):
    """``ChannelProperties`` for probe line channels."""

    center_frequency: float = 0.0
    """Center frequency for the channel."""

    integration_start_dead_time: int = 0
    """Dead time samples before integration."""

    integration_stop_dead_time: int = 0
    """Dead time samples after integration."""


@dataclass(frozen=True)
class ChannelConfiguration:
    """Base class for configuring channels."""


@dataclass(frozen=True)
class RealChannelConfig(ChannelConfiguration):
    """Requested configuration of a real channel."""

    sample_rate: float
    """sample rate of the instrument responsible for the channel (in Hz)"""


@dataclass(frozen=True)
class IQChannelConfig(RealChannelConfig):
    """Requested configuration of an IQ channel."""

    frequency: float
    """upconversion frequency for the IQ pulses (in Hz)"""


@dataclass
class ChannelDescription:
    """Channel specific data, including tables for the instructions and waveforms used.

    Args:
        name: name of the controller handling the channel, also the name of the channel
        config: properties of the channel
        instruction_table: mapping of the instructions to be executed on this channel. Each
            instruction should be unique.
        waveform_table: Contains mapping of the waveforms to be executed on this channel. Each
            waveform should be unique.

    """

    name: str
    config: RealChannelConfig
    instruction_table: list[Instruction] = field(repr=False, init=False, default_factory=list)
    waveform_table: list[Waveform] = field(repr=False, init=False, default_factory=list)

    _reverse_instruction_index: dict[Instruction, int] = field(repr=False, init=False, default_factory=dict)
    _reverse_waveform_index: dict[Waveform, int] = field(repr=False, init=False, default_factory=dict)

    def add_instruction(self, instruction: Instruction) -> int:
        """Add an instruction to the channel.

        Each unique instruction in a channel gets assigned an integer index that can be used to refer to it.
        If the instruction has associated :class:`Waveform` s, they are indexed in a similar manner.

        Args:
            instruction: instruction to be added

        Returns:
            index of the instruction

        """
        for child in instruction.get_child_instructions():
            self.add_instruction(child)

        for wave in instruction.get_waveforms():
            self._lookup_or_insert_waveform(wave)

        return self._lookup_or_insert_instruction(instruction)

    def _lookup_or_insert_waveform(self, wave: Waveform) -> int:
        new_idx = len(self.waveform_table)
        idx = self._reverse_waveform_index.setdefault(wave, new_idx)
        if idx == new_idx:
            self.waveform_table.append(wave)
        return idx

    def _lookup_or_insert_instruction(self, instruction) -> int:  # noqa: ANN001
        new_idx = len(self.instruction_table)
        idx = self._reverse_instruction_index.setdefault(instruction, new_idx)
        if idx == new_idx:
            self.instruction_table.append(instruction)
        return idx


def get_channel_properties_from_station_settings(
    settings: SettingNode, chip_topology: ChipTopology
) -> tuple[dict[str, ChannelProperties], dict[str, dict[str, str]]]:
    """Get channel properties from Station Control controller settings following the standard convention.

    Args:
        settings: Flat tree of all controllers.
        chip_topology: Chip topology.

    Returns:
        channel_properties: mapping from channel name to its properties
        component_to_channel: mapping from chip component to function name to channel name.

    """
    ### Naming convention for controllers.
    READOUT_CONTROLLER = "{}__readout"
    DRIVE_CONTROLLER = "{}__drive"
    FLUX_CONTROLLER = "{}__flux"

    drive_controllers = {}
    flux_controllers = {}
    readout_controllers = {}
    for component in chip_topology.probe_lines:
        if (node := settings.subtrees.get(READOUT_CONTROLLER.format(component))) is not None:
            readout_controllers[component] = node
    for component in chip_topology.qubits_sorted + chip_topology.couplers_sorted:
        if (node := settings.subtrees.get(DRIVE_CONTROLLER.format(component))) is not None:
            if (awg_node := node.subtrees.get("awg")) is not None:
                drive_controllers[component] = awg_node
        if (node := settings.subtrees.get(FLUX_CONTROLLER.format(component))) is not None:
            if (awg_node := node.subtrees.get("awg")) is not None:
                flux_controllers[component] = awg_node

    return get_channel_properties(chip_topology, drive_controllers, flux_controllers, readout_controllers)


FAST_FEEDBACK_CHANNEL_TEMPLATE = "{PROBE}__feedforward_bits_to_{AWG}"


def get_channel_properties(
    chip_topology: ChipTopology,
    drive_controllers: dict[str, SettingNode],
    flux_controllers: dict[str, SettingNode],
    readout_controllers: dict[str, SettingNode],
) -> tuple[dict[str, ChannelProperties], dict[str, dict[str, str]]]:
    """Internal function to get channel properties."""
    # TODO should this info come from a dedicated SC endpoint instead, so we would not have to parse SC settings:
    # EXA-1777

    RESONATOR_VIRTUAL_DRIVE_CHANNEL = "{}__drive_virtual"
    # Computational resonators use virtual drive channels to store VirtualRZ instructions coming
    # from two-component gates applied on them. These virtual channels are eventually emptied in a
    # schedule-level pass, and are never sent to the station. The z rotations are applied on one of
    # the qubits instead. See :class:`.MoveMarker`."""

    channel_properties: dict[str, ChannelProperties] = {}
    awg_sampling_rates = set()
    component_to_channel: dict[str, dict] = {}

    for component, node in readout_controllers.items():
        component_to_channel.setdefault(component, {})["readout"] = node.name

        channel_properties[node.name] = ProbeChannelProperties(
            sample_rate=node.sampling_rate.value,
            instruction_duration_granularity=node.instruction_duration_granularity.value,
            instruction_duration_min=node.instruction_duration_min.value,
            integration_start_dead_time=node.integration_start_dead_time.value,
            integration_stop_dead_time=node.integration_stop_dead_time.value,
            center_frequency=node.center_frequency.value,
        )
        component_to_channel.setdefault(component, {})["readout"] = node.name

    for function, controllers, is_iq in [("drive", drive_controllers, True), ("flux", flux_controllers, False)]:
        for component, awg_node in controllers.items():
            component_to_channel.setdefault(component, {})[function] = awg_node.name
            channel_properties[awg_node.name] = ChannelProperties(
                sample_rate=awg_node.sampling_rate.value,
                instruction_duration_granularity=awg_node.instruction_duration_granularity.value,
                instruction_duration_min=awg_node.instruction_duration_min.value,
                is_iq=is_iq,
            )
            awg_sampling_rates.add(awg_node.sampling_rate.value)
            if "feedback_sources" in awg_node.children:
                for source in awg_node.feedback_sources.value:
                    probe_line = source.split("__readout")[0]
                    if probe_line not in component_to_channel:
                        component_to_channel[probe_line] = {}
                    channel_name = FAST_FEEDBACK_CHANNEL_TEMPLATE.replace("{PROBE}", probe_line).replace(
                        "{AWG}", awg_node.name
                    )
                    channel_properties[channel_name] = ChannelProperties(
                        sample_rate=awg_node.sampling_rate.value,
                        instruction_duration_granularity=awg_node.instruction_duration_granularity.value,
                        instruction_duration_min=0,
                        is_virtual=True,
                        blocks_component=False,
                    )
                    component_to_channel[component][f"feedback_from_{probe_line}"] = channel_name
                    component_to_channel[probe_line][f"feedback_to_{awg_node.name}"] = channel_name

    # sample rate for virtual channels
    if len(awg_sampling_rates) > 1:
        raise ValueError(
            "ScheduleBuilder supports only a single sample rate for all AWG channels."
            " The station contains these sample rates:\n"
            f"{awg_sampling_rates}"
        )
    if awg_sampling_rates:
        rate = next(iter(awg_sampling_rates))

        # add virtual drive channels for all comp. resonators
        for component in chip_topology.computational_resonators_sorted:
            channel_name = RESONATOR_VIRTUAL_DRIVE_CHANNEL.format(component)
            component_to_channel.setdefault(component, {})["drive"] = channel_name
            channel_properties[channel_name] = ChannelProperties(
                sample_rate=rate,
                instruction_duration_granularity=1,
                instruction_duration_min=0,
                is_iq=True,
                is_virtual=True,
            )
    return channel_properties, component_to_channel
