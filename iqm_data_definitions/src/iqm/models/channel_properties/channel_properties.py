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
"""Hardware properties of station devices."""

from dataclasses import dataclass, field

from iqm.models.playlist.instructions import Instruction
import numpy as np


@dataclass()
class ChannelProperties:
    """Parent class of AWG and QA Channel properties that contains common attributes and methods."""

    sampling_rate: float
    """Sample rate of the instrument responsible for the channel (in Hz)."""

    instruction_duration_granularity: float
    """All instruction durations on this channel must be multiples of this granularity (in samples)."""

    instruction_duration_min: float
    """All instruction durations on this channel must at least this long (in samples)."""

    compatible_instructions: tuple[type[Instruction], ...] = field(default_factory=tuple)
    """Instruction types that are allowed on this channel."""

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
        return duration * self.sampling_rate

    def duration_to_seconds(self, duration: float) -> float:
        """Convert a time duration in samples at the channel sample rate to seconds.

        Args:
            duration: time duration in samples

        Returns:
            ``duration`` in seconds
        """
        return duration / self.sampling_rate

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
        message += f" ({duration} s at {self.sampling_rate} Hz sample rate)"
        # Do rounding to account for floating point issues, so we only need to specify a reasonable number of decimals.
        # If the number of samples is within 0.005 samples of an integer number, we assume that's what the user meant.
        samples = round(duration * self.sampling_rate, ndigits=2)
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
            round_up: whether to round the durations up to the closest granularity
            force_min_duration: whether to force the duration to be at least ``self.instruction_duration_min`` in
                seconds

        Returns:
            ``duration`` rounded to channel granularity, in seconds
        """
        granularity = self.instruction_duration_granularity / self.sampling_rate
        n_granularity = np.ceil(duration / granularity) if round_up else round(duration / granularity)
        rounded = n_granularity * granularity
        if force_min_duration:
            min_possible_duration = self.duration_to_seconds(self.instruction_duration_min)
            return max(min_possible_duration, rounded)
        return rounded



@dataclass(kw_only=True)
class AWGProperties(ChannelProperties):
    "Channel properties of an AWG channel."
    fast_feedback_sources: list[str] = field(default_factory=list)
    """Defines compatible fast feedback sources"""
    local_oscillator: bool = False
    """Whether this AWG contains a local oscillator or not."""
    mixer_correction: bool = False
    """Whether this AWG has mixer correction or not."""


@dataclass(kw_only=True)
class ReadoutProperties(ChannelProperties):
    """Channel properties of a QA channel."""
    integration_start_dead_time: float
    """ Minimum delay for probe pulse entries inside a ReadoutTrigger."""
    integration_stop_dead_time: float
    """
    Minimum delay after the last integrator has stopped, before a new
    ReadoutTrigger can be executed. This delay must be taken into account when calculating the duration of a
    ReadoutTrigger. The duration is the sum of:

        * This value,
        * The duration of the longest integration among the acquisitions in the ReadoutTrigger instruction,
        * The acquisition delay of the above integration.
    """
