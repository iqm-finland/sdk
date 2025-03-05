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
"""Instruction definitions."""
from __future__ import annotations

# pylint: disable=unused-import
from dataclasses import dataclass
from typing import Generic, TypeVar

from iqm.models.playlist.waveforms import CanonicalWaveform

@dataclass(frozen=True)
class Wait:
    """Class for WaitInstruction."""

    def __repr__(self):
        return f"{self.__class__.__name__}()"


@dataclass(frozen=True)
class VirtualRZ:
    """Class for Virtual Rz."""

    phase_increment: float


@dataclass(frozen=True)
class RealPulse:
    """Class for Real pulses. Contains a waveform object describing the waveform shape."""

    wave: CanonicalWaveform
    scale: float


@dataclass(frozen=True)
class IQPulse:
    """Class for IQ Pulses. Contains waveforms for both in-phase and quadrature waveforms."""

    wave_i: CanonicalWaveform
    wave_q: CanonicalWaveform
    scale_i: float
    scale_q: float
    phase: float = 0.0
    modulation_frequency: float = 0.0
    phase_increment: float = 0.0


@dataclass(frozen=True)
class ConditionalInstruction:
    """Class for Conditional Pulses."""

    condition: str
    if_true: Instruction
    if_false: Instruction


@dataclass(frozen=True)
class MultiplexedIQPulse:
    """Instruction to simultaneously play multiple IQ pulses.

    Each component pulse entry can be added with arbitrary delay from the beginning of this instruction.
    Where outside the duration of the MultiplexedIQPulse, the pulse entries are truncated.
    Where overlapping, samples of multiple pulse entries are summed.
    Where the interval of a MultiplexedIQPulse does not overlap with any of its entry pulse, its samples will be 0.
    """

    entries: tuple[tuple[Instruction, int], ...]
    """Pairs of instruction and `offset`.
    Instruction should be an IQPulse.
    `offset` is the number of samples the pulse is delayed from the beginning of the instruction.
    It has no granularity constraints. Negative values are allowed, but beginning will be truncated.
    """


@dataclass(frozen=True)
class MultiplexedRealPulse:
    """Instruction to simultaneously play multiple real pulses.

    Each component pulse entry can be added with arbitrary delay from the beginning of this instruction.
    Where outside the duration of the MultiplexedRealPulse, the pulse entries are truncated.
    Where overlapping, samples of multiple pulse entries are summed.
    Where the interval of a MultiplexedRealPulse does not overlap with any of its entry pulse, its samples will be 0.
    """

    entries: tuple[tuple[Instruction, int], ...]
    """Pairs of instruction and `offset`.
    Instruction should be valid a RealPulse.
    `offset` is the number of samples the pulse is delayed from the beginning of the instruction.
    It has no granularity constraints. Negative values are allowed, but beginning will be truncated.
    """


@dataclass(frozen=True)
class AcquisitionMethod:
    """Describes a way to acquire readout data."""

    label: str
    """Identifier for the returned data, like ``QB1__readout.time_trace``."""
    delay_samples: int
    """Delay from beginning of probe pulse to beginning of acquisition window, in samples."""


@dataclass(frozen=True)
class TimeTrace(AcquisitionMethod):
    """Capture the raw IQ signal without integration."""

    duration_samples: int
    """Length of the capture window, in samples."""


@dataclass(frozen=True)
class ComplexIntegration(AcquisitionMethod):
    """Perform a weighted integration of the IQ raw signal, resulting in a complex number."""

    weights: IQPulse
    """Integration weights."""


@dataclass(frozen=True)
class ThresholdStateDiscrimination(ComplexIntegration):
    """Perform a weighted integration of the IQ raw signal and compares the real part of the result against a
    threshold value, resulting in a single bit."""

    threshold: float
    """The real part of the integration result is compared against this."""
    feedback_signal_label: str = ""
    """In fast feedback routing, the transmitted signals are associated with this label.
    ConditionalInstructions whose "condition" field has the string value of `feedback_signal_label`
    will receive the signal from this ThresholdStateDiscrimination.
    Empty string (default) means the signal is not routed anywhere.
    The same `feedback_signal_label` may not be used multiple times within the same ReadoutTrigger.
    The same `feedback_signal_label` can be used in different ReadoutTriggers and different segments.
    """


@dataclass(frozen=True)
class ReadoutTrigger:
    """Instruction for playing a probe pulse and acquiring the associated readout results."""

    probe_pulse: Instruction
    """Index pointing to a probe pulse, usually a MultiplexedIQPulse."""

    acquisitions: tuple[AcquisitionMethod, ...]
    """Active readout acquisition methods associated with this trigger instance."""


Operation = TypeVar(
    "Operation",
    Wait,
    IQPulse,
    RealPulse,
    VirtualRZ,
    ConditionalInstruction,
    MultiplexedRealPulse,
    MultiplexedIQPulse,
    ReadoutTrigger,
)


@dataclass(frozen=True)
class Instruction(Generic[Operation]):
    """Wrapper class for Instructions."""

    duration_samples: int
    operation: Operation
