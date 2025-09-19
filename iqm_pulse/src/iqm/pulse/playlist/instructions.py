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
"""Instructions for control instruments."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
import random

from iqm.pulse.playlist.waveforms import Waveform


@dataclass(frozen=True)
class Instruction:
    """Command that can be executed by the quantum computer on a control channel.

    Has a well-specified time duration.
    """

    duration: int
    """Time duration of the instruction. In samples at the channel sample rate."""

    def __post_init__(self):
        """Add a unique id for caching purposes, which can be used if the instruction is not hashable."""
        object.__setattr__(self, "id", random.getrandbits(64))

    def validate(self) -> None:
        """Validate the instruction attributes.

        Raises:
            ValueError: something about the instruction is not ok

        """
        if self.duration < 0:
            raise ValueError(f"Instruction.duration {self.duration} is negative.")

    def copy(self, **changes) -> Instruction:
        """Make a copy of the Instruction with the given changes applied to its contents."""
        return dataclasses.replace(self, **changes)

    def get_child_instructions(self) -> tuple[Instruction, ...]:
        """Returns all the child Instructions the Instruction contains."""
        return ()

    def get_waveforms(self) -> tuple[Waveform, ...]:
        """Returns all the waveforms the Instruction contains."""
        return ()


@dataclass(frozen=True)
class Wait(Instruction):
    """Behave as if outputting zero-valued samples for the duration of the instruction.

    Used to idle QPU components. However, may be replaced with a dynamical decoupling sequence
    after the scheduling step. If you want to make sure that this does not happen, use
    :class:`Block` instead.
    """


@dataclass(frozen=True)
class Block(Instruction):
    """Behave strictly as if outputting zero-valued samples for the duration of the instruction.

    Used to block a control channel during compilation.
    A more strict version of :class:`Wait`, cannot be replaced with DD sequences during compilation.
    Converted to a :class:`.Wait` instruction at the end of compilation process.

    In "Tetris" scheduling, several Block instructions can overlap in time, whereas Waits cannot.
    """


@dataclass(frozen=True)
class VirtualRZ(Instruction):
    """Change the upconversion phase reference.

    The phase change can be done either by updating the phase of the local oscillator directly using
    a hardware instruction, or algebraically by incrementing :attr:`IQPulse.phase` of all the IQPulses
    following the VirtualRZ instruction in the :class:`.Segment`.
    """

    phase_increment: float
    """Phase increment for the local oscillator of a drive channel, in radians."""


@dataclass(frozen=True)
class RealPulse(Instruction):
    """Play a real-valued pulse."""

    wave: Waveform
    """Shape of the pulse."""
    scale: float
    """Scaling factor for the waveform."""

    def validate(self):  # noqa: ANN201
        super().validate()
        if abs(self.scale) > 1.0:
            raise ValueError(f"RealPulse.scale {self.scale} not in [-1, 1].")

    def get_waveforms(self) -> tuple[Waveform, ...]:
        return (self.wave,)


@dataclass(frozen=True)
class FluxPulse(RealPulse):
    """RealPulse representing a flux pulse.

    Can store RZ angles for correcting local phase shifts from the computational frame due to flux crosstalk.
    """

    rzs: tuple[tuple[str, float], ...] = field(default_factory=tuple)
    """Collection of (drive) channel names and RZ angles."""


@dataclass(frozen=True)
class IQPulse(Instruction):
    """Play an upconverted pulse that contains real in-phase and quadrature waveforms."""

    wave_i: Waveform
    """I quadrature envelope."""
    wave_q: Waveform
    """Q quadrature envelope."""
    scale_i: float = 1.0
    """Scaling factor for the I quadrature."""
    scale_q: float = 0.0
    """Scaling factor for the Q quadrature."""
    phase: float = 0.0
    """Phase of the pulse relative to the channel upconversion oscillator ("carrier wave"), in radians."""
    modulation_frequency: float = 0.0
    """Modulation frequency of the waveforms, in units of the sampling rate.
    This modulation is additional to the channel upconversion frequency.
    The default value of 0.0 does not modulate.
    Note that the phase of this modulation resets for every instruction, that is, successive instances of the same
    modulated pulse are not phase coherent.
    """
    phase_increment: float = 0.0
    """Phase increment for the channel upconversion oscillator ("carrier wave"), affecting this pulse and
    all pulses that are played after it on the channel, in radians.
    """

    def validate(self):  # noqa: ANN201
        super().validate()
        if abs(self.scale_i) > 1.0:
            raise ValueError(f"IQPulse.scale_i {self.scale_i} not in [-1, 1].")
        if abs(self.scale_q) > 1.0:
            raise ValueError(f"IQPulse.scale_q {self.scale_q} not in [-1, 1].")

    def get_waveforms(self) -> tuple[Waveform, ...]:
        return (self.wave_i, self.wave_q)


@dataclass(frozen=True)
class ConditionalInstruction(Instruction):
    """Choice between multiple Instructions, depending on a condition."""

    condition: str
    """Can be evaluated to an integer >= 0 representing an outcome."""
    outcomes: tuple[Instruction, ...]
    """Maps possible outcomes of the condition to the corresponding instructions."""

    def validate(self):  # noqa: ANN201
        super().validate()
        if not self.outcomes:
            raise ValueError("There must be at least one outcome.")
        durations = {instruction.duration for instruction in self.outcomes}
        durations.add(self.duration)
        if len(durations) != 1:
            raise ValueError(
                f"All the conditional instructions must have the same duration (now: {durations} samples)."
            )

    def get_child_instructions(self) -> tuple[Instruction, ...]:
        return self.outcomes


@dataclass(frozen=True)
class MultiplexedIQPulse(Instruction):
    """Play the sum of multiple IQ pulses.

    Each component pulse can have an arbitrary delay from the beginning of this instruction.
    Outside the interval of the MultiplexedIQPulse, the component pulses are truncated.
    Where overlapping, samples of component pulse entries are summed.
    Where the interval of a MultiplexedIQPulse does not overlap with any of its component pulses,
    its samples are zeroes.
    """

    entries: tuple[tuple[IQPulse, int], ...]
    """(``pulse``, ``offset``) pairs.
    ``offset`` is the number of samples ``pulse`` is delayed from the beginning of the instruction.
    It has no granularity constraints. Negative values are allowed, but beginning will be truncated.
    """


@dataclass(frozen=True)
class AcquisitionMethod:
    """Describes a way to acquire readout data."""

    label: str
    """Identifier for the returned data, like ``QB1__readout.time_trace``."""
    delay_samples: int
    """Delay from beginning of probe pulse to beginning of acquisition window, in samples."""
    implementation: str | None
    """Measure operation and implementation that created this AcquisitionMethod, in the format
    ``<operation name>.<implementation name>``. If the acquisition is not originated from a
    gate implementation, this should be ``None``."""


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
    """Perform a weighted integration of the IQ raw signal and compare the real part of the result
    against a threshold value, resulting in a boolean.
    """

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
class ReadoutTrigger(Instruction):
    """Instruction for playing a probe pulse and acquiring the associated readout results."""

    probe_pulse: MultiplexedIQPulse
    """Probe pulse to play, usually a MultiplexedIQPulse."""

    acquisitions: tuple[AcquisitionMethod, ...]
    """Active readout acquisition methods associated with this trigger instance."""

    def __add__(self, other: ReadoutTrigger) -> ReadoutTrigger:
        new_duration = max(self.duration, other.duration)
        new_probe_duration = max(self.probe_pulse.duration, other.probe_pulse.duration)
        return ReadoutTrigger(
            duration=new_duration,
            probe_pulse=MultiplexedIQPulse(
                duration=new_probe_duration, entries=self.probe_pulse.entries + other.probe_pulse.entries
            ),
            acquisitions=self.acquisitions + other.acquisitions,
        )
