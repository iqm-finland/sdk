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
r"""Single-qubit RZ gate.

The z rotation gate is defined as

.. math::
   R_z(\phi) = e^{-i Z \phi / 2}

where the rotation angle :math:`\phi` is in radians.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from iqm.models.playlist.waveforms import Waveform
import numpy as np

from exa.common.data.parameter import Parameter
from iqm.pulse.gate_implementation import (
    SINGLE_COMPONENTS_WITH_DRIVE_LOCUS_MAPPING,
    CompositeGate,
    GateImplementation,
    Locus,
    OILCalibrationData,
    get_waveform_parameters,
)
from iqm.pulse.gates.prx import Constant_PRX_with_smooth_rise_fall
from iqm.pulse.playlist.instructions import IQPulse, VirtualRZ
from iqm.pulse.playlist.schedule import Schedule
from iqm.pulse.playlist.waveforms import Constant, CosineFall, CosineRise, ModulatedCosineRiseFall
from iqm.pulse.utils import normalize_angle, phase_transformation

if TYPE_CHECKING:  # pragma: no cover
    from iqm.pulse.builder import ScheduleBuilder
    from iqm.pulse.quantum_ops import QuantumOp
from iqm.pulse.timebox import TimeBox


@lru_cache
def get_unitary_rz(angle: float) -> np.ndarray:
    """Unitary for an RZ gate.

    Args:
        angle: rotation angle (in rad)

    Returns:
        2x2 unitary representing ``rz(angle)``.

    """
    return np.array(
        [
            [np.exp(-1j * angle / 2), 0],
            [0, np.exp(1j * angle / 2)],
        ]
    )


class RZ_Virtual(GateImplementation):
    r"""Implementation of the RZ gate using the virtual z rotation technique.

    Implements the RZ gate on a specific qubit using a :class:`.VirtualRZ` instruction, which
    simply changes the phase of the local oscillator driving that qubit.
    This requires no calibration data as of now.
    The generated VirtualRZ instruction has the shortest possible duration allowed by the instruments.

    The virtual z rotation method is based on algebraically commuting the RZ gates towards the end
    of the circuit, until they hit a measurement operation, at which point they are eliminated. It assumes that

    1. all the multi-qubit gates in the circuit commute with arbitrary RZ gates (this holds e.g. for CZ
       since it is diagonal),
    2. measurements are projective and happen in the z basis, so that RZ gates that immediately
       precede them do not affect the measurement result or the state after the measurement, and thus
       can be removed, and
    3. conjugating the single-qubit gates in the circuit with RZ is equivalent to incrementing the phase of the drive
       (holds for :mod:`PRX <.prx>`),

       .. math::
          R_\phi(\theta) R_z(\alpha) = R_z(\alpha) R_{\phi - \alpha}(\theta),

       which can be accomplished either by incrementing the phase of
       the local oscillator of the drive channel, or incrementing the phases of all the :class:`.IQPulse` s
       following it on the drive channel.

    *If all these assumptions hold* we may implement an RZ gate using a VirtualRZ instruction,
    with :attr:`phase_increment` equal to the negated rotation angle.

    Args:
        channel: name of the drive channel on which the VirtualRZ acts
        duration: time duration of the VirtualRZ instruction, in seconds

    """

    def __init__(
        self,
        parent: QuantumOp,
        name: str,
        locus: Locus,
        calibration_data: OILCalibrationData,
        builder: ScheduleBuilder,
    ):
        super().__init__(parent, name, locus, calibration_data, builder)
        drive_channel_name = builder.get_drive_channel(*locus)
        drive_channel = builder.channels[drive_channel_name]
        duration = drive_channel.duration_to_seconds(drive_channel.instruction_duration_min)
        self.duration = duration
        self.channel = drive_channel_name

    def _call(self, angle: float) -> TimeBox:  # type: ignore[override]
        """Z rotation gate.

        Args:
            angle: rotation angle (in radians)

        Returns:
            pulse schedule implementing the z rotation gate

        """
        timebox = self.to_timebox(
            Schedule(
                {
                    self.channel: [
                        VirtualRZ(
                            duration=self.builder.channels[self.channel].duration_to_int_samples(self.duration),
                            phase_increment=-normalize_angle(angle),
                        )
                    ]
                }
            )
        )
        timebox.neighborhood_components[0] = set(self.locus)
        return timebox

    parameters: dict[str, Parameter] = {}  # type: ignore[assignment]

    def duration_in_seconds(self) -> float:
        return self.duration

    @classmethod
    def get_locus_mapping_name(cls, operation_name: str, implementation_name: str) -> str:
        return SINGLE_COMPONENTS_WITH_DRIVE_LOCUS_MAPPING


class RZ_ACStarkShift(GateImplementation):
    r"""Implementation of the RZ gate using an AC Stark pulse.

    An AC Stark pulse is a strong off-resonant drive on a qubit. This pulse leads to a frequency shift of the qubit due
    to the AC Stark effect. The qubit frequency shift depends on the AC Stark pulse amplitude and frequency.

    Args:
        ac_stark_pulse: AC Stark pulse.
        channel: Name of the drive channel on which the AC Stark pulse is played.

    """

    ac_stark_waveform: type[Waveform]

    def __init__(
        self,
        parent: QuantumOp,
        name: str,
        locus: Locus,
        calibration_data: OILCalibrationData,
        builder: ScheduleBuilder,
    ):
        """Constructs an instance of the AC Stark pulse for the given locus."""
        super().__init__(parent, name, locus, calibration_data, builder)
        drive_channel = builder.get_drive_channel(locus[0])
        params = self.convert_calibration_data(calibration_data, self.parameters, builder.channels[drive_channel])
        self.ac_stark_pulse = self._ac_stark_pulse(**params)
        self.channel = drive_channel

    def __init_subclass__(cls, /, ac_stark_waveform: type[Waveform]):
        """Store the Waveform types used by this subclass, and their parameters."""
        cls.ac_stark_waveform = ac_stark_waveform
        cls.symmetric = ac_stark_waveform is None
        parameters = get_waveform_parameters(ac_stark_waveform)
        parameters["amplitude"] = Parameter("", "amplitude", "")
        parameters["phase_increment"] = Parameter("", "phase increment", "rad")

        cls.parameters = {  # type: ignore[assignment]
            "duration": Parameter("", "Gate duration", "s"),
        } | parameters

    def _call(self) -> TimeBox:
        return self.to_timebox(Schedule({self.channel: [self.ac_stark_pulse]}))

    def duration_in_seconds(self) -> float:
        if self.ac_stark_pulse.duration == 0:
            return 0.0
        return self.builder.channels[self.channel].duration_to_seconds(self.ac_stark_pulse.duration)

    @classmethod
    def _ac_stark_pulse(
        cls,
        *,
        n_samples: int,
        amplitude: float,
        phase_increment: float,
        phase: float,
        **kwargs,
    ) -> IQPulse:
        """Returns an AC Stark pulse which consists of a modulated I and modulated Q waveform, where the Q quadrature
        has an additional phase of -pi/2.
        """
        _, phase_increment = phase_transformation(0, phase_increment)

        wave_i = cls.ac_stark_waveform(n_samples=n_samples, phase=phase, **kwargs)
        wave_q = cls.ac_stark_waveform(n_samples=n_samples, phase=phase - np.pi / 2, **kwargs)
        return IQPulse(
            n_samples,
            wave_i=wave_i,
            wave_q=wave_q,
            scale_i=amplitude,
            scale_q=amplitude,
            phase_increment=phase_increment,
        )


class RZ_ACStarkShift_CosineRiseFall(RZ_ACStarkShift, ac_stark_waveform=ModulatedCosineRiseFall):
    """AC stark pulse implemented as a modulated cosine rise fall pulse."""


class RZ_ACStarkShift_smoothConstant(  # type: ignore[call-arg]  # type: ignore[call-arg]  # type: ignore[call-arg]
    Constant_PRX_with_smooth_rise_fall,
    rise_waveform=CosineRise,  # type:ignore[call-arg]
    main_waveform=Constant,  # type:ignore[call-arg]
    fall_waveform=CosineFall,  # type:ignore[call-arg]
):
    """Constant AC stark pulse with cosine rise and fall padding.
    Implemented as a 3-instruction Schedule.
    """

    def __call__(self):
        return super().__call__(angle=np.pi)


class RZ_PRX_Composite(CompositeGate):
    """RZ gate implemented as a sequence of PRX gates."""

    registered_gates = ("prx",)

    def __init__(self, parent, name, locus, calibration_data, builder):  # noqa: ANN001
        super().__init__(parent, name, locus, calibration_data, builder)

    def __call__(self, angle: float) -> TimeBox:
        prx = self.build("prx", self.locus)
        return TimeBox.composite(
            [
                prx.ry(np.pi / 2),  # type: ignore[attr-defined]
                prx.rx(angle),  # type: ignore[attr-defined]
                prx.ry(-np.pi / 2),  # type: ignore[attr-defined]
            ]
        )
