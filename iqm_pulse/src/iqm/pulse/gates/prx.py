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
r"""Single-qubit PRX gate.

The phased x rotation (PRX) gate is defined as

.. math::
   R_\phi(\theta) = e^{-i (X \cos \phi + Y \sin \phi) \: \theta/2}
    = R_z(\phi) R_x(\theta) R_z^\dagger(\phi),

where the rotation angle :math:`\theta` and the phase angle :math:`\phi` are in radians.

It rotates the qubit state around an axis that lies in the XY plane of the Bloch sphere.
"""

from __future__ import annotations

from abc import abstractmethod
import copy
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from exa.common.data.parameter import Parameter, Setting
from iqm.pulse.gate_implementation import (
    SINGLE_COMPONENTS_WITH_DRIVE_LOCUS_MAPPING,
    CustomIQWaveforms,
    GateImplementation,
    Locus,
    OILCalibrationData,
    SinglePulseGate,
    get_waveform_parameters,
)
from iqm.pulse.gates.enums import XYGate
from iqm.pulse.playlist.fast_drag import FastDragI, FastDragQ
from iqm.pulse.playlist.hd_drag import HdDragI, HdDragQ
from iqm.pulse.playlist.instructions import Block, IQPulse
from iqm.pulse.playlist.schedule import TOLERANCE, Schedule
from iqm.pulse.playlist.waveforms import (
    Constant,
    Cosine,
    CosineFall,
    CosineRise,
    CosineRiseFall,
    TruncatedGaussian,
    Waveform,
)
from iqm.pulse.playlist.waveforms import CosineRiseFallDerivative as CosineRiseFallD
from iqm.pulse.playlist.waveforms import TruncatedGaussianDerivative as TruncatedGaussianD
from iqm.pulse.utils import normalize_angle, phase_transformation

if TYPE_CHECKING:  # pragma: no cover
    from iqm.pulse.builder import ScheduleBuilder
    from iqm.pulse.quantum_ops import QuantumOp
    from iqm.pulse.timebox import TimeBox


@lru_cache
def get_unitary_prx(angle: float, phase: float) -> np.ndarray:
    """Unitary for a PRX gate.

    Args:
        angle: rotation angle (in rad)
        phase: phase angle (in rad)

    Returns:
        2x2 unitary representing ``prx(angle, phase)``.

    """
    return np.array(
        [
            [np.cos(angle / 2), -1j * np.exp(-1j * phase) * np.sin(angle / 2)],
            [-1j * np.exp(1j * phase) * np.sin(angle / 2), np.cos(angle / 2)],
        ]
    )


class PrxGateImplementation(GateImplementation):
    """ABC for different implementations of the PRX gate."""

    def __init__(
        self, parent: QuantumOp, name: str, locus: Locus, calibration_data: OILCalibrationData, builder: ScheduleBuilder
    ) -> None:
        super().__init__(parent, name, locus, calibration_data, builder)
        self._cliffords: dict[XYGate, TimeBox] = {}

    @abstractmethod
    def _call(self, angle: float, phase: float) -> TimeBox:
        """Phased X rotation gate.

        Args:
            angle: Rotation angle in radians.
            phase: Phase angle in radians.

        Returns:
            Boxed instruction schedule implementing the phased X rotation gate.

        """
        raise NotImplementedError

    def rx(self, angle: float) -> TimeBox:
        """X rotation gate.

        Args:
            angle: rotation angle (in radians)

        Returns:
            boxed instruction schedule implementing the x rotation gate

        """
        box = self(angle=angle, phase=0)
        box.label = f"Rx on {self.locus[0]}"  # type: ignore[union-attr]
        return box  # type: ignore[return-value]

    def ry(self, angle: float) -> TimeBox:
        """Y rotation gate.

        Args:
            angle: rotation angle (in radians)

        Returns:
            boxed instruction schedule implementing the y rotation gate

        """
        box = self(angle=angle, phase=np.pi / 2)
        box.label = f"Ry on {self.locus[0]}"  # type: ignore[union-attr]
        return box  # type: ignore[return-value]

    def clifford(self, xy_gate: XYGate) -> TimeBox:
        """One-qubit XY Clifford gates.

        Args:
            xy_gate: Clifford gate

        Returns:
            boxed instruction schedule implementing the requested Clifford gate

        """
        if gate := self._cliffords.get(xy_gate):
            return gate

        # cache Cliffords on first use
        match xy_gate:
            case XYGate.IDENTITY:
                # TODO could be also implemented with a zero-duration Wait, which would be more accurate.
                gate = self.rx(0.0)
            case XYGate.X_90:
                gate = self.rx(np.pi / 2)
            case XYGate.X_180:
                gate = self.rx(np.pi)
            case XYGate.X_M90:
                gate = self.rx(-np.pi / 2)
            case XYGate.Y_90:
                gate = self.ry(np.pi / 2)
            case XYGate.Y_180:
                gate = self.ry(np.pi)
            case XYGate.Y_M90:
                gate = self.ry(-np.pi / 2)
            case _:
                raise ValueError(f"Unknown XYGate: {xy_gate}")
        gate.label = f"{xy_gate.name} on {self.locus[0]}"
        self._cliffords[xy_gate] = gate
        return gate

    @classmethod
    def get_locus_mapping_name(cls, operation_name: str, implementation_name: str) -> str:
        return SINGLE_COMPONENTS_WITH_DRIVE_LOCUS_MAPPING


class PRX_SinglePulse_GateImplementation(SinglePulseGate, PrxGateImplementation):
    r"""Base class for PRX gates implemented using a single IQ pulse.

    This class implements phased x rotation gates on a specific qubit using an :class:`.IQPulse`
    instance, derived from the pulse calibration data provided at construction by
    the static method :meth:`_single_iq_pulse`.
    The pulse is used to resonantly drive the qubit to effect the required rotation.

    The pulse calibration data consists of the parameters of an :math:`x_\pi` :class:`.IQPulse` only.
    It is assumed that

    * the transfer function from the AWG to the qubit is linear, i.e.,
      other rotation angles can be obtained by linearly scaling the pulse amplitude, and
    * other phase angles can be obtained by adjusting the IQ modulation phase.

    The generated pulses all have the same time duration, also for identity rotations. In the special case of the
    duration being zero, the gate implementation will apply a ``Block(0)`` instruction to the qubit's drive channel.
    """

    def _call(self, angle: float, phase: float = 0.0) -> TimeBox:  # type: ignore[override]
        scale, new_phase = _normalize_params(angle, phase)
        pulse = self.pulse.copy(
            scale_i=scale * self.pulse.scale_i,  # type: ignore[attr-defined]
            scale_q=scale * self.pulse.scale_q,  # type: ignore[attr-defined]
            phase=self.pulse.phase + new_phase,  # type: ignore[attr-defined]
        )
        if self.pulse.duration > TOLERANCE:
            timebox = self.to_timebox(Schedule({self.channel: [pulse]}))
        else:
            timebox = self.to_timebox(Schedule({self.channel: [Block(0)]}))
        timebox.neighborhood_components[0] = set(self.locus)
        return timebox


class PRX_CustomWaveforms(PRX_SinglePulse_GateImplementation, CustomIQWaveforms):
    """Base class for PRX gates implemented using a single IQ pulse and hot-swappable waveforms."""

    root_parameters: dict[str, Parameter | Setting] = {
        "duration": Parameter("", "pi pulse duration", "s"),
        "amplitude_i": Parameter("", "pi pulse I channel amplitude", ""),
        "amplitude_q": Parameter("", "pi pulse Q channel amplitude", ""),
    }

    @classmethod
    def _get_pulse(  # type: ignore[override]
        cls,
        *,
        amplitude_i: float,
        amplitude_q: float,
        n_samples: int,
        **rest_of_calibration_data,
    ) -> IQPulse:
        """Builds an x_pi pulse out of the calibration data."""
        if cls.dependent_waves:
            wave_i = cls.wave_i(n_samples=n_samples, **rest_of_calibration_data)
            wave_q = cls.wave_q(n_samples=n_samples, **rest_of_calibration_data)
        else:
            wave_i = cls.wave_i(n_samples=n_samples, **rest_of_calibration_data["i"])
            wave_q = cls.wave_q(n_samples=n_samples, **rest_of_calibration_data["q"])
        return IQPulse(
            n_samples,
            wave_i=wave_i,
            wave_q=wave_q,
            scale_i=amplitude_i,
            scale_q=amplitude_q,
            phase=0,
            phase_increment=0,
        )


class PRX_DRAGGaussian(PRX_CustomWaveforms, wave_i=TruncatedGaussian, wave_q=TruncatedGaussianD):  # type:ignore[call-arg]
    """PRX gate, DRAG / TruncatedGaussian IQ pulse implementation.

    See :class:`.PRX_CustomWaveforms`.
    """


class PRX_DRAGCosineRiseFall(PRX_CustomWaveforms, wave_i=CosineRiseFall, wave_q=CosineRiseFallD):  # type:ignore[call-arg]
    """PRX gate, DRAG / CosineRiseFall IQ pulse implementation.

    See :class:`.PRX_CustomWaveforms`.
    """

    excluded_parameters = ["rise_time"]

    @classmethod
    def _get_pulse(cls, **kwargs) -> IQPulse:
        kwargs["rise_time"] = kwargs["full_width"] / 2  # just a cosine, no flat part
        return super()._get_pulse(**kwargs)


class PRX_CustomWaveformsSX(PRX_SinglePulse_GateImplementation, CustomIQWaveforms):
    r"""Base class for PRX gates implemented using SX gate, hot-swappable waveforms and phase manipulation.

    The schedule used to implement the PRX gate depends on the arguments:
        1. If the rotation angle :math:`\theta = \pi/2`, the timebox will consist of just the SX IQ pulse, with phase.
        2. If the rotation angle :math:`\theta = 0.0`, the timebox will consist of a single zero-amplitude pulse.
        3. If not, the timebox will consist of two IQ pulses, with phase.

    The formula for the PRX gate implemented using SX gates and z rotations is

    .. math::
       R_\phi(\theta) = R_z(\phi-\pi/2) \: \text{SX} \: R_z(\pi-\theta) \: \text{SX} \: R_z(-\phi-\pi/2).

    The fusing of z rotations to IQPulses is done inside the :meth:`_call` method.

    All parameters in the pulse here is referring to the state of the qubits.
    """

    root_parameters: dict[str, Parameter | Setting] = {
        "duration": Parameter("", "pi pulse duration", "s"),
        "amplitude_i": Parameter("", "pi pulse I channel amplitude", ""),
        "amplitude_q": Parameter("", "pi pulse Q channel amplitude", ""),
        "rz_before": Parameter("", "RZ before IQ pulse", "rad"),
        "rz_after": Parameter("", "RZ after IQ pulse", "rad"),
    }

    def _call(self, angle: float, phase: float = 0) -> TimeBox:  # type: ignore[override]
        """Convert pulses into timebox, via extra Z rotations.

        There are exceptions while using 0, pi/2 and pi rotation in angle, for calibration reason. The duration of the
        timebox can be different.
        """
        new_angle, new_phase = _normalize_params(angle, phase)
        new_angle = new_angle * np.pi
        if self.pulse.duration < TOLERANCE:
            timebox = self.to_timebox(Schedule({self.channel: [Block(0)]}))
        elif np.isclose(new_angle, 0.0, atol=1e-8):
            # Play zero-amplitude pulse similarly to PRX_CustomWaveforms. The phase increment must be set to zero
            # so that the pulse induces no Z-rotation
            pulse = self.pulse.copy(scale_i=0.0, scale_q=0.0, phase=new_phase, phase_increment=0.0)
            timebox = self.to_timebox(Schedule({self.channel: [pulse]}))
        elif np.isclose(new_angle, np.pi / 2, atol=1e-8):
            pulse = self.pulse.copy(
                phase=self.pulse.phase + new_phase,  # type: ignore[attr-defined]
            )
            timebox = self.to_timebox(Schedule({self.channel: [pulse]}))
        elif np.isclose(new_angle, np.pi, atol=1e-8):
            pulse = self.pulse.copy(
                phase=self.pulse.phase + new_phase,  # type: ignore[attr-defined]
            )
            timebox = self.to_timebox(Schedule({self.channel: [pulse, pulse]}))
        else:
            rz_a = -np.pi / 2 - new_phase
            rz_b = np.pi - new_angle
            rz_c = -np.pi / 2 + new_phase
            phase_1, phase_increment_1 = phase_transformation(rz_a, 0)
            phase_2, phase_increment_2 = phase_transformation(rz_b, rz_c)
            pulse_1 = self.pulse.copy(
                phase=normalize_angle(self.pulse.phase + phase_1),  # type: ignore[attr-defined]
                phase_increment=normalize_angle(self.pulse.phase_increment + phase_increment_1),  # type: ignore[attr-defined]
            )
            pulse_2 = self.pulse.copy(
                phase=normalize_angle(self.pulse.phase + phase_2),  # type: ignore[attr-defined]
                phase_increment=normalize_angle(self.pulse.phase_increment + phase_increment_2),  # type: ignore[attr-defined]
            )
            timebox = self.to_timebox(Schedule({self.channel: [pulse_1, pulse_2]}))

        timebox.neighborhood_components[0] = set(self.locus)
        return timebox

    @classmethod
    def _get_pulse(  # type: ignore[override]
        cls,
        *,
        amplitude_i: float,
        amplitude_q: float,
        n_samples: int,
        **rest_of_calibration_data,
    ) -> IQPulse:
        """Builds a single sqrt(X) pulse from the calibration data."""
        rz_before = rest_of_calibration_data.pop("rz_before", 0)
        rz_after = rest_of_calibration_data.pop("rz_after", 0)
        phase, phase_increment = phase_transformation(rz_before, rz_after)

        if cls.dependent_waves:
            wave_i = cls.wave_i(n_samples=n_samples, **rest_of_calibration_data)
            wave_q = cls.wave_q(n_samples=n_samples, **rest_of_calibration_data)
        else:
            wave_i = cls.wave_i(n_samples=n_samples, **rest_of_calibration_data["i"])
            wave_q = cls.wave_q(n_samples=n_samples, **rest_of_calibration_data["q"])
        return IQPulse(
            n_samples,
            wave_i=wave_i,
            wave_q=wave_q,
            scale_i=amplitude_i,
            scale_q=amplitude_q,
            phase=phase,
            phase_increment=phase_increment,
        )


class PRX_DRAGGaussianSX(PRX_CustomWaveformsSX, wave_i=TruncatedGaussian, wave_q=TruncatedGaussianD):  # type:ignore[call-arg]
    """PRX gate, DRAG / Gaussian IQ pulse with VZ implementation.

    See :class:`.PRX_CustomWaveformsVZ`.
    """


class PRX_DRAGCosineRiseFallSX(PRX_CustomWaveformsSX, wave_i=CosineRiseFall, wave_q=CosineRiseFallD):  # type:ignore[call-arg]
    """PRX gate, DRAG / CosineRiseFall IQ pulse with VZ implementation.

    See :class:`.PRX_CustomWaveformsVZ`.
    """

    excluded_parameters = ["rise_time"]

    @classmethod
    def _get_pulse(cls, **kwargs) -> IQPulse:
        kwargs["rise_time"] = kwargs["full_width"] / 2  # just a cosine, no flat part
        return super()._get_pulse(**kwargs)


class PRX_FastDragSX(PRX_CustomWaveformsSX, wave_i=FastDragI, wave_q=FastDragQ):  # type:ignore[call-arg]
    """PRX gate, FAST DRAG IQ pulse with VZ-based SX-implementation.

    See :class:`.PRX_CustomWaveformsSX`.
    """


class PRX_FastDrag(PRX_CustomWaveforms, wave_i=FastDragI, wave_q=FastDragQ):  # type:ignore[call-arg]
    """PRX gate, FAST DRAG IQ pulse based on amplitude scaling.

    See :class:`.PRX_CustomWaveforms`.
    """


class PRX_HdDragSX(PRX_CustomWaveformsSX, wave_i=HdDragI, wave_q=HdDragQ):  # type:ignore[call-arg]
    """PRX gate, HD DRAG IQ pulse with VZ-based SX-implementation.

    See :class:`.PRX_CustomWaveformsSX`.
    """


class PRX_HdDrag(PRX_CustomWaveforms, wave_i=HdDragI, wave_q=HdDragQ):  # type:ignore[call-arg]
    """PRX gate, HD DRAG IQ pulse based on amplitude scaling

    See :class:`.PRX_CustomWaveforms`.
    """


class PRX_Cosine(PRX_CustomWaveforms, wave_i=Cosine, wave_q=Cosine):  # type:ignore[call-arg]
    """Special modulated pulse resulting in two frequency sidebands.

    See :class:`.PRX_CustomWaveforms`.
    """


def _normalize_params(angle: float, phase: float) -> tuple[float, float]:
    """Algebraic normalization of ``angle`` to [0, pi] and ``phase`` to (-pi, pi].

    Returns:
        amplitude scaling factor, phase for a pi pulse

    """
    # 1. Normalize angle to (-pi, pi].
    # 2. Use the symmetry PRX(-theta, phi ± pi) = PRX(theta, phi) to move rotation angle in [0, pi]
    # 3. Normalize phase to (-pi, pi].
    # FIXME: ideally, here we want to do normalization only. Step 2 is mostly motivated by some rules set in ZI
    # compiler (see COMP-544 for details), and we shall get rid of it or move somewhere more ZI specific.
    half_turn = np.pi
    angle = normalize_angle(angle)
    if angle < 0:
        angle = -angle
        phase += half_turn
    return angle / half_turn, normalize_angle(phase)


class ABC_Constant_smooth(PrxGateImplementation):
    r"""Base class for creating gates with an arbitrarily long Constant pulses with smooth rise and fall.
    This pulse creates a :'Segment' consisting of three instructions : [rise_waveform, main_waveform, fall_waveform].
    This class is created so that one can use middle waveform as a constant, thus enabling to use arbitrarily
    long pulses, not limited by the awg memory.

    Attributes::
        main_waveform: The middle part of the pulse, which should (but doesn't have to) be a Constant waveform
        rise_waveform: rise part of the pulse
        fall_waveform: fall part of the pulse
        channel: Name of the drive channel on which the AC Stark pulse is played.

    """

    main_waveform: type[Waveform]
    rise_waveform: type[Waveform]
    fall_waveform: type[Waveform]

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

        params_for_stark = copy.deepcopy(params)
        params_for_risefall = copy.deepcopy(params)

        params_for_stark["n_samples"] = max(
            int(round(params_for_stark["n_samples"] * (1 - 2 * params_for_stark["rise_time"]), 0)), 0
        )
        self.main_waveform = self._main_pulse(**params_for_stark)  # type: ignore[assignment]

        params_for_risefall["n_samples"] = int(
            round(params_for_risefall["n_samples"] * params_for_risefall["rise_time"], 0)
        )

        self.fall_waveform = self._fall_pulse(**params_for_risefall)  # type: ignore[assignment]
        self.rise_waveform = self._rise_pulse(**params_for_risefall)  # type: ignore[assignment]

        self.channel = drive_channel
        self.special_implementation = True  # DEBUG

    def __init_subclass__(
        cls,
        /,
        fall_waveform: type[Waveform],
        rise_waveform: type[Waveform],
        main_waveform: type[Waveform],
    ):
        """Store the Waveform types used by this subclass, and their parameters."""
        cls.main_waveform = main_waveform
        cls.rise_waveform = rise_waveform
        cls.fall_waveform = fall_waveform

        cls.symmetric = True
        parameters = get_waveform_parameters(main_waveform)
        parameters["amplitude_i"] = Parameter("", "amplitude_i", "")
        parameters["amplitude_q"] = Parameter("", "amplitude_q", "")

        parameters["rise_time"] = Parameter("", "gate rise time", "s")

        cls.parameters = {  # type: ignore[assignment]
            "duration": Parameter("", "Gate duration", "s"),
        } | parameters

    def _call(self, angle: float, phase: float = 0.0) -> TimeBox:
        scale, new_phase = _normalize_params(angle, phase)

        pulse_rise = self.rise_waveform.copy(
            scale_i=scale * self.rise_waveform.scale_i,
            scale_q=scale * self.rise_waveform.scale_q,
            phase=self.rise_waveform.phase + new_phase,
        )
        pulse_fall = self.fall_waveform.copy(
            scale_i=scale * self.fall_waveform.scale_i,
            scale_q=scale * self.fall_waveform.scale_q,
            phase=self.fall_waveform.phase + new_phase,
        )
        stark_pulse = self.main_waveform.copy(
            scale_i=scale * self.main_waveform.scale_i,
            scale_q=scale * self.main_waveform.scale_q,
            phase=self.main_waveform.phase + new_phase,
        )

        return self.to_timebox(
            Schedule(
                {self.channel: [pulse_rise, stark_pulse, pulse_fall]},
            )
        )

    @classmethod
    def _main_pulse(
        cls,
        *,
        n_samples: int,
        amplitude_i: float,
        amplitude_q: float,
        phase: float = 0.0,
        **kwargs,
    ) -> IQPulse:
        """Returns the main part pulse. Waveform is the same for both I and Q channels"""
        wave = cls.main_waveform(n_samples=n_samples)

        return IQPulse(
            n_samples,
            wave_i=wave,
            wave_q=wave,
            scale_i=amplitude_i,
            scale_q=amplitude_q,
        )

    @classmethod
    def _rise_pulse(
        cls,
        *,
        n_samples: int,
        amplitude_i: float,
        amplitude_q: float,
        **kwargs,
    ) -> IQPulse:
        """Returns a rise pulse."""
        wave_i = cls.rise_waveform(n_samples=n_samples)
        wave_q = cls.rise_waveform(n_samples=n_samples)

        return IQPulse(
            n_samples,
            wave_i=wave_i,
            wave_q=wave_q,
            scale_i=amplitude_i,
            scale_q=amplitude_q,
        )

    @classmethod
    def _fall_pulse(
        cls,
        *,
        n_samples: int,
        amplitude_i: float,
        amplitude_q: float,
        **kwargs,
    ) -> IQPulse:
        """Returns a fall pulse"""
        return IQPulse(
            n_samples,
            wave_i=cls.fall_waveform(n_samples=n_samples),
            wave_q=cls.fall_waveform(n_samples=n_samples),
            scale_i=amplitude_i,
            scale_q=amplitude_q,
        )


class Constant_PRX_with_smooth_rise_fall(
    ABC_Constant_smooth,
    rise_waveform=CosineRise,
    main_waveform=Constant,
    fall_waveform=CosineFall,
):
    """Constant PRX pulse with cosine rise and fall padding.
    Implemented as a 3-instruction Schedule.
    """


class PRX_ModulatedCustomWaveForms(PRX_CustomWaveforms):
    r"""Base class for PRX gates with modulated frequency, hot-swappable waveforms.

    The class takes baseband I and Q waveform as input, and modulates them with frequency in the root_parameters.
    The final pulse shape after modulation is:

    .. math::
        A_I^{\delta}\Omega_I(t)\cos((\omega_d + \delta)t) - A_Q^{\delta}\Omega_Q(t)\sin((\omega_d + \delta)t)

    where :math:`A_I` is `amplitude_i`, :math:`A_Q` is `amplitude_q`, :math:`\Omega` is arbitrary waveform in
    baseband, :math:`\omega_d/2\pi` is the drive frequency and :math:`\delta/2\pi` is the modulated `frequency`.

    """

    root_parameters: dict[str, Parameter | Setting] = {
        "duration": Parameter("", "pi pulse duration", "s"),
        "amplitude_i": Parameter("", "pi pulse amplitude of base band I waveform", ""),
        "amplitude_q": Parameter("", "pi pulse amplitude of base band Q waveform", ""),
        "frequency": Parameter("", "modulated pulse frequency", "Hz"),
    }

    @classmethod
    def _get_pulse(  # type: ignore[override]
        cls, *, amplitude_i: float, amplitude_q: float, n_samples: int, **rest_of_calibration_data
    ) -> IQPulse:
        """Return the IQPulse with modulated arbitrary waveform based on the calibration data."""
        frequency = rest_of_calibration_data.pop("frequency")
        if cls.dependent_waves:
            wave_i = cls.wave_i(n_samples=n_samples, **rest_of_calibration_data)
            wave_q = cls.wave_q(n_samples=n_samples, **rest_of_calibration_data)
        else:
            wave_i = cls.wave_i(n_samples=n_samples, **rest_of_calibration_data["i"])
            wave_q = cls.wave_q(n_samples=n_samples, **rest_of_calibration_data["q"])
        return IQPulse(
            n_samples,
            wave_i=wave_i,
            wave_q=wave_q,
            scale_i=amplitude_i,
            scale_q=amplitude_q,
            modulation_frequency=frequency / n_samples,
        )


class PRX_ModulatedDRAGCosineRiseFall(PRX_ModulatedCustomWaveForms, wave_i=CosineRiseFall, wave_q=CosineRiseFallD):  # type:ignore[call-arg]
    """Modulated PRX pulse with cosine rise fall waveform"""

    excluded_parameters = ["rise_time"]

    @classmethod
    def _get_pulse(cls, **kwargs) -> IQPulse:
        kwargs["rise_time"] = kwargs["full_width"] / 2  # just a cosine, no flat part
        return super()._get_pulse(**kwargs)
