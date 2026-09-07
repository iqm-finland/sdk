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
r"""Two-qubit controlled-Z (CZ) gate.

The CZ gate flips the relative phase of the :math:`|11⟩` state.
It can be represented by the unitary matrix

.. math:: \text{CZ} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & -1 \end{pmatrix}
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import replace
import logging
from typing import TYPE_CHECKING

import numpy as np

from exa.common.data.parameter import Parameter, Setting
from iqm.pulse.gate_implementation import (
    CompositeGate,
    GateImplementation,
    Locus,
    OILCalibrationData,
    get_waveform_parameters,
    init_subclass_composite,
)
from iqm.pulse.playlist.instructions import Block, FluxPulse, Instruction, IQPulse, VirtualRZ
from iqm.pulse.playlist.schedule import Schedule
from iqm.pulse.playlist.waveforms import (
    Constant,
    CosineFallFlex,
    CosineRiseFall,
    CosineRiseFlex,
    GaussianSmoothedSquare,
    ModulatedCosineRiseFall,
    Slepian,
    TruncatedGaussianSmoothedSquare,
    Waveform,
)
from iqm.pulse.utils import phase_transformation

if TYPE_CHECKING:  # pragma: no cover
    from iqm.pulse.builder import ScheduleBuilder
    from iqm.pulse.quantum_ops import QuantumOp
    from iqm.pulse.timebox import TimeBox


from iqm.pulse.locus_mappings import (
    two_component_coupler_flux,
    two_component_one_flux_coupler_flux,
    two_qubit_drive_coupler_flux,
    two_qubit_flux_coupler_flux,
)


class FluxPulseGateBase(GateImplementation):
    """Discrete arity-2 gate implemented using flux pulses, virtual RZs, and the interaction mediated by the coupler.

    Does not have any gate args since it is discrete.

    The two locus components of the gate must be connected via a tunable coupler.

    Consists of a flux pulse for the coupler with one waveform, optional flux pulses for the locus
    components with another waveform, and virtual RZs on the locus drive channels.

    Inherit from this class and assign waveforms to the ``coupler_wave`` and ``qubit_wave``
    pulse slots to create a specific implementation.

    Can be used as a base class for both CZ and MOVE gate implementations.

    .. note::

       The coupler and qubit pulses have the same duration (given in the calibration data), and in the
       special case of the duration being zero, the gate implementation will apply ``Block(0)`` instructions
       to all the channels where it would otherwise apply flux pulses or virtual z rotations.
    """

    coupler_wave: type[Waveform] | None
    """Flux pulse waveform for the coupler flux AWG."""
    qubit_wave: type[Waveform] | None
    """Flux pulse waveform for the locus component flux AWGs."""
    root_parameters: dict[str, Parameter | Setting | dict] = {
        "duration": Parameter("", "Gate duration", "s"),
        "rz": {
            "*": Parameter("", "Z rotation angle", "rad"),  # wildcard parameter
        },
    }
    """Parameters shared by all ``FluxPulseGateBase`` classes. Inheriting classes may override this if there's
    a need for additional calibration parameters."""
    excluded_parameters: list[str] = []
    """Parameters names to be excluded from ``self.parameters``. Inheriting classes may override this if certain
    parameters are not wanted in that class (also parameters defined by the waveforms can be excluded)."""

    def __init__(
        self,
        parent: QuantumOp,
        name: str,
        locus: Locus,
        calibration_data: OILCalibrationData,
        builder: ScheduleBuilder,
    ):
        super().__init__(parent, name, locus, calibration_data, builder)
        self._duration = calibration_data["duration"]
        """gate duration in seconds, shared between all channels"""
        flux_pulses: dict[str, FluxPulse] = {}

        coupler = builder.chip_topology.get_coupler_for(*locus)
        if self.coupler_wave is not None:
            flux_pulses |= self._build_flux_pulse(self.coupler_wave, coupler, "coupler", self._duration)

        flux_pulses |= self._build_qubit_flux_pulses(locus, self._duration)

        rz = calibration_data["rz"]
        # rz angles are required for the two locus components, and are optional for every other drivable QPU component
        # NOTE computational resonators cannot be driven directly, instead they have "virtual" drive channels
        # that are removed by the compiler at the end of the compilation, and implemented by other means.
        for c in locus:
            if c not in rz:
                raise ValueError(
                    f"{self.qualified_name}: {locus}: Calibration is missing an RZ angle for locus component {c}."
                )
        rz_locus = {builder.get_drive_channel(c): angle for c, angle in rz.items() if c in locus}
        rz_not_locus = tuple((builder.get_drive_channel(c), angle) for c, angle in rz.items() if c not in locus)
        # No driving must happen on any of the affected components during the flux pulses,
        # hence the virtual z rotations must use up their entire duration.
        duration_samples = {pulse.duration for pulse in flux_pulses.values()}
        if len(duration_samples) != 1:
            raise ValueError(
                f"Flux channels have different sample rates: pulse durations are {duration_samples} samples"
            )
        T = next(iter(duration_samples))
        # The gate takes no parameters, so we may build and cache the entire Schedule here.
        schedule: dict[str, list[Instruction]] = {}
        for channel, angle in rz_locus.items():
            # the virtual rz technique requires decrementing the drive phase by the rz angle
            schedule[channel] = [VirtualRZ(duration=T, phase_increment=-angle)]
        vzs_inserted = False  # insert the long-distance Vzs to the first flux pulse (whatever that is)
        for channel, flux_pulse in flux_pulses.items():
            if rz_not_locus and not vzs_inserted:
                schedule[channel] = [replace(flux_pulse, rzs=rz_not_locus)]
                vzs_inserted = True
            else:
                schedule[channel] = [flux_pulse]
        self._affected_components = set(locus) | {coupler}
        self._schedule = Schedule(schedule if T > 0 else {c: [Block(0)] for c in schedule}, duration=T)

    def __init_subclass__(
        cls,
        /,
        coupler_wave: type[Waveform] | None = None,
        qubit_wave: type[Waveform] | None = None,
    ) -> None:
        """Set the Waveform types used by this subclass.

        Create a :class:`.FluxPulseGateBase` subclass with specific waveforms using
        ``class MySubClass(FluxPulseGateBase, coupler_wave=A, qubit_wave=B)``.

        Further inheriting from that class like
        ``class MySubSubClass(MySubClass, coupler_wave=C, qubit_wave=D)``
        changes the waveforms accordingly. If you do not provide any waveforms,
        ``class MySubSubClass(MySubClass)``,
        the waveforms defined in ``MySubClass`` will be retained.
        If you provide just some of the waveforms,
        ``class MySubSubClass(MySubClass, coupler_wave=X)``,
        the other waveforms will be set to ``None``.

        Args:
            coupler_wave: Flux pulse waveform to be played by the coupler flux AWG. Can be set to ``None`` if
                no coupler flux pulse should be played.
            qubit_wave: Flux pulse waveform to be played by the locus component flux AWGs.
                Can be set to ``None`` if no flux pulses should be played on the locus components.
                Typically the locus components are qubits.

        """
        # fix __init_subclass__ behaviour for further inheritance from a subclass of FluxPulseGateBase
        # we can skip this function if the class attributes are already stored in the parent class
        # and the subsubclass definition does not change these
        # see more info in: https://stackoverflow.com/questions/55183288/inheriting-init-subclass-parameters
        # the unintuitive default ``None`` values and handling of these values is for overcoming this issue
        # so that the method itself behaves as expected in successive subclassing
        if coupler_wave is None and qubit_wave is None and hasattr(cls, "coupler_wave") and hasattr(cls, "qubit_wave"):
            return
        cls.coupler_wave = coupler_wave
        cls.qubit_wave = qubit_wave
        cls.locus_mapping_function = (
            two_component_coupler_flux if qubit_wave is None else two_component_one_flux_coupler_flux
        )

        parameters: dict = cls.root_parameters.copy()
        if coupler_wave is not None:
            parameters["coupler"] = get_waveform_parameters(coupler_wave, label_prefix="Coupler flux pulse ")
            parameters["coupler"]["amplitude"] = Parameter("", "Coupler flux pulse amplitude", "")
        parameters |= cls._get_qubit_pulse_parameters(qubit_wave)

        cls.parameters = {k: v for k, v in parameters.items() if k not in cls.excluded_parameters}
        if issubclass(cls, CompositeGate):
            init_subclass_composite(cls)

    def _build_flux_pulse(
        self, waveform_class: type[Waveform], component_name: str, cal_node_name: str, duration: float
    ) -> dict[str, FluxPulse]:
        """Uses a part of the gate calibration data to prepare a flux pulse for the given component."""
        flux_channel = self.builder.get_flux_channel(component_name)
        params = self.convert_calibration_data(
            self.calibration_data[cal_node_name],
            self.parameters[cal_node_name],  # type: ignore[arg-type]
            self.builder.channels[flux_channel],
            duration=duration,
        )
        amplitude = params.pop("amplitude")
        return {
            flux_channel: FluxPulse(
                duration=params["n_samples"],
                wave=waveform_class(**params),
                scale=amplitude,
            )
        }

    @abstractmethod
    def _build_qubit_flux_pulses(self, locus: Locus, duration: float) -> dict[str, FluxPulse]:
        """Uses a part of the gate calibration data to prepare flux pulses for the locus components.

        Called during ``__init__``.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _get_qubit_pulse_parameters(
        cls, qubit_wave: type[Waveform] | None
    ) -> dict[str, dict[str, Parameter | Setting]]:
        """Return the parameters for qubit flux pulses for the ``qubit_wave`` waveform.

        Called during ``__init_subclass__``.
        """
        raise NotImplementedError

    def _call(self) -> TimeBox:
        timebox = self.to_timebox(self._schedule)
        timebox.neighborhood_components[0] = self._affected_components
        return timebox

    def __call__(self, *args, **kwargs) -> TimeBox:  # For type narrowing
        return super().__call__(*args, **kwargs)  # type: ignore[return-value]

    def duration_in_seconds(self) -> float:
        return self._duration


class FluxPulseGate(FluxPulseGateBase):
    """Base class for flux pulse gate implementations that have a flux pulse for at most one locus component."""

    def _build_qubit_flux_pulses(self, locus: Locus, duration: float) -> dict[str, FluxPulse]:
        if self.qubit_wave is not None:
            # the pulsed qubit is always the first one of the locus
            return self._build_flux_pulse(self.qubit_wave, locus[0], "qubit", duration)
        return {}

    @classmethod
    def _get_qubit_pulse_parameters(
        cls, qubit_wave: type[Waveform] | None
    ) -> dict[str, dict[str, Parameter | Setting]]:
        if qubit_wave is None:
            cls.symmetric = True
            return {}

        cls.symmetric = False
        parameters = {
            "qubit": get_waveform_parameters(qubit_wave, label_prefix="Qubit flux pulse "),
        }
        parameters["qubit"]["amplitude"] = Parameter("", "Qubit flux pulse amplitude", "")
        return parameters


class CZ_GaussianSmoothedSquare(FluxPulseGate, coupler_wave=GaussianSmoothedSquare):
    """CZ gate using a GaussianSmoothedSquare flux pulse on the coupler."""


class CZ_Slepian(FluxPulseGate, coupler_wave=Slepian):
    """CZ gate using a Slepian flux pulse on the coupler."""


class CZ_TruncatedGaussianSmoothedSquare(FluxPulseGate, coupler_wave=TruncatedGaussianSmoothedSquare):
    """CZ gate using a TruncatedGaussianSmoothedSquare flux pulse on the coupler."""


class CZ_Slepian_CRF(FluxPulseGate, coupler_wave=Slepian, qubit_wave=CosineRiseFall):
    """CZ gate using a Slepian flux pulse on the coupler and a CosineRiseFall flux pulse on the qubit."""


class CZ_CRF(FluxPulseGate, coupler_wave=CosineRiseFall):
    """CZ gate using a CosineRiseFall flux pulse on the coupler."""


class FluxPulseGate_TGSS_CRF(FluxPulseGate, coupler_wave=TruncatedGaussianSmoothedSquare, qubit_wave=CosineRiseFall):
    """CZ gate using a TGSS flux pulse on the coupler and a CosineRiseFall flux pulse on the qubit."""


class FluxPulseGate_CRF_CRF(FluxPulseGate, coupler_wave=CosineRiseFall, qubit_wave=CosineRiseFall):
    """CZ gate using a CosineRiseFall flux pulse on the coupler and on the qubit."""


class CouplerFluxPulseQubitACStarkPulseGate(GateImplementation):
    """Base class for CZ gates with coupler flux pulse and a qubit AC Stark pulse.

    Analogous to the fast qubit flux pulse, the AC Stark pulse can tune the frequency of the qubit. Together with the
    coupler flux pulse, this can implement a fast qubit pulsed CZ gate.

    """

    coupler_wave: type[Waveform] | None
    """Flux pulse Waveform to be played in the coupler flux AWG."""
    qubit_drive_wave: type[Waveform] | None
    """Qubit drive pulse waveform to be played in the qubit drive AWG."""

    root_parameters: dict[str, Parameter | Setting | dict] = {
        "duration": Parameter("", "Gate duration", "s"),
        "rz": {
            "*": Parameter("", "Z rotation angle", "rad"),
        },
    }
    excluded_parameters: list[str] = []
    """Parameters names to be excluded from ``self.parameters``. Inheriting classes may override this if certain
    parameters are not wanted in that class (also parameters defined by the waveforms can be excluded)."""

    def __init__(
        self,
        parent: QuantumOp,
        name: str,
        locus: Locus,
        calibration_data: OILCalibrationData,
        builder: ScheduleBuilder,
    ):
        super().__init__(parent, name, locus, calibration_data, builder)
        duration = calibration_data["duration"]  # shared between all pulses
        flux_pulses = {}
        qubit_drive_pulses = {}
        rz = calibration_data["rz"]

        def build_flux_pulse(waveform_class: type[Waveform], component_name: str, cal_node_name: str) -> None:
            """Uses a part of the gate calibration data to prepare a flux pulse for the given component."""
            flux_channel = builder.get_flux_channel(component_name)
            params = self.convert_calibration_data(
                calibration_data[cal_node_name],
                self.parameters[cal_node_name],  # type: ignore[arg-type]
                builder.channels[flux_channel],
                duration=duration,
            )
            amplitude = params.pop("amplitude")
            flux_pulses[flux_channel] = FluxPulse(
                duration=params["n_samples"],
                wave=waveform_class(**params),
                scale=amplitude,
            )

        def build_ac_stark_pulse(component_name: str, cal_node_name: str) -> None:
            """Uses a part of the gate calibration data to prepare a flux pulse for the given component."""
            drive_channel = builder.get_drive_channel(component_name)
            params = self.convert_calibration_data(
                calibration_data[cal_node_name],
                self.parameters[cal_node_name],  # type: ignore[arg-type]
                builder.channels[drive_channel],
                duration=duration,
            )
            params["phase_increment"] = rz[component_name]
            qubit_drive_pulses[drive_channel] = self._ac_stark_pulse(**params)

        if self.coupler_wave is not None:
            build_flux_pulse(self.coupler_wave, builder.chip_topology.get_coupler_for(*locus), "coupler")

        if self.qubit_drive_wave is not None:
            # the pulsed qubit is always the first one of the locus
            build_ac_stark_pulse(locus[0], "first_qubit")
            build_ac_stark_pulse(locus[1], "second_qubit")

        T = max(pulse.duration for pulse in list(flux_pulses.values()) + list(qubit_drive_pulses.values()))
        schedule: dict[str, list[Instruction]] = {}

        for channel, qubit_drive_pulse in qubit_drive_pulses.items():
            schedule[channel] = [qubit_drive_pulse]
        rz_not_locus = tuple((builder.get_drive_channel(c), angle) for c, angle in rz.items() if c not in locus)
        for channel, flux_pulse in flux_pulses.items():  # just one flux pulse here
            if rz_not_locus:
                schedule[channel] = [replace(flux_pulse, rzs=rz_not_locus)]
            else:
                schedule[channel] = [flux_pulse]

        affected_components = set(locus)
        affected_components.add(builder.chip_topology.get_coupler_for(*locus))
        self._affected_components = affected_components

        self._schedule = Schedule(schedule) if T > 0 else Schedule({c: [Block(0)] for c in schedule})

    def __init_subclass__(
        cls, /, coupler_wave: type[Waveform] | None = None, qubit_drive_wave: type[Waveform] | None = None
    ):
        """Store the Waveform types used by this subclass, and their parameters."""
        cls.coupler_wave = coupler_wave
        cls.qubit_drive_wave = qubit_drive_wave
        cls.symmetric = True
        cls.locus_mapping_function = (
            two_component_coupler_flux if qubit_drive_wave is None else two_qubit_drive_coupler_flux
        )

        root_parameters = {k: v for k, v in cls.root_parameters.items() if k not in cls.excluded_parameters}
        parameters = {}
        if coupler_wave is not None:
            parameters["coupler"] = get_waveform_parameters(coupler_wave)
            parameters["coupler"]["amplitude"] = Parameter("", "amplitude", "")

        if qubit_drive_wave is not None:
            for cal_node_name in ["first_qubit", "second_qubit"]:
                # Same AC Stark pulse waveform for both qubits
                parameters[cal_node_name] = get_waveform_parameters(qubit_drive_wave)
                parameters[cal_node_name]["amplitude"] = Parameter("", "amplitude", "")

        cls.parameters = root_parameters | {k: v for k, v in parameters.items() if k not in cls.excluded_parameters}

    @classmethod
    def _ac_stark_pulse(
        cls,
        *,
        phase: float,
        amplitude: float,
        phase_increment: float,
        **kwargs,
    ) -> IQPulse:
        """AC Stark pulse with modulated I and Q waveforms, where the Q quadrature has an additional phase of -pi/2."""
        _, phase_increment = phase_transformation(0.0, phase_increment)

        if cls.qubit_drive_wave is not None:
            wave_i = cls.qubit_drive_wave(phase=phase, **kwargs)
            wave_q = cls.qubit_drive_wave(phase=phase - np.pi / 2, **kwargs)
        return IQPulse(
            kwargs["n_samples"],
            wave_i=wave_i,
            wave_q=wave_q,
            scale_i=amplitude,
            scale_q=amplitude,
            phase_increment=phase_increment,
        )

    def _call(self) -> TimeBox:
        timebox = self.to_timebox(self._schedule)
        timebox.neighborhood_components[0] = self._affected_components
        return timebox

    def duration_in_seconds(self) -> float:
        if self._schedule.duration == 0:
            return 0.0
        return self.builder.channels[list(self._schedule.channels())[0]].duration_to_seconds(self._schedule.duration)


class CZ_Slepian_ACStarkCRF(
    CouplerFluxPulseQubitACStarkPulseGate,
    coupler_wave=Slepian,
    qubit_drive_wave=ModulatedCosineRiseFall,
):
    """Controlled-Z two-qubit gate.

    CZ gate implemented using a slepian flux pulse for the coupler and a modulated cosine rise fall (CRF) AC Stark
    pulse on one qubit.
    """


class CZ_CRF_ACStarkCRF(
    CouplerFluxPulseQubitACStarkPulseGate,
    coupler_wave=CosineRiseFall,
    qubit_drive_wave=ModulatedCosineRiseFall,
):
    """Controlled-Z two-qubit gate.

    CZ gate implemented using a cosine rise fall flux pulse for the coupler and a modulated
    cosine rise fall (CRF) AC Stark pulse on one qubit.
    """


def round_to_granularity(value: float, granularity: float, precision: float = 1e-15) -> float:
    """Round a value to the nearest multiple of granularity.

    If the value is within a given precision of a multiple, round to that multiple.
    Otherwise, round down to the nearest lower multiple.

    Args:
        value: value to round
        granularity: granularity
        precision: rounding precision.

    Returns:
        ``value`` rounded to a granularity.

    """
    return np.floor(value / granularity + precision) * granularity


def split_flat_top_part_into_granular_parts(
    duration: float, full_width: float, rise_time: float, granularity: float, precision: float = 1e-10
) -> tuple[float, float, float, float]:
    """Split a flat-top pulse into three consecutive parts (rise, flat, and fall) to save waveform memory.

    All the parts conform to the granularity of the device.

    Args:
        duration: pulse duration in seconds.
        full_width: full width of the pulse.
        rise_time: rise time of the pulse.
        granularity: minimum allowed pulse duration.
        precision: precision of rounding to granularity,


    Returns:
        A tuple containing:
        - flat part duration
        - rise (or fall) part duration
        - rise time
        - flat part's non-granular leftover, which is transferred to the rise and fall parts

    Raises:
        ValueError: Error is raised if duration is not a multiple of granularity.
        ValueError: Error is raised if pulse parameters do not obey duration >= full_width >= 2*rise_time.

    """
    # Check if the number of samples is within 0.005 samples of an integer number, considered safe.
    if not round(duration / granularity, ndigits=2).is_integer():
        raise ValueError("Duration must be a multiple of granularity.")

    if (duration >= full_width) & (full_width >= 2 * rise_time):
        plateau_width = full_width - 2 * rise_time

        plateau_width_granular = round_to_granularity(plateau_width, granularity)
        rise_duration = (duration - plateau_width_granular) / 2

        if np.abs(rise_duration - np.round(rise_duration / granularity) * granularity) > precision:
            plateau_width_granular -= granularity
            rise_duration = (duration - plateau_width_granular) / 2

        flat_part = duration - 2 * rise_duration
        plateau_leftover = (full_width - 2 * rise_time - flat_part) / 2

        return plateau_width_granular, rise_duration, rise_time, plateau_leftover
    else:
        raise ValueError(
            f"Current pulse parameters (duration {duration}, full_width {full_width}, rise_time {rise_time}) "
            f"are impossible, please use duration >= full_width >= 2*rise_time."
        )


class FluxPulseGate_SmoothConstant(FluxPulseGate):
    """Flux pulse gate implementation which uses a 3-part pulse sequence (cosine rise, constant, cosine fall).

    Otherwise, works similar to FluxPulseGate.

    Args:
        flux_pulses: mapping from flux channel name to its flux pulse
        rz: mapping from drive channel name to the virtual z rotation angle, in radians, that should be performed on it

    """

    coupler_wave: Constant | None
    """Flux pulse Waveform to be played in the coupler flux AWG. Can be only Constant or None"""
    qubit_wave: Constant | None
    """Flux pulse Waveform to be played in the qubit flux AWG. Can be only Constant or None"""
    rise_wave: type[Waveform] = CosineRiseFlex
    """Waveform, rise part of the 3-pulse sequence to be played with qubit and coupler gates."""
    fall_wave: type[Waveform] = CosineFallFlex
    """Waveform, fall part of the 3-pulse sequence to be played with qubit and coupler gates."""

    root_parameters: dict[str, Parameter | Setting | dict] = {
        "duration": Parameter("", "Gate duration", "s"),
        "qubit": {
            "rise_time": Parameter("", "Qubit pulse rise time", "s"),
            "full_width": Parameter("", "Qubit pulse full width", "s"),
            "amplitude": Parameter("", "Qubit pulse amplitude", ""),
        },
        "coupler": {
            "rise_time": Parameter("", "Coupler pulse rise time", "s"),
            "full_width": Parameter("", "Coupler pulse full width", "s"),
            "amplitude": Parameter("", "Coupler pulse amplitude", ""),
        },
        "rz": {
            "*": Parameter("", "Z rotation angle", "rad"),
        },
    }

    def __init__(
        self,
        parent: QuantumOp,
        name: str,
        locus: Locus,
        calibration_data: OILCalibrationData,
        builder: ScheduleBuilder,
    ) -> None:
        GateImplementation.__init__(self, parent, name, locus, calibration_data, builder)
        duration = calibration_data["duration"]

        flux_pulses = {}
        rise_pulses = {}
        fall_pulses = {}

        def build_flux_pulse(waveform_class: type[Waveform], component_name: str, cal_node_name: str) -> None:
            """Uses a part of the gate calibration data to prepare a flux pulse for the given component."""
            flux_channel = builder.get_flux_channel(component_name)

            granularity = builder.channels[flux_channel].duration_to_seconds(
                builder.channels[flux_channel].instruction_duration_min
            )

            data = calibration_data[cal_node_name]
            calibration_data_constant = data.copy()
            calibration_data_rise = data.copy()

            plateau_width_granular, rise_duration, rise_time, plateau_leftover = (
                split_flat_top_part_into_granular_parts(duration, data["full_width"], data["rise_time"], granularity)
            )
            calibration_data_rise["rise_time"] = rise_time
            calibration_data_constant["duration"] = plateau_width_granular
            calibration_data_rise["duration"] = rise_duration
            calibration_data_rise["full_width"] = plateau_leftover + rise_time

            if plateau_width_granular > 0:
                params_for_flux_pulses = self.convert_calibration_data(
                    calibration_data=calibration_data_constant,
                    params=self.parameters[cal_node_name],  # type: ignore[arg-type]
                    channel_props=builder.channels[flux_channel],
                    duration=plateau_width_granular,
                )
            else:
                params_for_flux_pulses = {"n_samples": 0, "amplitude": calibration_data_constant["amplitude"]}

            params_for_risefall = self.convert_calibration_data(
                calibration_data=calibration_data_rise,
                params=self.parameters[cal_node_name],  # type: ignore[arg-type]
                channel_props=builder.channels[flux_channel],
                duration=rise_duration,
            )

            params_for_flux_pulses["n_samples"] = (
                builder.channels[flux_channel].duration_to_int_samples(plateau_width_granular)
                if plateau_width_granular > 0
                else 0
            )

            params_for_risefall["n_samples"] = (
                builder.channels[flux_channel].duration_to_int_samples(rise_duration) if rise_duration > 0 else 0
            )

            amplitude = params_for_flux_pulses.pop("amplitude")
            params_for_risefall.pop("amplitude")

            flux_pulses[flux_channel] = (
                FluxPulse(
                    duration=params_for_flux_pulses["n_samples"],
                    wave=waveform_class(n_samples=params_for_flux_pulses["n_samples"]),
                    scale=amplitude,
                )
                if params_for_flux_pulses["n_samples"] > 0
                else None
            )

            if params_for_risefall["n_samples"] > 0:
                rise_pulses[flux_channel] = FluxPulse(
                    duration=params_for_risefall["n_samples"],
                    wave=self.rise_wave(**params_for_risefall),
                    scale=amplitude,
                )
                fall_pulses[flux_channel] = FluxPulse(
                    duration=params_for_risefall["n_samples"],
                    wave=self.fall_wave(**params_for_risefall),
                    scale=amplitude,
                )
            else:
                rise_pulses[flux_channel] = None  # type: ignore[assignment]
                fall_pulses[flux_channel] = None  # type: ignore[assignment]

        if self.coupler_wave is not None:
            build_flux_pulse(self.coupler_wave, builder.chip_topology.get_coupler_for(*locus), "coupler")

        if self.qubit_wave is not None:
            # the pulsed qubit is always the first one of the locus
            build_flux_pulse(self.qubit_wave, locus[0], "qubit")

        rz = calibration_data["rz"]
        for c in locus:
            if c not in rz:
                raise ValueError(
                    f"{parent.name}.{name}: {locus}: Calibration is missing an RZ angle for locus component {c}."
                )
        rz_locus = {builder.get_drive_channel(c): angle for c, angle in rz.items() if c in locus}
        rz_not_locus = tuple((builder.get_drive_channel(c), angle) for c, angle in rz.items() if c not in locus)

        schedule: dict[str, list[Instruction]] = {
            channel: [
                VirtualRZ(
                    duration=builder.channels[channel].duration_to_int_samples(duration),
                    phase_increment=-angle,
                )
            ]
            for channel, angle in rz_locus.items()
        }
        vzs_inserted = False  # insert the long-distance Vzs to the first flux pulse (whatever that is)
        for channel, flux_pulse in flux_pulses.items():
            if rz_not_locus and not vzs_inserted and flux_pulse:
                schedule[channel] = [replace(flux_pulse, rzs=rz_not_locus)]
                vzs_inserted = True
            elif duration > 0:
                schedule[channel] = [
                    v for v in [rise_pulses[channel], flux_pulse, fall_pulses[channel]] if v is not None
                ]
            else:
                schedule[channel] = []
        affected_components = set(locus)
        affected_components.add(builder.chip_topology.get_coupler_for(*locus))
        self._affected_components = affected_components
        self._schedule = Schedule(schedule if duration > 0 else {c: [Block(0)] for c in schedule}, duration=duration)

    def __init_subclass__(
        cls,
        /,
        coupler_wave: type[Waveform] | None = None,
        qubit_wave: type[Waveform] | None = None,
        rise_wave: type[Waveform] = CosineRiseFlex,
        fall_wave: type[Waveform] = CosineFallFlex,
    ):
        if coupler_wave is None and qubit_wave is None and hasattr(cls, "coupler_wave") and hasattr(cls, "qubit_wave"):
            return
        if coupler_wave and (coupler_wave != Constant):
            logging.getLogger(__name__).warning(
                "Forcing coupler wave to be Constant",
            )
            coupler_wave = Constant
        if qubit_wave and (qubit_wave != Constant):
            logging.getLogger(__name__).warning(
                "Forcing qubit wave to be Constant",
            )
            qubit_wave = Constant

        cls.coupler_wave = coupler_wave
        cls.qubit_wave = qubit_wave
        cls.symmetric = qubit_wave is None
        cls.fall_wave = fall_wave
        cls.rise_wave = rise_wave
        cls.locus_mapping_function = (
            two_component_coupler_flux if qubit_wave is None else two_component_one_flux_coupler_flux
        )

        root_parameters = {k: v for k, v in cls.root_parameters.items() if k not in cls.excluded_parameters}
        parameters = {}
        if coupler_wave is not None:
            parameters["coupler"] = (
                get_waveform_parameters(rise_wave, label_prefix="Coupler flux pulse ")
                | get_waveform_parameters(fall_wave, label_prefix="Coupler flux pulse ")
                | get_waveform_parameters(coupler_wave, label_prefix="Coupler flux pulse ")
            )
            parameters["coupler"]["amplitude"] = Parameter("", "Coupler flux pulse amplitude", "")
            parameters["coupler"]["rise_time"] = Parameter("", "Coupler flux pulse rise time", "s")
            parameters["coupler"]["full_width"] = Parameter("", "Coupler flux pulse full width", "s")

        if qubit_wave is not None:
            parameters["qubit"] = (
                get_waveform_parameters(rise_wave, label_prefix="Qubit flux pulse ")
                | get_waveform_parameters(fall_wave, label_prefix="Qubit flux pulse ")
                | get_waveform_parameters(qubit_wave, label_prefix="Qubit flux pulse ")
            )
            parameters["qubit"]["amplitude"] = Parameter("", "Qubit flux pulse amplitude", "")
            parameters["qubit"]["rise_time"] = Parameter("", "Qubit flux pulse rise time", "s")
            parameters["qubit"]["full_width"] = Parameter("", "Qubit flux pulse full width", "s")

        cls.parameters = root_parameters | {k: v for k, v in parameters.items() if k not in cls.excluded_parameters}


class FluxPulse_SmoothConstant_qubit(FluxPulseGate_SmoothConstant, qubit_wave=Constant):
    """Constant flux pulse on qubit with smooth rise/fall."""


class FluxPulse_SmoothConstant_coupler(FluxPulseGate_SmoothConstant, coupler_wave=Constant):
    """Constant flux pulse on coupler with smooth rise/fall."""


class FluxPulse_SmoothConstant_SmoothConstant(FluxPulseGate_SmoothConstant, coupler_wave=Constant, qubit_wave=Constant):
    """Constant flux pulse on both qubit and coupler with smooth rise/fall."""


class TripleFluxPulseGate(FluxPulseGateBase):
    """Base class for flux pulse gate implementations that have flux pulses for both locus components.

    The parameters for the component flux pulses are under nodes ``first_qubit`` and ``second_qubit`` in the calibration
    data, where the names refer to the locus order.
    """

    def __init_subclass__(
        cls,
        /,
        coupler_wave: type[Waveform] | None = None,
        qubit_wave: type[Waveform] | None = None,
    ) -> None:
        """Set the Waveform types used by this subclass."""
        super().__init_subclass__(coupler_wave, qubit_wave)
        cls.locus_mapping_function = two_qubit_flux_coupler_flux

    def _build_qubit_flux_pulses(self, locus: Locus, duration: float) -> dict[str, FluxPulse]:
        flux_pulses: dict[str, FluxPulse] = {}
        if self.qubit_wave is not None:
            flux_pulses.update(self._build_flux_pulse(self.qubit_wave, locus[0], "first_qubit", duration))
            flux_pulses.update(self._build_flux_pulse(self.qubit_wave, locus[1], "second_qubit", duration))
        return flux_pulses

    @classmethod
    def _get_qubit_pulse_parameters(
        cls, qubit_wave: type[Waveform] | None
    ) -> dict[str, dict[str, Parameter | Setting]]:
        if qubit_wave is None:
            cls.symmetric = True
            return {}

        cls.symmetric = False
        parameters = {
            "first_qubit": get_waveform_parameters(qubit_wave, label_prefix="First qubit flux pulse "),
            "second_qubit": get_waveform_parameters(qubit_wave, label_prefix="Second qubit flux pulse "),
        }
        parameters["first_qubit"]["amplitude"] = Parameter("", "First qubit flux pulse amplitude", "")
        parameters["second_qubit"]["amplitude"] = Parameter("", "Second qubit flux pulse amplitude", "")
        return parameters


class TripleFluxPulseGate_CRF_CRF_CRF(TripleFluxPulseGate, coupler_wave=CosineRiseFall, qubit_wave=CosineRiseFall):
    """CZ gate using a CosineRiseFall flux pulse on the coupler and on both qubits."""


class TripleFluxPulseGate_Slepian_CRF_CRF(TripleFluxPulseGate, coupler_wave=Slepian, qubit_wave=CosineRiseFall):
    """CZ gate using a Slepian flux pulse on the coupler and CosineRiseFall flux pulse on both qubits."""
