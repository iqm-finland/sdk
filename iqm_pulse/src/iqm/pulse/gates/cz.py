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

from dataclasses import replace
import logging
from typing import TYPE_CHECKING

import numpy as np

from exa.common.data.parameter import Parameter, Setting
from exa.common.qcm_data.chip_topology import DEFAULT_2QB_MAPPING
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

FLUX_PULSED_QUBITS_2QB_MAPPING: str = "flux_pulsed_qubits_2qb_mapping"


class FluxPulseGate(GateImplementation):
    """Discrete two locus component gate implemented using flux pulses, virtual RZs,
    and the interaction mediated by the coupler.

    Does not take any parameters since it is discrete.

    The two locus components of the gate must be coupled by a tunable coupler.

    Consists of a flux pulse for the coupler, and possibly another one for the first locus component,
    assumed to be a qubit, both with arbitrary waveforms, and virtual RZs on both components.
    Inherit from this class and assign
    waveforms to the ``coupler_wave`` and ``qubit_wave`` pulse slots to create a specific implementation.

    Can be used as a base class for both CZ and MOVE gate implementations.

    Note: the coupler and qubit pulses typically have the same duration (given in the calibration data), and in the
    special case of the duration being zero, the gate implementation will apply ``Block(0)`` instructions
    to all the channels where it would otherwise apply flux pulses or virtual z rotations.

    Args:
        flux_pulses: mapping from flux channel name to its flux pulse
        rz: mapping from drive channel name to the virtual z rotation angle, in radians, that should be performed on it

    """

    coupler_wave: type[Waveform] | None
    """Flux pulse Waveform to be played in the coupler flux AWG."""
    qubit_wave: type[Waveform] | None
    """Flux pulse Waveform to be played in the qubit flux AWG."""
    root_parameters: dict[str, Parameter | Setting | dict] = {
        "duration": Parameter("", "Gate duration", "s"),
        "rz": {
            "*": Parameter("", "Z rotation angle", "rad"),  # wildcard parameter
        },
    }
    """Parameters shared by all ``FluxPulseGate`` classes. Inheriting classes may override this if there's
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
        duration = calibration_data["duration"]  # shared between all pulses
        flux_pulses = {}

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

        if self.coupler_wave is not None:
            build_flux_pulse(self.coupler_wave, builder.chip_topology.get_coupler_for(*locus), "coupler")

        if self.qubit_wave is not None:
            # the pulsed qubit is always the first one of the locus
            build_flux_pulse(self.qubit_wave, locus[0], "qubit")

        rz = calibration_data["rz"]
        # rz angles are required for the two locus components, and are optional for every other drivable QPU component
        # NOTE computational resonators cannot be driven directly, instead they have "virtual" drive channels
        # that are removed by the compiler at the end of the compilation, and implemented by other means.
        for c in locus:
            if c not in rz:
                # TODO self.qualified_name when __init__ and construct are merged
                raise ValueError(
                    f"{parent.name}.{name}: {locus}: Calibration is missing an RZ angle for locus component {c}."
                )
        rz_locus = {builder.get_drive_channel(c): angle for c, angle in rz.items() if c in locus}
        rz_not_locus = tuple((builder.get_drive_channel(c), angle) for c, angle in rz.items() if c not in locus)
        # No driving must happen on any of the affected components during the flux pulses,
        # hence the virtual z rotations must use up their entire duration.
        T = max(pulse.duration for pulse in flux_pulses.values())
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
        affected_components = set(locus)
        affected_components.add(builder.chip_topology.get_coupler_for(*locus))
        self._affected_components = affected_components
        self._schedule = Schedule(schedule if T > 0 else {c: [Block(0)] for c in schedule}, duration=T)

    def __init_subclass__(cls, /, coupler_wave: type[Waveform] | None = None, qubit_wave: type[Waveform] | None = None):
        """Store the Waveform types used by this subclass, and their parameters.

        NOTE: if ``MyCZ`` is a subclass of ``FluxPulseGate``, with some defined coupler and qubit waves, further
        inheriting from it like this ``class MySubSubClass(MyCZ, coupler_wave=Something, qubit_wave=SomethingElse)``
        changes the waves accordingly. If you do not provide any waves: ``class MySubSubClass(MyCZ)``,
        the waves defined in ``MyCZ`` will be retained. If you provide just one wave:
        ``class MySubSubClass(MyCZ, coupler_wave=Something)``, the other wave will be initialised as ``None``.

        Args:
            coupler_wave: flux pulse waveform to be played in the coupler flux AWG. Can be set as `None` if
                no coupler flux pulse should be played in this gate implementation.
            qubit_wave: flux pulse waveform to be played in the qubit flux AWG. Can be set as `None` if
                no qubit flux pulse should be played in this gate implementation.

        """
        # fix __init_subclass__ behaviour for further inheritance from a subclass of FluxPulseGate
        # we can skip this function if the class attributes are already stored in the parent class
        # and the subsubclass definition does not change these
        # see more info in: https://stackoverflow.com/questions/55183288/inheriting-init-subclass-parameters
        # the unintuitive default ``None`` values and handling of these values is for overcoming this issue
        # so that the method itself behaves as expected in successive subclassing
        if coupler_wave is None and qubit_wave is None and hasattr(cls, "coupler_wave") and hasattr(cls, "qubit_wave"):
            return
        cls.coupler_wave = coupler_wave
        cls.qubit_wave = qubit_wave
        cls.symmetric = cls.qubit_wave is None

        root_parameters = {k: v for k, v in cls.root_parameters.items() if k not in cls.excluded_parameters}
        parameters = {}
        if coupler_wave is not None:
            parameters["coupler"] = get_waveform_parameters(coupler_wave, label_prefix="Coupler flux pulse ")
            parameters["coupler"]["amplitude"] = Parameter("", "Coupler flux pulse amplitude", "")

        if qubit_wave is not None:
            parameters["qubit"] = get_waveform_parameters(qubit_wave, label_prefix="Qubit flux pulse ")
            parameters["qubit"]["amplitude"] = Parameter("", "Qubit flux pulse amplitude", "")

        cls.parameters = root_parameters | {k: v for k, v in parameters.items() if k not in cls.excluded_parameters}
        if issubclass(cls, CompositeGate):
            init_subclass_composite(cls)

    def _call(self) -> TimeBox:
        timebox = self.to_timebox(self._schedule)
        timebox.neighborhood_components[0] = self._affected_components
        return timebox

    def duration_in_seconds(self) -> float:
        if self._schedule.duration == 0:
            return 0.0
        return self.builder.channels[list(self._schedule.channels())[0]].duration_to_seconds(self._schedule.duration)

    @classmethod
    def get_locus_mapping_name(cls, operation_name: str, implementation_name: str) -> str:
        if cls.qubit_wave:
            return FLUX_PULSED_QUBITS_2QB_MAPPING
        return DEFAULT_2QB_MAPPING


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
    """CZ gate using a TruncatedGaussianSmoothedSquare flux pulse on the coupler and a CosineRiseFall
    flux pulse on the qubit.
    """


class FluxPulseGate_CRF_CRF(FluxPulseGate, coupler_wave=CosineRiseFall, qubit_wave=CosineRiseFall):
    """CZ gate using a CosineRiseFall flux pulse on the coupler and on the qubit."""


AC_STARK_PULSED_QUBITS_2QB_MAPPING: str = "ac_stark_pulsed_qubits_2qb_mapping"


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
        """Returns an AC Stark pulse which consists of a modulated I and modulated Q waveform, where the Q quadrature
        has an additional phase of -pi/2.
        """
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

    @classmethod
    def get_locus_mapping_name(cls, operation_name: str, implementation_name: str) -> str:
        if cls.qubit_drive_wave:
            return AC_STARK_PULSED_QUBITS_2QB_MAPPING
        return DEFAULT_2QB_MAPPING


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
        value:
        granularity: granularity
        precision: rounding precision.

    Returns:
        value rounded to a granularity.

    """
    return np.floor(value / granularity + precision) * granularity


def split_flat_top_part_into_granular_parts(
    duration: float, full_width: float, rise_time: float, granularity: float, precision: float = 1e-10
) -> tuple[float, float, float, float]:
    """To save waveform memory, a (long) flat-top pulse, which is defined by its duration, full_width and rise_time,
    is divided into three consecutive parts (rise, flat, and fall),
    all of which conform to the granularity of the device.

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
    """Flux pulse gate implementation realized as a 3-part pulse sequence,
    consisting of |cosine rise|Constant|cosine fall|. Otherwise, works similar to FluxPulseGate.

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
        cls.symmetric = cls.qubit_wave is None
        cls.fall_wave = fall_wave
        cls.rise_wave = rise_wave

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
    """Constant flux pulse on qubit with smooth rise/fall"""


class FluxPulse_SmoothConstant_coupler(FluxPulseGate_SmoothConstant, coupler_wave=Constant):
    """Constant flux pulse on coupler with smooth rise/fall."""


class FluxPulse_SmoothConstant_SmoothConstant(FluxPulseGate_SmoothConstant, coupler_wave=Constant, qubit_wave=Constant):
    """Constant flux pulse on both qubit and coupler with smooth rise/fall."""
