# Copyright 2022-2026 IQM Finland Oy
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
"""Flux Pulse Gate for implementing a flux pulse on a qubit or a coupler."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from exa.common.data import Parameter, Setting
from iqm.pulse.gate_implementation import GateImplementation, Locus, OILCalibrationData, get_waveform_parameters
from iqm.pulse.gates.cz import FluxPulseGate, split_flat_top_part_into_granular_parts
from iqm.pulse.locus_mappings import one_component_flux
from iqm.pulse.playlist.instructions import RealPulse
from iqm.pulse.playlist.schedule import Schedule
from iqm.pulse.playlist.waveforms import (
    Constant,
    Cosine,
    CosineFallFlex,
    CosineRiseFall,
    CosineRiseFlex,
    PiecewiseConstant,
    PolynomialCosine,
    Waveform,
)

if TYPE_CHECKING:  # pragma: no cover
    from iqm.pulse.builder import ScheduleBuilder
    from iqm.pulse.quantum_ops import QuantumOp
    from iqm.pulse.timebox import TimeBox


class FluxPulse(GateImplementation):
    """Base gate implementation for implementing flux pulses for any flux channel with arbitrary waveforms.

    Applies the flux pulse on the flux channel of the first locus component.
    Does not take any parameters since it is discrete.

    Inherit from this class and assign waveforms to the ``wave`` pulse slot to create a specific implementation.
    """

    wave: type[Waveform]
    locus_mapping_function = one_component_flux

    def __init__(
        self,
        parent: QuantumOp,
        name: str,
        locus: Locus,
        calibration_data: OILCalibrationData,
        builder: ScheduleBuilder,
    ):
        """Uses a part of the gate calibration data to prepare a flux pulse for the given component."""
        super().__init__(parent, name, locus, calibration_data, builder)
        flux_channel = builder.get_flux_channel(locus[0])
        params = self.convert_calibration_data(
            calibration_data,
            self.parameters,
            builder.channels[flux_channel],
        )
        amplitude = params.pop("amplitude")

        flux_pulse = RealPulse(
            duration=params["n_samples"],
            wave=self.wave(**params),
            scale=amplitude,
        )
        self.schedule = Schedule({flux_channel: [flux_pulse]})

    def __init_subclass__(cls, /, wave: type[Waveform]):
        """Store the Waveform type used by this subclass, and its parameters."""
        cls.wave = wave
        parameters = get_waveform_parameters(wave)
        parameters["amplitude"] = Parameter("", "amplitude", "")
        for parameter_name, parameter in parameters.items():
            parameters[parameter_name] = parameter.model_copy(
                update={"label": f"{parameter_name} flux pulse {parameter.label}"},
            )

        cls.parameters = {  # type: ignore[assignment]
            "duration": Parameter("", "Gate duration", "s"),
        } | parameters

    def __call__(self) -> TimeBox:
        return self.to_timebox(self.schedule)

    def duration_in_seconds(self) -> float:
        if self.schedule.duration == 0:
            return 0.0
        return self.builder.channels[list(self.schedule.channels())[0]].duration_to_seconds(self.schedule.duration)


class ConstantFluxPulse(FluxPulse, wave=Constant):
    """Flux pulse with a constant waveform."""


class CosineFluxPulse(FluxPulse, wave=Cosine):
    """Flux pulse with a cosine waveform."""


class CRFFluxPulse(FluxPulse, wave=CosineRiseFall):
    """Flux pulse with a constant waveform."""


class PiecewiseConstantFluxPulse(FluxPulse, wave=PiecewiseConstant):
    """Flux pulse with a piecewise constant waveform."""

    def __init__(
        self,
        parent: QuantumOp,
        name: str,
        locus: Locus,
        calibration_data: OILCalibrationData,
        builder: ScheduleBuilder,
    ):
        # Convert from user-facing [0, duration] to internal [-duration/2, duration/2]
        calibration_data["changepoints"] = calibration_data["changepoints"] - 0.5 * calibration_data["duration"]
        super().__init__(parent, name, locus, calibration_data, builder)


class PolynomialCosineFluxPulse(FluxPulse, wave=PolynomialCosine):
    """Flux pulse with a cosine waveform transformed by a polynomial.

    Used to implement oscillations in frequency domain.
    """


class CouplerFluxPulseSquare(FluxPulseGate, coupler_wave=Constant):
    """A gate to implement a single square flux pulse on the coupler only."""


Coupler_FluxPulse_Square = CouplerFluxPulseSquare


class FluxPulseSmoothConstantBase(FluxPulse, wave=Constant):
    """Qubit flux pulse gate implementation realized as a 3-part pulse sequence,
    consisting of (cosine rise, constant, cosine fall). Otherwise, works similar to exa-core FluxPulse.
    """

    wave: type[Waveform] = Constant
    """Flux pulse Waveform to be played in the coupler flux AWG. Can be only Constant or None"""
    rise_wave: type[Waveform] = CosineRiseFlex
    """Waveform, rise part of the 3-pulse sequence to be played with qubit and coupler gates."""
    fall_wave: type[Waveform] = CosineFallFlex
    """Waveform, fall part of the 3-pulse sequence to be played with qubit and coupler gates."""

    root_parameters: dict[str, Parameter | Setting | dict] = {
        "duration": Parameter("", "Gate duration", "s"),
        "rise_time": Parameter("", "Qubit pulse rise time", "s"),
        "full_width": Parameter("", "Qubit pulse full width", "s"),
        "amplitude": Parameter("", "Qubit pulse amplitude", ""),
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

        flux_channel = builder.get_flux_channel(locus[0])
        granularity = builder.channels[flux_channel].duration_to_seconds(
            builder.channels[flux_channel].instruction_duration_min
        )

        calibration_data_constant = calibration_data.copy()
        calibration_data_rise = calibration_data.copy()

        plateau_width_granular, rise_duration, rise_time, plateau_leftover = split_flat_top_part_into_granular_parts(
            calibration_data["duration"],
            calibration_data["full_width"],
            calibration_data["rise_time"],
            granularity,
        )
        calibration_data_rise["rise_time"] = rise_time
        calibration_data_constant["duration"] = plateau_width_granular
        calibration_data_rise["duration"] = rise_duration
        calibration_data_rise["full_width"] = plateau_leftover + rise_time

        if plateau_width_granular > 0:
            params_for_flux_pulses = self.convert_calibration_data(
                calibration_data=calibration_data_constant,
                params=self.parameters,
                channel_props=builder.channels[flux_channel],
                duration=plateau_width_granular,
            )
        else:
            params_for_flux_pulses = {"n_samples": 0, "amplitude": calibration_data_constant["amplitude"]}

        params_for_risefall = self.convert_calibration_data(
            calibration_data=calibration_data_rise,
            params=self.parameters,
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

        flux_pulse = RealPulse(
            duration=params_for_flux_pulses["n_samples"],
            wave=self.wave(n_samples=params_for_flux_pulses["n_samples"]),
            scale=amplitude,
        )
        rise_pulse = RealPulse(
            duration=params_for_risefall["n_samples"],
            wave=self.rise_wave(**params_for_risefall),
            scale=amplitude,
        )
        fall_pulse = RealPulse(
            duration=params_for_risefall["n_samples"],
            wave=self.fall_wave(**params_for_risefall),
            scale=amplitude,
        )

        if calibration_data["duration"] > 0:
            self.schedule = Schedule({flux_channel: [rise_pulse, flux_pulse, fall_pulse]})
        else:
            self.schedule = Schedule({flux_channel: []})

    def __init_subclass__(
        cls,
        /,
        wave: type[Waveform] = Constant,
        rise_wave: type[Waveform] = CosineRiseFlex,
        fall_wave: type[Waveform] = CosineFallFlex,
    ):
        if wave is None and hasattr(cls, "wave"):
            return
        if wave and (wave != Constant):
            logging.getLogger(__name__).warning(
                "Forcing coupler wave to be Constant",
            )
            wave = Constant

        cls.wave = wave
        cls.symmetric = cls.wave is None
        cls.fall_wave = fall_wave
        cls.rise_wave = rise_wave

        cls.parameters = cls.root_parameters
        root_parameters = (
            cls.root_parameters
        )  # TODO was {k: v for k, v in cls.root_parameters.items()}. check and remove the comment
        parameters = {}
        if wave is not None:
            parameters = (
                get_waveform_parameters(rise_wave, label_prefix="Flux pulse ")
                | get_waveform_parameters(fall_wave, label_prefix="Flux pulse ")
                | get_waveform_parameters(wave, label_prefix="Flux pulse ")
            )
            parameters["amplitude"] = Parameter("", "Flux pulse amplitude", "")
        cls.parameters = root_parameters | dict(parameters.items())


class FluxPulseSmoothConstant(FluxPulseSmoothConstantBase, wave=Constant):
    """A gate to implement a single square flux pulse on the coupler only."""
