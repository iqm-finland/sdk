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
r"""Qubit-coupler move gate.

Population exchange operation between a qubit and a coupler directly.

* COUPLER_MOVE is unitary.
* COUPLER_MOVE is an involution up to a phase on the target subspace
  :math:`S = \text{Span}(|00\rangle,|01\rangle,|10\rangle)`, and introduces a
  relative phase to the ground state.

In the target subspace, COUPLER_MOVE can be represented as

.. math::
   \text{COUPLER_MOVE}_{S} = |00\rangle \langle 00| + e^{i\phi} |10\rangle \langle 01| + e^{i\phi}|01\rangle \langle 10|

Behaviour in the subspace orthogonal to the target subspace S (including the :math:`|11\rangle` state) is not
defined, although is excitation number preserving. The phase :math:`\phi` is *not*
cancelled when applied a second time and is left uncontrolled. For applications where
this is important it should be determined by sweeping the phase of the final pulses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from exa.common.data.parameter import Parameter, Setting
from iqm.pulse.gate_implementation import (
    CompositeGate,
    GateImplementation,
    Locus,
    OILCalibrationData,
    get_waveform_parameters,
    init_subclass_composite,
)
from iqm.pulse.locus_mappings import LocusMapping, StationProperties
from iqm.pulse.playlist.instructions import Block, FluxPulse, Instruction
from iqm.pulse.playlist.schedule import Schedule
from iqm.pulse.playlist.waveforms import CosineRiseFall, Waveform

if TYPE_CHECKING:  # pragma: no cover
    from iqm.pulse.builder import ScheduleBuilder
    from iqm.pulse.quantum_ops import QuantumOp
    from iqm.pulse.timebox import TimeBox


def coupler_flux_locus_mapping(sp: StationProperties) -> LocusMapping:
    """Get custom locus mapping for this GateImplementation.

    Returns:
        Locus mapping of qubit-coupler pairs for coupler flux implementation.

    """
    chip_topology = sp.qpu_topology
    coupler_connections = chip_topology.coupler_to_components
    locus_mapping: LocusMapping
    locus_mapping = {
        (coupler, component): (
            coupler,
            component,
        )
        for coupler, components in coupler_connections.items()
        for component in components
    }

    return locus_mapping


def coupler_qubit_flux_locus_mapping(sp: StationProperties) -> LocusMapping:
    """Get custom locus mapping for this GateImplementation.

    Returns:
        Locus mapping of qubit-coupler pairs for coupler flux implementation. Returns only
        pairs for which the qubit has fast flux.

    """
    chip_topology = sp.qpu_topology
    coupler_connections = chip_topology.coupler_to_components

    locus_mapping: LocusMapping
    locus_mapping = {
        (coupler, component): (
            coupler,
            component,
        )
        for coupler, components in coupler_connections.items()
        for component in components
        if component in sp.has_fast_flux
    }
    return locus_mapping


class CouplerMOVEGate(GateImplementation):
    r"""Two component coupler_move gate implemented using flux pulse and the interaction mediated by the coupler.

    Implements a population exchange with single qubit phases on the invariant subspace.
    The two components should be a qubit and a coupler. Consists of a flux pulse on the coupler,
    and possibly another flux pulse on the qubit.
    """

    coupler_wave: type[Waveform] | None
    """Coupler flux pulse Waveform"""
    qubit_wave: type[Waveform] | None
    """Coupler flux pulse Waveform"""
    root_parameters: dict[str, Parameter | Setting | dict] = {
        "duration": Parameter("", "Gate duration", "s"),
    }
    excluded_parameters: list[str] = []

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
        flux_pulses: dict[str, FluxPulse] = {}

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

        # Locus mapping fixes locus[0] = coupler, locus[1] = qubit
        if self.coupler_wave is not None:
            build_flux_pulse(self.coupler_wave, locus[0], "coupler")
        if self.qubit_wave is not None:
            build_flux_pulse(self.qubit_wave, locus[1], "qubit")

        # Pulse schedule:
        T = max(pulse.duration for pulse in flux_pulses.values())
        schedule: dict[str, list[Instruction]] = {}
        for channel, flux_pulse in flux_pulses.items():
            schedule[channel] = [flux_pulse]
        self._affected_components = set(locus)
        self._schedule = Schedule(schedule if T > 0 else {c: [Block(0)] for c in schedule}, duration=T)

    def __init_subclass__(cls, /, coupler_wave: type[Waveform] | None = None, qubit_wave: type[Waveform] | None = None):
        """Store Waveform types used by a subclass.

        NOTE: As in :class:FluxPulsegate, care is needed in defining the waves for subclasses.
        Refer to the discussion in :class:FluxPulsegate for more details.
        """
        if coupler_wave is None and qubit_wave is None and hasattr(cls, "coupler_wave") and hasattr(cls, "qubit_wave"):
            return
        cls.coupler_wave = coupler_wave
        cls.qubit_wave = qubit_wave
        cls.symmetric = True
        cls.locus_mapping_function = (
            coupler_flux_locus_mapping if cls.qubit_wave is None else coupler_qubit_flux_locus_mapping
        )

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


class CouplerMOVE_CRF_CRF(CouplerMOVEGate, coupler_wave=CosineRiseFall, qubit_wave=CosineRiseFall):
    """COUPLER_MOVE gate using CRF on both qubit and coupler."""


class CouplerMOVE_CRF(CouplerMOVEGate, coupler_wave=CosineRiseFall):
    """COUPLER_MOVE gate using CRF on coupler only."""
