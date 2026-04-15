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
r"""Qubit reset operations.

The reset operation is a non-unitary quantum channel that sets the state of a qubit to :math:`|0\rangle`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

from exa.common.data.parameter import DataType, Parameter, Setting
from exa.common.qcm_data.chip_topology import ChipTopology
from iqm.pulse.gate_implementation import (
    SINGLE_COMPONENTS_WITH_DRIVE_LOCUS_MAPPING,
    SINGLE_COMPONENTS_WITH_READOUT_LOCUS_MAPPING,
    CompositeGate,
    GateImplementation,
    SinglePulseGate,
)
from iqm.pulse.playlist import IQPulse, MultiplexedIQPulse
from iqm.pulse.playlist.waveforms import CosineRiseFall, CosineRiseFallDerivative
from iqm.pulse.timebox import TimeBox

RESET_MEASUREMENT_KEY = "__reset"
"""Constant measurement key for the measure operation required in the fast-feedback reset gate (the double underscore
emphasizes the fact that this label should not be manually used in fast feedback)."""

RESET_FEEDBACK_KEY = "__default_feedback"
"""The feedback key used in fast-feedback reset gate (the double underscore emphasizes the fact that this label should
not be manually used in fast feedback)."""


class Reset_Conditional(CompositeGate):
    r"""Conditional reset operation.

    Uses a measurement followed by a conditional PRX gate with angle :math:`\pi`.
    It is assumed the measurement projects the state into the computational basis.

    The conditional PRX implementation handles any necessary waits to accommodate for the feedback result propagation
    delay.

    This reset implementation is factorizable. It can act upon any set of locus components, and the measurement
    used in the conditional reset will be multiplexed to those components. However, only locus components that have
    readout and drive can be reset via conditional reset. Otherwise, locus components will just have their channels
    blocked.
    """

    registered_gates = ("measure", "cc_prx")

    def _call(self) -> TimeBox:  # type: ignore[override]
        # find locus components that are resettable via conditional reset
        resettable = tuple(
            q
            for q in self.locus
            if "drive" in self.builder.component_channels[q] and q in self.builder.chip_topology.component_to_probe_line
        )
        probe_timebox = self.build("measure", resettable).probe_timebox(  # type: ignore[attr-defined]
            RESET_MEASUREMENT_KEY, feedback_key=RESET_FEEDBACK_KEY
        )
        virtual_channels = set()
        probes = set()
        resets = []
        for component in resettable:
            virtual_channels.add(
                self.builder.get_virtual_feedback_channel_for(self.builder.get_drive_channel(component), component)
            )
            probes.add(self.builder.chip_topology.component_to_probe_line[component])
            resets.append(self.build("cc_prx", (component,))(feedback_qubit=component, feedback_key=RESET_FEEDBACK_KEY))
        # adjust the blocking neighborhood, as we know the feedforward bits are sent only to self
        # block the probes and the fast feedback virtual channels for self-consistency
        blocks = set(self.locus).union(probes).union(virtual_channels)
        # block common couplers for multi-qubit loci
        if len(self.locus) > 1:
            blocks.update(set(self.builder.chip_topology.get_connecting_couplers(self.locus)))
        probe_timebox.neighborhood_components[0] = blocks
        measure = TimeBox.composite([probe_timebox])
        measure.neighborhood_components[0] = blocks
        return TimeBox.composite([measure, resets])  # type: ignore[list-item]

    @classmethod
    def get_locus_mapping_name(cls, operation_name: str, implementation_name: str) -> str:
        return SINGLE_COMPONENTS_WITH_READOUT_LOCUS_MAPPING


class Reset_Wait(GateImplementation):
    """Reset operation by relaxation (idling for a time comparable to the relaxation time of the component).

    Adds a Wait pulse for all the (non-virtual) channels of the locus components. In addition, blocks all the probes
    associated with the locus and in case the locus is connected via couplers, blocks them as well. The operation
    is factorizable, so that the calibration data it uses (i.e. the wait duration in seconds) is defined for each
    component, and when acting on multiple components, the maximum of the associated wait durations will be applied.

    Reset by relaxation is intended to be used in the circuit initialisation between shots / segments.
    It also can be called on specific qubits inside a segment, but as it works by
    waiting longer than the qubit relaxation time, the states of all the other qubits
    will likely have been destroyed as well by the time the reset has finished.
    """

    parameters = {"duration": Parameter("", "Wait duration", "s")}

    def wait_box(self) -> TimeBox:
        """TimeBox that contains Wait instructions for all non-virtual channels associated with ``self.locus``.

        Does not block any additional components beside the locus itself.
        """
        if len(self.locus) == 1:
            waits_box = self.builder.wait(self.locus, self.calibration_data["duration"], rounding=True)
        else:
            # factorizability: use the sub-implementations
            waits_box = TimeBox.composite([self.sub_implementations[c].wait_box() for c in self.locus])  # type: ignore[attr-defined]
        return waits_box

    def _call(self) -> TimeBox:
        waits_box = self.wait_box()
        # block also probe lines
        blocks = {
            self.builder.chip_topology.component_to_probe_line[q]
            for q in self.locus
            if q in self.builder.chip_topology.component_to_probe_line
        }.union(waits_box.locus_components)
        # block connecting couplers for a multi-qubit locus
        if len(self.locus) > 1:
            blocks.update(set(self.builder.chip_topology.get_connecting_couplers(self.locus)))
        waits_box.neighborhood_components[0] = blocks
        # wrap into one more TimeBox for self-consistency
        return TimeBox.composite([waits_box])

    def duration_in_seconds(self) -> float:
        if len(self.locus) == 1:
            return self.calibration_data["duration"]
        return max(self.sub_implementations[c].calibration_data["duration"] for c in self.locus)

    @classmethod
    def get_custom_locus_mapping(
        cls, chip_topology: ChipTopology, component_to_channels: Mapping[str, Iterable[str]]
    ) -> dict[tuple[str, ...] | frozenset[str], tuple[str, ...]] | None:
        """Supported loci: all components that have channels."""
        return {(c,): (c,) for c in chip_topology.all_components}


class Reset_F0G1(SinglePulseGate):
    r"""Reset by driving the :math:`|f0\rangle \leftrightarrow |g1\rangle` transition.

    This gate implementation resets the qubit by driving the :math:`|f0\rangle \leftrightarrow |g1\rangle` transition
    (basis here: :math:`|\text{qubit}, \text{readout resonator}\rangle`)
    and the :math:`|e\rangle \leftrightarrow |f\rangle` transition simultaneously.
    The principle is the same as for the ``lru`` gate (see :class:`.LRU_F0G1`), except here we additionally drive
    the :math:`|e\rangle \leftrightarrow |f\rangle` to also reset the :math:`|e\rangle` state,
    not just the :math:`|f\rangle` state.
    The reset happens due to quick dissipation in the resonator.
    This operation is therefore non-unitary.

    The gate is implemented via a :class:``MultiplexedIQPulse``,
    combining two modulated pulses which are supposed to drive each of the above mentioned transitions.

    See :cite:`Magnard_2018` for reference of an experiment where such a pulse was used.
    """

    parameters = {
        "duration": Parameter("", "Multiplexed pulse duration", "s"),
        "f0g1_pulse": {
            "amplitude": Parameter("", "F0G1 pulse amplitude", ""),
            "full_width": Parameter("", "F0G1 pulse duration", "s"),
            "rise_time": Parameter("", "F0G1 pulse rise time", "s"),
            "modulation_frequency": Setting(Parameter("", "F0G1 pulse modulation frequency", "Hz"), 0.0),
        },
        "ef_pulse": {
            "amplitude_i": Parameter("", "EF pulse channel I amplitude", ""),
            "amplitude_q": Parameter("", "EF pulse channel Q amplitude", ""),
            "full_width": Parameter("", "EF pulse full width", "s"),
            "rise_time": Parameter("", "EF pulse rise time", "s"),
            "modulation_frequency": Setting(Parameter("", "EF pulse modulation frequency", "Hz"), 0.0),
            "offset": Setting(Parameter("", "EF pulse offset from the F0G1 pulse", "s", data_type=DataType.INT), 0.0),
        },
    }

    @classmethod
    def _get_pulse(  # type: ignore[override]
        cls,
        *,
        n_samples: int,
        **rest_of_calibration_data,
    ) -> MultiplexedIQPulse:
        f0g1_cal = rest_of_calibration_data["f0g1_pulse"]
        ef_cal = rest_of_calibration_data["ef_pulse"]

        f0g1_mod_freq = f0g1_cal.pop("modulation_frequency") / n_samples
        ef_mod_freq = ef_cal.pop("modulation_frequency") / n_samples
        offset = int(ef_cal.pop("offset"))
        ef_scale_i = ef_cal.pop("amplitude_i")
        ef_scale_q = ef_cal.pop("amplitude_q")
        f0g1_scale = f0g1_cal.pop("amplitude")

        f0g1_wave = CosineRiseFall(n_samples=n_samples, **f0g1_cal)
        instruction = MultiplexedIQPulse(
            n_samples,
            entries=(
                (
                    IQPulse(
                        n_samples,
                        scale_i=f0g1_scale,
                        scale_q=0,
                        wave_i=f0g1_wave,
                        wave_q=f0g1_wave,
                        modulation_frequency=f0g1_mod_freq,
                        phase=0,
                        phase_increment=0,
                    ),
                    0,
                ),
                (
                    IQPulse(
                        n_samples,
                        scale_i=ef_scale_i,
                        scale_q=ef_scale_q,
                        wave_i=CosineRiseFall(n_samples=n_samples, **ef_cal),
                        wave_q=CosineRiseFallDerivative(n_samples=n_samples, **ef_cal),
                        modulation_frequency=ef_mod_freq,
                        phase=0,
                        phase_increment=0,
                    ),
                    offset,
                ),
            ),
        )
        return instruction

    @classmethod
    def get_locus_mapping_name(cls, operation_name: str, implementation_name: str) -> str:
        return SINGLE_COMPONENTS_WITH_DRIVE_LOCUS_MAPPING


class Reset_F0G1_Composite(CompositeGate):
    r"""Reset by applying a ``prx_12(pi)`` gate followed by the ``lru.f0g1`` operation.

    The ``prx_12(pi)`` swaps the populations of the states :math:`|1\rangle` and :math:`|2\rangle`,
    and the ``lru.f0g1`` moves the :math:`|2\rangle` state population to :math:`|0\rangle`.

    The parameter ``number_of_cycles`` specifies how many times the sequence of gates (prx_12, lru) will be repeated.

    Note that the F0G1 pulse here is take from the LRU gate, since this is a composite gate.
    """

    registered_gates: tuple[str, ...] = ("lru",)
    default_implementations: dict[str, str] = {"lru": "f0g1"}
    parameters = {
        "number_of_cycles": Setting(
            Parameter("", "Number of cycles of the [prx_12(pi), lru] sequence", "", data_type=DataType.INT), 1
        ),
    }

    def _call(self) -> TimeBox:  # type: ignore[override]
        f0g1 = self.build("lru", self.locus)()
        prx_12 = self.build("prx_12", self.locus)(np.pi)
        boxes = [prx_12, f0g1] * self.calibration_data["number_of_cycles"]
        return TimeBox.composite(boxes)

    @classmethod
    def get_locus_mapping_name(cls, operation_name: str, implementation_name: str) -> str:
        return SINGLE_COMPONENTS_WITH_DRIVE_LOCUS_MAPPING
