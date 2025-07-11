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
r"""Resetting qubits to the zero state.

The reset operation is a non-unitary quantum channel that sets the state of a qubit to :math:`|0\rangle`.
"""

from __future__ import annotations

from collections.abc import Iterable

from exa.common.data.parameter import Parameter
from exa.common.qcm_data.chip_topology import ChipTopology
from iqm.pulse.gate_implementation import (
    SINGLE_COMPONENTS_WITH_READOUT_LOCUS_MAPPING,
    CompositeGate,
    GateImplementation,
)
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

    registered_gates = ["measure"]

    def _call(self) -> TimeBox:  # type: ignore[override]
        # find locus components that are resettable via conditional reset
        resettable = tuple(
            q
            for q in self.locus
            if "drive" in self.builder.component_channels[q] and q in self.builder.chip_topology.component_to_probe_line
        )
        # try to get a qnd measurement, otherwise use default one
        try:  # TODO: should the QND measurement be its own QuantumOp instead?
            probe_timebox = self.build("measure", resettable, impl_name="constant_qnd").probe_timebox(  # type: ignore[attr-defined]
                RESET_MEASUREMENT_KEY, feedback_key=RESET_FEEDBACK_KEY
            )
        except (ValueError, KeyError):
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
            prio_calibration = self.calibration_data if self.calibration_data else None
            waits_box = TimeBox.composite(
                [
                    self.builder.get_implementation(  # type: ignore[attr-defined]
                        self.parent.name, (q,), impl_name=self.name, priority_calibration=prio_calibration
                    ).wait_box()
                    for q in self.locus
                ]
            )
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
        return max(
            self.builder.get_implementation(self.parent.name, (q,), impl_name=self.name).calibration_data["duration"]
            for q in self.locus
        )

    @classmethod
    def get_custom_locus_mapping(
        cls, chip_topology: ChipTopology, component_to_channels: dict[str, Iterable[str]]
    ) -> dict[tuple[str, ...] | frozenset[str], tuple[str, ...]] | None:
        """Supported loci: all components that have channels."""
        return {(c,): (c,) for c in chip_topology.all_components}
