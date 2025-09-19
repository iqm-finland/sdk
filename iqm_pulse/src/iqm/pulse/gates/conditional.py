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
"""Classically controlled gates."""

import numpy as np

from exa.common.data.parameter import CollectionType, Parameter
from iqm.pulse.gate_implementation import CompositeGate
from iqm.pulse.gates.measure import FEEDBACK_KEY
from iqm.pulse.gates.prx import PRX_SinglePulse_GateImplementation
from iqm.pulse.playlist.instructions import Block, ConditionalInstruction, Wait
from iqm.pulse.playlist.schedule import Schedule
from iqm.pulse.timebox import TimeBox


class CCPRX_Composite(CompositeGate):
    """Classically controlled PRX gate.

    Applies a PRX gate conditioned on a discriminated readout result obtained in the same segment (active feedback).
    Applies a PRX gate if the result is 1, and a Wait of equal duration if the result is 0.
    Uses the default implementation of PRX underneath, so no extra calibration is needed.
    """

    registered_gates = ("prx",)

    parameters = {"control_delays": Parameter("", "Control delays", "s", collection_type=CollectionType.NDARRAY)}
    """``control_delays`` contains the times it takes for the classical control signal from each
    probe line (readout instrument) to become usable for the drive AWG implementing the PRX gate.
    The delays must be in the same order as the probe lines are listed in
    the ``{drive_controller}.awg.feedback_sources`` station setting.
    """

    def _call(
        self, angle: float = np.pi, phase: float = 0.0, *, feedback_qubit: str, feedback_key: str
    ) -> list[TimeBox]:
        """Two TimeBoxes that together implement the classically controlled PRX gate.

        The first Timebox is for the control signal delay, and the second has a ConditionalInstruction.
        The delay TimeBox operates only on a virtual channel and is used to block the pulse TimeBox
        until there has been enough time for the control signal to arrive.
        The delay is specified by the ``control_delays`` gate parameter.

        In normal operation, the boxes can be placed sequentially without causing unnecessary delays.
        To care of the timing yourself, simply ignore the first TimeBox.

        Args:
            angle: The PRX rotation angle (rad).
            phase: The PRX rotation phase (rad).
            feedback_qubit: The qubit that was measured to create the feedback bit.
            feedback_key: Identifies the feedback signal if ``feedback_qubit`` was measured multiple times.
                The feedback label is then ``f"{feedback_qubit}__{feedback_key}"``.
                TODO: currently the HW does not support multiple feedback keys per drive channel, so this argument has
                no effect. The conditional prx will always listen feedback from the label
                ``f"{feedback_qubit}__{FEEDBACK_KEY}"``. When the HW is improved, the actual key the user inputs
                should be used.

        Returns:
            A TimeBox for the signal delay, and a TimeBox with a ConditionalInstruction inside.

        """
        qubit = self.locus[0]
        awg_name = self.builder.get_drive_channel(qubit)

        prx_gate: PRX_SinglePulse_GateImplementation = self.build("prx", self.locus)  # type: ignore[assignment]
        # FIXME assumes PRX gates only use this one implementation as the default,
        # with just a drive channel and a single IQPulse.
        pulse = prx_gate(angle, phase).atom[prx_gate.channel][0]  # type: ignore[union-attr,index]
        wait = Wait(pulse.duration)  # idling, can be replaced with a DD sequence later on

        # TODO: use the actual inputted label when the HW supports many labels per drive channel
        default_label = f"{feedback_qubit}__{FEEDBACK_KEY}"
        pulse_instruction = ConditionalInstruction(
            duration=pulse.duration, condition=default_label, outcomes=(wait, pulse)
        )
        delays = self.calibration_data["control_delays"]
        if len(delays) == 0:
            raise ValueError(f"'control_delays' for '{self.name}' on {qubit} is empty (not calibrated).")

        possible_sources = [c for c in self.builder.get_virtual_feedback_channels(qubit) if awg_name in c]
        if len(delays) != len(possible_sources):
            raise ValueError(
                f"Not the correct amount of calibration values for 'control_delays'. Need {len(possible_sources)}"
                f"values, got {delays}."
            )
        virtual_channel_name = self.builder.get_virtual_feedback_channel_for(awg_name, feedback_qubit)
        delay = delays[possible_sources.index(virtual_channel_name)]
        virtual_channel = self.builder.channels[virtual_channel_name]
        delay_samples = virtual_channel.duration_to_int_samples(
            virtual_channel.round_duration_to_granularity(delay, round_up=True), check_min_samples=False
        )
        delay_box = TimeBox.atomic(
            Schedule({virtual_channel_name: [Block(delay_samples)]}),
            locus_components=[],
            label=f"Feedback signal delay for {qubit}",
        )
        delay_box.neighborhood_components = {0: {virtual_channel_name}}
        cond = TimeBox.atomic(
            Schedule({virtual_channel_name: [Block(0)], prx_gate.channel: [pulse_instruction]}),
            locus_components=[qubit],
            label=f"Conditional PRX for {qubit}",
        )
        cond.neighborhood_components = {0: {virtual_channel_name, qubit}}
        return [
            delay_box,
            cond,
        ]


class CCPRX_Composite_DRAGCosineRiseFall(CCPRX_Composite):
    """Conditional drag_crf pulse."""

    default_implementations = {"prx": "drag_crf"}


class CCPRX_Composite_DRAGGaussian(CCPRX_Composite):
    """Conditional drag_gaussian pulse."""

    default_implementations = {"prx": "drag_gaussian"}
