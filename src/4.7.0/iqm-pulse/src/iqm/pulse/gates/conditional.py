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

from exa.common.data import CollectionType, Parameter
from iqm.pulse.gate_implementation import CompositeGate
from iqm.pulse.playlist.channel import _FAST_FEEDBACK_CHANNEL
from iqm.pulse.playlist.instructions import Block, ConditionalInstruction, IQPulse, Wait
from iqm.pulse.playlist.schedule import Schedule
from iqm.pulse.timebox import TimeBox
from iqm.pulse.utils import fuse_iq_pulses


class CCPRX_Composite(CompositeGate):
    """Classically controlled PRX gate.

    Applies a PRX gate conditioned on a discriminated readout result obtained in the same segment (active feedback).
    Applies a PRX gate if the result is 1, and a Wait of equal duration if the result is 0.
    Uses the default implementation of PRX underneath, so no extra calibration is needed.

    .. note::

       Assumes that the PRX gate implementation used only consists of IQPulse instructions on the drive channel
       of its locus qubit.

    """

    registered_gates = ("prx",)

    parameters = {"control_delays": Parameter("", "Control delays", "s", collection_type=CollectionType.NDARRAY)}
    """``control_delays`` contains the times it takes for the classical control signal from each
    probe line (readout instrument) to become usable for the drive AWG implementing the PRX gate.
    The delays must be in the same order as the probe lines are listed in
    the ``{drive_controller}.awg.feedback_sources`` station setting.
    """
    _needs_recalibration = False

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

        Returns:
            A TimeBox for the signal delay, and a TimeBox with a ConditionalInstruction inside.

        """
        qubit = self.locus[0]
        drive_channel = self.builder.get_drive_channel(qubit)
        prx_gate = self.build("prx", self.locus)

        # NOTE assumes that the PRX gate only has IQPulse instructions on drive_channel
        timebox: TimeBox = prx_gate(angle, phase)  # type: ignore[assignment]
        if not timebox.atom:
            raise RuntimeError("Received non-atomic PRX timebox.")
        prx_instructions = timebox.atom[drive_channel]
        iq_pulses = [inst for inst in prx_instructions if isinstance(inst, IQPulse)]
        if len(iq_pulses) != len(prx_instructions):
            raise RuntimeError(f"PRX drive channel has non-IQPulse instructions: {prx_instructions}")

        iq_pulse = fuse_iq_pulses(iq_pulses) if len(iq_pulses) > 1 else iq_pulses[0]

        wait = Wait(iq_pulse.duration)  # idling, can be replaced with a DD sequence later on

        feedback_label = f"{feedback_qubit}__{feedback_key}"
        conditional_instruction = ConditionalInstruction(
            duration=iq_pulse.duration,
            condition=feedback_label,
            outcomes=(wait, iq_pulse),
        )
        delays = self.calibration_data["control_delays"]
        if (len_delays := len(delays)) == 0:
            raise ValueError(f"'control_delays' for '{self.name}' on {qubit} is empty (not calibrated).")

        possible_sources = self.builder.feedback_sources[self.locus[0]]
        if len_delays != len(possible_sources) and len_delays != 1:
            raise ValueError(
                f"Not the correct amount of calibration values for 'control_delays'. Need {len(possible_sources)}"
                f"values, got {delays}."
            )
        virtual_channel_name = _FAST_FEEDBACK_CHANNEL.format(
            source_component=feedback_qubit, feedback_key=feedback_key, target_component=qubit
        )
        probe_line = self.builder.chip_topology.component_to_probe_line[feedback_qubit]
        if probe_line not in possible_sources:
            raise ValueError(
                f"{qubit} does not support fast feedback from {feedback_qubit} (probe line: {probe_line}). "
                f"Valid source probe lines are {possible_sources}."
            )
        delay = delays[possible_sources.index(probe_line)] if len_delays > 1 else delays[0]
        virtual_channel = self.builder.get_virtual_feedback_channel_for(feedback_qubit, self.locus[0])
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
            Schedule({virtual_channel_name: [Block(0)], drive_channel: [conditional_instruction]}),
            locus_components=[qubit],
            label=f"Conditional PRX for {qubit}",
        )
        cond.neighborhood_components = {0: {virtual_channel_name, qubit}}
        return [
            delay_box,
            cond,
        ]

    def __call__(self, *args, **kwargs) -> list[TimeBox]:  # for type narrowing
        return super().__call__(*args, **kwargs)


class CCPRX_Composite_DRAGCosineRiseFall(CCPRX_Composite):
    """Conditional drag_crf pulse."""

    default_implementations = {"prx": "drag_crf"}


class CCPRX_Composite_DRAGGaussian(CCPRX_Composite):
    """Conditional drag_gaussian pulse."""

    default_implementations = {"prx": "drag_gaussian"}
