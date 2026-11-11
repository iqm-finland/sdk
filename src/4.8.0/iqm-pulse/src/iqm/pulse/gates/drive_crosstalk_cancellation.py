# Copyright 2024-2026 IQM
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
"""GateImplementation that holds the drive crosstalk calibration matrix and builds compensation pulses."""

from __future__ import annotations

import logging

from exa.common.data import DataType, Parameter, Setting
from iqm.pulse.gate_implementation import GateImplementation
from iqm.pulse.locus_mappings import drive_crosstalk_pairs
from iqm.pulse.playlist import IQPulse, Schedule
from iqm.pulse.timebox import TimeBox

logger = logging.getLogger(__name__)


class ScaledWaveformCancellation(GateImplementation):
    r"""Calibration container and pulse builder for drive crosstalk compensation.

    The drive crosstalk is described by a complex matrix, where the element :math:`C_{tc}` describes the crosstalk
    drive seen by the **target** ``t`` when the **control** ``c`` is driven: its absolute value is the relative
    crosstalk amplitude, and its argument the crosstalk phase. To cancel it, ``cancel_drive_crosstalk`` replays the
    control's drive on an auxiliary drive channel of the target, scaled by the amplitude and phase-shifted accordingly.

    Parameters ``crosstalk_amplitude`` and ``crosstalk_phase`` contain the per-locus calibration data for the drive
    crosstalk matrix, i.e. the amplitude :math:`|C_{tc}|` and the phase :math:`\arg C_{tc}` (in radians).

    TODO: this is for now an experimental R&D implementation, and everything here is subject to change still.
    """

    parameters = {
        "crosstalk_amplitude": Setting(
            Parameter(
                "crosstalk_amplitude",
                label="Drive crosstalk relative amplitudes",
                data_type=DataType.FLOAT,
            ),
            0.0,
        ),
        "crosstalk_phase": Setting(
            Parameter(
                "crosstalk_phase",
                label="Drive crosstalk phases",
                unit="rad",
                data_type=DataType.FLOAT,
            ),
            0.0,
        ),
    }
    locus_mapping_function = drive_crosstalk_pairs

    def __call__(
        self,
        schedule: Schedule,
    ) -> TimeBox:
        """Drive crosstalk compensation pulses for a resolved schedule.

        For ``self.locus``, (i.e. the pair ``(target, control)``), the ``control`` drive channel content of
        ``schedule`` is replayed on the target's auxiliary drive channel ``<target_drive>.aux0``, with every
        :class:`.IQPulse` IQ-scaled by the calibrated crosstalk amplitude. The whole replay is phase-shifted by adding
        the calibrated crosstalk phase to the ``phase_increment`` of the first replayed :class:`.IQPulse`, which
        shifts the carrier for that pulse and all that follow it on the channel. At most one auxiliary channel is
        produced per target. Pairs whose calibration is missing, whose control is not driven in ``schedule``, or
        whose target already has compensation are skipped.

        Args:
            schedule: Resolved schedule whose control drive channels are replayed onto target aux channels.

        Returns:
            Atomic TimeBox carrying the compensation aux channel corresponding to the correction between
            ``self.locus``'s target and control, or an empty composite TimeBox if the compensation is skipped.

        """
        target, control = self.locus
        channels = schedule.channels()
        aux_channel = f"{self.builder.get_drive_channel(target)}.aux0"
        if aux_channel in channels:
            logger.warning(
                "Drive crosstalk compensation for target %s already exists, skipping control %s",
                target,
                control,
            )
            return TimeBox.composite([])
        control_channel = self.builder.get_drive_channel(control)
        if control_channel not in channels:
            # control is not driven in this schedule, so there is nothing to compensate for.
            # just return empty box silently
            return TimeBox.composite([])
        amplitude = self.calibration_data["crosstalk_amplitude"]
        phase = self.calibration_data["crosstalk_phase"]
        replayed: list = []
        phase_applied = False
        for inst in schedule[control_channel]:
            if not isinstance(inst, IQPulse):
                replayed.append(inst)
                continue
            scaling = {"scale_i": inst.scale_i * amplitude, "scale_q": inst.scale_q * amplitude}
            if not phase_applied:
                # Apply the calibrated crosstalk phase to the first replayed IQPulse via its phase_increment
                # (which shifts the carrier for this pulse and all that follow it on the channel) instead of
                # prepending a separate VirtualRZ.
                scaling["phase_increment"] = inst.phase_increment + phase
                phase_applied = True
            replayed.append(inst.copy(**scaling))
        if not phase_applied:
            # control was not actually driven (no IQPulse), so there is nothing to compensate
            return TimeBox.composite([])
        return TimeBox.atomic(
            Schedule({aux_channel: replayed}),
            locus_components=(target,),
            label=f"Drive cross-talk compensation on {aux_channel}",
        )
