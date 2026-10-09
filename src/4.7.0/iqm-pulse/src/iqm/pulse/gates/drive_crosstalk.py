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

from collections.abc import Iterable
import logging

from exa.common.data.parameter import CollectionType, DataType, Parameter
from iqm.pulse.gate_implementation import GateImplementation
from iqm.pulse.locus_mappings import LocusMapping, StationProperties
from iqm.pulse.playlist.instructions import IQPulse
from iqm.pulse.playlist.schedule import Schedule
from iqm.pulse.timebox import TimeBox

logger = logging.getLogger(__name__)


class DriveCrossTalk(GateImplementation):
    r"""Calibration container and pulse builder for drive crosstalk compensation.

    This implementation carries the drive crosstalk calibration data so that it can be calibrated and fetched like
    any other gate, and it builds the compensation pulses for a resolved schedule. The ``apply_drive_crosstalk``
    compiler pass fetches this implementation via the builder and calls it with a schedule, e.g.::

        impl = builder.get_implementation("drive_crosstalk", (), impl_name="drive_crosstalk")
        compensation_box = impl(schedule, crosstalk_loci)  # TimeBox with the target aux-channel compensation pulses

    The drive crosstalk is described by a complex matrix, where the element :math:`C_{tc}` describes the crosstalk
    drive seen by the **target** ``t`` when the **control** ``c`` is driven: its absolute value is the relative
    crosstalk amplitude, and its argument the crosstalk phase. To cancel it, ``apply_drive_crosstalk`` replays the
    control's drive on an auxiliary drive channel of the target, scaled by the amplitude and phase-shifted accordingly.

    The matrix is given in a sparse form via three calibration parameters (we do not support dict- or xarray-valued
    Parameters yet). Parameter ``matrix_index`` lists the relevant (non-zero) elements as a flat list of strings of the
    form ``<target>__<control>``. Parameters ``crosstalk_amplitudes`` and ``crosstalk_phases`` list the corresponding
    amplitudes :math:`|C_{tc}|` and phases :math:`\arg C_{tc}` (in radians). The three arrays must have equal lengths.

    TODO: this is for now an experimental R&D implementation, and everything here is subject to change still.
    """

    parameters = {
        "matrix_index": Parameter(
            "matrix_index",
            label="Drive crosstalk sparse matrix index",
            data_type=DataType.STRING,
            collection_type=CollectionType.LIST,
        ),
        "crosstalk_amplitudes": Parameter(
            "crosstalk_amplitudes",
            label="Drive crosstalk relative amplitudes",
            data_type=DataType.FLOAT,
            collection_type=CollectionType.NDARRAY,
        ),
        "crosstalk_phases": Parameter(
            "crosstalk_phases",
            label="Drive crosstalk phases",
            unit="rad",
            data_type=DataType.FLOAT,
            collection_type=CollectionType.NDARRAY,
        ),
    }

    # HACK: Locus is "global" (the whole QPU) represented by an empty tuple for now.
    @staticmethod
    def locus_mapping_function(station_properties: StationProperties) -> LocusMapping:
        return {(): ()}

    def _calibration(self) -> dict[tuple[str, str], dict[str, float]]:
        """Drive crosstalk calibration parsed into a per-pair mapping.

        Returns:
            Mapping from ``(target, control)`` component pairs to their calibration data, i.e. a mapping with the keys
            ``"crosstalk_amplitude"`` and ``"crosstalk_phase"`` (in radians).

        """
        matrix_index = self.calibration_data["matrix_index"]
        amplitudes = self.calibration_data["crosstalk_amplitudes"]
        phases = self.calibration_data["crosstalk_phases"]
        calibration: dict[tuple[str, str], dict[str, float]] = {}
        for entry, amplitude, phase in zip(matrix_index, amplitudes, phases):
            target, control = entry.split("__")
            calibration[(target, control)] = {
                "crosstalk_amplitude": float(amplitude),
                "crosstalk_phase": float(phase),
            }
        return calibration

    def __call__(
        self,
        schedule: Schedule,
        crosstalk_loci: Iterable[str],
    ) -> TimeBox:
        """Drive crosstalk compensation pulses for a resolved schedule.

        For each ``(target, control)`` pair, the ``control`` drive channel content of ``schedule`` is replayed on
        the target's auxiliary drive channel ``<target_drive>.aux0``, with every :class:`.IQPulse` IQ-scaled by the
        calibrated crosstalk amplitude. The whole replay is phase-shifted by adding the calibrated crosstalk phase to
        the ``phase_increment`` of the first replayed :class:`.IQPulse`, which shifts the carrier for that pulse and
        all that follow it on the channel. At most one auxiliary channel is produced per target (the first matching
        ``(target, control)`` pair wins). Pairs whose calibration is missing, whose control is not driven in
        ``schedule``, or whose target already has compensation are skipped.

        Args:
            schedule: Resolved schedule whose control drive channels are replayed onto target aux channels.
            crosstalk_loci: ``<target>__<control>`` strings specifying the pairs to compensate. Pairs without a
                calibration entry are skipped.

        Returns:
            Atomic TimeBox carrying the compensation aux channels (an empty schedule if there is nothing to
            compensate).

        """
        calibration = self._calibration()
        channels = schedule.channels()
        compensation: dict[str, list] = {}
        targets: set[str] = set()
        for locus in crosstalk_loci:
            target, control = locus.split("__")
            if target in targets:
                logger.warning(
                    "Drive crosstalk compensation for target %s already exists, skipping control %s",
                    target,
                    control,
                )
                continue
            cal = calibration.get((target, control))
            if cal is None:
                logger.warning(
                    "Drive crosstalk calibration for target %s and control %s is missing, skipping",
                    target,
                    control,
                )
                continue
            control_channel = self.builder.get_drive_channel(control)
            if control_channel not in channels:
                logger.warning(
                    "Control %s is not driven in the schedule, skipping target %s",
                    control,
                    target,
                )
                continue
            amplitude = cal["crosstalk_amplitude"]
            phase = cal["crosstalk_phase"]
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
                continue
            aux_channel = f"{self.builder.get_drive_channel(target)}.aux0"
            compensation[aux_channel] = replayed
            targets.add(target)
        return TimeBox.atomic(
            Schedule(compensation),
            locus_components=targets,
            label=f"Drive cross-talk compensation on {sorted(targets)}",
        )
