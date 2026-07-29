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
"""Barrier metaoperation.

The barrier is an n-qubit metaoperation that forces a specific temporal ordering on the quantum
operations on different sides of it (the ones preceding the barrier are always executed first).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from iqm.pulse.gate_implementation import GateImplementation, Locus, OILCalibrationData
from iqm.pulse.playlist.instructions import Block
from iqm.pulse.playlist.schedule import Schedule

if TYPE_CHECKING:  # pragma: no cover
    from iqm.pulse.builder import ScheduleBuilder
    from iqm.pulse.quantum_ops import QuantumOp
    from iqm.pulse.timebox import TimeBox


class Barrier(GateImplementation):
    """GateImplementation for the n-qudit ``barrier`` metaoperation.

    Returns a schedule with zero-duration :class:`.Block` metainstructions.
    When this is appended to another :class:`.Schedule`,
    it causes the affected channels to be padded with :class:`.Wait` instructions to the same length,
    which in turn imposes a definite temporal order for the operations on different sides of
    the barrier (the ones preceding it are always executed first).

    .. note::
       Assumes that all instructions involve either the drive, flux or probe channels of the locus QPU components.

    Args:
        channels: channels related to the locus QPU components, to be blocked

    """

    symmetric: bool = True

    def __init__(
        self,
        parent: QuantumOp,
        name: str,
        locus: Locus,
        calibration_data: OILCalibrationData,
        builder: ScheduleBuilder,
    ):
        super().__init__(parent, name, locus, calibration_data, builder)
        channels = builder.get_control_channels(locus)
        self.schedule = Schedule({c: [Block(0)] for c in channels})

    def _call(self) -> TimeBox:
        return self.to_timebox(self.schedule)  # TODO make a copy of schedule??

    def duration_in_seconds(self) -> float:
        return 0.0
