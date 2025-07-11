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
r"""Single-qubit sqrt(X) gate.

The gate is doing pi/2 X gate, with additional Z rotation to correct phase.

.. math::
   R = e^{-i\pi \^{\sigma}_X/4}

It rotates the qubit state in XZ plane (or around Y axis) for 90 degree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from iqm.pulse.gate_implementation import CompositeGate
from iqm.pulse.gates.prx import PRX_CustomWaveformsSX, PRX_SinglePulse_GateImplementation
from iqm.pulse.playlist.schedule import Schedule

if TYPE_CHECKING:  # pragma: no cover
    from iqm.pulse.timebox import TimeBox


class SXGate(CompositeGate):
    """SX gate implementation based on PRX gate, by limiting the angle to pi / 2."""

    registered_gates = ["prx"]

    def _call(self) -> TimeBox:  # type: ignore[override]
        """Call PRX gate with angle equals to pi / 2."""
        prx_gate: PRX_SinglePulse_GateImplementation | PRX_CustomWaveformsSX = self.build("prx", self.locus)  # type: ignore[assignment]
        pulse = prx_gate(np.pi / 2, 0.0).atom[prx_gate.channel][0]  # type: ignore[union-attr,index]
        return self.to_timebox(Schedule({prx_gate.channel: [pulse]}))
