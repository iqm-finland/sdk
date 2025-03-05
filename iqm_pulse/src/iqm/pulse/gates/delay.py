# Copyright 2025 IQM
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
"""Force a delay between instructions on the control channels of specific locus components.

Ideally the delay corresponds to an identity gate. In reality it of course allows decoherence
to act on the quantum state for some time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from iqm.pulse.gate_implementation import GateImplementation
from iqm.pulse.playlist.channel import round_duration_to_granularity_samples
from iqm.pulse.playlist.instructions import Wait
from iqm.pulse.playlist.schedule import Schedule

if TYPE_CHECKING:
    from iqm.pulse.timebox import TimeBox


DELAY_MAX_DURATION_SECONDS = 0.1
"""Maximum duration for individual Delay operations, in seconds."""


class Delay(GateImplementation):
    r"""Applies a delay on the control channels of its locus components.

    This operation applies :class:`.Wait` instructions on all the control channels of all its locus
    components.  The duration of all the Waits is the same, and it is given as a parameter for the
    operation, rounded up to the nearest possible duration the hardware can handle.

    .. note::

       We can only guarantee that the delay is *at least* of the requested duration.  Also, when
       Delay is used in a quantum circuit, the delay between the preceding and following operations is
       again *at least* the requested duration, but could be much more depending on the other operations
       in the circuit.  To see why, consider e.g. the circuit
       ``[CZ(a, b), Delay(1, a), Delay(10, b), CZ(a, b)]`` where a and b are qubits.
       In this case the actual delay between the two CZ gates will be 10 time units rounded up to
       hardware granularity.
    """

    symmetric: bool = True

    def _call(self, duration: float) -> TimeBox:  # type: ignore[override]
        """Delay instruction.

        Args:
            duration: Duration of the requested wait (in seconds). Will be rounded up to the nearest
                duration that the hardware enables, with the exception that a duration of zero will
                cause no waiting. However, as usual, during scheduling all channels the TimeBox is
                acting on will be extended to the duration of the longest channel in the TimeBox.

        """
        if duration > DELAY_MAX_DURATION_SECONDS:
            raise ValueError(
                f"Requested delay duration {duration} s exceeds the allowed maximum {DELAY_MAX_DURATION_SECONDS} s."
                " You can use several delay operations in a row if this is not enough."
            )
        channel_names = self.builder.get_control_channels(self.locus)
        channels = [self.builder.channels[name] for name in channel_names]

        duration_samples: int = (
            0
            if duration == 0.0
            else round_duration_to_granularity_samples(channels, duration, round_up=True, force_min_duration=True)
        )
        # NOTE we assume that all the control channels here have the same sample rate!
        wait = Wait(duration_samples)
        timebox = self.to_timebox(Schedule({ch: [wait] for ch in channel_names}))
        timebox.neighborhood_components[0] = set(self.locus)
        return timebox
