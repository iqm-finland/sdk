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
r"""Qubit leakage reduction unit.

The leakage reduction unit (LRU) is a non-unitary quantum channel that removes population
from the :math:`|2\rangle` state of a qubit.
If there is no population in the :math:`|2\rangle` state, the computational subspace is left unchanged
(LRU will act as identity on :math:`\mathrm{Span} \: \{|0\rangle, |1\rangle\}`).
However, any leaked population is moved to one of the computational states, thus changing the computational subspace.

Different LRU implementations may move the population to different states within the computational subspace.
"""

from __future__ import annotations

from exa.common.data.parameter import Parameter, Setting
from iqm.pulse.gate_implementation import (
    SINGLE_COMPONENTS_WITH_DRIVE_LOCUS_MAPPING,
    SinglePulseGate,
)
from iqm.pulse.playlist import IQPulse
from iqm.pulse.playlist.waveforms import CosineRiseFall


class LRU_F0G1(SinglePulseGate):
    r"""Leakage reduction unit implementation driving the :math:`|f0\rangle \leftrightarrow |g1\rangle` transition.

    This gate implementation removes qubit leakage (population of the :math:`|2\rangle` state)
    by driving at the frequency of the :math:`|f0\rangle \leftrightarrow |g1\rangle` transition
    which causes Rabi oscillation between those two states. The excited readout resonator quickly
    decays to its ground state, thus eventually causing the :math:`|2\rangle` state population of the qubit to
    move to the :math:`|0\rangle` state (basis here: :math:`|\text{qubit}, \text{readout resonator}\rangle`).

    See :cite:`Magnard_2018` for reference of an experiment where such a pulse was used.
    """

    parameters = {
        "duration": Parameter("", "F0G1 pulse duration", "s"),
        "amplitude": Parameter("", "F0G1 pulse amplitude", ""),
        "full_width": Parameter("", "F0G1 pulse width", "s"),
        "rise_time": Parameter("", "F0G1 pulse rise time", "s"),
        "modulation_frequency": Setting(Parameter("", "F0G1 pulse modulation frequency", "Hz"), 0.0),
        "phase_increment": Setting(Parameter("", "F0G1 pulse phase increment", ""), 0.0),
    }

    @classmethod
    def _get_pulse(  # type: ignore[override]
        cls,
        *,
        amplitude: float,
        n_samples: int,
        **rest_of_calibration_data,
    ) -> IQPulse:
        mod_freq = rest_of_calibration_data.pop("modulation_frequency") / n_samples
        phase_inc = rest_of_calibration_data.pop("phase_increment")
        wave = CosineRiseFall(n_samples=n_samples, **rest_of_calibration_data)

        instruction = IQPulse(
            n_samples,
            scale_i=amplitude,
            scale_q=0,
            wave_i=wave,
            wave_q=wave,
            modulation_frequency=mod_freq,
            phase=0,
            phase_increment=phase_inc,
        )
        return instruction

    @classmethod
    def get_locus_mapping_name(cls, operation_name: str, implementation_name: str) -> str:
        return SINGLE_COMPONENTS_WITH_DRIVE_LOCUS_MAPPING
