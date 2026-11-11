# Copyright 2022-2026 IQM Finland Oy
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
r"""Single-qubit RZ gate.

The z rotation gate is defined as

.. math::
   R_z(\phi) = e^{-i Z \phi / 2}

where the rotation angle :math:`\phi` is in radians.
"""

from exa.common.data import Parameter
from iqm.pulse.gate_implementation import SinglePulseGate, get_waveform_parameters
from iqm.pulse.playlist.instructions import RealPulse
from iqm.pulse.playlist.waveforms import Constant, Waveform


class RZ_FluxPulse(SinglePulseGate):  # noqa: N801
    r"""Implementation of the RZ gate on qubit components using a single flux pulse.

    Args:
        flux_pulse_waveform: Flux pulse.
        channel: Name of the flux channel on which the flux is played.

    """

    flux_pulse_waveform: type[Waveform]

    def __init_subclass__(cls, /, flux_pulse_waveform: type[Waveform]):
        """Store the Waveform types used by this subclass, and their parameters."""
        cls.flux_pulse_waveform = flux_pulse_waveform
        cls.symmetric = flux_pulse_waveform is None
        parameters = get_waveform_parameters(flux_pulse_waveform)
        parameters["amplitude"] = Parameter("", "amplitude", "")
        cls.parameters = {"duration": Parameter("", "Gate duration", "s")} | parameters  # type: ignore[assignment]

    def _get_pulse_channel(self) -> str:
        return self.builder.get_flux_channel(self.locus[0])

    @classmethod
    def _get_pulse(  # type: ignore[override]
        cls,
        *,
        n_samples: int,
        amplitude: float,
    ) -> RealPulse:
        """Returns a flux pulse with given amplitude and duration."""
        return RealPulse(duration=n_samples, wave=cls.flux_pulse_waveform(n_samples=n_samples), scale=amplitude)


class RZ_ConstantFluxPulse(RZ_FluxPulse, flux_pulse_waveform=Constant):  # noqa: N801
    """RZ flux pulse gate acting on qubits, implemented with a constant (square) waveform."""
