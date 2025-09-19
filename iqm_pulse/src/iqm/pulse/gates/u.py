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
r"""Single-qubit SU(2) gate.

The SU(2) matrix in the computational basis is parametrized using Euler angles:


.. math::
   U(\theta, \phi, \lambda) =
    \begin{pmatrix}
    cos(\theta / 2) & -e^{i\lambda}\sin{\theta/2} \\
    e^{i\phi}\sin{\theta/2} & e^{i(\lambda+\phi)}\cos{\theta/2}
    \end{pmatrix}

where the angles :math:`\theta`, :math:`\phi` and :math:`\lambda` are in radians. They are the angles of subsequent
Z, Y and Z Euler rotations:

.. math::
    U(\theta, \phi, \lambda) = R_Z(\phi) \: R_Y(\theta) \: R_Z(\lambda)

It rotates the qubit state around an arbitrary axis on the Bloch sphere.

Some common single-qubit gates expressed as U gates:

.. math::
   X = U(\pi, -\pi/2, \pi/2)\\
   Y = U(\pi, 0, 0)\\
   Z = U(0, 0, \pi)\\
   H = U(\pi / 2, 0, \pi)\\
   S = U(0, \pi / 4, \pi / 4)\\
   T = U(0, \pi / 8, \pi / 8)

References
----------
https://openqasm.com/language/gates.html#built-in-gates

"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from iqm.pulse.gate_implementation import CompositeGate
from iqm.pulse.playlist.schedule import Schedule
from iqm.pulse.utils import normalize_angle, phase_transformation

if TYPE_CHECKING:  # pragma: no cover
    from iqm.pulse.timebox import TimeBox


@lru_cache
def get_unitary_u(theta: float, phi: float, lam: float) -> np.ndarray:
    """Unitary for an SU(2) gate.

    See :mod:`iqm.pulse.gates.u` for the definition of the gate parameters.

    Args:
        theta: y rotation angle
        phi: z rotation angle
        lam: another z rotation angle

    Returns:
        2x2 unitary representing ``u(theta, phi, lam)``.

    """
    return np.array(
        [
            [np.cos(theta / 2), -np.exp(1j * lam) * np.sin(theta / 2)],
            [np.exp(1j * phi) * np.sin(theta / 2), np.exp(1j * (lam + phi)) * np.cos(theta / 2)],
        ]
    )


class UGate(CompositeGate):
    r"""SU(2) gate implemented using PRX.

    Assumes the chosen PRX implementation uses resonant driving, and that the virtual RZ technique can be used.
    """

    registered_gates = ("prx",)

    def _call(self, theta: float, phi: float = 0.0, lam: float = 0.0) -> TimeBox:  # type: ignore[override]
        r"""Convert pulses into timebox, via Euler decomposition.

        .. math::
            U(\theta, \phi, \lambda) = R_Z(\phi) \cdot R_Y(\theta) \cdot R_Z(\lam)
        """
        # TODO we directly modify the PRX timebox contents here which makes a lot of assumptions about
        # the PRX implementation. This isn't safe in general, can we find a better solution?

        prx_gate = self.build("prx", self.locus)
        pulse_train = prx_gate(theta, np.pi / 2).atom[  # type: ignore[union-attr]
            prx_gate.channel  # type: ignore[index,attr-defined]
        ]  # RY pulse

        # Check if the pulse train have one or several pulses.
        if len(pulse_train) == 1:
            # Assumes the PRX consists of a single IQPulse.
            pulse = pulse_train[0]
            new_phase, new_phase_increment = phase_transformation(lam, phi)
            new_pulse = pulse.copy(
                scale_i=pulse.scale_i,
                scale_q=pulse.scale_q,
                phase=normalize_angle(pulse.phase + new_phase),
                phase_increment=normalize_angle(pulse.phase_increment + new_phase_increment),
            )
            timebox = self.to_timebox(Schedule({prx_gate.channel: [new_pulse]}))  # type: ignore[attr-defined]

        else:
            # Assumes the PRX pulse train begins and ends with IQPulses.
            # Only the first and last pulse need to be changed to implement the RZs.
            pulse_a = pulse_train[0]
            pulse_b = pulse_train[-1]
            _lam_phase, _lam_phase_increment = phase_transformation(lam, 0)
            _phi_phase, _phi_phase_increment = phase_transformation(0, phi)
            new_pulse_a = pulse_a.copy(
                scale_i=pulse_a.scale_i,
                scale_q=pulse_a.scale_q,
                phase=normalize_angle(pulse_a.phase + _lam_phase),
                phase_increment=normalize_angle(pulse_a.phase_increment + _lam_phase_increment),
            )
            new_pulse_b = pulse_b.copy(
                scale_i=pulse_b.scale_i,
                scale_q=pulse_b.scale_q,
                phase=normalize_angle(pulse_b.phase + _phi_phase),
                phase_increment=normalize_angle(pulse_b.phase_increment + _phi_phase_increment),
            )
            other_pulses = [pulse.copy() for pulse in pulse_train[1:-1]]
            new_pulses = [new_pulse_a] + other_pulses + [new_pulse_b]
            timebox = self.to_timebox(Schedule({prx_gate.channel: new_pulses}))  # type: ignore[attr-defined]

        timebox.neighborhood_components[0] = set(self.locus)
        return timebox
