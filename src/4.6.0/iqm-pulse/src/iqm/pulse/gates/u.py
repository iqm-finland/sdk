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
   SH = C_XYZ = U(\pi / 2, 0, \pi /2)\\
   HS^\dagger = C_{ZYX} = U(\pi / 2, \pi / 2, \pi)\\

References
----------
https://openqasm.com/language/gates.html#built-in-gates

"""

from __future__ import annotations

from functools import lru_cache
from types import MethodType
from typing import TYPE_CHECKING

import numpy as np

from iqm.pulse.gate_implementation import CompositeGate
from iqm.pulse.playlist import VirtualRZ
from iqm.pulse.playlist.schedule import Schedule
from iqm.pulse.utils import normalize_angle, phase_transformation

if TYPE_CHECKING:  # pragma: no cover
    from iqm.pulse.builder import ScheduleBuilder
    from iqm.pulse.gate_implementation import Locus, OILCalibrationData
    from iqm.pulse.quantum_ops import QuantumOp
    from iqm.pulse.timebox import TimeBox

PI_2 = np.pi / 2
"""Fundamental mathematical constant pi divided by two for performance."""

CLIFFORDS_TO_EULER_DECOMPOSITION: dict[str, tuple[float, float, float]] = {
    "x": (np.pi, -PI_2, PI_2),
    "y": (np.pi, 0.0, 0.0),
    "z": (0.0, 0.0, np.pi),
    "c_nxyz": (PI_2, np.pi, PI_2),
    "c_nzyx": (PI_2, -PI_2, 0.0),
    "c_xnyz": (PI_2, 0.0, -PI_2),
    "c_xynz": (PI_2, np.pi, -PI_2),
    "c_xyz": (PI_2, 0.0, PI_2),
    "c_znyx": (PI_2, -PI_2, np.pi),
    "c_zynx": (PI_2, PI_2, 0.0),
    "c_zyx": (PI_2, PI_2, np.pi),
    "h": (PI_2, 0.0, np.pi),
    "h_xz": (PI_2, 0.0, np.pi),
    "h_nxy": (np.pi, 0.0, -PI_2),
    "h_nxz": (PI_2, np.pi, 0.0),
    "h_nyz": (PI_2, -PI_2, -PI_2),
    "h_xy": (np.pi, 0.0, PI_2),
    "h_yz": (PI_2, PI_2, PI_2),
    "s": (0.0, 0.0, PI_2),
    "sz": (0.0, 0.0, PI_2),
    "sqrt_z": (0.0, 0.0, PI_2),
    "s_dag": (0.0, 0.0, -PI_2),
    "szd": (0.0, 0.0, -PI_2),
    "sqrt_z_dag": (0.0, 0.0, -PI_2),
    "sx": (PI_2, -PI_2, PI_2),
    "sqrt_x": (PI_2, -PI_2, PI_2),
    "sxd": (PI_2, PI_2, -PI_2),
    "sqrt_x_dag": (PI_2, PI_2, -PI_2),
    "sy": (PI_2, 0.0, 0.0),
    "sqrt_y": (PI_2, 0.0, 0.0),
    "syd": (PI_2, np.pi, np.pi),
    "sqrt_y_dag": (PI_2, np.pi, np.pi),
}
r"""Mapping of names used to describe Clifford gates to the tuples :math:`\theta, \phi, \lambda` representing their
ZYZ Euler angle decompositions. The names follow both Qiskit and Stim conventions, so some angle tuples repeat."""

ANGLE_THRESHOLD: float = 1e-13
"""If the angle is below this threshold, it is understood to be zero."""


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


def get_su2_params(unitary: np.ndarray, threshold: float = 1e-13) -> tuple[float, float, float]:
    """Calculate parameters for an SU(2) gate given its unitary.

    The parameters are calculated up to global phase, and such that ``0 <= theta <= pi`` and ``-pi <= phi, lam < pi``.
    See :mod:`iqm.pulse.gates.u` for the definition of the gate parameters.

    Args:
        unitary: U(2) matrix to calculate parameters for, ignoring the global phase.
        threshold: Precision threshold to decide if the unitary describes a pure Z rotation, a pi rotation, or neither.

    Returns:
        Parameters theta, phi, lambda for the ``u`` gate.

    """
    global_phase = np.angle(np.linalg.det(unitary)) / 2
    special_unitary = np.exp(-1j * global_phase) * unitary

    # pi XY rotation
    if abs(unitary[0, 0]) < threshold:
        theta = np.pi
        phi = np.angle(special_unitary[1, 0])
        lam = -phi

    # Z rotation
    elif abs(unitary[0, 1]) < threshold:
        theta = 0.0
        lam = 0.0
        phi = np.angle(special_unitary[1, 1] ** 2)
        if phi < threshold:
            phi = 0.0

    # full unitary
    else:
        theta = 2 * np.arccos(abs(unitary[0, 0]))
        phi = np.angle(special_unitary[1, 1] * special_unitary[1, 0])
        lam = np.angle(special_unitary[1, 1] / special_unitary[1, 0])

    return theta, phi, lam


def get_haar_random_unitary(d: int = 2, rng: np.random.Generator | None = None) -> np.ndarray:
    """Generates a random unitary matrix drawn uniformly from Haar measure.

    Follows the procedure outlined in :cite:`Mezzadri_2007`:

    1. Generate a n x n matrix of complex numbers such that both real and imaginary parts of each number are drawn
        independently from a normal distribution with mean 0 and variance 1.
    2. Perform a QR decomposition of this matrix, obtaining a unitary and upper triangular matrices.
    3. Normalize the unitary, multiplying each column by the sign of a corresponding diagonal entry in the upper
        triangular matrix, which are guaranteed to be real.

    The last step is necessary, because otherwise the non-uniqueness of the QR decomposition introduces bias to the
    sampling process.

    Args:
        d: Dimension of the generated unitary.
        rng: Random number generator object.

    Returns:
        A random unitary matrix of requested dimension.

    """
    rng = rng or np.random.default_rng()
    base_matrix = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    unitary, upper = np.linalg.qr(base_matrix, mode="complete")
    diagonal = np.diag(upper)
    normalizing = np.diag(diagonal / np.abs(diagonal))
    unitary = unitary @ normalizing

    return unitary


class UGate(CompositeGate):
    r"""SU(2) gate implemented using PRX.

    Assumes the chosen PRX implementation uses resonant driving, and that the virtual RZ technique can be used.
    """

    registered_gates = ("prx",)

    def __init__(
        self,
        parent: QuantumOp,
        name: str,
        locus: Locus,
        calibration_data: OILCalibrationData,
        builder: ScheduleBuilder,
    ) -> None:
        super().__init__(parent, name, locus, calibration_data, builder)
        for clifford_name, clifford_angles in CLIFFORDS_TO_EULER_DECOMPOSITION.items():
            self._set_clifford_shortcut(clifford_name, clifford_angles)

    def _call(self, theta: float, phi: float = 0.0, lam: float = 0.0) -> TimeBox:  # type: ignore[override]
        r"""Convert pulses into timebox, via Euler decomposition.

        .. math::
            U(\theta, \phi, \lambda) = R_Z(\phi) \cdot R_Y(\theta) \cdot R_Z(\lam)
        """
        # TODO we directly modify the PRX timebox contents here which makes a lot of assumptions about
        # the PRX implementation. This isn't safe in general, can we find a better solution?
        prx_gate = self.build("prx", self.locus)

        # If the unitary is a pure Z rotation, output a virtual Z. This implementation is using virtual z-s
        # anyway for z rotations before and after, so now the pulse is also not performed.
        if abs(normalize_angle(theta)) < ANGLE_THRESHOLD:
            angle = phi + lam
            timebox = self.to_timebox(
                Schedule(
                    {
                        prx_gate.channel: [  # type: ignore[attr-defined]
                            VirtualRZ(
                                duration=self.builder.min_allowed_instruction_duration,
                                phase_increment=-normalize_angle(angle),
                                standalone=True,
                            )
                        ]
                    }
                )
            )
            timebox.neighborhood_components[0] = set(self.locus)
            return timebox

        pulse_train = prx_gate(theta, np.pi / 2).atom[  # type: ignore[union-attr]
            prx_gate.channel  # type: ignore[index, attr-defined]
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

    def rx(self, angle: float) -> TimeBox:
        """X rotation gate.

        Args:
            angle: rotation angle (in radians)

        Returns:
            boxed instruction schedule implementing the x rotation gate

        """
        box = self(theta=angle, phi=-np.pi / 2, lam=np.pi / 2)
        box.label = f"Rx on {self.locus[0]}"
        return box

    def ry(self, angle: float) -> TimeBox:
        """Y rotation gate.

        Args:
            angle: rotation angle (in radians)

        Returns:
            boxed instruction schedule implementing the y rotation gate

        """
        box = self(theta=angle, phi=0.0, lam=0.0)
        box.label = f"Ry on {self.locus[0]}"
        return box

    def from_unitary(self, unitary: np.ndarray) -> TimeBox:
        """Make a box directly from a single qubit unitary matrix.

        Args:
            unitary: 2x2 unitary matrix

        Returns:
            boxed instruction schedule implementing the unitary as a gate

        """
        if not np.allclose(unitary @ unitary.T.conj(), np.eye(2), atol=ANGLE_THRESHOLD):
            raise ValueError(f"Matrix {unitary} is malformed or not a unitary.")

        angles = get_su2_params(unitary)
        box = self(*angles)
        return box

    def _set_clifford_shortcut(self, name: str, angles: tuple[float, float, float]) -> None:
        """Add the convenience methods for Clifford gates."""

        def _clifford(
            self: UGate,
        ) -> TimeBox:
            box = self(*angles)
            box.label = f"{name.capitalize()} on {self.locus[0]}"
            return box

        _clifford.__doc__ = (
            f"{name.upper()} gate.\n\nReturns:\n\n    boxed instruction schedule implementing the {name.upper()} gate"
        )
        setattr(self, name.lower(), MethodType(_clifford, self))
