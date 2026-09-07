# Copyright (c) 2024-2026 IQM Quantum Computers
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted (subject to the
# limitations in the disclaimer below) provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this list of conditions and the following
#   disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
#   disclaimer in the documentation and/or other materials provided with the distribution.
# * Neither the name of IQM Quantum Computers nor the names of its contributors may be used to endorse or promote
#   products derived from this software without specific prior written permission.
#
# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY
# THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""A module for the HUBOQAOA class."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from dimod.higherorder.polynomial import BinaryPolynomial
from iqm.applications.hubo import HUBOInstance
from iqm.qaoa.generic_qaoa import QAOA
import numpy as np

if TYPE_CHECKING:
    from iqm.qaoa.backends import EstimatorStateVector


class HUBOQAOA(QAOA[HUBOInstance]):
    r"""The class for QAOA with higher-order unconstrained binary (HUBO) cost function.

    The class inherits a lot of functionality from its parent :class:`iqm.qaoa.generic_qaoa.QAOA`. One new addition is
    the attribute :attr:`hamiltonian_bp` which stores the coefficient of the problem Hamiltonian.

    Args:
        problem: A :class:`~iqm.applications.hubo.HUBOInstance` object describing the HUBO problem to be solved.
        num_layers: The number of QAOA layers, commonly referred to as *p* in the literature.
        betas: An optional list of the initial *beta* angles of QAOA. Has to be provided together with ``gammas``.
        gammas: An optional list of the initial *gamma* angles of QAOA. Has to be provided together with ``betas``.
        initial_angles: An optional list of the initial QAOA angles as one variable. Shouldn't be provided together
            with either ``betas`` or ``gammas``. The *gamma* and *beta* angles are interleaved, so that the first pair
            of entries corresponds to :math:`\gamma_1` and :math:`\beta_1` (the angles of the first QAOA layer).
            The second pair of entries corresponds to :math:`\gamma_2` and :math:`\beta_2`, etc. ...

    """

    def __init__(
        self,
        problem: HUBOInstance,
        num_layers: int,
        *,
        betas: Sequence[float] | np.ndarray | None = None,
        gammas: Sequence[float] | np.ndarray | None = None,
        initial_angles: Sequence[float] | np.ndarray | None = None,
    ) -> None:
        self._problem: HUBOInstance
        super().__init__(problem, num_layers, betas=betas, gammas=gammas, initial_angles=initial_angles)
        # This BP contains the coefficients of the Hamiltonian (in front of the Z, ZZ, ZZZ, ... terms).
        self._hamiltonian_bp = self._problem._spin_bp.copy()
        # To reconcile the difference in convention between ``dimod`` and ``qiskit``, the odd terms' signs are flipped.
        # This way (``qiskit`` convention), the variable that was 0 in the binary formulation corresponds to the state
        # |0⟩, i.e., spin 1. Correspondingly, the value 1 in binary corresponds to the state |1⟩, i.e., spin -1.
        for term, bias in self._hamiltonian_bp.items():
            if len(term) % 2 == 1:
                self._hamiltonian_bp[term] = -bias

    @property
    def hamiltonian_bp(self) -> BinaryPolynomial:
        """The BP representation of the problem, taken from the input :class:`~iqm.applications.hubo.HUBOInstance`."""
        return self._hamiltonian_bp

    def _get_default_estimator(self) -> EstimatorStateVector:
        # Lazy import to prevent cyclical imports.
        from iqm.qaoa.backends import EstimatorStateVector  # noqa: PLC0415

        return EstimatorStateVector()
