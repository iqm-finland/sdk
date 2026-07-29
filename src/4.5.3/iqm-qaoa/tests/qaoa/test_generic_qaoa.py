# Copyright (c) 2024-2025 IQM Quantum Computers
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
"""The testing module for the iqm/qaoa/generic_qaoa.py file."""

from iqm.applications.mis import MISInstance
from iqm.qaoa.qubo_qaoa import QUBOQAOA
import numpy as np


def test_linear_ramp_schedule(sparse_mis_instance: MISInstance) -> None:
    """Tests some basic properties of :meth:`~iqm.qaoa.generic_qaoa.QAOA.linear_ramp_schedule`.

    This method sets the angles to be linearly spaced from 0 (excluded) to the maximum values given on input.
    """
    p = 5
    max_gamma = 3
    max_beta = 5

    # We create an instance of QUBOQAOA, but we use it to test a method from QAOA.
    my_qaoa = QUBOQAOA(sparse_mis_instance, num_layers=p)
    my_qaoa.linear_ramp_schedule(max_beta, max_gamma)

    gammas = my_qaoa.gammas
    betas = my_qaoa.betas

    # Check length.
    assert gammas.shape == (p,)
    assert betas.shape == (p,)

    # Check equal spacing (i.e., "linear ramp schedule").
    diffs_g = np.diff(gammas)
    assert np.allclose(diffs_g, max_gamma / p)
    diffs_b = np.diff(betas)
    assert np.allclose(diffs_b, -max_beta / p)  # The beta angles are decreasing.

    # Check that the maximum values agree.
    assert betas[0] == max_beta  # The beta angles are decreasing.
    assert gammas[-1] == max_gamma
