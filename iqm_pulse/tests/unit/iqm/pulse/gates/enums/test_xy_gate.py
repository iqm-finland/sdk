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
import numpy as np
import pytest

from iqm.pulse.gates.enums import XYGATE_UNITARIES, XYGate


def test_get_unitaries():
    unitaries = XYGATE_UNITARIES

    assert len(unitaries) == 7

    for key, U in unitaries.items():
        assert isinstance(key, XYGate)
        assert isinstance(U, np.ndarray)
        # U must be an SU(2) matrix
        assert U.shape == (2, 2)
        assert U.dtype == complex
        W = U.T.conj()
        assert U @ W == pytest.approx(np.eye(2))
        assert W @ U == pytest.approx(np.eye(2))
        # det(U) == 1
        assert U[0, 0] * U[1, 1] - U[0, 1] * U[1, 0] == pytest.approx(1)
