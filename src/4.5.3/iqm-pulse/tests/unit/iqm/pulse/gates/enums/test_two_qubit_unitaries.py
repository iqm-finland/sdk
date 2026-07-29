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

from iqm.pulse.gates.enums import TWO_QUBIT_UNITARIES, TwoQubitGate


def test_give_correct_unitary_for_cz():
    cz = TWO_QUBIT_UNITARIES[TwoQubitGate.CZ]

    assert np.array_equal(cz, np.diag([1, 1, 1, -1]))


def test_give_correct_unitary_for_iswap():
    iswap = TWO_QUBIT_UNITARIES[TwoQubitGate.ISWAP]

    assert np.array_equal(iswap, np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]]))


def test_give_correct_unitary_for_sqrt_iswap():
    sqrt_iswap = TWO_QUBIT_UNITARIES[TwoQubitGate.SQRT_ISWAP]
    iswap = TWO_QUBIT_UNITARIES[TwoQubitGate.ISWAP]
    I = np.eye(4)  # noqa: E741

    # unitary
    assert sqrt_iswap @ sqrt_iswap.T.conj() == pytest.approx(I)
    assert sqrt_iswap.T.conj() @ sqrt_iswap == pytest.approx(I)
    # square root
    assert sqrt_iswap @ sqrt_iswap == pytest.approx(iswap)
