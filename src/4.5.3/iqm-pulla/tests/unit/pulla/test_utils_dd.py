# Copyright 2024-2025 IQM
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

from math import pi

import pytest

from iqm.pulla.utils_dd import generate_phases_of_urn_sequence


@pytest.mark.parametrize("n", [-1, 0, 1, 3, 5])
def test_generate_phases_of_urn_sequences_raises_error(n):
    with pytest.raises(ValueError):
        generate_phases_of_urn_sequence(n)


@pytest.mark.parametrize(
    ("n", "seq"),
    [(2, [0, pi / 2]), (4, [0, pi / 2, 0, pi / 2]), (6, [0, pi / 2, 5 / 3 * pi, 3 * pi / 2, 0, 7 * pi / 6])],
)
def test_generate_phases_of_urn_sequences_even(n, seq):
    sequence = generate_phases_of_urn_sequence(n)
    assert len(sequence) == n
    assert sequence == pytest.approx(seq)
