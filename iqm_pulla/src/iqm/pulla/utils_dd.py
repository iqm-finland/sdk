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
"""Utilities for working with Dynamical Decoupling."""

from math import pi


def generate_phases_of_urn_sequence(n: int) -> list[float]:
    """Generate PRX gate phases for the URn sequence.

    More information on the URn sequence is available in :cite:`Ezzell_2022`, at the end in Appendix A.1.

    Args:
        n: Number of single qubit PRX pulses with different phases to apply, must be a positive even number.

    Returns:
        Phases to be used by the URn sequence.

    """
    if (n <= 0) or (n % 2 != 0):
        raise ValueError("n should be a positive even number.")

    phases = []

    for index_pulse in range(n):
        if n % 4 == 0:
            phi = pi / (n // 4)
            angle = index_pulse * (index_pulse - 1) * phi / 2 + index_pulse * pi / 2
            phases.append(angle)
        elif n % 2 == 0:
            phi = 2 * (n // 4) * pi / (2 * (n // 4) + 1)
            angle = index_pulse * (index_pulse - 1) * phi / 2 + index_pulse * pi / 2
            phases.append(angle)

    phases = [p % (2 * pi) for p in phases]  # Normalize phase angles to interval [0, 2 * pi)
    return phases
