# Copyright 2022-2026 IQM
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

# This code is a Qiskit project.

# (C) Copyright IBM 2023.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# NOTE: The code has been altered from the original (https://github.com/Qiskit/qiskit-addon-cutting/blob/stable/0.10/qiskit_addon_cutting/utils/simulation.py)

"""Test utilities for circuit randomization tests.

This module provides helper functions for simulating quantum circuits and
computing distances between probability distributions
"""

from collections import defaultdict

from iqm.error_reduction_tools.utils.general_utils import (
    total_variational_distance,  # noqa: F401
)
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

_TOLERANCE = 1e-16


def simulate_statevector_outcomes(  # noqa: PLR0912
    qc: QuantumCircuit, bin: bool = True, little_endian: bool = True
) -> dict[str | int, float]:
    """Return each classical outcome along with its precise probability.

    Circuit can contain mid-circuit, projective measurements.

    All gates are supported, along with measurements and reset operations.

    Args:
        qc: The quantum circuit to simulate.
        bin: If True, return binary string keys. If False, return integer keys.
        little_endian: If True, use little-endian bit order for binary strings.

    Returns:
        Dictionary mapping outcomes to probabilities.
    """
    current = defaultdict(list)
    current[0].append((1.0, Statevector.from_int(0, 2**qc.num_qubits)))
    for inst in qc.data:
        opname = inst.operation.name
        if opname in ("measure", "reset"):
            # The current instruction is not unitary: it's either a measurement
            # or a reset.
            qubit_idx = qc.find_bit(inst.qubits[0])[0]
            if opname == "measure":
                # We will need to set a classical bit depending on the
                # measurement result.  `k_flipper` locates that bit.
                k_flipper = 1 << qc.find_bit(inst.clbits[0])[0]
            else:
                # It's a reset operation, so we will not be modifying any
                # classical bits.
                k_flipper = 0
            # We need to keep track of the statevector and corresponding
            # probability of *both* possible outcomes (although, we truncate
            # states if their probability becomes less than _TOLERANCE).  In
            # the following, we loop through each outcome so far and prepare to
            # update the state.
            pending_delete: list[tuple[int, int]] = []
            pending_insert: list[tuple[int, tuple[float, Statevector]]] = []
            for k, svs in current.items():
                k0 = k ^ (k & k_flipper)  # like k, but k_flipper bit will NOT be set
                k1 = k | k_flipper  # like k, but k_flipper bit (if any) will be set
                for i, (prob, sv) in enumerate(svs):
                    prob0, prob1 = sv.probabilities([qubit_idx])
                    dims = sv.dims([qubit_idx])  # always going to be (2,) for a qubit
                    pending_delete.append((k, i))
                    # Handle the 0 branch of the wave function
                    if not np.isclose(prob0, 0, atol=_TOLERANCE):
                        proj0 = np.diag([1 / np.sqrt(prob0), 0.0])
                        sv0 = sv.evolve(
                            Operator(proj0, input_dims=dims, output_dims=dims),
                            qargs=[qubit_idx],
                        )
                        pending_insert.append((k0, (prob * prob0, sv0)))
                    # Handle the 1 branch of the wave function
                    if not np.isclose(prob1, 0, atol=_TOLERANCE):
                        proj1 = np.diag([0.0, 1 / np.sqrt(prob1)])
                        if k_flipper == 0:
                            # It's a reset operation, so we need to rotate the 1
                            # result back to 0 by applying the same rotation as
                            # the X gate.
                            proj1 = np.array([(0, 1), (1, 0)]) @ proj1
                        sv1 = sv.evolve(
                            Operator(proj1, input_dims=dims, output_dims=dims),
                            qargs=[qubit_idx],
                        )
                        pending_insert.append((k1, (prob * prob1, sv1)))
            # A dict's keys cannot be changed while iterating it, so we perform
            # all such updates now that iteration over the dict is complete.
            for k, i in reversed(pending_delete):
                del current[k][i]
            for k, v in pending_insert:
                current[k].append(v)
            # We might as well clean up empty lists, too.
            for k in [k for k, v in current.items() if not v]:
                del current[k]
        else:
            # The current instruction is a unitary operation (i.e., a gate).
            if len(inst.clbits) != 0:  # pragma: no cover
                raise ValueError("Circuit cannot contain a non-measurement operation on classical bit(s).")
            # Evolve each statevector according to the current instruction
            for svs in current.values():
                for _, sv in svs:
                    # Calling `_evolve_instruction` rather than `evolve` allows
                    # us to avoid a copy.
                    Statevector._evolve_instruction(sv, inst.operation, [qc.find_bit(q)[0] for q in inst.qubits])

    counts = {outcome: sum(prob for prob, _ in svs) for outcome, svs in current.items()}

    if bin:
        bin_counts = to_bin_counts(counts)
        return {k if not little_endian else k[::-1]: v for k, v in bin_counts.items()}

    else:
        return counts


def to_bin_counts(counts: dict[int, float]) -> dict[str, float]:
    """Convert integer-based counts to binary string-based counts.

    Args:
        counts: Dictionary with integer keys representing measurement outcomes.

    Returns:
        Dictionary with binary string keys (little-endian bit order).
    """
    max_key = max(counts.keys())
    num_qubits = max_key.bit_length()
    bin_counts = {}
    for key, value in counts.items():
        bin_key = format(key, f"0{num_qubits}b")[::-1]  # Qiskit uses little-endian bit order
        bin_counts[bin_key] = value
    return bin_counts
