# Copyright 2026 IQM
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

"""Optimize single-qubit U3 gates by commuting phase information past SWAP gates."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from qrisp import QuantumCircuit, U3Gate
from qrisp.circuit import Qubit
from qrisp.circuit.pass_management.circuit_pass import CircuitPass


def commute_phases(preserve_unitary: bool = False) -> Callable[[QuantumCircuit], QuantumCircuit]:
    """Create a pass that optimizes U3 gates by commuting phase information past SWAP gates.

    This pass tracks and accumulates phases on each qubit, allowing single-qubit
    rotations to be optimized by combining phase information across SWAP operations.
    It modifies U3 gates to absorb accumulated phases and can optionally preserve
    the overall unitary by adding final phase gates.

    Parameters
    ----------
    preserve_unitary : bool, optional
        If True, adds final U3 gates to preserve overall unitary. Default is False.

    Returns
    -------
    Callable[[QuantumCircuit], QuantumCircuit]
        A pass function that transforms the circuit.

    Algorithm
    ---------
    - Tracks accumulated phase for each qubit
    - For U3 gates: incorporates accumulated phase into lambda parameter
    - Skips two-qubit gates if either qubit hasn't been initialized
    - Optionally adds final phase gates if preserve_unitary=True

    Example
    -------
    >>> from qrisp import QuantumCircuit, PassManager
    >>> from iqm.qrisp_iqm import commute_phases
    >>> qc = QuantumCircuit(2); qc.h(0); qc.cx(0, 1); qc.measure(qc.qubits)
    >>> pm = PassManager()
    >>> pm += commute_phases(preserve_unitary=True)
    >>> transpiled_qc = pm.run(qc)

    """

    @CircuitPass
    def _commute_phases(qc: QuantumCircuit) -> QuantumCircuit:
        qc_new = qc.clearcopy()

        last_instruction: dict[Qubit, bool | None] = {qb: None for qb in qc_new.qubits}
        phase_dict: dict[Qubit, float] = {qb: 0.0 for qb in qc_new.qubits}

        for i in range(len(qc.data)):
            instr = qc.data[i]
            op = qc.data[i].op.copy()

            if op.num_qubits == 1:
                if "alloc" in op.name:
                    continue

                if isinstance(op, U3Gate):
                    phase = phase_dict[instr.qubits[0]]
                    phase = (phase + op.lam) % (2 * np.pi)

                    new_op = U3Gate(op.theta, -phase, phase)

                    phase_dict[instr.qubits[0]] = phase + op.phi

                    op = new_op

            if op.num_qubits == 2:  # noqa: PLR2004
                skip_cz = False
                for qb in instr.qubits:
                    if last_instruction[qb] is None:
                        skip_cz = True
                        break
                if skip_cz:
                    continue

            for qb in instr.qubits:
                last_instruction[qb] = True

            qc_new.append(op, instr.qubits, instr.clbits)

        if preserve_unitary:
            for qb in qc_new.qubits:
                op = U3Gate(0, 0, phase_dict[qb])
                qc_new.append(op, [qb], [])

        return qc_new

    _commute_phases.__name__ = "commute_phases"
    return _commute_phases
