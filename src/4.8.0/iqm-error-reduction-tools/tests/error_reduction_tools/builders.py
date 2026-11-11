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

"""Builders of small native circuits for the workflow tests."""

from collections.abc import Sequence

import numpy as np

from iqm.pulse import Circuit, CircuitOperation


def prx(qubit, angle, phase=0.0):
    return CircuitOperation(name="prx", locus=(qubit,), args={"angle": float(angle), "phase": float(phase)})


def cz(first, second):
    return CircuitOperation(name="cz", locus=(first, second), args={})


def measurements(qubits: Sequence[str], clbits: Sequence[int] | None = None):
    clbits = list(range(len(qubits))) if clbits is None else list(clbits)
    return [
        CircuitOperation(name="measure", locus=(qubit,), args={"key": f"m_{len(qubits)}_0_{clbit}"})
        for qubit, clbit in zip(qubits, clbits)
    ]


def circuit(gates, measured: Sequence[str], name="circuit", clbits=None) -> Circuit:
    return Circuit(name=name, instructions=tuple(list(gates) + measurements(measured, clbits)))


def bell_circuit(qubits=("QB1", "QB2")) -> Circuit:
    """Prepare (|00> + |11>)/sqrt(2): <ZZ> = 1, <ZI> = <IZ> = 0."""
    first, second = qubits
    gates = [prx(first, np.pi / 2, np.pi / 2), prx(second, np.pi / 2, np.pi / 2), cz(first, second)]
    gates.append(prx(second, -np.pi / 2, np.pi / 2))
    return circuit(gates, qubits, name="bell")


def rotated_circuit(theta: float, qubit="QB1") -> Circuit:
    """Rotate |0> by theta about X: <Z> = cos(theta)."""
    return circuit([prx(qubit, theta, 0.0)], [qubit], name=f"rx_{theta:.3f}")
