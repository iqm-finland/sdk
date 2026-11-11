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

"""Exact output distributions of small native circuits, and an executor that returns them as counts."""

from uuid import uuid4

from iqm.error_reduction_tools.circuits import measurement_map
from iqm.error_reduction_tools.execution import SubmittedJob
import numpy as np

_PASSIVE = {"measure", "barrier", "delay"}


def _prx(angle, phase):
    c, s = np.cos(angle / 2), np.sin(angle / 2)
    return np.array([[c, -1j * np.exp(-1j * phase) * s], [-1j * np.exp(1j * phase) * s, c]])


def num_gates(circuit):
    return sum(operation.name in {"prx", "cz"} for operation in circuit.instructions)


def exact_distribution(circuit, *, readout=0.0, depolarizing=0.0):
    """Return the exact probability of every outcome.

    ``readout`` flips each measured bit independently: one probability for every qubit, or per qubit a pair
    ``(P(0 -> 1), P(1 -> 0))``. ``depolarizing`` is a global depolarizing probability per gate: the ideal
    distribution survives with weight ``(1 - depolarizing) ** num_gates``, the rest is uniform, so every Z-type
    expectation value decays as exactly that power.
    """
    qubits = sorted({qubit for operation in circuit.instructions for qubit in operation.locus})
    axis = {qubit: position for position, qubit in enumerate(qubits)}
    state = np.zeros([2] * len(qubits), dtype=complex)
    state[(0,) * len(qubits)] = 1.0
    for operation in circuit.instructions:
        if operation.name == "prx":
            gate = _prx(float(operation.args["angle"]), float(operation.args["phase"]))
            state = np.moveaxis(
                np.tensordot(gate, state, axes=([1], [axis[operation.locus[0]]])), 0, axis[operation.locus[0]]
            )
        elif operation.name == "cz":
            index = [slice(None)] * len(qubits)
            for qubit in operation.locus:
                index[axis[qubit]] = 1
            state[tuple(index)] *= -1
        elif operation.name not in _PASSIVE:
            raise ValueError(f"Unsupported operation {operation.name!r}.")

    clbits = measurement_map(circuit)
    width = len(clbits)
    probabilities = np.zeros(2**width)
    for outcome, amplitude in np.ndenumerate(state):
        value = sum(outcome[axis[qubit]] << clbit for qubit, clbit in clbits.items())
        probabilities[value] += abs(amplitude) ** 2

    survival = (1.0 - depolarizing) ** num_gates(circuit)
    probabilities = survival * probabilities + (1.0 - survival) / 2**width
    flips = readout if isinstance(readout, dict) else dict.fromkeys(clbits, (readout, readout))
    outcomes = np.arange(2**width)
    for qubit, clbit in clbits.items():
        to_one, to_zero = flips.get(qubit, (0.0, 0.0))
        partner = probabilities[outcomes ^ (1 << clbit)]
        probabilities = np.where(
            (outcomes >> clbit) & 1,
            (1.0 - to_zero) * probabilities + to_one * partner,
            (1.0 - to_one) * probabilities + to_zero * partner,
        )
    return {format(value, f"0{width}b"): float(p) for value, p in enumerate(probabilities) if p > 1e-15}


def exact_counts(circuit, shots, **noise):
    """Return the expected counts, as floats, so that results computed from them are exact."""
    return {bitstring: p * shots for bitstring, p in exact_distribution(circuit, **noise).items()}


class ExactExecutor:
    """Executor that returns the expected counts of every circuit instead of sampling them."""

    quantum_computer = "exact"
    supports_twirling = True

    def __init__(self, *, readout=0.0, depolarizing=0.0):
        self.noise = {"readout": readout, "depolarizing": depolarizing}
        self._results = {}

    def topology(self):
        return None

    def submit(self, circuits, shots, *, calibration=False):
        # Calibration circuits only see readout noise, so the characterization recovers it exactly.
        noise = {"readout": self.noise["readout"]} if calibration else self.noise
        job = SubmittedJob(job_id=str(uuid4()), num_circuits=len(circuits), shots=shots)
        self._results[job.job_id] = [exact_counts(circuit, shots, **noise) for circuit in circuits]
        return job

    def collect(self, job):
        return self._results[job.job_id]
