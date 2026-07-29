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
"""Job handle for circuit execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .iqm_job import IQMJob

if TYPE_CHECKING:
    from iqm.station_control.interface.models.circuit import CircuitMeasurementResultsBatch


class IQMCircuitJob(IQMJob):
    """Job handle for gate-level circuit execution.

    Returned by :meth:`~iqm.qrisp_iqm.backends.IQMBackend.run_async` when the
    submitted circuit contains no :class:`~iqm.qrisp_iqm.pulse_operation.IQMPulseOperation`
    instructions.

    Example::

        >>> from qrisp import QuantumCircuit
        >>> from iqm.qrisp_iqm import IQMBackend, IQMCircuitJob
        >>> backend = IQMBackend(device_instance="garnet", token="...")
        >>> qc = QuantumCircuit(2)
        >>> qc.h(0); qc.cx(0, 1); qc.measure(qc.qubits)
        >>> job = backend.run_async(qc, shots=1000)
        >>> isinstance(job, IQMCircuitJob)
        True
        >>> result = job.result()  # waits for completion, returns JobResult
    """

    # ------------------------------------------------------------------
    # Result parsing helpers
    # ------------------------------------------------------------------

    def _get_results(self) -> list[dict[str, int]]:
        """Parse IQMClient-style results: one dict per circuit."""
        job = self._iqm_job
        raw_result: CircuitMeasurementResultsBatch | None = job.result()
        if raw_result is None:
            raise RuntimeError(f"IQM job {job.job_id} returned no results — execution may have failed")
        return self._parse_results(raw_result)

    @staticmethod
    def _parse_results(raw_result: CircuitMeasurementResultsBatch) -> list[dict[str, int]]:
        """Parse IQMClient-style results: one dict per circuit."""
        return [IQMCircuitJob._parse_single_circuit_result(circuit_result) for circuit_result in raw_result]

    @staticmethod
    def _parse_single_circuit_result(circuit_result: dict[str, list[list[int]]]) -> dict[str, int]:
        """Parse a single circuit's measurement result into counts.

        Args:
            circuit_result: Mapping from measurement keys to list of shot results.

        Returns:
            Mapping from bitstring outcomes to their counts.

        """
        if not circuit_result:
            return {}

        first_value = next(iter(circuit_result.values()))
        n_shots = len(first_value)
        n_keys = len(circuit_result)
        # measurement keys refer to single classical bits (and are lexically sortable)
        sorted_keys = sorted(circuit_result.keys(), reverse=False)

        # TODO this could be more efficient (we could just query the counts directly from server)
        bitstrings = np.empty((n_shots, n_keys), dtype=int)
        for i, key in enumerate(sorted_keys):
            v = circuit_result[key]
            res = np.array(v, dtype=int)
            # In Qrisp each measurement is a separate single-qubit instruction, so only one column is expected
            # per measurement key. qrisp-iqm assigns a unique measurement key to each classical bit.
            if res.shape != (n_shots, 1):
                raise ValueError(f"Measurement result {key} has the wrong shape {res.shape}, expected ({n_shots}, 1)")
            bitstrings[:, i] = res[:, 0]

        unique_bitstrings, counts = np.unique(bitstrings, axis=0, return_counts=True)
        return {
            "".join(map(lambda x: str(x), bitstr))[::-1]: int(
                count
            )  # Reverse bitstring to ensure Qrisp endian convention
            for bitstr, count in zip(unique_bitstrings, counts, strict=True)
        }
