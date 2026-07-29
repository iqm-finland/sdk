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
"""Job handle for circuit execution as a pulse-level job."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .iqm_job import IQMJob

if TYPE_CHECKING:
    from iqm.cpc.compiler.post_process import CircuitExecutionResults
    from iqm.station_control.interface.models.circuit import CircuitMeasurementResultsBatch


class IQMPulseJob(IQMJob):
    """Job handle for pulse-level circuit execution.

    Returned by :meth:`~iqm.qrisp_iqm.backends.IQMBackend.run_async` when the
    submitted circuit contains :class:`~iqm.qrisp_iqm.pulse_operation.IQMPulseOperation`
    instructions.  The circuit is compiled locally to a pulse schedule and submitted
    via Pulla.

    Example::

        >>> from qrisp import QuantumCircuit
        >>> from iqm.qrisp_iqm import IQMBackend, IQMPulseOperation, IQMPulseJob
        >>> from iqm.qrisp_iqm.custom_pulse_operations import delay_quantum_op
        >>> backend = IQMBackend(device_instance="garnet", token="...")
        >>> qc = QuantumCircuit(2)
        >>> qc.h(0); qc.cx(0, 1)
        >>> qc.append(IQMPulseOperation(delay_quantum_op, {"duration": 100e-9}), [qc.qubits[0]])
        >>> qc.measure(qc.qubits)
        >>> job = backend.run_async(qc, shots=1000)
        >>> isinstance(job, IQMPulseJob)
        True
        >>> result = job.result()  # waits for completion, returns JobResult
    """

    # ------------------------------------------------------------------
    # Result parsing helpers
    # ------------------------------------------------------------------

    def _get_results(self) -> list[dict[str, int]]:
        job = self._iqm_job
        raw_result: CircuitExecutionResults | None = job.result()
        if raw_result is None:
            raise RuntimeError(f"IQM job {job.job_id} returned no results — execution may have failed")
        return self._parse_results(raw_result.circuit_measurement_results)

    @staticmethod
    def _parse_results(raw_result: CircuitMeasurementResultsBatch) -> list[dict[str, int]]:
        """Parse results: one dict per circuit."""
        return [IQMPulseJob._parse_single_circuit_result(circuit_result) for circuit_result in raw_result]

    @staticmethod
    def _parse_single_circuit_result(circuit_result: dict[str, list[list[int]]]) -> dict[str, int]:
        """Parse Pulla-style results, respecting per-circuit shot counts.

        For individual (non-parallelized) measurements the keys are
        simple strings like ``"cb_0"``.  For **parallelized**
        measurements (produced by measurement_parallelization) the
        key is a comma-separated string of classical-bit identifiers
        (e.g. ``"cb_0,cb_1,cb_2"``) and each shot result is a list
        whose length equals the number of measured qubits.

        The assembled bitstring is then reversed to match Qrisp's
        convention where ``clbits[0]`` (qubit 0) is the rightmost (LSB)
        bit.

        Args:
            circuit_result: Maps measurement key names to arrays of shot results.

        Returns:
            Bitstring to counts.

        """
        # TODO should not differ from the IQMCircuitJob results parsing

        # Filter out internal labels from reset instructions
        # TODO may be unnecessary
        circuit_result = {k: v for k, v in circuit_result.items() if "____reset" not in k}
        if not circuit_result:
            return {}

        first_value = next(iter(circuit_result.values()))
        n_shots = len(first_value)
        if n_shots == 0:
            return {}

        # Build mapping: cb_name → (key, index_within_key).
        # The converter guarantees cb_N names where N = clbit index,
        # so sorting by numeric suffix recovers circuit order.
        cb_to_location: dict[str, tuple[str, int]] = {}
        for key in circuit_result:
            # "cb_0,cb_1,cb_2" → ["cb_0", "cb_1", "cb_2"]
            # "cb_0"             → ["cb_0"]
            cb_names = key.split(",")
            for idx, cb_name in enumerate(cb_names):
                cb_to_location[cb_name] = (key, idx)

        # Sort lexically: zero-padded cb names (cb_000, cb_001, …) sort correctly to match clbit index order.
        # TODO this could be more efficiently implemented
        sorted_cb_names = sorted(cb_to_location)
        bitstring_counts: dict[str, int] = {}
        for shot_idx in range(n_shots):
            bitstring = "".join(
                str(circuit_result[cb_to_location[cb][0]][shot_idx][cb_to_location[cb][1]]) for cb in sorted_cb_names
            )
            # Reverse to match Qrisp's convention:
            # clbits[0] (qubit 0) is the rightmost (LSB) bit.
            bitstring = bitstring[::-1]
            bitstring_counts[bitstring] = bitstring_counts.get(bitstring, 0) + 1

        return bitstring_counts
