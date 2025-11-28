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
"""Utilities for working with Qiskit objects."""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Sequence
from datetime import date
from typing import TYPE_CHECKING

from iqm.qiskit_iqm.iqm_backend import IQMBackendBase
from iqm.qiskit_iqm.iqm_job import IQMJob
from iqm.qiskit_iqm.qiskit_to_iqm import serialize_instructions
from qiskit import QuantumCircuit
from qiskit.providers import JobStatus, JobV1, Options
from qiskit.result import Counts, Result

from iqm.cpc.interface.compiler import Circuit, CircuitExecutionOptions, HeraldingMode
from iqm.pulla.pulla import JobStatus as IQMServerJobStatus

if TYPE_CHECKING:
    from iqm.qiskit_iqm.iqm_backend import DynamicQuantumArchitecture
    from iqm.qiskit_iqm.iqm_provider import IQMBackend

    from iqm.cpc.compiler.compiler import Compiler
    from iqm.pulla.pulla import Pulla, SweepJob


def qiskit_circuits_to_pulla(
    qiskit_circuits: QuantumCircuit | Sequence[QuantumCircuit],
    qubit_idx_to_name: dict[int, str],
    custom_gates: Collection[str] = (),
) -> list[Circuit]:
    """Convert Qiskit quantum circuits into IQM Pulse quantum circuits.

    Lower-level method, you may want to use :func:`qiskit_to_pulla` instead.

    Args:
        qiskit_circuits: One or many Qiskit quantum circuits to convert.
        qubit_idx_to_name: Mapping from Qiskit qubit indices to the names of the corresponding
            qubit names.
        custom_gates: Names of custom gates that should be treated as additional native gates
            by qiskit-iqm, i.e. they should be passed as-is to Pulla.

    Returns:
        Equivalent IQM Pulse circuit(s).

    """
    if isinstance(qiskit_circuits, QuantumCircuit):
        qiskit_circuits = [qiskit_circuits]

    return [
        Circuit(
            name=qiskit_circuit.name,
            instructions=tuple(
                serialize_instructions(
                    qiskit_circuit,
                    qubit_idx_to_name,
                    custom_gates,
                ),
            ),
        )
        for qiskit_circuit in qiskit_circuits
    ]


def qiskit_to_pulla(
    pulla: Pulla,
    backend: IQMBackend,
    qiskit_circuits: QuantumCircuit | Sequence[QuantumCircuit],
) -> tuple[list[Circuit], Compiler]:
    """Convert transpiled Qiskit quantum circuits to IQM Pulse quantum circuits.

    Also provides the Compiler object for compiling them, with the correct
    calibration set and component mapping initialized.

    Args:
        pulla: Quantum computer pulse level access object.
        backend: qiskit-iqm backend used to transpile the circuits. Determines
            the calibration set to be used by the returned compiler.
        qiskit_circuits: One or many transpiled Qiskit QuantumCircuits to convert.

    Returns:
        Equivalent IQM Pulse circuit(s), compiler for compiling them.

    """
    # TODO backend is connected to Cocos, which must be connected to the same Station Control as pulla.
    # The pieces here still don't fit perfectly together.

    # build a qiskit-iqm RunRequest, then prepare to compile and execute it using Pulla
    run_request = backend.create_run_request(qiskit_circuits, shots=1)
    if run_request.calibration_set_id is None:
        raise ValueError("RunRequest created by IQMBackend has no calibration set id.")

    # create a compiler containing all the required station information
    compiler = pulla.get_standard_compiler(
        calibration_set_values=pulla.fetch_calibration_set_values_by_id(run_request.calibration_set_id),
    )
    compiler.component_mapping = run_request.qubit_mapping
    # We can be certain run_request contains only Circuit objects, because we created it
    # right in this method with qiskit.QuantumCircuit objects
    circuits: list[Circuit] = [c for c in run_request.circuits if isinstance(c, Circuit)]
    return circuits, compiler


def sweep_job_to_qiskit(
    job: SweepJob,
    *,
    shots: int,
    execution_options: CircuitExecutionOptions,
) -> Result:
    """Convert a completed Pulla job to a Qiskit Result.

    Args:
        job: The completed job to convert.
        shots: Number of shots requested.
        execution_options: Circuit execution options used to produce the result.

    Returns:
        The equivalent Qiskit Result.

    """
    result = job.result()
    if result is None:
        raise ValueError(
            f'Cannot format Qiskit result without result measurements. Job status is "{job.status.upper()}"'
        )

    used_heralding = execution_options.heralding_mode == HeraldingMode.NONE

    # Convert the measurement results from a batch of circuits into the Qiskit format.
    batch_results: list[tuple[str, list[str]]] = [
        # TODO: Proper naming instead of "index"
        (
            f"{index}",
            IQMJob._iqm_format_measurement_results(
                circuit_measurements, requested_shots=shots, expect_exact_shots=used_heralding
            ),
        )
        for index, circuit_measurements in enumerate(result)
    ]

    result_dict = {
        "backend_name": "IQMPullaBackend",
        "backend_version": "",
        "qobj_id": "",
        "job_id": str(job.job_id),
        "success": job.status == IQMServerJobStatus.COMPLETED,
        "date": date.today().isoformat(),
        "status": str(job.status),
        "timeline": job.data.timeline.copy(),
        "results": [
            {
                "shots": len(measurement_results),
                "success": True,
                "data": {
                    "memory": measurement_results,
                    "counts": Counts(Counter(measurement_results)),
                    "metadata": {},
                },
                "header": {"name": name},
                "calibration_set_id": job.data.compilation.calibration_set_id if job.data.compilation else None,
            }
            for name, measurement_results in batch_results
        ],
    }
    return Result.from_dict(result_dict)


class IQMPullaBackend(IQMBackendBase):
    """A backend that compiles circuits locally using Pulla and submits them to Station Control for execution.

    Args:
        architecture: Describes the backend architecture.
        pulla: Instance of Pulla used to execute the circuits.
        compiler: Instance of Compiler used to compile the circuits.

    """

    def __init__(self, architecture: DynamicQuantumArchitecture, pulla: Pulla, compiler: Compiler):
        super().__init__(architecture, name="IQMPullaBackend")
        self.pulla = pulla
        self.compiler = compiler

    def run(self, run_input: QuantumCircuit | list[QuantumCircuit], shots: int = 1024, **options) -> DummyJob:
        # Convert Qiskit circuits to Pulla circuits
        pulla_circuits = qiskit_circuits_to_pulla(run_input, self._idx_to_qb)

        # Compile the circuits, build settings and execute
        playlist, context = self.compiler.compile(pulla_circuits)
        settings, context = self.compiler.build_settings(context, shots=shots)

        # submit the playlist for execution
        job = self.pulla.submit_playlist(playlist, settings, context=context)
        # wait for the job to finish, no timeout (user can use Ctrl-C to stop)
        # TODO it would be better if we did not wait and instead returned a Qiskit JobV1 containing
        # a SweepJob that can be used to actually track the job.
        job.wait_for_completion(timeout_secs=0.0)

        # Convert the response data to a Qiskit result
        qiskit_result = sweep_job_to_qiskit(job, shots=shots, execution_options=context["options"])

        # Return a dummy job object that can be used to retrieve the result
        dummy_job = DummyJob(self, qiskit_result)
        return dummy_job

    @classmethod
    def _default_options(cls) -> Options:
        return Options()

    @property
    def max_circuits(self) -> int | None:
        return None


class DummyJob(JobV1):
    """A dummy job object that can be used to retrieve the result of a locally compiled circuit.

    The ``job_id`` is the same as the ``sweep_id`` of the ``StationControlResult``.
    """

    def __init__(self, backend: IQMBackend, qiskit_result: Result) -> None:
        super().__init__(backend=backend, job_id=qiskit_result.job_id)
        self.qiskit_result = qiskit_result

    def result(self) -> Result:
        return self.qiskit_result

    def status(self) -> JobStatus:
        return JobStatus.DONE

    def submit(self) -> None:
        return None
