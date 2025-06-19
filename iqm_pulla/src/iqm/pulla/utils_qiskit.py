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
from typing import TYPE_CHECKING

from iqm.qiskit_iqm.iqm_backend import IQMBackendBase
from iqm.qiskit_iqm.iqm_job import IQMJob
from iqm.qiskit_iqm.qiskit_to_iqm import serialize_instructions
from qiskit import QuantumCircuit
from qiskit.providers import JobStatus, JobV1, Options
from qiskit.result import Counts, Result

from iqm.cpc.interface.compiler import Circuit as CPC_Circuit
from iqm.cpc.interface.compiler import CircuitExecutionOptions, HeraldingMode
from iqm.pulla.interface import StationControlResult, TaskStatus
from iqm.pulse.builder import CircuitOperation

if TYPE_CHECKING:
    from iqm.iqm_client import Circuit, Instruction
    from iqm.qiskit_iqm.iqm_backend import DynamicQuantumArchitecture
    from iqm.qiskit_iqm.iqm_provider import IQMBackend

    from iqm.cpc.compiler.compiler import Compiler
    from iqm.pulla.pulla import Pulla


def _instruction_to_dataclass(instruction: Instruction) -> CircuitOperation:
    """Convert the iqm-client model to an iqm-pulse dataclass."""
    return CircuitOperation(
        name=instruction.name,
        implementation=instruction.implementation,
        locus=instruction.qubits,
        args=instruction.args,
    )


def _circuit_to_dataclass(circuit: Circuit) -> CPC_Circuit:
    """Convert the iqm-client model to a CPC dataclass."""
    return CPC_Circuit(
        name=circuit.name,
        instructions=tuple(_instruction_to_dataclass(instruction) for instruction in circuit.instructions),
    )


def qiskit_circuits_to_pulla(
    qiskit_circuits: QuantumCircuit | Sequence[QuantumCircuit],
    qubit_idx_to_name: dict[int, str],
    custom_gates: Collection[str] = (),
) -> list[CPC_Circuit]:
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
        CPC_Circuit(
            name=qiskit_circuit.name,
            instructions=tuple(
                map(
                    _instruction_to_dataclass,
                    serialize_instructions(
                        qiskit_circuit,
                        qubit_idx_to_name,
                        custom_gates,
                    ),
                )
            ),
        )
        for qiskit_circuit in qiskit_circuits
    ]


def qiskit_to_pulla(
    pulla: Pulla,
    backend: IQMBackend,
    qiskit_circuits: QuantumCircuit | Sequence[QuantumCircuit],
) -> tuple[list[CPC_Circuit], Compiler]:
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
        calibration_set=pulla.fetch_calibration_set_by_id(run_request.calibration_set_id),
    )
    compiler.component_mapping = (
        None
        if run_request.qubit_mapping is None
        else {m.logical_name: m.physical_name for m in run_request.qubit_mapping}
    )

    circuits = [_circuit_to_dataclass(c) for c in run_request.circuits]
    return circuits, compiler


def station_control_result_to_qiskit(
    station_control_result: StationControlResult,
    *,
    shots: int,
    execution_options: CircuitExecutionOptions,
) -> Result:
    """Convert a Station Control result to a Qiskit Result.

    Args:
        station_control_result: The Station Control result to convert.
        shots: number of shots requested
        execution_options: Circuit execution options used to produce the result.

    Returns:
        The equivalent Qiskit Result.

    """
    if station_control_result.result is None:
        raise ValueError(
            f"Cannot format station control result without result."
            f'Job status is "{station_control_result.status.value.upper()}"'
        )

    used_heralding = execution_options.heralding_mode == HeraldingMode.NONE

    # Convert the measurement results from a batch of circuits into the Qiskit format.
    batch_results: list[tuple[str, list[str]]] = [
        # TODO: Proper naming instead of "index"
        (
            f"{index}",
            IQMJob._format_measurement_results(
                circuit_measurements, requested_shots=shots, expect_exact_shots=used_heralding
            ),
        )
        for index, circuit_measurements in enumerate(station_control_result.result)
    ]

    result_dict = {
        "backend_name": "",
        "backend_version": "",
        "qobj_id": "",
        "job_id": str(station_control_result.sweep_id),
        "success": station_control_result.status == TaskStatus.READY,
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
                "calibration_set_id": None,
                # TODO: calibration set id is known to Pulla, but not the compiler; this is probably good, because
                #  the compiler is not conceptually linked to the storage of calibration data (id being the property of
                #  the storage). We need to find a nice way to pass calibration set id to this function.
            }
            for name, measurement_results in batch_results
        ],
        "date": None,
        "status": station_control_result.status.value,
        "timestamps": {
            "start_time": station_control_result.start_time,
            "end_time": station_control_result.end_time,
        },
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
        super().__init__(architecture)
        self.pulla = pulla
        self.name = "IQMPullaBackend"
        self.compiler = compiler

    def run(self, run_input, **options):
        # Convert Qiskit circuits to Pulla circuits
        pulla_circuits = qiskit_circuits_to_pulla(run_input, self._idx_to_qb)

        # Compile the circuits, build settings and execute
        playlist, context = self.compiler.compile(pulla_circuits)
        shots = options.get("shots")
        settings, context = self.compiler.build_settings(context, shots=shots)
        # Get the response data from Station Control
        response_data = self.pulla.execute(playlist, context, settings, verbose=False)

        # Convert the response data to a Qiskit result
        qiskit_result = station_control_result_to_qiskit(
            response_data, shots=shots, execution_options=context["options"]
        )

        # Return a dummy job object that can be used to retrieve the result
        dummy_job = DummyJob(self, qiskit_result)
        return dummy_job

    @classmethod
    def _default_options(cls) -> Options:
        return Options(shots=1024)

    @property
    def max_circuits(self) -> int | None:
        return None


class DummyJob(JobV1):
    """A dummy job object that can be used to retrieve the result of a locally compiled circuit.

    The ``job_id`` is the same as the ``sweep_id`` of the ``StationControlResult``.
    """

    def __init__(self, backend, qiskit_result):
        super().__init__(backend=backend, job_id=qiskit_result.job_id)
        self.qiskit_result = qiskit_result

    def result(self):
        return self.qiskit_result

    def status(self):
        return JobStatus.DONE

    def submit(self):
        return None
