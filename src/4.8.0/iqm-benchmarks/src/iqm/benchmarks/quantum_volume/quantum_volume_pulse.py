# Copyright 2024 IQM Benchmarks developers
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

"""Quantum Volume benchmark executed at pulse level.

The transpiled QV circuits are compiled client-side and submitted through the backend's
``IQMClient``, using :mod:`iqm.benchmarks.pulse_utils`. Circuit generation and analysis are the same
as in :mod:`iqm.benchmarks.quantum_volume.quantum_volume`.
"""

from datetime import UTC, datetime
import time
from typing import Any

from iqm.benchmarks.benchmark_definition import (
    BENCHMARK_TIMESTAMP_FORMAT,
    Benchmark,
    add_counts_to_dataset,
)
from iqm.benchmarks.circuit_containers import BenchmarkCircuit, CircuitGroup, Circuits
from iqm.benchmarks.logging_config import qcvv_logger
from iqm.benchmarks.pulse_utils import compile_pulse, prepare_pulse_compiler, retrieve_pulse, submit_pulse
from iqm.benchmarks.quantum_volume.quantum_volume import (
    QuantumVolumeBenchmark,
    QuantumVolumeConfiguration,
)
from iqm.benchmarks.readout_mitigation import M3IQM, M3IQMPulse
from iqm.benchmarks.utils import (
    PhysicalLayout,
    count_native_gates,
    set_coupling_map,
    xrvariable_to_counts,
)
from iqm.qiskit_iqm import IQMCircuit as QuantumCircuit
from iqm.qiskit_iqm.iqm_backend import IQMBackendBase
from pydantic import ConfigDict, Field
import xarray as xr


class QuantumVolumePulseBenchmark(QuantumVolumeBenchmark):
    """Quantum Volume executed at pulse level."""

    @classmethod
    def name(cls) -> str:
        """Returns the name of the benchmark."""
        return "quantum_volume_pulse"

    def __init__(
        self,
        backend_arg: IQMBackendBase | str,
        configuration: "QuantumVolumePulseConfiguration",
    ):
        """Construct the QuantumVolumePulseBenchmark class.

        Args:
            backend_arg: the backend to execute the benchmark on
            configuration: the configuration of the benchmark

        Raises:
            ValueError: If the backend exposes no ``IQMClient``, or if a batching physical layout
                is requested.

        """
        super().__init__(backend_arg, configuration)

        self.pulse_settings = configuration.pulse_settings
        self.use_timeslot = configuration.use_timeslot
        self.timeout_secs = configuration.timeout_secs

        # Pulse-level execution needs a real IQMClient, which simulator backends do not have
        if getattr(self.backend, "client", None) is None:
            raise ValueError(
                f"Backend {self.backend_name} exposes no IQMClient, so it cannot execute pulse-level jobs. "
                f"Use the circuit-level QuantumVolumeBenchmark instead."
            )

        if self.physical_layout != PhysicalLayout.FIXED:
            raise ValueError(
                f'Pulse-level Quantum Volume requires physical_layout="fixed", got "{self.physical_layout}". '
                f"Circuits are compiled against the configured qubit layout."
            )

    def add_all_meta_to_dataset(self, dataset: xr.Dataset) -> None:
        """Adds all configuration metadata and circuits to the dataset variable.

        Args:
            dataset: The xarray dataset

        """
        dataset.attrs["session_timestamp"] = self.session_timestamp
        dataset.attrs["execution_timestamp"] = self.execution_timestamp
        dataset.attrs["backend_name"] = self.backend_name

        for key, value in self.configuration:
            if key == "benchmark":  # Avoid saving the class object
                dataset.attrs[key] = value if isinstance(value, str) else value.name()
            else:
                dataset.attrs[key] = value

    def compile_and_submit_single_qv_job(
        self,
        qubits: list[int],
        transpiled_qc_list: list[QuantumCircuit],
    ) -> dict[str, Any]:
        """Compile the QV circuits of one layout to pulse level and submit them.

        Args:
            qubits: The qubit layout to identify the submitted job.
            transpiled_qc_list: The transpiled circuits for this layout.

        Returns:
            Dict with qubit layout, job tracker, results type and the compilation and submission times.

        """
        t_start = time.monotonic()
        compiler, settings_tree = prepare_pulse_compiler(
            self.backend,
            qubits,
            transpiled_qc_list,
            self.shots,
            self.pulse_settings,
            self.configuration.active_reset_cycles,
        )
        run_definition, context, _ = compile_pulse(self.backend, compiler, settings_tree, transpiled_qc_list)
        time_compile = time.monotonic() - t_start
        job_tracker, time_submit = submit_pulse(self.backend, run_definition, context, use_timeslot=self.use_timeslot)

        return {
            "qubits": qubits,
            "job_tracker": job_tracker,
            "qv_results_type": "vanilla",
            "time_compile": time_compile,
            "time_submit": time_submit,
        }

    def readout_mitigator(self) -> M3IQM:
        """Creates an M3 mitigator whose calibration circuits run at pulse level with the benchmark settings.

        A shortened reset wait leaves more residual excited population, which the readout calibration only
        captures if it runs with the same reset conditions as the QV circuits.

        Returns:
            An uncalibrated pulse-level M3 mitigator.

        """
        return M3IQMPulse(
            self.backend,
            pulse_settings=self.pulse_settings,
            active_reset_cycles=self.configuration.active_reset_cycles,
            use_timeslot=self.use_timeslot,
            timeout_secs=self.timeout_secs,
        )

    def execute(self, backend: IQMBackendBase) -> xr.Dataset:
        """Executes the benchmark.

        Args:
            backend: The backend to execute the benchmark on.

        Returns:
            An xarray dataset with the counts and timings of every qubit layout.

        """
        self.execution_timestamp = datetime.now(UTC).strftime(BENCHMARK_TIMESTAMP_FORMAT)
        total_compile: float = 0
        total_submit: float = 0
        total_retrieve: float = 0

        dataset = xr.Dataset()
        self.add_all_meta_to_dataset(dataset)

        # Compile and submit jobs for all qubit layouts first, then collect the results
        all_qv_jobs: list[dict[str, Any]] = []
        time_circuit_generation = {}
        time_transpilation = {}
        transpile_seeds = {}
        sorted_qc_list_indices = {}

        # Initialize the variable to contain the QV circuits of each layout
        self.circuits = Circuits()
        self.untranspiled_circuits = BenchmarkCircuit(name="untranspiled_circuits")
        self.transpiled_circuits = BenchmarkCircuit(name="transpiled_circuits")
        all_op_counts = {}

        for qubits in self.custom_qubits_array:  # NB: jobs will be submitted for qubit layouts in the specified order
            num_qubits = len(qubits)
            depth = num_qubits
            qcvv_logger.info(f"Executing pulse-level QV on qubits {qubits}")

            qc_list, time_circuit_generation[str(qubits)] = self.generate_circuit_list(num_qubits, depth=depth)
            qcvv_logger.info(f"Successfully generated all {self.num_circuits} circuits to be executed")
            # Set the coupling map
            coupling_map = set_coupling_map(qubits, backend, self.physical_layout)
            # Perform transpilation to backend
            qcvv_logger.info(f'Will transpile according to "{self.physical_layout}" physical layout')
            (transpiled_qc_list, transpile_seeds[str(qubits)]), time_transpilation[str(qubits)] = (
                self.transpile_circuits(qc_list, qubits, coupling_map, backend)
            )
            # Fixed physical layout, so a single batch
            sorted_qc_list_indices[str(qubits)] = {tuple(qubits): list(range(len(qc_list)))}

            self.untranspiled_circuits.circuit_groups.append(CircuitGroup(name=str(qubits), circuits=qc_list))
            self.transpiled_circuits.circuit_groups.append(CircuitGroup(name=str(qubits), circuits=transpiled_qc_list))

            # Count operations
            all_op_counts[str(qubits)] = count_native_gates(backend, transpiled_qc_list)

            # Compile to pulse level and submit
            t_start = time.monotonic()
            job_dict = self.compile_and_submit_single_qv_job(list(qubits), transpiled_qc_list)
            total_compile += job_dict["time_compile"]
            total_submit += time.monotonic() - t_start
            all_qv_jobs.append(job_dict)
            qcvv_logger.info(f"Job for layout {qubits} submitted successfully!")

        # Retrieve counts of jobs for all qubit layouts
        for job_idx, job_dict in enumerate(all_qv_jobs):
            qubits = job_dict["qubits"]
            execution_results, all_job_metadata, time_retrieve = retrieve_pulse(
                job_dict["job_tracker"], self.shots, timeout_secs=self.timeout_secs
            )
            total_retrieve += time_retrieve
            # Export all to dataset
            dataset.attrs.update(
                {
                    job_idx: {
                        "qubits": qubits,
                        "qv_results_type": job_dict["qv_results_type"],
                        "time_circuit_generation": time_circuit_generation[str(qubits)],
                        "time_transpilation": time_transpilation[str(qubits)],
                        "transpile_seeds": transpile_seeds[str(qubits)],
                        "time_batching": 0,
                        "time_compile": job_dict["time_compile"],
                        "time_submit": job_dict["time_submit"],
                        "time_retrieve": time_retrieve,
                        "all_job_metadata": all_job_metadata,
                        "sorted_qc_list_indices": {
                            str(key): value for key, value in sorted_qc_list_indices[str(qubits)].items()
                        },
                        "operation_counts": all_op_counts[str(qubits)],
                    }
                }
            )

            qcvv_logger.info(f"Adding counts of {qubits} run to the dataset")
            dataset, _ = add_counts_to_dataset(execution_results, str(qubits), dataset)

        self.circuits = Circuits([self.transpiled_circuits, self.untranspiled_circuits])

        if self.rem:
            # The readout calibration runs at pulse level with the same settings, see readout_mitigator
            rem_quasidistros = {}
            for qubits in self.custom_qubits_array:
                exec_counts = xrvariable_to_counts(dataset, str(qubits), self.num_circuits)
                transpiled_circuit = self.transpiled_circuits[str(qubits)]
                if transpiled_circuit is not None:
                    rem_quasidistros[f"REM_quasidist_{str(qubits)}"] = self.get_rem_quasidistro(
                        {tuple(qubits): transpiled_circuit.circuits},
                        sorted_qc_list_indices[str(qubits)],
                        exec_counts,
                        self.mit_shots,
                    )
            dataset.attrs.update({"REM_quasidistributions": rem_quasidistros})

        dataset.attrs["total_compile_time"] = total_compile
        dataset.attrs["total_submit_time"] = total_submit
        dataset.attrs["total_retrieve_time"] = total_retrieve
        qcvv_logger.info("Pulse-level QV experiment execution concluded !")
        return dataset


class QuantumVolumePulseConfiguration(QuantumVolumeConfiguration):
    """Pulse-level Quantum Volume configuration.

    Attributes:
        benchmark: ``QuantumVolumePulseBenchmark``.
        rem: Whether Readout Error Mitigation is applied. The readout calibration circuits are executed at
            pulse level with ``pulse_settings`` and ``active_reset_cycles``, so that they run with the same reset
            conditions as the QV circuits.
        pulse_settings: Compiler settings-tree values keyed by dotted path, e.g.
            ``{"gates.reset_wait.reset_wait.QB1.duration": 4e-4}``. Paths outside a layout are skipped.
        use_timeslot: Whether jobs are submitted into a reserved timeslot.
        timeout_secs: Maximum time to wait for each job. ``0.0`` waits indefinitely.
        physical_layout: Must be "fixed".
        use_dd: Not used; set dynamical decoupling through ``pulse_settings``.

    """

    benchmark: type[Benchmark] = QuantumVolumePulseBenchmark
    model_config = ConfigDict(arbitrary_types_allowed=True)
    pulse_settings: dict[str, Any] = Field(default_factory=dict)
    use_timeslot: bool = False
    timeout_secs: float = 0.0
