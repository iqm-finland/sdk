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

"""CLOPS_v benchmark executed at pulse level.

The parametrized QV circuits are compiled client-side and submitted through the backend's
``IQMClient``, using :mod:`iqm.benchmarks.pulse_utils`. The procedure follows Sec. III.3 of
arXiv:2110.14108:

* Before the timer: the templates are generated and transpiled, and the compiler is built.
* Timed, every update: parameter assignment, compilation, submission, execution and retrieval.
* The parameters of each update after the first are seeded with the counts of the previous update.
"""

from time import perf_counter
from typing import Any

from iqm.benchmarks import Benchmark
from iqm.benchmarks.logging_config import qcvv_logger
from iqm.benchmarks.pulse_utils import compile_pulse, prepare_pulse_compiler, retrieve_pulse, submit_pulse
from iqm.benchmarks.quantum_volume.clops import CLOPSBenchmark, CLOPSConfiguration
from iqm.benchmarks.utils import PhysicalLayout
from iqm.qiskit_iqm import IQMCircuit as QuantumCircuit
from iqm.qiskit_iqm.iqm_backend import IQMBackendBase
from pydantic import ConfigDict, Field
import xarray as xr


class CLOPSPulseBenchmark(CLOPSBenchmark):
    """CLOPS reflect the speed of execution of parametrized QV circuits, measured at pulse level."""

    @classmethod
    def name(cls) -> str:
        """Returns the name of the benchmark."""
        return "clops_pulse"

    def __init__(self, backend_arg: IQMBackendBase | str, configuration: "CLOPSPulseConfiguration"):
        """Construct the CLOPSPulseBenchmark class.

        Args:
            backend_arg: The backend to execute the benchmark on
            configuration: The configuration of the benchmark

        Raises:
            ValueError: If the backend exposes no ``IQMClient``, or if a batching physical layout
                is requested.

        """
        super().__init__(backend_arg, configuration)

        self.pulse_settings = configuration.pulse_settings
        self.use_timeslot = configuration.use_timeslot
        self.timeout_secs = configuration.timeout_secs
        self.polling_interval = configuration.polling_interval

        # Pulse-level execution needs a real IQMClient, which simulator backends do not have
        if getattr(self.backend, "client", None) is None:
            raise ValueError(
                f"Backend {self.backend_name} exposes no IQMClient, so it cannot execute pulse-level jobs. "
                f"Use the circuit-level CLOPSBenchmark instead."
            )

        if self.physical_layout != PhysicalLayout.FIXED:
            raise ValueError(
                f'Pulse-level CLOPS requires physical_layout="fixed", got "{self.physical_layout}". '
                f"Circuits are compiled against the configured qubit set."
            )

        self.time_compile_per_update: dict[str, float] = {}
        self.time_compiler_setup: float = 0.0

    def prepare_timed_loop(self, sorted_transpiled_qc_list: dict[tuple, list[QuantumCircuit]]) -> None:
        """Builds the pulse compiler and settings tree before the CLOPS timer starts.

        The compiler does not depend on the parameters, so it is built once from a template bound to zeros.

        Args:
            sorted_transpiled_qc_list: A dictionary of lists of transpiled quantum circuits

        """
        qcvv_logger.info("Building the pulse compiler and settings tree")
        template = next(iter(sorted_transpiled_qc_list.values()))[0]
        t_setup = perf_counter()
        self.compiler, self.settings_tree = prepare_pulse_compiler(
            self.backend,
            self.qubits,
            [template.assign_parameters(dict.fromkeys(template.parameters, 0.0))],
            self.num_shots,
            self.pulse_settings,
            self.configuration.active_reset_cycles,
        )
        self.time_compiler_setup = perf_counter() - t_setup

    def submit_and_retrieve(
        self,
        backend: IQMBackendBase,  # noqa: ARG002 - pulse jobs go through self.backend's IQMClient
        sorted_qc_list_parametrized: dict[tuple, list[QuantumCircuit]],
        update: int,
    ) -> tuple[list[dict[str, int]], dict[str, dict[str, Any]], float, float]:
        """Compiles the parametrized circuits of one CLOPS update to pulse level, executes them and retrieves results.

        Args:
            backend: Not used; jobs are submitted through the ``IQMClient`` of the benchmark backend
            sorted_qc_list_parametrized: A dictionary of lists of parametrized quantum circuits
            update: The current cycle update

        Returns:
            The counts of every circuit, in the order of ``sorted_qc_list_parametrized``, the job metadata,
            and the elapsed times for submission and retrieval of jobs

        """
        qcvv_logger.info("Compiling the corresponding circuit batch to pulse level")
        all_parametrized = [qc for circuits in sorted_qc_list_parametrized.values() for qc in circuits]
        run_definition, context, time_compile = compile_pulse(
            self.backend, self.compiler, self.settings_tree, all_parametrized
        )
        # Compilation happens client-side, inside the timed CLOPS loop, so keep track of it
        self.time_compile_per_update["update_" + str(update + 1)] = time_compile

        qcvv_logger.info("Executing the corresponding circuit batch")
        job_tracker, time_submit = submit_pulse(self.backend, run_definition, context, use_timeslot=self.use_timeslot)

        qcvv_logger.info("Retrieving counts")
        all_counts, all_job_metadata, time_retrieve = retrieve_pulse(
            job_tracker, self.num_shots, timeout_secs=self.timeout_secs, polling_interval=self.polling_interval
        )

        return all_counts, all_job_metadata, time_submit, time_retrieve

    def execute(self, backend: IQMBackendBase) -> xr.Dataset:
        """Executes the benchmark.

        Args:
            backend: The backend to execute the benchmark on.

        Returns:
            An xarray dataset with the CLOPS timing data, including the client-side compilation times.

        """
        dataset = super().execute(backend)
        dataset.attrs.update(
            {
                "time_compiler_setup": self.time_compiler_setup,
                "time_compile_per_update": self.time_compile_per_update,
            }
        )
        return dataset


class CLOPSPulseConfiguration(CLOPSConfiguration):
    """Pulse-level CLOPS configuration.

    Attributes:
        benchmark: ``CLOPSPulseBenchmark``.
        pulse_settings: Compiler settings-tree values keyed by dotted path, e.g.
            ``{"gates.reset_wait.reset_wait.QB1.duration": 4e-4}``. Paths outside ``qubits`` are skipped.
        use_timeslot: Whether jobs are submitted into a reserved timeslot.
        timeout_secs: Maximum time to wait for each job. ``0.0`` waits indefinitely.
        polling_interval: Seconds between job status checks.
        physical_layout: Must be "fixed".
        use_dd: Not used; set dynamical decoupling through ``pulse_settings``.

    """

    benchmark: type[Benchmark] = CLOPSPulseBenchmark
    model_config = ConfigDict(arbitrary_types_allowed=True)
    pulse_settings: dict[str, Any] = Field(default_factory=dict)
    use_timeslot: bool = False
    timeout_secs: float = 0.0
    polling_interval: float = 0.1
