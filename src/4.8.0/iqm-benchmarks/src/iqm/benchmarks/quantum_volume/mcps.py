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

"""Max Circuits Per Second (MCPS) benchmark, at circuit and pulse level.

MCPS is the device-level speed ceiling of Sec. II.B, Eq. (1) of arXiv:2608.18044::

    MCPS = 1 / (t_gate + t_sp + t_m + t_delay)

measured by repeatedly running a single layer of Hadamard gates on all qubits followed by a
measurement. The benchmark reports the rate measured on the device, the ceiling computed from the
compiled schedule, and the wall-clock rate of the whole job.
"""

from collections.abc import Sequence
from datetime import UTC, datetime
from time import perf_counter
from typing import Any

from iqm.benchmarks.benchmark_definition import (
    BENCHMARK_TIMESTAMP_FORMAT,
    Benchmark,
    BenchmarkAnalysisResult,
    BenchmarkConfigurationBase,
    BenchmarkObservation,
    BenchmarkObservationIdentifier,
    BenchmarkRunResult,
)
from iqm.benchmarks.circuit_containers import BenchmarkCircuit, CircuitGroup, Circuits
from iqm.benchmarks.logging_config import qcvv_logger
from iqm.benchmarks.pulse_utils import compile_pulse, prepare_pulse_compiler, retrieve_pulse, submit_pulse
from iqm.benchmarks.utils import (
    PhysicalLayout,
    perform_backend_transpilation,
    retrieve_all_job_metadata,
    set_coupling_map,
    submit_execute,
)
from iqm.qiskit_iqm import IQMCircuit as QuantumCircuit
from iqm.qiskit_iqm.iqm_backend import IQMBackendBase
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from pydantic import Field
import xarray as xr

JOB_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%f%z"
"""Format of job timestamps stored as strings."""

EXECUTION_TIME_METRICS_SETTING = (
    "stages.schedule_resolution.build_playlist_and_merge_contexts.compute_execution_time_metrics"
)
"""Compiler setting that makes it store the schedule duration of each circuit in the context."""


def generate_mcps_circuit(num_qubits: int) -> QuantumCircuit:
    """Generate the MCPS circuit: a Hadamard on every qubit, followed by a measurement.

    Args:
        num_qubits: Number of qubits.

    Returns:
        The MCPS circuit.

    """
    qc = QuantumCircuit(num_qubits)
    for qubit in range(num_qubits):
        qc.h(qubit)
    qc.measure_all()
    return qc


def schedule_timing(context: dict[str, Any]) -> tuple[float, float] | None:
    """Read the duration of one shot from a compiler context.

    Args:
        context: The context returned by :func:`~iqm.benchmarks.pulse_utils.compile_pulse`, compiled with
            :data:`EXECUTION_TIME_METRICS_SETTING` enabled.

    Returns:
        The schedule duration of the first circuit, including its reset, and the end delay between
        shots, in seconds. ``None`` if the context has no schedule duration.

    """
    circuit_metrics = context.get("circuit_metrics")
    if not circuit_metrics or not circuit_metrics[0].schedule_duration:
        return None
    end_delay = float(context["settings"].controllers.options.end_delay.value)
    return float(circuit_metrics[0].schedule_duration), end_delay


def _to_datetime(value: str | datetime) -> datetime:
    """Return a job timestamp as a datetime."""
    return datetime.strptime(value, JOB_TIME_FORMAT) if isinstance(value, str) else value


def job_rates(job_meta: dict[str, Any]) -> tuple[float, float] | None:
    """Device and wall-clock circuit rates of one job.

    Args:
        job_meta: Metadata of one job, with ``timestamps``, ``shots`` and ``circuits_in_batch``.

    Returns:
        Shots per second between ``execution_started`` and ``execution_ended``, and between
        ``received`` and ``ready``. ``None`` if the job has no timestamps (e.g. on a simulator).

    """
    timestamps = job_meta.get("timestamps")
    required = ("execution_started", "execution_ended", "received", "ready")
    if not timestamps or any(key not in timestamps for key in required):
        return None
    total_shots = job_meta["shots"] * job_meta["circuits_in_batch"]
    execution = (
        _to_datetime(timestamps["execution_ended"]) - _to_datetime(timestamps["execution_started"])
    ).total_seconds()
    wall_clock = (_to_datetime(timestamps["ready"]) - _to_datetime(timestamps["received"])).total_seconds()
    return total_shots / execution, total_shots / wall_clock


def plot_mcps(values: dict[str, float], qubits: list[int], backend_name: str, timestamp: str) -> tuple[str, Figure]:
    """Bar plot of the MCPS values.

    Args:
        values: MCPS values keyed by observation name.
        qubits: The benchmarked qubits.
        backend_name: Name of the backend.
        timestamp: Execution timestamp.

    Returns:
        The figure name and the figure.

    """
    fig, ax = plt.subplots()
    labels = list(values)
    bars = ax.bar(labels, [values[k] for k in labels], color=["tab:blue", "tab:green", "tab:gray"][: len(labels)])
    ax.bar_label(bars, fmt="%.0f")
    ax.set_ylabel("Circuits per second")
    ax.set_title(f"MCPS on {len(qubits)} qubits\n{backend_name}, {timestamp}")
    plt.close(fig)
    return f"mcps_{len(qubits)}_qubits", fig


def mcps_analysis(run: BenchmarkRunResult) -> BenchmarkAnalysisResult:
    """Analysis function for an MCPS experiment.

    Args:
        run: An MCPS experiment run.

    Returns:
        The measured, ceiling and wall-clock MCPS observations and a bar plot.

    """
    dataset = run.dataset
    qubits = dataset.attrs["benchmarked_qubits"]
    identifier = BenchmarkObservationIdentifier(qubits)
    observations = []
    values: dict[str, float] = {}

    rates = [rate for meta in dataset.attrs["job_meta_per_repetition"].values() if (rate := job_rates(meta))]
    if rates:
        device, wall_clock = np.array(rates).T
        for name, samples in (("mcps_measured", device), ("mcps_wall_clock", wall_clock)):
            values[name] = float(np.mean(samples))
            observations.append(
                BenchmarkObservation(
                    name=name, value=values[name], uncertainty=float(np.std(samples)), identifier=identifier
                )
            )
    else:
        qcvv_logger.info("The jobs have no timestamps (e.g. on a simulator), so MCPS was not measured")

    if dataset.attrs.get("schedule_duration"):
        values["mcps_ceiling"] = 1.0 / (dataset.attrs["schedule_duration"] + dataset.attrs["end_delay"])
        observations.append(
            BenchmarkObservation(name="mcps_ceiling", value=values["mcps_ceiling"], identifier=identifier)
        )

    plots = {}
    if values:
        dataset.attrs.update(values)
        fig_name, fig = plot_mcps(values, qubits, dataset.attrs["backend_name"], dataset.attrs["execution_timestamp"])
        plots[fig_name] = fig

    return BenchmarkAnalysisResult(dataset=dataset, observations=observations, plots=plots)


class MCPSBenchmark(Benchmark):
    """MCPS measured with circuit-level execution."""

    analysis_function = staticmethod(mcps_analysis)

    @classmethod
    def name(cls) -> str:
        """Returns the name of the benchmark."""
        return "mcps"

    def __init__(self, backend_arg: IQMBackendBase | str, configuration: "MCPSConfiguration"):
        """Construct the MCPSBenchmark class.

        Args:
            backend_arg: The backend to execute the benchmark on.
            configuration: The configuration of the benchmark.

        """
        super().__init__(backend_arg, configuration)
        self.num_repetitions = configuration.num_repetitions
        self.active_reset_cycles = configuration.active_reset_cycles
        if configuration.qubits is not None:
            self.qubits = list(configuration.qubits)
        else:
            self.qubits = [self.backend.qubit_name_to_index(name) for name in self.backend.architecture.qubits]
        self.session_timestamp = datetime.now(UTC).strftime(BENCHMARK_TIMESTAMP_FORMAT)
        self.execution_timestamp = ""

    def add_all_meta_to_dataset(self, dataset: xr.Dataset) -> None:
        """Adds all configuration metadata to the dataset variable.

        Args:
            dataset: The xarray dataset

        """
        dataset.attrs["session_timestamp"] = self.session_timestamp
        dataset.attrs["execution_timestamp"] = self.execution_timestamp
        dataset.attrs["backend_name"] = self.backend_name
        for key, value in self.configuration:
            if key == "benchmark":  # Avoid saving the class object
                dataset.attrs[key] = value if isinstance(value, str) else value.name()
            elif key == "dd_strategy":
                dataset.attrs[key] = str(value)
            else:
                dataset.attrs[key] = value
        dataset.attrs["benchmarked_qubits"] = self.qubits
        dataset.attrs["reset_mode"] = "passive" if self.active_reset_cycles is None else "active"

    def compile_settings(self) -> dict[str, Any]:
        """Settings used to compile the circuit client-side, for the ceiling.

        Returns:
            Settings-tree values keyed by dotted path.

        """
        return {EXECUTION_TIME_METRICS_SETTING: True}

    def run_repetitions(self, circuit: QuantumCircuit) -> tuple[dict[str, Any], tuple[float, float] | None]:
        """Execute the circuit ``num_repetitions`` times at circuit level.

        Args:
            circuit: The transpiled MCPS circuit.

        Returns:
            The job metadata of every repetition, and the schedule duration and end delay of one shot
            (``None`` if the backend has no ``IQMClient`` to compile the circuit with).

        """
        timing = None
        if getattr(self.backend, "client", None) is not None:
            compiler, settings_tree = prepare_pulse_compiler(
                self.backend, self.qubits, [circuit], self.shots, self.compile_settings(), self.active_reset_cycles
            )
            _, context, _ = compile_pulse(self.backend, compiler, settings_tree, [circuit])
            timing = schedule_timing(context)

        job_meta = {}
        for repetition in range(self.num_repetitions):
            qcvv_logger.info(f"Repetition {repetition + 1}/{self.num_repetitions}")
            jobs, _ = submit_execute(
                {tuple(self.qubits): [circuit]},
                self.backend,
                self.shots,
                circuit_compilation_options=self.circuit_compilation_options,
            )
            job_meta[f"repetition_{repetition + 1}"] = retrieve_all_job_metadata(jobs)["batch_job_1"]
        return job_meta, timing

    def execute(self, backend: IQMBackendBase) -> xr.Dataset:
        """Executes the benchmark.

        Args:
            backend: The backend to execute the benchmark on.

        Returns:
            An xarray dataset with the job metadata and the schedule timing.

        """
        self.execution_timestamp = datetime.now(UTC).strftime(BENCHMARK_TIMESTAMP_FORMAT)
        dataset = xr.Dataset()
        self.add_all_meta_to_dataset(dataset)

        circuit = generate_mcps_circuit(len(self.qubits))
        # The circuit has no two-qubit gates, so the qubits need not be connected.
        coupling_map = set_coupling_map(self.qubits, backend, self.physical_layout, check_if_connected=False)
        (transpiled,), _ = perform_backend_transpilation(
            [circuit], backend, self.qubits, coupling_map, qiskit_optim_level=3, optimize_sqg=True
        )
        self.untranspiled_circuits = BenchmarkCircuit(
            name="untranspiled_circuits", circuit_groups=[CircuitGroup(name=str(self.qubits), circuits=[circuit])]
        )
        self.transpiled_circuits = BenchmarkCircuit(
            name="transpiled_circuits", circuit_groups=[CircuitGroup(name=str(self.qubits), circuits=[transpiled])]
        )
        self.circuits = Circuits([self.transpiled_circuits, self.untranspiled_circuits])

        t_start = perf_counter()
        job_meta, timing = self.run_repetitions(transpiled)
        dataset.attrs["total_time"] = perf_counter() - t_start
        dataset.attrs["job_meta_per_repetition"] = job_meta
        if timing is not None:
            dataset.attrs["schedule_duration"], dataset.attrs["end_delay"] = timing
        return dataset


class MCPSPulseBenchmark(MCPSBenchmark):
    """MCPS measured with pulse-level execution."""

    @classmethod
    def name(cls) -> str:
        """Returns the name of the benchmark."""
        return "mcps_pulse"

    def __init__(self, backend_arg: IQMBackendBase | str, configuration: "MCPSPulseConfiguration"):
        """Construct the MCPSPulseBenchmark class.

        Args:
            backend_arg: The backend to execute the benchmark on.
            configuration: The configuration of the benchmark.

        Raises:
            ValueError: If the backend exposes no ``IQMClient``, or if a batching physical layout is requested.

        """
        super().__init__(backend_arg, configuration)
        self.pulse_settings = configuration.pulse_settings
        self.use_timeslot = configuration.use_timeslot
        self.timeout_secs = configuration.timeout_secs
        self.polling_interval = configuration.polling_interval

        if getattr(self.backend, "client", None) is None:
            raise ValueError(
                f"Backend {self.backend_name} exposes no IQMClient, so it cannot execute pulse-level jobs. "
                f"Use the circuit-level MCPSBenchmark instead."
            )
        if self.physical_layout != PhysicalLayout.FIXED:
            raise ValueError(f'Pulse-level MCPS requires physical_layout="fixed", got "{self.physical_layout}".')

    def compile_settings(self) -> dict[str, Any]:
        """Settings used to compile the circuit: the pulse settings plus the schedule timing.

        Returns:
            Settings-tree values keyed by dotted path.

        """
        return {**self.pulse_settings, EXECUTION_TIME_METRICS_SETTING: True}

    def run_repetitions(self, circuit: QuantumCircuit) -> tuple[dict[str, Any], tuple[float, float] | None]:
        """Compile the circuit once and execute it ``num_repetitions`` times at pulse level.

        Args:
            circuit: The transpiled MCPS circuit.

        Returns:
            The job metadata of every repetition, and the schedule duration and end delay of one shot.

        """
        compiler, settings_tree = prepare_pulse_compiler(
            self.backend, self.qubits, [circuit], self.shots, self.compile_settings(), self.active_reset_cycles
        )
        run_definition, context, _ = compile_pulse(self.backend, compiler, settings_tree, [circuit])

        job_meta = {}
        for repetition in range(self.num_repetitions):
            qcvv_logger.info(f"Repetition {repetition + 1}/{self.num_repetitions}")
            job, _ = submit_pulse(self.backend, run_definition, context, use_timeslot=self.use_timeslot)
            _, meta, _ = retrieve_pulse(
                job, self.shots, timeout_secs=self.timeout_secs, polling_interval=self.polling_interval
            )
            job_meta[f"repetition_{repetition + 1}"] = meta["batch_job_1"]
        return job_meta, schedule_timing(context)


class MCPSConfiguration(BenchmarkConfigurationBase):
    """MCPS configuration.

    Attributes:
        benchmark: ``MCPSBenchmark``.
        qubits: Physical qubits to run on. ``None`` uses all qubits of the backend.
        shots: Shots per job. Large, so that per-job overheads do not affect the measured rate.
        num_repetitions: Number of jobs to run.
        active_reset_cycles: Number of active reset cycles. ``None`` resets passively, by waiting.

    """

    benchmark: type[Benchmark] = MCPSBenchmark
    qubits: Sequence[int] | None = None
    shots: int = 10_000
    num_repetitions: int = 5


class MCPSPulseConfiguration(MCPSConfiguration):
    """Pulse-level MCPS configuration.

    Attributes:
        benchmark: ``MCPSPulseBenchmark``.
        pulse_settings: Compiler settings-tree values keyed by dotted path, e.g.
            ``{"gates.reset_wait.reset_wait.QB1.duration": 4e-4}``. Paths outside ``qubits`` are skipped.
        use_timeslot: Whether jobs are submitted into a reserved timeslot.
        timeout_secs: Maximum time to wait for each job. ``0.0`` waits indefinitely.
        polling_interval: Seconds between job status checks.
        physical_layout: Must be "fixed".
        use_dd: Not used; set dynamical decoupling through ``pulse_settings``.

    """

    benchmark: type[Benchmark] = MCPSPulseBenchmark
    pulse_settings: dict[str, Any] = Field(default_factory=dict)
    use_timeslot: bool = False
    timeout_secs: float = 0.0
    polling_interval: float = 0.1
