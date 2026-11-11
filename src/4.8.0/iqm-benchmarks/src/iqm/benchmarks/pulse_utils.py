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

"""Helpers for running benchmarks at pulse level.

Typical usage::

    compiler, settings_tree = prepare_pulse_compiler(backend, qubits, transpiled_circuits, shots, settings)
    run_definition, context, _ = compile_pulse(backend, compiler, settings_tree, transpiled_circuits)
    job, _ = submit_pulse(backend, run_definition, context)
    counts, job_meta, _ = retrieve_pulse(job, shots)

Settings are given as settings-tree values keyed by dotted path, e.g.
``{"stages.dynamical_decoupling.apply_dd_strategy.dd_is_disabled": False}``.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from time import perf_counter
from typing import Any, cast

from iqm.benchmarks.logging_config import qcvv_logger
from iqm.qiskit_iqm.iqm_job import IQMJob
from iqm.qiskit_iqm.iqm_provider import IQMBackend
from iqm.qiskit_iqm.pulse_level_utils import qiskit_to_iqm
from qiskit import QuantumCircuit

from exa.common.data import SettingNode
from exa.common.errors import NotFoundError
from iqm.cpc.compiler.compiler import Compiler
from iqm.cpc.compiler.post_process import CircuitExecutionResults
from iqm.cpc.compiler.standard_stages import HERALDING_KEY
from iqm.pulse import Circuit
from iqm.station_control.client.job_trackers import RunJobTracker
from iqm.station_control.interface.models.jobs_new import JobStatus
from iqm.station_control.interface.models.run import RunDefinition

INTERNAL_MEASUREMENT_KEY_PREFIX = "__"
"""Prefix of the compiler's internal measurement keys (e.g. ``__reset``), which are dropped from the counts."""


def prepare_pulse_compiler(  # noqa: PLR0913
    backend: IQMBackend,
    qubits: Sequence[int],
    circuits: Sequence[QuantumCircuit],
    shots: int,
    settings: Mapping[str, Any] | None = None,
    active_reset_cycles: int | None = None,
) -> tuple[Compiler, SettingNode]:
    """Build a compiler and its settings tree for the given qubits.

    Args:
        backend: The IQM backend the circuits were transpiled for. Must expose an ``IQMClient``.
        qubits: Physical qubit indices the circuits act on.
        circuits: Transpiled Qiskit circuits. Only the first one is used, to resolve the compilation stages.
        shots: Number of measurement shots per circuit.
        settings: Settings-tree values to set, keyed by dotted path. Paths not in the settings tree of
            ``qubits`` are skipped.
        active_reset_cycles: Number of active reset cycles. ``None`` keeps the compiler default.

    Returns:
        The compiler and the settings tree to compile with.

    Raises:
        ValueError: If a value of ``settings`` cannot be assigned to its path.

    """
    qubit_names = [backend.index_to_qubit_name(q) for q in qubits]
    coupler_to_components = backend.client.get_chip_topology().coupler_to_components
    couplers = [c for c, comps in coupler_to_components.items() if all(q in qubit_names for q in comps)]

    iqm_circuits, compiler = qiskit_to_iqm(backend, list(circuits[:1]))
    settings_tree = compiler.get_settings(circuits=iqm_circuits, qubits=qubit_names, couplers=couplers)
    settings_tree.set_shots(shots)
    if active_reset_cycles is not None:
        settings_tree.stages.timebox_stage.prepend_reset.active_reset_cycles = active_reset_cycles

    settings = settings or {}
    skipped = []
    for path, value in settings.items():
        try:
            settings_tree[path] = value
        except NotFoundError:
            skipped.append(path)
        except Exception as exc:
            raise ValueError(f"Cannot set pulse setting '{path}': {exc}") from exc
    if settings:
        log = qcvv_logger.warning if len(skipped) == len(settings) else qcvv_logger.info
        log(
            f"Pulse settings: applied {len(settings) - len(skipped)} of {len(settings)} paths to the settings tree "
            f"of qubits {qubit_names}; skipped (not found): {skipped}"
        )
    return compiler, settings_tree


def compile_pulse(
    backend: IQMBackend,
    compiler: Compiler,
    settings_tree: SettingNode,
    circuits: Sequence[QuantumCircuit],
) -> tuple[RunDefinition, dict[str, Any], float]:
    """Compile transpiled Qiskit circuits into a pulse-level run definition.

    Args:
        backend: The IQM backend the circuits were transpiled for.
        compiler: The compiler returned by :func:`prepare_pulse_compiler`.
        settings_tree: The settings tree returned by :func:`prepare_pulse_compiler`.
        circuits: Transpiled Qiskit circuits.

    Returns:
        The run definition, the compiler context and the compilation time in seconds.

    Raises:
        ValueError: If the backend produced a job definition without a calibration set id.

    """
    t0 = perf_counter()
    job_definition = backend.create_circuit_job_definition(list(circuits), shots=1)
    if job_definition.calibration_set_id is None:
        raise ValueError("CircuitJobDefinition created by IQMBackend has no calibration set id.")
    iqm_circuits = [c for c in job_definition.circuits if isinstance(c, Circuit)]
    compiler.component_mapping = job_definition.qubit_mapping
    run_definition, context = compiler.compile(iqm_circuits, settings=settings_tree)
    return cast(RunDefinition, run_definition), context, perf_counter() - t0


def submit_pulse(
    backend: IQMBackend,
    run_definition: RunDefinition,
    context: dict[str, Any],
    use_timeslot: bool = False,
) -> tuple[RunJobTracker, float]:
    """Submit a compiled run definition for execution, without waiting for results.

    Args:
        backend: The IQM backend whose ``IQMClient`` submits the job.
        run_definition: The run definition returned by :func:`compile_pulse`.
        context: The compiler context returned by :func:`compile_pulse`.
        use_timeslot: Whether to submit the job into a reserved timeslot.

    Returns:
        The job tracker and the submission time in seconds.

    """
    t0 = perf_counter()
    job = backend.client.submit_playlist(run_definition, context=context, use_timeslot=use_timeslot)
    t_submit = perf_counter() - t0
    qcvv_logger.info(f"Pulse level: submitted job {job.job_id}")
    return job, t_submit


def retrieve_pulse(
    job: RunJobTracker,
    shots: int,
    timeout_secs: float = 0.0,
    polling_interval: float = 1.0,
) -> tuple[list[dict[str, int]], dict[str, Any], float]:
    """Wait for a submitted pulse-level job and return its counts and metadata.

    Args:
        job: The job tracker returned by :func:`submit_pulse`.
        shots: Number of measurement shots per circuit.
        timeout_secs: Maximum time to wait for the job. ``0.0`` waits indefinitely.
        polling_interval: Seconds between job status checks.

    Returns:
        The counts of every circuit, the job metadata and the retrieval time in seconds.

    Raises:
        RuntimeError: If the job did not complete successfully.

    """
    t0 = perf_counter()
    status = job.wait_for_completion(timeout_secs=timeout_secs, polling_interval=polling_interval)
    if status != JobStatus.COMPLETED:
        errors = "\n".join(str(e) for e in job.job_data.errors)
        raise RuntimeError(f"Pulse level execution failed with status {status}:\n{errors}")

    results = job.result()
    if results is None:
        raise RuntimeError(f"Pulse level job {job.job_id} completed without results")
    counts = _counts(results, shots)

    timestamps = {entry.status: entry.timestamp for entry in job.job_data.timeline}
    job_meta = {
        "batch_job_1": {
            "job_id": str(job.job_id),
            "status": str(status),
            "circuits_in_batch": len(counts),
            "shots": shots,
            "timestamps": timestamps,
        }
    }
    return counts, job_meta, perf_counter() - t0


def _counts(results: CircuitExecutionResults, shots: int) -> list[dict[str, int]]:
    """Convert per-circuit measurement results into Qiskit-style counts."""
    used_heralding = any(HERALDING_KEY in label for label in results.sweep_results)
    counts = []
    for circuit_results in results.circuit_measurement_results:
        user_results = {k: v for k, v in circuit_results.items() if not k.startswith(INTERNAL_MEASUREMENT_KEY_PREFIX)}
        bitstrings = IQMJob._iqm_format_measurement_results(
            user_results, requested_shots=shots, expect_exact_shots=not used_heralding
        )
        counts.append(dict(Counter(bitstrings)))
    return counts
