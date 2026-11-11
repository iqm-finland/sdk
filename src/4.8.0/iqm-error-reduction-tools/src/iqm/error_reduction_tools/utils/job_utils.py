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

"""Compile, submit and retrieve pulse-level circuits on an IQM quantum computer."""

from __future__ import annotations

from collections.abc import Sequence
from time import sleep
from typing import TYPE_CHECKING, Any

from iqm.qiskit_iqm.pulse_level_utils import sweep_job_to_qiskit

from iqm.pulse import Circuit as PulseCircuit
from iqm.pulse.playlist.instructions import ReadoutMetrics

if TYPE_CHECKING:
    from iqm.iqm_client import IQMClient, RunJobTracker
    from qiskit.result import Result

_MAX_RETRIES = 10
"""How many times to wait for a job whose results are not yet available."""

_RETRY_DELAY_S = 30.0
"""Seconds between two such waits."""


def compile_and_submit(
    client: IQMClient,
    circuits: Sequence[PulseCircuit],
    shots: int,
    compilation_options: dict[str, Any] | None = None,
) -> tuple[RunJobTracker, dict[str, Any]]:
    """Compile circuits with the standard compiler and submit them as one playlist job.

    Args:
        client: Client connected to the quantum computer.
        circuits: Circuits in the native gate set, with terminal measurements.
        shots: Shots for every circuit of the job.
        compilation_options: Extra entries for the compiler context, e.g. a dynamical decoupling strategy.
            The dictionary is copied, never modified.

    Returns:
        The job tracker, and the compiler context needed to post-process its results.

    """
    compiler = client.get_standard_compiler()
    settings = compiler.get_settings(circuits=list(circuits))
    settings.set_shots(shots)
    compile_context = dict(compilation_options) if compilation_options else None
    job_definition, context = compiler.compile(circuits=list(circuits), settings=settings, context=compile_context)
    job = client.submit_playlist(job_definition, context=context)
    return job, context


def wait_for_results(job: RunJobTracker, shots: int) -> Result:
    """Wait for a playlist job and convert its results to a qiskit result.

    Retries while the server reports the job as still waiting.

    Args:
        job: Tracker of a submitted playlist job.
        shots: Shots per circuit the job was compiled with.

    Returns:
        The results, with ``get_counts(i)`` giving the counts of circuit ``i``.

    Raises:
        RuntimeError: If the job did not complete after all retries.

    """
    for attempt in range(_MAX_RETRIES):
        job.wait_for_completion()
        try:
            return sweep_job_to_qiskit(job, shots=shots)
        except ValueError as exc:
            if "WAITING" in str(exc) and attempt < _MAX_RETRIES - 1:
                sleep(_RETRY_DELAY_S)
            else:
                raise
    raise RuntimeError("Job did not complete after multiple wait attempts.")


def serialize_compiler_context(context: dict[str, Any] | None) -> dict[str, Any] | None:
    """Reduce a compiler context to the JSON-safe entries a job tracker needs to post-process results.

    ``RunJobTracker.result()`` needs the client-side compiler context, which never reaches the server, so a job
    restored with ``IQMClient.get_job`` cannot produce results without it. Only ``readout_metrics`` and
    ``mapped_readout_keys`` are needed. Remove this once the context is recoverable from the server.

    Args:
        context: The context returned by the compiler, or ``None``.

    Returns:
        A JSON-serializable dictionary, or ``None`` if there is nothing to keep.

    """
    if context is None:
        return None
    metrics = context.get("readout_metrics")
    if metrics is None:
        return None
    return {
        "readout_metrics": {
            "num_segments": metrics.num_segments,
            "integration_occurrences": metrics.integration_occurrences,
            "timetrace_occurrences": metrics.timetrace_occurrences,
            "timetrace_lengths": metrics.timetrace_lengths,
            # sets are not JSON, and their order does not matter
            "implementations": {label: sorted(values) for label, values in metrics.implementations.items()},
        },
        "mapped_readout_keys": context.get("mapped_readout_keys", {}),
    }


def deserialize_compiler_context(data: dict[str, Any] | None) -> dict[str, Any] | None:
    """Rebuild the compiler context entries written by :func:`serialize_compiler_context`.

    Args:
        data: The serialized context, or ``None``.

    Returns:
        The context entries, or ``None`` if ``data`` is empty.

    """
    if not data:
        return None
    stored = data["readout_metrics"]
    return {
        "readout_metrics": ReadoutMetrics(
            num_segments=stored["num_segments"],
            integration_occurrences=stored["integration_occurrences"],
            timetrace_occurrences=stored.get("timetrace_occurrences", {}),
            timetrace_lengths=stored.get("timetrace_lengths", {}),
            implementations={label: set(values) for label, values in stored.get("implementations", {}).items()},
        ),
        "mapped_readout_keys": data.get("mapped_readout_keys", {}),
    }
