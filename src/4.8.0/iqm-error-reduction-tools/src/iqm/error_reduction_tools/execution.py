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

"""Executors run native circuits and return their counts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol
from uuid import UUID

from iqm.iqm_client import RunJobTracker

from iqm.pulse import Circuit

from .utils.job_utils import (
    compile_and_submit,
    deserialize_compiler_context,
    serialize_compiler_context,
    wait_for_results,
)
from .utils.topology_utils import topology_from_qc, uses_move_gates

if TYPE_CHECKING:
    from iqm.iqm_client import IQMClient

    from .readout_characterization.topologies import QPUTopology


@dataclass(frozen=True)
class SubmittedJob:
    """Everything needed to retrieve the counts of a submitted job, in this session or a later one."""

    job_id: str
    """Identifier of the job on the executor."""

    num_circuits: int
    """Number of circuits in the job, in the order they were submitted."""

    shots: int
    """Shots of every circuit in the job."""

    compiler_context: dict[str, Any] | None = None
    """Serialized compiler context, needed by IQM Resonance to post-process the results of a restored job."""


@dataclass(frozen=True)
class Submission:
    """The jobs of a submitted execution plan."""

    quantum_computer: str
    """Name of the quantum computer the jobs were submitted to."""

    jobs: tuple[SubmittedJob, ...]
    """One job per batch of the plan, in the same order."""

    calibration_job: SubmittedJob | None = None
    """The readout error characterization job, if there is one."""


class Executor(Protocol):
    """Runs native circuits and returns their counts."""

    @property
    def quantum_computer(self) -> str:
        """Return the name of the quantum computer the circuits run on."""
        ...

    @property
    def supports_twirling(self) -> bool:
        """Return whether twirled circuits can be executed."""
        ...

    def topology(self) -> QPUTopology | None:
        """Return the QPU topology, which the ``"LOCAL"`` readout-twirling strategy needs."""
        ...

    def submit(self, circuits: Sequence[Circuit], shots: int, *, calibration: bool = False) -> SubmittedJob:
        """Submit circuits as one job, every circuit with the same number of shots.

        A ``calibration`` job characterizes the bare readout, so it is executed without extra compilation options.
        """
        ...

    def collect(self, job: SubmittedJob) -> list[dict[str, int]]:
        """Wait for a job and return one counts dictionary per circuit, in submission order."""
        ...


class ResonanceExecutor(Executor):
    """Runs circuits on an IQM quantum computer through IQM Resonance or a local IQM server.

    Args:
        client: Client connected to the quantum computer.
        compilation_options: Extra entries for the compiler context, e.g. a dynamical decoupling strategy.

    """

    def __init__(self, client: IQMClient, compilation_options: dict[str, Any] | None = None) -> None:
        self._client = client
        self._compilation_options = compilation_options
        self._trackers: dict[str, RunJobTracker] = {}

    @property
    def quantum_computer(self) -> str:
        """Return the name of the quantum computer the client is connected to."""
        return self._client.quantum_computer_name

    @property
    def supports_twirling(self) -> bool:
        """Return ``False`` for QPUs with computational resonators, where twirling is not supported."""
        return not uses_move_gates(self._client)

    def topology(self) -> QPUTopology | None:
        """Return the QPU topology, or ``None`` if the server does not provide it."""
        return topology_from_qc(self._client)

    def submit(self, circuits: Sequence[Circuit], shots: int, *, calibration: bool = False) -> SubmittedJob:
        """Compile the circuits and submit them as one playlist job.

        Args:
            circuits: Circuits in the native gate set.
            shots: Shots per circuit.
            calibration: Whether the job characterizes the readout, so that it is compiled without the compilation
                options.

        Returns:
            The submitted job.

        """
        tracker, context = compile_and_submit(
            self._client, circuits, shots, None if calibration else self._compilation_options
        )
        job_id = str(tracker.job_id)
        self._trackers[job_id] = tracker
        return SubmittedJob(
            job_id=job_id,
            num_circuits=len(circuits),
            shots=shots,
            compiler_context=serialize_compiler_context(context),
        )

    def collect(self, job: SubmittedJob) -> list[dict[str, int]]:
        """Wait for a job and return its counts.

        A job submitted in another session is restored from its identifier and compiler context.

        Args:
            job: The submitted job.

        Returns:
            One counts dictionary per circuit, in submission order.

        Raises:
            TypeError: If the identifier does not belong to a playlist job.
            ValueError: If a restored job carries no compiler context.

        """
        tracker = self._trackers.get(job.job_id) or self._restore(job)
        results = wait_for_results(tracker, job.shots)
        return [
            {bitstring.replace(" ", ""): int(count) for bitstring, count in results.get_counts(index).items()}
            for index in range(job.num_circuits)
        ]

    def _restore(self, job: SubmittedJob) -> RunJobTracker:
        """Rebuild the tracker of a job submitted in another session."""
        tracker = self._client.get_job(UUID(job.job_id))
        if not isinstance(tracker, RunJobTracker):
            raise TypeError(f"Job {job.job_id} is a {type(tracker).__name__}, not a playlist job.")
        context = deserialize_compiler_context(job.compiler_context)
        if context is None:
            raise ValueError(f"Job {job.job_id} carries no compiler context, so its results cannot be processed.")
        # get_job() offers no way to pass the context, see serialize_compiler_context.
        tracker._context.update(context)
        self._trackers[job.job_id] = tracker
        return tracker
