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
"""Abstract base class for IQM job handles.

Defines shared status polling, cancellation, and result-retrieval logic
that does not differ between gate-level and pulse-level submissions.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Final
import warnings

from iqm.iqm_server_client.models import JobStatus as IQMJobStatus
from qrisp.interface import Job, JobResult, JobStatus

if TYPE_CHECKING:
    from iqm.station_control.interface.models.circuit import CircuitMeasurementResultsBatch

    from .backend import IQMBackend as qrisp_IQMBackend

# Map IQM's JobStatus values to Qrisp's JobStatus enum
_IQM_TO_QRISP_STATUS: Final = {
    IQMJobStatus.WAITING: JobStatus.QUEUED,
    IQMJobStatus.PROCESSING: JobStatus.RUNNING,
    IQMJobStatus.COMPLETED: JobStatus.DONE,
    IQMJobStatus.FAILED: JobStatus.ERROR,
    IQMJobStatus.CANCELLED: JobStatus.CANCELLED,
}


class IQMJob(Job):
    """Abstract base for IQM job trackers, shared by circuit-level and pulse-level jobs.

    Implements routines that do not differ between the two submission paths:
    status polling, cancellation, and the high-level result-retrieval flow.

    Args:
        job: The IQM job for an existing job. Used when retrieving a
            previously submitted job via :meth:`~iqm.qrisp_iqm.backends.IQMBackend.retrieve_job`.
        backend: The backend instance that created this job.

    Notes:
        Jobs are normally created by :meth:`~iqm.qrisp_iqm.backends.IQMBackend.run_async` and should
        not be instantiated directly by the user.

    """

    def __init__(
        self,
        job: Any,
        *,
        backend: qrisp_IQMBackend,
    ) -> None:
        super().__init__(backend=backend, job_id=str(job.job_id))
        self._iqm_job = job
        self._last_known_status = JobStatus.QUEUED
        self._cached_result: JobResult | None = None

    # ------------------------------------------------------------------
    # Narrow backend type for IQM-specific access
    # ------------------------------------------------------------------

    @property
    def backend(self) -> qrisp_IQMBackend:
        """The :class:`~iqm.qrisp_iqm.backends.IQMBackend` that created this job."""
        return self._backend

    # ------------------------------------------------------------------
    # Abstract methods (subclass-specific)
    # ------------------------------------------------------------------

    @abstractmethod
    def _get_results(self) -> list[dict[str, int]]:
        """Retrieve and parse the raw IQM result object into a list of bitstring-count dicts.

        Returns:
            One count dictionary per submitted circuit.

        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _parse_results(raw_result: CircuitMeasurementResultsBatch) -> list[dict[str, int]]:
        """Parse the raw IQM result object into a list of bitstring-count dicts.

        Returns:
            One count dictionary per submitted circuit.

        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared methods
    # ------------------------------------------------------------------

    def result(self, timeout: float | None = None) -> JobResult:
        """Block until the job completes and return its measurement results.

        Waits for the IQM job to reach a terminal state on the server, and
        then parses and returns the measurement counts, or raises an appropriate error.

        Args:
            timeout:
                Maximum number of seconds to wait for the job to finish.
                ``None`` means use the default client timeout.

        Returns:
            The measurement results.

        Raises:
            TimeoutError: If ``timeout`` is specified and the job does not finish in time.
            JobFailureError: If the job terminated with an error.
            JobCancelledError: If the job was cancelled.

        """
        if self._cached_result is None:
            # wait for the results to be available
            job = self._iqm_job
            if timeout is not None:
                status = job.wait_for_completion(timeout_secs=timeout)
            else:
                status = job.wait_for_completion()
            self._set_last_known_status(status)

            if status not in IQMJobStatus.terminal_statuses():
                raise TimeoutError(f"IQM job {job.job_id} did not finish within the timeout")

            # handle failures and cancellations
            self._raise_for_status()

            # process results
            counts_batch = self._get_results()
            self._cached_result = JobResult(counts_batch)
        return self._cached_result

    def cancel(self) -> bool:
        """Attempt to cancel the job on the IQM server.

        Returns:
            ``True`` iff cancellation was successful.

        """
        if self.in_final_state():
            return False
        try:
            self._iqm_job.cancel()
            return True
        except Exception as e:
            warnings.warn(f"Failed to cancel job {self._job_id}: {e}")
            return False

    def _set_last_known_status(self, iqm_status: IQMJobStatus) -> JobStatus:
        """Convert the given IQM job status to Qrisp job status, set it as _last_known_status, and return it."""
        qrisp_status = _IQM_TO_QRISP_STATUS.get(iqm_status)
        if qrisp_status is None:
            raise RuntimeError(f"Received unknown job status {iqm_status}")
        self._last_known_status = qrisp_status
        return qrisp_status

    def status(self) -> JobStatus:
        """Query and return the current job status.

        This is a live query that fetches the latest status from the
        IQM server, which involves a network call.

        Returns:
            The current status of the job.

        """
        iqm_status = self._iqm_job.update()
        return self._set_last_known_status(iqm_status)
