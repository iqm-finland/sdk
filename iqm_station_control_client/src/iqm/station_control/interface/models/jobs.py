# Copyright 2025 IQM
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
"""Job executor artifact and state models."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
import functools
from uuid import UUID

from iqm.station_control.interface.pydantic_base import PydanticBase


class TimelineEntry(PydanticBase):
    """Status and timestamp pair as described in a job timeline."""

    status: JobExecutorStatus
    timestamp: datetime


class JobResult(PydanticBase):
    """Progress information about a running job."""

    job_id: UUID
    parallel_sweep_progress: list[tuple[str, int, int]]
    interrupted: bool


class JobError(PydanticBase):
    """Error log for a job."""

    full_error_log: str
    user_error_message: str


class JobData(PydanticBase):
    """Status, artifacts and metadata of a job."""

    job_id: UUID
    """Unique ID of the job."""
    job_status: JobExecutorStatus
    """Current job status."""
    job_result: JobResult
    """Progress information for the job."""  # FIXME why is it called JobResult? can it be None?
    # job_result: The output of a progressing or a successful job. This includes progress indicators.
    job_error: JobError | None
    """Error message(s) for a failed job."""
    position: int | None
    """If the job is not completed, its position in the current queue.
    1 means this task will be executed next. In other cases the value is 0."""


@functools.total_ordering
class JobExecutorStatus(Enum):
    """Different states a job can be in.

    The ordering of these statuses is important, and execution logic relies on it.
    Thus, if a new status is added, ensure that it is slotted
    in at the appropriate place. See the :meth:`__lt__` implementation for further details.
    """

    # Received by the server
    RECEIVED = "received"
    """The job has been received by the server."""

    # Validating the job
    VALIDATION_STARTED = "validation_started"
    """The job is being validated."""
    VALIDATION_ENDED = "validation_ended"
    """The job passed validation."""

    # Running PulLA
    FETCH_CALIBRATION_STARTED = "fetch_calibration_started"
    """Calibration data for the job is being fetched."""
    FETCH_CALIBRATION_ENDED = "fetch_calibration_ended"
    """Calibration data for the job has been fetched."""
    COMPILATION_STARTED = "compilation_started"
    """The job is being compiled."""
    COMPILATION_ENDED = "compilation_ended"
    """The job has been succesfully compiled."""

    # Executing sweep
    SAVE_SWEEP_METADATA_STARTED = "save_sweep_metadata_started"
    """Metadata about the sweep is being stored to database."""
    SAVE_SWEEP_METADATA_ENDED = "save_sweep_metadata_ended"
    """Metadata about the sweep has been stored to database."""
    PENDING_EXECUTION = "pending_execution"
    """The job is ready for execution and is waiting for its turn in the queue."""
    EXECUTION_STARTED = "execution_started"
    """The job has started executing on the instruments."""
    EXECUTION_ENDED = "execution_ended"
    """The job has finished execution on the instruments."""
    POST_PROCESSING_PENDING = "post_processing_pending"
    """The job has finished execution and is awaiting further processing."""
    POST_PROCESSING_STARTED = "post_processing_started"
    """Execution artifacts are being constructed and persisted."""
    POST_PROCESSING_ENDED = "post_processing_ended"
    """Execution artifacts have been constructed and persisted."""

    READY = "ready"
    """The job has completed."""

    # Job failed, can happen at any stage
    FAILED = "failed"
    """The job has failed. Error message(s) may be available."""

    ABORTED = "aborted"
    """The job has been aborted."""

    def __str__(self):
        return self.name.lower()

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, str):
            try:
                other = JobExecutorStatus(other.lower())
            except ValueError:
                return False
        elif not isinstance(other, JobExecutorStatus):
            return NotImplemented
        return self.name == other.name

    def __lt__(self, other):
        """Comparison according to definition order.

        :meta public:

        .. doctest::

            >>> JobExecutorStatus.RECEIVED < JobExecutorStatus.VALIDATION_STARTED
            True
            >>> sorted(list(JobExecutorStatus.__members__))
            [
                JobExecutorStatus.RECEIVED,
                JobExecutorStatus.VALIDATION_STARTED,
                JobExecutorStatus.VALIDATION_ENDED,
                JobExecutorStatus.FETCH_CALIBRATION_STARTED,
                JobExecutorStatus.FETCH_CALIBRATION_ENDED,
                JobExecutorStatus.COMPILATION_STARTED,
                JobExecutorStatus.COMPILATION_ENDED,
                JobExecutorStatus.SAVE_SWEEP_METADATA_STARTED,
                JobExecutorStatus.SAVE_SWEEP_METADATA_ENDED,
                JobExecutorStatus.PENDING_EXECUTION,
                JobExecutorStatus.EXECUTION_STARTED,
                JobExecutorStatus.EXECUTION_ENDED,
                JobExecutorStatus.POST_PROCESSING_PENDING,
                JobExecutorStatus.POST_PROCESSING_STARTED,
                JobExecutorStatus.POST_PROCESSING_ENDED,
                JobExecutorStatus.READY,
                JobExecutorStatus.FAILED,
                JobExecutorStatus.ABORTED,
            ]

        """
        if isinstance(other, str):
            try:
                other = JobExecutorStatus(other.lower())
            except ValueError:
                return NotImplemented
        elif not isinstance(other, JobExecutorStatus):
            return NotImplemented
        members = list(JobExecutorStatus.__members__.values())
        return members.index(self) < members.index(other)

    @classmethod
    def terminal_statuses(cls) -> set[JobExecutorStatus]:
        """Statuses from which the execution can't continue."""
        return {cls.ABORTED, cls.FAILED, cls.READY}
