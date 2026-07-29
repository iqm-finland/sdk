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
"""Job-related models."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from enum import Enum
import functools
from typing import TypeAlias
from uuid import UUID

from iqm.station_control.interface.pydantic_base import PydanticBase

_Progress: TypeAlias = tuple[str, int, int]
"""Describes the progress of an arbitrary task: (label, value, max_value)."""

ProgressCallback: TypeAlias = Callable[[list[_Progress]], None]
"""Callback function for reporting progress on a list of tasks."""

Statuses: TypeAlias = list[_Progress]
"""Progress of the parallel sweeps of a job.
Used in Station Control. Deprecated, should not be used anywhere else."""


class TimelineEntry(PydanticBase):
    """Status and timestamp pair for a job timeline."""

    status: JobExecutorStatus
    """Job status that was reached."""
    timestamp: datetime
    """Time at which ``status`` was reached."""


class JobResult(PydanticBase):
    """Progress information for the JobExecutorStatus.EXECUTION_STARTED stage of a job."""

    # TODO redesign, should not be called JobResult since it's a progress indicator

    job_id: UUID
    """ID of the job."""
    parallel_sweep_progress: list[_Progress]
    """Progress of the sweeps if we are in the JobExecutorStatus.EXECUTION_STARTED stage, otherwise empty."""
    interrupted: bool
    """True iff the job was canceled."""


class JobError(PydanticBase):
    """Error message for a job."""

    full_error_log: str
    """Full error message for logging."""
    user_error_message: str
    """Short, human-readable error message for users."""


class JobData(PydanticBase):
    """Status, artifacts and metadata of a job."""

    # TODO redesign

    job_id: UUID
    """Unique ID of the job."""
    job_status: JobExecutorStatus
    """Current job status."""
    job_result: JobResult  # TODO should not be called JobResult, it's progress info for the execution stage
    """Progress information for the JobExecutorStatus.EXECUTION_STARTED stage of a job."""
    job_error: JobError | None
    """Error message(s) for a failed job, otherwise None."""
    position: int | None
    """Number of jobs ahead of this job in its current queue.
    None means the job has reached a terminal status.
    """


# NOTE: Keep JobExecutorStatus inheriting from Enum (not StrEnum).
# Our tests do ordering like:  str(JobExecutorStatus.X) > JobExecutorStatus.Y
# With Enum, the right operand isn’t a str, so Python dispatches to the Enum’s comparison methods,
# where we implement definition-order (<, >) logic.
# With StrEnum, members are str subclasses; when a plain str is on the left, Python uses str.__gt__
# (lexicographic) and never calls our enum’s ordering, so string↔enum ordering fails.
@functools.total_ordering
class JobExecutorStatus(Enum):
    """Different statuses a job can be in.

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

    # Running Pulla
    FETCH_CALIBRATION_STARTED = "fetch_calibration_started"
    """Calibration data for the job is being fetched."""
    FETCH_CALIBRATION_ENDED = "fetch_calibration_ended"
    """Calibration data for the job has been fetched."""
    COMPILATION_STARTED = "compilation_started"
    """The job is being compiled."""
    COMPILATION_ENDED = "compilation_ended"
    """The job has been successfully compiled."""

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

    def __eq__(self, other):  # noqa: ANN001
        if isinstance(other, str):
            try:
                other = JobExecutorStatus(other.lower())
            except ValueError:
                return False
        elif not isinstance(other, JobExecutorStatus):
            return NotImplemented
        return self.name == other.name

    def __lt__(self, other):  # noqa: ANN001
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

    @classmethod
    def in_progress_statuses(cls) -> set[JobExecutorStatus]:
        """Statuses representing jobs that should be cleaned up on restart.

        Returns all non-terminal statuses.
        """
        all_statuses = set(cls)
        return all_statuses - cls.terminal_statuses()
