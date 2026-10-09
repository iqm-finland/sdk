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
"""Job related interface models."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal, TypeAlias
from uuid import UUID

from pydantic import BaseModel, Field

from iqm.station_control.interface.errors import IQMServerError
from iqm.station_control.interface.models.circuit import CircuitJobDefinition
from iqm.station_control.interface.models.run import RunDefinition
from iqm.station_control.interface.models.type_aliases import Source
from iqm.station_control.interface.pydantic_base import PydanticBase

JobType = Literal["run", "circuit"]
JobDefinition: TypeAlias = RunDefinition | CircuitJobDefinition


class ProgressInfo(PydanticBase):
    """Progress information about completing an arbitrary task.

    Used e.g. for tracking job completion.
    """

    value: int
    """Current progress indicator value."""
    max_value: int
    """When we hit this, the task is done."""


class JobExecution(PydanticBase):
    """Progress information about job execution."""

    progress: dict[str, ProgressInfo]
    """Mapping from label to its progress information (value, max_value)."""


class JobCompilation(PydanticBase):
    """Progress information about job compilation.

    Only circuit-level jobs are compiled on the server side, for pulse-level jobs
    this struct is empty.
    """

    calibration_set_id: UUID | None = None
    """ID of the calibration set used by the compiler."""


class JobMessage(PydanticBase):
    """Message log for a job."""

    source: Source
    """Source of the message."""
    message: str
    """Content of the message."""

    def __str__(self) -> str:
        """Pretty printing."""
        return f"{self.source}: {self.message}"


class TimelineEntry(BaseModel):
    """Timeline entry for a job."""

    source: Source
    """Source of the timeline entry."""
    status: str
    """Name of the execution step that was reached."""
    timestamp: datetime
    """Time at which ``status`` was reached."""


class JobArtifactInfo(PydanticBase):
    """Description of an artifact for a job, currently available on the server."""

    type: str
    """Name of the available artifact type."""


class JobStatus(StrEnum):
    """Job statuses in IQMServer."""

    WAITING = "waiting"
    """Job is in a queue, waiting to be executed."""
    PROCESSING = "processing"
    """Job is being executed."""
    COMPLETED = "completed"
    """Job has completed successfully."""
    FAILED = "failed"
    """Job has failed."""
    CANCELLED = "cancelled"
    """Job has been cancelled by the user or the admin."""

    @classmethod
    def terminal_statuses(cls) -> frozenset[JobStatus]:
        """Statuses from which the execution no longer continues.

        Once a job reaches a terminal status, its status will no longer change.
        """
        return frozenset({cls.COMPLETED, cls.FAILED, cls.CANCELLED})


class JobData(PydanticBase):
    """Status, lightweight artifacts and metadata of a job."""

    id: UUID
    """Unique ID of the job."""
    type: JobType
    """What kind of job it is."""
    # qc: QCInfo  # TODO
    # """Quantum computer on which the job is executed."""
    status: JobStatus
    """Current job status."""
    execution: JobExecution | None = None
    """Execution information for the job."""
    compilation: JobCompilation | None = None
    """Compilation information for the job."""
    messages: list[JobMessage] = Field(default=[])
    """Informational messages for the job."""
    errors: list[IQMServerError] = Field(default=[])
    """Errors for a failed job."""
    queue_position: int | None = None
    """Iff the status is JobStatus.WAITING, the number of jobs ahead of this job in the queue.
    Otherwise None."""
    timeline: list[TimelineEntry] = Field(default=[])
    """Server-side statuses reached by the job so far. May include statuses from several services."""
    artifacts: list[JobArtifactInfo] = Field(default=[])
    """Generated artifacts that represent the output of the job.

    Lists the currently available artifacts. More may be added as long as the job is not in a terminal status.
    """
