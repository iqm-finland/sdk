# Copyright 2025 IQM client developers
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
"""Data models used by IQMServerClient."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal, TypeAlias
from uuid import UUID

from pydantic import BaseModel, Field

from iqm.station_control.interface.models import ObservationSetWithObservations
from iqm.station_control.interface.pydantic_base import PydanticBase

CalibrationSet: TypeAlias = ObservationSetWithObservations
QualityMetricSet: TypeAlias = ObservationSetWithObservations


# Keep 'str' for potential new sources
Source: TypeAlias = Literal["iqm-server", "iqm-station-control"] | str
"""Type indicating the source of a particular job-related object from the server."""


class QuantumComputer(PydanticBase):
    """Quantum computer attributes."""

    id: UUID
    """Unique ID of the quantum computer."""
    alias: str
    """Quantum computer alias that can be used as a substitute for id in API calls."""


class ListQuantumComputersResponse(PydanticBase):
    """Response of GET /v1/quantum-computers"""

    quantum_computers: list[QuantumComputer]
    """List of available quantum computers."""


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
    """Progress information about job compilation."""

    calibration_set_id: UUID | None = None
    """ID of the calibration set used by the compiler."""


class JobMessage(PydanticBase):
    """Message log for a job."""

    source: Source
    """Source of the message."""
    message: str
    """Content of the message."""

    def __str__(self) -> str:
        """Prettyprinting."""
        return f"{self.source}: {self.message}"


class JobError(PydanticBase):
    """Error log for a job."""

    source: Source
    """Source of the error."""
    message: str
    """Verbose error message."""
    error_code: str | None = None
    """Short error code classifying the error category."""

    def __str__(self) -> str:
        """Prettyprinting."""
        return f"{self.source}: {self.error_code}: {self.message}"


class TimelineEntry(BaseModel):
    """Timeline entry for a job."""

    source: Source
    """Source of the timeline entry."""
    status: str
    """Name of the execution step that was reached."""
    timestamp: datetime
    """Time at which ``status`` was reached."""


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
        """Statuses from which the execution can't continue."""
        return frozenset({cls.COMPLETED, cls.FAILED, cls.CANCELLED})


# Note: there is another, related JobData class for StationControlClient
class JobData(PydanticBase):
    """Status, artifacts and metadata of a job."""

    id: UUID
    """Unique ID of the job."""
    status: JobStatus
    """Current job status."""
    execution: JobExecution | None = None
    """Execution information for the job."""
    compilation: JobCompilation | None = None
    """Compilation information for the job."""
    messages: list[JobMessage] = Field(default=[])
    """Informational messages for the job."""
    errors: list[JobError] = Field(default=[])
    """Errors for a failed job."""
    queue_position: int | None = None
    """Iff the status is JobStatus.WAITING, the number of jobs ahead of this job in the queue.
    Otherwise None."""
    timeline: list[TimelineEntry] = Field(default=[])
    """Server-side statuses reached by the job so far. May include statuses from several services."""

    def find_timeline_entry(
        self,
        *,
        status: str,
        source: Source | None = None,
    ) -> TimelineEntry | None:
        """Search the timeline for an entry matching the given criteria.

        Args:
            status: Status of the searched timeline entry.
            source: Source of the searched timeline entry. If None, accepts any source.

        Returns:
            The first matching entry or ``None`` if the job timeline does not have any matching entries.

        """
        for timeline_entry in self.timeline:
            if timeline_entry.status == status and (timeline_entry.source == source or source is None):
                return timeline_entry
        return None
