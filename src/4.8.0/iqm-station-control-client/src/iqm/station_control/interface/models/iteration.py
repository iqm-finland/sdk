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
"""Iteration-related interface models."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from uuid import UUID

from pydantic import Field

from iqm.station_control.interface.models.jobs_new import QCInfo
from iqm.station_control.interface.pydantic_base import PydanticBase


class IterationStatus(StrEnum):
    """Iteration statuses in IQMServer."""

    WAITING = "waiting"
    """Iteration request was accepted and is in a queue, waiting to start."""
    ACTIVE = "active"
    """Iteration is currently active, and the quantum computer will only execute jobs submitted
    into this iteration."""
    COMPLETED = "completed"
    """Iteration has ended normally."""
    FAILED = "failed"
    """Iteration failed to complete normally, see :attr:`IterationData.messages` for the reason."""
    CANCELLED = "cancelled"
    """Iteration was ended by the user before it started."""

    @classmethod
    def terminal_statuses(cls) -> frozenset[IterationStatus]:
        """Statuses in which the iteration is not, and will not be active anymore.

        Once an iteration reaches a terminal status, its status or :class:`IterationData` will
        no longer change.
        """
        return frozenset({cls.COMPLETED, cls.FAILED, cls.CANCELLED})


class IterationMetrics(PydanticBase):
    """Runtime and billing metrics for an iteration."""

    runtime_seconds: float | None
    """Time (in seconds) the iteration has been active."""
    billed_seconds: int | None
    """Time (in seconds) billed for the iteration."""
    cost: Decimal | None
    """Iteration cost (in IQM Server credits), until now."""


class IterationDefinition(PydanticBase):
    """Request for creating an iteration."""

    qc: str
    """Quantum computer on which to request an iteration."""
    max_duration_seconds: int
    """Maximum duration of the iteration (in seconds).
    The iteration ends automatically once it has been active this long."""
    idle_timeout_seconds: int | None = None
    """Maximum time (in seconds) the iteration job queue may be empty. None means use the server default."""


class IterationData(PydanticBase):
    """Status, metrics and metadata of an iteration."""

    id: UUID
    """Unique ID for the iteration."""
    status: IterationStatus
    """Current iteration status."""
    qc: QCInfo
    """Quantum computer on which the iteration takes place."""
    queue_position: int | None = None
    """For iterations in the "waiting" state, position in the IQM Server queue, otherwise None."""
    messages: list[str] = Field(default=[])
    """Informational messages for the iteration, e.g. the reason for a :attr:`.IterationStatus.FAILED`
    status."""
    max_duration_seconds: int
    """Maximum duration of the iteration (in seconds).
    The iteration ends automatically once it has been active this long."""
    idle_timeout_seconds: int
    """Maximum time (in seconds) the iteration job queue may be empty.
    The iteration ends automatically once this limit has been passed."""
    created_at: datetime
    """When the iteration was requested/created."""
    started_at: datetime | None = None
    """When the iteration became active, or None if this has not happened yet."""
    ends_at: datetime | None = None
    """When the iteration will end automatically, or None if it has not started yet."""
    finished_at: datetime | None = None
    """When the iteration ended, or None if this has not happened yet."""
    metrics: IterationMetrics
    """Runtime and billing metrics."""
