# Copyright 2024 IQM
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
"""job executor artifact and state models."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from iqm.station_control.interface.models import JobExecutorStatus
from iqm.station_control.interface.pydantic_base import PydanticBase


class JobStateTimestamp(PydanticBase):
    """Represents a single timestamped state for a job."""

    job_id: UUID
    status: JobExecutorStatus
    timestamp: datetime


class JobsInQueue(PydanticBase):
    """List of jobs in a particular queue, corresponding to some job state."""

    jobs: list[JobStateTimestamp] = []
    job_count: int = 0


class QueueState(PydanticBase):
    """Describes the state of a single job queue."""

    queue: JobExecutorStatus
    jobs_in_queue: int
    completed_jobs: int
