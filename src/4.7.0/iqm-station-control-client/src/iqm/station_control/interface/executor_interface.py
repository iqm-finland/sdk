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
"""Standardized interface for quantum job execution and lifecycle management.

This module provides two primary abstractions to manage the active lifecycle of quantum tasks:

1. :class:`ExecutorInterface`: A stateless backend adapter that standardizes how
   jobs are submitted, queried, and cancelled.
2. :class:`JobTracker`: A stateful handle for an in-flight job, encapsulating
   synchronization logic, progress tracking, and results retrieval.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging
from time import sleep
from typing import Any, ClassVar, Generic, TypeVar
from uuid import UUID
import warnings

from iqm.models.channel_properties import ChannelProperties

from exa.common.data.setting_node import SettingNode
from iqm.station_control.interface.models import (
    CircuitMeasurementCounts,
    CircuitMeasurementResults,
    SoftwareVersionSet,
    StaticQuantumArchitecture,
    StrUUID,
    SweepResults,
)
from iqm.station_control.interface.models.jobs import ProgressCallback, _Progress
from iqm.station_control.interface.models.jobs_new import (
    JobData as JobDataNew,
)
from iqm.station_control.interface.models.jobs_new import (
    JobDefinition,
    JobStatus,
    JobType,
    TimelineEntry,
)
from iqm.station_control.interface.models.type_aliases import Source
from iqm.station_control.interface.serializable import Serializable

logger = logging.getLogger(__name__)

T_JobDefinition = TypeVar("T_JobDefinition", bound=JobDefinition)

DEFAULT_TIMEOUT_SECONDS: float = 10800.0  # 3 hours
"""Default timeout (in seconds) when waiting for a job to reach a terminal state."""

DEFAULT_POLLING_INTERVAL: float = 1.0
"""Default interval (in seconds) between successive status synchronization requests."""


@dataclass(kw_only=True)
class JobTracker(ABC, Generic[T_JobDefinition]):
    """Orchestrates the lifecycle of a job, bridging raw data with execution logic.

    A JobTracker wraps a static :class:`~iqm.station_control.interface.models.jobs_new.JobData`
    snapshot and uses an :class:`ExecutorInterface` to perform live actions like status
    synchronization, waiting for completion, or cancellation.
    """

    _JOB_TYPE: ClassVar[JobType]  # A class-level constant that subclasses MUST provide

    job_data: JobDataNew
    """The underlying job status and metadata."""

    _executor: ExecutorInterface
    """The executor instance used to manage the job."""

    _context: dict[str, Any] = field(default_factory=dict)
    """Optional metadata used for post-processing or result interpretation."""

    _payload: T_JobDefinition | None = field(default=None, repr=False)
    """The original definition/payload used to submit this job."""

    _max_seen_queue_pos: int = field(default=0, init=False, repr=False)
    """Internal tracker to ensure queue progress bars only move forward."""

    def __post_init__(self) -> None:
        """Validate the job type."""
        if self.job_data.type != self._JOB_TYPE:
            raise ValueError(f"Tried to initialize {self.__class__.__name__} for job type {self.job_data.type}")

    @property
    def job_id(self) -> UUID:
        """Unique ID of the job."""
        return self.job_data.id

    @property
    def status(self) -> JobStatus:
        """Last queried status of the job.

        Note that this is not necessarily the same as the current status of the job,
        unless the status is terminal. To get the current status, use :meth:`update`.
        """
        return self.job_data.status

    def update(self) -> JobStatus:
        """Update the job data by querying the execution backend.

        Modifies :attr:`job_data`. If the job is already in a terminal state (COMPLETED,
        FAILED, or CANCELLED), no query is performed to save bandwidth.

        Returns:
            Current status of the job.

        """
        if self.status not in JobStatus.terminal_statuses():
            self.job_data = self._executor.get_job(self.job_id)
        return self.status

    def cancel(self) -> None:
        """Cancel the job."""
        self._executor.cancel_job(self.job_id)

    def payload(self) -> T_JobDefinition:
        """Retrieve the original definition/payload used to submit this job.

        If the payload was provided at initialization (e.g., during submission),
        it is returned immediately. Otherwise, it is fetched from the executor.
        """
        if self._payload is None:
            logger.debug("Payload not found locally for job %s. Fetching from server...", self.job_id)
            self._payload = self._executor.get_job_payload(self.job_id, job_type=self._JOB_TYPE)
        return self._payload

    def wait_for_completion(
        self,
        *,
        timeout_secs: float = DEFAULT_TIMEOUT_SECONDS,
        polling_interval: float = DEFAULT_POLLING_INTERVAL,
        progress_callback: Callable | None = None,
    ) -> JobStatus:
        """Poll the backend updating the job status, until the job reaches a terminal state,
        or until we hit a timeout.

        The terminal states are "completed", "failed", and "cancelled".

        Will stop the polling (but does not cancel the job) upon receiving a
        KeyboardInterrupt (Ctrl-C). If you want to cancel the job, call :meth:`cancel`.

        Modifies ``self``.

        Args:
            timeout_secs: Maximum time to wait in seconds. If nonzero, the method will
                return the current status after this period even if it is non-terminal.
            polling_interval: Time to wait in seconds between successive status
                synchronization requests.
            progress_callback: Optional callback function triggered on every status update loop.
                Receives progress data packets (e.g., queue positions or execution metrics)
                to update UI progress bars. Defaults to None.

        Returns:
            Last seen job status.

        Raises:
            KeyboardInterrupt: Received Ctrl-C while waiting for the job to finish.

        """
        try:
            self._wait_for_completion(
                timeout_secs=timeout_secs,
                polling_interval=polling_interval,
                progress_callback=progress_callback or self._get_progress_bar_callback(),
            )
        except KeyboardInterrupt:
            # Refresh data one last time after interruption to reflect current state
            self.update()
            raise

        self._log()
        return self.status

    def _log(self) -> None:
        """Log currently available job-related messages, errors etc."""
        if self.job_data.messages:
            message_str = "\n".join(f"  {message.source}: {message.message}" for message in self.job_data.messages)
            logger.debug("Job messages:\n%s", message_str)

        if self.status == JobStatus.FAILED:
            logger.error("Job failed! Error(s):\n%s", self.errors)
        elif self.status == JobStatus.CANCELLED:
            logger.warning("Job was cancelled.")

    # TODO (Marko): Consider removing from JobTracker, move it to JobData or standalone utility function
    def find_timeline_entry(self, status: str, source: Source | None = None) -> TimelineEntry | None:
        """Search the job's execution timeline for an entry matching the specified criteria.

        The timeline is an ordered log of state transitions and events. This method
        performs a linear search from the beginning of the timeline and returns the
        first entry that matches the provided status and, optionally, the source.

        Args:
            status: The status string to search for (e.g., "pending", "running").
            source: The component or service that generated the timeline entry.
                If provided, only entries from this source are considered.
                If None (default), entries from any source matching the status
                will be returned.

        Returns:
            The first matching :class:`TimelineEntry` found, or ``None`` if no
            entry in the timeline satisfies the criteria.

        """
        for entry in self.job_data.timeline:
            if entry.status == status and (source is None or entry.source == source):
                return entry
        return None

    @property
    def errors(self) -> str:
        """All errors formatted as a string."""
        return "\n".join(f"  {error}" for error in self.job_data.errors)

    @staticmethod
    def _get_progress_bar_callback() -> ProgressCallback:
        """Returns a callback that manages tqdm progress bars."""
        try:
            from tqdm import tqdm  # noqa: PLC0415
        except ImportError:
            logger.warning("tqdm not installed; progress bars will not be displayed.")
            return lambda _: None

        progress_bars: dict[str, tqdm] = {}

        def _update_bars(statuses: list[_Progress]) -> None:
            for label, value, total in statuses:
                if label not in progress_bars:
                    progress_bars[label] = tqdm(total=total, desc=label, leave=True)

                bar = progress_bars[label]
                bar.total = total
                bar.n = value
                bar.refresh()

                # Automatically close the bar if it reaches 100%
                if bar.n >= bar.total > 0:
                    bar.close()
                    # We don't pop it yet to avoid flickering, but closing ensures it's cleaned up in the terminal.

        return _update_bars

    def _wait_for_completion(
        self,
        *,
        timeout_secs: float,
        polling_interval: float,
        progress_callback: ProgressCallback | None,
    ) -> None:
        """Internal polling loop for job completion."""
        logger.info("Waiting for job %s to finish...", self.job_id)
        callback = progress_callback or (lambda _: None)
        start_time = datetime.now()

        while True:
            status = self.update()
            self._report_progress(callback)

            if status in JobStatus.terminal_statuses():
                break

            if timeout_secs and (datetime.now() - start_time).total_seconds() >= timeout_secs:
                logger.warning(
                    f"Job {self.job_id} reached the timeout of {timeout_secs}s. "
                    "Stopping polling and returning the current non-terminal status {status}."
                )
                break

            sleep(polling_interval)

    def _report_progress(self, callback: ProgressCallback) -> None:
        """Parses current job_data and executes the progress callback."""
        if self.job_data.queue_position is not None:
            pos = self.job_data.queue_position
            # Ensure the "total" of the progress bar doesn't shrink if the queue grows
            self._max_seen_queue_pos = max(self._max_seen_queue_pos, pos)
            callback([("Progress in queue", self._max_seen_queue_pos - pos, self._max_seen_queue_pos)])
        elif self.job_data.execution:
            statuses = [(label, v.value, v.max_value) for label, v in self.job_data.execution.progress.items()]
            callback(statuses)

            # TODO maybe ExecutorInterface should not be generic like this

    def _is_completed(self) -> bool:
        """Checks job status and retuns True iff the job is completed.

        Also shows warnings related to the completed job.
        """
        status = self.update()
        if status != JobStatus.COMPLETED:
            return False

        for message in self.job_data.messages:
            warnings.warn(f"{message.source}: {message.message}")
        return True


T_JobTracker = TypeVar("T_JobTracker", bound=JobTracker)


# (some executors like IQMClient can handle several job types),
# but just be typed with the JobTracker and JobDefinition base types.
class ExecutorInterface(Serializable, Generic[T_JobDefinition, T_JobTracker]):
    """Interface for managing quantum job execution and results retrieval.

    Implementations are responsible for protocol-specific communication with execution
    backends, such as remote services, local processes, or hardware simulators.
    """

    @abstractmethod
    def get_experiment_configuration(self) -> dict[str, Any]:
        """Get the default experiment configuration for the backend.

        Returns:
            Default experiment configuration.

        """

    @abstractmethod
    def get_or_create_software_version_set(self, software_version_set: SoftwareVersionSet) -> int:
        """Get software version set ID from the database, or create one if it doesn't exist."""

    @abstractmethod
    def get_dut_label(self) -> str:
        """Get the QPU chip label."""

    @abstractmethod
    def get_chip_design_record(self) -> dict[str, Any]:
        """Get the chip design record for the QPU."""

    @abstractmethod
    def get_static_quantum_architecture(self) -> StaticQuantumArchitecture:
        """Get the simplified QPU topology.

        Use :meth:`get_chip_design_record` for more detailed QPU topology information.
        """

    @abstractmethod
    def get_channel_properties(self) -> dict[str, ChannelProperties]:
        """Get control channel properties, including hardware limitations and supported instructions."""

    @abstractmethod
    def get_settings(self) -> SettingNode:
        """Default settings tree of the quantum computer.

        Contains the available settings of the control instruments, and their default values.
        """

    @abstractmethod
    def submit_job(self, job_definition: T_JobDefinition, *, use_timeslot: bool = False) -> T_JobTracker:
        """Submit a job definition for execution.

        Args:
            job_definition: The content of the job to be created (e.g. and experiment run,
                or a batch of quantum circuits).
            use_timeslot: Iff ``True``, submit the job to the priority timeslot queue;
                otherwise, submit it to the shared FIFO queue.

        Returns:
            A :class:`JobTracker` instance wrapping the submitted job,
            providing methods to track progress and retrieve results.

        """

    @abstractmethod
    def get_job(self, job_id: StrUUID) -> JobDataNew:
        """Get status information about an existing job.

        Allows for managing jobs submitted in previous sessions or by other clients.

        Args:
            job_id: The ID of the job to query.

        Returns:
            Current status and metadata of the job.

        """

    @abstractmethod
    def get_job_payload(self, job_id: StrUUID, job_type: JobType) -> T_JobDefinition:
        """Get the job payload, i.e. the definition used to create the job.

        Args:
            job_id: ID of the job to query.

        Returns:
            Job payload.

        """

    @abstractmethod
    def get_artifact_sweep_results(self, job_id: StrUUID) -> SweepResults:
        """Get the results of an N-dimensional sweep job.

        Raises an error if the results are not available.

        Args:
            job_id: ID of the job to query.

        Returns:
            Results of the sweep job.

        """

    @abstractmethod
    def get_artifact_measurements(self, job_id: StrUUID) -> list[CircuitMeasurementResults]:
        """Get the measurement results of a circuit batch execution job.

        Raises an error if the results are not available.

        Args:
            job_id: ID of the job to query.

        Returns:
            For each circuit in the batch, the measurement results.

        """

    @abstractmethod
    def get_artifact_measurement_counts(self, job_id: StrUUID) -> list[CircuitMeasurementCounts]:
        """Get the measurement counts of a circuit batch execution job.

        Raises an error if the results are not available.

        Args:
            job_id: ID of the job to query.

        Returns:
            For each circuit in the batch, the measurement counts.

        """

    @abstractmethod
    def cancel_job(self, job_id: StrUUID) -> None:
        """Cancel a job that has been submitted for execution.

        A canceled job will remain in the execution record, but it will not proceed
        further. If the job is currently being executed, it is interrupted.
        If the job has already reached a terminal state (completed or failed),
        it will remain in that state.

        Args:
            job_id: ID of the job to be canceled.

        """

    @abstractmethod
    def delete_job(self, job_id: StrUUID) -> None:
        """Delete a job that has been submitted for execution.

        Works like :meth:`cancel_job`, but also removes the job from the IQM Server database.

        .. warning:: There is no way to recover a deleted job.

        Args:
            job_id: ID of the job to be deleted.

        """
