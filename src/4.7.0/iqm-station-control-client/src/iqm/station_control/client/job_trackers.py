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
"""Concrete implementations of job trackers for different job types.

This module contains specialized :class:`.JobTracker` subclasses that extend the
base tracking logic with type-specific result processing.

Each tracker is responsible for:

1. **Lifecycle Monitoring**: Inheriting the polling and status synchronization
   logic from the base interface.
2. **Result Interpretation**: Implementing the ``result()`` method to fetch
   specific artifacts (e.g., N-dimensional arrays for sweeps or bitstrings
   for circuits).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, ClassVar, cast

if TYPE_CHECKING:
    from iqm.cpc.compiler.compiler import Compiler
    from iqm.cpc.compiler.post_process import CircuitExecutionResults
    from iqm.cpc.core.run_result import RunResult

from iqm.station_control.interface.executor_interface import JobTracker
from iqm.station_control.interface.models import (
    CircuitJobDefinition,
    CircuitMeasurementCounts,
    CircuitMeasurementResults,
    RunData,
    RunDefinition,
    SweepData,
    SweepResults,
)
from iqm.station_control.interface.models.jobs import JobExecutorStatus
from iqm.station_control.interface.models.jobs_new import JobData, JobType
from iqm.station_control.interface.models.run import _RunTimestamps


def _run_data_from_run_definition(run_definition: RunDefinition, job_data: JobData) -> RunData:
    """Temporary helper for creating a somewhat fake RunData from a RunDefinition.

    RunData contains timestamps that RunDefinition does not have, here we replace them
    with the job timeline timestamps.

    For now only called on runs that have successfully completed.
    """
    # TODO (Ville): Get rid of this function and RunData/SweepData.
    # IQM Server interface can provide the RunDefinition object, which is the payload for the submit job request.
    # Refactor everything in EXA to use RunDefinition, and get the timestamps from JobData.timeline
    sd = run_definition.sweep_definition
    now = datetime.now(UTC)

    def find_timestamp(status: str, source: str) -> datetime:
        """Find a timestamp from job timeline, or return current time if not found."""
        for entry in job_data.timeline:
            if entry.status == status and entry.source == source:
                return entry.timestamp
        return now

    timestamps = _RunTimestamps(
        created_timestamp=find_timestamp("created", "iqm-server"),
        modified_timestamp=find_timestamp("ready", "iqm-server"),
        begin_timestamp=find_timestamp("received", "iqm-station-control"),
        end_timestamp=find_timestamp("completed", "iqm-station-control"),
    )
    return RunData(
        # RunBase
        run_id=run_definition.run_id,
        username=run_definition.username,
        experiment_name=run_definition.experiment_name,
        experiment_label=run_definition.experiment_label,
        options=run_definition.options,
        software_version_set_id=run_definition.software_version_set_id,
        # RunConfigurationBase
        additional_run_properties=run_definition.additional_run_properties,
        hard_sweeps=run_definition.hard_sweeps,
        components=run_definition.components,
        default_data_parameters=run_definition.default_data_parameters,
        default_sweep_parameters=run_definition.default_sweep_parameters,
        # _RunTimestamps
        created_timestamp=timestamps.created_timestamp,
        modified_timestamp=timestamps.modified_timestamp,
        begin_timestamp=timestamps.begin_timestamp,
        end_timestamp=timestamps.end_timestamp,
        # create SweepData out of SweepDefinition
        sweep_data=SweepData(
            # SweepBase
            sweep_id=sd.sweep_id,
            dut_label=sd.dut_label,
            settings=sd.settings,
            sweeps=sd.sweeps,
            return_parameters=sd.return_parameters,
            # timestamps and old SC job status
            created_timestamp=timestamps.created_timestamp,
            modified_timestamp=timestamps.modified_timestamp,
            begin_timestamp=timestamps.begin_timestamp,
            end_timestamp=timestamps.end_timestamp,
            # TODO (Marko): Don't use JobExecutorStatus on the client side, use JobStatus instead
            job_status=JobExecutorStatus.READY,  # this function is only called on completed jobs
        ),
    )


# TODO (Marko): Consider where to keep these job tracker implementations
#  Can't be in iqm-client package, unless we make exa-core to depend on it.
@dataclass(kw_only=True)
class RunJobTracker(JobTracker[RunDefinition]):
    """Tracker for experiment run jobs."""

    _JOB_TYPE: ClassVar[JobType] = cast(JobType, "run")

    _result_sweep: SweepResults | None = field(default=None, init=False, repr=False)
    """Cached job result. Existence implies execution has completed successfully."""
    _result: CircuitExecutionResults | None = field(default=None, init=False, repr=False)
    _result_exa: RunResult | None = field(default=None, init=False, repr=False)

    def result(self) -> CircuitExecutionResults | None:
        """Fetch and process the job results.

        Returns:
            Results for the job, or None if the results are not available.

        """
        if self._result is None:
            # TODO (Ville): this is weird, we take no compiler but instead run manually a compiler pass
            #  and use a compiler context. Because the job tracker does not store a compiler.
            if (sweep_results := self.result_sweep()) is None:
                return None

            run_data = _run_data_from_run_definition(self.payload(), self.job_data)
            self._context["run_data"] = run_data
            # FIXME: Deferred import to break cyclic dependency
            from iqm.cpc.compiler.post_process import construct_circuit_execution_results  # noqa: PLC0415

            self._result = construct_circuit_execution_results(sweep_results, self._context)

        return self._result

    def result_exa(self, *, compiler: Compiler | None = None) -> RunResult | None:
        """Fetch and process the job results.

        Args:
            compiler: Compiler for post-processing the job results. If None, just do default
                EXA post-processing.

        Returns:
            Results for the job, or None if the results are not available.

        """
        if self._result_exa is None:
            if (sweep_results := self.result_sweep()) is None:
                return None

            run_data = _run_data_from_run_definition(self.payload(), self.job_data)
            # TODO (Marko): Check performance for this or if there is any better way to do this?
            #  The client sends the "settings" with empty paths. In the current implementation,
            #  station control adds the paths during deserialization, but with IQM Server it will respond
            #  with the payload as it was sent in the request, i.e. paths are still empty here.
            #  Should we just change the logic so that client is responsible settings the paths correctly
            #  already before making the request?
            if run_data.sweep_data.settings:
                run_data.sweep_data.settings._generate_paths_and_names()

            if compiler is None:
                # FIXME: Deferred import to break cyclic dependency
                from iqm.cpc.core.run_result import construct_run_result  # noqa: PLC0415

                self._result_exa = construct_run_result(run_data, sweep_results)
            else:
                # For EXA, for now the post-processing context cannot contain anything that cannot
                # be derived from the run data.
                # Cannot cache result since it depends on ``compiler``.
                self._context["run_data"] = run_data
                result, _ = compiler.post_process(sweep_results, self._context)
                return result

        return self._result_exa

    def result_sweep(self) -> SweepResults | None:
        """Fetch the raw sweep results."""
        if self._result_sweep is None:
            if not self._is_completed():
                return None
            # TODO: Could we return also partial sweep data?
            self._result_sweep = self._executor.get_artifact_sweep_results(self.job_id)

        return self._result_sweep


@dataclass(kw_only=True)
class CircuitJobTracker(JobTracker[CircuitJobDefinition]):
    """Tracker for circuit jobs."""

    _JOB_TYPE: ClassVar[JobType] = cast(JobType, "circuit")

    _result: list[CircuitMeasurementResults] | None = field(default=None, init=False, repr=False)
    """Cached job result once execution is terminal."""

    _result_counts: list[CircuitMeasurementCounts] | None = field(default=None, init=False, repr=False)
    """Cached job results as aggrated bitstring counts."""

    def result(self) -> list[CircuitMeasurementResults] | None:
        """Fetch the job results as raw bitstrings."""
        if self._result is None:
            if not self._is_completed():
                return None

            self._result = self._executor.get_artifact_measurements(self.job_id)

        return self._result

    def result_counts(self) -> list[CircuitMeasurementCounts] | None:
        """Fetch the job results as aggrated bitstring counts."""
        if self._result_counts is None:
            if not self._is_completed():
                return None

            self._result_counts = self._executor.get_artifact_measurement_counts(self.job_id)

        return self._result_counts
