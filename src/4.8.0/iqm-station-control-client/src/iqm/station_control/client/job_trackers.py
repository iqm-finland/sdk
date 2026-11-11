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
    SweepResults,
)
from iqm.station_control.interface.models.jobs_new import JobType, TimelineEntry


def _run_data_to_timeline(run_data: RunData) -> list[TimelineEntry]:
    """Build a best-effort approximate run job timeline using the legacy timestamps in RunData."""
    timeline = [
        TimelineEntry(source="iqm-server", status="created", timestamp=run_data.created_timestamp),
        TimelineEntry(source="iqm-station-control", status="received", timestamp=run_data.begin_timestamp),
    ]
    if run_data.end_timestamp is not None:
        timeline.append(TimelineEntry(source="iqm-station-control", status="ready", timestamp=run_data.end_timestamp))
    timeline.append(TimelineEntry(source="iqm-server", status="completed", timestamp=run_data.modified_timestamp))
    return timeline


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

            self._context["job_timeline"] = tuple(self.job_data.timeline)
            self._context["run_definition"] = self.payload()
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

            run_definition = self.payload()
            # TODO (Marko): Check performance for this or if there is any better way to do this?
            #  The client sends the "settings" with empty paths. In the current implementation,
            #  station control adds the paths during deserialization, but with IQM Server it will respond
            #  with the payload as it was sent in the request, i.e. paths are still empty here.
            #  Should we just change the logic so that client is responsible settings the paths correctly
            #  already before making the request?
            if run_definition.sweep_definition.settings:
                run_definition.sweep_definition.settings._generate_paths_and_names()

            if compiler is None:
                # FIXME: Deferred import to break cyclic dependency
                from iqm.cpc.core.run_result import construct_run_result  # noqa: PLC0415

                self._result_exa = construct_run_result(run_definition, sweep_results, self.job_data.timeline)
            else:
                # For EXA, for now the post-processing context cannot contain anything that cannot
                # be derived from the run definition
                # Cannot cache result since it depends on ``compiler``.
                self._context["job_timeline"] = tuple(self.job_data.timeline)
                self._context["run_definition"] = run_definition
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
