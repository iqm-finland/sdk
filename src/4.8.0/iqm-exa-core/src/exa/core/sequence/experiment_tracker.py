#  ********************************************************************************
#    _____  ____ _
#   / _ \ \/ / _` |  Framework for control
#  |  __/>  < (_| |  and measurement of
#   \___/_/\_\__,_|  superconducting qubits
#
#  Copyright 2019-2026 IQM Finland Oy
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ********************************************************************************
"""Proxy class that wraps Experiment and captures its output."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
import uuid

from exa.core import Experiment
from exa.core.analyze_result import AnalyzeResult

from .interface import CustomExperimentAnalysis, NodeExecution
from .wrap_analyze import wrap_experiment_analyze


class ExperimentTracker:
    """Wraps an instance of :class:`.Experiment` and tracks the run IDs and analysis.

    ExperimentTracker behaves like a normal Experiment, but also keeps track of run IDs generated with :meth:`run` and
    arguments given to :meth:`analyze`.

    The ExperimentTracker is provided so that users may subclass it to customize the behavior of :meth:`run` and
    :meth:`analyze`. See :mod:`exa.sequence.definitions.trackers` for examples.

    Note:
        The ExperimentTracker is only meant to be used for one round of run-analyze. To do multiple rounds,
        use a new ExperimentTracker object. Otherwise, only the latest results are tracked.

    Args:
         experiment: An instance of Experiment that is wrapped.

    """

    custom_analysis: CustomExperimentAnalysis | None = None
    """Custom analysis function to call for the experiment instead of the native analyze method."""

    def __init__(self, experiment: Experiment):
        self._experiment = experiment
        self._plot_method_args: list[dict[str, Any]] = []
        self._analyze_result: AnalyzeResult | None = None
        self._analyze_args: dict[str, Any] = {}
        self._run_id: uuid.UUID | None = None

    def __getattr__(self, item: str) -> Any:
        return getattr(self._experiment, item)

    def __repr__(self) -> str:
        return f"ExperimentTracker for {self._experiment.label} (experiment={self._experiment})"

    @property
    def experiment_class_name(self) -> str:
        """Returns the class name of the experiment."""
        return self._experiment.__class__.__name__

    def run(self, *args, **kwargs) -> ExperimentTracker:
        """Wraps :meth:`.Experiment.run` and captures the resulting run ID."""
        run_result = self._experiment.run(*args, **kwargs).last_run
        self._run_id = run_result.run_definition.run_id
        return self

    def analyze(self, custom_analysis: Callable[..., AnalyzeResult] | None = None, **kwargs) -> AnalyzeResult:
        """Wraps :experiment analyze and captures the used arguments and resulting :class:`AnalyzeResult`.

        Args:
            custom_analysis: Custom analysis function to call for the experiment instead of :meth:`.Experiment.analyze`.
            kwargs: Additional arguments for the analysis function.

        Returns:
            The analyze result.

        """
        analyze_result = wrap_experiment_analyze(self._experiment, custom_analysis=custom_analysis, **kwargs)
        self._analyze_args = kwargs
        self._analyze_result = analyze_result
        self.custom_analysis = custom_analysis
        return analyze_result

    def tracked_content(self) -> tuple[NodeExecution, AnalyzeResult]:
        """Returns the captured content.

        Returns:
            The captured result of analyze, and other data needed to reconstruct the analysis.

        """
        if self._run_id is None:
            raise RuntimeError(
                f"{self._experiment.name}.run should be called before ending experiment tracking, but it wasn't called."
            )
        if self._analyze_result is None:
            self._analyze_result = AnalyzeResult(run_result=self._experiment.last_run)
        other = NodeExecution(run_id=self._run_id, analyze_args=self._analyze_args, type="experiment")
        return other, self._analyze_result
