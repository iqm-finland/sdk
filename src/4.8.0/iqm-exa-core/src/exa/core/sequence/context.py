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
"""Context for handling the execution state of multiple experiments."""

from __future__ import annotations

import inspect
import logging
from traceback import format_exc
from typing import Any, Protocol
import uuid

from exa.core import Experiment
from exa.core.analyze_result import AnalyzeResult
from exa.core.observation import LatestFromStash, ObservationStash
import exa.core.sequence.data_access as serialize

from .experiment_tracker import ExperimentTracker
from .interface import CustomExperimentAnalysis, NodeExecution, SequenceContextData, SequenceData
from .result import SequenceAnalyzeResult, SequenceExecuteResult
from .sequence_executor import SequenceExecutor
from .util import update_options
from .wrap_analyze import wrap_sequence_analyze

SAVE_STASH = "save_stash"
IGNORE_ANALYZE_ERRORS = "ignore_analyze_errors"

ATTRIBUTE_OPTIONS = [SAVE_STASH, IGNORE_ANALYZE_ERRORS]
"""Sequence attributes that are definable as options and change the behaviour of the context."""

logger = logging.getLogger(__name__)


class _Sequence(Protocol):
    """A Sequence.

    This is just a dummy class to overcome circular imports.
    The actual class is :class:`exa.sequence.core.sequence.Sequence`.
    """

    ignore_analyze_errors: bool | None = None
    save_stash: bool = True
    observation_stash: ObservationStash | None = None
    options: dict[str, Any] = {}

    def execute(self) -> SequenceExecuteResult:
        """Execute."""

    def set_attribute_options(self) -> None:
        """Set attribute options."""


class SequenceContext:
    """A stateful object that provides the tools for Sequence execution and captures the inputs and outputs.

    SequenceContext is a context object that captures the run IDs, analysis results,
    and analysis arguments for each :class:`Experiment` or :class:`Sequence` execution done within the context.

    A user does not need to construct a SequenceContext. The intended way is to use it as a context manager,
    see :meth:`.Sequence.context`.

    If exceptions are caught in ``execute_sequence`` and ``execute_experiment``, error summary and traceback are stored
    in results. Execution errors are stored in sequence result data which will be stored to persistence.
    On the contrary, analysis errors are stored only locally in analyze results.

    Logging of errors is handled exclusively in ``__exit__``.

    Args:
        data: The serializable part of the Sequence that will be executed with this context.
        tracker_class: Optional subclass of :class:`.ExperimentTracker`.
            This class will be used when creating a tracker object with :meth:`track`.
        observation_stash: The stash to save observations to instead of the DB.
        sequence_executer_class: Will be used when executing a sequence.

    """

    sub_stashes: dict[uuid.UUID, ObservationStash | None]
    """Stores the observation stashes of any subsequences executed in this context
    (mapped to the ``sequence_run_id``)."""

    def __init__(
        self,
        data: SequenceData,
        tracker_class: type[ExperimentTracker] = ExperimentTracker,
        observation_stash: ObservationStash | None = None,
        attribute_options: dict[str, Any] | None = None,
        sequence_executor_class: type[SequenceExecutor] = SequenceExecutor,
        sequence_run_id: uuid.UUID | None = None,
    ):
        self._data = SequenceContextData(
            sequence=data, sequence_run_id=sequence_run_id or uuid.uuid4(), tags=data.options.get("tags", [])
        )
        self._analyze_results: list[AnalyzeResult | SequenceAnalyzeResult] = []
        self._tracker_class = tracker_class
        self._exited_context = False
        self._attribute_options = attribute_options or {}
        self._observation_stash = observation_stash
        self._sequence_executor_class = sequence_executor_class
        self.sub_stashes = {}

    def __enter__(self):
        if self._exited_context:
            raise self.InternalError(
                "Context was already entered and closed before. Reinitialize the sequence to avoid data duplication."
            )

        serialize.save_sequence_metadata(self._data)
        logger.info("Executing %s, %s", self._data.sequence.name, self._data.sequence_run_id)
        self._analyze_results.clear()
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback) -> bool:  # noqa: ANN001
        # Complicated error logging is handled here in exiting the context manager.
        if exception_type is None:
            logger.info("%s completed successfully", self._data.sequence.name)
        elif exception_type is self.InternalError:
            # Propagate exception on special internal error
            return False
        elif exception_type is self.AnalyzeError:
            # Identify analysis-related exceptions here.
            if not self._analyze_results or sum(res.error is not None for res in self._analyze_results) > 1:
                raise self.InternalError(
                    f"No errors found in analyze results when {self.AnalyzeError.__name__} is caught in exiting the "
                    "context. This should not happen."
                )
            last_result = self._analyze_results[-1]

            # Log the last analyze error only in one context
            if last_result.error:
                logger.error("%s\n%s", last_result.error, last_result.error_traceback)
            if (
                isinstance(last_result, SequenceAnalyzeResult)
                and last_result.execution_result.get_analyze_result_errors()
            ):
                logger.error("Caught exception inside subsequence analysis.")

            logger.info("%s partially completed with errors", self._data.sequence.name)
        else:
            # Catch any other exception here, also KeyboardInterrupt.
            # These exceptions will be added to the persisted sequence result data.
            if self._data.executed_nodes:
                last_node = self._data.executed_nodes[-1]
                describe_node = f"Last executed node was {str(last_node.run_id)} ({last_node.type})"
            else:
                describe_node = "No nodes were executed"
            self._data.error = f"Exception caught during execution: {repr(exception_value)}. {describe_node}."
            self._data.error_traceback = format_exc()
            logger.error("%s\n\n%s", self._data.error, self._data.error_traceback)
            logger.info("%s partially completed with errors", self._data.sequence.name)

        self._exited_context = True
        # Do not propagate exceptions because we want `to_result()` to be called in the end to save current results
        return True

    def __repr__(self) -> str:
        return f"SequenceContext(tracker_class={self._tracker_class}, data={self._data})"

    def execute_experiment(
        self,
        experiment: Experiment | type[Experiment],
        custom_analysis: CustomExperimentAnalysis | None = None,
        run_args: dict[str, Any] | None = None,
        analyze_args: dict[str, Any] | None = None,
        observation_stash: ObservationStash | None = None,
    ) -> AnalyzeResult:
        """Run a single experiment and its analysis.

        This includes:

        - running the experiment on hardware
        - analyzing the execution results with given arguments
        - tagging the observations from the analysis. Note: the ``tags`` entry of :attr:`.Sequence.options` will be
          used for tagging, if any.
        - saving the observations.

        Under the hood, the Experiment is wrapped with :attr:`tracker_class`, which means that a tracker class can be
        used to customize the behavior without calling :meth:`track` explicitly.

        Args:
            experiment: The experiment to execute.
                Can be either an instance or a class, in which case a new instance is created.
            custom_analysis: Custom analysis function for the experiment.
            run_args: Arguments to pass to the ``experiment.run``, most commonly at least the list of components.
            analyze_args: Arguments to pass to the ``experiment.analyze``.
            observation_stash: The stash to save observations to instead of the DB. If not provided, will use
                the stash in ``self._observation_stash``, and if that is empty as well, the observations will
                be saved to the DB.

        Returns:
             The result of the ``experiment.analyze``.

        """
        run_args = run_args or {}
        analyze_args = analyze_args or {}
        if observation_stash is None:
            observation_stash = self._observation_stash

        tracker = self.track(experiment, observation_stash)
        update_options(experiment, self._data.sequence.options)  # type: ignore[arg-type]

        ignore_analyze_errors = self._resolve_attribute_option_for(tracker, IGNORE_ANALYZE_ERRORS)

        logger.debug("Executing %s", tracker.label)
        tracker.run(**run_args)
        tracker.analyze(custom_analysis=custom_analysis, **analyze_args)
        analyze_result = self.end_tracking(tracker)

        if analyze_result.error is None or ignore_analyze_errors:
            try:
                self._save_observations(analyze_result, observation_stash)  # type: ignore[arg-type]
            except Exception as err:
                run_id = self._data.executed_nodes[-1].run_id
                analyze_result.error = (
                    f"Error in handling observations of the analyze result of experiment {str(run_id)}: {str(err)}"
                )
                analyze_result.error_traceback = format_exc()

        if analyze_result.error is not None and not ignore_analyze_errors:
            self._raise_analyze_error()

        return analyze_result

    def execute_sequence(
        self, sequence: _Sequence, analyze_args: dict[str, Any] | None = None
    ) -> SequenceAnalyzeResult:
        """Execute and analyze another sequence in this context.

        The analyze_results of each Experiment or sub-sequence within `sequence` are captured.

        If ``sequence`` does not have a stash and the context has ``self._observation_stash`` populated, will copy
        the contents of ``self._observation_stash`` and run ``sequence`` on that stash. At the end of execution,
        the stash of ``sequence`` will be saved (i.e. either merged to ``self._observation_stash`` or if that does not
        exist, saved to the DB) based on the value of the attribute option ``save_stash``.

        Args:
            sequence: The sequence instance to execute.
            analyze_args: Arguments to pass to the ``sequence.analyze``.

        Returns:
            The analyzed result of executing `sequence`.

        """
        update_options(sequence, self._data.sequence.options)
        self._propagate_attribute_options(sequence, [IGNORE_ANALYZE_ERRORS])
        sequence.set_attribute_options()

        save_stash = sequence.save_stash
        ignore_analyze_errors = sequence.ignore_analyze_errors

        analyze_args = analyze_args or {}

        if self._observation_stash and not sequence.observation_stash:
            sequence.observation_stash = self._observation_stash.copy()

        executor = self._sequence_executor_class(sequence)
        execute_result = executor.execute()
        if execute_result.context.error:
            raise self.ChildSequenceExecuteError(
                f"Error executing child sequence {execute_result.context.sequence_run_id}. "
                f"{execute_result.context.error}"
            )

        child_analyze_errors = execute_result.get_analyze_result_errors()
        if child_analyze_errors and not ignore_analyze_errors:
            # If analysis errors happened in the context of the child sequence, do not proceed to analyze the sequence.
            # Instead, we pass the execution results forward.
            analyze_result = SequenceAnalyzeResult(execution_result=execute_result)
        else:
            analyze_result = wrap_sequence_analyze(execute_result, **analyze_args)

        self._data.executed_nodes.append(
            NodeExecution(
                run_id=analyze_result.execution_result.context.sequence_run_id,
                analyze_args=analyze_args,
                type="sequence",
            )
        )
        self._analyze_results.append(analyze_result)

        if (child_analyze_errors or analyze_result.error is not None) and not ignore_analyze_errors:
            self._raise_analyze_error()
        if analyze_result.error is None:
            self._save_observations(analyze_result, observation_stash=sequence.observation_stash)
            if save_stash:
                self._merge_subsequence_stash(sequence.observation_stash)

        self.sub_stashes[execute_result.context.sequence_run_id] = sequence.observation_stash

        return analyze_result

    def to_result(self) -> SequenceExecuteResult:
        """Converts the captured execution results into a :class:`SequenceExecuteResult` data object.

        As a side effect, the final result of this sequence is saved to persistence. The result is marked final after
        which further changes to it are prevented. Thus, this method is callable only once per object.

        Should be called at the end of :meth:`.Sequence.execute`.

        Returns:
            A new instance of SequenceExecuteResult.

        """
        if not self._exited_context:
            raise self.InternalError(
                "SequenceExecuteResult.to_result can only be called after exiting the ``with`` context."
            )
        final_result = self._save_current_result(final=True)
        return final_result

    def track(
        self,
        experiment: Experiment | type[Experiment] | ExperimentTracker,
        observation_stash: ObservationStash | None = None,
    ) -> ExperimentTracker:
        """Create a tracker object that wraps an Experiment instance and captures the actions done with it.

        As an alternative to :meth:`execute_experiment`, the tracker allows a more fluent workflow with the
        experiment instance.
        The tracker behaves like a normal Experiment.
        The use cases are, for example, tuning settings, sweep ranges, and options before running.

        The tracker should only be used for one round of ``run`` & ``analyze``. After this, :meth:`end_tracking`
        should be called.

        Args:
            experiment: An Experiment instance or an Experiment class. In case of class, a new instance is created.
            observation_stash: The stash to save observations to instead of the DB. If not provided, the observations
            will be saved to the DB.

        Returns:
             A new instance of ExperimentTracker. The created class is determined be :attr:`tracker_class`.

        """
        if isinstance(experiment, ExperimentTracker):
            logger.warning(
                "Can't retrack an already tracked experiment."
                " This may be caused by an experiment being explicitly tracked inside of a sequence before executing"
                " within sequence context, and / or by not calling ´end_tracking()´. Returning the previous tracker."
            )
            return experiment
        if inspect.isclass(experiment):
            experiment = experiment()
        if observation_stash:
            experiment.observation_handler.change_load_rules([LatestFromStash(observation_stash)])  # type: ignore[union-attr,arg-type]
        return self._tracker_class(experiment)  # type: ignore[arg-type]

    def end_tracking(self, tracker: ExperimentTracker) -> AnalyzeResult:
        """Get the captured data from an ExperimentTracker instance and store it to the context.

        This should be called after performing the desired ``run`` and ``analyze`` actions with the
        tracker, so that the results and used arguments are stored.

        Args:
            tracker: the instance to get data from.

        Returns:
            The AnalyzeResult captured by the tracker, if any.

        """
        node, analyze_result = tracker.tracked_content()
        self._data.executed_nodes.append(node)
        self._analyze_results.append(analyze_result)
        self._save_current_result(final=False)
        if tracker.custom_analysis:
            self._data.custom_experiment_analyses[str(node.run_id)] = tracker.custom_analysis
        return analyze_result

    def _propagate_attribute_options(self, subsequence: _Sequence, options_to_propagate: list[str]):  # noqa: ANN202
        """Propagate attribute options to subsequences."""
        attribute_options = {o: v for o, v in self._attribute_options.items() if o in options_to_propagate}
        for option, value in attribute_options.items():
            if hasattr(subsequence, option) and value is not None:
                setattr(subsequence, option, value)

    def _resolve_attribute_option_for(self, tracker: ExperimentTracker, attribute_name: str) -> Any:
        """Resolve attribute option for an experiment."""
        exp_name = tracker.experiment_class_name
        attr_option_name = f"{exp_name}_{attribute_name}"
        if attr_option_name in self._data.sequence.options:
            return self._data.sequence.options[attr_option_name]
        return self._attribute_options.get(attribute_name, None)

    def _save_current_result(self, final: bool) -> SequenceExecuteResult:
        """Converts the execution results that have been captured so far into a :class:`SequenceExecuteResult` data
        object and saves it to the database.
        """
        result = SequenceExecuteResult(
            context=self._data,
            analyze_results=self._analyze_results,  # type: ignore[arg-type]
        )
        serialize.save_sequence_result(result, final=final)
        return result

    def _save_observations(
        self, analyze_result: SequenceAnalyzeResult, observation_stash: ObservationStash | None = None
    ) -> None:
        observations = analyze_result.observations.validate().tag(self._data.tags)  # type: ignore[union-attr]
        if observation_stash:
            observations.stash(observation_stash)
        else:
            observations.save()

    def _merge_subsequence_stash(self, subsequence_stash: ObservationStash | None):  # noqa: ANN202
        if not subsequence_stash:
            return
        if self._observation_stash:
            subsequence_stash.merge_to(self._observation_stash)
        else:
            subsequence_stash.save()

    def _raise_analyze_error(self) -> None:
        for result in reversed(self._analyze_results):
            if result.error is not None:
                raise self.AnalyzeError(self._format_error(result))
            children = getattr(getattr(result, "execution_result", None), "analyze_results", None) or []
            for child in reversed(children):
                if child.error is not None:
                    raise self.AnalyzeError(self._format_error(child))
        raise self.AnalyzeError("Analysis failed but no error details were found in results.")

    @staticmethod
    def _format_error(result: AnalyzeResult | SequenceAnalyzeResult) -> str:
        msg = str(result.error)
        tb = getattr(result, "error_traceback", "") or ""
        if last := tb.rstrip().rsplit("\n", 1)[-1].strip():
            return f"{msg}. {last}"

        return msg

    class AnalyzeError(Exception):
        """Internal helper type for propagating analysis errors.

        Analysis error can happen:

        - subsequence or experiment analysis in ``execute_sequence`` method inside ``sequence.execute``.
        - in sequence analysis in ``execute_sequence`` method.
        """

    class ChildSequenceExecuteError(Exception):
        """Internal helper type for propagating child sequence execution errors to upper level context."""

    class InternalError(Exception):
        """Internal helper type for propagating internal sequence execution errors to upper level context."""
