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
"""Wrap error-handling for experiment and sequence analysis."""

from __future__ import annotations

from traceback import format_exc

from exa.core.analyze_result import AnalyzeResult
from exa.core.experiment import Experiment
from exa.core.experiment.experiment import error_analyze

from .interface import CustomExperimentAnalysis
from .result import SequenceAnalyzeResult, SequenceExecuteResult


def wrap_experiment_analyze(
    experiment: Experiment, custom_analysis: CustomExperimentAnalysis | None = None, **kwargs
) -> AnalyzeResult:
    """Wraps :meth:`.Experiment.analyze` by recording any caught exception detail into the result.

    In the event of an exception, a form of stub analysis is performed. This merely conveys the latest run results to
    the returned analyze result in order to have the data available. Additionally, the full exception traceback is
    stored in the result.

    Args:
        experiment: the experiment instance to call `.analyze` on.
        custom_analysis: Custom analysis function to call for the experiment instead of :meth:`.Experiment.analyze`.
            If not provided, will call ``experiment.analyze(**kwargs)``.
        kwargs: keyword arguments to pass to `.analyze`.

    Returns:
        An instance of :class:`.AnalyzeResult`. Any error detail is recorded to the returned instance.

    """
    try:
        analyze_result = custom_analysis(experiment, **kwargs) if custom_analysis else experiment.analyze(**kwargs)
    except Exception:
        analyze_result = error_analyze(experiment)
        analyze_result.error = (
            f"Error in {str(experiment.last_run.run_definition.run_id)} "
            f"`{experiment.__class__.__name__}.analyze({kwargs})`. "
            "Any changes to the dataset made in the analysis until this point are propagated to the analyze result "
            "as is."
        )
        analyze_result.error_traceback = format_exc()
    return analyze_result


def wrap_sequence_analyze(sequence_execute_result: SequenceExecuteResult, **kwargs) -> SequenceAnalyzeResult:
    """Wraps :meth:`.SequenceExecuteResult.analyze` by recording any caught exception detail into the result.

    In the event of an exception, the real analysis is skipped and the latest sequence execution results are merely
    conveyed to the returned analyze result in order to have the data available.
    Additionally, the full exception traceback is stored in the result.

    Args:
        sequence_execute_result: the sequence_execute_result instance to call `.analyze` on.
        kwargs: keyword arguments to pass to `.analyze`.

    Returns:
        An instance of :class:`.sequence.result.SequenceAnalyzeResult`.
        Any error detail is recorded to the returned instance.

    """
    try:
        result = sequence_execute_result.analyze(**kwargs)
    except Exception:
        result = SequenceAnalyzeResult(execution_result=sequence_execute_result)
        result.error = (
            f"Error in {str(sequence_execute_result.context.sequence_run_id)} "
            f"`{sequence_execute_result.__class__.__name__}.analyze({kwargs})`. "
            f"Sequence execution data is merely propagated to the analyze result as is."
        )
        result.error_traceback = format_exc()
    return result
