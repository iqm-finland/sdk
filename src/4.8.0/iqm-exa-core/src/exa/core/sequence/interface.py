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
"""Data classes and protocols that define the Sequence framework."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import importlib
from typing import Annotated, Any, Literal, Protocol, TypeAlias
import uuid

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FieldSerializationInfo,
    PlainSerializer,
    PlainValidator,
    field_serializer,
    field_validator,
)
import xarray as xr

from exa.common.data import ObservationValue, serialize_value, validate_value
from exa.core.analyze_result import AnalyzeResult, PlotDefinition
from exa.core.observation import LocalObservationSequence

SERIALIZER_VERSION = 2
"""Version of the serialization protocol. It determines if a serialized Sequence result is compatible with this version
of the code. That is, when loading back and reconstructing a Sequence result, the version has to be equal to the version
which the Sequence was executed with."""
# Changing this version breaks backwards and forwards compatibility of loading results.
# I.e. from that point on, older results are not loadable, and older versions of the code cannot load these results.


def _fqn_import(fqn: str) -> type:
    module_name, _, function_name = fqn.rpartition(".")

    # Workaround for backwards compatible module name import after exa-core split.
    module_name = module_name.replace("exa.sequence.core", "exa.core.sequence")
    module_name = module_name.replace("exa.sequence.definitions", "exa.experiments.sequences")

    return getattr(importlib.import_module(module_name), function_name)


def _fully_qualified_name(type_: type) -> str:
    return f"{type_.__module__}.{type_.__name__}"


NestedValue: TypeAlias = None | ObservationValue | tuple | list | dict[str, Any]
"""Types we can serialize to JSON, possibly using our custom serializers."""
JSONSafeScalar: TypeAlias = None | str | bool | int | float
JSONSafeValue: TypeAlias = JSONSafeScalar | tuple | list | dict[str, Any]
"""Corresponding trivially JSON-serializable types."""


def _serialize_nested_value(value: NestedValue) -> JSONSafeValue:
    """Recursively serialize possibly nested values into JSON-safe payloads."""
    if isinstance(value, dict):
        return {key: _serialize_nested_value(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_nested_value(inner) for inner in value]
    # scalar serializer
    return serialize_value(value)


def _deserialize_nested_value(value: JSONSafeValue) -> NestedValue:
    """Recursively deserialize possibly nested values from serialized payloads."""
    if isinstance(value, dict):
        decoded = validate_value(value)
        if decoded is not value:
            # break recursion, if validate changed value (successful decode)
            return decoded
        return {key: _deserialize_nested_value(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_deserialize_nested_value(inner) for inner in value]
    return value


class SequenceAnalyzeResult(Protocol):
    """The analyzed end result of a Sequence.

    This is just a dummy class to overcome circular references.
    The actual class is :class:`exa.sequence.core.result.SequenceAnalyzeResult`.
    """


class SequenceExecuteResult(Protocol):
    """The result of the execution of a Sequence.

    This is just a dummy class to overcome circular imports.
    The actual class is :class:`exa.sequence.core.result.SequenceExecuteResult`.
    """


Analysis = Callable[..., SequenceAnalyzeResult]
"""Protocol that any analysis function used with Sequences should implement.

The callable should take a :class:`SequenceResult` as the first argument and return
a :class:`SequenceAnalyzeResult`.
"""

CustomExperimentAnalysis = Callable[..., AnalyzeResult]
"""Protocol that a custom analysis function for an experiment node should implement.

The callable should take a :class:`.Experiment` instance as the first argument and return
a class:`.AnalyzeResult`.
"""


class SequenceData(BaseModel):
    """The serializable part of a :class:`.Sequence`."""

    name: str
    """Name of the sequence."""
    analysis: Analysis
    """Callable that analyzes the sequence-level :class:`SequenceExecuteResultData`."""
    options: Annotated[
        dict[str, Any],
        PlainSerializer(_serialize_nested_value),
        PlainValidator(_deserialize_nested_value),
    ] = {}
    """Arbitrary options to control sequence behavior. Must contain only basic Python types."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __repr__(self) -> str:
        header = [
            self.__class__.__name__,
            f".name: {self.name}",
            f".analysis: {self.analysis.__name__}",
            ".options:",
        ]
        options = [f"\t'{key}': {value}" for key, value in self.options.items()] or ["{}"]
        options.append("\n")
        return "\n".join(header + options)

    @field_serializer("analysis")
    def serialize_analysis(self, value: Any, _info: FieldSerializationInfo) -> str:
        """Serialize analysis."""
        return _fully_qualified_name(value)

    @field_validator("analysis", mode="before")
    @classmethod
    def validate_analysis(cls, value: Any):  # noqa: ANN001, ANN206
        """Cast str fqdn to type."""
        if isinstance(value, str):
            return _fqn_import(value)
        return value


class NodeExecution(BaseModel):
    """Captured data of an executed Experiment or Sequence, for traceability purposes."""

    run_id: uuid.UUID
    """ID of the run or execution."""
    analyze_args: dict[str, Any]
    """Keyword arguments used in the analyze method of the Experiment or Sequence."""
    type: Literal["experiment", "sequence"]
    """Type of the executed node. Either ``"experiment"`` or ``"sequence"``."""

    model_config = ConfigDict(arbitrary_types_allowed=True)


class SequenceContextData(BaseModel):
    """Data representation of the execution context."""

    sequence: SequenceData
    """Sequence that used this context."""
    sequence_run_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    """Unique ID of the execution."""
    executed_nodes: list[NodeExecution] = []
    """List of run IDs and analysis args of individual executed nodes."""
    tags: list[str] = []
    """Tags used to tag any observations during execution and the resulting final analysis."""
    custom_experiment_analyses: dict[str, CustomExperimentAnalysis] = {}
    """Custom analyze functions to call instead of the normal experiment analysis."""
    error: str | None = None
    """Optional string to store the summary of a possible error during execution."""
    error_traceback: str | None = None
    """Optional string to store the traceback of a possible error during execution."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_serializer("custom_experiment_analyses")
    def serialize_custom_experiment_analyses(self, value: Any, _info: FieldSerializationInfo) -> dict[str, str]:
        """Serialize custom experiment analyses."""
        return {key: _fully_qualified_name(value_) for key, value_ in value.items()}

    @field_validator("custom_experiment_analyses", mode="before")
    @classmethod
    def validate_custom_experiment_analyses(cls, value):  # noqa: ANN001, ANN206
        """Cast str fqdn to type."""
        analyses = {}
        for idn, func in value.items():
            if isinstance(func, str):
                func = _fqn_import(func)  # noqa: PLW2901
            analyses[idn] = func
        return analyses


class SequenceExecuteResultDataModel(BaseModel):
    """Serialization model for SequenceExecuteResultData."""

    context: SequenceContextData
    """Execution context data of the sequence."""
    version: int = SERIALIZER_VERSION
    """Version of the serialized format."""

    model_config = ConfigDict(arbitrary_types_allowed=True)


@dataclass
class SequenceExecuteResultData:
    """Contains the results of all nodes executed within a sequence, and the context information."""

    context: SequenceContextData
    """Data of the context in which this result was gathered."""
    analyze_results: list[AnalyzeResult | SequenceAnalyzeResult]
    """List of analysis results from individual executed nodes."""

    def get_analyze_result_errors(self) -> list[str]:
        """Compile a list of error summary strings inside ``analyze_results``."""
        return [res.error for res in self.analyze_results if res.error is not None]  # type:ignore[union-attr]

    def __repr__(self) -> str:
        analyze_errors = self.get_analyze_result_errors()
        analyze_results_error_str = (
            f" Of which {len(analyze_errors)} had errors during analyze." if analyze_errors else ""
        )
        execution_error_str = f" (Execution error: {self.context.error})" if self.context.error else ""
        return f"""{self.__class__.__name__}:
.analyze_results: {len(self.analyze_results)} items.{analyze_results_error_str}
.context: {self.context.__class__.__name__} of {self.context.sequence.name} {self.context.sequence_run_id}\
{execution_error_str}
"""


@dataclass(repr=False)
class SequenceAnalyzeResultData:
    """The analyzed end result of a Sequence.

    Analogous to the :class:`AnalyzeResult` of Experiments. Contains the dataset, observations, and plots from the
    sequence-level analysis, as well as the execution result where it was all derived from.
    """

    execution_result: SequenceExecuteResultData
    """Results of the whole sequence, including analysis of the individual experiments."""
    dataset: xr.Dataset = field(default_factory=xr.Dataset)
    """Dataset representing the whole sequence. Empty by default."""
    observations: LocalObservationSequence | None = field(default_factory=LocalObservationSequence)
    """Observations made by analyzing the whole sequence. Empty by default."""
    plots: dict[str, PlotDefinition] = field(default_factory=dict)
    """Generators for plotters. Empty by default."""
    experiment_template: str | None = None
    """Template to use when displaying an experiment report."""
    sequence_template: str | None = None
    """Template for aggregated report containing sequence and experiment."""
    error: str | None = None
    """Optional string to store the summary of a possible error during analysis."""
    error_traceback: str | None = None
    """Optional string to store the traceback of a possible error during analysis."""

    _all_errors: list[str] | None = None

    def __repr__(self):
        ret = f"""SequenceAnalyzeResult
-- .observations --
{self.observations}

-- .plots --
{self.plots.keys()}

-- .dataset --
{self.dataset}

-- .execution_result --
{self.execution_result}
"""
        if self.error is not None:
            ret += f"""
-- .error --
    {self.error}
"""
        return ret

    def get_errors(self) -> list[str]:
        """Convenience method to recursively collect and store all error summary strings in this structure."""
        if self._all_errors is None:

            def find_errors(
                result: AnalyzeResult | SequenceAnalyzeResultData,
            ) -> list[str]:
                found_errors = []
                if result.error:
                    found_errors.append(result.error)
                if isinstance(result, SequenceAnalyzeResultData):
                    if result.execution_result.context.error:
                        found_errors.append(result.execution_result.context.error)
                    for child_analyze_result in result.execution_result.analyze_results:
                        found_errors += find_errors(child_analyze_result)  # type: ignore[arg-type]
                return found_errors

            self._all_errors = find_errors(self)
        return self._all_errors
