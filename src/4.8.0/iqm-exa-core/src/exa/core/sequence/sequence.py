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
"""Base class for Sequences. Sequences are defined by subclassing this."""

from abc import ABC, abstractmethod
from copy import deepcopy
import json
from typing import Any
import uuid

from pydantic_core import PydanticSerializationError

from exa.core.observation import ObservationStash

from .context import ATTRIBUTE_OPTIONS, SequenceContext
from .experiment_tracker import ExperimentTracker
from .interface import Analysis, SequenceData
from .result import SequenceAnalyzeResult, SequenceExecuteResult
from .sequence_executor import SequenceExecutor


def default_analysis(result: SequenceExecuteResult, **kwargs) -> SequenceAnalyzeResult:
    """Default Analysis that only converts :class:`~exa.sequence.core.result.SequenceExecuteResult` into
    :class:`~exa.sequence.core.result.SequenceAnalyzeResult`.

    Args:
        result: Data from execution.
        kwargs: Not used.

    Returns:
        Same data wrapped as an analysis result.

    """
    return SequenceAnalyzeResult(execution_result=result)


class Sequence(ABC):
    """Base class for Sequences. New sequences are defined by subclassing this.

    A subclass of Sequence should at least define ``name`` as a class attribute, and implement
    :meth:`execute`, which is the main body of the sequence execution.

    Args:
        observation_stash: If provided, the observations created by the execution nodes will not
            be saved to the DB but to the given stash instead. In this case, any subsequences will
            by default copy that stash, add their observations to the copy, and only at the end of
            the subsequence execution, save them to the parent's stash. The user may turn off the merging
            of stashes via the attribute ``save_stash`` and/or use a different stash (or no stash)
            for a given execution node in the actual implementation of :meth:`execute`.
        kwargs: All keyword arguments to the constructor are copied to :attr:`options` attribute.
            The :meth:`execute` can refer to the options to modify the behavior of the execution.
            The meaning of the options depends entirely on how they are used by :meth:`execute`.
            The user may apply options to specific execution nodes via prefixing them with the node's class names.
            E.g. to apply option ``foo = 1`` for experiment class ``MyExperiment`` that is a part of the
            subsequence ``MySequence``, the user should add ``MySequence_MyExperiment_foo = 1`` to the kwargs.

    """

    name: str
    """Identifier of the sequence type."""
    analysis: Analysis = staticmethod(default_analysis)
    """Function that performs sequence-level analysis based on the results of the executed nodes.
    It must take a :class:`SequenceResult` as the first argument and return a :class:`SequenceAnalyzeResult`.
    Other arguments are optional.
    `analysis` should be given as a class attribute, decorated with ``staticmethod``.
    If a subclass does not define :attr:`analysis`, :func:`.default_analysis` is used."""
    default_options: dict[str, Any] = {}
    """Required keys for the sequence :attr:`options` and their default values."""
    tracker_class: type[ExperimentTracker] = ExperimentTracker
    """:class:`.ExperimentTracker` or a subclass of it.
    This class will be used when creating a tracker object in the execution context with :meth:`.track`.
    Giving your own tracker class allows modifying the execution of all experiments, for example by wrapping
    the run logic with custom conditions."""
    sequence_executor_class: type[SequenceExecutor] = SequenceExecutor
    """:class:`.SequenceExecutor` or a subclass of it.
    This class will be used when executing a sequence, and can be used to modify the sequence execution."""
    save_stash: bool = True
    """Whether to save this sequence's stash to the parent sequence's stash (or the DB if the parent does not have
    stash). This attribute has an effect only if set at the subsequence level, and effect will not propagate to
    further subsequences. This attribute is also settable by including ``save_stash`` in the sequence options."""
    ignore_analyze_errors: bool | None = None
    """Ignore analyze errors for this sequence and its subsequences. The observations and other analyze results will
    be retained if they were created successfully, but any analyze errors will still be shown in the report. If set,
    as `True` or `False`, the effect will propagate recursively to all subsequences and their experiments.
    This attribute is also settable by including ``ignore_analyze_errors`` in the sequence options."""

    def __init__(self, observation_stash: ObservationStash | None = None, **kwargs):
        self.observation_stash = observation_stash
        self.options: dict[str, Any] = deepcopy(self.default_options | kwargs)
        """Options that modify the execution. The values are either class-level defaults from :attr:`default_options`
        or kwargs given in the constructor. Options must contain only basic Python types."""
        self.set_attribute_options()

        if not hasattr(self, "name"):
            raise NotImplementedError(
                f"Class {self.__class__} is missing a name attribute. "
                f"Add it as a class variable when defining the Sequence subclass."
            )
        try:
            json.loads(SequenceData(name=self.name, analysis=self.analysis, options=self.options).model_dump_json())
        except (TypeError, PydanticSerializationError) as err:
            raise TypeError(
                "Only JSON-serializable option types are allowed in Sequence options. "
                "NumPy arrays and complex values are supported. "
                f"{err}."
            ) from err

    def __repr__(self) -> str:
        header = [
            self.__class__.__name__,
            f".name: {self.name}",
            f".analysis: {self.analysis.__name__}",
            ".options:",
        ]
        options = [f"\t'{key}': {value}" for key, value in self.options.items()] or ["{}"]
        return "\n".join(header + options)

    def context(self, sequence_run_id: uuid.UUID | None = None) -> SequenceContext:
        """Generate a context manager inside :meth:`execute`.

        Args:
            sequence_run_id: If specified uses this id for the sequence.

        Returns:
            A new instance of SequenceContext.

        """
        return SequenceContext(
            data=SequenceData(
                name=self.name,
                analysis=self.analysis,
                options=self.options,
            ),
            attribute_options={option: getattr(self, option) for option in ATTRIBUTE_OPTIONS},
            tracker_class=self.tracker_class,
            observation_stash=self.observation_stash,
            sequence_executor_class=self.sequence_executor_class,
            sequence_run_id=sequence_run_id,
        )

    def set_attribute_options(self):  # noqa: ANN201
        """Set the attribute values from options for the options defined in ``ATTRIBUTE_OPTIONS``."""
        for option, value in self.options.items():
            if option in ATTRIBUTE_OPTIONS:
                setattr(self, option, value)

    @abstractmethod
    def execute(self) -> SequenceExecuteResult:
        """Executes the sequence.

        This is the main body of the execution code and should be implemented in a subclass.

        Inside this method

        - A :class:`.SequenceContext` should be created by making a ``with self.context() as context:`` statement.
        - Under the ``with`` statement, all calls to :meth:`.Experiment.run`, :meth:`.Experiment.analyze`, and
          :meth:`.Sequence.execute` (of subsequences) should be made using the methods of ``context``.
          Otherwise, those actions will not be tracked.
        - The method must end with ``return context.to_result()``, called outside the scope of the above-mentioned
          ``with`` statement. It converts the captured results for further processing.

        Returns:
            A new instance of SequenceExecuteResult.

        """
