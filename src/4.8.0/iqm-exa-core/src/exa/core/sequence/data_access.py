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
"""Sequence execute result serialization and deserialization with saving and loading helpers."""

import getpass
import logging
import os
import platform
from uuid import UUID

from exa.core.experiment import Experiment
from exa.core.provider.provider import ExaRuntime
from exa.core.util.param_parsing import chunk_list
from iqm.station_control.interface.models import (
    ObservationData,
    SequenceMetadataDefinition,
    SequenceResultData,
    SequenceResultDefinition,
)
from iqm.station_control.interface.storage_interface import StorageInterface

from .interface import (
    SERIALIZER_VERSION,
    SequenceContextData,
    SequenceExecuteResultData,
    SequenceExecuteResultDataModel,
)
from .result import SequenceExecuteResult
from .wrap_analyze import wrap_experiment_analyze, wrap_sequence_analyze

logger = logging.getLogger(__name__)


def deserialize_sequence_result(sequence_result: SequenceResultData) -> SequenceExecuteResult:
    """Deserialize SequenceResultData object to SequenceExecuteResult.

    This function recursively loads and analyzes the results of each executed node (experiment or subsequence) inside
    the result.

    If any custom analysis functions were used, they must be defined in the current run time.

    Args:
        sequence_result: SequenceResult object.

    Returns:
        SequenceExecuteResult object.

    """
    result = SequenceExecuteResultDataModel(**sequence_result.data)
    analyze_results = []
    if result.version != SERIALIZER_VERSION:
        raise ValueError(
            f"Incorrect serialization version, expected != actual: {SERIALIZER_VERSION} != {result.version}"
        )
    for execution in result.context.executed_nodes:
        logger.debug("Loading %s %s", execution.type, execution.run_id)

        if execution.type == "experiment":
            loaded_experiment = Experiment.get_by_run_id(execution.run_id)
            custom_analysis = result.context.custom_experiment_analyses.get(str(execution.run_id), None)
            analyze_result = wrap_experiment_analyze(
                loaded_experiment, custom_analysis=custom_analysis, **execution.analyze_args
            )
        elif execution.type == "sequence":
            loaded_sequence_result = load_sequence_result(execution.run_id)
            analyze_result = wrap_sequence_analyze(loaded_sequence_result, **execution.analyze_args)  # type: ignore[assignment, arg-type]
        else:
            raise ValueError(f"Unrecognized sequence node: {execution.type}")

        if analyze_result.error is not None:
            logger.warning(
                "Could not fully reconstruct %s with analysis args %s: %s. Check the result for traceback.",
                execution.run_id,
                execution.analyze_args,
                analyze_result.error,
            )
        else:
            logger.debug(
                "Successfully reconstructed %s with analysis args %s", execution.run_id, execution.analyze_args
            )
        analyze_results.append(analyze_result)
    return SequenceExecuteResult(context=result.context, analyze_results=analyze_results)  # type: ignore[arg-type, call-arg, misc]


def serialize_sequence_result(
    sequence_execute_result: SequenceExecuteResultData, final: bool
) -> SequenceResultDefinition:
    """Serialize SequenceExecuteResultData object to SequenceResultDefinition.

    Args:
        sequence_execute_result: Sequence execute result.
        final: A boolean indicating whether the result is marked final.

    Returns:
        SequenceResultDefinition object.

    """
    model = SequenceExecuteResultDataModel(context=sequence_execute_result.context)
    return SequenceResultDefinition(
        sequence_id=sequence_execute_result.context.sequence_run_id,
        data=model.model_dump(mode="json"),
        final=final,
    )


def save_sequence_result(sequence_execute_result: SequenceExecuteResultData, final: bool) -> None:
    """Save the result of a sequence to persistence.

    This function can be called repeatedly on the same sequence. Subsequent calls will update the data in the database.
    Thus, for incremental saving, a sequence will call this function with updated results after each run.

    This function does not save the results of child sequences recursively.

    Before calling this function, metadata has to exist in persistence so that results can be linked.
    That is, for the sequence in question, :func:`save_sequence_metadata` has to be called before this function.

    Args:
        sequence_execute_result: The results data model to be serialized and saved to persistence.
        final: A boolean indicating whether the result is marked final. After being marked final, further saving of
            results for the same sequence are prevented.

    """
    sequence_id = sequence_execute_result.context.sequence_run_id

    logger.debug(
        "Saving %s sequence %s result",
        "final" if final else "intermediate",
        sequence_id,
    )
    sequence_result_definition = serialize_sequence_result(sequence_execute_result, final)
    ExaRuntime.storage().save_sequence_result(sequence_result_definition)


def save_sequence_metadata(data: SequenceContextData) -> None:
    """Save metadata of a sequence to persistence.

    This function is to be called once in the beginning of sequence execution.
    It creates an entity for the sequence in persistence that will be linked to the result data when calling
    :func:`save_sequence_result` for the same sequence.

    Along with the UUID of the sequence, the following metadata is needed along with the results in order to create a
    usable entity in persistence:

    - ``origin_id``: See ``EXA_SEQUENCE_ORIGIN_ID`` in sequences user documentation.
    - ``origin_uri``: See ``EXA_SEQUENCE_ORIGIN_URI`` in sequences user documentation.

    Sensible default values are set for the additional metadata. Thus, a regular notebook user does not have
    to tweak anything.

    Args:
        data: context of the sequence.

    """
    origin_id = os.environ.get("EXA_SEQUENCE_ORIGIN_ID")
    if origin_id is None:
        origin_id = ExaRuntime.experiment_configuration().user

    origin_uri = os.environ.get("EXA_SEQUENCE_ORIGIN_URI")
    if origin_uri is None:
        origin_uri = f"{getpass.getuser()}@{platform.node()}"

    logger.debug(
        "Saving sequence %s metadata with origin_id: %s, origin_uri: %s",
        data.sequence_run_id,
        origin_id,
        origin_uri,
    )

    sequence_metadata_definition = SequenceMetadataDefinition(
        sequence_id=data.sequence_run_id,
        origin_id=origin_id,
        origin_uri=origin_uri,
    )
    ExaRuntime.storage().create_sequence_metadata(sequence_metadata_definition)


def load_sequence_result(sequence_run_id: UUID) -> SequenceExecuteResult:
    """Load sequence execute result from persistence.

    Args:
        sequence_run_id: Sequence ID.

    Returns:
        SequenceExecuteResult

    """
    logger.debug("Loading sequence %s result", sequence_run_id)
    sequence_result = ExaRuntime.storage().get_sequence_result(sequence_run_id)
    return deserialize_sequence_result(sequence_result)


def load_observations_for(
    sequence_run_id: UUID, tags: list[str] | None = None
) -> dict[tuple[str, ...], ObservationData]:
    """Load all the observations for a given sequence and all of its child nodes.

    The observations are returned from any level of subsequences and their experiment nodes. The returned values
    are mapped to tuples that contain the "path" of IDs for the sequences and/or experiment that created them.
    For example, if a sequence with ID=0 has a subsequence with ID=1 which further has an experiment with ID=3
    that creates an observation with ``dut_field`` "QB1.readout.amplitude", the observation will be mapped to a path
    ``("0","1","2","QB1.readout_amplitude")``.

    Args:
        sequence_run_id: The sequence ID.
        tags: Optionally filter with these tags.

    Returns:
        Observations mapped to the paths of IDs for the sequences or experiments that created them.

    """

    def _filter_by_tags(tags: list[str] | None, obs_tags: list[str] | None) -> bool:
        if not tags:
            return True
        if not obs_tags:
            return False
        return set(tags).issubset(set(obs_tags))

    max_uids_to_query = 25  # Any more than this risk creating query urls over 2000 characters.
    storage = ExaRuntime.storage()
    run_ids, sequence_ids, paths = _get_child_node_ids(sequence_run_id, storage)

    # Have to query separately for sequence IDs and run IDs
    # The station control interface works only with "and" logic, thus all parameters must be true at the same time.
    # Since observation can have only one source, both sequence ID and run ID can't be true the same time.
    observations = []  # type: ignore[var-annotated]
    for ids in chunk_list(sequence_ids, max_uids_to_query):
        observations += storage.query_observations(source__sequence_id__in=ids, limit=0)
    for ids in chunk_list(run_ids, max_uids_to_query):
        observations += storage.query_observations(source__run_id__in=ids, limit=0)

    observations_mapped = {}
    for observation in observations:
        source = observation.source
        execution_id = source["run_id"] if source["type"] == "analysis_source" else source["sequence_id"]
        for path in paths:
            if path[-1].replace("-", "") == str(execution_id).replace("-", "") and _filter_by_tags(
                tags, observation.tags
            ):
                observations_mapped[path + (observation.dut_field,)] = observation
    return observations_mapped


def _get_child_node_ids(
    sequence_id: UUID, storage: StorageInterface
) -> tuple[list[UUID], list[UUID], list[tuple[str, ...]]]:
    """Recursively find all the child node sequences and run IDs associated with a sequence."""
    sequence_result = storage.get_sequence_result(sequence_id)
    data = SequenceExecuteResultDataModel(**sequence_result.data)

    run_ids: list[UUID] = []
    sequence_ids: list[UUID] = [sequence_id]
    paths: list[tuple[str, ...]] = [(str(sequence_id),)]
    for execution in data.context.executed_nodes:
        if execution.type == "experiment":
            run_ids.append(execution.run_id)
            paths.append((str(sequence_id),) + (str(execution.run_id),))
        elif execution.type == "sequence":
            child_run_ids, child_sequence_ids, child_paths = _get_child_node_ids(execution.run_id, storage)
            run_ids += child_run_ids
            sequence_ids += [execution.run_id] + child_sequence_ids
            paths += [(str(sequence_id),) + path for path in child_paths]
    return run_ids, sequence_ids, paths
