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
"""Convenience functions for accessing data."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Iterator, Sequence
import logging
from typing import Any

import numpy as np
from typing_extensions import deprecated

from exa.common.errors import ConflictError
from exa.common.helpers import format_deprecated
from exa.core.provider import ExaRuntime

# The following functions have moved to iqm.cpc.core.observation.observation_load_utils. They are imported
# here to retain backwards compatibility for imports.
from iqm.cpc.core.observation.observation_load_utils import (  # noqa: F401
    _get_dict_of_observations,
    _handle_observation_set_observations,
    get_all_latest_observations,
    get_default_calset_observations,
    get_latest_observation,
    get_latest_observation_set_data,
    get_latest_observation_set_observations,
    get_observation_set_observations,
    get_observation_set_type,
    load_get_latest_for_tags_cache,
)
from iqm.station_control.interface.models import (
    ObservationData,
    ObservationDefinition,
    ObservationUpdate,
)
from iqm.station_control.interface.storage_interface import StorageInterface

logger = logging.getLogger(__name__)


def _resolve_storage(storage: StorageInterface | None) -> StorageInterface:
    """Return ``storage`` if given, otherwise fall back to the ambient ``ExaRuntime.storage()``.

    Raises a clearer error than the bare ``ExaRuntime`` one when neither is available, since the most likely
    cause is a caller (e.g. an observation loading rule) that forgot to pass an explicit ``storage``.
    """
    if storage is not None:
        return storage
    try:
        return ExaRuntime.storage()
    except ConflictError as err:
        raise ConflictError(
            "No storage was given and no EXA runtime is active. Pass storage=<StorageInterface> explicitly, "
            "or initialize an ExaRuntime."
        ) from err


def save_observation(
    dut_field: str,
    value: bool | int | float | complex | str | np.ndarray,
    unit: str,
    source: dict[str, Any],
    uncertainty: int | float | complex | None = None,
    tags: list[str] | None = None,
    invalid: bool = False,
    dut_label: str | None = None,
) -> ObservationData:
    """Save an observation for chip that is specified in configuration file.

    Args:
        dut_field: The name of the observable. For example, ``QB1.flux.voltage``.
        value: Value that is saved as an observation for ``dut_field``.
        unit: The observation unit (base SI unit, e.g. Hz and never GHz).
            Can be an empty string for quantities without unit (e.g. fidelity values).
        source: How the observation was made, e.g. experiment analysis or manual specification.
        uncertainty: The uncertainty of the observation value. ``None`` means unknown.
        tags: Human-readable tags about the observation.
        invalid: Boolean flag indicating if observation is invalid. Defaults to False.
        dut_label: DUT label of the device being used.
            If not given, defaults to the one used in experiment configuration.

    """
    if not dut_label:
        dut_label = ExaRuntime.experiment_configuration().dut_label
    if not tags:
        tags = []

    observation = ExaRuntime.storage().create_observations(
        [
            ObservationDefinition(
                dut_label=dut_label,
                dut_field=dut_field,
                value=value,
                unit=unit,
                source=source,
                uncertainty=uncertainty,
                tags=tags,
                invalid=invalid,
            )
        ]
    )[0]

    logger.info("Observation saved: dut_field=%s, value=%s", dut_field, value)

    return observation


def batch_save_observations(
    observations: Sequence[ObservationDefinition],
    storage: StorageInterface | None = None,
) -> list[ObservationData]:
    """Save a list of observations in a batch.

    Args:
        observations: A list of local observations to be saved.
        storage: If given, overrides the default (ambient) storage backend.

    Returns:
        The ObservationData objects that were successfully saved into the database. If some
        observations failed to save, they are omitted here (and logged as a warning by the
        storage backend); callers needing an all-or-nothing guarantee must check the count.

    """
    if not observations:
        return []

    storage = _resolve_storage(storage)
    saved_observations = storage.create_observations(observations)

    logger.info("Batch-saved %d observations.", len(saved_observations))
    if logger.level <= logging.DEBUG:
        log_msg = "\n".join([f"{observation.dut_field}:\t{observation.value}" for observation in saved_observations])
        logger.debug("Saved observations:\n%s", log_msg)
    return saved_observations


@deprecated(format_deprecated(old="get_all_dut_fields", new=None, since="2026-08-21"))
def get_all_dut_fields() -> set[str]:
    """Get all dut field paths for all DUTs the global IQM Server is configured with.

    Scans every field ever observed for the DUT, independent of any actual query need. Prefer filtering
    observations directly (e.g. by tags or observation set) instead of enumerating all dut fields upfront.

    Returns: set of all DUT field paths.
    """
    all_dut_fields = set()
    dut_label = ExaRuntime.experiment_configuration().dut_label
    if dut_label:
        dut_fields = ExaRuntime.storage().get_dut_fields(dut_label=dut_label)
        all_dut_fields.update([dut_field.path for dut_field in dut_fields])
    return all_dut_fields


def _latest_by_tag_combinations(observations: Iterable[ObservationData]) -> dict[str, list[ObservationData]]:
    """Parses a collection of ObservationData objects into a dictionary, where they key is dut_field, and the value
    a list of the latest observation corresponding to that dut_field for each unique tag combination.

    The method assumes that observations are provided in descending order by time created.
    """
    dict_of_observation_lists: dict[str, list[ObservationData]] = {}
    for observation in observations:
        dut_field = observation.dut_field
        if dut_field not in dict_of_observation_lists:
            dict_of_observation_lists[dut_field] = [observation]
        else:
            # DUT field already exist with some tags, add observation only if tags are different
            tags = [frozenset(o.tags) for o in dict_of_observation_lists[dut_field]]
            if frozenset(observation.tags) not in tags:
                dict_of_observation_lists[dut_field].append(observation)
    return dict_of_observation_lists


def get_latest_observations_for_tags(tags: list[str]) -> dict[str, ObservationData]:
    """Returns the latest observations per ``dut_field`` that have the provided tags (and possibly more tags).

    Args:
        tags:  List of tags to query for. The returned observations contain at least these tags.

    Returns:
        Dict of the latest observations mapped to their ``dut_field``.

    """
    return _get_dict_of_observations(
        storage=ExaRuntime.storage(),
        dut_label=ExaRuntime.experiment_configuration().dut_label,
        tags__contains=tags,
        latest="dut_field",
    )


def save_observations_with_new_tags(
    tags: Sequence[str] = (),
    add_tags: Collection[str] = (),
    remove_tags: Collection[str] = (),
    remove_observations: Collection[str] = (),
) -> list[ObservationData]:
    """Queries the latest observations (per dut_field) for set of tags and allows saving them with new set of tags.

    If both `add_tags` and `remove_tags` are None will not save anything but just returns the queried observations.

    Args:
        tags: the tags of the observations to be queried. If left empty, queries the latest observations for each
            dut_field.
        add_tags: the new tags to be added to the observations.
        remove_tags: the tags to be removed from the observations. The tags to be removed need to be defined for
            an observation in order for it to be included in the returned observations and be saved.
        remove_observations: Observations with dut fields appearing in `remove_observations` do not carry over to the
            next tag.

    Returns:
        The observations with the new tags (or the original ones if both `add_tags` and `remove_tags` are None)

    """
    observations = list(_find_latest_observations_matching_all_tags(tags))
    observation_definitions: list[ObservationDefinition] = []
    for observation in observations:
        if observation.dut_field in remove_observations:
            continue
        if add_tags or remove_tags:
            new_tags = list(set(observation.tags or []).union(set(add_tags)) - set(remove_tags))
            if new_tags != observation.tags:
                observation_definitions.append(
                    ObservationDefinition(
                        dut_label=observation.dut_label,
                        dut_field=observation.dut_field,
                        value=observation.value,
                        unit=observation.unit,
                        source=observation.source,
                        uncertainty=observation.uncertainty,
                        tags=new_tags,
                        invalid=observation.invalid,
                    )
                )
    if not observation_definitions:
        return observations
    return batch_save_observations(observation_definitions)


def _find_latest_observations_matching_all_tags(tags: Sequence[str]) -> Iterator[ObservationData]:
    """Utility for querying latest observations with tags per every dut_field or just per every dut_field."""
    return ExaRuntime.storage().query_observations(
        tags__contains=tags if tags else None,
        latest="dut_field",
        dut_label=ExaRuntime.experiment_configuration().dut_label,
        limit=0,
    )


def invalidate_observation(observation: ObservationData | None) -> ObservationData | None:
    """Invalidates observation.

    Observation is marked as invalid by setting the invalid field to True.
    If there is no observation, then the function does not have any side-effects.

    Args:
        observation: Observation to invalidate, or None.

    Returns:
        Observation with invalid flag set or None.

    """
    return set_observation_validity(observation, invalid=True)


def set_observation_validity(observation: ObservationData | None, *, invalid: bool) -> ObservationData | None:
    """Set observation validity flag.

    Observation field `invalid` is set to a value passed as an argument.
    If observation is None, then the function returns None without side-effects.

    Args:
        observation: Observation to invalidate, or None.
        invalid: value for the invalid field.

    Returns:
        Observation with invalid flag set or None.

    """
    if observation is not None:
        return ExaRuntime.storage().update_observations(
            [
                ObservationUpdate(
                    observation_id=observation.observation_id,
                    invalid=invalid,
                ),
            ]
        )[0]
    return None


def _get_observations_from_database(**kwargs) -> Iterator[ObservationData]:
    """Get observations from database, filtering based on the criteria provided.

    This method is used by :class:`QPUPlot`.

    Args:
        **kwargs: Filters used in the query.

    Returns:
        The observations that match the given filters.

    """
    query_arguments = {key: value for key, value in kwargs.items() if value}
    return ExaRuntime.storage().query_observations(**query_arguments, limit=0)
