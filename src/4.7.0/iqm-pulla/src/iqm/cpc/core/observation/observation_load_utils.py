# Copyright 2024-2025 IQM
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
"""Utilities for loading observations from :class:`.StorageInterface`.

Contains the storage queries used by the observation loading rules, as well as the cache loading functions
that :class:`.RuleCache` maps the cacheable loading rules to.

All functions here are self-contained: they take the :class:`.StorageInterface` instance they operate on as an argument
and have no dependency on any ambient runtime.
"""

from __future__ import annotations

from collections.abc import Iterable
import logging
from typing import Any
import uuid

from iqm.cpc.core.config import ExperimentConfiguration
from iqm.station_control.interface.models import (
    ObservationData,
    ObservationSetData,
    ObservationSetType,
)
from iqm.station_control.interface.models.type_aliases import StrUUIDOrDefault
from iqm.station_control.interface.storage_interface import StorageInterface

logger = logging.getLogger(__name__)

# TODO: The storage backend and the DUT label are mandatory arguments of the functions below, since these no
#  longer depend on any ambient runtime (they used to fall back to `ExaRuntime.storage()` and
#  `ExaRuntime.experiment_configuration().dut_label`). The cache loading functions at the end of this module
#  are called by `RuleCache.load_cache` with the loading rule attributes only, so the caller still needs to be
#  taught to pass the storage and the DUT label down to them.

_EXP_CONFIG_NODES_AND_GLOBAL_KEYS: dict[str, Any] = {
    "controllers": "common",  # The controllers section uses "common" for generic settings for legacy reasons
    "gates": (),
    "characterization": (),
    "stages": (),
}
"""Mapping for handling wildcards (generic loci) in experiment configuration nodes."""

_GLOBAL_KEY_STR_REPR = "%%%ANY%%%"
"""Intermediate representation of the global key in experiment configuration nodes used during processing."""


def _by_dut_field(observations: Iterable[ObservationData]) -> dict[str, ObservationData]:
    """Map observations to their ``dut_field``."""
    return {observation.dut_field: observation for observation in observations}


def _get_dict_of_observations(*, storage: StorageInterface, dut_label: str, **kwargs) -> dict[str, ObservationData]:
    """Get a dict of observations mapped to their ``dut_field`` queried from a :class:`.StorageInterface instance."""
    dict_of_observations: dict[str, ObservationData] = {}
    observations = storage.query_observations(
        dut_label=dut_label,
        **kwargs,
        limit=0,
    )
    for observation in observations:
        if observation.dut_field not in dict_of_observations:
            # TODO: What happens if multiple DUTs and they are using the same DUT fields?
            #  In the current solution, only the first DUT is returned in the data,
            #  rest will be ignored which is likely not the correct behaviour.
            dict_of_observations[observation.dut_field] = observation
    return dict_of_observations


def _handle_observation_set_observations(
    observations: list[ObservationData],
    set_id: uuid.UUID,
    set_type: str,
    dut_label: str,
    result_dict: dict[str, ObservationData],
) -> dict[str, ObservationData]:
    """Parses a list of observations associated with some observation set into a dictionary.

    Args:
        observations: list of observations to be parsed
        set_id: the observation set id all observations in observations list belong to.
        set_type: the type of the observation set. Affects whether an error or warning is raised upon conflicts with
            dut fields or dut labels.
        dut_label: Valid dut label.
        result_dict: the dictionary observations are to be parsed into. It can already contain observations from before.

    Raises:
        ValueError: In case an observations dut label does not match ``dut_label``, or in case a duplicate
            dut field is encountered and the set is not a generic-set.

    Returns:
        result_dict with the observations in observations list appended and dut_field as key. the set_id is appended to
            observation.observation_set_id.

    """
    for observation in observations:
        if set_type != "generic-set" and observation.dut_label != dut_label:
            raise ValueError(f"Encountered observation {observation.dut_field} not belonging to station DUTs")

        if observation.dut_field not in result_dict:
            observation.observation_set_ids = [set_id]
            result_dict[observation.dut_field] = observation
        elif set_id not in result_dict[observation.dut_field].observation_set_ids:
            # Observation already loaded as part of another set.
            result_dict[observation.dut_field].observation_set_ids.append(set_id)
        elif set_type == "generic-set":
            logger.warning("encountered duplicate dut field %s. Including only the latest one.", observation.dut_field)
        #    # latest observation has already been added in the branch above
        else:
            raise ValueError(f"loaded {set_type} contains multiple observation for dut field {observation.dut_field}.")
    return result_dict


def get_latest_observation(
    dut_field: str,
    *,
    storage: StorageInterface,
    dut_label: str,
    include_invalid: bool = False,
    tags: list[str] | None = None,
    set_id: uuid.UUID | None = None,
) -> ObservationData | None:
    """Get the latest valid observation for the given DUT, based on the created timestamp.

    If no observation was found, None is returned.

    Args:
        dut_field: the name of the observable to search by.
        storage: The storage backend to query.
        dut_label: DUT label of the device being used.
        include_invalid: when True, also observations marked as ``invalid`` are included in the query.
        tags: The tags to filter the observation with.
            The observations need to have every one of the given tags associated with them.
        set_id: The observation set the observation is associated with.

    Returns:
        Observation or None

    """
    observations = storage.query_observations(
        dut_label=dut_label,
        dut_field=dut_field,
        tags__contains=tags,
        observation_set_ids__overlap=[set_id] if set_id else None,
        invalid=None if include_invalid else False,
        limit=1,
    )
    if not observations:
        return None
    return observations[0]


def get_all_latest_observations(storage: StorageInterface | None, dut_label: str) -> dict[str, ObservationData]:
    """Get the latest observation corresponding to each ``dut_field``.

    Args:
        storage: The storage backend to query.
        dut_label: DUT label of the device being used.

    Returns: dict of the latest observations mapped to their ``dut_field`` paths.

    """
    if not storage:
        raise ValueError("Storage backend is required to load latest observations.")
    return _get_dict_of_observations(latest="dut_field", storage=storage, dut_label=dut_label)


def load_get_latest_for_tags_cache(
    tags: list[str], storage: StorageInterface | None, dut_label: str
) -> dict[str, list[ObservationData]]:
    """Load the cache for the :class:`.GetLatestForTags` load rule.

    The latest observation is returned per each DUT and ``dut_field`` and a set of tags such that the ``tags``
    given as the function argument is a subset of it.

    NOTE: This function is mainly used under the hood in :class:`.ObservationHandler`, and may behave unintuitively
    (e.g. return several observations per ``dut_field``).

    Args:
        tags: List of tags to query for. Only the observations that have at least one tag in this list are returned.
        storage: The storage backend to query.
        dut_label: DUT label of the device being used.

    Returns:
        Dict of the latest observations mapped to their ``dut_field``.

    """
    if not storage:
        raise ValueError("Storage backend is required to load latest observations.")

    dict_of_observation_lists: dict[str, list[ObservationData]] = {}

    observations = storage.query_observations(
        # FIXME: These query parameters are using custom implementation on the server side
        #  We are actually doing a query similar to latest=["dut_field", "tags"] on the server side,
        #  which isn't supported by generic query API. This will result to much less data than
        #  latest="dut_field" or latest="tags" only, thus it is much more performant.
        dut_label=dut_label,
        tags__overlap=tags,
        latest="tags",
        limit=0,
    )
    for observation in observations:
        dut_field = observation.dut_field
        if dut_field not in dict_of_observation_lists:
            dict_of_observation_lists[dut_field] = []
        dict_of_observation_lists[dut_field].append(observation)

    return dict_of_observation_lists


def get_observation_set_type(set_id: uuid.UUID, storage: StorageInterface | None) -> ObservationSetType:
    """Get the type of an observation set.

    Args:
        set_id: The observation set ID to get the type for.
        storage: The storage backend to query.

    """
    if not storage:
        raise ValueError("Storage backend is required to load latest observations.")
    return storage.get_observation_set(set_id).observation_set_type


def get_observation_set_observations(
    set_ids: tuple[uuid.UUID, ...], storage: StorageInterface | None, dut_label: str
) -> dict[str, ObservationData]:
    """Loads all observations related to a given observation set, one set at a time.

    Args:
        set_ids: The observation set IDs to load observations for.
        storage: The storage backend to query.
        dut_label: DUT label of the device being used.

    """
    if not storage:
        raise ValueError("Storage backend is required to load latest observations.")

    dict_of_observations: dict[str, ObservationData] = {}
    for set_id in set(set_ids):
        set_type = get_observation_set_type(set_id, storage=storage)
        observations = storage.query_observations(
            observation_set_ids__contains=[set_id],
            limit=0,
        )
        dict_of_observations = _handle_observation_set_observations(
            observations, set_id, set_type, dut_label, dict_of_observations
        )

    return dict_of_observations


def get_default_calset_observations(storage: StorageInterface | None) -> dict[str, ObservationData]:
    """Loads all observations related to the default calibration set.

    Args:
        storage: The storage backend to query.

    """
    if not storage:
        raise ValueError("Storage backend is required to load latest observations.")

    set_data = storage.get_calibration_set("default")
    observations = storage.query_observations(
        observation_set_ids__contains=[set_data.observation_set_id],
        invalid=False,
        limit=0,
    )
    logger.info("Loaded calibration set %s created at %s", set_data.observation_set_id, set_data.created_timestamp)
    return _handle_observation_set_observations(
        observations, set_data.observation_set_id, set_data.observation_set_type, str(set_data.dut_label), {}
    )


def get_latest_observation_set_observations(
    set_type: ObservationSetType, finalized: bool = True, *, storage: StorageInterface | None, dut_label: str
) -> dict[str, ObservationData]:
    """Loads all observations related to the latest finalized observation set of given type.

    Args:
        set_type: Type of set to look for.
        finalized: If True, only finalized sets are considered.
        storage: The storage backend to query.
        dut_label: DUT label of the device being used.

    Returns:
        Dict of observations in the found set. If no set is found, the dict is empty.

    """
    if not storage:
        raise ValueError("Storage backend is required to load latest observations.")

    set_data = get_latest_observation_set_data(set_type, finalized, storage=storage, dut_label=dut_label)
    if not set_data:
        logger.info("No %s%s found", "finalized " if finalized else "", set_type)
        return {}
    observations = storage.query_observations(
        observation_set_ids__contains=[set_data.observation_set_id],
        limit=0,
    )
    logger.info("Loaded %s %s created at %s", set_type, set_data.observation_set_id, set_data.created_timestamp)
    return _handle_observation_set_observations(
        observations, set_data.observation_set_id, set_data.observation_set_type, dut_label, {}
    )


def get_latest_observation_set_data(
    set_type: ObservationSetType, finalized: bool = True, *, storage: StorageInterface | None, dut_label: str
) -> ObservationSetData | None:
    """Find the latest finalized observation set of given type.

    Args:
        set_type: Type of set to look for.
        finalized: If True, only finalized sets are considered.
        storage: The storage backend to query.
        dut_label: DUT label of the device being used.

    Returns:
        Metadata of the found set. None if no set is found.

    """
    if not storage:
        raise ValueError("Storage backend is required to load latest observations.")

    if finalized:
        sorting_args = {"end_timestamp__isnull": False, "order_by": "-end_timestamp"}
    else:
        sorting_args = {"latest": "observation_set_type"}

    sets = storage.query_observation_sets(
        observation_set_type=set_type.value, dut_label__in=[dut_label], limit=1, **sorting_args
    )
    if not sets:
        return None
    return sets[0]


def populate_from_configs(
    items: list[tuple[tuple[ExperimentConfiguration, ...], StorageInterface | None]],
    dut_label: str,
) -> dict[uuid.UUID, dict[str, Any]]:
    """Populate the settings values from multiple ExperimentConfigurations into a dict.

    The resulting dict maps the contents of each of the configs to :attr:`.ExperimentConfiguration.id` of that
    configuration. The contents for a particular configuration are observation paths mapped to the corresponding values.

    Args:
        items: list of ``(tuple[ExperimentConfiguration, ...], storage)`` to populate settings values from. NOTE: The
            storage is not used in this function, but is included to match the signature of the other cache loading
            functions.
        dut_label: DUT label NOTE: not used in this function, but is included to match the signature of the other cache
            loading functions.

    Returns:
        The dict containing the populated observation values.

    """
    all_data: dict[uuid.UUID, dict[str, Any]] = {}

    def _populate_recursive(data: dict, config_data: dict, global_key: Any, path: list[str]) -> None:
        for key, node in config_data.items():
            key_str = _GLOBAL_KEY_STR_REPR if key == global_key else ("__".join(key) if isinstance(key, tuple) else key)
            new_path = path + [key_str]
            if isinstance(node, dict):
                _populate_recursive(data, node, global_key, new_path)
            else:
                path_str = ".".join(new_path)
                data[path_str] = node

    configs = items[0][0]  # only one storage `None` here, so `items` is of length 1
    for config in configs:
        all_data[config.id] = {}
        for root_node, global_key in _EXP_CONFIG_NODES_AND_GLOBAL_KEYS.items():
            config_data = getattr(config, root_node)
            _populate_recursive(all_data[config.id], config_data, global_key, path=[root_node])
    return all_data


def _load_latest_observations_cache(
    items: list[tuple[StorageInterface | None]],
    dut_label: str,
) -> dict[StorageInterface | None, dict[str, ObservationData]]:
    """Load the cache for the :class:`.GetLatest` load rule, one storage backend at a time."""
    storages = [storage for (storage,) in items]
    return {storage: get_all_latest_observations(storage=storage, dut_label=dut_label) for storage in set(storages)}


def _load_latest_for_tags_observations_cache(
    items: list[tuple[tuple[str, ...], StorageInterface | None]],
    dut_label: str,
) -> dict[StorageInterface | None, dict[str, list[ObservationData]]]:
    """Load the cache for the :class:`.GetLatestForTags` load rule, one storage backend at a time.

    Only the tags of rules targeting a given storage are queried for that storage; tags from rules
    targeting a different storage are never mixed in.
    """
    cache: dict[StorageInterface | None, dict[str, list[ObservationData]]] = {}
    for tag_tuples, storage in items:
        merged_tags = sorted({tag for tag_tuple in tag_tuples for tag in tag_tuple})
        cache[storage] = load_get_latest_for_tags_cache(merged_tags, storage=storage, dut_label=dut_label)
    return cache


def _load_observation_set_observations_cache(
    items: list[tuple[tuple[uuid.UUID], StorageInterface | None]],
    dut_label: str,
) -> dict[StorageInterface | None, dict[str, ObservationData]]:
    """Load the cache for the :class:`.GetObservationSet` load rule, one storage backend at a time."""
    return {
        storage: get_observation_set_observations(set_ids, storage=storage, dut_label=dut_label)
        for set_ids, storage in items
    }


def _load_calibration_set_observations_cache(
    items: list[tuple[tuple[StrUUIDOrDefault, ...], StorageInterface]],
    dut_label: str,
) -> dict[StorageInterface, dict[StrUUIDOrDefault, dict[str, ObservationData]]]:
    """Load the cache for the :class:`.GetCalibrationSet` load rule, one storage backend at a time.

    NOTE: This method should be considered internal to the observation_stash. It should not
    be accessed outside of it.

    Args:
        items: (calibration set ID, storage) pairs to load observations for.
        dut_label: DUT label of the device (not used in this function, but is included to match the signature of the
            other cache loading functions).

    Returns:
        Dict mapping each storage backend to a dict mapping each calibration set ID to its observations,
        keyed by ``dut_field``.

    """
    return {
        storage: {
            calibration_set_id: _by_dut_field(storage.get_calibration_set(calibration_set_id).observations)
            for calibration_set_id in set(calibration_set_ids)
        }
        for calibration_set_ids, storage in items
    }


def _load_calibration_set_quality_metric_set_observations_cache(
    items: list[tuple[tuple[StrUUIDOrDefault, ...], StorageInterface]], dut_label: str
) -> dict[StorageInterface, dict[StrUUIDOrDefault, dict[str, ObservationData]]]:
    """Load the cache for the :class:`.GetQualityMetricSet` load rule, one storage backend at a time.

    NOTE: This method should be considered internal to the observation_stash. It should not
    be accessed outside of it.

    Args:
        items: (calibration set ID, storage) pairs to load quality metric observations for.
        dut_label: DUT label of the device (not used in this function, but is included to match the signature of the
            other cache loading functions).

    Returns:
        Dict mapping each storage backend to a dict mapping each calibration set ID to its quality metric set's
        observations, keyed by ``dut_field``.

    """
    return {
        storage: {
            calibration_set_id: _by_dut_field(
                storage.get_calibration_set_quality_metric_set(calibration_set_id).observations
            )
            for calibration_set_id in set(calibration_set_ids)
        }
        for calibration_set_ids, storage in items
    }


def _load_default_calibration_set_observations_cache(
    storages: list[tuple[StorageInterface]], dut_label: str
) -> dict[StorageInterface, dict[str, ObservationData]]:
    """Load the cache for the (deprecated) :class:`.GetDefaultCalibrationSet` load rule, one storage at a time."""
    return {
        storage_tuple[0]: get_default_calset_observations(storage=storage_tuple[0]) for storage_tuple in set(storages)
    }


def _load_latest_characterization_set_observations_cache(
    storages: list[tuple[StorageInterface]],
    dut_label: str,
) -> dict[StorageInterface, dict[str, ObservationData]]:
    """Load the cache for the :class:`.GetLatestCharacterizationSet` load rule, one storage at a time."""
    return {
        storage_tuple[0]: get_latest_observation_set_observations(
            ObservationSetType.CHARACTERIZATION_SET, finalized=True, storage=storage_tuple[0], dut_label=dut_label
        )
        for storage_tuple in set(storages)
    }
