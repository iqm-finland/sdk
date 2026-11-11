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
"""Class for storing & loading observations in run time."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import MINYEAR, UTC, datetime
import logging
import uuid

import numpy as np
from typing_extensions import deprecated

from exa.common.data import ObservationValue
from exa.common.helpers import format_deprecated
from exa.core.database_wrapper import get_all_dut_fields
from iqm.cpc.core.observation.observation_loading_rules import (
    LatestFromStash,
    RuleType,
)
from iqm.station_control.interface.models import ObservationData, ObservationDefinition
from iqm.station_control.interface.storage_interface import StorageInterface

from .observation import LocalObservation, LocalObservationSequence
from .observation_handler import ObservationHandler, load_observations_by_rules
from .observation_loading_rules import (
    GetLatestForTags,
    GetObservationSet,
    set_global_default_observation_loading_rules,
)
from ..provider.provider import ExaRuntime

logger = logging.getLogger(__name__)


@deprecated(format_deprecated(old="get_loaded_stash_with", new="get_loaded_stash_from_rules", since="2026-09-07"))
def get_loaded_stash_with(
    load_rules: list[RuleType], compare_db_and_local_time: bool | None = None
) -> ObservationStash:
    """Utility for loading all observations for a given dut from the DB and mapping them to their dut field paths.

    Uses given observation rules to load the values from the DB.

    Args:
        load_rules: The load rules for loading the observations.
        compare_db_and_local_time: Allow comparing local and database timestamps when fetching latest observations.
            If disabled, local observations will always have priority to database observations, and a warning will be
            shown in case a database observation may be more recent. Setting the flag explicitly to ´False´ Suppresses
            the warning. Disabled by default.

    Returns: The observations mapped to their dut field paths.

    """
    handler = ObservationHandler(
        list(get_all_dut_fields()),
        load_rules=load_rules,
        dut_label=ExaRuntime.experiment_configuration().dut_label,
        default_storage=ExaRuntime.storage(),
    )
    handler.load_observations()

    observations_for_stash = {}
    for observation_and_rule in handler.loaded_observations.values():
        observation = observation_and_rule[0]
        if observation:
            observations_for_stash[observation.dut_field] = [observation]
    return ObservationStash(
        local_observations={},
        db_observations=observations_for_stash,  # type: ignore[arg-type]
        compare_db_and_local_time=compare_db_and_local_time,
    )


def get_loaded_stash_from_rules(
    load_rules: list[RuleType],
    *,
    dut_label: str | None = None,
    default_storage: StorageInterface | None = None,
    compare_db_and_local_time: bool | None = None,
) -> ObservationStash:
    """Utility for loading observations resolved by the given rules and mapping them to their dut field paths.

    Unlike :func:`get_loaded_stash_with`, this does not scan every dut field ever recorded for the DUT. It only
    touches the dut fields that ``load_rules`` themselves surface from the DB (see
    :func:`.load_observations_by_rules`), so it needs no prior knowledge of which dut fields exist.

    NOTE: This may slightly alter how the settings tree is built. In particular, its possible that certain
    branches are not initially created, if they do no exist in the observations the stash was initialized with.
    Such observations can however still be added to the stash.

    Args:
        load_rules: The load rules for loading the observations.
        dut_label: The dut label to use. If not provided, will use the dut label from the ExaRuntime experiment
            configuration.
        default_storage: The default storage to use when no storage is pinned by the loading rule. If not provided,
            will use the ExaRuntime storage.
        compare_db_and_local_time: Allow comparing local and database timestamps when fetching latest observations.
            If disabled, local observations will always have priority to database observations, and a warning will be
            shown in case a database observation may be more recent. Setting the flag explicitly to ´False´ Suppresses
            the warning. Disabled by default.

    Returns: The observations mapped to their dut field paths.

    """
    if dut_label is None:
        dut_label = ExaRuntime.experiment_configuration().dut_label
    if default_storage is None:
        default_storage = ExaRuntime.storage()
    observations_for_stash = {
        dut_field: [observation]
        for dut_field, observation in load_observations_by_rules(load_rules, dut_label, default_storage).items()
    }
    return ObservationStash(
        local_observations={},
        db_observations=observations_for_stash,  # type: ignore[arg-type]
        compare_db_and_local_time=compare_db_and_local_time,
    )


class ObservationStash:
    """Class for saving local observations in the runtime instead of the DB.

    Args:
        local_observations: Dictionary mapping lists of local observations to their dut field path
            (The full one which includes the component name). Duplicates are removed.
        db_observations: Dictionary mapping lists of db observations to their dut field path
            (The full one which includes the component name). Duplicates are removed. The DB observations will be
            returned by :meth:`.get_latest_observation` in case there is no local observation for that ``dut_field``
            found.
        compare_db_and_local_time: Allow comparing local and database timestamps when fetching latest observations.
            If disabled, local observations will always have priority to database observations, and a warning will be
            shown in case a database observation may be more recent. Setting the flag explicitly to ´False´ Suppresses
            the warning. Disabled by default.

    """

    def __init__(
        self,
        local_observations: Mapping[str, Sequence[LocalObservation]],
        db_observations: Mapping[str, Sequence[ObservationData]],
        compare_db_and_local_time: bool | None = None,
    ):
        self._db_observations: dict[str, list[ObservationData]] = {path: [] for path in db_observations}
        self._local_observations: dict[str, list[LocalObservation]] = {path: [] for path in local_observations}
        for dut_field, obs_seq in local_observations.items():
            for obs in obs_seq:
                if self._is_not_yet_stashed(obs):
                    self._local_observations[dut_field].append(obs)
        for dut_field, obs_seq in db_observations.items():  # type: ignore[assignment]
            for obs in obs_seq:
                if self._is_not_yet_stashed(obs):
                    self._db_observations[dut_field].append(obs)  # type: ignore[arg-type]
        self.compare_db_and_local_time = compare_db_and_local_time

    @staticmethod
    def from_tags(tags: Iterable[Iterable[str]] = ()) -> ObservationStash:
        """Create an observation stash containing latest observations from the DB filtered using ``tags``.

        Only contains observations for the ``dut_id`` of the station.

        Args:
            tags: Observation tags to choose which observations to load. Outer iterable is a short-circuit OR, inner is
                AND. Example: ``[['awesome', 'monday'], ['base']]`` means "For every observation (dut_field) we need,
                first try to load the latest such observation that is tagged with both ``'awesome'`` AND ``'monday'``.
                If one with these tags cannot be found, try finding one tagged with ``'base'``" instead. If ``tags`` is
                empty, no observations will be loaded.

        Returns:
            observation stash containing the requested observations

        """
        if tags:
            rules: list[RuleType] = [GetLatestForTags(list(t)) for t in tags]
            stash = get_loaded_stash_from_rules(rules)
            return stash
        return ObservationStash({}, {}, compare_db_and_local_time=False)

    @staticmethod
    def from_observation_set(set_id: uuid.UUID) -> ObservationStash:
        """Create an observation stash containing latest observations from the DB filtered using ``set_id``.

        Only contains observations for the ``dut_id`` of the station.

        Args:
            set_id: The observation set ID for which to load observations.

        Returns:
            observation stash containing the requested observations

        """
        rules: list[RuleType] = [GetObservationSet(set_id)]
        stash = get_loaded_stash_from_rules(rules)
        return stash

    def copy(self) -> ObservationStash:
        """Copies the observation stash."""
        return ObservationStash(self.get_local_observations(), self.get_db_observations())

    def stash(self, observations: LocalObservation | list[LocalObservation]) -> None:
        """Adds local observations to the stash.

        Only observations not yet found in self are added.

        Args:
            observations: A list of observations or a singular observation to be added.

        """
        if isinstance(observations, LocalObservation):
            observations = [observations]
        for observation in observations:
            dut_field = observation.dut_field
            if self._is_not_yet_stashed(observation):
                if dut_field not in self._local_observations:
                    self._local_observations[dut_field] = []
                self._local_observations[dut_field].append(observation)

    def get_latest_observation(
        self, dut_field: str, tags: list[str] | None = None, set_ids: list[uuid.UUID] | None = None
    ) -> ObservationDefinition | None:
        """Gives the latest observation for the specified name (i.e. DUT field path).

        By default, any local observations added to the stash are considered more recent than database observations
        stored in the stash when initialising it. This can be altered using the compare_local_and_db_time flag when
        initializing the ObservationStash

        Args:
            dut_field: The full DUT field path of the observation, including the component name.
            tags: The tags used for filtering the observations.
            set_ids: TODO

        Returns:
            The latest observation.

        """
        latest_from_db = None
        latest_local = None
        if dut_field in self._db_observations:
            latest_from_db = _find_latest_observation(self._db_observations[dut_field], tags=tags, set_ids=set_ids)  # type: ignore[arg-type]
        if dut_field in self._local_observations:
            latest_local = _find_latest_observation(self._local_observations[dut_field], tags=tags, set_ids=set_ids)  # type: ignore[arg-type]

        if latest_local and latest_from_db:
            latest = _find_latest_observation([latest_local, latest_from_db])
            if self.compare_db_and_local_time:
                return latest
            if self.compare_db_and_local_time is None and isinstance(latest, ObservationData):
                logger.warning(
                    "Found database observation which may be more recent than returned LocalObservation. "
                    "Set flag 'compare_db_and_local_time' to `True` when initializing ObservationStash to compare "
                    "database and local times. Set flag 'compare_db_and_local_time' to `False` to supress this warning."
                )
        return latest_local if latest_local else latest_from_db

    def get_local_observations(
        self,
        components: list[str] | None = None,
        observations: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, list[LocalObservation]]:
        """Get the local observations stored in the stash.

        Args:
            components: Components for which to get the observations. If not provided, will get for all components.
            observations: Observation parameter names for which to get the observations. If not provided, will
                get for all observation parameters.
            tags: If provided, will get only the observations containing these tags.

        """
        return self._get_observations(components, observations, tags, True)  # type: ignore[return-value]

    def get_db_observations(
        self,
        components: list[str] | None = None,
        observations: list[str] | None = None,
        tags: list[str] | None = None,
        set_ids: list[uuid.UUID] | None = None,
    ) -> dict[str, list[ObservationData]]:
        """Get the DB observations stored in the stash.

        Args:
            components: Components for which to get the observations. If not provided, will get for all components.
            observations: Observation parameter names for which to get the observations. If not provided, will
                get for all observation parameters.
            tags: If provided, will get only the observations containing these tags.
            set_ids: If provided, will get only the observations associated with the given set ids. Can not be used
                together with tags.

        """
        return self._get_observations(components, observations, tags, False, set_ids)  # type: ignore[return-value]

    def get_latest_observations(
        self,
        components: list[str] | None = None,
        observations: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, ObservationDefinition]:
        """Get the latest local or DB observations stored in the stash.

        Args:
            components: Components for which to get the latest observations.
                If not provided, will get for all components.
            observations: Observation parameter names for which to get the latest observations.
                If not provided, will get for all observation parameters.
            tags: If provided, will get only the latest observations containing these tags.

        """
        local_observation_keys = list(self.get_local_observations(components, observations, tags).keys())
        db_observation_keys = list(self.get_db_observations(components, observations, tags).keys())
        keys = list(set(local_observation_keys + db_observation_keys))
        return {key: self.get_latest_observation(key, tags) for key in keys}  # type: ignore[misc]

    def modify_tags(
        self,
        tags_to_add: list[str] | None = None,
        tags_to_remove: list[str] | None = None,
        components: list[str] | None = None,
        observations: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """Tag observations in the stash.

        Args:
            tags_to_add: Tags to be added.
            tags_to_remove: Tags to be removed.
            components: Components for which to add/remove tags. If not provided, will apply to all components.
            observations: Observations for which to add/remove tags. If not provided, will apply to all observations.
            tags: Tags for which to add/remove more tags.

        """
        tags_to_add = tags_to_add or []
        tags_to_remove = tags_to_remove or []
        for obs_path, obs_seq in self._local_observations.items():
            if _filter_paths(obs_path, components, observations):
                for obs in obs_seq:
                    if not tags or set(tags).issubset(obs.tags):
                        for tag_to_be_removed in tags_to_remove:
                            if tag_to_be_removed in obs.tags:
                                obs.tags.remove(tag_to_be_removed)
                        obs.tags += tags_to_add

    def merge_to(
        self,
        other: ObservationStash,
        components: list[str] | None = None,
        observations: list[str] | None = None,
        tags: list[str] | None = None,
        only_latest: bool = False,
    ) -> None:
        """Merge the local observations from self to another stash.

        Args:
            other: The stash into which to merge the local observations.
            components: Components for which to merge the observations. If not provided, will merge for all components.
            observations: Observation parameter names for which to merge the observations. If not provided, will
                merge for all observation parameters.
            tags: If provided, will merge only the observations containing these tags.
            only_latest: Option to merge only the latest observation per dut field.

        """
        if only_latest:
            to_be_merged = list(self._get_latest_for_saving(components, observations, tags))
        else:
            to_be_merged = []
            for obs_seq in self._local_observations.values():
                to_be_merged += obs_seq
        other.stash(to_be_merged)

    def save(
        self,
        components: list[str] | None = None,
        observations: list[str] | None = None,
        tags: list[str] | None = None,
        new_tags: list[str] | None = None,
        only_latest: bool = False,
        convert_to_db_observations: bool = False,
        storage: StorageInterface | None = None,
    ) -> list[ObservationData]:
        """Saves the local observations to the DB.

        Args:
            components: Components for which to save the observations. If not provided, will save for all components.
            observations: Observation parameter names for which to save the observations. If not provided, will
                save for all observation parameters.
            tags: If provided, will save only the observations containing these tags.
            new_tags: Optionally tag the observations with these tags when saving.
            only_latest: Option to save only the latest observation per dut field.
            convert_to_db_observations: If set ``True``, moves the saved observations as DB observations, so that
                they cannot be saved again from this same stash.
            storage: If given, overrides the default (ambient) storage backend for this call.

        Returns:
            The saved observations as in a list of DB observations.

        """
        num_saves = 1 if only_latest else max(len(s) for s in self._local_observations.values())
        # The batch saving to DB creates a common timestamp for the whole batch, so we must save observations
        # with the same dut_field in different batches and in the same order as they were created.
        # For this purpose, we copy the stash contents and find and remove the latest per dut field iteratively.
        tmp_stash = self if convert_to_db_observations else self.copy()
        to_be_saved = []
        for _ in range(num_saves):
            next_sequence = tmp_stash._get_latest_for_saving(components, observations, tags, drop_from_stash=True)
            if next_sequence:
                to_be_saved.append(LocalObservationSequence(next_sequence))
            else:
                break
        to_be_saved.reverse()

        # Now let's actually save the observations in several batches.
        saved: list[ObservationData] = []
        for seq in to_be_saved:
            if new_tags:
                seq.tag(new_tags)
            saved += seq.save(storage=storage)

        if convert_to_db_observations:
            for obs in saved:
                if obs.dut_field not in self._db_observations:
                    self._db_observations[obs.dut_field] = []
                self._db_observations[obs.dut_field].append(obs)
        return saved

    def _save_latest(
        self,
        *,
        tags: tuple[str, ...] | None = None,
    ) -> list[ObservationData]:
        """Saves the latest LocalObservation for each dut_field in the stash to the DB.

        Args:
            tags: Tags for the saved observations. Any existing tags are cleared.
                None means keep existing tags.

        Returns:
            Saved observations.

        Raises:
            RuntimeError: Saving was not fully successful.

        """
        # TODO batching if there are lots of observations?

        # latest local observation for each dut_field
        observations = [obs_list[-1] for obs_list in self._local_observations.values()]
        if tags is not None:
            # has to be able to handle GBCObservations which are frozen
            observations = [obs.model_copy(update={"tags": list(tags)}) for obs in observations]

        saved = LocalObservationSequence(observations).save()
        if len(saved) != len(observations):
            raise RuntimeError(
                f"Saving was not fully successful: expected to save {len(observations)} observations, "
                f"but only {len(saved)} were saved."
            )
        return saved

    def set_observation_value(self, dut_field: str, value: ObservationValue) -> None:
        """Change the value of an observation in the stash by adding a new LocalObservation.

        Args:
            dut_field: The full DUT field path of the observation, including the component name.
            value: The new value for the observation.

        """
        observation = self.get_latest_observation(dut_field)
        if observation is None:
            raise ValueError(f"Observation '{dut_field}' not found in the stash.")

        observation = LocalObservation.create(
            dut_field=observation.dut_field,
            value=value,
            unit=observation.unit,
            uncertainty=observation.uncertainty,
            invalid=observation.invalid,
            tags=observation.tags,
            configurator="ObservationStash.set_observation_value",
        )
        self.stash(observation)

    def _get_latest_for_saving(
        self,
        components: list[str] | None = None,
        observations: list[str] | None = None,
        tags: list[str] | None = None,
        drop_from_stash: bool = False,
    ) -> list[LocalObservation]:
        to_be_saved = []
        for dut_field_path in self._local_observations:
            if _filter_paths(dut_field_path, components, observations):
                latest_observation = _find_latest_observation(
                    self._local_observations[dut_field_path],  # type: ignore[arg-type]
                    tags,
                    drop_from_stash,
                )
                if latest_observation:
                    to_be_saved.append(latest_observation)
        return to_be_saved  # type: ignore[return-value]

    def _get_observations(
        self,
        components: list[str] | None = None,
        observations: list[str] | None = None,
        tags: list[str] | None = None,
        local: bool = True,
        set_ids: list[uuid.UUID] | None = None,
    ) -> dict[str, list[ObservationData] | list[LocalObservation]]:
        """Get the observations stored in the stash."""
        if tags and set_ids:
            raise ValueError("simultaneous use of tags and observation sets not supported.")
        storage = self._local_observations if local else self._db_observations
        returned = {}
        for obs_path, obs_seq in storage.items():
            if _filter_paths(obs_path, components, observations):
                # Only db observations can belong to an observation set.
                collected = [
                    obs
                    for obs in obs_seq
                    if (
                        (not tags or set(tags).issubset(set(obs.tags)))
                        and (
                            not set_ids
                            or (
                                isinstance(obs, ObservationData) and set(set_ids).issubset(set(obs.observation_set_ids))
                            )
                        )
                    )
                ]
                if len(collected) > 0:
                    returned[obs_path] = collected
        return returned  # type: ignore[return-value]

    def _is_not_yet_stashed(self, observation: ObservationDefinition) -> bool:
        if isinstance(observation, LocalObservation):
            storage = self._local_observations
            dut_field_path = observation.dut_field
        else:
            storage = self._db_observations  # type: ignore[assignment]
            dut_field_path = observation.dut_field
        if dut_field_path not in storage:
            return True
        for stashed_obs in storage[dut_field_path]:
            if stashed_obs.created_timestamp == observation.created_timestamp and np.all(  # type: ignore[attr-defined]
                observation.value == stashed_obs.value
            ):
                return False
        return True


def _find_latest_observation(
    obs_sequence: list[ObservationDefinition],
    tags: list[str] | None = None,
    drop_from_stash: bool = False,
    set_ids: list[uuid.UUID] | None = None,
) -> ObservationDefinition | None:
    latest_timestamp = datetime(MINYEAR, 1, 1, tzinfo=UTC)
    latest_observation = None
    latest_index = 0
    for index, obs in enumerate(obs_sequence):
        utc_timestamp = obs.created_timestamp.astimezone(UTC)  # type: ignore[attr-defined]
        if (
            utc_timestamp > latest_timestamp
            and (not tags or set(tags).issubset(set(obs.tags)))
            and (
                not set_ids
                or isinstance(obs, ObservationData)  # LocalObservations can't belong to observation sets.
                and set(set_ids).issubset(set(obs.observation_set_ids))
            )
        ):
            latest_observation = obs
            latest_timestamp = utc_timestamp
            latest_index = index
    if drop_from_stash and latest_observation:
        obs_sequence.pop(latest_index)  # remove by index to avoid errors from __eq__ comparison
    return latest_observation


def _filter_paths(path: str, components: list[str] | None = None, observations: list[str] | None = None) -> bool:
    """Filter observation paths by components and observations."""
    split_path = path.split(".")
    for ind in range(len(split_path)):
        path_prefix = ".".join(split_path[: ind + 1])
        path_suffix = ".".join(split_path[ind + 1 :])
        if (not components or path_prefix in components) and (not observations or path_suffix in observations):
            return True
    return False


def set_global_observation_stash(rules: list[RuleType] | None) -> ObservationStash:
    """Shorthand to populate an observation stash with given rules, and then set the global loading rules
    to get from that stash.

    Args:
        rules: The rules used to populate the stash, applied when this function is called.

    Returns:
        The populated stash.

    """
    dut_label = ExaRuntime.experiment_configuration().dut_label
    default_storage = ExaRuntime.storage()
    obs_stash = get_loaded_stash_from_rules(rules or [], dut_label=dut_label, default_storage=default_storage)
    set_global_default_observation_loading_rules([LatestFromStash(obs_stash)])  # type: ignore[arg-type]
    return obs_stash
