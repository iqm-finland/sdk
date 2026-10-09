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
"""Loading rule interfaces and base classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias
import uuid

from typing_extensions import deprecated

from exa.common.helpers.deprecation import format_deprecated
from iqm.cpc.core.config import ExperimentConfiguration
from iqm.cpc.core.observation.observation_load_utils import (
    _EXP_CONFIG_NODES_AND_GLOBAL_KEYS,
    _GLOBAL_KEY_STR_REPR,
    _load_latest_characterization_set_observations_cache,
    get_latest_observation,
    populate_from_configs,
)
from iqm.station_control.interface.models import ObservationBase, ObservationData, ObservationDefinition
from iqm.station_control.interface.models.type_aliases import StrUUIDOrDefault
from iqm.station_control.interface.storage_interface import StorageInterface

RuleCacheDataType: TypeAlias = Any
"""Represents cached data for a single CacheableLoadRule subclass."""

LoadFunction: TypeAlias = Callable[..., RuleCacheDataType]
"""Function for loading :class:`RuleCache` data for a single subclass of CacheableLoadRule."""

DEFAULTS_FROM_YML_SOURCE = {"type": "configuration_source", "configurator": "exa_yml"}
"""Source for settings that get their values from the EXA configuration YML."""


class RuleType(ABC):
    """Base class for observation loading rules.

    Args:
        storage: The storage backend the rule loads observations from. Rules that don't connect to a storage
            backend can leave this as ``None`` (default).

    """

    def __init__(self, storage: StorageInterface | None = None) -> None:
        self.storage = storage

    @property
    def attributes(self) -> tuple[Any, ...]:
        """Returns the attributes, including the storage backend.

        The storage backend is always the last attribute.
        """
        return (self.storage,)

    @abstractmethod
    def __call__(
        self, observation_name: str, dut_label: str, default_storage: StorageInterface | None
    ) -> ObservationBase | None:
        """Apply the rule to (potentially) return an observation.

        Args:
            observation_name: The ``dut_field`` of the observation to be loaded.
            dut_label: The label of the DUT for which the observation is being loaded.
            default_storage: The default storage backend to use if the rule does not have its own storage backend.

        Returns:
            The loaded observation, or ``None`` if the rule could not load the observation.

        """
        raise NotImplementedError


class _ObservationStash(Protocol):
    """Dummy for preventing circular import."""

    def get_latest_observation(self, observation_name: str, **kwargs) -> ObservationBase | None:
        """Get the latest observation."""


@dataclass
class _ObservationSetStash:
    """Minimal ObservationStash adapter that can be used with the :class:`.LatestFromStash` observation loading rule."""

    observations: dict[str, ObservationBase]
    """Observation.dut_field mapped to the observation itself."""

    def get_latest_observation(self, observation_name: str, **kwargs) -> ObservationBase | None:
        """Get the observation value if it exists, otherwise ``None``."""
        return self.observations.get(observation_name)


class CacheableLoadRule(RuleType):
    """Interface for an observation loading rule for which the observation data can be preloaded to a cache.

    Note that generally load rules do not need to implement this interface — it is enough that they
    subclass :class:`RuleType`.
    """

    @property
    def full_name(self) -> str:
        """Returns the full name including the attributes."""
        return self.__class__.__name__

    @abstractmethod
    def resolve_from_cache(
        self, observation_name: str, cache_data: RuleCacheDataType, default_storage: StorageInterface | None
    ) -> ObservationDefinition | None:
        """Applies the rule for preloaded cache.

        Args:
            observation_name: The ``dut_field`` of the observation to be loaded.
            cache_data: The cache data to be used for loading the observation.
            default_storage: The default storage backend to use if the rule does not have its own storage backend.

        Returns:
            The loaded observation, or ``None`` if the rule could not load the observation.

        """
        raise NotImplementedError


_DEFAULT_OBSERVATION_LOADING_RULES: list[RuleType] = []


class Fail(RuleType):
    """Terminates rule application chain on error."""

    def __call__(
        self, observation_name: str, dut_label: str, default_storage: StorageInterface | None
    ) -> ObservationBase | None:
        """Apply the rule to (potentially) return an observation."""
        raise ValueError(f"Observation {observation_name} was not resolved")


class LatestFromStash(RuleType):
    """Load rule that gets the observation from the local stash."""

    def __init__(self, stash: _ObservationStash, tags: list[str] | None = None):
        super().__init__()
        self.stash = stash
        self.tags = tags

    def __call__(
        self, observation_name: str, dut_label: str, default_storage: StorageInterface | None
    ) -> ObservationBase | None:
        """Apply the rule to (potentially) return an observation."""
        return self.stash.get_latest_observation(observation_name, tags=self.tags)


class RuleCache:
    """Stores cached observation values for the CacheableLoadingRule subclasses.

    Args:
        load_function_mapping: Maps the load rule names to the functions used to load the cached observations.
            Note that each rule should have its own function/database query as they are (generally) not applicable
            for other rules.

    Attributes:
        data: maps rule name to the loaded data for that rule
        load_function_mapping: maps the rule names to the associated load functions.

    """

    def __init__(self, load_function_mapping: dict[str, LoadFunction]):
        self.data: dict[str, RuleCacheDataType] = {}
        self.load_function_mapping = load_function_mapping

    def load_cache(self, rule_name_to_attributes: dict[str, list[Any]]) -> None:
        """Loops through the mapped load functions and applies them to load the cache.

        If the rule has attributes, those are passed to the load function.

        Args:
            rule_name_to_attributes: the rule names mapped to the attributes of that rule.

        """
        for rule_name, load_func in self.load_function_mapping.items():
            if (attributes := rule_name_to_attributes.get(rule_name)) is not None:
                if attributes:
                    self.data[rule_name] = load_func(attributes)
                else:
                    self.data[rule_name] = load_func()


def resolve_rules(
    observation_name: str,
    rules: Sequence[RuleType],
    dut_label: str,
    default_storage: StorageInterface | None | None,
    rule_cache: RuleCache | None = None,
) -> tuple[ObservationBase, str] | tuple[None, None]:
    """Resolves the rules related to loading and fall back one by one.

    If an observation is loaded successfully from the DB, the rule resolving is terminated.
    Returns the loaded observation and the rule that ended up producing it.

    If global observation loading rules has been set with :func:`.set_global_default_observation_loading_rules`,
    then those rules will override anything given in ``rules``.
    Global observation rules can be set to an empty list, in that case no rules will be used,
    i.e. observation will not be loaded at all.
    If global observations rules are set to None, then those will be ignored
    and original rules given as a parameter will be used.

    Args:
        observation_name: the name of the observation to search for in the database.
        rules: the list of rules to be resolved in loading the observation.
        dut_label: the label of the DUT for which this observation will be loaded.
        default_storage: the default storage to be used for loading observations if the rule does not have its own
            storage backend.
        rule_cache: cache of preloaded observations for certain load rules. Loading from the rule cache takes
            precedence over the normal rule execution.

    Returns:
        - The observation that was loaded (None if no rule successfully fetched an observation).
        - The name of the rule class that produced the above observation
          (None if no rule successfully fetched an observation).

    Raises:
        ValueError: in case a rule could not be applied and a ValueError was thrown.

    """
    loaded_observation: ObservationBase | None
    for rule in rules:
        rule_name = rule.__class__.__name__
        try:
            if rule_cache and isinstance(rule, CacheableLoadRule) and rule_name in rule_cache.data:
                loaded_observation = rule.resolve_from_cache(
                    observation_name, rule_cache.data[rule_name], default_storage
                )
            else:
                loaded_observation = rule(observation_name, dut_label, default_storage)
            if loaded_observation:
                if isinstance(rule, CacheableLoadRule):
                    rule_name = rule.full_name
                return loaded_observation, rule_name
        except ValueError as err:
            raise ValueError(f"Applying rules {rules} for {observation_name} failed:") from err
    return None, None


class FromExperimentConfiguration(CacheableLoadRule):
    """Load observations from :class:`.ExperimentConfiguration`.

    The settings tree root nodes ("controllers", "gates", "characterization", "stages") are given as attrs in
    the :class:`.ExperimentConfiguration`. These are dicts that reflect the settings tree path structure. For each
    ``dut_field``, this rule picks the corresponding root attribute and traverses the dict to find the value. Global
    default values for a component/locus are used if they exist and no specific value for that component/locus does.
    If no applicable value or global default value is found a ``None`` is returned.

    The global default dict key is `"common"` for the "controllers" node, and an empty tuple for the other nodes.

    """

    def __init__(self, config: ExperimentConfiguration, storage: StorageInterface | None = None):
        if storage is not None:
            raise ValueError(
                "FromExperimentConfiguration reads from an in-memory ExperimentConfiguration and does not use a "
                "storage backend; do not pass 'storage'."
            )
        super().__init__()
        self.config = config

    @property
    def attributes(self) -> tuple[ExperimentConfiguration, None]:
        """The experiment configuration the rule loads the observations from (no storage backend)."""
        return (self.config, None)

    def __call__(
        self, observation_name: str, dut_label: str, default_storage: StorageInterface | None
    ) -> ObservationDefinition | None:
        """Apply the rule to (potentially) return an observation."""
        data = populate_from_configs([((self.config,), None)], dut_label)[self.config.id]
        return self._call(observation_name, data)

    def resolve_from_cache(
        self,
        observation_name: str,
        cache_data: dict[uuid.UUID, dict[str, Any]],
        default_storage: StorageInterface | None,
    ) -> ObservationDefinition | None:
        """Applies the rule for preloaded cache."""
        return self._call(observation_name, cache_data[self.config.id])

    def _call(self, observation_name: str, data: dict[str, Any]) -> ObservationDefinition | None:
        path = observation_name.split(".")
        if path[0] not in _EXP_CONFIG_NODES_AND_GLOBAL_KEYS:
            return None
        traversed_node = getattr(self.config, path[0])
        paths = [path]
        for frag_idx, frag in enumerate(path[1:]):
            split_frag = frag.split("__")
            if len([f for f in split_frag if f in self.config.components]) == len(split_frag):
                # this is a locus node of the path (contains just QPU components), so we need to also search with the
                # corresponding global locus path. Let's add that to the paths
                global_path = path.copy()
                global_path[frag_idx + 1] = _GLOBAL_KEY_STR_REPR
                paths.append(global_path)
                break
            elif frag not in traversed_node:
                return None
            traversed_node = traversed_node[frag]

        for path in paths:
            path_str = ".".join(path)
            if path_str in data:
                return ObservationDefinition(
                    dut_field=observation_name,
                    value=data[path_str],
                    unit="",  # FIXME: unit None (we can't get this from the yml, must fix when saving to DB).
                    dut_label=self.config.dut_label,
                    source=DEFAULTS_FROM_YML_SOURCE,
                )
        return None


class GetLatest(CacheableLoadRule):
    """Get the latest observation."""

    def __call__(
        self, observation_name: str, dut_label: str, default_storage: StorageInterface | None
    ) -> ObservationDefinition | None:
        """Apply the rule to (potentially) return an observation."""
        storage = self.storage or default_storage
        if storage is None:
            raise ValueError(
                "GetLatest rule requires a storage backend to be set either in the rule or as the default storage."
            )
        return get_latest_observation(
            observation_name,
            storage=storage,
            dut_label=dut_label,  # type: ignore[arg-type]
        )

    def resolve_from_cache(
        self,
        observation_name: str,
        cache_data: dict[StorageInterface | None, dict[str, ObservationDefinition]],
        default_storage: StorageInterface | None,
    ) -> ObservationDefinition | None:
        """Applies the rule for preloaded cache."""
        return cache_data.get(self.storage or default_storage, {}).get(observation_name)


class GetLatestForTags(CacheableLoadRule):
    """Get the latest observation for a given list of tags.

    The observation must have all the defined tags in order to be loaded, but it may also contain additional tags.
    """

    def __init__(self, tags: list[str], storage: StorageInterface | None = None):
        super().__init__(storage=storage)
        self._tags = sorted(tags)

    @property
    def full_name(self) -> str:
        """Returns the full name including the attributes."""
        return f"{self.__class__.__name__}({self._tags})"

    @property
    def attributes(self) -> tuple[tuple[str, ...], StorageInterface | None]:
        """The tags and the storage backend the rule loads the observations from."""
        return (tuple(self._tags), self.storage)

    def __call__(
        self, observation_name: str, dut_label: str, default_storage: StorageInterface | None
    ) -> ObservationData | None:
        """Apply the rule to (potentially) return an observation."""
        return get_latest_observation(
            observation_name,
            tags=self._tags,
            storage=self.storage or default_storage,  # type: ignore[arg-type]
            dut_label=dut_label,
        )

    def resolve_from_cache(
        self,
        observation_name: str,
        cache_data: dict[StorageInterface | None, dict[str, list[ObservationData]]],
        default_storage: StorageInterface | None,
    ) -> ObservationData | None:
        """Applies the rule for preloaded cache.

        Args:
            observation_name: The name of the observation.
            cache_data: Cache data for loading the observation.
            default_storage: Default storage for loading the observation in case the rule does not have its own storage
                backend.

        Returns:
            The loaded observation or ``None`` if the rule could not load the observation.

        """
        latest_observation = None
        observations_by_dut_field = cache_data.get(self.storage or default_storage, {})
        if observations := observations_by_dut_field.get(observation_name):
            self_tags = set(self._tags)
            latest_timestamp = None
            for observation in observations:
                if self_tags.issubset(set(observation.tags)) and (
                    latest_timestamp is None or observation.created_timestamp > latest_timestamp
                ):
                    latest_timestamp = observation.created_timestamp
                    latest_observation = observation
        return latest_observation


class GetObservationSet(CacheableLoadRule):
    """Get the observations for a given observation set.

    The observations must belong to the specified set, but may also belong to others.

    Raises: ValueError in case there are several observations for the same dut_field, or observations in the set belong
        to more than one DUT.
        If the set is of type 'generic-set' a warning is shown instead and the set is loaded as usual.
    """

    def __init__(
        self,
        set_id: uuid.UUID | str,
        storage: StorageInterface | None = None,
    ):
        super().__init__(storage=storage)
        if isinstance(set_id, str):
            set_id = uuid.UUID(set_id)
        self._set_ids = [set_id]

    @property
    def full_name(self) -> str:
        """Returns the full name including the attributes."""
        return f"{self.__class__.__name__}_for_id_{self.set_id}"

    @property
    def attributes(self) -> tuple[uuid.UUID, StorageInterface | None]:
        """The observation set id and the storage backend the rule loads the observations from."""
        return (self.set_id, self.storage)

    @property
    def set_id(self) -> uuid.UUID:
        """The observation set id the loading rule was initialized with."""
        return self._set_ids[0]

    def __call__(
        self, observation_name: str, dut_label: str, default_storage: StorageInterface | None
    ) -> ObservationData | None:
        """Apply the rule to (potentially) return an observation."""
        return get_latest_observation(
            observation_name,
            set_id=self.set_id,
            storage=self.storage or default_storage,  # type: ignore[arg-type]
            dut_label=dut_label,
        )

    def resolve_from_cache(
        self,
        observation_name: str,
        cache_data: dict[StorageInterface | None, dict[str, ObservationData]],
        default_storage: StorageInterface | None,
    ) -> ObservationData | None:
        """Applies the rule for preloaded cache."""
        observations_by_dut_field = cache_data.get(self.storage or default_storage, {})
        if (
            observation := observations_by_dut_field.get(observation_name)
        ) and self.set_id in observation.observation_set_ids:
            return observation
        return None


class GetCalibrationSet(CacheableLoadRule):
    """Get the observations of a specific calibration set, or the current default one.

    Maps directly to :meth:`.StorageInterface.get_calibration_set`.

    Args:
        calibration_set_id: ID of the calibration set to get.
        storage: The storage backend the rule loads the observations from. If ``None``, the default storage
            will be used.

    """

    def __init__(
        self,
        calibration_set_id: StrUUIDOrDefault,
        storage: StorageInterface | None = None,
    ):
        super().__init__(storage=storage)
        self._calibration_set_id = calibration_set_id

    @property
    def full_name(self) -> str:
        """Returns the full name including the attributes."""
        return f"{self.__class__.__name__}_for_id_{self.calibration_set_id}"

    @property
    def attributes(self) -> tuple[StrUUIDOrDefault, StorageInterface | None]:
        """The calibration set id and the storage backend the rule loads the observations from."""
        return (self.calibration_set_id, self.storage)

    @property
    def calibration_set_id(self) -> StrUUIDOrDefault:
        """The calibration set id the loading rule was initialized with."""
        return self._calibration_set_id

    def __call__(
        self, observation_name: str, dut_label: str, default_storage: StorageInterface | None
    ) -> ObservationData | None:
        """Apply the rule to (potentially) return an observation."""
        return None  # If it was not in the cache ( = calset), there is nothing we can do.

    def resolve_from_cache(
        self,
        observation_name: str,
        cache_data: dict[StorageInterface | None, dict[StrUUIDOrDefault, dict[str, ObservationData]]],
        default_storage: StorageInterface | None,
    ) -> ObservationData | None:
        """Applies the rule for preloaded cache."""
        return (
            cache_data.get(self.storage or default_storage, {}).get(self.calibration_set_id, {}).get(observation_name)
        )


class GetQualityMetricSet(CacheableLoadRule):
    """Get the observations of a specific calibration set's latest quality metric set.

    Maps directly to :meth:`.StorageInterface.get_calibration_set_quality_metric_set`.

    Args:
        calibration_set_id: ID of the calibration set whose latest quality metric set to get.

    """

    def __init__(
        self,
        calibration_set_id: StrUUIDOrDefault,
        storage: StorageInterface | None = None,
    ):
        super().__init__(storage=storage)
        self._calibration_set_id = calibration_set_id

    @property
    def full_name(self) -> str:
        """Returns the full name including the attributes."""
        return f"{self.__class__.__name__}_for_id_{self.calibration_set_id}"

    @property
    def attributes(self) -> tuple[StrUUIDOrDefault, StorageInterface | None]:
        """The calibration set id and the storage backend the rule loads the observations from."""
        return (self.calibration_set_id, self.storage)

    @property
    def calibration_set_id(self) -> StrUUIDOrDefault:
        """The calibration set id the loading rule was initialized with."""
        return self._calibration_set_id

    def __call__(
        self, observation_name: str, dut_label: str, default_storage: StorageInterface | None
    ) -> ObservationData | None:
        """Apply the rule to (potentially) return an observation."""
        return None  # If it was not in the cache ( = quality metric set), there is nothing we can do.

    def resolve_from_cache(
        self,
        observation_name: str,
        cache_data: dict[StorageInterface | None, dict[StrUUIDOrDefault, dict[str, ObservationData]]],
        default_storage: StorageInterface | None,
    ) -> ObservationData | None:
        """Applies the rule for preloaded cache."""
        return (
            cache_data.get(self.storage or default_storage, {}).get(self.calibration_set_id, {}).get(observation_name)
        )


@deprecated(format_deprecated(old="GetDefaultCalibrationSet", new="GetCalibrationSet('default')", since="2026-09-01"))
class GetDefaultCalibrationSet(CacheableLoadRule):
    """Get the observations for the default calibration set.

    The default calibration set results from a calibration run and contains all observations needed for successful
    circuit execution.
    """

    def __call__(
        self, observation_name: str, dut_label: str, default_storage: StorageInterface | None
    ) -> ObservationDefinition | None:
        """Apply the rule to (potentially) return an observation."""
        return None  # If it was not in the cache ( = calset), there is nothing we can do.

    def resolve_from_cache(
        self,
        observation_name: str,
        cache_data: dict[StorageInterface | None, dict[str, ObservationDefinition]],
        default_storage: StorageInterface | None,
    ) -> ObservationDefinition | None:
        """Applies the rule for preloaded cache."""
        return cache_data.get(self.storage or default_storage, {}).get(observation_name)


class GetLatestCharacterizationSet(CacheableLoadRule):
    """Get the observations for the latest finalized characterization set.

    A characterization set results from a calibration run.
    It is a superset of the calibration set, containing observations that aid the calibration process itself.
    """

    def __call__(
        self, observation_name: str, dut_label: str, default_storage: StorageInterface | None
    ) -> ObservationData | None:
        """Apply the rule to (potentially) return an observation."""
        return None  # If it was not in the cache ( = char set), there is nothing we can do.

    @staticmethod
    def wrap_load(
        storages: list[StorageInterface | None], dut_label: str
    ) -> dict[StorageInterface, dict[str, ObservationData]]:
        """Load the cache for this rule, one storage backend at a time."""
        return _load_latest_characterization_set_observations_cache(storages, dut_label)  # type: ignore[arg-type]

    def resolve_from_cache(
        self,
        observation_name: str,
        cache_data: dict[StorageInterface | None, dict[str, ObservationData]],
        default_storage: StorageInterface | None,
    ) -> ObservationData | None:
        """Applies the rule for preloaded cache."""
        return cache_data.get(self.storage or default_storage, {}).get(observation_name)
