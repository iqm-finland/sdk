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
from collections.abc import Callable
from typing import Any, Protocol, TypeAlias

from iqm.station_control.interface.models import ObservationBase, ObservationDefinition

RuleType: TypeAlias = Callable[[str], ObservationBase | None]
"""Callable that returns an observation for the given dut_field, or None if it cannot find one."""

RuleCacheDataType: TypeAlias = Any
"""Represents cached data for a single CacheableLoadRule subclass."""

LoadFunction: TypeAlias = Callable[..., RuleCacheDataType]
"""Function for loading :class:`RuleCache` data for a single subclass of CacheableLoadRule."""


class _ObservationStash(Protocol):
    """Dummy for preventing circular import."""

    def get_latest_observation(self, observation_name: str, **kwargs) -> ObservationBase | None:
        """Get the latest observation."""


def set_global_default_observation_loading_rules(rules: list[RuleType]) -> None:
    """Set global default rules which are to be applied to observation handlers in experiments.

    Args:
        rules: The rules to be applied globally.

    """
    if not isinstance(rules, list):
        raise TypeError("Argument 'rules' must be a list")
    if not all(callable(rule) for rule in rules):
        raise TypeError("Every rule in 'rules' must be a callable")

    global _DEFAULT_OBSERVATION_LOADING_RULES  # noqa: PLW0603
    _DEFAULT_OBSERVATION_LOADING_RULES = rules


def get_global_default_observation_loading_rules() -> list[RuleType]:
    """Return the global default loading rules."""
    return _DEFAULT_OBSERVATION_LOADING_RULES


class CacheableLoadRule(ABC):
    """Interface for an observation loading rule for which the observation data can be preloaded to a cache.

    Note that generally load rules do not need to implement this interface. It is enough that they are of the
    type :type:`RuleType`.
    """

    @property
    def full_name(self) -> str:
        """Returns the full name including the attributes."""
        return self.__class__.__name__

    @property
    def attributes(self) -> list[Any]:
        """Returns the attributes."""
        return []

    @abstractmethod
    def __call__(self, observation_name: str) -> ObservationBase | None:
        """Applies the rule to (potentially) return an observation."""
        raise NotImplementedError

    @abstractmethod
    def resolve_from_cache(
        self,
        observation_name: str,
        cache_data: RuleCacheDataType,
    ) -> ObservationDefinition | None:
        """Applies the rule for preloaded cache."""
        raise NotImplementedError


_DEFAULT_OBSERVATION_LOADING_RULES: list[RuleType] = []


class Fail:
    """Terminates rule application chain on error."""

    def __call__(self, observation_name: str) -> ObservationBase | None:
        raise ValueError(f"Observation {observation_name} was not resolved")


class LatestFromStash:
    """Load rule that gets the observation from the local stash."""

    def __init__(self, stash: _ObservationStash, tags: list[str] | None = None):
        self.stash = stash
        self.tags = tags

    def __call__(self, observation_name: str) -> ObservationBase | None:
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
    rules: list[RuleType],
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
                loaded_observation = rule.resolve_from_cache(observation_name, rule_cache.data[rule_name])
            else:
                loaded_observation = rule(observation_name)
            if loaded_observation:
                if isinstance(rule, CacheableLoadRule):
                    rule_name = rule.full_name
                return loaded_observation, rule_name
        except ValueError as err:
            raise ValueError(f"Applying rules {rules} for {observation_name} failed:") from err
    return None, None
