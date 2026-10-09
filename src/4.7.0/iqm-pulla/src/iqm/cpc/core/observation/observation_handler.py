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
"""Base class for handling observation loading."""

from collections import defaultdict
from collections.abc import Iterable, Sequence
from functools import partial
import logging
from math import log
import re
from typing import Any

from exa.common.data.parameter import Parameter
from exa.common.data.setting_node import SettingNode
from iqm.cpc.core.observation.observation_load_utils import (
    _load_calibration_set_observations_cache,
    _load_calibration_set_quality_metric_set_observations_cache,
    _load_default_calibration_set_observations_cache,
    _load_latest_for_tags_observations_cache,
    _load_latest_observations_cache,
    _load_observation_set_observations_cache,
    populate_from_configs,
)
from iqm.cpc.core.observation.observation_loading_rules import (
    CacheableLoadRule,
    LoadFunction,
    RuleCache,
    RuleType,
    resolve_rules,
)
from iqm.station_control.interface.models import ObservationBase
from iqm.station_control.interface.storage_interface import StorageInterface

from .observation_loading_rules import (
    FromExperimentConfiguration,
    GetCalibrationSet,
    GetDefaultCalibrationSet,
    GetLatest,
    GetLatestCharacterizationSet,
    GetLatestForTags,
    GetObservationSet,
    GetQualityMetricSet,
)

logger = logging.getLogger(__name__)

_LOAD_FUNCTION_MAPPING: dict[str, LoadFunction] = {
    GetLatest.__name__: _load_latest_observations_cache,
    GetLatestForTags.__name__: _load_latest_for_tags_observations_cache,
    GetObservationSet.__name__: _load_observation_set_observations_cache,
    GetDefaultCalibrationSet.__name__: _load_default_calibration_set_observations_cache,
    GetLatestCharacterizationSet.__name__: GetLatestCharacterizationSet.wrap_load,
    GetCalibrationSet.__name__: _load_calibration_set_observations_cache,
    GetQualityMetricSet.__name__: _load_calibration_set_quality_metric_set_observations_cache,
    FromExperimentConfiguration.__name__: populate_from_configs,
}


def load_observations_by_rules(
    load_rules: Sequence[RuleType], dut_label: str, default_storage: StorageInterface
) -> dict[str, ObservationBase]:
    """Load observations resolved by the given rules, keyed by whichever dut fields they resolve to.

    Unlike populating an :class:`.ObservationHandler` with every known dut field, this only touches the dut
    fields that the given ``load_rules`` themselves surface from the DB, so it works without any prior
    knowledge of which dut fields exist for the DUT. As a consequence, a rule that isn't a
    :class:`.CacheableLoadRule` (e.g. :class:`.Fail`, :class:`.LatestFromStash`) never contributes any dut
    field to resolve against, since there's no cached query result to discover them from. If ``load_rules``
    contains no ``CacheableLoadRule`` at all, a warning is logged and no observations are returned.

    Args:
        load_rules: Loading rules to resolve, tried in order per dut field.
        dut_label: DUT to resolve the rule for.
        default_storage: Default storage to use for loading observations in case the rules do not pin a storage backend.


    Returns:
        Mapping from resolved dut field paths to their observations.

    """
    if load_rules and not any(isinstance(rule, CacheableLoadRule) for rule in load_rules):
        logger.warning(
            "None of the given load_rules (%s) are cacheable, so no dut fields can be discovered to resolve "
            "them against. No observations will be loaded.",
            [rule.__class__.__name__ for rule in load_rules],
        )

    cache = RuleCache({n: partial(f, dut_label=dut_label) for n, f in _LOAD_FUNCTION_MAPPING.items()})
    rule_name_to_attributes = _get_attrs_by_rule_name(load_rules, default_storage)
    cache.load_cache(rule_name_to_attributes)

    dut_fields: set[str] = set()
    for rule in load_rules:
        rule_name = rule.__class__.__name__
        if not (isinstance(rule, CacheableLoadRule) and rule_name in cache.data):
            continue
        rule_cache_data = cache.data[rule_name]
        # Every cacheable rule's cache is keyed by storage first.
        storage_scoped_cache = rule_cache_data.get(rule.storage or default_storage, {})
        if isinstance(rule, (GetCalibrationSet, GetQualityMetricSet)):
            # ...and by calibration_set_id second.
            dut_fields.update(storage_scoped_cache.get(rule.calibration_set_id, {}))
        else:
            dut_fields.update(storage_scoped_cache)

    resolved: dict[str, ObservationBase] = {}
    for dut_field in dut_fields:
        observation, _ = resolve_rules(dut_field, load_rules, dut_label, default_storage, cache)
        if observation:
            resolved[dut_field] = observation
    return resolved


def _get_attrs_by_rule_name(
    load_rules: Sequence[RuleType], default_storage: StorageInterface | None
) -> dict[str, list[Any]]:
    """Return a mapping from rule names to their attributes for caching.

    The mapping's values ore lists of tuples ``((attr1, attr2, ...), storage)`` for each unique combination of
    attributes and storage for that rule name. The attributes are the rule's attributes excluding the storage,
    which is always the last attribute of a rule. THe attributes are aggregated across all rules of the same name and
    the storage implementation.
    """
    attrs_per_rule_and_storage: dict[tuple[str, StorageInterface | None], list[Any]] = defaultdict(list)
    for rule in load_rules:
        rule_name = rule.__class__.__name__
        if isinstance(rule, CacheableLoadRule):
            # storage is the last attribute of the rule, if it is None, use the default storage.
            storage = rule.attributes[-1] if rule.attributes[-1] is not None else default_storage
            for attr in rule.attributes[:-1]:
                if attr not in attrs_per_rule_and_storage[(rule_name, storage)]:
                    attrs_per_rule_and_storage[(rule_name, storage)].append(attr)
            if not rule.attributes[:-1]:
                # rule has no attributes beside storage, so we still need to add the storage to the dict.
                attrs_per_rule_and_storage[(rule_name, storage)] = []

    unique_rules: dict[str, list[tuple[Any, ...]]] = defaultdict(list)
    for rule_and_storage, attrs in attrs_per_rule_and_storage.items():
        if not attrs:
            unique_rules[rule_and_storage[0]].append((rule_and_storage[1],))
        else:
            unique_rules[rule_and_storage[0]].append((tuple(attrs), rule_and_storage[1]))
    return unique_rules


class ObservationHandler:
    """Base class for loading observations from the DB or another source.

    The load_rules of for a given set of observations can be set and the observations loaded accordingly.

    Args:
        observation_parameters: list of observation names or parameters.
        dut_label: The label of the DUT for which this ObservationHandler is associated with.
        default_storage: The default storage to be used for loading observations when the load rules
            do not specify a storage.
        load_rules: The loading rules to be set for the above ``observation_parameters``.
        rule_cache: optional user inputted custom rule cache.

    Attributes:
        load_rules: The load rules for each observation.
        loaded_observations: The loaded observation and the load rule that produced it mapped to the observation
            ``dut_field``.

    """

    def __init__(
        self,
        observation_parameters: Iterable[Parameter | str],
        dut_label: str,
        default_storage: StorageInterface | None,
        load_rules: Sequence[RuleType] | None = None,
        rule_cache: RuleCache | None = None,
    ):
        self.dut_label = dut_label
        self.default_storage = default_storage
        self.default_rules = load_rules or []
        self.load_rules: dict[str, Sequence[RuleType]] = {
            p.name if isinstance(p, Parameter) else p: self.default_rules for p in observation_parameters
        }
        self.loaded_observations: dict[str, tuple[ObservationBase, str] | tuple[None, None]] = {
            p.name if isinstance(p, Parameter) else p: (None, None) for p in observation_parameters
        }
        self._cache = rule_cache or RuleCache(
            {n: partial(f, dut_label=dut_label) for n, f in _LOAD_FUNCTION_MAPPING.items()}
        )

    def _filter_observation_names(
        self,
        matches: list[str] | None = None,
        starts_with: list[str] | None = None,
        includes: list[str] | None = None,
    ) -> list[str]:
        """Filter the observation names."""
        obs_names = list(self.loaded_observations.keys())
        if matches:
            obs_names = [n for n in obs_names if n in matches]
        if starts_with:
            obs_names = [n for n in obs_names if max(re.match(s, n) is not None for s in starts_with)]
        if includes:
            obs_names = [n for n in obs_names if max(re.search(i, n) is not None for i in includes)]
        return obs_names

    def set_load_rules(
        self,
        load_rules: Sequence[RuleType],
        matches: list[str] | None = None,
        starts_with: list[str] | None = None,
        includes: list[str] | None = None,
    ) -> None:
        """Set given load rule to specific observations.

        Args:
            load_rules: the rules to be set.
            matches: Apply function to only the observations whose names that match one of the strings exactly. Does
                not support regex.
            starts_with: Apply to only the observations whose names start with one of these strings. Supports regex.
            includes: Apply to only the observations whose names contain one of the given strings. Supports regex.

        """
        obs_names = self._filter_observation_names(matches=matches, starts_with=starts_with, includes=includes)
        for name in obs_names:
            self.load_rules[name] = load_rules

    def change_load_rules(self, load_rules: list[RuleType]) -> None:
        """Change all the non-empty load rules into the provided rules.

        Args:
            load_rules: the rules to be applied.

        """
        for name, previous_rules in self.load_rules.items():
            if previous_rules:
                self.load_rules[name] = load_rules

    def load_observations(
        self,
        matches: list[str] | None = None,
        starts_with: list[str] | None = None,
        includes: list[str] | None = None,
        load_rule_cache: bool = True,
    ) -> None:
        """Load the observations for the given components based on their loading rules.

        In case a list of component names is not provided, load observations for all components.
        Allows preloading the observation into the rule cache for certain load rules in order improve the performance.

        Args:
            matches: Apply function to only the observations whose names match one of these strings exactly. Does
                not support regex.
            starts_with: Apply to only the observations whose names start with one of these strings. Supports regex.
            includes: Apply to only the observations whose names contain one of the given strings. Supports regex.
            load_rule_cache: Iff True, preload observations into the cache.

        """
        obs_names = self._filter_observation_names(matches, starts_with, includes)
        if load_rule_cache and self._cache is not None:
            rule_name_to_attributes = self.get_rule_names_and_attributes(matches, starts_with, includes)
            self._cache.load_cache(rule_name_to_attributes)
        for name in obs_names:
            self.loaded_observations[name] = resolve_rules(
                name, self.load_rules[name], self.dut_label, self.default_storage, self._cache
            )

    def print_observations(
        self,
        matches: list[str] | None = None,
        starts_with: list[str] | None = None,
        includes: list[str] | None = None,
    ) -> None:
        """Prints the observations and the load rule that produces them.

        Args:
            matches: Apply function to only the observations whose names that match one of these strings exactly. Does
                not support regex.
            starts_with: Apply to only the observations whose names start with one of these strings. Supports regex.
            includes: Apply to only the observations whose names contain one of the given strings. Supports regex.

        """
        obs_names = self._filter_observation_names(matches, starts_with, includes)
        obs_names.sort()
        max_len = 0
        value_strings: dict[str, str] = {}
        printout = ""
        for name in obs_names:
            loaded_observation = self.loaded_observations[name]
            if loaded_observation[0] is not None:
                value_string = f"{name}={loaded_observation[0].value}"
                if loaded_observation[0].uncertainty is not None:
                    uncertainty = loaded_observation[0].uncertainty
                    uncertainty = round(uncertainty, -round(log(uncertainty, 10) - 2)) if uncertainty > 0 else 0.0  # type: ignore[arg-type,operator]
                    value_string += " " + "\u00b1" + " " + f"{uncertainty}"
                unit = loaded_observation[0].unit
                if unit:
                    value_string += f" {unit}"
                value_strings[name] = value_string
                max_len = max(len(value_string), max_len)
        if len(value_strings) == 0:
            return
        for name, value_string in value_strings.items():
            printout += value_string
            printout += (max_len + 1 - len(value_string)) * " "
            printout += f"  | load_rule: {self.loaded_observations[name][1]}\n"
        print(printout)

    def add_observations(
        self,
        observation_parameters: Iterable[str | Parameter],
        load_rules: Sequence[RuleType] | None = None,
        load: bool = False,
    ) -> None:
        """Adds observations, load rules for them and optionally loads them.

        Args:
            observation_parameters: The observation dut_fields or Parameters to be added to.
            load_rules: Sets the given load rules for the aforementioned paths/dut_fields.
                The global default rules will be used if left ``None``.
            load: If set ``True`` will load the newly added observations using the specified ``load_rules``.
                Note: for cached load rules, assumes the cache is loaded before this call and will not reload it.

        """
        load_rules = self.default_rules if load_rules is None else load_rules
        obs_names = []
        for param in observation_parameters:
            name = param.name if isinstance(param, Parameter) else param
            self.load_rules[name] = load_rules
            self.loaded_observations[name] = (None, None)
            obs_names.append(name)
        if load:
            self.load_observations(matches=obs_names, load_rule_cache=False)

    def populate_from(
        self, settings: SettingNode, load_rules: Sequence[RuleType] | None = None, override_rules: bool = False
    ) -> None:
        """Set up the handler's loading from a settings tree.

        Will add all the settings tree paths to be loaded.

        Args:
            settings: The settings tree from which to read the paths.
            load_rules: Sets the given load rules for the aforementioned paths/dut_fields.
                The global default rules will be used if left ``None``.
            override_rules: If ``True``, will override any already existing rules with the given ``load_rules``.

        """
        paths = []

        def _get_paths(prefix: str | None, branch: SettingNode) -> None:
            for key, child in branch.children.items():
                new_prefix = key if prefix is None else f"{prefix}.{key}"
                if isinstance(child, SettingNode):
                    _get_paths(new_prefix, child)
                else:
                    paths.append(new_prefix)

        new_load_rules = self.default_rules if load_rules is None else load_rules
        _get_paths(None, settings)
        if not override_rules:
            paths = [p for p in paths if p not in self.load_rules]
        self.add_observations(observation_parameters=paths, load_rules=new_load_rules)

    def get_value(self, name: str, load: bool = False, load_rules: Sequence[RuleType] | None = None) -> Any:
        """Get the loaded observation value for a given observation path.

        If the specified observation is not yet found in ``self``, it can optionally be loaded first and then returned.

        Args:
            name: Observation path.
            load: Whether to try to add & load the observation if no value is found. Note: for cached load rules,
                assumes the rule cache has been loaded before this call and will not reload it.
            load_rules: Load rules to add for the observation named ``name`` in case it was not yet found in ``self``.
                If not given, the global default rules are used. Has no effect if ``load == False``.

        Returns:
            The observation value or ``None`` if the observation was not found in ``self``.

        """
        if load:
            if name not in self.load_rules or name not in self.loaded_observations:
                self.add_observations([name], load_rules=load_rules, load=True)
            elif self.loaded_observations[name][0] is None:
                self.load_observations(matches=[name], load_rule_cache=False)
        obs_and_rule = self.loaded_observations.get(name, None)
        if obs_and_rule is None:
            return None
        return obs_and_rule[0].value if obs_and_rule[0] else None

    def __getitem__(self, item: str):
        return self.get_value(item, load=False, load_rules=None)

    def value_dict(
        self,
        matches: list[str] | None = None,
        starts_with: list[str] | None = None,
        includes: list[str] | None = None,
        return_full_observation: bool = False,
    ) -> dict[str, Any]:
        """Return loaded observation values mapped to the respective dut_fields.

        Args:
            matches: Apply function to only the observations whose names that match one of these strings exactly. Does
                not support regex.
            starts_with: Apply to only the observations whose names start with one of these strings. Supports regex.
            includes: Apply to only the observations whose names contain one of the given strings. Supports regex.
            return_full_observation: Return the full observations in the dict instead of just the observation values.

        Returns:
            A dictionary mapping the loaded values to the respective dut_fields. ``None`` values are skipped
                and not found in the dict.

        """
        obs_names = self._filter_observation_names(matches=matches, starts_with=starts_with, includes=includes)
        return {
            name: loaded[0] if return_full_observation else loaded[0].value
            for name, loaded in self.loaded_observations.items()
            if loaded[0] is not None and name in obs_names
        }

    def get_rule_names_and_attributes(
        self,
        matches: list[str] | None = None,
        starts_with: list[str] | None = None,
        includes: list[str] | None = None,
    ) -> dict[str, list[Any]]:
        """Returns the rule names mapped to all their attributes (e.g. the tags) for caching of cacheable load rules.

        Args:
            matches: Apply function to only the observations whose names that match one of these strings exactly. Does
                not support regex.
            starts_with: Apply to only the observations whose names start with one of these strings. Supports regex.
            includes: Apply to only the observations whose names contain one of the given strings. Supports regex.

        Returns:
            Mapping from CacheableLoadRule subclass names to the rule attributes of all the rules of the same type.

        """
        all_rules: list[RuleType] = []
        obs_names = self._filter_observation_names(matches=matches, starts_with=starts_with, includes=includes)
        for name in obs_names:
            all_rules.extend(self.load_rules[name])
        return _get_attrs_by_rule_name(all_rules, self.default_storage)
