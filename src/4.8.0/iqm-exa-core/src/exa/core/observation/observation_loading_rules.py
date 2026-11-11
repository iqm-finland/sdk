# Copyright 2019-2026 IQM Finland Oy
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
"""Rules for loading observations (e.g. from the DB).

The loading rules themselves live in :mod:`iqm.cpc.core.observation.observation_loading_rules`, and the
functions they use to query the storage backend in
:mod:`iqm.cpc.core.observation.observation_load_utils`. They are imported here to retain backwards
compatibility for imports.
"""

from iqm.cpc.core.observation.observation_load_utils import (  # noqa: F401
    _EXP_CONFIG_NODES_AND_GLOBAL_KEYS,
    _GLOBAL_KEY_STR_REPR,
    _load_calibration_set_observations_cache,
    _load_calibration_set_quality_metric_set_observations_cache,
    _load_default_calibration_set_observations_cache,
    _load_latest_characterization_set_observations_cache,
    _load_latest_for_tags_observations_cache,
    _load_latest_observations_cache,
    _load_observation_set_observations_cache,
    populate_from_configs,
)
from iqm.cpc.core.observation.observation_loading_rules import (  # noqa: F401
    _DEFAULT_OBSERVATION_LOADING_RULES,
    DEFAULTS_FROM_YML_SOURCE,
    CacheableLoadRule,
    Fail,
    FromExperimentConfiguration,
    GetCalibrationSet,
    GetDefaultCalibrationSet,
    GetLatest,
    GetLatestCharacterizationSet,
    GetLatestForTags,
    GetObservationSet,
    GetQualityMetricSet,
    LatestFromStash,
    RuleCache,
    RuleType,
    resolve_rules,
)


def set_global_default_observation_loading_rules(rules: list[RuleType]) -> None:
    """Set global default rules which are to be applied to observation handlers in :class:`.Experiment`.

    Args:
        rules: The rules to be applied globally.

    """
    if not isinstance(rules, list):
        raise TypeError("Argument 'rules' must be a list")
    if not all(isinstance(rule, RuleType) for rule in rules):
        raise TypeError("Every rule in 'rules' must be an instance of RuleType")

    global _DEFAULT_OBSERVATION_LOADING_RULES  # noqa: PLW0603
    _DEFAULT_OBSERVATION_LOADING_RULES = rules


def get_global_default_observation_loading_rules() -> list[RuleType]:
    """Return the global default loading rules."""
    return _DEFAULT_OBSERVATION_LOADING_RULES
