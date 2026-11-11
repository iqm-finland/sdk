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
"""Utility functions for helping with observation handling and experiments."""

from collections.abc import Iterable
from typing import Any

from exa.core.database_wrapper import get_all_dut_fields
from exa.core.observation import GetLatest, ObservationHandler, get_global_default_observation_loading_rules
from exa.core.provider import ExaRuntime


# TODO: this should probably move into observation
def load_observations(
    observation_names: Iterable[str] | None = None, return_full_observation: bool = False
) -> dict[str, Any]:
    """Loads all latest observations that match the loading rules from a connected database or observation stash.

    Args:
         observation_names:
             If None, all matching observations are loaded, otherwise only observations that
             match the names.
         return_full_observation: If set, return full ObservationData incl. metadata. Else, only observation.value.

    Returns:
         obs_dict: Dictionary with requested observation.

    """
    handler = ObservationHandler(
        observation_names or get_all_dut_fields(),
        load_rules=get_global_default_observation_loading_rules() or [GetLatest()],
        dut_label=ExaRuntime.experiment_configuration().dut_label,
        default_storage=ExaRuntime.storage(),
    )

    handler.load_observations()
    ret_dict = handler.value_dict(return_full_observation=return_full_observation)

    return ret_dict
