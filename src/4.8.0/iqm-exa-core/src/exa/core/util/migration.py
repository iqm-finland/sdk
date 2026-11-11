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
"""Temporary tools for migrating code and data between versions."""

import uuid

from exa.common.data import SettingNode
from exa.core.database_wrapper import (
    batch_save_observations,
    get_latest_observations_for_tags,
    get_observation_set_observations,
    get_observation_set_type,
    save_observations_with_new_tags,
)
from exa.core.experiment import Experiment
from exa.core.observation_set_utils import create_observation_set, finalize_observation_set
from exa.core.provider import ExaRuntime
from iqm.cpc.core.observation.observation_parameters import (
    FLUX_PULSE_GATE_CHARACTERIZATION_PARAMETERS,
    SINGLE_QUBIT_MODEL_CHARACTERIZATION_PARAMETERS,
)
from iqm.station_control.interface.models import ObservationData

OLD_READOUT_PATHS = (
    "QUBIT.readout.channel_pulse_amplitude",
    "QUBIT.readout.phase",
    "QUBIT.readout.frequency",
    "QUBIT.readout.pulse_length",
    "QUBIT.readout.integration_weights_I",
    "QUBIT.readout.integration_weights_Q",
    "QUBIT.readout.integration_threshold",
    "PL.readout.integration_length",
    "PL.readout.delay_before_integration",
)


def drop_legacy_readout_observations(old_tags: list[str], new_tags: list[str]) -> list[ObservationData]:
    """Copy observations from old tags to new tags, and drop observations related to legacy readout in the process.

    See :const:`.OLD_READOUT_PATHS` for list of dropped observations.
    Advisable to use when transitioning to exa-core 7.0.

    Args:
        old_tags: Tags to start from.
        new_tags: New tags.

    Returns:
        Newly saved observations.

    """
    chip_toplogy = Experiment().chip_topology  # type: ignore[operator]

    old_obs = [
        path.replace("QUBIT", qubit)
        for qubit in chip_toplogy.qubits
        for path in OLD_READOUT_PATHS
        if path.startswith("QUBIT")
    ]
    old_obs.extend(
        [
            path.replace("PL", probeline)
            for probeline in chip_toplogy.probe_lines
            for path in OLD_READOUT_PATHS
            if path.startswith("PL")
        ]
    )

    return save_observations_with_new_tags(
        tags=old_tags, add_tags=new_tags, remove_tags=old_tags, remove_observations=old_obs
    )


def _migrate_to_settings_refactoring(settings: SettingNode, observation: ObservationData) -> ObservationData:  # noqa: PLR0911,PLR0912
    _SINGLE_QUBIT_CHARACTERIZATION_NAMES = [p.name for p in SINGLE_QUBIT_MODEL_CHARACTERIZATION_PARAMETERS]
    _FLUX_PULSE_CHARACTERIZATION_NAMES = [p.name for p in FLUX_PULSE_GATE_CHARACTERIZATION_PARAMETERS]
    _QUBIT_CONTROLLER_DUT_FIELD_WHITELIST = ["drive.frequency", "drive.awg", "flux.voltage", "flux.awg"]
    _CPHASE_CHARACTERIZATION_NAMES = ["cphase_derivative", "cphase_pi_distance"]
    _AVG_RESPONSE_NAMES = ["average_response_g", "average_response_e", "average_response_phase"]

    dut_field = observation.dut_field
    fragments = dut_field.split(".")
    if len(fragments) == 2 and fragments[1] in _SINGLE_QUBIT_CHARACTERIZATION_NAMES:
        return observation.model_copy(update={"dut_field": f"characterization.model.{dut_field}"})
    if "QB" in fragments[0] or "TC" in fragments[0] and "controllers" not in dut_field:
        for whitelisted in _QUBIT_CONTROLLER_DUT_FIELD_WHITELIST:
            if whitelisted in dut_field:
                return observation.model_copy(update={"dut_field": f"controllers.{dut_field}"})
    if len(fragments) == 2 and "COMP" in fragments[0] and fragments[1] == "frequency":
        return observation.model_copy(update={"dut_field": f"characterization.model.{dut_field}"})
    if "PL" in fragments[0] and "readout" in dut_field and "controllers" not in dut_field:
        return observation.model_copy(update={"dut_field": f"controllers.{dut_field}"})
    if fragments[-1] in _AVG_RESPONSE_NAMES:
        return observation.model_copy(
            update={"dut_field": f"characterization.gate_properties.measure.constant.{fragments[0]}.{fragments[-1]}"}
        )
    if fragments[0] == "gates" and len(fragments) == 3 and fragments[2] == "default_implementation":
        return observation.model_copy(update={"dut_field": f"gate_definitions.{fragments[1]}.default_implementation"})
    if fragments[0] == "gates" and len(fragments) == 4 and fragments[3] == "override_default_for_loci":
        return observation.model_copy(
            update={"dut_field": f"gate_definitions.{fragments[1]}.{fragments[2]}.override_default_for_loci"}
        )
    if dut_field == "all_qubits.options.end_delay":
        return observation.model_copy(update={"dut_field": "controllers.options.end_delay"})
    if len(fragments) > 4 and fragments[1] == "flux":
        prefix = "characterization.gate_properties"
        if "TC" in fragments[0]:
            if len(fragments[0].split("-")) == 3:
                locus_strs = [  # we don't know the correct locus order, so we must try with both orders...
                    "__".join(["QB" + fragments[0].split("-")[1], "QB" + fragments[0].split("-")[2]]),
                    "__".join(["QB" + fragments[0].split("-")[2], "QB" + fragments[0].split("-")[1]]),
                ]
            else:  # star topology, save for both COMP_R and COMPR1
                qubit_num = fragments[0].replace("TC", "")
                locus_strs = [  # we don't know the resonator name, so we must try with both names...
                    "__".join(["QB" + qubit_num, "COMP_R"]),
                    "__".join(["QB" + qubit_num, "COMPR1"]),
                ]
            if fragments[2] in _CPHASE_CHARACTERIZATION_NAMES or fragments[2] in _FLUX_PULSE_CHARACTERIZATION_NAMES:
                prefix = f"{prefix}.{fragments[-2]}.{fragments[-1]}"
                coupler_str = ".coupler." if fragments[2] in _FLUX_PULSE_CHARACTERIZATION_NAMES else "."
                # find which locus string actually is found in the settings
                for locus in locus_strs:
                    if (
                        fragments[-2] in settings.gates.children
                        and fragments[-1] in settings.gates[fragments[-2]].children
                        and locus in settings.gates[fragments[-2]][fragments[-1]].children
                    ):
                        return observation.model_copy(
                            update={"dut_field": f"{prefix}.{locus}{coupler_str}{fragments[2]}"}
                        )
        if "QB" in fragments[0] and "TC" in fragments[-1]:
            if len(fragments[-1].split("-")) == 3:
                locus_strs = [  # we don't know the correct locus order, so we must try with both orders...
                    "__".join(["QB" + fragments[-1].split("-")[1], "QB" + fragments[-1].split("-")[2]]),
                    "__".join(["QB" + fragments[-1].split("-")[2], "QB" + fragments[-1].split("-")[1]]),
                ]
            else:  # star topology, save for both COMP_R and COMPR1
                qubit_num = fragments[-1].replace("TC", "")
                locus_strs = [  # we don't know the resonator name, so we must try with both names...
                    "__".join(["QB" + qubit_num, "COMP_R"]),
                    "__".join(["QB" + qubit_num, "COMPR1"]),
                ]
            if fragments[2] in _CPHASE_CHARACTERIZATION_NAMES or fragments[2] in _FLUX_PULSE_CHARACTERIZATION_NAMES:
                prefix = f"{prefix}.{fragments[-3]}.{fragments[-2]}"
                qubit_str = ".qubit." if fragments[2] in _FLUX_PULSE_CHARACTERIZATION_NAMES else "."
                for locus in locus_strs:
                    if (
                        fragments[-3] in settings.gates.children
                        and fragments[-2] in settings.gates[fragments[-3]].children
                        and locus in settings.gates[fragments[-3]][fragments[-2]].children
                    ):
                        return observation.model_copy(
                            update={"dut_field": f"{prefix}.{locus}{qubit_str}{fragments[2]}"}
                        )
        if fragments[2] in _QUBIT_CONTROLLER_DUT_FIELD_WHITELIST:
            return observation.model_copy(update={"dut_field": "characterization.gate_properties"})
    return observation


def migrate_observation_set_to_settings_refactoring(
    observation_set_id: uuid.UUID | str, finalize: bool = True
) -> uuid.UUID:
    """Loads an existing observation set, and modifies it to the new settings tree structure.

    NOTE: you must initialize the station-control and experiment config for the same station the observations belong
    to before using this function.

    Args:
        observation_set_id: The id of the set to migrate to the new structure
        finalize: Determines whether to automatically finalize the set. Default = True.

    """
    settings = Experiment().get_settings()  # type: ignore[operator]
    if isinstance(observation_set_id, str):
        observation_set_id = uuid.UUID(observation_set_id)
    observation_set_type = get_observation_set_type(observation_set_id, storage=ExaRuntime.storage())
    set_observations = get_observation_set_observations(
        (observation_set_id,),
        storage=ExaRuntime.storage(),
        dut_label=ExaRuntime.experiment_configuration().dut_label,
    )

    fixed_observations = []
    for dut_field, obs in set_observations.items():
        migrated = _migrate_to_settings_refactoring(settings, obs)
        fixed_observations.append(migrated)
        if migrated.dut_field != dut_field:
            print(f"migrated: {dut_field} => {migrated.dut_field}")

    new_observations = batch_save_observations(fixed_observations)  # type: ignore[arg-type]
    new_set_id = create_observation_set(new_observations, observation_set_type)

    if finalize:
        finalize_observation_set(new_set_id)
    return new_set_id


def migrate_observation_tag_to_settings_refactoring(
    observation_tag: str, new_tags: list[str] | str, clear_old_tags: bool = True
) -> list[ObservationData]:
    """Loads an existing observation tag, and duplicates it according to the new settings tree structure.

    NOTE: you must initialize the station-control and experiment config for the same station the observations belong
    to before using this function.

    Args:
        observation_tag: The tag to load.
        new_tags: The new tags for which to save the migrated observations.
        clear_old_tags: If true, clear the old tags before saving the observation.

    """
    if isinstance(new_tags, str):
        new_tags = [new_tags]
    tag_observations = get_latest_observations_for_tags([observation_tag])
    settings = Experiment().get_settings()  # type: ignore[operator]
    fixed_observations = []
    for dut_field, obs in tag_observations.items():
        migrated = _migrate_to_settings_refactoring(settings, obs)
        fixed_observations.append(migrated)
        if clear_old_tags:
            migrated.tags = []
        migrated.tags.extend(new_tags)
        if migrated.dut_field != dut_field:
            print(f"migrated: {dut_field} => {migrated.dut_field}")

    new_observations = batch_save_observations(fixed_observations)  # type: ignore[arg-type]
    return new_observations
