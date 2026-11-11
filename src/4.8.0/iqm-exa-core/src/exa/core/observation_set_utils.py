# Copyright 2021-2026 IQM Finland Oy
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
"""Querying, creating and modifying calibration sets."""

from __future__ import annotations

from collections.abc import Iterable
import logging
from uuid import UUID

from exa.common.errors import UnknownObservationError, ValidationError
from exa.core.observation import LocalObservation, ObservationStash
from exa.core.provider import ExaRuntime, get_current_chip_label
from iqm.station_control.client.qon import QON, QONControllerSetting
from iqm.station_control.interface.models import (
    ObservationBase,
    ObservationData,
    ObservationLite,
    ObservationSetDefinition,
    ObservationSetType,
    ObservationSetUpdate,
)

logger = logging.getLogger(__name__)


def get_observation_set(
    set_id: UUID,
    observation_set_type: ObservationSetType | None = None,
) -> tuple[dict[str, ObservationData], str]:
    """Get the contents of an observation set.

    Args:
        set_id: ID of a calibration set.
        observation_set_type: Optional parameter denoting the type of the observation set. If set, an exception is
        raised in case the loaded observation set is not of the provided observation_set_type.

    Returns:
        Mapping from observation dut_field names to the observations themselves, and the type of the set.

    Raises:
        ValueError: If observation_set_type is set and does not conform to the set defined by set_id.

    """
    storage = ExaRuntime.storage()
    observation_set = storage.get_observation_set(set_id)
    if observation_set_type and observation_set.observation_set_type != observation_set_type:
        raise ValueError(f"Observation set {set_id} is not a {observation_set_type}.")
    observations = storage.get_observation_set_observations(set_id)
    return {obs.dut_field: obs for obs in observations}, observation_set.observation_set_type


def get_latest_observation_set_id(observation_set_type: ObservationSetType | None = None) -> UUID | None:
    """Get the latest finalized observation set id from Station Control.

    Args:
        observation_set_type: Optionally specifies which observation set type to fetch the latest of.

    Returns:
        The observation set id of the latest observation set belonging to the stations dut_label.
        If ``observation_set_type`` is not None, returns the latest set of that type.
        If no such sets are found, returns None.

    """
    chip_label = get_current_chip_label()
    observation_sets = ExaRuntime.storage().query_observation_sets(
        observation_set_type=observation_set_type,
        invalid=False,
        end_timestamp__isnull=False,
        dut_label=chip_label,
        order_by="-end_timestamp",
        limit=1,
    )
    observation_set = next(observation_sets, None)
    return observation_set.observation_set_id if observation_set is not None else None


def create_observation_set(
    observations: Iterable[ObservationLite],
    observation_set_type: ObservationSetType,
) -> UUID:
    """Create an observation set and return the ID.

    Only creates the set, the observations have to already exist in the DB.

    Args:
        observations: Observations to include in the newly created observation set.
            If ``observations`` is empty, an empty observation set will be created.
        observation_set_type: Specifies which observation set type to create.

    Returns:
        The ID of the created observation set.

    """
    observation_set_definition = ObservationSetDefinition(
        observation_set_type=observation_set_type,
        observation_ids=[obs.observation_id for obs in observations],
    )
    observation_set_data = ExaRuntime.storage().create_observation_set(observation_set_definition)
    return observation_set_data.observation_set_id


def create_quality_metric_set(observations: Iterable[ObservationLite], calibration_set_id: UUID) -> UUID:
    """Create a quality metric set and return the ID.

    Args:
        observations: Observations to include in the newly created quality metric set. Can be empty.
        calibration_set_id: UUID of the calibration set to be associated with the quality metric set.

    Returns:
        The ID of the created quality metric set.

    """
    quality_metric_set_definition = ObservationSetDefinition(
        observation_set_type=ObservationSetType.QUALITY_METRIC_SET,
        observation_ids=[obs.observation_id for obs in observations],
        describes_id=calibration_set_id,
    )
    quality_metric_set_data = ExaRuntime.storage().create_observation_set(quality_metric_set_definition)
    return quality_metric_set_data.observation_set_id


def add_to_observation_set(set_id: UUID, observations: Iterable[ObservationLite]) -> None:
    """Add observations to an existing observation set.

    Args:
        set_id: ID of an observation set.
        observations: List of observations to add.

    """
    ExaRuntime.storage().update_observation_set(
        ObservationSetUpdate(
            observation_set_id=set_id,
            observation_ids=[obs.observation_id for obs in observations],
            invalid=False,
        )
    )


def finalize_observation_set(set_id: UUID) -> None:
    """Finalize (make immutable) an existing observation set.

    Args:
        set_id: ID of the observation set.

    """
    ExaRuntime.storage().finalize_observation_set(set_id)


def copy_station_specific_observations(
    set_id: UUID,
    dut_label: str,
    *,
    set_type: ObservationSetType | None = None,
    dry_run: bool = False,
) -> UUID | None:
    """Make copies of existing station-specific observations in an existing observation set, linking them to a new QPU.

    Currently all :attr:`observations <ObservationDefinition.dut_label>` and most
    :attr:`observation sets <ObservationSetData.dut_label>` are linked to a specific QPU via the
    ``dut_label`` attribute.
    However, some observations are more correctly associated with other parts of the station,
    e.g. the instrumentation and wiring. Their values do not change (much) when the QPU is changed.
    These include e.g. instrument trigger delays, pulse precompensation settings, and TWPA settings.

    This function takes an existing observation set, and makes copies of all station-specific observations
    in it in the DB, linking them with a new ``dut_label``. It then groups them in a new observation set,
    which can e.g. be used as the starting point of a new calibration run for the station after the QPU
    has been changed.

    .. note::

       The station-specific observations are determined using a fixed whitelist which must be updated
       when new observations are introduced.

    Args:
        set_id: ID of an existing observation set.
        dut_label: DUT label of the new QPU.
        set_type: Observation set type for the new set. If None, use the type of the existing set.
        dry_run: Iff True, only log the observations that would be copied, do not make actual copies.

    Returns:
        ID of the new observation set, containing of copies of all the station-specific observations
        in the existing observation set but linked to ``dut_label``, or None on a dry run.

    """
    storage = ExaRuntime.storage()
    # get the old set
    old_set = storage.get_observation_set(set_id)
    if old_set.dut_label == dut_label:
        raise ValueError(f"The observations are already associated with {dut_label}.")

    # get the full observation data for the old set
    old_obs_data = storage.query_observations(observation_set_ids__contains=[set_id], limit=0)

    # find the station-specific observations and associate them with the new QPU
    _WHITELIST_CONTROLLER_PATH = {
        ("twpa",),  # all TWPA settings, e.g. controllers.PL-1.twpa.power
        # all precompensation settings
        ("drive", "awg", "precompensation"),
        ("flux", "awg", "precompensation"),  # e.g. controllers.TC-1-3.flux.awg.precompensation.fir_kernel
    }
    _WHITELIST_CONTROLLER_SETTING = {
        "trigger_delay",  # all trigger delays, e.g. controllers.QB4.drive.awg.trigger_delay
        # all input/output ranges
        "output_range",  # e.g. controllers.QB4.drive.awg.output_range
        "input_range",  # e.g controllers.PL-1.readout.input_range
        # all attenuations
        "attenuation_in",  # e.g controllers.PL-1.readout.attenuation_in
        "attenuation_out",  # e.g controllers.PL-1.readout.attenuation_out
        # athene output mixer bias
        "mixer_out_dc_bias_i",
        "mixer_out_dc_bias_q",
        # athene input mixer IQ imbalance settings
        "mixer_in_iq_phase_error",
        "mixer_in_iq_gain_error",
    }
    new_obs = []
    unparseable_obs = []
    for obs in old_obs_data:
        try:
            qon = QON.from_str(obs.dut_field)
        except (ValidationError, UnknownObservationError):
            unparseable_obs.append(obs)
            continue
        if isinstance(qon, QONControllerSetting) and (
            qon.controller_path in _WHITELIST_CONTROLLER_PATH or qon.setting in _WHITELIST_CONTROLLER_SETTING
        ):
            obs.dut_label = dut_label  # associate with the new QPU
            new_obs.append(obs)

    logger.info(
        "Copying %d station-specific observations from observation set %s (for '%s') to '%s'):\n%s",
        len(new_obs),
        set_id,
        old_set.dut_label,
        dut_label,
        "\n".join([f"  {obs.dut_field}:\t{obs.value}" for obs in new_obs]),
    )
    n_uniques = len({obs.dut_field for obs in new_obs})
    if n_uniques != len(new_obs):
        raise ValueError(
            f"Observation set {set_id}: out of the {len(new_obs)} station-specific observations, "
            f"only {n_uniques} are unique."
        )
    if unparseable_obs:
        logger.info(
            "%d unparseable observations found:\n%s",
            len(unparseable_obs),
            "\n".join([f"  {obs.dut_field}" for obs in unparseable_obs]),
        )
    if dry_run:
        return None

    # save copies of the station-specific observations in the DB
    new_obs_data = storage.create_observations(new_obs)
    # create the new set
    return create_observation_set(new_obs_data, set_type or old_set.observation_set_type)


def _migrate_observations_for_measure_operation(
    to_be_migrated: dict[str, ObservationBase],
    dut_label: str,
    from_impl: str,
    to_impl: str,
    convert_qnd_observations: bool,
    migrated_from: list[str],
) -> list[LocalObservation]:
    converted_observations = []
    components: set[str] = set()
    for dut_field, obs in to_be_migrated.items():
        if from_impl in dut_field:
            if f"gates.{from_impl}" in dut_field:
                components.update(dut_field.split(".")[-2])
            converted_observations.append(
                LocalObservation(
                    dut_field=obs.dut_field.replace(f".{from_impl}.", f".{to_impl}."),
                    value=obs.value,
                    unit=obs.unit,
                    uncertainty=obs.uncertainty,
                    invalid=obs.invalid,
                    dut_label=dut_label,
                    source={"type": "configuration_source", "configurator": f"migration_from_{migrated_from}"},
                )
            )
        if convert_qnd_observations and "gates.measure.constant_qnd" in dut_field:
            converted_observations.append(
                LocalObservation(
                    dut_field=obs.dut_field.replace(".measure.constant_qnd.", ".measure.constant."),
                    value=obs.value,
                    unit=obs.unit,
                    uncertainty=obs.uncertainty,
                    invalid=obs.invalid,
                    dut_label=dut_label,
                    source={"type": "configuration_source", "configurator": f"migration_from_{str(migrated_from)}"},
                )
            )
    # add shelved impl extra observations
    if to_impl == "measure_fidelity.shelved_constant":
        for component in components:
            converted_observations.append(
                LocalObservation(
                    dut_field=f"gates.{to_impl}.{component}.do_prx_12",
                    value=True,
                    unit="",
                    invalid=False,
                    dut_label=dut_label,
                    source={"type": "configuration_source", "configurator": f"migration_from_{migrated_from}"},
                )
            )
            converted_observations.append(
                LocalObservation(
                    dut_field=f"gates.{to_impl}.{component}.second_prx_12_offset",
                    value=0.0,
                    unit="s",
                    invalid=False,
                    dut_label=dut_label,
                    source={"type": "configuration_source", "configurator": f"migration_from_{migrated_from}"},
                )
            )
    return converted_observations


def migrate_observation_set_for_measure_fidelity_gate(
    set_id: UUID,
    from_impl: str = "measure.constant",
    to_impl: str = "measure_fidelity.constant",
    convert_qnd_observations: bool = False,
) -> UUID:
    """Copy the ``measure.constant`` observations into ``measure_fidelity.constant`` and create a new observation set.

    Args:
        set_id: ID of the observation set to be migrated.
        from_impl: From which gate implementation (in the format `"<gate name>.<impl name>"` to migrate the
            observations.
        to_impl: To which gate implementation (in the format `"<gate name>.<impl name>"` to migrate the
            observations.
        convert_qnd_observations: Convert the ``measure.constant_qnd`` observations into ``measure.constant``
            observations. Only set to ``True`` if you are certain the set defined by ``set_id`` has a valid calibration
            for ``measure.constant_qnd``.

    Returns:
        The ID of new observation set which now contains the ``measure_fidelity`` observations.

    """
    to_be_migrated, _ = get_observation_set(set_id)
    dut_label = ExaRuntime.storage().get_observation_set(set_id).dut_label or "NOT_FOUND"
    converted_observations = _migrate_observations_for_measure_operation(
        to_be_migrated,  # type: ignore[arg-type]
        dut_label,
        from_impl,
        to_impl,
        convert_qnd_observations,
        [str(set_id)],  # type: ignore[arg-type]
    )
    saved_observations = ExaRuntime.storage().create_observations(converted_observations)
    rest_of_calset = [
        o
        for df, o in to_be_migrated.items()
        if (not convert_qnd_observations or "gates.measure.constant_qnd" not in df)
    ]
    return create_observation_set(
        saved_observations + rest_of_calset, observation_set_type=ObservationSetType.CALIBRATION_SET
    )


def migrate_observation_stash_for_measure_operation(
    stash: ObservationStash,
    tag_with: list[str],
    from_impl: str = "measure.constant",
    to_impl: str = "measure_fidelity.constant",
    convert_qnd_observations: bool = False,
    save: bool = False,
) -> ObservationStash:
    """Copy one measure operation implementation observations into another one and create a new ObservationStash
    instance which contains the new observations as LocalObservations.

    Args:
        stash: stash that contains the observations to be migrated.
        tag_with: Tag the new observations with these tags.
        from_impl: From which gate implementation (in the format `"<gate name>.<impl name>"` to migrate the
            observations.
        to_impl: To which gate implementation (in the format `"<gate name>.<impl name>"` to migrate the
            observations.
        convert_qnd_observations: Convert the ``measure.constant_qnd`` observations into ``measure.constant``
            observations. Only set to ``True`` if you are certain the stash has a valid calibration
            for ``measure.constant_qnd``.
        save: Save the new observations to the DB.

    Returns:
            The ID of new observation set which now contains the ``measure_fidelity`` observations.

    """
    to_be_migrated = {df: observations[0] for df, observations in stash._db_observations.items()}
    dut_label = next(o.dut_label for o in to_be_migrated.values())
    converted_observations = _migrate_observations_for_measure_operation(
        to_be_migrated,  # type: ignore[arg-type]
        dut_label,
        from_impl,
        to_impl,
        convert_qnd_observations,
        tag_with,
    )
    rest_of_obs = {
        o.dut_field: [o]
        for df, o in to_be_migrated.items()
        if (not convert_qnd_observations or "gates.measure.constant_qnd" not in df)
    }
    new_stash = ObservationStash(
        local_observations={o.dut_field: [o] for o in converted_observations},
        db_observations=rest_of_obs,
    )
    new_stash.modify_tags(tags_to_add=tag_with)
    if save:
        new_stash.save()
    return new_stash


def adjust_trigger_delays_in_stash(
    stash: ObservationStash,
    tag_with: list[str],
    controller_types: list[str],
    shift: float,
    save: bool = False,
) -> ObservationStash:
    """Copy ``trigger_delay`` observations shifted by ``shift`` into a new :class:`ObservationStash`.

    Args:
        stash: Stash whose DB observations are the source values.
        tag_with: Tags applied to the new (shifted) local observations and to the stash.
        controller_types: Substrings identifying which trigger_delay controller type to shift,
            e.g. ``["drive", "flux", "readout"]``. An empty list shifts nothing.
        shift: Amount added to each matched observation's value, in the observation's own unit.
        save: If True, save the new observations to the DB.

    Returns:
        A new :class:`ObservationStash` containing the shifted trigger_delay observations as
        local observations and everything else preserved as DB observations.

    """
    shifted: list[LocalObservation] = []
    passthrough: dict[str, list[ObservationData]] = {}
    for dut_field, observations in stash._db_observations.items():
        obs = observations[0]
        matches = "trigger_delay" in dut_field and any(ct in dut_field for ct in controller_types)
        if matches and isinstance(obs.value, float):
            shifted.append(
                LocalObservation(
                    dut_field=obs.dut_field,
                    value=obs.value + shift,
                    unit=obs.unit,
                    uncertainty=obs.uncertainty,
                    invalid=obs.invalid,
                    dut_label=obs.dut_label or "NOT_FOUND",
                    tags=list(tag_with),
                    source={
                        "type": "configuration_source",
                        "configurator": f"trigger_delay_shift_from_{tag_with}",
                    },
                )
            )
        else:
            passthrough[dut_field] = [obs]

    new_stash = ObservationStash(
        local_observations={o.dut_field: [o] for o in shifted},
        db_observations=passthrough,
    )
    new_stash.modify_tags(tags_to_add=tag_with)
    if save:
        new_stash.save()
    return new_stash
