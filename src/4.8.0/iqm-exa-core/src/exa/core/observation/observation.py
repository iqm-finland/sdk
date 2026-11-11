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
"""Classes for local observations."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from decimal import InvalidOperation
import logging
from typing import Protocol, TypeAlias
import uuid

from pydantic import Field

from exa.common.data import ObservationUncertainty, ObservationValue, Setting, setting_value_to_observation
from exa.core import database_wrapper
from exa.core.provider import ExaRuntime
from exa.core.util.formatting import scientific_rounding
from iqm.station_control.interface.models import ObservationData, ObservationDefinition
from iqm.station_control.interface.storage_interface import StorageInterface

from .observation_validation_rules import DUTFieldValidationRule

logger = logging.getLogger(__name__)


class _ObservationStash(Protocol):
    """Dummy definition to avoid circular import."""

    def stash(self, observation: LocalObservation | list[LocalObservation]) -> None:
        """Add local observations to the stash."""


def _utcnow() -> datetime:
    """Construct a UTC datetime."""
    return datetime.now(UTC)


ValidationRuleType: TypeAlias = Callable[[ObservationDefinition], None]
"""Validator for (some) ObservationDefinition attributes, raises on errors."""


class LocalObservation(ObservationDefinition):
    """Data object representing a locally generated observation that can be later saved to database.

    The responsibility of the class is to hold the data of the observation
    in order to defer the saving to the database and use the observations
    in subsequent parts.
    """

    validation_rules: list[ValidationRuleType] = []
    """The list of validation rules for the observation."""
    default_validation_rules: list[ValidationRuleType] = [DUTFieldValidationRule()]
    """The list of default validation rules for the observation."""
    created_timestamp: datetime = Field(default_factory=_utcnow)
    """Time when this object was created locally. Not to be confused with the timestamp assigned when saving to
    database."""

    # FIXME signature clashes with Pydantic BaseModel.validate
    def validate(self) -> None:  # type: ignore[override]
        """Validate the observation using the validation rules of the observation."""
        for validation_rule in self.default_validation_rules + self.validation_rules:
            validation_rule(self)

    def tag(self, tags: list[str]) -> None:
        """Add provided tags to the observation."""
        self.tags = sorted(set(self.tags + tags))  # type: ignore[has-type]

    def save(self, storage: StorageInterface | None = None) -> ObservationData | None:
        """Save the observation to the DB.

        Args:
            storage: If given, overrides the default (ambient) storage backend.

        Returns:
            The saved observation or None in case the saving did not succeed.

        """
        storage = storage if storage is not None else ExaRuntime.storage()
        try:
            observations = storage.create_observations([self])
        except RuntimeError:
            logger.warning("Saving observation failed. Returning None instead.")
            return None
        observation = observations[0]
        logger.info("Observation saved: dut_field=%s, value=%s", observation.dut_field, observation.value)
        return observation

    def stash(self, observation_stash: _ObservationStash) -> LocalObservation:
        """Stash the observation.

        Args:
            observation_stash: The stash where the observation is added.

        Returns:
            Self.

        """
        observation_stash.stash(self)
        return self

    def pretty_value(self) -> str:
        """Scientifically formatted value with uncertainty."""
        try:
            return scientific_rounding(self.value, self.uncertainty)
        except (ValueError, OverflowError, TypeError, InvalidOperation):
            return str(self.value)

    def __str__(self) -> str:
        return f"{self.dut_field}: {self.pretty_value()} {self.unit}".rstrip()

    @classmethod
    def from_setting(
        cls,
        setting: Setting,
        *,
        dut_field: str = "",
        run_id: uuid.UUID | None = None,
        sequence_id: uuid.UUID | None = None,
        configurator: str | None = None,
    ) -> LocalObservation:
        """Creates a local observation from a setting.

        Observations have several attributes that Settings are lacking. They are handled as follows:

        * :attr: ``source`` is given using ``run_id``, ``sequence_id`` or ``configurator``.
        * :attr: ``uncertainty`` is None.
        * :attr: ``invalid`` is False.
        * :attr: ``tags`` is empty.

        Args:
            setting: Setting to be converted into an observation.
            dut_field: If given, will override ``setting.parameter.name``, which is arbitrary and
                does not have to match the path of ``setting`` in a settings tree.
                The ``dut_field`` of an observation should match the path of the corresponding
                setting in the EXA settings tree.
                If not given, ``setting.parameter.name`` must include a QPU component.
            run_id: If given, the run ID of the experiment that generated ``setting``.
            sequence_id: If given, the ID of the sequence that generated ``setting``.
            configurator: If given, the configuration source for ``setting``.

        """
        # TODO setting and observation type systems still do not match perfectly
        # TODO some settings like flux.voltages use DataType.ANYTHING!
        if not dut_field:
            dut_field = setting.path

        return cls.create(
            dut_field=dut_field,
            value=setting_value_to_observation(setting.value),
            # handles the case where setting.parameter.collection_type == CollectionType.LIST
            unit=setting.parameter.unit,
            tags=[],
            run_id=run_id,
            sequence_id=sequence_id,
            configurator=configurator,
        )

    @classmethod
    def create(  # noqa: PLR0913
        cls,
        dut_field: str,
        value: ObservationValue,
        unit: str,
        *,
        uncertainty: ObservationUncertainty | None = None,
        invalid: bool = False,
        tags: list[str] | None = None,
        run_id: uuid.UUID | None = None,
        sequence_id: uuid.UUID | None = None,
        configurator: str | None = None,
        validation_rules: list[ValidationRuleType] | None = None,
    ) -> LocalObservation:
        """Create a local observation from the given parameters."""
        return cls(
            dut_label=ExaRuntime.experiment_configuration().dut_label,
            dut_field=dut_field,
            value=value,
            unit=unit,
            uncertainty=uncertainty,
            invalid=invalid,
            source=LocalObservation._generate_source(run_id=run_id, sequence_id=sequence_id, configurator=configurator),
            tags=tags or [],
            validation_rules=validation_rules or [],
        )

    @staticmethod
    def _generate_source(
        *, run_id: uuid.UUID | None = None, sequence_id: uuid.UUID | None = None, configurator: str | None = None
    ) -> dict:
        """Generate source from the given attributes `run_id`, `sequence_id`, and `configurator`."""
        values = [run_id, sequence_id, configurator]
        number_of_input_nones = sum(value is not None for value in values)
        if number_of_input_nones > 1:
            raise ValueError(
                "LocalObservation should not have more than one of the attributes 'run_id', 'sequence_id', and"
                "'configurator' populated with a non-empty value."
            )
        if number_of_input_nones == 3:
            raise ValueError(
                "One of the attributes 'run_id', 'sequence_id', and"
                "'configurator' should be populated with a non-empty value."
            )
        if sequence_id is not None:
            return {
                "type": "sequence_analysis_source",
                "sequence_id": sequence_id,
            }
        if run_id is not None:
            return {
                "type": "analysis_source",
                "run_id": run_id,
            }
        else:
            return {
                "type": "configuration_source",
                "configurator": configurator,
            }


class LocalObservationSequence(Sequence):
    """A sequence of local observations that can be later saved to database."""

    def __init__(self, observations: Sequence[LocalObservation] | None = None):
        self._observations: list[LocalObservation] = list(observations or [])

    def __str__(self) -> str:
        lines = [str(observation) for observation in self._observations]
        return "\n".join(lines)

    def __getitem__(self, item):  # noqa: ANN001
        return self._observations[item]

    def __len__(self):
        return len(self._observations)

    def append(self, observation: LocalObservation) -> None:
        """Add an observation."""
        self._observations.append(observation)

    def extend(self, observations: Sequence[LocalObservation]) -> None:
        """Add multiple observations."""
        self._observations.extend(observations)

    def validate(self) -> LocalObservationSequence:
        """Validate all observations."""
        for observation in self:
            observation.validate()
        return self

    def tag(self, tags: list[str]) -> LocalObservationSequence:
        """Add provided tags to all observations."""
        for observation in self:
            observation.tag(tags)
        return self

    def save(self, storage: StorageInterface | None = None) -> list[ObservationData]:
        """Save all observations to the database.

        Args:
            storage: If given, overrides the default (ambient) storage backend.

        Returns:
            List of persisted observation objects.

        """
        return database_wrapper.batch_save_observations(list(self), storage=storage)

    def stash(self, observation_stash: _ObservationStash) -> LocalObservationSequence:
        """Stash all the observations.

        Args:
            observation_stash: The stash where the observations are added.

        Returns:
            Self.

        """
        observation_stash.stash(self._observations)
        return self


def get_components_for(dut_field: str) -> list[str]:
    """Get components from the DUT field."""
    # FIXME: Consider a better way to get component information from the DUT field
    #  Component is not always the first one anymore, thus we need to find a way to find it from the DUT field.
    #  Also, DUT field might have more than just one component.
    #  The current implementation should be considered a temporary solution, until we have some more reliable
    #  way to get the components for observation. Ideally, we would store it to the database and both
    #  ObservationDefinition and ObservationData would have the same information so that we can get the information
    #  the same way, no matter whether we are dealing with local or database observations.
    segments = dut_field.split(".")
    for segment in segments:
        if "QB" in segment:
            return [segment]
    return [segments[0]]
