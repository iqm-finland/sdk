# Copyright 2025 IQM
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
"""Observation related Station Control interface models."""

from datetime import datetime
from typing import Any
import uuid
import warnings

from pydantic import ConfigDict, model_validator

from exa.common.data.value import ObservationUncertainty, ObservationValue
from exa.common.helpers.deprecation import format_deprecated
from iqm.station_control.interface.pydantic_base import PydanticBase

_OBSERVATION_LITE_DEPRECATION_MSG = format_deprecated(
    old="Direct instantiation of `ObservationLite`",
    new="`ObservationData`",
    since="2026-08-31",
)


class ObservationBase(PydanticBase):
    """Abstract base class of the observation models."""

    dut_field: str
    """Name of the property the observation is about."""
    value: ObservationValue
    """Value of the observation."""
    unit: str
    """SI unit of the value. Empty string means the value is dimensionless."""
    uncertainty: ObservationUncertainty | None = None
    """Uncertainty of the observation value. ``None`` means unknown."""
    invalid: bool = False
    """Flag indicating if the observation is invalid. Automated systems must not use invalid observations."""


class ObservationDefinition(ObservationBase):
    """The content of the observation definition."""

    dut_label: str
    """DUT label of the device the observation is about."""
    source: dict[str, Any]
    """How the observation was made, e.g. experiment analysis or manual specification.
    ``source`` always has the key ``"type"`` whose ``str`` value determines the other contents of the dict.
    The currently supported source types are:
    - analysis_source
    - configuration_source
    - measurement_source
    - sequence_analysis_source
    - specification_source
    """
    tags: list[str] = []
    """Human-readable tags of the observation."""


class ObservationLite(ObservationBase):
    """The lightweight version of the observation data.

    This model can be used when not all observation data is needed, to speed up retrieval.

    Deprecated:
        Direct instantiation is deprecated since 2026-08-31, use :class:`ObservationData` instead.
        Subclasses of ``ObservationLite``, such as :class:`ObservationData`, are unaffected.
    """

    observation_id: int
    """Unique identifier of the observation."""
    created_timestamp: datetime
    """Time when the object was created in the database."""
    modified_timestamp: datetime
    """Time when the object was last modified in the database."""

    @model_validator(mode="before")
    @classmethod
    def _warn_direct_instantiation(cls, data: Any) -> Any:
        """Warn when `ObservationLite` itself is instantiated directly, but not for its subclasses."""
        if cls is ObservationLite:
            warnings.warn(_OBSERVATION_LITE_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
        return data


class ObservationData(ObservationLite, ObservationDefinition):
    """The content of the observation stored in the database."""

    model_config = ConfigDict(
        extra="ignore",  # Ignore any extra attributes
    )

    observation_set_ids: list[uuid.UUID] = []
    """List of observation set UUIDs this observation belongs to."""

    @classmethod
    def from_observation_lite(cls, observation: ObservationLite, dut_label: str | None = None) -> "ObservationData":
        """Create an observation data from an observation lite, using placeholder values for the missing fields.

        Args:
            observation: Lightweight observation to convert.
            dut_label: DUT label of the device the observation is about. A placeholder is used if not given.

        Returns:
            Observation data corresponding to ``observation``, with placeholder ``dut_label``,
            ``source`` and ``tags`` if not otherwise derivable from ``observation``.

        """
        return cls(
            dut_field=observation.dut_field,
            value=observation.value,
            unit=observation.unit,
            uncertainty=observation.uncertainty,
            invalid=observation.invalid,
            observation_id=observation.observation_id,
            created_timestamp=observation.created_timestamp,
            modified_timestamp=observation.modified_timestamp,
            dut_label=dut_label if dut_label is not None else "unknown",
            source={"type": "source_not_loaded"},
            tags=["tags_not_loaded"],
        )


class ObservationUpdate(PydanticBase):
    """The observation data to be updated in the database."""

    model_config = ConfigDict(
        extra="forbid",  # Forbid any extra attributes
    )

    observation_id: int
    """Unique identifier of the observation."""
    invalid: bool
    """Flag indicating if the observation is invalid. Automated systems must not use invalid observations."""
