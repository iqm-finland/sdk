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
"""Observation set related station control interface models."""

from datetime import datetime
import uuid

from pydantic import ConfigDict, Field

from iqm.station_control.interface.models.observation import ObservationLite
from iqm.station_control.interface.models.type_aliases import ObservationSetType
from iqm.station_control.interface.pydantic_base import PydanticBase


class ObservationSetBase(PydanticBase):
    """Abstract base class of the observation set definition and data."""

    observation_set_type: ObservationSetType
    """Indicates the type (i.e. purpose) of the observation set."""
    observation_ids: list[int]
    """Database IDs of the observations belonging to the observation set."""
    describes_id: uuid.UUID | None = Field(default=None)
    """Unique identifier of the observation set this observation set describes."""
    invalid: bool = Field(default=False)
    """Flag indicating if the set is invalid. Automated systems must not use invalid sets."""


class ObservationSetDefinition(ObservationSetBase):
    """The content of the observation set object when creating it."""

    model_config = ConfigDict(
        extra="forbid",  # Forbid any extra attributes
    )


class ObservationSetData(ObservationSetBase):
    """The content of the observation set stored in the database."""

    dut_label: str | None
    """String representation of the DUT the observation set is associated with. Can only be None for generic sets."""
    observation_set_id: uuid.UUID
    """Unique identifier of the observation set."""
    created_timestamp: datetime
    """Time when the object was created in the database."""
    end_timestamp: datetime | None
    """Time when the observation set was finalized. If ``None``, the set is not finalized yet."""


class ObservationSetUpdate(PydanticBase):
    """The observation set data to be updated in the database."""

    model_config = ConfigDict(
        extra="forbid",  # Forbid any extra attributes
    )

    observation_set_id: uuid.UUID
    """Unique identifier of the observation set."""

    observation_ids: list[int] | None = Field(default=None)
    """Database IDs of the observations belonging to the observation set.

    This will only add new observations to the observation set, deleting existing ones is not possible.
    Setting this to ``None`` or omitting it will leave existing :attr:`observation_ids` as is with no changes.
    """
    invalid: bool
    """Flag indicating if the set is invalid. Automated systems must not use invalid sets."""


class ObservationSetWithObservations(ObservationSetData):
    """The content of the observation set stored in the database, with a list of observations."""

    observations: list[ObservationLite]
    """Observations belonging to the observation set."""


class QualityMetrics(ObservationSetWithObservations):
    """The content of the quality metric set stored in the database, with a list of observations and calibration set."""

    calibration_set: ObservationSetData
