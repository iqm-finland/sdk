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

from pydantic import ConfigDict

from exa.common.data.value import Uncertainty, Value
from iqm.station_control.interface.pydantic_base import PydanticBase


class ObservationBase(PydanticBase):
    """Abstract base class of the observation models."""

    dut_field: str
    """Name of the property the observation is about."""
    value: Value
    """Value of the observation."""
    unit: str
    """SI unit of the value. Empty string means the value is dimensionless."""
    uncertainty: Uncertainty | None = None
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
    """

    observation_id: int
    """Unique identifier of the observation."""
    created_timestamp: datetime
    """Time when the object was created in the database."""
    modified_timestamp: datetime
    """Time when the object was last modified in the database."""


class ObservationData(ObservationLite, ObservationDefinition):
    """The content of the observation stored in the database."""

    model_config = ConfigDict(
        extra="ignore",  # Ignore any extra attributes
    )

    observation_set_ids: list[uuid.UUID] = []
    """List of observation set UUIDs this observation belongs to."""


class ObservationUpdate(PydanticBase):
    """The observation data to be updated in the database."""

    model_config = ConfigDict(
        extra="forbid",  # Forbid any extra attributes
    )

    observation_id: int
    """Unique identifier of the observation."""
    invalid: bool
    """Flag indicating if the observation is invalid. Automated systems must not use invalid observations."""
