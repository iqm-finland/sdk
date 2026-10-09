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
"""Sequence related station control interface models."""

from datetime import datetime
from typing import Any
import uuid

from pydantic import ConfigDict

from iqm.station_control.interface.pydantic_base import PydanticBase


class SequenceMetadataBase(PydanticBase):
    """Abstract base class of the sequence metadata definition and data."""

    sequence_id: uuid.UUID
    """Unique identifier of the sequence."""

    origin_id: str
    """Unique identifier of the creator. E.g. notebook researcher username, or calibration service ID."""

    origin_uri: str
    """Uniform resource identifier (weak reference) for the creator. E.g. calibration service ID."""


class SequenceMetadataDefinition(SequenceMetadataBase):
    """The content of the sequence metadata object when creating it."""

    model_config = ConfigDict(
        extra="forbid",  # Forbid any extra attributes
    )


class SequenceMetadataData(SequenceMetadataBase):
    """The content of the sequence metadata stored in the database."""

    created_timestamp: datetime
    """Time when the object was created in the database."""


class SequenceResultBase(PydanticBase):
    """Abstract base class of the sequence result definition and data."""

    sequence_id: uuid.UUID
    """Unique identifier of the sequence result."""

    data: dict[str, Any]
    """JSON serializable dict."""
    # Use simple "dict" here instead of creating a Pydantic model for data.
    # Pydantic model is kept in exa-experiment to avoid having complex sequence details in the interface.

    final: bool
    """Indicates whether this result was marked as final."""


class SequenceResultDefinition(SequenceResultBase):
    """The content of the sequence result object when creating it."""

    model_config = ConfigDict(
        extra="forbid",  # Forbid any extra attributes
    )


class SequenceResultData(SequenceResultBase):
    """The content of the sequence result stored in the database."""

    created_timestamp: datetime
    """Time when the object was created in the database."""

    modified_timestamp: datetime | None
    """Time when the object was last modified in the database."""
