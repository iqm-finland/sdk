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

from datetime import datetime, timezone
import uuid

from pydantic import ValidationError
import pytest

from iqm.station_control.interface.models import (
    SequenceMetadataData,
    SequenceMetadataDefinition,
    SequenceResultData,
    SequenceResultDefinition,
)


def test_sequence_metadata_definition_forbids_extra_attributes():
    with pytest.raises(ValidationError):
        _ = SequenceMetadataDefinition(sequence_id=uuid.uuid4(), origin_id="foo", origin_uri="bar", extra="foobar")


def test_sequence_metadata_ignores_extra_attributes():
    metadata = SequenceMetadataData(
        sequence_id=uuid.uuid4(),
        origin_id="foo",
        origin_uri="bar",
        created_timestamp=datetime.now(timezone.utc),
        extra="foobar",
    )
    assert not hasattr(metadata, "extra")


def test_sequence_result_definition_forbids_extra_attributes():
    with pytest.raises(ValidationError):
        _ = SequenceResultDefinition(sequence_id=uuid.uuid4(), data={}, final=False, extra="foobar")


def test_sequence_result_data_ignores_extra_attributes():
    result = SequenceResultData(
        sequence_id=uuid.uuid4(),
        data={},
        final=False,
        created_timestamp=datetime.now(timezone.utc),
        modified_timestamp=datetime.now(timezone.utc),
        extra="foobar",
    )
    assert not hasattr(result, "extra")
