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
import json
import math

import numpy as np
from pydantic import ValidationError
import pytest

from iqm.station_control.interface.models import (
    ObservationData,
    ObservationDefinition,
    ObservationLite,
    ObservationUpdate,
)

# These tests technically only verify that Pydantic does what it promises to do.
# They are still included to make sure that we don't accidentally alter the extra behaviour of our interface models.


@pytest.mark.parametrize(
    "value",
    [
        42,
        42.7,
        4 + 2.0j,
        True,
        "foobar",
        np.arange(10),
        np.array([[1.0, 2.4]]),
        np.exp([[1j, 2j], [0, 0]]),
        np.array([], dtype=np.int32),
        np.eye(4).T,
        np.arange(16).reshape(4, 4)[1:, 1:],
        np.nan,
        np.inf,
        np.array([np.nan, 2]),
        np.array([np.inf, 2]),
    ],
)
def test_observation_definition_roundtrips_value(value):
    observation_definition = ObservationDefinition(
        dut_field="foo",
        unit="bar",
        value=value,
        dut_label="foo",
        source={},
    )

    observation_definition_json = observation_definition.model_dump_json()
    observation_definition_roundtripped = ObservationDefinition.model_validate_json(observation_definition_json)
    value_roundtripped = observation_definition_roundtripped.value

    match value:
        case np.ndarray():
            assert isinstance(value_roundtripped, np.ndarray)
            assert np.array_equal(value_roundtripped, value, equal_nan=True)
            assert value_roundtripped.dtype == value.dtype
        case float(value) if math.isnan(value):
            assert "NaN" in observation_definition_json
            json.loads(observation_definition_json)
            assert math.isnan(value_roundtripped)
        case _:
            assert type(value_roundtripped) is type(value)
            assert value_roundtripped == value


def test_observation_lite_ignores_extra_attributes():
    observation = ObservationLite(
        dut_field="foo",
        value=42,
        unit="bar",
        uncertainty=None,
        invalid=False,
        observation_id=1337,
        created_timestamp=datetime.now(timezone.utc),
        modified_timestamp=datetime.now(timezone.utc),
        extra="foobar",
    )
    assert not hasattr(observation, "extra")


def test_observation_data_ignores_extra_attributes():
    observation = ObservationData(
        dut_field="foo",
        value=42,
        unit="bar",
        uncertainty=None,
        invalid=False,
        observation_id=1337,
        dut_label="foo",
        source={},
        tags=[],
        created_timestamp=datetime.now(timezone.utc),
        modified_timestamp=datetime.now(timezone.utc),
        observation_set_ids=[],
        extra="foobar",
    )
    assert not hasattr(observation, "extra")


def test_observation_data_requires_created_timestamp():
    # Only ObservationDefinition should assign "created_timestamp" automatically.
    # ObservationData should always read it from the database, thus no automatic creation should be allowed.
    with pytest.raises(ValidationError, match="Field required"):
        ObservationData(
            dut_field="foo",
            value=42,
            unit="bar",
            uncertainty=None,
            invalid=False,
            observation_id=1337,
            dut_label="foo",
            source={},
            tags=[],
            # created_timestamp=datetime.now(timezone.utc),
            modified_timestamp=datetime.now(timezone.utc),
            observation_set_ids=[],
            extra="foobar",
        )


def test_observation_update_forbids_extra_attributes():
    with pytest.raises(ValidationError):
        _ = ObservationUpdate(
            observation_id=42,
            invalid=False,
            extra="foobar",
        )
