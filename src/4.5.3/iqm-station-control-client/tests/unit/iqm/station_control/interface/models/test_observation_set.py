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
    ObservationLite,
    ObservationSetData,
    ObservationSetDefinition,
    ObservationSetType,
    ObservationSetUpdate,
    ObservationSetWithObservations,
)


def test_observation_set_definition_forbids_extra_attributes():
    with pytest.raises(ValidationError):
        _ = ObservationSetDefinition(
            observation_set_type=ObservationSetType.GENERIC_SET,
            observation_ids=[1, 2, 3],
            describes_id=None,
            invalid=False,
            extra="foobar",
        )


def test_observation_set_data_ignores_extra_attributes():
    observation_set = ObservationSetData(
        observation_set_type=ObservationSetType.GENERIC_SET,
        observation_ids=[1, 2, 3],
        describes_id=None,
        invalid=False,
        dut_label="M138_W36_A22_N05",
        observation_set_id=uuid.uuid4(),
        created_timestamp=datetime.now(timezone.utc),
        end_timestamp=datetime.now(timezone.utc),
        extra="foobar",
    )
    assert not hasattr(observation_set, "extra")


def test_observation_set_update_forbids_extra_attributes():
    with pytest.raises(ValidationError):
        _ = ObservationSetUpdate(
            observation_set_id=uuid.uuid4(),
            observation_ids=[1, 2, 3],
            invalid=False,
            extra="foobar",
        )


def test_observation_set_with_observations_deprecated_observation_ids_still_serialized_to_output():
    observation_set_with_observations = ObservationSetWithObservations(
        observation_set_type=ObservationSetType.GENERIC_SET,
        describes_id=None,
        invalid=False,
        dut_label="M138_W36_A22_N05",
        observation_set_id=uuid.uuid4(),
        created_timestamp=datetime.now(timezone.utc),
        end_timestamp=datetime.now(timezone.utc),
        observations=[
            ObservationLite(
                observation_id=1,
                dut_field="QB1.t1_time",
                unit="s",
                value=4.408139707188389e-05,
                uncertainty=None,
                invalid=False,
                created_timestamp=datetime.fromisoformat("2023-02-10T08:57:04.605956+00:00"),
                modified_timestamp=datetime.fromisoformat("2023-02-10T08:57:04.605956+00:00"),
            ),
            ObservationLite(
                observation_id=2,
                dut_field="QB1.t2_time",
                unit="s",
                value=3.245501974471748e-05,
                uncertainty=None,
                invalid=False,
                created_timestamp=datetime.fromisoformat("2023-02-10T08:57:04.605956+00:00"),
                modified_timestamp=datetime.fromisoformat("2023-02-10T08:57:04.605956+00:00"),
            ),
        ],
    )
    assert [observation.observation_id for observation in observation_set_with_observations.observations] == [1, 2]

    # "observation_ids" should be in the output for older clients to work
    assert observation_set_with_observations.model_dump()["observation_ids"] == [1, 2]
    assert "[1,2]" in observation_set_with_observations.model_dump_json()
