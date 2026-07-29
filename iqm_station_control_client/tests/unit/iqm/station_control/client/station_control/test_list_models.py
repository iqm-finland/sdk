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

import json
import math

import numpy as np
import pytest

from iqm.station_control.client.list_models import ObservationDefinitionList
from iqm.station_control.interface.models import ObservationDefinition


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
    observation = ObservationDefinition(
        dut_label="M138_W36_A22_N05",
        dut_field="TC-1-2.flux.cphase_derivative.cz.tgss_crf",
        value=value,
        unit="",
        source={
            "type": "analysis_source",
            "run_id": "9f66988c-7abd-4b33-8534-7b801fa350b2",
        },
    )

    observation_definitions_json = ObservationDefinitionList([observation]).model_dump_json()
    observation_definitions_roundtripped = ObservationDefinitionList.model_validate(
        json.loads(observation_definitions_json)
    )

    value_roundtripped = observation_definitions_roundtripped[0].value

    match value:
        case np.ndarray():
            assert isinstance(value_roundtripped, np.ndarray)
            assert np.array_equal(value_roundtripped, value, equal_nan=True)
            assert value_roundtripped.dtype == value.dtype
        case float(value) if math.isnan(value):
            assert "NaN" in str(observation_definitions_json)
            json.loads(observation_definitions_json)
            assert math.isnan(value_roundtripped)
        case _:
            assert type(value_roundtripped) is type(value)
            assert value_roundtripped == value
