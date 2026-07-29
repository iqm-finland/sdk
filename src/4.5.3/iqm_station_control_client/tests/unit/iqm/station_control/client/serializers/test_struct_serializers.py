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

import pytest

from iqm.station_control.client.serializers.struct_serializer import deserialize_struct, serialize_struct


def test_serialize_struct_round_trip():
    struct_data = {
        "additional_run_property_0": None,
        "additional_run_property_1": 1,
        "additional_run_property_2": 2.0,
        "additional_run_property_3": ["a", "b", "c"],
        "additional_run_property_4": [[1, 2], [1], [2.0], [["a", "b"], [1, 1.0]]],
        "additional_run_property_5": {
            "a_result": "a_value",
            "b_result": "b_value",
            "c_result": "c_value",
        },
    }
    serialized_struct_data = serialize_struct(struct_data)
    deserialized_struct_data = deserialize_struct(serialized_struct_data)

    assert deserialized_struct_data == struct_data
    assert deserialized_struct_data["additional_run_property_0"] is None
    assert isinstance(deserialized_struct_data["additional_run_property_1"], int)
    assert deserialized_struct_data["additional_run_property_1"] == 1
    assert isinstance(deserialized_struct_data["additional_run_property_2"], float)
    assert deserialized_struct_data["additional_run_property_2"] == 2.0
    assert deserialized_struct_data["additional_run_property_3"] == ["a", "b", "c"]
    assert deserialized_struct_data["additional_run_property_4"] == [
        [1, 2],
        [1],
        [2.0],
        [["a", "b"], [1, 1.0]],
    ]
    assert deserialized_struct_data["additional_run_property_5"] == {
        "a_result": "a_value",
        "b_result": "b_value",
        "c_result": "c_value",
    }


def test_serialize_large_integer_should_fail():
    struct_data = {
        "additional_run_property_0": 2**63 - 1,  # MAX 2**63-1
    }
    serialized_struct_data = serialize_struct(struct_data)
    deserialized_struct_data = deserialize_struct(serialized_struct_data)

    assert deserialized_struct_data == struct_data
    assert isinstance(deserialized_struct_data["additional_run_property_0"], int)
    assert deserialized_struct_data["additional_run_property_0"] == 2**63 - 1

    struct_data = {
        "additional_run_property_0": 2**63,  # MAX 2**63-1
    }
    with pytest.raises(ValueError, match="Value out of range"):
        serialize_struct(struct_data)
