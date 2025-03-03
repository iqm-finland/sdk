# Copyright 2021-2023 IQM client developers
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
"""Tests for the IQM client utilities."""
import numpy as np
import pytest

from iqm.iqm_client.util import to_json_dict


def test_serialize_dict():
    """
    Tests that util.to_json_dict can handle a dict that is already JSON compatible
    """
    data = {"key1": True, "key2": None, "key3": 123.456, "key4": {"key5": ["a", "b", "c"], "key6": "value6"}}
    assert to_json_dict(data) == data


def test_serialize_dict_with_ndarray():
    """
    Tests that util.to_json_dict can handle a dict that contains numpy.ndarray value
    """
    original = np.array([[1, 2, 3], [4, 5, 6]])
    expected = [[1, 2, 3], [4, 5, 6]]
    json_dict = to_json_dict({"key1": {"key2": original}})
    assert json_dict == {"key1": {"key2": expected}}


def test_serialize_dict_with_unsupported_value():
    """
    Tests that util.to_json_dict raises ValueError if there is unsupported data in the dict.

    `to_json_dict` catches TypeError raised from JSON serialization and raises a ValueError from it.
    """
    original = {"key1": complex(1.0, 2.0)}
    with pytest.raises(ValueError):
        to_json_dict(original)


def test_serialize_dict_with_nan_value():
    """
    Tests that util.to_json_dict raises TypeError if there is unsupported data in the dict

    `to_json_dict` catches ValueError raised from JSON serialization and raises a new ValueError from it.
    """
    original = {"key1": float("NaN")}
    with pytest.raises(ValueError):
        to_json_dict(original)
