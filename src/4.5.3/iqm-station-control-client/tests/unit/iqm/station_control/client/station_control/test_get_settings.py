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
from http import HTTPStatus

from mockito import mock, when
import pytest
import requests

from exa.common.api import proto_serialization
from exa.common.data.parameter import Parameter, Setting
from exa.common.data.setting_node import SettingNode
from exa.common.errors.station_control_errors import UnauthorizedError

# Dummy `SettingNode` model with nested structure.
#
# Structure of the tree looks like:
# root
# ├── child
# └── setting
nested_setting_node = SettingNode(
    "root", child_key=SettingNode("child"), setting_key=Setting(parameter=Parameter("parameter"), value=10)
)


def _setup_get_settings(station_control_client):
    binary = proto_serialization.setting_node.pack(nested_setting_node, minimal=False).SerializeToString()
    response = mock({"ok": True, "content": binary}, spec=requests.Response)
    when(requests).get(f"{station_control_client.root_url}/v1/settings").thenReturn(response)
    station_control_client._get_cached_settings.cache_clear()


def test_binary_is_decoded(station_control_client):
    _setup_get_settings(station_control_client)
    result = station_control_client.get_settings()
    assert result == nested_setting_node


def test_cache_is_functioning_correctly(station_control_client):
    _setup_get_settings(station_control_client)

    # First get should be a miss, and the result should be added to the cache.
    settings_1 = station_control_client.get_settings()

    # Second get should be a hit, and the result shouldn't be added to the cache again.
    settings_2 = station_control_client.get_settings()

    assert settings_1 is not settings_2
    assert settings_1 == settings_2


def test_get_settings_with_auth(station_control_client_with_auth):
    binary = proto_serialization.setting_node.pack(nested_setting_node, minimal=False).SerializeToString()
    response = mock({"ok": True, "content": binary}, spec=requests.Response)
    when(requests).get(
        f"{station_control_client_with_auth.root_url}/v1/settings",
        headers={"Authorization": "Bearer VERY_SECRET_TOKEN"},
    ).thenReturn(response)
    station_control_client_with_auth.get_settings()


def test_get_settings_with_auth_error(station_control_client_with_auth):
    response = mock(
        {
            "status_code": 401,
            "json": lambda: {"status_code": HTTPStatus.UNAUTHORIZED, "message": "Unauthorized"},
            "ok": False,
        },
        spec=requests.Response,
    )
    when(requests).get(
        f"{station_control_client_with_auth.root_url}/v1/settings",
        headers={"Authorization": "Bearer VERY_SECRET_TOKEN"},
    ).thenReturn(response)
    with pytest.raises(UnauthorizedError, match=r"[SomeError]"):
        station_control_client_with_auth.get_settings()
