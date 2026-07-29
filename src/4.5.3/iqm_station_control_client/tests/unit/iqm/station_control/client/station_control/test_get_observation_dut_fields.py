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
import json

from mockito import mock, when
import pytest
import requests

from exa.common.errors.station_control_errors import UnauthorizedError


def test_get_observation_dut_fields(station_control_client):
    response = mock(
        {
            "text": json.dumps([{"path": "dut_label_1", "unit": "unit_1"}]),
            "ok": True,
        },
        spec=requests.Response,
    )
    when(requests).get(
        f"{station_control_client.root_url}/v1/dut-fields",
        params={"dut_label": "M101_W1_X01_A01"},
    ).thenReturn(response)

    response_dut_fields = station_control_client.get_dut_fields("M101_W1_X01_A01")
    assert len(response_dut_fields) == 1
    assert response_dut_fields[0].path == "dut_label_1"
    assert response_dut_fields[0].unit == "unit_1"


def test_get_observation_dut_fields_with_auth(station_control_client_with_auth):
    response = mock(
        {
            "text": json.dumps([{"path": "dut_label_1", "unit": "unit_1"}]),
            "ok": True,
        },
        spec=requests.Response,
    )
    when(requests).get(
        f"{station_control_client_with_auth.root_url}/v1/dut-fields",
        params={"dut_label": "M101_W1_X01_A01"},
        headers={"Authorization": "Bearer VERY_SECRET_TOKEN"},
    ).thenReturn(response)

    response_dut_fields = station_control_client_with_auth.get_dut_fields("M101_W1_X01_A01")
    assert len(response_dut_fields) == 1
    assert response_dut_fields[0].path == "dut_label_1"
    assert response_dut_fields[0].unit == "unit_1"


def test_get_observation_dut_fields_with_auth_error(station_control_client_with_auth):
    response = mock(
        {
            "status_code": HTTPStatus.UNAUTHORIZED,
            "json": lambda: {"status_code": HTTPStatus.UNAUTHORIZED, "message": "Unauthorized"},
            "ok": False,
        },
        spec=requests.Response,
    )
    when(requests).get(
        f"{station_control_client_with_auth.root_url}/v1/dut-fields",
        params={"dut_label": "M101_W1_X01_A01"},
        headers={"Authorization": "Bearer VERY_SECRET_TOKEN"},
    ).thenReturn(response)
    with pytest.raises(UnauthorizedError, match=r"[SomeError]"):
        station_control_client_with_auth.get_dut_fields("M101_W1_X01_A01")
