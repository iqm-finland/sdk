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
from mockito.matchers import Any
import pytest
import requests

from exa.common.errors.station_control_errors import UnauthorizedError


def test_post_abort_is_called_on_keyboard_interrupt(station_control_client, mock_run_definition):
    response_1 = mock(
        {
            "json": lambda: {"job_id": "2e0b8eb5-2599-4db9-86ca-91dff88a1506"},
            "ok": True,
        },
        spec=requests.Response,
    )
    when(requests).post(
        f"{station_control_client.root_url}/v1/runs",
        data=Any(bytes),
        headers={"Content-Type": "application/protobuf"},
    ).thenReturn(response_1)

    response_2 = mock({"ok": True}, spec=requests.Response)
    when(requests).get(
        f"{station_control_client.root_url}/v1/jobs/2e0b8eb5-2599-4db9-86ca-91dff88a1506",
    ).thenRaise(KeyboardInterrupt())

    when(requests).post(
        f"{station_control_client.root_url}/v1/jobs/2e0b8eb5-2599-4db9-86ca-91dff88a1506/abort",
    ).thenReturn(response_2)
    assert station_control_client.run(mock_run_definition)


def test_run_with_auth(station_control_client_with_auth, mock_run_definition):
    response_1 = mock(
        {
            "json": lambda: {"job_id": "2e0b8eb5-2599-4db9-86ca-91dff88a1506"},
            "ok": True,
        },
        spec=requests.Response,
    )
    when(requests).post(
        f"{station_control_client_with_auth.root_url}/v1/runs",
        data=Any(bytes),
        headers={"Content-Type": "application/protobuf", "Authorization": "Bearer VERY_SECRET_TOKEN"},
    ).thenReturn(response_1)

    when(requests).get(
        f"{station_control_client_with_auth.root_url}/v1/jobs/2e0b8eb5-2599-4db9-86ca-91dff88a1506",
        headers={"Authorization": "Bearer VERY_SECRET_TOKEN"},
    ).thenRaise(KeyboardInterrupt())

    response_2 = mock({"ok": True}, spec=requests.Response)
    when(requests).post(
        f"{station_control_client_with_auth.root_url}/v1/jobs/2e0b8eb5-2599-4db9-86ca-91dff88a1506/abort",
        headers={"Authorization": "Bearer VERY_SECRET_TOKEN"},
    ).thenReturn(response_2)

    station_control_client_with_auth.run(mock_run_definition)


def test_run_with_auth_error(station_control_client_with_auth, mock_run_definition):
    response = mock(
        {
            "status_code": HTTPStatus.UNAUTHORIZED,
            "json": lambda: {"status_code": HTTPStatus.UNAUTHORIZED, "message": "Unauthorized"},
            "ok": False,
        },
        spec=requests.Response,
    )
    when(requests).post(
        f"{station_control_client_with_auth.root_url}/v1/runs",
        data=Any(bytes),
        headers={"Content-Type": "application/protobuf", "Authorization": "Bearer VERY_SECRET_TOKEN"},
    ).thenReturn(response)

    with pytest.raises(UnauthorizedError, match=r"[SomeError]"):
        station_control_client_with_auth.run(mock_run_definition)
