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
import uuid

from mockito import expect, mock
from mockito.matchers import Any
import pytest
import requests

from exa.common.errors.station_control_errors import InternalServerError, UnauthorizedError


def test_sweep_is_called_successfully(
    station_control_client,
    mock_sweep_definition,
    mocked_response_generator,
    sweep_response_generator,
):
    sweep_id = str(mock_sweep_definition.sweep_id)
    job_id = str(uuid.uuid4())
    expect(requests, times=1).post(
        f"{station_control_client.root_url}/v1/sweeps",
        data=Any(bytes),
        headers={"Content-Type": "application/protobuf"},
    ).thenReturn(
        mocked_response_generator(HTTPStatus.CREATED, sweep_response_generator(job_id, sweep_id)),
    )
    response = station_control_client.sweep(mock_sweep_definition)
    assert response["job_id"] == job_id
    assert response["sweep_id"] == sweep_id


def test_sweep_with_exception(station_control_client, mock_sweep_definition, mocked_response_generator):
    expect(requests, times=1).post(
        f"{station_control_client.root_url}/v1/sweeps",
        data=Any(bytes),
        headers={"Content-Type": "application/protobuf"},
    ).thenReturn(
        mocked_response_generator(HTTPStatus.INTERNAL_SERVER_ERROR),
    )
    with pytest.raises(InternalServerError, match=r"Ohno"):
        station_control_client.sweep(mock_sweep_definition)


def test_sweep_with_auth(
    station_control_client_with_auth,
    mock_sweep_definition,
    mocked_response_generator,
    sweep_response_generator,
):
    sweep_id = str(mock_sweep_definition.sweep_id)
    job_id = str(uuid.uuid4())
    expect(requests, times=1).post(
        f"{station_control_client_with_auth.root_url}/v1/sweeps",
        data=Any(bytes),
        headers={"Content-Type": "application/protobuf", "Authorization": "Bearer VERY_SECRET_TOKEN"},
    ).thenReturn(
        mocked_response_generator(HTTPStatus.CREATED, sweep_response_generator(job_id, sweep_id)),
    )
    response = station_control_client_with_auth.sweep(mock_sweep_definition)
    assert response["job_id"] == job_id
    assert response["sweep_id"] == sweep_id


def test_unauthorized_sweep(station_control_client_with_auth, mock_sweep_definition, mocked_response_generator):
    expect(requests, times=1).post(
        f"{station_control_client_with_auth.root_url}/v1/sweeps",
        data=Any(bytes),
        headers={"Content-Type": "application/protobuf", "Authorization": "Bearer VERY_SECRET_TOKEN"},
    ).thenReturn(
        mocked_response_generator(HTTPStatus.UNAUTHORIZED),
    )
    with pytest.raises(UnauthorizedError, match="Unauthorized"):
        station_control_client_with_auth.sweep(mock_sweep_definition)


def test_abort_is_called_successfully(station_control_client):
    sweep_id = uuid.uuid4()
    expect(requests, times=1).post(
        f"{station_control_client.root_url}/v1/jobs/{sweep_id}/abort",
    ).thenReturn(
        mock({"ok": True}, spec=requests.Response),
    )
    station_control_client.abort_job(sweep_id)


def test_http_requests_have_full_headers(station_control_client_with_full_headers):
    """HTTP requests should have some default headers and other boilerplate.
    The client method we call here is arbitrary."""
    sweep_id = uuid.uuid4()
    expect(requests, times=1).post(
        f"{station_control_client_with_full_headers.root_url}/v1/jobs/{sweep_id}/abort",
        headers={"User-Agent": station_control_client_with_full_headers._signature},
        timeout=600,
    ).thenReturn(
        mock({"ok": True}, spec=requests.Response),
    )
    station_control_client_with_full_headers.abort_job(sweep_id)
