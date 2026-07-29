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

from mockito import expect
import pytest
import requests

from exa.common.errors.station_control_errors import InternalServerError, UnauthorizedError


def test_getting_job_by_job_id(station_control_client, mocked_response_generator_pydantic, job_response):
    job_id = job_response.job_id
    expect(requests, times=1).get(
        f"{station_control_client.root_url}/v1/jobs/{job_id}",
    ).thenReturn(
        mocked_response_generator_pydantic(HTTPStatus.OK, job_response),
    )
    assert station_control_client.get_job(job_id) == job_response


def test_getting_job_fails(station_control_client, mocked_response_generator_pydantic):
    job_id = uuid.uuid4()
    expect(requests, times=1).get(
        f"{station_control_client.root_url}/v1/jobs/{job_id}",
    ).thenReturn(
        mocked_response_generator_pydantic(HTTPStatus.INTERNAL_SERVER_ERROR),
    )
    with pytest.raises(InternalServerError, match=r"Ohno"):
        station_control_client.get_job(job_id)


def test_getting_job_by_job_id_with_auth(
    station_control_client_with_auth, mocked_response_generator_pydantic, job_response
):
    job_id = job_response.job_id
    expect(requests, times=1).get(
        f"{station_control_client_with_auth.root_url}/v1/jobs/{job_id}",
        headers={"Authorization": "Bearer VERY_SECRET_TOKEN"},
    ).thenReturn(
        mocked_response_generator_pydantic(HTTPStatus.OK, job_response),
    )
    assert station_control_client_with_auth.get_job(job_id) == job_response


def test_getting_unauthorized_job(station_control_client_with_auth, mocked_response_generator_pydantic):
    job_id = uuid.uuid4()
    expect(requests, times=1).get(
        f"{station_control_client_with_auth.root_url}/v1/jobs/{job_id}",
        headers={"Authorization": "Bearer VERY_SECRET_TOKEN"},
    ).thenReturn(
        mocked_response_generator_pydantic(HTTPStatus.UNAUTHORIZED),
    )
    with pytest.raises(UnauthorizedError, match="Unauthorized"):
        station_control_client_with_auth.get_job(job_id)
