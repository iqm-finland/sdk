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

from exa.common.errors.station_control_errors import UnauthorizedError


def test_only_kwargs_given_should_succeed(station_control_client, mock_list_response):
    when(requests).get(f"{station_control_client.root_url}/v1/observation-sets", params={"invalid": True}).thenReturn(
        mock_list_response
    )

    station_control_client.query_observation_sets(invalid=True)


def test_invalid_false_is_added_to_request_params_by_default(station_control_client, mock_list_response):
    # We want to get only valid observations by default, thus we use default "False" instead of "None",
    # which would return all observations regardless of the "invalid" value.
    when(requests).get(f"{station_control_client.root_url}/v1/observation-sets", params={"invalid": False}).thenReturn(
        mock_list_response
    )

    # Call query_observations() without "invalid" argument,
    # but it should still use the stub above with "invalid": False.
    # If not, then this test would fail with mockito.invocation.InvocationError: Called but not expected.
    station_control_client.query_observation_sets()


def test_args_given_should_fail(station_control_client):
    with pytest.raises(TypeError, match="takes 1 positional argument but 2 were given"):
        station_control_client.query_observation_sets(1)


def test_get_observations_with_auth(station_control_client_with_auth, mock_list_response):
    when(requests).get(
        f"{station_control_client_with_auth.root_url}/v1/observation-sets",
        params={"invalid": False},
        headers={"Authorization": "Bearer VERY_SECRET_TOKEN"},
    ).thenReturn(mock_list_response)

    station_control_client_with_auth.query_observation_sets()


def test_get_observations_with_auth_error(station_control_client_with_auth):
    response = mock(
        {
            "status_code": HTTPStatus.UNAUTHORIZED,
            "json": lambda: {"status_code": HTTPStatus.UNAUTHORIZED, "message": "Unauthorized"},
            "ok": False,
        },
        spec=requests.Response,
    )
    when(requests).get(
        f"{station_control_client_with_auth.root_url}/v1/observation-sets",
        params={"invalid": False},
        headers={"Authorization": "Bearer VERY_SECRET_TOKEN"},
    ).thenReturn(response)

    with pytest.raises(UnauthorizedError, match=r"[SomeError]"):
        station_control_client_with_auth.query_observation_sets()
