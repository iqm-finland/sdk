# Copyright 2025 IQM client developers
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Mocks server calls for testing
"""

from importlib.metadata import version
import json
import os

from iqm.iqm_client import IQMClient, TokenManager
from mockito import ANY, expect, when
from packaging.version import parse
import pytest
import requests
from requests import HTTPError, Response

from iqm.station_control.client.station_control import StationControlClient
from iqm.station_control.interface import models


@pytest.fixture()
def base_url() -> str:
    # NOTE: You should mock all HTTP requests in the tests, so we do not send out actual HTTP requests here!
    return "https://example.com"


@pytest.fixture(scope="function")
def iqm_client_mock(base_url) -> IQMClient:
    expect(requests, times=1).get(
        f"{base_url}/info/client-libraries",
        headers=ANY,
        timeout=ANY,
    ).thenReturn(mock_supported_client_libraries_response())
    when(requests).get(f"{base_url}/about", headers=ANY).thenReturn(MockJsonResponse(200, {}))

    when(StationControlClient)._check_api_versions().thenReturn(None)
    client = IQMClient(base_url)
    client._token_manager = TokenManager()  # Do not use authentication
    when(client._token_manager).get_bearer_token().thenReturn(None)
    return client


@pytest.fixture
def crystal_5_calibration_set():
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
        "tests/iqm_client/resources/crystal_5_calibration_set.json",
    )
    with open(path, "r", encoding="utf-8") as data:
        return json.load(data)

        # return models.ObservationSetData(
        #     observation_set_type="calibration-set",
        #     observation_ids=[obs["observation_id"] for obs in calset],
        # )


@pytest.fixture
def crystal_5_quality_metrics(crystal_5_calibration_set):
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
        "tests/iqm_client/resources/crystal_5_quality_metric_set.json",
    )
    with open(path, "r", encoding="utf-8") as data:
        quality_metric_set = json.load(data)

    calibration_set_data = models.ObservationSetData(
        observation_set_type="calibration-set",
        # Extract all observation IDs from the list of observations
        observation_ids=[obs["observation_id"] for obs in crystal_5_calibration_set if "observation_id" in obs],
        # Use metadata from the quality metric set as it's the context
        describes_id=quality_metric_set["describes_id"],
        invalid=quality_metric_set["invalid"],
        dut_label=quality_metric_set["dut_label"],
        observation_set_id=quality_metric_set["observation_set_id"],
        created_timestamp=quality_metric_set["created_timestamp"],
        end_timestamp=quality_metric_set["end_timestamp"],
    )

    # Convert the list of observation dicts to a list of ObservationLite model instances
    observations_models = [models.ObservationLite.model_construct(**obs) for obs in quality_metric_set["observations"]]

    return models.QualityMetrics(
        observation_set_type=quality_metric_set["observation_set_type"],
        observation_ids=quality_metric_set["observation_ids"],
        describes_id=quality_metric_set["describes_id"],
        invalid=quality_metric_set["invalid"],
        dut_label=quality_metric_set["dut_label"],
        observation_set_id=quality_metric_set["observation_set_id"],
        created_timestamp=quality_metric_set["created_timestamp"],
        end_timestamp=quality_metric_set["end_timestamp"],
        calibration_set=calibration_set_data,
        observations=observations_models,
    )


class MockBytesResponse:
    def __init__(self, status_code: int, content: bytes, history: list[Response] | None = None):
        self.status_code = status_code
        self.content = content
        self.media_type = "application/octet-stream"
        self.history = history
        self.url = "https://example.com"

    def raise_for_status(self):
        if 400 <= self.status_code < 600:
            raise HTTPError(f"{self.status_code}", response=self)


class MockJsonResponse:
    def __init__(self, status_code: int, json_data: dict | list[dict], history: list[Response] | None = None):
        self.status_code = status_code
        self.json_data = json_data
        self.history = history
        self.url = "https://example.com"

    @property
    def text(self):
        # NOTE cannot handle UUIDs
        return json.dumps(self.json_data)

    def json(self):
        return self.json_data

    def raise_for_status(self):
        if 400 <= self.status_code < 600:
            raise HTTPError(f"{self.status_code}", response=self)


def mock_supported_client_libraries_response(
    iqm_client_name: str = "iqm-client", max_version: str | None = None, min_version: str | None = None
) -> MockJsonResponse:
    client_version = parse(version("iqm-client"))
    min_version = f"{client_version.major}.0" if min_version is None else min_version
    max_version = f"{client_version.major + 1}.0" if max_version is None else max_version
    return MockJsonResponse(
        200,
        {
            iqm_client_name: {
                "name": iqm_client_name,
                "min": min_version,
                "max": max_version,
            }
        },
    )


def mock_about_response():
    return MockJsonResponse(200, {})
