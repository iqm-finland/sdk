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

from collections.abc import Callable
from http import HTTPStatus
import json
import os
from typing import Any
import uuid

from iqm.models.playlist import Playlist, Segment
from mockito import mock
from pydantic import BaseModel
import pytest
import requests

from exa.common.data.setting_node import SettingNode
from iqm.station_control.client.station_control import StationControlClient
from iqm.station_control.interface.models import JobData, JobError, JobResult, RunDefinition, SweepDefinition


class StationControlClientMock(StationControlClient):
    def __init__(self, get_token_callback: Callable[[], str] | None = None):
        root_url = "http://localhost"
        super().__init__(root_url=root_url, get_token_callback=get_token_callback)

    # TODO SW-1387: Remove when using v1 API
    def _check_api_versions(self) -> None:
        pass

    def _send_request(
        self, http_method: Callable[..., requests.Response], url_path: str, **kwargs
    ) -> requests.Response:
        # TODO SW-1387: Remove when using v1 API
        #  We add "v1" here for now, until station control client is updated to use "v1" API,
        #  and after that this temporary hack can be removed.
        return super()._send_request(http_method, "v1/" + url_path, **kwargs)


class StationControlClientMockNoDefaultHeaders(StationControlClientMock):
    """Remove default headers and timeout from ``requests`` calls to make mockito.expect calls simpler."""

    def _build_request_kwargs(self, **kwargs) -> dict[str, Any]:
        kwargs = super()._build_request_kwargs(**kwargs)
        del kwargs["headers"]["User-Agent"]
        if not kwargs["headers"]:
            del kwargs["headers"]
        del kwargs["timeout"]
        return kwargs


@pytest.fixture()
def station_control_client_with_full_headers() -> StationControlClient:
    """StationControlClient that does not actually connect anywhere and passes full headers to _send_request."""
    station_control_client = StationControlClientMock()
    return station_control_client


@pytest.fixture()
def station_control_client() -> StationControlClient:
    """StationControlClient that does not actually connect anywhere."""
    station_control_client = StationControlClientMockNoDefaultHeaders()
    return station_control_client


@pytest.fixture()
def station_control_client_with_auth() -> StationControlClient:
    """StationControlClient that does not actually connect anywhere and has bearer token for auth."""
    station_control_client = StationControlClientMockNoDefaultHeaders(
        get_token_callback=lambda: "Bearer VERY_SECRET_TOKEN"
    )
    return station_control_client


@pytest.fixture
def sweep_response_generator():
    def _sweep_response(job_id: str, sweep_id: str):
        return {
            "job_id": job_id,
            "sweep_id": sweep_id,
            "job_href": f"jobs/{job_id}",
            "sweep_href": f"sweeps/{sweep_id}",
        }

    return _sweep_response


@pytest.fixture
def job_response() -> JobData:
    job_id = uuid.uuid4()
    return JobData(
        job_id=job_id,
        job_status="ready",
        job_result=JobResult(job_id=job_id, parallel_sweep_progress=[], interrupted=False),
        job_error=JobError(full_error_log="", user_error_message=""),
        position=0,
        is_position_capped=False,
    )


@pytest.fixture
def mocked_response_generator():
    def _mocked_response_generator(status_code: int, body: dict[str, Any] | None = None):
        response_body: dict[str, Any] | None = None
        match status_code:
            case HTTPStatus.UNAUTHORIZED:
                response_body = {"status_code": HTTPStatus.UNAUTHORIZED, "message": "Unauthorized"}
            case HTTPStatus.INTERNAL_SERVER_ERROR:
                response_body = {"status_code": HTTPStatus.INTERNAL_SERVER_ERROR, "message": "Ohno"}
            case _:
                response_body = body
        return mock(
            {
                "status_code": status_code,
                "ok": status_code < 400,
                "json": lambda: response_body,
                "text": json.dumps(response_body),
            },
            spec=requests.Response,
        )

    return _mocked_response_generator


@pytest.fixture
def mocked_response_generator_pydantic():
    """All responses that have Pydantic models should use this mock generator."""

    def _mocked_response_generator_pydantic(status_code: int, body: BaseModel | None = None):
        response_body: dict[str, Any] | None = None
        match status_code:
            case HTTPStatus.UNAUTHORIZED | HTTPStatus.INTERNAL_SERVER_ERROR:
                if status_code == HTTPStatus.UNAUTHORIZED:
                    response_body = json.dumps({"status_code": HTTPStatus.UNAUTHORIZED, "message": "Unauthorized"})
                else:
                    response_body = json.dumps({"status_code": HTTPStatus.INTERNAL_SERVER_ERROR, "message": "Ohno"})
                return mock(
                    {
                        "status_code": status_code,
                        "ok": status_code < 400,
                        "json": lambda: json.loads(response_body),
                    },
                    spec=requests.Response,
                )
            case _:
                response_body = "" if body is None else body.model_dump_json()
                return mock(
                    {
                        "status_code": status_code,
                        "ok": status_code < 400,
                        "text": response_body,
                    },
                    spec=requests.Response,
                )

    return _mocked_response_generator_pydantic


@pytest.fixture
def chip_design_record_data_1_1():
    return {
        "mask_set_name": "M139",
        "variant": "N70",
        "content_format_version": "1.1",
        "content": {
            "id": {"variant": "N70", "mask_set_name": "M139"},
            "schema": "https://iqm.tech/chip_architecture_definition_schema_v1.1.json",
            "components": {
                "qubit": [
                    {"name": "QB1", "connections": ["FL-QB1", "DL-QB1", "TC-1-3", "PL_RO-1", "COMP_R"]},
                    {"name": "QB2", "connections": ["FL-QB2", "DL-QB2", "TC-2-3", "PL_RO-1"]},
                    {"name": "QB3", "connections": ["FL-QB3", "DL-QB3", "TC-1-3", "TC-2-3", "PL_RO-1"]},
                ],
                "launcher": [
                    {"pin": "1", "name": "RO-1", "function": "probe_in", "connections": ["PL_RO-1"]},
                    {"pin": "2", "name": "RO-2", "function": "probe_out", "connections": ["PL_RO-1"]},
                    {"pin": "3", "name": "FL-QB1", "function": "flux", "connections": ["QB1"]},
                    {"pin": "4", "name": "FL-QB2", "function": "flux", "connections": ["QB2"]},
                    {"pin": "5", "name": "FL-QB3", "function": "flux", "connections": ["QB3"]},
                    {"pin": "6", "name": "FL-TC-1-2", "function": "flux", "connections": ["TC-1-2"]},
                    {"pin": "7", "name": "FL-TC-2-3", "function": "flux", "connections": ["TC-2-3"]},
                    {"pin": "8", "name": "DL-QB1", "function": "drive", "connections": ["QB1"]},
                    {"pin": "9", "name": "DL-QB2", "function": "drive", "connections": ["QB2"]},
                    {"pin": "10", "name": "DL-QB3", "function": "drive", "connections": ["QB3"]},
                ],
                "probe_line": [
                    {
                        "name": "PL_RO-1",
                        "connections": ["QB1", "QB2", "QB3", "RO-1", "RO-2"],
                    }
                ],
                "tunable_coupler": [
                    {"name": "TC-1-2", "connections": ["FL-TC-1-2", "QB1", "QB2", "COMP_R"]},
                    {"name": "TC-2-3", "connections": ["FL-TC-2-3", "QB2", "QB3", "COMP_R"]},
                ],
                "computational_resonator": [{"name": "COMP_R", "connections": ["TC-1-2", "TC-2-3", "QB1"]}],
            },
        },
    }


@pytest.fixture
def mock_sweep_definition():
    return SweepDefinition(
        sweep_id=uuid.uuid4(),
        dut_label="M138_W36_A22_N05",
        settings=SettingNode("root"),
        sweeps=[],
        return_parameters=[],
        playlist=Playlist(channel_descriptions={}, segments=[Segment()]),
    )


@pytest.fixture
def mock_run_definition(mock_sweep_definition):
    return RunDefinition(
        run_id=uuid.uuid4(),
        username="user",
        experiment_name="experiment_name",
        experiment_label="experiment_label",
        sweep_definition=mock_sweep_definition,
        software_version_set_id=42,
    )


@pytest.fixture
def set_fallback_env():
    """Sets an environment variable for the scope of one test. This needs to be defined before station control init."""
    # setup
    env_variable = "CHIP_DESIGN_RECORD_FALLBACK_URL"
    os.environ[env_variable] = "www.some.url"

    # run test
    yield

    # teardown
    os.environ.pop(env_variable)


@pytest.fixture
def mock_list_response():
    return mock(
        {
            "text": json.dumps(
                {
                    "items": [],
                    "meta": {
                        "count": 1,
                        "order_by": "-created_timestamp",
                        "limit": 20,
                        "offset": 0,
                    },
                }
            ),
            "ok": True,
        },
        spec=requests.Response,
    )
