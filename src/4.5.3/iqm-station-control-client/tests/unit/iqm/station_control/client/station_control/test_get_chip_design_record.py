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

from mockito import contains, mock, when
import pytest
import requests

from exa.common.errors.station_control_errors import NotFoundError, UnauthorizedError
from exa.common.qcm_data.qcm_data_client import QCMDataClient


def test_cache_is_functioning_correctly(station_control_client, chip_design_record_data_1_1):
    response = mock({"ok": True, "json": lambda: chip_design_record_data_1_1}, spec=requests.Response)
    when(requests).get(contains(f"{station_control_client.root_url}/v1/chip-design-records/")).thenReturn(response)

    # First get should be a miss, and the result should be added to the cache.
    chip_design_record_1 = station_control_client.get_chip_design_record("SL1")

    # Second get should be a hit, and the result shouldn't be added to the cache again.
    chip_design_record_2 = station_control_client.get_chip_design_record("SL1")

    assert chip_design_record_2 is chip_design_record_1
    assert station_control_client.get_chip_design_record.cache_info().hits == 1
    assert station_control_client.get_chip_design_record.cache_info().misses == 1

    _ = station_control_client.get_chip_design_record("SL2")

    assert station_control_client.get_chip_design_record.cache_info().hits == 1
    assert station_control_client.get_chip_design_record.cache_info().misses == 2


def test_falls_back_to_local_qcm_on_not_found_error_with_env(
    set_fallback_env, station_control_client, chip_design_record_data_1_1
):
    err = NotFoundError("Not found")
    when(requests).get(contains(f"{station_control_client.root_url}/v1/chip-design-records/")).thenRaise(err)

    when(QCMDataClient).get_chip_design_record(...).thenReturn(chip_design_record_data_1_1)
    result = station_control_client.get_chip_design_record("foo")
    assert result == chip_design_record_data_1_1


def test_does_not_fall_back_to_local_qcm_on_404_without_env(station_control_client):
    err = NotFoundError("Not found")
    when(requests).get(contains(f"{station_control_client.root_url}/v1/chip-design-records/")).thenRaise(err)

    with pytest.raises(NotFoundError):
        station_control_client.get_chip_design_record("foo")


def test_get_chip_design_record_with_auth(station_control_client_with_auth, chip_design_record_data_1_1):
    response_1 = mock(
        {"ok": True, "json": lambda: chip_design_record_data_1_1},
        spec=requests.Response,
    )
    when(requests).get(
        f"{station_control_client_with_auth.root_url}/v1/chip-design-records/SL1",
        headers={"Authorization": "Bearer VERY_SECRET_TOKEN"},
    ).thenReturn(response_1)
    station_control_client_with_auth.get_chip_design_record("SL1")


def test_get_chip_design_record_with_auth_error(station_control_client_with_auth):
    response = mock(
        {
            "status_code": HTTPStatus.UNAUTHORIZED,
            "json": lambda: {"status_code": HTTPStatus.UNAUTHORIZED, "message": "Unauthorized"},
            "ok": False,
        },
        spec=requests.Response,
    )
    when(requests).get(
        f"{station_control_client_with_auth.root_url}/v1/chip-design-records/SL1",
        headers={"Authorization": "Bearer VERY_SECRET_TOKEN"},
    ).thenReturn(response)
    with pytest.raises(UnauthorizedError, match=r"[SomeError]"):
        station_control_client_with_auth.get_chip_design_record("SL1")
