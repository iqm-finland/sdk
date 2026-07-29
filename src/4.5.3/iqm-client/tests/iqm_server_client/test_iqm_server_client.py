# Copyright 2021-2025 IQM client developers
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
"""Tests for the IQMServerClient."""

import uuid

from iqm.iqm_server_client.iqm_server_client import CircuitCountsBatchAdapter, _IQMServerClient
from mockito import ANY, expect, mock
import pytest
import requests

from iqm.station_control.client.authentication import ClientConfigurationError
from iqm.station_control.interface.models import CircuitMeasurementCounts

from ..conftest import mock_quantum_computers_list_request

pytestmark = pytest.mark.usefixtures("unstub")

JOB_ID = uuid.uuid4()


@pytest.fixture()
def iqm_server_client_mock(base_url) -> _IQMServerClient:
    return _IQMServerClient(base_url)


@pytest.fixture()
def measurement_counts_response(base_url):
    response = mock(
        {
            "text": CircuitCountsBatchAdapter.dump_json(
                [
                    CircuitMeasurementCounts(
                        measurement_keys=["m1"],
                        counts={"0": 5, "1": 5},
                    ),
                    CircuitMeasurementCounts(
                        measurement_keys=["m2"],
                        counts={"0": 1, "1": 9},
                    ),
                ]
            ),
            "ok": True,
        },
        spec=requests.Response,
    )
    expect(requests, times=1).get(
        f"{base_url}/api/v1/jobs/{JOB_ID}/artifacts/measurement_counts", headers=ANY, timeout=ANY
    ).thenReturn(response)
    return response


def test_quantum_computer_is_used_automatically_if_there_is_only_one_quantum_computer():
    base_url = "http://example.localhost"
    mock_quantum_computers_list_request(base_url, ["ixion"])
    client = _IQMServerClient(base_url)
    assert client.quantum_computer == "ixion"


def test_explicit_quantum_computer_is_required_if_there_are_multiple_quantum_computers():
    base_url = "http://example.localhost"
    mock_quantum_computers_list_request(base_url, ["ixion", "varda"])
    with pytest.raises(
        ClientConfigurationError, match="Quantum computer not selected. Available quantum computers are: ixion, varda"
    ):
        _IQMServerClient(base_url)

    client = _IQMServerClient(base_url, quantum_computer="varda")
    assert client.quantum_computer == "varda"


def test_explicitly_defined_quantum_computer_must_be_in_the_available_quantum_computers():
    base_url = "http://example.localhost"
    mock_quantum_computers_list_request(base_url, ["ixion", "varda"])
    with pytest.raises(
        ClientConfigurationError,
        match='Quantum computer "garnet" does not exist. Available quantum computers are: ixion, varda',
    ):
        _IQMServerClient(base_url, quantum_computer="garnet")


def test_get_job_artifact_measurement_counts(iqm_server_client_mock, measurement_counts_response):
    measurement_counts = iqm_server_client_mock.get_job_artifact_measurement_counts(JOB_ID)
    assert isinstance(measurement_counts, list)
    assert len(measurement_counts) == 2


def test_debug_info(base_url, monkeypatch):
    """Test the _debug_info method."""
    monkeypatch.setenv("IQM_TOKEN", "a" * 20)
    client = _IQMServerClient(base_url)
    about = {
        "iqm_server": True,
        "qccsw_version": "x.y.z-abcd",
        "server_version": "000",
        "station_control_version": "111",
    }
    expect(client, times=1).get_about().thenReturn(about)
    expect(client, times=1).get_about_station().thenReturn(
        {
            "version": "111",
            "software_versions": {
                "iqm-xxx": "222",
                "other_pkg": "333",
            },
        }
    )

    info = client._debug_info()
    local_pkgs = info["local packages"]
    assert "iqm-client" in local_pkgs
    assert "iqm-station-control-client" in local_pkgs
    assert "qiskit" in local_pkgs
    for key, value in {
        "platform.platform": None,
        "platform.version": None,
        "platform.python_version": None,
        "root_url": "https://example.com",
        "quantum_computer": "default",
        "use_timeslot": False,
        "len(IQM_TOKEN)": 20,
        "len(IQM_TOKENS_FILE)": None,
        "token_provider": None,
        "auth_header_callback": None,
        "about": about,
        "about_station": {
            "software_versions": {"iqm-xxx": "222"},
            "version": "111",
        },
    }.items():
        assert key in info
        if value is not None:
            assert info[key] == value


# TODO: Add # get_job_artifact tests for other types as well
#  Correct Python object should be returned rather than raw JSON

# TODO: Add missing unit tests
#  get_about
#  get_health
#  get_settings
#  get_chip_design_record
#  get_channel_properties
#  submit_sweep
#  get_sweep
#  get_sweep_results
#  get_observation_set
#  get_observation_set_observations
#  get_default_calibration_set
#  get_dynamic_quantum_architecture
#  get_default_calibration_set_quality_metrics
#  get_calibration_set_quality_metrics
#  get_duts
#  get_static_quantum_architecture
#  submit_circuits
#  get_job
#  get_job_status
#  get_job_timeline
#  cancel_job
#  wait_job_completion
