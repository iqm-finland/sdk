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

import json
from math import pi
import os
import uuid

from iqm.iqm_client import IQMClient
from iqm.iqm_server_client.models import CalibrationSet, ListQuantumComputersResponse, QuantumComputer
from mockito import ANY, mock, when
import pytest
import requests
from requests import HTTPError, Response

from iqm.pulse import Circuit, CircuitOperation
from iqm.station_control.interface import models as sc_models
from iqm.station_control.interface.models import ObservationLite, ObservationSetData, QualityMetrics


def mock_quantum_computers_list_request(base_url: str, aliases: list[str]):
    response = mock(
        {
            "text": ListQuantumComputersResponse(
                quantum_computers=[QuantumComputer(id=uuid.uuid4(), alias=alias) for alias in aliases],
            ).model_dump_json(),
            "ok": True,
        },
        spec=requests.Response,
    )
    when(requests).get(f"{base_url}/api/v1/quantum-computers", headers=ANY, timeout=ANY).thenReturn(response)


@pytest.fixture()
def base_url() -> str:
    # NOTE: You should mock all HTTP requests in the tests, so we do not send out actual HTTP requests here!
    url = "https://example.com"
    mock_quantum_computers_list_request(url, ["default"])
    return url


@pytest.fixture(scope="function")
def iqm_client_mock(base_url) -> IQMClient:
    client = IQMClient(base_url)
    return client


@pytest.fixture()
def sample_calset_id() -> uuid.UUID:
    return uuid.UUID("9ddb9586-8f27-49a9-90ed-41086b47f6bd")


@pytest.fixture
def crystal_5_quality_metrics(crystal_5_calibration_set_observations):
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
        "tests/iqm_client/resources/crystal_5_quality_metric_set.json",
    )
    with open(path, "r", encoding="utf-8") as data:
        quality_metric_set_data = json.load(data)

    # Convert the list of observation dicts to a list of ObservationLite model instances
    observations = [
        ObservationLite.model_construct(**observation) for observation in quality_metric_set_data["observations"]
    ]

    return QualityMetrics(
        observation_set_type=quality_metric_set_data["observation_set_type"],
        describes_id=quality_metric_set_data["describes_id"],
        invalid=quality_metric_set_data["invalid"],
        dut_label=quality_metric_set_data["dut_label"],
        observation_set_id=quality_metric_set_data["observation_set_id"],
        created_timestamp=quality_metric_set_data["created_timestamp"],
        end_timestamp=quality_metric_set_data["end_timestamp"],
        observations=observations,
        calibration_set=ObservationSetData(
            observation_set_type=quality_metric_set_data["observation_set_type"],
            describes_id=quality_metric_set_data["describes_id"],
            invalid=quality_metric_set_data["invalid"],
            dut_label=quality_metric_set_data["dut_label"],
            observation_set_id=quality_metric_set_data["observation_set_id"],
            created_timestamp=quality_metric_set_data["created_timestamp"],
            end_timestamp=quality_metric_set_data["end_timestamp"],
            observation_ids=[observation.observation_id for observation in crystal_5_calibration_set_observations],
        ),
    )


@pytest.fixture
def crystal_5_calibration_set(crystal_5_quality_metrics, crystal_5_calibration_set_observations):
    return CalibrationSet(
        observation_set_type=crystal_5_quality_metrics.observation_set_type,
        describes_id=crystal_5_quality_metrics.describes_id,
        invalid=crystal_5_quality_metrics.invalid,
        dut_label=crystal_5_quality_metrics.dut_label,
        observation_set_id=crystal_5_quality_metrics.observation_set_id,
        created_timestamp=crystal_5_quality_metrics.created_timestamp,
        end_timestamp=crystal_5_quality_metrics.end_timestamp,
        observations=crystal_5_calibration_set_observations,
    )


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

    @property
    def ok(self):
        return self.status_code < 400


@pytest.fixture
def sample_dynamic_architecture(sample_calset_id) -> sc_models.DynamicQuantumArchitecture:
    return sc_models.DynamicQuantumArchitecture(
        calibration_set_id=sample_calset_id,
        qubits=["QB1", "QB2", "QB3"],
        computational_resonators=[],
        gates={
            "prx": sc_models.GateInfo(
                implementations={
                    "drag_gaussian": sc_models.GateImplementationInfo(loci=(("QB1",), ("QB2",), ("QB3",))),
                    "drag_crf": sc_models.GateImplementationInfo(loci=(("QB1",), ("QB3",))),
                },
                default_implementation="drag_gaussian",
                override_default_implementation={},
            ),
            "cc_prx": sc_models.GateInfo(
                implementations={
                    "prx_composite": sc_models.GateImplementationInfo(loci=(("QB1",), ("QB2",), ("QB3",))),
                },
                default_implementation="prx_composite",
                override_default_implementation={},
            ),
            "cz": sc_models.GateInfo(
                implementations={
                    "tgss": sc_models.GateImplementationInfo(loci=(("QB1", "QB2"), ("QB1", "QB3"))),
                    "crf": sc_models.GateImplementationInfo(loci=(("QB1", "QB2"),)),
                },
                default_implementation="tgss",
                override_default_implementation={},
            ),
            "measure": sc_models.GateInfo(
                implementations={"constant": sc_models.GateImplementationInfo(loci=(("QB1",), ("QB2",)))},
                default_implementation="constant",
                override_default_implementation={},
            ),
        },
    )


def create_sample_circuit(qubits: list[str], metadata) -> Circuit:
    return Circuit(
        # All unicode chars must work
        name="The circuit 😈",
        instructions=(
            CircuitOperation(
                name="cz",
                locus=tuple(qubits),
                args={},
            ),
            CircuitOperation(
                name="prx",
                implementation="drag_gaussian",
                locus=(qubits[0],),
                args={"phase": 1.4 * pi, "angle": 0.5 * pi},
            ),
            CircuitOperation(
                name="prx",
                locus=(qubits[0],),
                args={"phase": 0.6 * pi, "angle": -0.4 * pi},
            ),
            CircuitOperation(name="measure", locus=(qubits[0],), args={"key": "A"}),
            CircuitOperation(name="measure", locus=(qubits[1],), args={"key": "B"}),
        ),
        metadata=metadata,
    )


@pytest.fixture()
def sample_circuit_metadata():
    return {"experiment_type": "test", "qubits": (0, 1), "values": [0.01686514, 0.05760602]}


@pytest.fixture
def sample_circuit(sample_circuit_metadata) -> Circuit:
    """
    A sample circuit for testing submit_circuit
    """
    return create_sample_circuit(["QB1", "QB2"], metadata=sample_circuit_metadata)
