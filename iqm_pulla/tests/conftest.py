# Copyright 2024-2025 IQM
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
"""Pulla tests root."""

from http import HTTPStatus
from importlib.metadata import version
import json
import os
from pathlib import Path
from unittest.mock import Mock
from uuid import UUID

from httpx import Response as HTTPResponse
from iqm.iqm_client.models import (
    DynamicQuantumArchitecture,
    GateImplementationInfo,
    GateInfo,
)
from iqm.qiskit_iqm.iqm_provider import IQMBackend, IQMClient
from mockito import ANY, mock, when
import pytest
import requests
from requests import Response

from exa.common.api import proto_serialization
from exa.common.data.setting_node import SettingNode
from exa.common.qcm_data.chip_topology import ChipTopology
from iqm.pulla.calibration import CalibrationDataProvider
from iqm.pulla.pulla import Pulla
from iqm.pulla.utils_qiskit import IQMPullaBackend
from iqm.station_control.client.iqm_server import proto
from iqm.station_control.client.iqm_server.testing.iqm_server_mock import IqmServerMockBase
from iqm.station_control.client.station_control import StationControlClient

RESOURCES = Path(os.path.abspath(__name__)).parent / "tests" / "resources"


@pytest.fixture(scope="module")
def chip_topology() -> ChipTopology:
    """ChipTopology constructed from chip design record."""
    path = RESOURCES / "fake_chip_design_record.json"
    with open(path, mode="r", encoding="utf-8") as f:
        record = json.load(f)
    return ChipTopology.from_chip_design_record(record)


@pytest.fixture(scope="module")
def chip_topology_star() -> ChipTopology:
    """ChipTopology for Star variant, constructed from chip design record."""
    path = RESOURCES / "chip_design_record_star.json"
    with open(path, mode="r", encoding="utf-8") as f:
        record = json.load(f)
    return ChipTopology.from_chip_design_record(record)


@pytest.fixture(params=["station-control", "iqm-server"])
def pulla_on_spark(request, monkeypatch):
    """Pulla instance that mocks connection with a Spark system."""
    backend = request.param
    root_url = "https://fake.iqm.fi"

    # Provide mocks for Station Control backend
    def mocked_requests_get(*args, **kwargs):
        if args[0] == f"{root_url}/spark/about":
            response = Response()
            response.status_code = HTTPStatus.OK
            response.json = lambda: {
                "iqm_server": True,
            }
            return response
        # TODO SW-1387: Use v1 API
        # if args[0] == f"{root_url}/station/v1/about":
        if args[0] == f"{root_url}/station/about":
            response = Mock(spec=Response)
            response.status_code = HTTPStatus.OK
            data = {"software_versions": {"iqm-station-control-client": version("iqm-station-control-client")}}
            response.text = json.dumps(data)
            response.json = lambda: data
            response.ok = True
            return response
        # TODO SW-1387: Use v1 API
        # if args[0] == f"{root_url}/station/v1/duts":
        if args[0] == f"{root_url}/station/duts":
            response = Mock(spec=Response)
            response.status_code = HTTPStatus.OK
            response.text = json.dumps([{"label": "M000_fake_0_0", "dut_type": "chip"}])
            return response
        # TODO SW-1387: Use v1 API
        # if args[0].startswith(f"{root_url}/station/v1/sweeps/"):
        if args[0].startswith(f"{root_url}/station/sweeps/"):
            response = Response()
            response.status_code = HTTPStatus.INTERNAL_SERVER_ERROR
            return response
        if args[0].startswith(f"{root_url}/cocos/info/client-libraries"):
            response = Response()
            response.status_code = HTTPStatus.INTERNAL_SERVER_ERROR
            return response

        return HTTPResponse(404)

    def mocked_requests_post(*args, **kwargs):
        # TODO SW-1387: Use v1 API
        # if args[0] == f"{root_url}/station/v1/sweeps":
        if args[0] == f"{root_url}/station/sweeps":
            response = Response()
            response.status_code = HTTPStatus.OK
            response.json = lambda: {
                "job_id": "c2d31ae9-e749-4835-8450-0df10be5d1c1",
                # TODO SW-1387: Use v1 API
                # "job_href": f"{root_url}/station/v1/jobs/c2d31ae9-e749-4835-8450-0df10be5d1c1",
                # "sweep_href": f"{root_url}/station/v1/sweeps/8a28be71-b819-419d-bfcb-9ed9186b7473",
                "job_href": f"{root_url}/station/jobs/c2d31ae9-e749-4835-8450-0df10be5d1c1",
            }
            return response
        return HTTPResponse(404)

    monkeypatch.setattr(requests, "get", mocked_requests_get)
    monkeypatch.setattr(requests, "post", mocked_requests_post)

    with open(RESOURCES / "spark_settings.json", "r", encoding="utf-8") as file:
        settings = SettingNode(**json.loads(file.read()))
        settings_proto = proto_serialization.setting_node.pack(settings, minimal=False)
        settings_bytes = settings_proto.SerializeToString()
    monkeypatch.setattr(StationControlClient, "get_settings", lambda self: settings)

    with open(RESOURCES / "spark_chip_design_record.json", "r", encoding="utf-8") as file:
        design_record_str = file.read()
        record = json.loads(design_record_str)
    monkeypatch.setattr(StationControlClient, "get_chip_design_record", lambda self, label: record)

    with open(RESOURCES / "spark_calibration_set_raw.json", "r", encoding="utf-8") as file:
        cal = json.loads(file.read()), "fbaa6256-ab83-4217-8b7b-07c1952ec236"
    monkeypatch.setattr(CalibrationDataProvider, "get_latest_calibration_set", lambda self, label: cal)
    monkeypatch.setattr(CalibrationDataProvider, "get_calibration_set", lambda self, label: cal[0])

    # Provide mocks for IQM server backend
    class IqmServerMockBackend(IqmServerMockBase):
        def ListQuantumComputersV1(self, request: proto.ListQuantumComputerFiltersV1, context):
            return proto.QuantumComputersListV1(
                items=[
                    proto.QuantumComputerV1(
                        id=self.proto_uuid(UUID("c2d31ae9-e749-4835-8450-0df10be5d1c1")),
                        alias="spark",
                        display_name="Spark",
                    )
                ]
            )

        def GetQuantumComputerResourceV1(self, request: proto.QuantumComputerResourceLookupV1, context):
            match request.resource_name:
                case "duts":
                    return self.chunk_stream(
                        json.dumps([{"label": "M000_fake_0_0", "dut_type": "chip"}]).encode("utf-8")
                    )
                case "settings":
                    return self.chunk_stream(settings_bytes)
                case "chip-design-records/M000_fake_0_0":
                    return self.chunk_stream(design_record_str.encode("utf-8"))
                case "about":
                    return self.chunk_stream(
                        json.dumps(
                            {"software_versions": {"iqm-station-control-client": version("iqm-station-control-client")}}
                        ).encode("utf-8")
                    )
            return self.chunk_stream(bytearray())

        def SubmitJobV1(self, request: proto.SubmitJobRequestV1, context):
            now = self.proto_timestamp()
            return proto.JobV1(
                id=self.proto_uuid(UUID("8a28be71-b819-419d-bfcb-9ed9186b7473")),
                type=proto.JobType.PULSE,
                status=proto.JobStatus.IN_QUEUE,
                created_at=now,
                updated_at=now,
            )

    if backend == "iqm-server":
        pulla = Pulla(
            station_control_url=f"{root_url}/spark",
            grpc_channel=IqmServerMockBackend().channel(),
        )
    else:
        pulla = Pulla(station_control_url=f"{root_url}/station")
    return pulla


@pytest.fixture
def qiskit_backend_spark(monkeypatch) -> IQMBackend:
    """IQMBackend instance that mocks connection with a Spark system."""
    root_url = "https://fake.iqm.fi"
    calset_id = UUID("26c5e70f-bea0-43af-bd37-6212ec7d04cb")
    dqa = DynamicQuantumArchitecture(
        calibration_set_id=calset_id,
        qubits=["QB1", "QB2", "QB3", "QB4", "QB5"],
        computational_resonators=[],
        gates={
            "prx": GateInfo(
                implementations={
                    "drag_gaussian": GateImplementationInfo(loci=(("QB1",), ("QB2",), ("QB3",))),
                },
                default_implementation="drag_gaussian",
                override_default_implementation={},
            ),
            "cz": GateInfo(
                implementations={
                    "tgss": GateImplementationInfo(
                        loci=(
                            ("QB1", "QB3"),
                            ("QB2", "QB3"),
                        )
                    ),
                },
                default_implementation="tgss",
                override_default_implementation={},
            ),
            "measure": GateInfo(
                implementations={
                    "constant": GateImplementationInfo(loci=(("QB1",), ("QB2",), ("QB3",), ("QB4",), ("QB5",)))
                },
                default_implementation="constant",
                override_default_implementation={},
            ),
        },
    )

    monkeypatch.setattr(IQMClient, "get_dynamic_quantum_architecture", lambda self, calset_id: dqa)
    monkeypatch.setattr(StationControlClient, "_check_api_versions", lambda self: None)
    mock_about_response = mock(spec=Response)
    when(mock_about_response).raise_for_status().thenReturn(None)
    when(mock_about_response).json().thenReturn({})
    when(requests).get(f"{root_url}/station/about", headers=ANY).thenReturn(mock_about_response)
    iqm_client = IQMClient(f"{root_url}/station", client_signature="test fixture")
    return IQMBackend(iqm_client, calibration_set_id=calset_id)


@pytest.fixture
def pulla_backend_spark(pulla_on_spark) -> IQMPullaBackend:
    compiler = pulla_on_spark.get_standard_compiler()

    def loci_1q(qubits: list) -> tuple:
        """One-qubit loci for the given qubits."""
        return tuple((q,) for q in qubits)

    qubits = ["QB1", "QB2", "QB3", "QB4", "QB5"]
    calset_id = UUID("26c5e70f-bea0-43af-bd37-6212ec7d04cb")
    dqa = DynamicQuantumArchitecture(
        calibration_set_id=calset_id,
        qubits=list(qubits),
        computational_resonators=[],
        gates={
            "prx": GateInfo(
                implementations={
                    "drag_gaussian": GateImplementationInfo(loci=loci_1q(qubits)),
                },
                default_implementation="drag_gaussian",
                override_default_implementation={},
            ),
            "cc_prx": GateInfo(
                implementations={
                    "prx_composite": GateImplementationInfo(loci=loci_1q(qubits)),
                },
                default_implementation="prx_composite",
                override_default_implementation={},
            ),
            "cz": GateInfo(
                implementations={
                    "tgss": GateImplementationInfo(
                        loci=(
                            ("QB1", "QB3"),
                            ("QB2", "QB3"),
                            ("QB3", "QB4"),
                            ("QB3", "QB5"),
                        )
                    ),
                },
                default_implementation="tgss",
                override_default_implementation={},
            ),
            "measure": GateInfo(
                implementations={
                    "constant": GateImplementationInfo(loci=loci_1q(qubits)),
                },
                default_implementation="constant",
                override_default_implementation={},
            ),
        },
    )

    return IQMPullaBackend(dqa, pulla_on_spark, compiler)
