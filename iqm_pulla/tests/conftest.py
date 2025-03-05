# Copyright 2024 IQM
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
from uuid import UUID

from httpx import Response as HTTPResponse
from iqm.iqm_client.models import (
    DynamicQuantumArchitecture,
    GateImplementationInfo,
    GateInfo,
    QuantumArchitectureSpecification,
)
from iqm.qiskit_iqm.iqm_provider import IQMBackend, IQMClient
import pytest
import requests
from requests import Response

from exa.common.data.setting_node import SettingNode
from exa.common.qcm_data.chip_topology import ChipTopology
from iqm.pulla.calibration import CalibrationDataProvider
from iqm.pulla.pulla import Pulla
from iqm.pulla.utils_qiskit import IQMPullaBackend
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
def chip_topology_ndonis() -> ChipTopology:
    """ChipTopology for NDonis variant, constructed from chip design record."""
    path = RESOURCES / "chip_design_record_ndonis.json"
    with open(path, mode="r", encoding="utf-8") as f:
        record = json.load(f)
    return ChipTopology.from_chip_design_record(record)


@pytest.fixture
def pulla_on_spark(monkeypatch):
    """Pulla instance that mocks connection with a Spark system."""
    root_url = "https://fake.iqm.fi"

    def mocked_requests_get(*args, **kwargs):
        if args[0] == f"{root_url}/station/about":
            response = Response()
            response.status_code = HTTPStatus.OK
            response.json = lambda: {
                "software_versions": {"iqm-station-control-client": version("iqm-station-control-client")}
            }
            return response
        if args[0] == f"{root_url}/station/duts":
            response = Response()
            response.status_code = HTTPStatus.OK
            response.json = lambda: [{"label": "M000_fake_0_0", "dut_type": "chip"}]
            return response
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
        if args[0] == f"{root_url}/station/sweeps":
            response = Response()
            response.status_code = HTTPStatus.OK
            response.json = lambda: {
                "task_id": "c2d31ae9-e749-4835-8450-0df10be5d1c1",
                "sweep_id": "8a28be71-b819-419d-bfcb-9ed9186b7473",
                "task_href": f"{root_url}/station/tasks/c2d31ae9-e749-4835-8450-0df10be5d1c1",
                "sweep_href": f"{root_url}/station/sweeps/8a28be71-b819-419d-bfcb-9ed9186b7473",
            }
            return response

        return HTTPResponse(404)

    monkeypatch.setattr(requests, "get", mocked_requests_get)
    monkeypatch.setattr(requests, "post", mocked_requests_post)

    with open(RESOURCES / "spark_settings.json", "r", encoding="utf-8") as file:
        settings = SettingNode(**json.loads(file.read()))
    monkeypatch.setattr(StationControlClient, "get_settings", lambda self: settings)

    with open(RESOURCES / "spark_chip_design_record.json", "r", encoding="utf-8") as file:
        record = json.loads(file.read())
    monkeypatch.setattr(StationControlClient, "get_chip_design_record", lambda self, label: record)

    with open(RESOURCES / "spark_calibration_set_raw.json", "r", encoding="utf-8") as file:
        cal = json.loads(file.read()), "fbaa6256-ab83-4217-8b7b-07c1952ec236"
    monkeypatch.setattr(CalibrationDataProvider, "get_latest_calibration_set", lambda self, label: cal)
    monkeypatch.setattr(CalibrationDataProvider, "get_calibration_set", lambda self, label: cal[0])

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
    if hasattr(IQMClient, "_check_versions"):  # _check_versions was introduced in a minor version of IQM Client
        monkeypatch.setattr(IQMClient, "_check_versions", lambda self: None)
    client = IQMClient(f"{root_url}/cocos", client_signature="test fixture")
    return IQMBackend(client, calibration_set_id=calset_id)


@pytest.fixture
def pulla_backend_spark(pulla_on_spark) -> IQMPullaBackend:
    compiler = pulla_on_spark.get_standard_compiler()
    architecture = QuantumArchitectureSpecification(
        name="Adonis",
        operations={
            "prx": [["QB1"], ["QB2"], ["QB3"], ["QB4"], ["QB5"]],
            "cc_prx": [["QB1"], ["QB2"], ["QB3"], ["QB4"], ["QB5"]],
            "cz": [["QB1", "QB3"], ["QB2", "QB3"], ["QB4", "QB3"], ["QB5", "QB3"]],
            "measure": [["QB1"], ["QB2"], ["QB3"], ["QB4"], ["QB5"]],
            "barrier": [],
        },
        qubits=["QB1", "QB2", "QB3", "QB4", "QB5"],
        qubit_connectivity=[["QB1", "QB3"], ["QB2", "QB3"], ["QB3", "QB4"], ["QB3", "QB5"]],
    )

    return IQMPullaBackend(architecture, pulla_on_spark, compiler)
