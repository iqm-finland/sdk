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
from pathlib import Path
from unittest.mock import Mock
from uuid import UUID

from httpx import Response as HTTPResponse
from iqm.iqm_server_client.iqm_server_client import _IQMServerClient
from iqm.iqm_server_client.models import QuantumComputer
from iqm.qiskit_iqm.iqm_provider import IQMBackend, IQMClient
import pytest
import requests
from requests import Response

from exa.common.data.setting_node import SettingNode
from exa.common.qcm_data.chip_topology import ChipTopology
from iqm.pulla.calibration import CalibrationDataProvider
from iqm.pulla.pulla import Pulla
from iqm.pulla.utils_qiskit import IQMPullaBackend
from iqm.station_control.interface.models import DynamicQuantumArchitecture, GateImplementationInfo, GateInfo

RESOURCES = Path(__file__).parent / "resources"


@pytest.fixture(scope="module")
def chip_topology() -> ChipTopology:
    """ChipTopology constructed from chip design record."""
    path = RESOURCES / "chip_design_record.json"
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


@pytest.fixture
def pulla_on_spark(request, monkeypatch):
    """Pulla instance that mocks connection with a Spark system."""
    root_url = "https://fake.iqm.fi"

    # Provide mock responses for IQM Server requests
    def mocked_requests_get(*args, **kwargs):
        if args[0] == f"{root_url}/api/v1/quantum-computers":
            response = Mock(spec=Response)
            response.status_code = HTTPStatus.OK
            data = {
                "quantum_computers": [
                    QuantumComputer(id=UUID("1887449d-627e-48b0-a3d9-d971fa3bbd91"), alias="default").model_dump(
                        mode="json"
                    )
                ]
            }
            response.text = json.dumps(data)
            response.json = lambda: data
            response.ok = True
            return response
        if args[0] == f"{root_url}/api/v1/about":
            response = Mock(spec=Response)
            response.status_code = HTTPStatus.OK
            data = {"software_versions": {"iqm-station-control-client": version("iqm-station-control-client")}}
            response.text = json.dumps(data)
            response.json = lambda: data
            response.ok = True
            return response
        if args[0] == f"{root_url}/api/v1/quantum-computers/default/artifacts/duts":
            response = Mock(spec=Response)
            response.status_code = HTTPStatus.OK
            response.text = json.dumps([{"label": "M000_fake_0_0", "dut_type": "chip"}])
            return response
        if args[0].startswith(f"{root_url}/api/v1/jobs/"):
            response = Mock(spec=Response)
            response.ok = False
            response.status_code = HTTPStatus.INTERNAL_SERVER_ERROR
            return response
        return HTTPResponse(404)

    def mocked_requests_post(*args, **kwargs):
        if args[0] == f"{root_url}/api/v1/jobs/default/sweep":
            response = Mock(spec=Response)
            response.status_code = HTTPStatus.OK
            response.json = lambda: {
                "job_id": "c2d31ae9-e749-4835-8450-0df10be5d1c1",
                "job_href": f"{root_url}/api/v1/jobs/c2d31ae9-e749-4835-8450-0df10be5d1c1",
            }
            return response
        return HTTPResponse(404)

    monkeypatch.setattr(requests, "get", mocked_requests_get)
    monkeypatch.setattr(requests, "post", mocked_requests_post)

    with open(RESOURCES / "spark_settings.json", "r", encoding="utf-8") as file:
        settings = SettingNode(**json.loads(file.read()))
    monkeypatch.setattr(_IQMServerClient, "get_settings", lambda self: settings)

    with open(RESOURCES / "spark_chip_design_record.json", "r", encoding="utf-8") as file:
        design_record_str = file.read()
        record = json.loads(design_record_str)
    monkeypatch.setattr(_IQMServerClient, "get_chip_design_records", lambda self: [record])

    with open(RESOURCES / "spark_calibration_set_raw.json", "r", encoding="utf-8") as file:
        cal = json.loads(file.read()), "fbaa6256-ab83-4217-8b7b-07c1952ec236"
    monkeypatch.setattr(CalibrationDataProvider, "get_default_calibration_set", lambda self: cal)
    monkeypatch.setattr(CalibrationDataProvider, "get_calibration_set_values", lambda self, calibration_set_id: cal[0])

    pulla = Pulla(iqm_server_url=root_url)
    return pulla


@pytest.fixture
def qiskit_backend_spark(monkeypatch, pulla_on_spark) -> IQMBackend:
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
    iqm_client = IQMClient(f"{root_url}", client_signature="test fixture")
    return IQMBackend(iqm_client, calibration_set_id=calset_id, use_metrics=False)


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
