# Copyright 2021-2022 IQM client developers
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

"""
Mocks server calls for testing
"""

from datetime import datetime
import json
from math import pi
import os
import platform
from typing import Any
from uuid import UUID

from iqm.iqm_client import DIST_NAME, CircuitCompilationOptions, IQMClient, __version__
from iqm.iqm_server_client.iqm_server_client import REQUESTS_TIMEOUT
from iqm.iqm_server_client.models import CalibrationSet, QualityMetricSet
from iqm.models.channel_properties import AWGProperties, ChannelProperties, ReadoutProperties
from mockito import expect
import pytest
import requests

from iqm.pulse import Circuit
from iqm.station_control.client.authentication import TokenManager
from iqm.station_control.interface.list_with_meta import ListWithMeta, Meta
from iqm.station_control.interface.models import (
    DDMode,
    DDStrategy,
    DutData,
    DynamicQuantumArchitecture,
    GateImplementationInfo,
    GateInfo,
    HeraldingMode,
    MoveGateFrameTrackingMode,
    MoveGateValidationMode,
    ObservationLite,
    ObservationSetData,
    ObservationSetType,
    QualityMetrics,
    RunRequest,
    StaticQuantumArchitecture,
)

from ..conftest import MockJsonResponse, create_sample_circuit

# Delete any env variables that were set during e.g. cocos e2e testing to prevent ClientConfigurationErrors
os.environ.pop("IQM_TOKEN", None)


@pytest.fixture()
def client_signature() -> str:
    return "some-signature"


@pytest.fixture()
def sample_calset_id_2() -> UUID:
    return UUID("3902d525-d8f4-42c0-9fa9-6bbd535b6c80")


@pytest.fixture()
def existing_job_id() -> UUID:
    return UUID("0cf95701-ee23-4aca-8332-90f1a1fe6808")


@pytest.fixture()
def iqm_client_mock_with_signature(base_url) -> IQMClient:
    client = IQMClient(base_url, client_signature="some-signature")
    client._token_manager = TokenManager()  # Do not use authentication
    return client


@pytest.fixture()
def jobs_url(base_url) -> str:
    return f"{base_url}/api/v1/jobs"


@pytest.fixture()
def mock_circuit_job_internals_for_client(
    existing_job_id,
    run_request_with_dd,
):
    """
    Factory fixture to mock internal GET calls of CircuitJob.result() and CircuitJob.payload().
    """

    def _setup_mocks(client_obj: IQMClient):
        expect(client_obj._iqm_server_client, times=1).get_job_artifact_measurements(existing_job_id).thenReturn(
            [{"result": [[1, 0, 1, 1], [1, 0, 0, 1], [1, 0, 1, 1], [1, 0, 1, 1]]}]
        )
        expect(client_obj._iqm_server_client, times=1).get_submit_circuits_payload(existing_job_id).thenReturn(
            run_request_with_dd
        )

    return _setup_mocks


# Helper fixture for access token (replace with actual token logic if needed)
@pytest.fixture
def access_token():
    return "dummy-access-token"


@pytest.fixture()
def existing_job_url(jobs_url, existing_job_id) -> str:
    return f"{jobs_url}/{existing_job_id}"


@pytest.fixture()
def sample_circuit_metadata():
    return {"experiment_type": "test", "qubits": (0, 1), "values": [0.01686514, 0.05760602]}


@pytest.fixture
def sample_circuit(sample_circuit_metadata):
    """
    A sample circuit for testing submit_circuit
    """
    return create_sample_circuit(["QB1", "QB2"], metadata=sample_circuit_metadata)


@pytest.fixture
def sample_circuit_logical(sample_circuit_metadata):
    """
    A sample circuit with logical names for testing submit_circuit
    """
    return create_sample_circuit(["Qubit A", "Qubit B"], metadata=sample_circuit_metadata)


@pytest.fixture
def sample_circuit_with_raw_instructions(sample_circuit_metadata):
    """
    A sample circuit with instructions defined by dicts for testing if
    we do not break pydantic parsing logic with custom validators
    """
    return Circuit(
        name="The circuit",
        instructions=[
            {
                "name": "cz",
                "qubits": (
                    "Qubit A",
                    "Qubit B",
                ),
                "args": {},
            },
            {
                "name": "prx",
                "implementation": "drag_gaussian",
                "qubits": ("Qubit A",),
                "args": {"phase": 1.4 * pi, "angle": 0.5 * pi},
            },
            {
                "name": "prx",
                "qubits": ("Qubit A",),
                "args": {"phase": 0.6 * pi, "angle": -0.4 * pi},
            },
            {"name": "measure", "qubits": ("Qubit A",), "args": {"key": "A"}},
            {"name": "measure", "qubits": ("Qubit B",), "args": {"key": "B"}},
        ],
        metadata=sample_circuit_metadata,
    )


@pytest.fixture()
def minimal_run_request(sample_circuit, sample_calset_id) -> RunRequest:
    return RunRequest(
        circuits=[sample_circuit],
        shots=10,
        calibration_set_id=sample_calset_id,
    )


@pytest.fixture()
def run_request_with_heralding(sample_circuit, sample_calset_id) -> RunRequest:
    return RunRequest(
        circuits=[sample_circuit],
        shots=10,
        heralding_mode=HeraldingMode.ZEROS,
        calibration_set_id=sample_calset_id,
    )


@pytest.fixture()
def run_request_with_dd(sample_circuit, sample_calset_id) -> RunRequest:
    return RunRequest(
        circuits=[sample_circuit],
        shots=10,
        dd_mode=DDMode.ENABLED,
        dd_strategy=DDStrategy(gate_sequences=[(9, "XYXYYXYX", "asap"), (5, "YXYX", "asap"), (2, "XX", "center")]),
        calibration_set_id=sample_calset_id,
    )


@pytest.fixture()
def run_request_with_move_validation(sample_circuit, sample_calset_id) -> RunRequest:
    return RunRequest(
        circuits=[sample_circuit],
        shots=10,
        move_gate_validation=MoveGateValidationMode.STRICT,
        calibration_set_id=sample_calset_id,
    )


@pytest.fixture()
def run_request_with_incompatible_options(sample_circuit, sample_calset_id) -> RunRequest:
    return RunRequest(
        circuits=[sample_circuit],
        shots=10,
        move_gate_validation=MoveGateValidationMode.NONE,
        move_gate_frame_tracking=MoveGateFrameTrackingMode.FULL,
        calibration_set_id=sample_calset_id,
    )


@pytest.fixture()
def run_request_without_prx_move_validation(sample_circuit, sample_calset_id) -> RunRequest:
    return RunRequest(
        circuits=[sample_circuit],
        shots=10,
        move_gate_validation=MoveGateValidationMode.ALLOW_PRX,
        calibration_set_id=sample_calset_id,
    )


@pytest.fixture()
def run_request_with_move_gate_frame_tracking(sample_circuit, sample_calset_id) -> RunRequest:
    return RunRequest(
        circuits=[sample_circuit],
        shots=10,
        move_gate_frame_tracking=MoveGateFrameTrackingMode.FULL,
        calibration_set_id=sample_calset_id,
    )


@pytest.fixture()
def run_request_without_qubit_mapping(sample_circuit, sample_calset_id) -> RunRequest:
    return RunRequest(
        circuits=[sample_circuit],
        shots=10,
        heralding_mode=HeraldingMode.NONE,
        calibration_set_id=sample_calset_id,
    )


@pytest.fixture()
def run_request_with_invalid_qubit_mapping(sample_circuit_logical, sample_calset_id) -> RunRequest:
    return RunRequest(
        circuits=[sample_circuit_logical],
        shots=10,
        qubit_mapping={
            "Qubit A": "QB1",
            "Qubit B": "QB1",
        },
        heralding_mode=HeraldingMode.NONE,
        calibration_set_id=sample_calset_id,
    )


@pytest.fixture()
def run_request_with_incomplete_qubit_mapping(sample_circuit_logical, sample_calset_id) -> RunRequest:
    return RunRequest(
        circuits=[sample_circuit_logical],
        shots=10,
        qubit_mapping={
            "Qubit A": "QB1",
        },
        heralding_mode=HeraldingMode.NONE,
        calibration_set_id=sample_calset_id,
    )


@pytest.fixture()
def run_request_with_calibration_set_id(sample_circuit_logical, sample_calset_id) -> RunRequest:
    return RunRequest(
        circuits=[sample_circuit_logical],
        shots=10,
        qubit_mapping={
            "Qubit A": "QB1",
            "Qubit B": "QB2",
        },
        calibration_set_id=sample_calset_id,
        heralding_mode=HeraldingMode.NONE,
    )


@pytest.fixture()
def run_request_with_duration_check_disabled(sample_circuit_logical, sample_calset_id) -> RunRequest:
    return RunRequest(
        circuits=[sample_circuit_logical],
        shots=10,
        qubit_mapping={
            "Qubit A": "QB1",
            "Qubit B": "QB2",
        },
        max_circuit_duration_over_t2=0.0,
        heralding_mode=HeraldingMode.NONE,
        calibration_set_id=sample_calset_id,
    )


@pytest.fixture
def sample_quality_metrics_observations() -> list[ObservationLite]:
    return [
        ObservationLite(
            observation_id=123456,
            dut_field="QB1.t1_time",
            unit="s",
            value=4.408139707188389e-05,
            uncertainty=2.83049498694448e-06,
            invalid=False,
            created_timestamp=datetime.fromisoformat("2023-02-10T08:57:04.605956+00:00"),
            modified_timestamp=datetime.fromisoformat("2023-02-10T08:57:04.605956+00:00"),
        ),
        ObservationLite(
            observation_id=234567,
            dut_field="QB1.t2_time",
            unit="s",
            value=3.245501974471748e-05,
            uncertainty=2.39049697699448e-06,
            invalid=False,
            created_timestamp=datetime.fromisoformat("2023-02-10T08:57:04.605956+00:00"),
            modified_timestamp=datetime.fromisoformat("2023-02-10T08:57:04.605956+00:00"),
        ),
    ]


@pytest.fixture
def sample_quality_metrics_sc(sample_calibration_set_sc, sample_quality_metrics_observations) -> QualityMetrics:
    return QualityMetrics(
        observation_set_type=ObservationSetType.QUALITY_METRIC_SET,
        describes_id=sample_calibration_set_sc.observation_set_id,
        invalid=False,
        dut_label="M194_W0_P08_Z99",
        observation_set_id=UUID("3fbcdb11-6e92-43e9-85a8-cbca94fbec92"),
        created_timestamp=datetime.fromisoformat("2023-02-10T08:57:04.605956+00:00"),
        end_timestamp=datetime.fromisoformat("2023-02-10T08:57:04.605956+00:00"),
        calibration_set=sample_calibration_set_sc,
        observations=sample_quality_metrics_observations,
    )


@pytest.fixture
def sample_calibration_set_sc(sample_calset_id, sample_observation_set_sc) -> ObservationSetData:
    return ObservationSetData(
        observation_set_type=ObservationSetType.CALIBRATION_SET,
        observation_ids=[123456, 234567],
        describes_id=None,
        invalid=False,
        dut_label="M194_W0_P08_Z99",
        observation_set_id=sample_calset_id,
        created_timestamp=datetime.fromisoformat("2023-02-10T08:57:04.605956+00:00"),
        end_timestamp=datetime.fromisoformat("2023-02-10T08:57:04.605956+00:00"),
    )


@pytest.fixture
def crystal_5_calibration_set_observations():
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
        "iqm_client/resources/crystal_5_calibration_set_observations.json",
    )
    with open(path, "r", encoding="utf-8") as data:
        calibration_set_data = json.load(data)

    # Convert to list of ObservationLite objects
    observations = []
    for item in calibration_set_data:
        observation = ObservationLite(
            observation_id=item["observation_id"],
            dut_field=item["dut_field"],
            unit=item["unit"],
            value=item["value"],
            uncertainty=item["uncertainty"],
            invalid=item["invalid"],
            created_timestamp=datetime.fromisoformat(item["created_timestamp"]),
            modified_timestamp=datetime.fromisoformat(item["modified_timestamp"]),
        )
        observations.append(observation)

    return observations


@pytest.fixture
def sample_parsed_metrics():
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
        "iqm_client/resources/crystal_5_parsed_metrics.json",
    )
    with open(path, "r", encoding="utf-8") as data:
        return json.load(data)


@pytest.fixture
def sample_crystal_5_dynamic_architecture() -> DynamicQuantumArchitecture:
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
        "iqm_client/resources/crystal_5_dynamic_quantum_architecture.json",
    )
    with open(path, "r", encoding="utf-8") as dqa:
        return DynamicQuantumArchitecture(**json.load(dqa))


@pytest.fixture
def sample_move_architecture(sample_calset_id) -> DynamicQuantumArchitecture:
    return DynamicQuantumArchitecture(
        calibration_set_id=sample_calset_id,
        qubits=["QB1", "QB2", "QB3"],
        computational_resonators=["CR1", "CR2"],
        gates={
            "prx": GateInfo(
                implementations={"drag_gaussian": GateImplementationInfo(loci=(("QB1",), ("QB2",), ("QB3",)))},
                default_implementation="drag_gaussian",
                override_default_implementation={},
            ),
            "cz": GateInfo(
                implementations={"tgss": GateImplementationInfo(loci=(("QB1", "CR1"), ("QB2", "CR1")))},
                default_implementation="tgss",
                override_default_implementation={},
            ),
            "move": GateInfo(
                implementations={"tgss_crf": GateImplementationInfo(loci=(("QB3", "CR1"),))},
                default_implementation="tgss_crf",
                override_default_implementation={},
            ),
            "measure": GateInfo(
                implementations={"constant": GateImplementationInfo(loci=(("QB1",), ("QB2",), ("QB3",)))},
                default_implementation="constant",
                override_default_implementation={},
            ),
        },
    )


@pytest.fixture
def sample_move_architecture_2() -> DynamicQuantumArchitecture:
    return DynamicQuantumArchitecture(
        calibration_set_id=sample_calset_id_2,
        qubits=["QB1", "QB2"],
        computational_resonators=[],
        gates={
            "prx": GateInfo(
                implementations={
                    "drag_gaussian": {
                        "loci": [["QB1"], ["QB2"]],
                    }
                },
                default_implementation="drag_gaussian",
                override_default_implementation={},
            ),
        },
    )


@pytest.fixture
def hybrid_move_architecture() -> DynamicQuantumArchitecture:
    """Contains both q-r and q-q gate loci.

       QB1
        |
       CR1
      *  |*
    QB2   QB3 - QB4
      *  |*
       CR2
        |*
       QB5

    Here, | signifies a CZ connection and * a MOVE connection.
    """
    return DynamicQuantumArchitecture(
        calibration_set_id=UUID("26c5e70f-bea0-43af-bd37-6212ec7d04cb"),
        qubits=["QB1", "QB2", "QB3", "QB4", "QB5"],
        computational_resonators=["CR1", "CR2"],
        gates={
            "prx": GateInfo(
                implementations={
                    "drag_gaussian": GateImplementationInfo(loci=(("QB1",), ("QB2",), ("QB3",), ("QB4",), ("QB5",))),
                },
                default_implementation="drag_gaussian",
                override_default_implementation={},
            ),
            "cz": GateInfo(
                implementations={
                    "tgss": GateImplementationInfo(
                        loci=(
                            ("QB1", "CR1"),
                            ("QB3", "CR1"),
                            ("QB3", "QB4"),
                            ("QB3", "CR2"),
                            ("QB5", "CR2"),
                        )
                    ),
                },
                default_implementation="tgss",
                override_default_implementation={},
            ),
            "move": GateInfo(
                implementations={
                    "tgss_crf": GateImplementationInfo(
                        loci=(
                            ("QB2", "CR1"),
                            ("QB2", "CR2"),
                            ("QB3", "CR1"),
                            ("QB3", "CR2"),
                            ("QB5", "CR2"),
                        ),
                    )
                },
                default_implementation="tgss_crf",
                override_default_implementation={},
            ),
            "measure": GateInfo(
                implementations={
                    "constant": GateImplementationInfo(
                        loci=(("QB1",), ("QB2",), ("QB3",), ("QB4",), ("QB5",)),
                    )
                },
                default_implementation="constant",
                override_default_implementation={},
            ),
        },
    )


class MockTextResponse:
    def __init__(self, status_code: int, text: str, history: list[requests.Response] | None = None):
        self.status_code = status_code
        self.text = text
        self.history = history

    def json(self):
        return json.loads(self.text)

    def raise_for_status(self):
        if 400 <= self.status_code < 600:
            raise requests.HTTPError("")


@pytest.fixture()
def sample_dut_label() -> list[DutData]:
    return [DutData(label="M194_W0_P08_Z99", dut_type="chip")]


@pytest.fixture()
def sample_static_architecture() -> StaticQuantumArchitecture:
    return StaticQuantumArchitecture(
        dut_label="M138_W0_XXX_Z99",
        qubits=["QB1", "QB2", "QB3"],
        computational_resonators=["CR1", "CR2"],
        connectivity=[("QB1", "CR1"), ("QB2", "CR2"), ("QB1", "QB3"), ("QB3", "CR1"), ("QB3", "CR2")],
    )


@pytest.fixture()
def submit_failed_auth() -> MockJsonResponse:
    return MockJsonResponse(401, {"detail": "unauthorized"})


@pytest.fixture()
def static_architecture_success(sample_static_architecture) -> MockJsonResponse:
    return MockJsonResponse(200, sample_static_architecture.model_dump())


@pytest.fixture()
def dynamic_architecture_success(sample_dynamic_architecture) -> MockJsonResponse:
    return MockJsonResponse(200, sample_dynamic_architecture.model_dump())


@pytest.fixture
def sample_channel_properties() -> dict[str, ChannelProperties]:
    return {
        "QB1__flux.awg": AWGProperties(
            sampling_rate=2.4e9,
            instruction_duration_granularity=16,
            instruction_duration_min=32,
            is_virtual=False,
            blocks_component=True,
            fast_feedback_sources=[],
        ),
        "QB1__drive.awg": AWGProperties(
            sampling_rate=2.4e9,
            instruction_duration_granularity=16,
            instruction_duration_min=32,
            is_virtual=False,
            blocks_component=True,
            fast_feedback_sources=["PL-1__readout"],
        ),
        "QB2__drive.awg": AWGProperties(
            sampling_rate=2.4e9,
            instruction_duration_granularity=16,
            instruction_duration_min=32,
            is_virtual=False,
            blocks_component=True,
            fast_feedback_sources=["PL-1__readout"],
        ),
        "QB3__drive.awg": AWGProperties(
            sampling_rate=2.4e9,
            instruction_duration_granularity=16,
            instruction_duration_min=32,
            is_virtual=False,
            blocks_component=True,
            fast_feedback_sources=["PL-2__readout"],
        ),
        "PL-1__readout": ReadoutProperties(
            sampling_rate=2.4e9,
            instruction_duration_granularity=16,
            instruction_duration_min=16,
            integration_start_dead_time=2e-16,
            integration_stop_dead_time=4e-16,
        ),
        "PL-2__readout": ReadoutProperties(
            sampling_rate=2.4e9,
            instruction_duration_granularity=16,
            instruction_duration_min=16,
            integration_start_dead_time=2e-16,
            integration_stop_dead_time=4e-16,
        ),
    }


def get_jobs_args(
    user_agent: str | None = None,
    access_token: str | None = None,
) -> dict[str, Any]:
    """Returns expected kwargs of POST /jobs request"""
    headers = {}
    signature = f"{platform.platform(terse=True)}, python {platform.python_version()}, {DIST_NAME} {__version__}"
    headers["User-Agent"] = signature if user_agent is None else user_agent
    if access_token is not None:
        headers["Authorization"] = f"Bearer {access_token}"
    return {
        "headers": headers if headers else None,
        "timeout": REQUESTS_TIMEOUT,
    }


def submit_circuits_args(run_request: RunRequest) -> dict[str, Any]:
    """Return args to be used with submit_circuits to generate the expected RunRequest"""
    return {
        "circuits": run_request.circuits,
        "qubit_mapping": run_request.qubit_mapping,
        "calibration_set_id": run_request.calibration_set_id,
        "shots": run_request.shots,
        "options": CircuitCompilationOptions(
            max_circuit_duration_over_t2=run_request.max_circuit_duration_over_t2,
            heralding_mode=run_request.heralding_mode,
            move_gate_validation=run_request.move_gate_validation,
            move_gate_frame_tracking=run_request.move_gate_frame_tracking,
            dd_mode=run_request.dd_mode,
            dd_strategy=run_request.dd_strategy,
        ),
    }


@pytest.fixture
def mock_default_dynamic_architecture_retrieval(
    iqm_client_mock,
    sample_dynamic_architecture,  # iqm_client.models.DynamicQuantumArchitecture
):
    """
    Sets up mocks for IQMClient.get_dynamic_quantum_architecture()
    when it retrieves the architecture for the default calibration set.
    """
    expect(iqm_client_mock._iqm_server_client, times=1).get_dynamic_quantum_architecture("default").thenReturn(
        sample_dynamic_architecture
    )


@pytest.fixture
def mock_dynamic_architecture_retrieval(
    iqm_client_mock,
    sample_calset_id,
    sample_dynamic_architecture,  # iqm_client.models.DynamicQuantumArchitecture
):
    """
    Sets up mocks for IQMClient.get_dynamic_quantum_architecture()
    when it retrieves the architecture for the specified calibration set id.
    """
    expect(iqm_client_mock._iqm_server_client, times=1).get_dynamic_quantum_architecture(sample_calset_id).thenReturn(
        sample_dynamic_architecture
    )


@pytest.fixture
def mock_calset_retrieval(
    iqm_client_mock,
    sample_calset_id,
    sample_calibration_set_iqm_server,
):
    """
    Sets up mocks for IQMClient.get_calibration_set(calset_id)
    when it retrieves the calibration set for the specified calibration set id.
    """
    expect(iqm_client_mock._iqm_server_client, times=1).get_calibration_set(sample_calset_id).thenReturn(
        sample_calibration_set_iqm_server
    )


@pytest.fixture
def mock_default_calset_retrieval(
    iqm_client_mock,
    sample_calibration_set_iqm_server,
):
    """
    Sets up mocks for IQMClient.get_calibration_set()
    when it retrieves the default calibration set.
    """
    expect(iqm_client_mock._iqm_server_client, times=1).get_calibration_set("default").thenReturn(
        sample_calibration_set_iqm_server
    )


@pytest.fixture
def mock_static_architecture_retrieval(
    iqm_client_mock,
    sample_dut_label,  # station_control.interface.models.DUTLabel,
    sample_static_architecture,  # iqm_client.models.StaticQuantumArchitecture
):
    """
    Sets up mocks for IQMClient.get_static_quantum_architecture(dut_label)
    when it retrieves the architecture for the specified dut label.
    """
    expect(iqm_client_mock._iqm_server_client, times=1).get_duts().thenReturn(sample_dut_label)
    expect(iqm_client_mock._iqm_server_client, times=1).get_static_quantum_architectures().thenReturn(
        [sample_static_architecture]
    )


@pytest.fixture
def sample_observation_set_query(sample_dut_label, sample_calset_id):
    return ListWithMeta[ObservationSetData](
        items=[
            ObservationSetData(
                dut_label=sample_dut_label[0].label,
                describes_id=None,
                observation_set_type=ObservationSetType.CALIBRATION_SET,
                observation_ids=[123456, 234567],
                observation_set_id=sample_calset_id,
                created_timestamp=datetime.fromisoformat("2023-02-10T08:57:04.605956+00:00"),
                end_timestamp=datetime.fromisoformat("2023-02-10T08:57:04.605956+00:00"),
                invalid=False,
            ),
        ],
        meta=Meta(count=1663, order_by="-created_timestamp", limit=20, offset=0, errors=[]),
    )


@pytest.fixture
def sample_quality_metric_set_iqm_server(
    sample_dut_label, sample_calset_id, sample_quality_metrics_observations
) -> QualityMetricSet:
    return QualityMetricSet(
        observation_set_type=ObservationSetType.QUALITY_METRIC_SET,
        describes_id=sample_calset_id,
        invalid=False,
        dut_label="M194_W0_P08_Z99",
        observation_set_id=UUID("3fbcdb11-6e92-43e9-85a8-cbca94fbec92"),
        created_timestamp=datetime.fromisoformat("2023-02-10T08:57:04.605956+00:00"),
        end_timestamp=datetime.fromisoformat("2023-02-10T08:57:04.605956+00:00"),
        observations=sample_quality_metrics_observations,
    )


@pytest.fixture
def sample_calibration_set_iqm_server(sample_calset_id, sample_observation_set_sc) -> CalibrationSet:
    return CalibrationSet(
        dut_label="M194_W0_P08_Z99",
        observation_set_type=ObservationSetType.CALIBRATION_SET,
        observation_set_id=sample_calset_id,
        created_timestamp=datetime.fromisoformat("2023-02-10T08:57:04.605956+00:00"),
        end_timestamp=datetime.fromisoformat("2023-02-10T08:57:04.605956+00:00"),
        observations=sample_observation_set_sc,
    )


@pytest.fixture
def sample_calibration_set_iqm_server_2(sample_calset_id_2, sample_observation_set_sc) -> CalibrationSet:
    return CalibrationSet(
        dut_label="M194_W0_P08_Z99",
        observation_set_type=ObservationSetType.CALIBRATION_SET,
        observation_set_id=sample_calset_id_2,
        created_timestamp=datetime.fromisoformat("2023-02-10T08:57:04.605956+00:00"),
        end_timestamp=datetime.fromisoformat("2023-02-10T08:57:04.605956+00:00"),
        observations=sample_observation_set_sc,
    )


@pytest.fixture
def sample_observation_set_sc() -> list[ObservationLite]:
    return [
        ObservationLite(
            observation_id=123456,
            dut_field="QB4.flux.voltage",
            unit="V",
            value=-0.158,
            uncertainty=None,
            invalid=False,
            created_timestamp=datetime.fromisoformat("2023-02-10T08:57:04.605956+00:00"),
            modified_timestamp=datetime.fromisoformat("2023-02-10T08:57:04.605956+00:00"),
        ),
        ObservationLite(
            observation_id=234567,
            dut_field="PL-1.readout.center_frequency",
            unit="Hz",
            value=5.5e9,
            uncertainty=None,
            invalid=False,
            created_timestamp=datetime.fromisoformat("2023-02-10T08:57:04.605956+00:00"),
            modified_timestamp=datetime.fromisoformat("2023-02-10T08:57:04.605956+00:00"),
        ),
    ]


@pytest.fixture
def sample_dynamic_architecture_2(sample_calset_id_2) -> DynamicQuantumArchitecture:
    return DynamicQuantumArchitecture(
        calibration_set_id=sample_calset_id_2,
        qubits=["QB1", "QB2"],
        computational_resonators=[],
        gates={
            "prx": GateInfo(
                implementations={
                    "drag_gaussian": {
                        "loci": [["QB1"], ["QB2"]],
                    }
                },
                default_implementation="drag_gaussian",
                override_default_implementation={},
            ),
        },
    )
