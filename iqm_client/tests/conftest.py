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

from base64 import b64encode
from importlib.metadata import version
import json
import os
import platform
import time
from typing import Any, Optional
from uuid import UUID

from mockito import ANY, when
from packaging.version import parse
import pytest
import requests
from requests import HTTPError, Response

from iqm.iqm_client import (
    DIST_NAME,
    REQUESTS_TIMEOUT,
    Circuit,
    CircuitCompilationOptions,
    DDMode,
    DDStrategy,
    DynamicQuantumArchitecture,
    GateImplementationInfo,
    GateInfo,
    HeraldingMode,
    Instruction,
    IQMClient,
    MoveGateFrameTrackingMode,
    MoveGateValidationMode,
    RunRequest,
    SingleQubitMapping,
    __version__,
)


@pytest.fixture()
def base_url() -> str:
    return 'https://example.com'


@pytest.fixture()
def client_signature() -> str:
    return 'some-signature'


@pytest.fixture()
def sample_calibration_set_id() -> UUID:
    return UUID('9ddb9586-8f27-49a9-90ed-41086b47f6bd')


@pytest.fixture()
def existing_run_id() -> UUID:
    return UUID('3c3fcda3-e860-46bf-92a4-bcc59fa76ce9')


@pytest.fixture()
def missing_run_id() -> UUID:
    return UUID('059e4186-50a3-4e6c-ba1f-37fe6afbdfc2')


@pytest.fixture()
def sample_client(base_url) -> IQMClient:
    when(requests).get(f'{base_url}/info/client-libraries', headers=ANY, timeout=ANY).thenReturn(
        mock_supported_client_libraries_response()
    )
    client = IQMClient(url=base_url)
    client._token_manager = None  # Do not use authentication
    return client


@pytest.fixture()
def client_with_signature(base_url) -> IQMClient:
    when(requests).get(f'{base_url}/info/client-libraries', headers=ANY, timeout=ANY).thenReturn(
        mock_supported_client_libraries_response()
    )
    client = IQMClient(url=base_url, client_signature='some-signature')
    client._token_manager = None  # Do not use authentication
    return client


@pytest.fixture()
def jobs_url(base_url) -> str:
    return f'{base_url}/jobs'


@pytest.fixture()
def existing_job_url(jobs_url, existing_run_id) -> str:
    return f'{jobs_url}/{existing_run_id}'


@pytest.fixture()
def existing_job_status_url(existing_job_url) -> str:
    return f'{existing_job_url}/status'


@pytest.fixture()
def quantum_architecture_url(base_url) -> str:
    return f'{base_url}/quantum-architecture'


@pytest.fixture()
def dynamic_architecture_url(base_url) -> str:
    return f'{base_url}/api/v1/calibration/default/gates'


@pytest.fixture()
def channel_properties_url(base_url) -> str:
    # Only in API V2
    return f'{base_url}/station/channel-properties'


@pytest.fixture
def settings_dict():
    """
    Reads and parses settings file into a dictionary
    """
    settings_path = os.path.dirname(os.path.realpath(__file__)) + '/resources/settings.json'
    with open(settings_path, 'r', encoding='utf-8') as f:
        return json.loads(f.read())


@pytest.fixture()
def sample_circuit_metadata():
    return {'experiment_type': 'test', 'qubits': (0, 1), 'values': [0.01686514, 0.05760602]}


@pytest.fixture
def sample_circuit(sample_circuit_metadata):
    """
    A sample circuit for testing submit_circuit
    """
    return create_sample_circuit(['QB1', 'QB2'], metadata=sample_circuit_metadata)


@pytest.fixture
def sample_circuit_logical(sample_circuit_metadata):
    """
    A sample circuit with logical names for testing submit_circuit
    """
    return create_sample_circuit(['Qubit A', 'Qubit B'], metadata=sample_circuit_metadata)


def create_sample_circuit(qubits: list[str], metadata) -> Circuit:
    return Circuit(
        name='The circuit',
        instructions=[
            Instruction(
                name='cz',
                qubits=tuple(qubits),
                args={},
            ),
            Instruction(
                name='prx',
                implementation='drag_gaussian',
                qubits=(qubits[0],),
                args={'phase_t': 0.7, 'angle_t': 0.25},
            ),
            Instruction(
                name='prx',
                qubits=(qubits[0],),
                args={'phase_t': 0.3, 'angle_t': -0.2},
            ),
            Instruction(name='measure', qubits=(qubits[0],), args={'key': 'A'}),
            Instruction(name='measure', qubits=(qubits[1],), args={'key': 'B'}),
        ],
        metadata=metadata,
    )


@pytest.fixture
def sample_circuit_with_raw_instructions(sample_circuit_metadata):
    """
    A sample circuit with instructions defined by dicts for testing if
    we do not break pydantic parsing logic with custom validators
    """
    return Circuit(
        name='The circuit',
        instructions=[
            {
                'name': 'cz',
                'qubits': (
                    'Qubit A',
                    'Qubit B',
                ),
                'args': {},
            },
            {
                'name': 'prx',
                'implementation': 'drag_gaussian',
                'qubits': ('Qubit A',),
                'args': {'phase_t': 0.7, 'angle_t': 0.25},
            },
            {
                'name': 'prx',
                'qubits': ('Qubit A',),
                'args': {'phase_t': 0.3, 'angle_t': -0.2},
            },
            {'name': 'measure', 'qubits': ('Qubit A',), 'args': {'key': 'A'}},
            {'name': 'measure', 'qubits': ('Qubit B',), 'args': {'key': 'B'}},
        ],
        metadata=sample_circuit_metadata,
    )


@pytest.fixture()
def minimal_run_request(sample_circuit) -> RunRequest:
    return RunRequest(
        circuits=[sample_circuit],
        shots=10,
    )


@pytest.fixture()
def run_request_with_heralding(sample_circuit) -> RunRequest:
    return RunRequest(
        circuits=[sample_circuit],
        shots=10,
        heralding_mode=HeraldingMode.ZEROS,
    )


@pytest.fixture()
def run_request_with_dd(sample_circuit) -> RunRequest:
    return RunRequest(
        circuits=[sample_circuit],
        shots=10,
        dd_mode=DDMode.ENABLED,
        dd_strategy=DDStrategy(gate_sequences=[(9, 'XYXYYXYX', 'asap'), (5, 'YXYX', 'asap'), (2, 'XX', 'center')]),
    )


@pytest.fixture()
def run_request_with_move_validation(sample_circuit) -> RunRequest:
    return RunRequest(
        circuits=[sample_circuit],
        shots=10,
        move_validation_mode=MoveGateValidationMode.STRICT,
    )


@pytest.fixture()
def run_request_with_incompatible_options(sample_circuit) -> RunRequest:
    return RunRequest(
        circuits=[sample_circuit],
        shots=10,
        move_validation_mode=MoveGateValidationMode.NONE,
        move_gate_frame_tracking_mode=MoveGateFrameTrackingMode.FULL,
    )


@pytest.fixture()
def run_request_without_prx_move_validation(sample_circuit) -> RunRequest:
    return RunRequest(
        circuits=[sample_circuit],
        shots=10,
        move_validation_mode=MoveGateValidationMode.ALLOW_PRX,
    )


@pytest.fixture()
def run_request_with_move_gate_frame_tracking(sample_circuit) -> RunRequest:
    return RunRequest(
        circuits=[sample_circuit],
        shots=10,
        move_gate_frame_tracking_mode=MoveGateFrameTrackingMode.FULL,
    )


@pytest.fixture()
def run_request_with_custom_settings(sample_circuit_logical, settings_dict) -> RunRequest:
    return RunRequest(
        circuits=[sample_circuit_logical],
        shots=10,
        qubit_mapping=[
            SingleQubitMapping(logical_name='Qubit A', physical_name='QB1'),
            SingleQubitMapping(logical_name='Qubit B', physical_name='QB2'),
        ],
        custom_settings=settings_dict,
        heralding_mode=HeraldingMode.NONE,
    )


@pytest.fixture()
def run_request_without_qubit_mapping(sample_circuit) -> RunRequest:
    return RunRequest(
        circuits=[sample_circuit],
        shots=10,
        heralding_mode=HeraldingMode.NONE,
    )


@pytest.fixture()
def run_request_with_invalid_qubit_mapping(sample_circuit_logical) -> RunRequest:
    return RunRequest(
        circuits=[sample_circuit_logical],
        shots=10,
        qubit_mapping=[
            SingleQubitMapping(logical_name='Qubit A', physical_name='QB1'),
            SingleQubitMapping(logical_name='Qubit B', physical_name='QB1'),
        ],
        heralding_mode=HeraldingMode.NONE,
    )


@pytest.fixture()
def run_request_with_incomplete_qubit_mapping(sample_circuit_logical) -> RunRequest:
    return RunRequest(
        circuits=[sample_circuit_logical],
        shots=10,
        qubit_mapping=[
            SingleQubitMapping(logical_name='Qubit A', physical_name='QB1'),
        ],
        heralding_mode=HeraldingMode.NONE,
    )


@pytest.fixture()
def run_request_with_calibration_set_id(sample_circuit_logical, sample_calibration_set_id) -> RunRequest:
    return RunRequest(
        circuits=[sample_circuit_logical],
        shots=10,
        qubit_mapping=[
            SingleQubitMapping(logical_name='Qubit A', physical_name='QB1'),
            SingleQubitMapping(logical_name='Qubit B', physical_name='QB2'),
        ],
        calibration_set_id=sample_calibration_set_id,
        heralding_mode=HeraldingMode.NONE,
    )


@pytest.fixture()
def run_request_with_duration_check_disabled(sample_circuit_logical) -> RunRequest:
    return RunRequest(
        circuits=[sample_circuit_logical],
        shots=10,
        qubit_mapping=[
            SingleQubitMapping(logical_name='Qubit A', physical_name='QB1'),
            SingleQubitMapping(logical_name='Qubit B', physical_name='QB2'),
        ],
        max_circuit_duration_over_t2=0.0,
        heralding_mode=HeraldingMode.NONE,
    )


@pytest.fixture()
def pending_compilation_job_result(sample_circuit):
    return MockJsonResponse(
        200, {'status': 'pending compilation', 'metadata': {'request': {'shots': 10, 'circuits': [sample_circuit]}}}
    )


@pytest.fixture()
def pending_execution_job_result(sample_circuit):
    return MockJsonResponse(
        200, {'status': 'pending execution', 'metadata': {'request': {'shots': 10, 'circuits': [sample_circuit]}}}
    )


@pytest.fixture()
def pending_deletion_job_result(sample_circuit):
    return MockJsonResponse(
        200, {'status': 'pending deletion', 'metadata': {'request': {'shots': 10, 'circuits': [sample_circuit]}}}
    )


@pytest.fixture()
def deleted_job_result():
    return MockJsonResponse(200, {'status': 'deleted', 'metadata': {'request': {'shots': 1, 'circuits': []}}})


@pytest.fixture()
def ready_job_result(sample_circuit, sample_calibration_set_id):
    return MockJsonResponse(
        200,
        {
            'status': 'ready',
            'measurements': [{'result': [[1, 0, 1, 1], [1, 0, 0, 1], [1, 0, 1, 1], [1, 0, 1, 1]]}],
            'metadata': {
                'calibration_set_id': str(sample_calibration_set_id),
                'request': {
                    'shots': 42,
                    'circuits': [sample_circuit],
                    'calibration_set_id': str(sample_calibration_set_id),
                },
                'timestamps': {
                    'job_start': '0.0',
                    'job_end': '1.1',
                },
            },
        },
    )


@pytest.fixture()
def job_result_with_warnings(sample_circuit, sample_calibration_set_id):
    return MockJsonResponse(
        200,
        {
            'status': 'ready',
            'metadata': {
                'calibration_set_id': str(sample_calibration_set_id),
                'request': {
                    'shots': 42,
                    'circuits': [sample_circuit],
                    'calibration_set_id': str(sample_calibration_set_id),
                },
                'timestamps': {
                    'job_start': '0.0',
                    'job_end': '1.1',
                },
            },
            'warnings': ['This is a warning message'],
        },
    )


@pytest.fixture()
def pending_compilation_status():
    return MockJsonResponse(200, {'status': 'pending compilation'})


@pytest.fixture()
def pending_execution_status():
    return MockJsonResponse(200, {'status': 'pending execution'})


@pytest.fixture()
def ready_status():
    return MockJsonResponse(200, {'status': 'ready'})


@pytest.fixture()
def pending_deletion_status():
    return MockJsonResponse(200, {'status': 'pending deletion'})


@pytest.fixture()
def deleted_status():
    return MockJsonResponse(200, {'status': 'deleted'})


@pytest.fixture
def sample_static_architecture():
    return {
        'quantum_architecture': {
            'name': 'hercules',
            'qubits': ['QB1', 'QB2', 'QB3'],
            'qubit_connectivity': [['QB1', 'QB2']],
            'operations': {
                'prx': [['QB1'], ['QB2'], ['QB3']],
                'cz': [['QB1', 'QB2']],
                'measure': [['QB1'], ['QB2'], ['QB3']],
                'barrier': [],
            },
        }
    }


@pytest.fixture
def sample_dynamic_architecture() -> DynamicQuantumArchitecture:
    return DynamicQuantumArchitecture(
        calibration_set_id=UUID('26c5e70f-bea0-43af-bd37-6212ec7d04cb'),
        qubits=['QB1', 'QB2', 'QB3'],
        computational_resonators=[],
        gates={
            'prx': GateInfo(
                implementations={
                    'drag_gaussian': GateImplementationInfo(loci=(('QB1',), ('QB2',), ('QB3',))),
                    'drag_crf': GateImplementationInfo(loci=(('QB1',), ('QB3',))),
                },
                default_implementation='drag_gaussian',
                override_default_implementation={('QB3',): 'drag_crf'},
            ),
            'cc_prx': GateInfo(
                implementations={
                    'prx_composite': GateImplementationInfo(loci=(('QB1',), ('QB2',), ('QB3',))),
                },
                default_implementation='prx_composite',
                override_default_implementation={},
            ),
            'cz': GateInfo(
                implementations={
                    'tgss': GateImplementationInfo(loci=(('QB1', 'QB2'), ('QB1', 'QB3'))),
                    'crf': GateImplementationInfo(loci=(('QB1', 'QB2'),)),
                },
                default_implementation='tgss',
                override_default_implementation={},
            ),
            'measure': GateInfo(
                implementations={'constant': GateImplementationInfo(loci=(('QB1',), ('QB2',)))},
                default_implementation='constant',
                override_default_implementation={},
            ),
        },
    )


@pytest.fixture
def sample_move_architecture() -> DynamicQuantumArchitecture:
    return DynamicQuantumArchitecture(
        calibration_set_id=UUID('26c5e70f-bea0-43af-bd37-6212ec7d04cb'),
        qubits=['QB1', 'QB2', 'QB3'],
        computational_resonators=['CR1', 'CR2'],
        gates={
            'prx': GateInfo(
                implementations={'drag_gaussian': GateImplementationInfo(loci=(('QB1',), ('QB2',), ('QB3',)))},
                default_implementation='drag_gaussian',
                override_default_implementation={},
            ),
            'cz': GateInfo(
                implementations={'tgss': GateImplementationInfo(loci=(('QB1', 'CR1'), ('QB2', 'CR1')))},
                default_implementation='tgss',
                override_default_implementation={},
            ),
            'move': GateInfo(
                implementations={'tgss_crf': GateImplementationInfo(loci=(('QB3', 'CR1'),))},
                default_implementation='tgss_crf',
                override_default_implementation={},
            ),
            'measure': GateInfo(
                implementations={'constant': GateImplementationInfo(loci=(('QB1',), ('QB2',), ('QB3',)))},
                default_implementation='constant',
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
        calibration_set_id=UUID('26c5e70f-bea0-43af-bd37-6212ec7d04cb'),
        qubits=['QB1', 'QB2', 'QB3', 'QB4', 'QB5'],
        computational_resonators=['CR1', 'CR2'],
        gates={
            'prx': GateInfo(
                implementations={
                    'drag_gaussian': GateImplementationInfo(loci=(('QB1',), ('QB2',), ('QB3',), ('QB4',), ('QB5',))),
                },
                default_implementation='drag_gaussian',
                override_default_implementation={},
            ),
            'cz': GateInfo(
                implementations={
                    'tgss': GateImplementationInfo(
                        loci=(
                            ('QB1', 'CR1'),
                            ('QB3', 'CR1'),
                            ('QB3', 'QB4'),
                            ('QB3', 'CR2'),
                            ('QB5', 'CR2'),
                        )
                    ),
                },
                default_implementation='tgss',
                override_default_implementation={},
            ),
            'move': GateInfo(
                implementations={
                    'tgss_crf': GateImplementationInfo(
                        loci=(
                            ('QB2', 'CR1'),
                            ('QB2', 'CR2'),
                            ('QB3', 'CR1'),
                            ('QB3', 'CR2'),
                            ('QB5', 'CR2'),
                        ),
                    )
                },
                default_implementation='tgss_crf',
                override_default_implementation={},
            ),
            'measure': GateInfo(
                implementations={
                    'constant': GateImplementationInfo(
                        loci=(('QB1',), ('QB2',), ('QB3',), ('QB4',), ('QB5',)),
                    )
                },
                default_implementation='constant',
                override_default_implementation={},
            ),
        },
    )


class MockTextResponse:
    def __init__(self, status_code: int, text: str, history: Optional[list[Response]] = None):
        self.status_code = status_code
        self.text = text
        self.history = history

    def json(self):
        return json.loads(self.text)

    def raise_for_status(self):
        if 400 <= self.status_code < 600:
            raise HTTPError('')


@pytest.fixture()
def not_valid_json_response() -> MockTextResponse:
    return MockTextResponse(200, 'not a valid json')


class MockJsonResponse:
    def __init__(self, status_code: int, json_data: dict, history: Optional[list[Response]] = None):
        self.status_code = status_code
        self.json_data = json_data
        self.history = history
        self.url = 'https://example.com'

    @property
    def text(self):
        return json.dumps(self.json_data)

    def json(self):
        return self.json_data

    def raise_for_status(self):
        if 400 <= self.status_code < 600:
            raise HTTPError(f'{self.status_code}', response=self)


def mock_supported_client_libraries_response(
    iqm_client_name: str = 'iqm-client', max_version: Optional[str] = None, min_version: Optional[str] = None
) -> MockJsonResponse:
    client_version = parse(version('iqm-client'))
    min_version = f'{client_version.major}.0' if min_version is None else min_version
    max_version = f'{client_version.major + 1}.0' if max_version is None else max_version
    return MockJsonResponse(
        200,
        {
            iqm_client_name: {
                'name': iqm_client_name,
                'min': min_version,
                'max': max_version,
            }
        },
    )


@pytest.fixture()
def not_valid_client_configuration_response() -> MockJsonResponse:
    return MockJsonResponse(400, {'detail': 'not a valid client configuration'})


@pytest.fixture()
def submit_success(existing_run_id) -> MockJsonResponse:
    return MockJsonResponse(201, {'id': str(existing_run_id)})


@pytest.fixture()
def submit_failed_auth() -> MockJsonResponse:
    return MockJsonResponse(401, {'detail': 'unauthorized'})


@pytest.fixture()
def static_architecture_success(sample_static_architecture) -> MockJsonResponse:
    return MockJsonResponse(200, sample_static_architecture)


@pytest.fixture()
def dynamic_architecture_success(sample_dynamic_architecture) -> MockJsonResponse:
    return MockJsonResponse(200, sample_dynamic_architecture.model_dump())


@pytest.fixture()
def channel_properties_success() -> MockJsonResponse:
    content = {
        'QB1__flux.awg': {'fast_feedback_sources': []},
        'QB1__drive.awg': {'fast_feedback_sources': ['PL-1__readout']},
        'QB2__drive.awg': {'fast_feedback_sources': ['PL-1__readout']},
        'QB3__drive.awg': {'fast_feedback_sources': ['PL-2__readout']},
        'PL-1__readout': {},
        'PL-2__readout': {},
    }
    return MockJsonResponse(200, content)


@pytest.fixture()
def move_architecture_success(sample_move_architecture) -> MockJsonResponse:
    return MockJsonResponse(200, sample_move_architecture.model_dump())


@pytest.fixture()
def abort_job_success() -> MockJsonResponse:
    return MockJsonResponse(200, {})


@pytest.fixture()
def abort_job_failed() -> MockJsonResponse:
    return MockJsonResponse(400, {'detail': 'failed to abort job'})


def make_token(token_type: str, lifetime: int) -> str:
    """Encode given token type and expire time as a token.

    Args:
        token_type: 'Bearer' for access tokens, 'Refresh' for refresh tokens
        lifetime: seconds from current time to token's expire time

    Returns:
        Encoded token
    """
    empty = b64encode('{}'.encode('utf-8')).decode('utf-8')
    body = f'{{ "typ": "{token_type}", "exp": {int(time.time()) + lifetime} }}'
    body = b64encode(body.encode('utf-8')).decode('utf-8')
    return f'{empty}.{body}.{empty}'


def post_jobs_args(
    run_request: Optional[RunRequest] = None,
    user_agent: Optional[str] = None,
    access_token: Optional[str] = None,
) -> dict[str, Any]:
    """Returns expected kwargs of POST /jobs request"""
    headers = {'Expect': '100-Continue'} if run_request is not None else {}
    signature = f'{platform.platform(terse=True)}, python {platform.python_version()}, {DIST_NAME} {__version__}'
    headers['User-Agent'] = signature if user_agent is None else user_agent
    if access_token is not None:
        headers['Authorization'] = f'Bearer {access_token}'
    if run_request is None:
        return {'headers': headers, 'timeout': REQUESTS_TIMEOUT}
    return {
        'json': json.loads(run_request.model_dump_json(exclude_none=True)),
        'headers': headers,
        'timeout': REQUESTS_TIMEOUT,
    }


def get_jobs_args(
    user_agent: Optional[str] = None,
    access_token: Optional[str] = None,
) -> dict[str, Any]:
    """Returns expected kwargs of POST /jobs request"""
    headers = {}
    signature = f'{platform.platform(terse=True)}, python {platform.python_version()}, {DIST_NAME} {__version__}'
    headers['User-Agent'] = signature if user_agent is None else user_agent
    if access_token is not None:
        headers['Authorization'] = f'Bearer {access_token}'
    return {
        'headers': headers if headers else None,
        'timeout': REQUESTS_TIMEOUT,
    }


def submit_circuits_args(run_request: RunRequest) -> dict[str, Any]:
    """Return args to be used with submit_circuits to generate the expected RunRequest"""
    qm_dict = None
    if run_request.qubit_mapping is not None:
        qm_dict = {qm.logical_name: qm.physical_name for qm in run_request.qubit_mapping}
    return {
        'circuits': run_request.circuits,
        'qubit_mapping': qm_dict,
        'custom_settings': run_request.custom_settings,
        'calibration_set_id': run_request.calibration_set_id,
        'shots': run_request.shots,
        'options': CircuitCompilationOptions(
            max_circuit_duration_over_t2=run_request.max_circuit_duration_over_t2,
            heralding_mode=run_request.heralding_mode,
            move_gate_validation=run_request.move_validation_mode,
            move_gate_frame_tracking=run_request.move_gate_frame_tracking_mode,
            dd_mode=run_request.dd_mode,
            dd_strategy=run_request.dd_strategy,
        ),
    }
