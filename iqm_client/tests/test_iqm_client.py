# Copyright 2021-2023 IQM client developers
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
"""Tests for the IQM client."""
# pylint: disable=too-many-arguments,too-many-lines
from importlib.metadata import version
import re
import uuid

from mockito import ANY, expect, unstub, verifyNoUnwantedInteractions, when
from packaging.version import parse
import pytest
import requests
from requests import HTTPError

from iqm.iqm_client import (
    APIEndpoint,
    APIVariant,
    ArchitectureRetrievalError,
    Circuit,
    CircuitCompilationOptions,
    CircuitExecutionError,
    CircuitValidationError,
    ClientConfigurationError,
    Counts,
    DDMode,
    DDStrategy,
    DynamicQuantumArchitecture,
    HeraldingMode,
    Instruction,
    IQMClient,
    JobAbortionError,
    QuantumArchitectureSpecification,
    SingleQubitMapping,
    Status,
    serialize_qubit_mapping,
    validate_circuit,
)
from tests.conftest import (
    MockJsonResponse,
    get_jobs_args,
    mock_supported_client_libraries_response,
    post_jobs_args,
    submit_circuits_args,
)


@pytest.fixture
def move_circuit():
    instructions = (
        Instruction(
            name='prx',
            qubits=('QB1',),
            args={'phase_t': 0.3, 'angle_t': -0.2},
        ),
        Instruction(
            name='move',
            qubits=('QB3', 'CR1'),
            args={},
        ),
        Instruction(
            name='cz',
            qubits=('QB1', 'CR1'),
            args={},
        ),
        Instruction(
            name='cz',
            qubits=('QB2', 'CR1'),
            args={},
        ),
        Instruction(
            name='move',
            qubits=('QB3', 'CR1'),
            args={},
        ),
    )
    return Circuit(name='CR1 circuit', instructions=instructions)


@pytest.fixture
def move_circuit_with_prx_in_the_sandwich():
    instructions = (
        Instruction(
            name='prx',
            qubits=('QB1',),
            args={'phase_t': 0.3, 'angle_t': -0.2},
        ),
        Instruction(
            name='move',
            qubits=('QB3', 'CR1'),
            args={},
        ),
        Instruction(
            name='prx',
            qubits=('QB3',),
            args={'phase_t': 0.3, 'angle_t': -0.2},
        ),
        Instruction(
            name='move',
            qubits=('QB3', 'CR1'),
            args={},
        ),
    )
    return Circuit(name='CR1 circuit with PRX in the sandwich', instructions=instructions)


def test_serialize_qubit_mapping():
    """
    Tests that serialize_qubit_mapping returns a list of SingleQubitMapping objects
    """
    qm_dict = {'Alice': 'QB1', 'Bob': 'qubit_3', 'Charlie': 'physical 0'}
    assert serialize_qubit_mapping(qm_dict) == [
        SingleQubitMapping(logical_name='Alice', physical_name='QB1'),
        SingleQubitMapping(logical_name='Bob', physical_name='qubit_3'),
        SingleQubitMapping(logical_name='Charlie', physical_name='physical 0'),
    ]


def test_submit_circuits_adds_user_agent(
    sample_client, jobs_url, minimal_run_request, submit_success, dynamic_architecture_url, dynamic_architecture_success
):
    """
    Tests that submit_circuit without client signature adds correct User-Agent header
    """
    expect(requests, times=1).post(jobs_url, **post_jobs_args(minimal_run_request)).thenReturn(submit_success)
    expect(requests, times=1).get(dynamic_architecture_url, ...).thenReturn(dynamic_architecture_success)

    sample_client.submit_circuits(**submit_circuits_args(minimal_run_request))

    verifyNoUnwantedInteractions()
    unstub()


def test_submit_circuits_adds_user_agent_with_client_signature(
    client_with_signature,
    jobs_url,
    minimal_run_request,
    submit_success,
    dynamic_architecture_url,
    dynamic_architecture_success,
):
    """
    Tests that submit_circuit with client signature adds correct User-Agent header
    """
    expect(requests, times=1).post(
        jobs_url, **post_jobs_args(minimal_run_request, user_agent=client_with_signature._signature)
    ).thenReturn(submit_success)
    expect(requests, times=1).get(dynamic_architecture_url, ...).thenReturn(dynamic_architecture_success)

    client_with_signature.submit_circuits(**submit_circuits_args(minimal_run_request))

    verifyNoUnwantedInteractions()
    unstub()


@pytest.mark.parametrize(
    'run_request_name, valid_request, error',
    [
        ('minimal_run_request', True, None),
        ('run_request_with_heralding', True, None),
        ('run_request_with_custom_settings', True, None),
        (
            'run_request_with_invalid_qubit_mapping',
            False,
            CircuitValidationError('Multiple logical qubits map to the same physical qubit.'),
        ),
        (
            'run_request_with_incomplete_qubit_mapping',
            False,
            CircuitValidationError(
                "The qubits {'Qubit B'} in circuit 'The circuit' "
                'at index 0 are not found in the provided qubit mapping.'
            ),
        ),
        ('run_request_without_qubit_mapping', True, None),
        ('run_request_with_calibration_set_id', True, None),
        ('run_request_with_duration_check_disabled', True, None),
        (
            'run_request_with_incompatible_options',
            False,
            ValueError(
                'Unable to perform full MOVE gate frame tracking if MOVE gate validation '
                'is not "strict" or "allow_prx".'
            ),
        ),
    ],
)
def test_submit_circuits_returns_id(
    sample_client,
    jobs_url,
    run_request_name,
    valid_request,
    error,
    request,
    submit_success,
    existing_run_id,
    dynamic_architecture_success,
):
    """
    Tests submitting circuits for execution
    """
    run_request = request.getfixturevalue(run_request_name)
    calibration_set_id = uuid.UUID('ec6f6478-a99b-4e75-8a94-2f9cb0511bce')
    run_request.calibration_set_id = calibration_set_id
    dynamic_architecture_success.json_data['calibration_set_id'] = calibration_set_id
    when(requests).get(sample_client._api.url(APIEndpoint.CALIBRATED_GATES, str(calibration_set_id)), ...).thenReturn(
        dynamic_architecture_success
    )

    if valid_request:
        expect(requests, times=1).post(jobs_url, **post_jobs_args(run_request)).thenReturn(submit_success)
    if error is None:
        assert sample_client.submit_circuits(**submit_circuits_args(run_request)) == existing_run_id
    else:
        with pytest.raises(type(error), match=str(error)):
            sample_client.submit_circuits(**submit_circuits_args(run_request))

    verifyNoUnwantedInteractions()
    unstub()


def test_submit_circuits_does_not_activate_heralding_by_default(
    sample_client, jobs_url, minimal_run_request, submit_success, dynamic_architecture_url, dynamic_architecture_success
):
    """
    Test submitting run request without heralding
    """
    # Expect request to have heralding mode NONE by default
    assert post_jobs_args(minimal_run_request)['json']['heralding_mode'] == HeraldingMode.NONE.value
    expect(requests, times=1).post(jobs_url, **post_jobs_args(minimal_run_request)).thenReturn(submit_success)
    when(requests).get(dynamic_architecture_url, ...).thenReturn(dynamic_architecture_success)

    # Specify no heralding mode in submit_circuits
    sample_client.submit_circuits(circuits=minimal_run_request.circuits, shots=minimal_run_request.shots)

    verifyNoUnwantedInteractions()
    unstub()


def test_submit_circuits_does_not_activate_dd_by_default(
    sample_client, jobs_url, minimal_run_request, submit_success, dynamic_architecture_url, dynamic_architecture_success
):
    """
    Test submitting run request without dynamical decoupling
    """
    # Expect request to have dynamical decoupling mode DISABLED by default
    assert post_jobs_args(minimal_run_request)['json']['dd_mode'] == DDMode.DISABLED.value

    expect(requests, times=1).post(jobs_url, **post_jobs_args(minimal_run_request)).thenReturn(submit_success)
    when(requests).get(dynamic_architecture_url, ...).thenReturn(dynamic_architecture_success)

    # Specify no dynamical decoupling mode in submit_circuits
    sample_client.submit_circuits(circuits=minimal_run_request.circuits, shots=minimal_run_request.shots)

    verifyNoUnwantedInteractions()
    unstub()


def test_submit_circuits_raises_with_invalid_shots(sample_client, minimal_run_request):
    """
    Test that submitting run request with invalid number of shots raises ValueError
    """
    args = submit_circuits_args(minimal_run_request)
    args['shots'] = 0
    with pytest.raises(ValueError, match='Number of shots must be greater than zero.'):
        sample_client.submit_circuits(**args)


def test_submit_circuits_sets_heralding_mode_in_run_request(
    sample_client,
    jobs_url,
    run_request_with_heralding,
    submit_success,
    dynamic_architecture_url,
    dynamic_architecture_success,
):
    """
    Test submitting run request with heralding
    """
    # Expect heralding mode to be the same as in run request
    expected_heralding_mode = run_request_with_heralding.heralding_mode.value
    assert post_jobs_args(run_request_with_heralding)['json']['heralding_mode'] == expected_heralding_mode
    expect(requests, times=1).post(jobs_url, **post_jobs_args(run_request_with_heralding)).thenReturn(submit_success)
    expect(requests, times=1).get(dynamic_architecture_url, ...).thenReturn(dynamic_architecture_success)

    assert submit_circuits_args(run_request_with_heralding)['options'].heralding_mode == expected_heralding_mode
    sample_client.submit_circuits(**submit_circuits_args(run_request_with_heralding))

    verifyNoUnwantedInteractions()
    unstub()


def test_submit_circuits_sets_dd_mode_in_run_request(
    sample_client,
    jobs_url,
    run_request_with_dd,
    submit_success,
    dynamic_architecture_url,
    dynamic_architecture_success,
):
    """
    Test submitting run request with dynamical decoupling
    """
    # Expect dynamical decoupling mode to be the same as in run request
    expected_dd_mode = run_request_with_dd.dd_mode.value
    expected_dd_strategy = run_request_with_dd.dd_strategy

    assert post_jobs_args(run_request_with_dd)['json']['dd_mode'] == expected_dd_mode
    assert DDStrategy(**post_jobs_args(run_request_with_dd)['json']['dd_strategy']) == expected_dd_strategy
    expect(requests, times=1).post(jobs_url, **post_jobs_args(run_request_with_dd)).thenReturn(submit_success)
    expect(requests, times=1).get(dynamic_architecture_url, ...).thenReturn(dynamic_architecture_success)

    assert submit_circuits_args(run_request_with_dd)['options'].dd_mode == expected_dd_mode
    assert submit_circuits_args(run_request_with_dd)['options'].dd_strategy == expected_dd_strategy
    sample_client.submit_circuits(**submit_circuits_args(run_request_with_dd))

    verifyNoUnwantedInteractions()
    unstub()


def test_submit_circuits_gets_architecture_once(
    sample_client,
    jobs_url,
    minimal_run_request,
    submit_success,
    dynamic_architecture_success,
):
    """
    Test that dynamic quantum architecture is only requested once from the QC when calset id is specified
    """
    calibration_set_id = uuid.UUID('ec6f6478-a99b-4e75-8a94-2f9cb0511bce')
    minimal_run_request.calibration_set_id = calibration_set_id
    dynamic_architecture_success.json_data['calibration_set_id'] = calibration_set_id
    expect(requests, times=1).get(
        sample_client._api.url(APIEndpoint.CALIBRATED_GATES, str(calibration_set_id)), ...
    ).thenReturn(dynamic_architecture_success)
    expect(requests, times=1).post(jobs_url, **post_jobs_args(minimal_run_request)).thenReturn(submit_success)
    # Get architecture explicitly and then submit job
    sample_client.get_dynamic_quantum_architecture(calibration_set_id)
    sample_client.submit_circuits(**submit_circuits_args(minimal_run_request))
    verifyNoUnwantedInteractions()
    unstub()


def test_submit_circuits_raises_with_invalid_heralding_mode(
    sample_client,
    dynamic_architecture_url,
    dynamic_architecture_success,
):
    """
    Test that submitting run request with invalid heralding mode raises an error
    """
    when(requests).get(dynamic_architecture_url, ...).thenReturn(dynamic_architecture_success)
    with pytest.raises(ValueError, match="Input should be 'none' or 'zeros'"):
        sample_client.submit_circuits(
            circuits=[], shots=10, options=CircuitCompilationOptions(heralding_mode='invalid')
        )


def test_get_run_adds_user_agent(
    sample_client,
    existing_job_url,
    existing_run_id,
    pending_compilation_job_result,
    dynamic_architecture_url,
    dynamic_architecture_success,
):
    """
    Tests that get_run without client signature adds the correct User-Agent header
    """
    expect(requests, times=1).get(existing_job_url, **get_jobs_args()).thenReturn(pending_compilation_job_result)
    when(requests).get(dynamic_architecture_url, ...).thenReturn(dynamic_architecture_success)

    sample_client.get_run(existing_run_id)

    verifyNoUnwantedInteractions()
    unstub()


def test_get_run_adds_user_agent_with_client_signature(
    client_with_signature, client_signature, existing_job_url, existing_run_id, pending_compilation_job_result
):
    """
    Tests that get_run with client signature adds the correct User-Agent header
    """
    assert client_signature in client_with_signature._signature
    expect(requests, times=1).get(
        existing_job_url,
        **get_jobs_args(user_agent=client_with_signature._signature),
    ).thenReturn(pending_compilation_job_result)

    client_with_signature.get_run(existing_run_id)

    verifyNoUnwantedInteractions()
    unstub()


def test_get_run_status_and_results_for_existing_run(
    sample_client,
    existing_job_url,
    sample_circuit_metadata,
    sample_calibration_set_id,
    pending_compilation_job_result,
    pending_execution_job_result,
    pending_deletion_job_result,
    deleted_job_result,
    ready_job_result,
    existing_run_id,
):
    """
    Tests getting the run status
    """
    expect(requests, times=5).get(existing_job_url, **get_jobs_args()).thenReturn(
        pending_compilation_job_result
    ).thenReturn(pending_execution_job_result).thenReturn(ready_job_result).thenReturn(
        pending_deletion_job_result
    ).thenReturn(
        deleted_job_result
    )

    # First request gets status 'pending compilation'
    assert sample_client.get_run(existing_run_id).status == Status.PENDING_COMPILATION

    # Second requests gets status 'pending execution'
    assert sample_client.get_run(existing_run_id).status == Status.PENDING_EXECUTION

    # Third request gets status 'ready'
    job_result = sample_client.get_run(existing_run_id)
    assert job_result.status == Status.READY
    assert job_result.measurements is not None
    assert job_result.metadata.request.calibration_set_id == sample_calibration_set_id
    assert job_result.metadata.request.circuits[0].metadata == sample_circuit_metadata

    # Fourth request gets status 'pending deletion'
    assert sample_client.get_run(existing_run_id).status == Status.PENDING_DELETION

    # Fifth request gets status 'deleted'
    job_result = sample_client.get_run(existing_run_id)
    assert job_result.status == Status.DELETED
    assert job_result.measurements is None
    assert job_result.metadata.request.circuits == []
    assert job_result.metadata.request.shots == 1

    verifyNoUnwantedInteractions()
    unstub()


def test_get_run_status_for_existing_run(
    sample_client,
    existing_job_status_url,
    pending_compilation_status,
    pending_execution_status,
    ready_status,
    pending_deletion_status,
    deleted_status,
    existing_run_id,
):
    """
    Tests getting the run status
    """
    expect(requests, times=5).get(existing_job_status_url, **get_jobs_args()).thenReturn(
        pending_compilation_status
    ).thenReturn(pending_execution_status).thenReturn(ready_status).thenReturn(pending_deletion_status).thenReturn(
        deleted_status
    )

    # First request gets status 'pending compilation'
    assert sample_client.get_run_status(existing_run_id).status == Status.PENDING_COMPILATION
    # Second request gets status 'pending execution'
    assert sample_client.get_run_status(existing_run_id).status == Status.PENDING_EXECUTION
    # Third request gets status 'ready'
    assert sample_client.get_run_status(existing_run_id).status == Status.READY
    # Fourth request gets status 'pending deletion'
    assert sample_client.get_run_status(existing_run_id).status == Status.PENDING_DELETION
    # Fifth request gets status 'deleted'
    assert sample_client.get_run_status(existing_run_id).status == Status.DELETED

    verifyNoUnwantedInteractions()
    unstub()


def test_get_run_status_and_results_for_missing_run(sample_client, jobs_url, missing_run_id):
    """
    Tests getting a task that was not created
    """
    expect(requests, times=1).get(f'{jobs_url}/{missing_run_id}', **get_jobs_args()).thenReturn(
        MockJsonResponse(404, {'detail': 'not found'})
    )

    with pytest.raises(HTTPError) as e:
        sample_client.get_run(missing_run_id)
    assert e.value.response.status_code == 404

    verifyNoUnwantedInteractions()
    unstub()


def test_get_run_status_for_missing_run(sample_client, jobs_url, missing_run_id):
    """
    Tests getting a task that was not created
    """
    expect(requests, times=1).get(f'{jobs_url}/{missing_run_id}/status', **get_jobs_args()).thenReturn(
        MockJsonResponse(404, {'detail': 'not found'})
    )

    with pytest.raises(HTTPError) as e:
        sample_client.get_run_status(missing_run_id)
    assert e.value.response.status_code == 404

    verifyNoUnwantedInteractions()
    unstub()


def test_waiting_for_compilation(
    sample_client,
    existing_job_status_url,
    pending_compilation_status,
    pending_execution_status,
    existing_job_url,
    pending_execution_job_result,
    existing_run_id,
):
    """
    Tests waiting for compilation for an existing task
    """
    expect(requests, times=3).get(existing_job_status_url, **get_jobs_args()).thenReturn(
        pending_compilation_status
    ).thenReturn(pending_compilation_status).thenReturn(pending_execution_status)
    expect(requests, times=1).get(existing_job_url, **get_jobs_args()).thenReturn(pending_execution_job_result)

    assert sample_client.wait_for_compilation(existing_run_id).status == Status.PENDING_EXECUTION

    verifyNoUnwantedInteractions()
    unstub()


def test_wait_for_compilation_adds_user_agent_with_signature(
    client_with_signature,
    client_signature,
    existing_job_status_url,
    pending_compilation_status,
    pending_execution_status,
    existing_job_url,
    pending_execution_job_result,
    existing_run_id,
):
    """
    Tests that wait_for_compilation with client signature adds the correct User-Agent header
    """
    assert client_signature in client_with_signature._signature
    expect(requests, times=3).get(
        existing_job_status_url, **get_jobs_args(user_agent=client_with_signature._signature)
    ).thenReturn(pending_compilation_status).thenReturn(pending_compilation_status).thenReturn(pending_execution_status)
    expect(requests, times=1).get(
        existing_job_url, **get_jobs_args(user_agent=client_with_signature._signature)
    ).thenReturn(pending_execution_job_result)

    assert client_with_signature.wait_for_compilation(existing_run_id).status == Status.PENDING_EXECUTION

    verifyNoUnwantedInteractions()
    unstub()


def test_waiting_for_results(
    sample_client,
    existing_job_status_url,
    pending_compilation_status,
    pending_execution_status,
    ready_status,
    existing_job_url,
    ready_job_result,
    existing_run_id,
):
    """
    Tests waiting for results for an existing task
    """
    expect(requests, times=3).get(existing_job_status_url, **get_jobs_args()).thenReturn(
        pending_compilation_status
    ).thenReturn(pending_execution_status).thenReturn(ready_status)
    expect(requests, times=1).get(existing_job_url, **get_jobs_args()).thenReturn(ready_job_result)

    assert sample_client.wait_for_results(existing_run_id).status == Status.READY

    verifyNoUnwantedInteractions()
    unstub()


def test_wait_for_results_adds_user_agent_with_signature(
    client_with_signature,
    client_signature,
    existing_job_status_url,
    pending_compilation_status,
    pending_execution_status,
    ready_status,
    existing_job_url,
    ready_job_result,
    existing_run_id,
):
    """
    Tests that wait_for_results without client signature adds the correct User-Agent header
    """
    assert client_signature in client_with_signature._signature
    expect(requests, times=3).get(
        existing_job_status_url, **get_jobs_args(user_agent=client_with_signature._signature)
    ).thenReturn(pending_compilation_status).thenReturn(pending_execution_status).thenReturn(ready_status)
    expect(requests, times=1).get(
        existing_job_url, **get_jobs_args(user_agent=client_with_signature._signature)
    ).thenReturn(ready_job_result)

    assert client_with_signature.wait_for_results(existing_run_id).status == Status.READY

    verifyNoUnwantedInteractions()
    unstub()


def test_get_quantum_architecture(
    sample_client, quantum_architecture_url, sample_static_architecture, static_architecture_success
):
    """Test retrieving the quantum architecture"""
    expect(requests, times=1).get(quantum_architecture_url, ...).thenReturn(static_architecture_success)

    assert sample_client.get_quantum_architecture() == QuantumArchitectureSpecification(
        **sample_static_architecture['quantum_architecture']
    )

    verifyNoUnwantedInteractions()
    unstub()


def test_get_feedback_groups(base_url, channel_properties_url, channel_properties_success, static_architecture_success):
    """Test retrieving the feedback groups."""
    when(requests).get(f'{base_url}/info/client-libraries', headers=ANY, timeout=ANY).thenReturn(
        mock_supported_client_libraries_response()
    )
    expect(requests, times=1).get(f'{base_url}/cocos/quantum-architecture', ...).thenReturn(static_architecture_success)
    iqm_client = IQMClient(base_url, api_variant=APIVariant.V2)
    expect(requests, times=1).get(channel_properties_url, ...).thenReturn(channel_properties_success)

    assert iqm_client.get_feedback_groups() == (frozenset({'QB1', 'QB2'}), frozenset({'QB3'}))

    verifyNoUnwantedInteractions()
    unstub()


def test_user_warning_is_emitted_when_warnings_in_response(
    sample_client, existing_job_url, job_result_with_warnings, existing_run_id
):
    """Test that a warning is emitted when warnings are present in the response"""
    expect(requests, times=1).get(existing_job_url, ...).thenReturn(job_result_with_warnings)

    expected_message = job_result_with_warnings.json()['warnings'][0]
    with pytest.warns(UserWarning, match=expected_message):
        sample_client.get_run(existing_run_id)

    verifyNoUnwantedInteractions()
    unstub()


def test_base_url_is_invalid():
    """Test that an exception is raised when the base URL is invalid"""
    invalid_base_url = 'xyz://example.com'
    with pytest.raises(ClientConfigurationError) as exc:
        IQMClient(invalid_base_url)
    assert f'The URL schema has to be http or https. Incorrect schema in URL: {invalid_base_url}' == str(exc.value)


def test_run_result_throws_json_decode_error_if_received_not_json(
    sample_client, existing_job_url, existing_run_id, not_valid_json_response
):
    """Test that an exception is raised when the response is not a valid JSON"""
    expect(requests, times=1).get(existing_job_url, ...).thenReturn(not_valid_json_response)

    with pytest.raises(CircuitExecutionError):
        sample_client.get_run(existing_run_id)

    verifyNoUnwantedInteractions()
    unstub()


def test_run_result_status_throws_json_decode_error_if_received_not_json(
    sample_client, existing_job_status_url, existing_run_id, not_valid_json_response
):
    """Test that an exception is raised when the response is not a valid JSON"""
    expect(requests, times=1).get(existing_job_status_url, ...).thenReturn(not_valid_json_response)

    with pytest.raises(CircuitExecutionError):
        sample_client.get_run_status(existing_run_id)

    verifyNoUnwantedInteractions()
    unstub()


def test_quantum_architecture_throws_json_decode_error_if_received_not_json(
    sample_client, quantum_architecture_url, not_valid_json_response
):
    """Test that an exception is raised when the response is not a valid JSON"""
    expect(requests, times=1).get(quantum_architecture_url, ...).thenReturn(not_valid_json_response)

    with pytest.raises(ArchitectureRetrievalError):
        sample_client.get_quantum_architecture()

    verifyNoUnwantedInteractions()
    unstub()


def test_submit_circuits_throws_json_decode_error_if_received_not_json(
    sample_client, jobs_url, not_valid_json_response, dynamic_architecture_url, dynamic_architecture_success
):
    """Test that an exception is raised when the response is not a valid JSON"""
    expect(requests, times=1).post(jobs_url, ...).thenReturn(not_valid_json_response)
    expect(requests, times=1).get(dynamic_architecture_url, ...).thenReturn(dynamic_architecture_success)

    with pytest.raises(CircuitExecutionError):
        sample_client.submit_circuits([])

    verifyNoUnwantedInteractions()
    unstub()


def test_submit_circuits_throws_client_configuration_error_on_400(
    sample_client,
    jobs_url,
    not_valid_client_configuration_response,
    dynamic_architecture_url,
    dynamic_architecture_success,
):
    """Test that an exception is raised when the response is 400"""
    expect(requests, times=1).post(jobs_url, ...).thenReturn(not_valid_client_configuration_response)
    expect(requests, times=1).get(dynamic_architecture_url, ...).thenReturn(dynamic_architecture_success)

    with pytest.raises(ClientConfigurationError):
        sample_client.submit_circuits([])

    verifyNoUnwantedInteractions()
    unstub()


def test_submit_circuits_validates_circuits(sample_client, sample_circuit):
    """
    Tests that <submit_circuits> validates the batch of provided circuits
    before submitting them for execution
    """
    invalid_circuit = sample_circuit.model_copy()
    invalid_circuit.name = ''  # Invalidate the circuit on purpose
    with pytest.raises(CircuitValidationError, match='The circuit at index 1 failed the validation'):
        sample_client.submit_circuits(circuits=[sample_circuit, invalid_circuit], shots=10)


def test_validate_circuit_accepts_valid_circuit(sample_circuit):
    """
    Tests that custom Pydantic validator (triggered via <validate_circuit>)
    accepts a valid circuit.
    """
    circuit = sample_circuit.model_copy()
    validate_circuit(circuit)


def test_validate_circuit_detects_circuit_name_is_empty_string(sample_circuit):
    """
    Tests that custom Pydantic validator (triggered via <validate_circuit>)
    catches empty name of a circuit
    """
    circuit = sample_circuit.model_copy()
    circuit.name = ''
    with pytest.raises(ValueError, match='A circuit should have a non-empty string for a name'):
        validate_circuit(circuit)


def test_validate_circuit_checks_circuit_has_at_least_one_instruction(sample_circuit):
    """
    Tests that custom Pydantic validator (triggered via <validate_circuit>)
    catches when circuit instructions container has 0 instructions
    """
    circuit = sample_circuit.model_copy()
    circuit.instructions = tuple()
    with pytest.raises(ValueError, match='Each circuit should have at least one instruction'):
        validate_circuit(circuit)


def test_validate_circuit_checks_instruction_name_is_supported(sample_circuit):
    """
    Tests that custom Pydantic validator (triggered via <validate_circuit>)
    catches when instruction name is set to an unknown instruction type
    """
    circuit = sample_circuit.model_copy()
    circuit.instructions[0].name = 'kaboom'
    with pytest.raises(ValueError, match='Unknown operation "kaboom"'):
        validate_circuit(circuit)


def test_validate_circuit_checks_instruction_implementation_is_string(sample_circuit):
    """
    Tests that custom Pydantic validator (triggered via <validate_circuit>)
    catches when instruction implementation is set to an empty string
    """
    circuit = sample_circuit.model_copy()
    circuit.instructions[0].implementation = ''
    with pytest.raises(ValueError, match='Implementation of the instruction should be None, or a non-empty string'):
        validate_circuit(circuit)


def test_validate_circuit_checks_instruction_qubit_count(sample_circuit):
    """
    Tests that custom Pydantic validator (triggered via <validate_circuit>)
    catches when qubit count of the instruction does not match the arity of
    that instruction
    """
    circuit = sample_circuit.model_copy()
    circuit.instructions[0].qubits += ('Qubit C',)
    with pytest.raises(ValueError, match=r'The "cz" operation acts on 2 qubit\(s\), but 3 were given'):
        validate_circuit(circuit)


def test_validate_circuit_extra_arguments(sample_circuit):
    """
    Tests that custom Pydantic validator (triggered via <validate_circuit>)
    catches when submitted argument names of the instruction are not supported
    """
    circuit = sample_circuit.model_copy()
    circuit.instructions[1].args['arg_x'] = 'This argument name is not supported by the operation'
    with pytest.raises(ValueError, match='The operation "prx" allows'):
        validate_circuit(circuit)


def test_validate_circuit_missing_arguments():
    """
    Tests that custom Pydantic validator (triggered via <validate_circuit>)
    catches when submitted argument names of the instruction are not supported
    """
    with pytest.raises(ValueError, match='The operation "prx" requires'):
        Circuit(
            name='The circuit',
            instructions=[
                Instruction(
                    name='prx',
                    qubits=('QB1',),
                    args={'phase_t': 0.3},
                ),
            ],
        )


def test_validate_circuit_checks_instruction_argument_types(sample_circuit):
    """
    Tests that custom Pydantic validator (triggered via <validate_circuit>)
    catches when submitted argument types of the instruction are not supported
    """
    circuit = sample_circuit.model_copy()
    circuit.instructions[1].args['phase_t'] = '0.7'
    with pytest.raises(TypeError, match='The argument "phase_t" should be of one of the following supported types'):
        validate_circuit(circuit)


def test_validate_circuit_can_handle_raw_instructions(sample_circuit_with_raw_instructions):
    """
    Tests that custom Pydantic validator (triggered via <validate_circuit>)
    accepts a circuit with raw instructions
    """
    circuit = sample_circuit_with_raw_instructions.model_copy()
    validate_circuit(circuit)


def test_abort_job_successful(sample_client, existing_job_url, existing_run_id, abort_job_success):
    """
    Tests aborting a job
    """
    expect(requests, times=1).post(f'{existing_job_url}/abort', **post_jobs_args()).thenReturn(abort_job_success)

    sample_client.abort_job(existing_run_id)

    verifyNoUnwantedInteractions()
    unstub()


@pytest.mark.parametrize('status_code', [404, 409])
def test_abort_job_failed(status_code, sample_client, existing_job_url, existing_run_id, abort_job_failed):
    """
    Tests aborting a job raises JobAbortionError if server returned error response
    """
    response = abort_job_failed
    response.status_code = status_code
    expect(requests, times=1).post(f'{existing_job_url}/abort', **post_jobs_args()).thenReturn(response)

    with pytest.raises(JobAbortionError):
        sample_client.abort_job(existing_run_id)

    verifyNoUnwantedInteractions()
    unstub()


@pytest.mark.parametrize(
    'params',
    [
        {},
        {'options': CircuitCompilationOptions(heralding_mode=HeraldingMode.ZEROS, active_reset_cycles=1)},
        {'custom_settings': {'some_setting': 1}},
        {'calibration_set_id': uuid.uuid4()},
        {'options': CircuitCompilationOptions(max_circuit_duration_over_t2=0.0)},
        {'qubit_mapping': {'QB1': 'QB2', 'QB2': 'QB1'}},
    ],
)
def test_create_and_submit_run_request(
    sample_client,
    sample_circuit,
    jobs_url,
    submit_success,
    existing_run_id,
    dynamic_architecture_url,
    dynamic_architecture_success,
    params,
):
    """
    Tests that calling create_run_request and then submit_run_request is equivalent to calling submit_circuits.
    """
    if 'calibration_set_id' in params:
        when(requests).get(
            sample_client._api.url(APIEndpoint.CALIBRATED_GATES, params['calibration_set_id']), ...
        ).thenReturn(dynamic_architecture_success)
    else:
        when(requests).get(dynamic_architecture_url, ...).thenReturn(dynamic_architecture_success)

    run_request = sample_client.create_run_request([sample_circuit], **params)
    if 'options' in params:
        assert run_request.active_reset_cycles == params['options'].active_reset_cycles
    expect(requests, times=2).post(jobs_url, **post_jobs_args(run_request)).thenReturn(submit_success)
    assert sample_client.submit_run_request(run_request) == existing_run_id
    assert sample_client.submit_circuits([sample_circuit], **params) == existing_run_id

    verifyNoUnwantedInteractions()
    unstub()


@pytest.mark.parametrize(
    'run_request_name, quantum_architecture_name, sample_circuit_name',
    [
        (run_request, success_result, sample_circuit)
        for run_request in [
            'run_request_with_move_validation',
            'run_request_without_prx_move_validation',
            'run_request_with_move_gate_frame_tracking',
        ]
        for success_result, sample_circuit in zip(
            ['dynamic_architecture_success', 'move_architecture_success', 'move_architecture_success'],
            ['sample_circuit', 'move_circuit', 'move_circuit_with_prx_in_the_sandwich'],
        )
    ],
)
def test_compiler_options_are_used_and_sent(
    sample_client,
    sample_circuit_name,
    jobs_url,
    run_request_name,
    request,
    submit_success,
    dynamic_architecture_url,
    quantum_architecture_name,
):
    """
    Tests submitting circuits for execution
    """
    run_request = request.getfixturevalue(run_request_name)
    quantum_architecture_success = request.getfixturevalue(quantum_architecture_name)
    run_request.circuits = [request.getfixturevalue(sample_circuit_name)]

    when(requests).get(dynamic_architecture_url, ...).thenReturn(quantum_architecture_success)
    if (
        sample_circuit_name != 'move_circuit_with_prx_in_the_sandwich'  # Valid circuit
        or run_request_name == 'run_request_without_prx_move_validation'  # Validation is turned off
    ):
        expect(requests, times=1).post(jobs_url, **post_jobs_args(run_request)).thenReturn(submit_success)
        sample_client.submit_circuits(**submit_circuits_args(run_request))
    else:  # Invalid circuit and validation is turned on.
        with pytest.raises(CircuitValidationError):
            sample_client.submit_circuits(**submit_circuits_args(run_request))

    verifyNoUnwantedInteractions()
    unstub()


def test_get_dynamic_quantum_architecture_with_calset_id(
    sample_client, base_url, dynamic_architecture_success, sample_dynamic_architecture
):
    """Tests that the correct dynamic quantum architecture for the given ``calibration_set_id`` is returned."""
    calset_id = sample_dynamic_architecture.calibration_set_id
    expect(requests, times=1).get(f'{base_url}/api/v1/calibration/{calset_id}/gates', ...).thenReturn(
        dynamic_architecture_success
    )
    assert sample_client.get_dynamic_quantum_architecture(calset_id) == sample_dynamic_architecture
    verifyNoUnwantedInteractions()
    unstub()


def test_get_dynamic_quantum_architecture_with_calset_id_caches(
    sample_client, base_url, dynamic_architecture_success, sample_dynamic_architecture
):
    """
    Tests that cached dynamic quantum architecture is returned when requesting it for the second time for
    a given calibration set id.
    """
    calset_id = sample_dynamic_architecture.calibration_set_id
    expect(requests, times=1).get(f'{base_url}/api/v1/calibration/{calset_id}/gates', ...).thenReturn(
        dynamic_architecture_success
    )

    assert sample_client.get_dynamic_quantum_architecture(calset_id) == sample_dynamic_architecture
    assert sample_client.get_dynamic_quantum_architecture(calset_id) == sample_dynamic_architecture

    verifyNoUnwantedInteractions()
    unstub()


def test_get_dynamic_quantum_architecture_without_calset_id(
    sample_client, base_url, dynamic_architecture_success, sample_dynamic_architecture
):
    """Tests that the correct dynamic quantum architecture for the default calibration set is returned."""
    expect(requests, times=1).get(f'{base_url}/api/v1/calibration/default/gates', ...).thenReturn(
        dynamic_architecture_success
    )
    assert sample_client.get_dynamic_quantum_architecture() == sample_dynamic_architecture
    verifyNoUnwantedInteractions()
    unstub()


def test_get_dynamic_quantum_architecture_without_calset_id_does_not_cache(
    sample_client, base_url, dynamic_architecture_success, sample_dynamic_architecture
):
    """
    Tests that the correct dynamic quantum architecture is returned in the case where default calset
    changes between two invocations of get_dynamic_quantum_architecture().
    """
    dynamic_quantum_architecture_2 = {
        'calibration_set_id': '3902d525-d8f4-42c0-9fa9-6bbd535b6c80',
        'qubits': ['QB1', 'QB2'],
        'computational_resonators': [],
        'gates': {
            'prx': {
                'implementations': {
                    'drag_gaussian': {
                        'loci': [['QB1'], ['QB2']],
                    }
                },
                'default_implementation': 'drag_gaussian',
                'override_default_implementation': {},
            },
        },
    }
    dynamic_architecture_success_2 = MockJsonResponse(200, dynamic_quantum_architecture_2)
    expect(requests, times=2).get(f'{base_url}/api/v1/calibration/default/gates', ...).thenReturn(
        dynamic_architecture_success
    ).thenReturn(dynamic_architecture_success_2)

    assert sample_client.get_dynamic_quantum_architecture() == sample_dynamic_architecture
    assert sample_client.get_dynamic_quantum_architecture() == DynamicQuantumArchitecture(
        **dynamic_quantum_architecture_2
    )

    verifyNoUnwantedInteractions()
    unstub()


def test_get_dynamic_quantum_architecture_throws_json_decode_error_if_received_not_json(
    sample_client, base_url, not_valid_json_response
):
    """Test that an exception is raised when the response is not a valid JSON"""
    expect(requests, times=1).get(f'{base_url}/api/v1/calibration/default/gates', ...).thenReturn(
        not_valid_json_response
    )

    with pytest.raises(ArchitectureRetrievalError):
        sample_client.get_dynamic_quantum_architecture()

    verifyNoUnwantedInteractions()
    unstub()


def test_get_dynamic_quantum_architecture_not_found(base_url, sample_client):
    """Test that an informative error message is returned when 404 is returned due to version incompatibility."""
    client_version = parse(version('iqm-client'))
    min_version = f'{client_version.major + 2}.0'
    max_version = f'{client_version.major + 3}.0'
    when(requests).get(f'{base_url}/info/client-libraries', headers=ANY, timeout=ANY).thenReturn(
        mock_supported_client_libraries_response(min_version=min_version, max_version=max_version)
    )
    when(requests).get(f'{base_url}/api/v1/calibration/default/gates', ...).thenReturn(MockJsonResponse(404, {}))
    with pytest.raises(
        HTTPError,
        match=re.escape(
            f'Your IQM Client version {client_version} was built for a different version of IQM Server. '
            f'You might encounter issues. For the best experience, consider using a version '
            f'of IQM Client that satisfies {min_version} <= iqm-client < {max_version}.'
        ),
    ):
        sample_client.get_dynamic_quantum_architecture()
    unstub()


@pytest.mark.parametrize(
    'iqm_client_name,server_version_diff',
    [
        ('iqm_client', 0),
        ('iqm-client', 1),
    ],
)
def test_check_versions_success(base_url, iqm_client_name, server_version_diff, recwarn):
    """Test that a warning about version incompatibility is shown when initializing client with incompatible server."""
    client_version = parse(version('iqm-client'))
    min_version = f'{client_version.major + server_version_diff}.0'
    max_version = f'{client_version.major + server_version_diff + 1}.0'
    when(requests).get(f'{base_url}/info/client-libraries', headers=ANY, timeout=ANY).thenReturn(
        mock_supported_client_libraries_response(
            iqm_client_name=iqm_client_name, min_version=min_version, max_version=max_version
        )
    )
    if server_version_diff == 0:
        IQMClient(base_url, api_variant=APIVariant.V2)
        assert len(recwarn) == 0
    else:
        with pytest.warns(
            UserWarning,
            match=re.escape(
                f'Your IQM Client version {client_version} was built for a different version of IQM Server. '
                f'You might encounter issues. For the best experience, consider using a version '
                f'of IQM Client that satisfies {min_version} <= iqm-client < {max_version}.'
            ),
        ):
            IQMClient(base_url)
    unstub()


def test_check_versions_bad_response(base_url):
    """Test that unexpected response from /client-libraries endpoint does not break the client."""
    when(requests).get(f'{base_url}/info/client-libraries', headers=ANY, timeout=ANY).thenReturn(
        MockJsonResponse(
            200,
            {'iqm_client': 'invalid-payload'},
        )
    )
    with pytest.warns(
        UserWarning,
        match=re.escape('Could not verify IQM Client compatibility with the server. You might encounter issues.'),
    ):
        IQMClient(base_url)
    unstub()


def test_check_versions_request_exception(base_url):
    """Test that exception raised by request to /client-libraries endpoint does not break the client."""
    when(requests).get(f'{base_url}/info/client-libraries', headers=ANY, timeout=ANY).thenRaise(
        requests.exceptions.RequestException
    )
    with pytest.warns(
        UserWarning,
        match=re.escape('Could not verify IQM Client compatibility with the server. You might encounter issues.'),
    ):
        IQMClient(base_url)
    unstub()


def test_get_run_counts(base_url, sample_client, existing_run_id):
    """Test that the number of runs is returned."""
    expect(requests, times=1).get(f'{base_url}/jobs/{existing_run_id}/counts', ...).thenReturn(
        MockJsonResponse(
            200,
            {
                'status': 'ready',
                'counts_batch': [
                    {
                        'measurement_keys': ['m1'],
                        'counts': {'0': 5, '1': 5},
                    },
                    {
                        'measurement_keys': ['m2'],
                        'counts': {'0': 1, '1': 9},
                    },
                ],
                'warnings': [],
            },
        )
    )
    counts = sample_client.get_run_counts(existing_run_id)
    assert counts.status == Status.READY
    assert counts.counts_batch == [
        Counts(measurement_keys=['m1'], counts={'0': 5, '1': 5}),
        Counts(measurement_keys=['m2'], counts={'0': 1, '1': 9}),
    ]
    verifyNoUnwantedInteractions()
    unstub()


def test_get_supported_client_libraries(base_url, sample_client):
    """Test retrieving client library information from server."""
    libraries_data = {
        'iqm-client': {
            'name': 'IQM Client',
            'package_name': 'iqm-client',
            'repo_url': 'https://github.com/iqm-finland/iqm-client',
            'package_url': 'https://pypi.org/project/iqm-client',
            'min': '14.0.0',
            'max': '15.0.0',
            'images': None,
        },
        'iqm-cortex-cli': {
            'name': 'IQM Cortex CLI',
            'package_name': 'iqm-cortex-cli',
            'repo_url': 'https://github.com/iqm-finland/cortex-cli',
            'package_url': 'https://pypi.org/project/iqm-cortex-cli',
            'min': '1.0.0',
            'max': '2.0.0',
            'images': [],
        },
    }

    expect(requests, times=1).get(f'{base_url}/info/client-libraries', headers=ANY, timeout=ANY).thenReturn(
        MockJsonResponse(200, libraries_data)
    )

    libraries = sample_client.get_supported_client_libraries()

    # Verify returned data matches expected structure
    assert len(libraries) == 2
    assert 'iqm-client' in libraries
    assert 'iqm-cortex-cli' in libraries

    # Verify specific library details
    iqm_client = libraries['iqm-client']
    assert iqm_client.name == 'IQM Client'
    assert iqm_client.package_name == 'iqm-client'
    assert iqm_client.repo_url == 'https://github.com/iqm-finland/iqm-client'
    assert iqm_client.package_url == 'https://pypi.org/project/iqm-client'
    assert iqm_client.min == '14.0.0'
    assert iqm_client.max == '15.0.0'

    verifyNoUnwantedInteractions()
    unstub()


def test_check_api_version_deprecation_warning(base_url):
    """Test that deprecation warning is raised when API version is deprecated."""
    with pytest.warns(
        DeprecationWarning,
        match=re.escape(
            'The V1 API is deprecated and will be removed in a future release. Please use the V2 API instead.'
        ),
    ):
        IQMClient(base_url, api_variant=APIVariant.V1)
    unstub()
