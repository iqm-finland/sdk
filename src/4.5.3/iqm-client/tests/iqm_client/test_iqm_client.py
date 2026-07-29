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

from base64 import b64encode
from copy import copy
from http import HTTPStatus
from math import pi
import time
import uuid

from iqm.iqm_client import (
    CircuitCompilationOptions,
    CircuitJob,
    CircuitValidationError,
    IQMClient,
    validate_circuit,
)
from iqm.iqm_server_client.iqm_server_client import REQUESTS_TIMEOUT, _IQMServerClient
from iqm.iqm_server_client.models import JobCompilation, JobData, JobError, JobMessage, JobStatus
from mockito import ANY, expect, patch, unstub, verifyNoUnwantedInteractions, when
import pytest
import requests

from exa.common.errors.station_control_errors import InternalServerError, NotFoundError
from iqm.pulse import Circuit, CircuitOperation
from iqm.station_control.client.authentication import ClientConfigurationError
from iqm.station_control.interface.models import (
    CircuitMeasurementCounts,
    CircuitMeasurementCountsBatch,
    DDMode,
    HeraldingMode,
    RunRequest,
)

from .conftest import MockJsonResponse, submit_circuits_args


def _patch_env(patcher, **patched):
    for key in ["IQM_TOKEN", "IQM_TOKENS_FILE"]:
        if patched.get(key):
            patcher(key, patched[key])
        else:
            patcher(key, "")


pytestmark = pytest.mark.usefixtures("unstub")


@pytest.fixture
def move_circuit():
    instructions = (
        CircuitOperation(
            name="prx",
            locus=("QB1",),
            args={"phase": 0.6 * pi, "angle": -0.4 * pi},
        ),
        CircuitOperation(
            name="move",
            locus=("QB3", "CR1"),
            args={},
        ),
        CircuitOperation(
            name="cz",
            locus=("QB1", "CR1"),
            args={},
        ),
        CircuitOperation(
            name="cz",
            locus=("QB2", "CR1"),
            args={},
        ),
        CircuitOperation(
            name="move",
            locus=("QB3", "CR1"),
            args={},
        ),
    )
    return Circuit(name="CR1 circuit", instructions=instructions)


@pytest.fixture
def move_circuit_with_prx_in_the_sandwich():
    instructions = (
        CircuitOperation(
            name="prx",
            locus=("QB1",),
            args={"phase": 0.6 * pi, "angle": -0.4 * pi},
        ),
        CircuitOperation(
            name="move",
            locus=("QB3", "CR1"),
            args={},
        ),
        CircuitOperation(
            name="prx",
            locus=("QB3",),
            args={"phase": 0.6 * pi, "angle": -0.4 * pi},
        ),
        CircuitOperation(
            name="move",
            locus=("QB3", "CR1"),
            args={},
        ),
    )
    return Circuit(name="CR1 circuit with PRX in the sandwich", instructions=instructions)


def sample_job_data(
    job_id: uuid.UUID, status: JobStatus = JobStatus.PROCESSING, compilation: JobCompilation | None = None
) -> JobData:
    return JobData(
        id=job_id,
        status=status,
        compilation=compilation,
        timeline=[],
    )


def test_submit_circuits_adds_user_agent(
    iqm_client_mock,
    minimal_run_request,
    sample_calset_id,
    sample_dynamic_architecture,
):
    """
    Tests that submit_circuit without client signature adds correct User-Agent header
    """
    client = iqm_client_mock
    expect(client._iqm_server_client, times=1).get_dynamic_quantum_architecture(sample_calset_id).thenReturn(
        sample_dynamic_architecture
    )
    expect(client._iqm_server_client, times=1).submit_circuits(minimal_run_request, use_timeslot=False).thenReturn(
        sample_calset_id
    )

    client.submit_circuits(**submit_circuits_args(minimal_run_request))

    verifyNoUnwantedInteractions()


def test_submit_circuits_adds_user_agent_with_client_signature(
    iqm_client_mock_with_signature,
    minimal_run_request,
    sample_calset_id,
    sample_dynamic_architecture,
):
    """
    Tests that submit_circuit with client signature adds correct User-Agent header
    """
    client = iqm_client_mock_with_signature
    expect(client._iqm_server_client, times=1).get_dynamic_quantum_architecture(sample_calset_id).thenReturn(
        sample_dynamic_architecture
    )
    expect(client._iqm_server_client, times=1).submit_circuits(minimal_run_request, use_timeslot=False).thenReturn(
        sample_calset_id
    )

    iqm_client_mock_with_signature.submit_circuits(**submit_circuits_args(minimal_run_request))

    verifyNoUnwantedInteractions()


@pytest.mark.parametrize(
    "run_request_name, valid_request, error",
    [
        ("minimal_run_request", True, None),
        ("run_request_with_heralding", True, None),
        (
            "run_request_with_invalid_qubit_mapping",
            False,
            CircuitValidationError("Multiple logical qubits map to the same physical qubit."),
        ),
        (
            "run_request_with_incomplete_qubit_mapping",
            False,
            CircuitValidationError(
                "The qubits {'Qubit B'} in circuit 'The circuit 😈' "
                "at index 0 are not found in the provided qubit mapping."
            ),
        ),
        ("run_request_without_qubit_mapping", True, None),
        ("run_request_with_calibration_set_id", True, None),
        ("run_request_with_duration_check_disabled", True, None),
        (
            "run_request_with_incompatible_options",
            False,
            ValueError(
                "Unable to perform full MOVE gate frame tracking if MOVE gate validation "
                'is not "strict" or "allow_prx".'
            ),
        ),
    ],
)
def test_submit_circuits_returns_id(
    iqm_client_mock,
    sample_calset_id,
    run_request_name,
    valid_request,
    error,
    request,
    existing_job_id,
    sample_dynamic_architecture,
):
    """
    Tests submitting circuits for execution
    """
    run_request = request.getfixturevalue(run_request_name)
    run_request.calibration_set_id = sample_calset_id
    if not (
        error is not None
        and isinstance(error, ValueError)
        and "Unable to perform full MOVE gate frame tracking" in str(error)
    ):
        expect(iqm_client_mock._iqm_server_client, times=1).get_dynamic_quantum_architecture(
            sample_calset_id
        ).thenReturn(sample_dynamic_architecture)

    if valid_request:
        expect(iqm_client_mock._iqm_server_client, times=1).submit_circuits(run_request, use_timeslot=False).thenReturn(
            JobData(id=existing_job_id, status=JobStatus.WAITING)
        )
    if error is None:
        assert iqm_client_mock.submit_circuits(**submit_circuits_args(run_request)).job_id == existing_job_id
    else:
        with pytest.raises(type(error), match=str(error)):
            iqm_client_mock.submit_circuits(**submit_circuits_args(run_request))

    verifyNoUnwantedInteractions()


def test_submit_circuits_does_not_activate_heralding_by_default(
    iqm_client_mock,
    minimal_run_request,
    sample_dynamic_architecture,
    sample_calset_id,
):
    """
    Test submitting run request without heralding
    """

    def mock_submit_circuits(run_request: RunRequest, use_timeslot: bool) -> uuid.UUID:
        # Expect request to have heralding mode NONE by default
        assert run_request.heralding_mode == HeraldingMode.NONE.value
        return run_request.calibration_set_id

    # Expect request to have heralding mode NONE by default
    patch(_IQMServerClient.submit_circuits, mock_submit_circuits)
    expect(iqm_client_mock._iqm_server_client, times=1).get_dynamic_quantum_architecture(sample_calset_id).thenReturn(
        sample_dynamic_architecture
    )

    # Specify no heralding mode in submit_circuits
    iqm_client_mock.submit_circuits(**submit_circuits_args(minimal_run_request))

    verifyNoUnwantedInteractions()


def test_submit_circuits_does_not_activate_dd_by_default(
    iqm_client_mock,
    minimal_run_request,
    sample_calset_id,
    sample_dynamic_architecture,
):
    """
    Test submitting run request without dynamical decoupling
    """

    def mock_submit_circuits(run_request: RunRequest, use_timeslot: bool) -> uuid.UUID:
        # Expect request to have heralding mode NONE by default
        assert run_request.dd_mode == DDMode.DISABLED.value
        return run_request.calibration_set_id

    patch(_IQMServerClient.submit_circuits, mock_submit_circuits)
    expect(iqm_client_mock._iqm_server_client, times=1).get_dynamic_quantum_architecture(sample_calset_id).thenReturn(
        sample_dynamic_architecture
    )

    # Specify no dynamical decoupling mode in submit_circuits
    iqm_client_mock.submit_circuits(**submit_circuits_args(minimal_run_request))

    verifyNoUnwantedInteractions()


def test_submit_circuits_raises_with_invalid_shots(iqm_client_mock, minimal_run_request, sample_calset_id):
    """
    Test that submitting run request with invalid number of shots raises ValueError
    """
    args = submit_circuits_args(minimal_run_request)
    args["shots"] = 0
    args["calibration_set_id"] = sample_calset_id
    with pytest.raises(ValueError, match="Number of shots must be greater than zero."):
        iqm_client_mock.submit_circuits(**args)


def test_submit_circuits_sets_heralding_mode_in_run_request(
    iqm_client_mock,
    run_request_with_heralding,
    sample_calset_id,
    sample_dynamic_architecture,
):
    """
    Test submitting run request with heralding
    """
    expected_heralding_mode = run_request_with_heralding.heralding_mode.value

    def mock_submit_circuits(run_request: RunRequest, use_timeslot: bool) -> uuid.UUID:
        # Expect request to have heralding mode NONE by default
        assert run_request.heralding_mode == expected_heralding_mode
        return run_request.calibration_set_id

    patch(_IQMServerClient.submit_circuits, mock_submit_circuits)
    expect(iqm_client_mock._iqm_server_client, times=1).get_dynamic_quantum_architecture(sample_calset_id).thenReturn(
        sample_dynamic_architecture
    )

    assert submit_circuits_args(run_request_with_heralding)["options"].heralding_mode == expected_heralding_mode
    iqm_client_mock.submit_circuits(**submit_circuits_args(run_request_with_heralding))

    verifyNoUnwantedInteractions()


def test_submit_circuits_sets_dd_mode_in_run_request(
    iqm_client_mock,
    run_request_with_dd,
    sample_calset_id,
    sample_dynamic_architecture,
):
    """
    Test submitting run request with dynamical decoupling
    """
    # Expect dynamical decoupling mode to be the same as in run request
    expected_dd_mode = run_request_with_dd.dd_mode.value
    expected_dd_strategy = run_request_with_dd.dd_strategy

    def mock_submit_circuits(run_request: RunRequest, use_timeslot: bool) -> uuid.UUID:
        # Expect request to have heralding mode NONE by default
        assert run_request.dd_mode == expected_dd_mode
        assert run_request.dd_strategy == expected_dd_strategy
        return run_request.calibration_set_id

    patch(_IQMServerClient.submit_circuits, mock_submit_circuits)
    expect(iqm_client_mock._iqm_server_client, times=1).get_dynamic_quantum_architecture(sample_calset_id).thenReturn(
        sample_dynamic_architecture
    )

    assert submit_circuits_args(run_request_with_dd)["options"].dd_mode == expected_dd_mode
    assert submit_circuits_args(run_request_with_dd)["options"].dd_strategy == expected_dd_strategy
    iqm_client_mock.submit_circuits(**submit_circuits_args(run_request_with_dd))

    verifyNoUnwantedInteractions()


def test_submit_circuits_gets_architecture_once(
    iqm_client_mock,
    minimal_run_request,
    sample_calset_id,
    sample_dynamic_architecture,
):
    """
    Test that dynamic quantum architecture is only requested once from the QC when calset id is specified
    """
    minimal_run_request.calibration_set_id = sample_calset_id
    expect(iqm_client_mock._iqm_server_client, times=1).get_dynamic_quantum_architecture(sample_calset_id).thenReturn(
        sample_dynamic_architecture
    )
    expect(iqm_client_mock._iqm_server_client, times=1).submit_circuits(
        minimal_run_request, use_timeslot=False
    ).thenReturn(sample_calset_id)
    # Get architecture explicitly and then submit job
    iqm_client_mock.get_dynamic_quantum_architecture(sample_calset_id)
    iqm_client_mock.submit_circuits(**submit_circuits_args(minimal_run_request))
    verifyNoUnwantedInteractions()


def test_submit_circuits_raises_with_invalid_heralding_mode(
    iqm_client_mock,
    sample_calset_id,
    sample_dynamic_architecture,
):
    """
    Test that submitting run request with invalid heralding mode raises an error
    """
    expect(iqm_client_mock._iqm_server_client, times=1).get_dynamic_quantum_architecture(sample_calset_id).thenReturn(
        sample_dynamic_architecture
    )
    with pytest.raises(ValueError, match="Input should be 'none' or 'zeros'"):
        iqm_client_mock.submit_circuits(
            circuits=[],
            shots=10,
            options=CircuitCompilationOptions(heralding_mode="invalid"),
            calibration_set_id=sample_calset_id,
        )


def test_get_job_and_results_for_existing_run(
    iqm_client_mock,
    sample_circuit_metadata,
    mock_circuit_job_internals_for_client,
    existing_job_id,
    sample_calset_id,
):
    """
    Tests getting the job status
    """
    expect(iqm_client_mock._iqm_server_client, times=4).get_job(existing_job_id).thenReturn(
        sample_job_data(existing_job_id, JobStatus.WAITING)
    ).thenReturn(sample_job_data(existing_job_id, JobStatus.PROCESSING)).thenReturn(
        sample_job_data(existing_job_id, JobStatus.COMPLETED, JobCompilation(calibration_set_id=sample_calset_id))
    )
    mock_circuit_job_internals_for_client(iqm_client_mock)

    assert iqm_client_mock.get_job(existing_job_id).status == JobStatus.WAITING
    assert iqm_client_mock.get_job(existing_job_id).status == JobStatus.PROCESSING

    job = iqm_client_mock.get_job(existing_job_id)
    assert job.status == JobStatus.COMPLETED

    assert job.result() is not None
    circuits_batch, job_params = job.payload()
    assert job_params.calibration_set_id == sample_calset_id
    assert circuits_batch[0]["metadata"] == sample_circuit_metadata

    verifyNoUnwantedInteractions()


def test_get_job_for_existing_run(
    iqm_client_mock,
    existing_job_id,
):
    """
    Tests getting the run status
    """
    expect(iqm_client_mock._iqm_server_client, times=3).get_job(existing_job_id).thenReturn(
        sample_job_data(existing_job_id, JobStatus.WAITING)
    ).thenReturn(sample_job_data(existing_job_id, JobStatus.PROCESSING)).thenReturn(
        sample_job_data(existing_job_id, JobStatus.COMPLETED)
    )

    # First request gets status 'pending compilation'
    assert iqm_client_mock.get_job(existing_job_id).status == JobStatus.WAITING
    # Second request gets status 'pending execution'
    assert iqm_client_mock.get_job(existing_job_id).status == JobStatus.PROCESSING
    # Third request gets status 'ready'
    assert iqm_client_mock.get_job(existing_job_id).status == JobStatus.COMPLETED

    verifyNoUnwantedInteractions()


def test_get_job_for_missing_run(iqm_client_mock):
    """
    Tests getting a task that was not created
    """
    missing_run_id = uuid.uuid4()

    expect(iqm_client_mock._iqm_server_client, times=1).get_job(missing_run_id).thenRaise(
        NotFoundError("Job not found.")
    )
    with pytest.raises(NotFoundError, match="Job not found."):
        iqm_client_mock.get_job(missing_run_id)

    verifyNoUnwantedInteractions()


def test_waiting_for_results(
    iqm_client_mock,
    existing_job_id,
):
    """
    Tests waiting for results for an existing task
    """
    expect(iqm_client_mock._iqm_server_client, times=4).get_job(existing_job_id).thenReturn(
        sample_job_data(existing_job_id, JobStatus.WAITING)
    ).thenReturn(sample_job_data(existing_job_id, JobStatus.WAITING)).thenReturn(
        sample_job_data(existing_job_id, JobStatus.PROCESSING)
    ).thenReturn(
        sample_job_data(existing_job_id, JobStatus.COMPLETED, JobCompilation(calibration_set_id=existing_job_id))
    )

    job = iqm_client_mock.get_job(existing_job_id)
    assert job.wait_for_completion() == JobStatus.COMPLETED

    verifyNoUnwantedInteractions()


def test_wait_for_completion_adds_user_agent_with_signature(
    iqm_client_mock_with_signature,
    client_signature,
    existing_job_id,
):
    """
    Tests that wait_for_completion without client signature adds the correct User-Agent header
    """
    assert client_signature in iqm_client_mock_with_signature._iqm_server_client._signature

    expect(iqm_client_mock_with_signature._iqm_server_client, times=4).get_job(existing_job_id).thenReturn(
        sample_job_data(existing_job_id, JobStatus.WAITING)
    ).thenReturn(sample_job_data(existing_job_id, JobStatus.WAITING)).thenReturn(
        sample_job_data(existing_job_id, JobStatus.PROCESSING)
    ).thenReturn(
        sample_job_data(existing_job_id, JobStatus.COMPLETED, JobCompilation(calibration_set_id=existing_job_id))
    )

    job = iqm_client_mock_with_signature.get_job(existing_job_id)
    assert job.wait_for_completion() == JobStatus.COMPLETED

    verifyNoUnwantedInteractions()


def test_get_quality_metrics_with_calset_id_calset_exists(
    iqm_client_mock,
    sample_calset_id,
    sample_quality_metric_set_iqm_server,
    sample_calibration_set_iqm_server,
    sample_quality_metrics_sc,
):
    """Tests that the correct quality metric set for the given ``calibration_set_id`` is returned."""

    expect(iqm_client_mock._iqm_server_client, times=1).get_calibration_set_quality_metric_set(
        sample_calset_id
    ).thenReturn(sample_quality_metrics_sc)

    quality_metric_set = iqm_client_mock.get_quality_metric_set(sample_calset_id)
    assert quality_metric_set == sample_quality_metric_set_iqm_server

    verifyNoUnwantedInteractions()


def test_get_quality_metrics_without_calset_id_calset_exists(
    iqm_client_mock,
    sample_quality_metric_set_iqm_server,
):
    """Tests that the correct quality metric set for the given ``calibration_set_id`` is returned."""

    expect(iqm_client_mock._iqm_server_client, times=1).get_calibration_set_quality_metric_set("default").thenReturn(
        sample_quality_metric_set_iqm_server
    )

    assert iqm_client_mock.get_quality_metric_set() == sample_quality_metric_set_iqm_server

    verifyNoUnwantedInteractions()


def test_get_feedback_groups(
    sample_channel_properties,
    sample_static_architecture,
    iqm_client_mock,
    sample_dut_label,
):
    """Test retrieving the feedback groups."""
    expect(iqm_client_mock._iqm_server_client, times=1).get_channel_properties().thenReturn(sample_channel_properties)
    expect(iqm_client_mock._iqm_server_client, times=1).get_duts().thenReturn(sample_dut_label)
    expect(iqm_client_mock._iqm_server_client, times=1).get_static_quantum_architectures().thenReturn(
        [sample_static_architecture]
    )

    assert iqm_client_mock.get_feedback_groups() == (frozenset({"QB1", "QB2"}), frozenset({"QB3"}))

    verifyNoUnwantedInteractions()


def test_user_warning_is_emitted_when_errors_in_response(
    iqm_client_mock,
    existing_job_id,
):
    """Test that a warning is emitted when errors are present in the response"""
    job_data = JobData(
        id=existing_job_id,
        status=JobStatus.FAILED,
        timeline=[],
        errors=[
            JobError(
                source="iqm-server",
                message="Some job error",
                error_code=None,
            )
        ],
    )

    expect(iqm_client_mock._iqm_server_client, times=1).get_job(existing_job_id).thenReturn(job_data)
    with pytest.warns(UserWarning, match="Some job error"):
        iqm_client_mock.get_job(existing_job_id)

    verifyNoUnwantedInteractions()


def test_user_warning_is_emitted_when_messages_in_response(
    iqm_client_mock,
    existing_job_id,
):
    """Test that a warning is emitted when messages are present in the response"""
    job_data = JobData(
        id=existing_job_id,
        status=JobStatus.FAILED,
        timeline=[],
        messages=[
            JobMessage(
                source="iqm-server",
                message="Some job message",
            )
        ],
    )

    expect(iqm_client_mock._iqm_server_client, times=1).get_job(existing_job_id).thenReturn(job_data)
    with pytest.warns(UserWarning, match="Some job message"):
        iqm_client_mock.get_job(existing_job_id)

    verifyNoUnwantedInteractions()


def test_base_url_is_invalid():
    """Test that an exception is raised when the base URL is invalid"""
    invalid_base_url = "xyz://example.com"
    with pytest.raises(ClientConfigurationError) as exc:
        IQMClient(invalid_base_url)
    assert f"The URL schema has to be http or https. Incorrect schema in URL: {invalid_base_url}" == str(exc.value)


def test_submit_circuits_validates_circuits(iqm_client_mock, sample_circuit):
    """
    Tests that <submit_circuits> validates the batch of provided circuits
    before submitting them for execution
    """
    invalid_circuit = copy(sample_circuit)
    invalid_circuit.name = ""  # Invalidate the circuit on purpose
    with pytest.raises(CircuitValidationError, match="The circuit at index 1 failed the validation"):
        iqm_client_mock.submit_circuits(circuits=[sample_circuit, invalid_circuit], shots=10)


def test_validate_circuit_accepts_valid_circuit(sample_circuit):
    """
    Tests that ``validate_circuit`` accepts a valid circuit.
    """
    validate_circuit(sample_circuit)


def test_validate_circuit_rejects_invalid_circuit(sample_circuit):
    """
    Tests that ``validate_circuit`` rejects an invalid circuit.
    """
    circuit = copy(sample_circuit)
    expect(circuit, times=1).validate(ANY).thenRaise(ValueError("invalid circuit"))
    with pytest.raises(ValueError, match="invalid circuit"):
        validate_circuit(circuit)


def test_cancel_job_successful(iqm_client_mock, existing_job_url, existing_job_id):
    """
    Tests canceling a job
    """
    expect(iqm_client_mock._iqm_server_client, times=1).cancel_job(existing_job_id).thenReturn(None)
    iqm_client_mock.cancel_job(existing_job_id)

    verifyNoUnwantedInteractions()


@pytest.mark.parametrize("status_code", [404, 409])
def test_cancel_job_failed(status_code, iqm_client_mock, existing_job_url, existing_job_id):
    """
    Tests canceling a job raises StationControlError if server returned error response
    """
    expect(iqm_client_mock._iqm_server_client, times=1).cancel_job(existing_job_id).thenRaise(
        InternalServerError("Internal Server Error.")
    )

    with pytest.raises(InternalServerError):
        iqm_client_mock.cancel_job(existing_job_id)

    verifyNoUnwantedInteractions()


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"options": CircuitCompilationOptions(heralding_mode=HeraldingMode.ZEROS, active_reset_cycles=1)},
        {"calibration_set_id": uuid.uuid4()},
        {"options": CircuitCompilationOptions(max_circuit_duration_over_t2=0.0)},
        {"qubit_mapping": {"QB1": "QB2", "QB2": "QB1"}},
    ],
)
def test_create_and_submit_run_request(
    iqm_client_mock,
    sample_circuit,
    existing_job_id,
    sample_dynamic_architecture,
    sample_calibration_set_iqm_server,
    params,
):
    """
    Tests that calling create_run_request and then submit_run_request is equivalent to calling submit_circuits.
    """
    if "calibration_set_id" in params:
        calset_id = params["calibration_set_id"]
    else:
        calset_id = "default"

    expect(iqm_client_mock._iqm_server_client).get_dynamic_quantum_architecture(calset_id).thenReturn(
        sample_dynamic_architecture
    )

    run_request = iqm_client_mock.create_run_request([sample_circuit], **params)
    if "options" in params:
        assert run_request.active_reset_cycles == params["options"].active_reset_cycles
    expect(iqm_client_mock._iqm_server_client, times=2).submit_circuits(run_request, use_timeslot=False).thenReturn(
        sample_job_data(existing_job_id)
    )
    assert iqm_client_mock.submit_run_request(run_request).job_id == existing_job_id
    assert iqm_client_mock.submit_circuits([sample_circuit], **params).job_id == existing_job_id

    verifyNoUnwantedInteractions()


@pytest.mark.parametrize(
    "run_request_name, quantum_architecture_name, sample_circuit_name",
    [
        (run_request, success_result, sample_circuit)
        for run_request in [
            "run_request_with_move_validation",
            "run_request_without_prx_move_validation",
            "run_request_with_move_gate_frame_tracking",
        ]
        for success_result, sample_circuit in zip(
            ["sample_dynamic_architecture", "sample_move_architecture", "sample_move_architecture"],
            ["sample_circuit", "move_circuit", "move_circuit_with_prx_in_the_sandwich"],
        )
    ],
)
def test_compiler_options_are_used_and_sent(
    iqm_client_mock,
    sample_circuit_name,
    run_request_name,
    request,
    quantum_architecture_name,
    existing_job_id,
    sample_calset_id,
):
    """
    Tests submitting circuits for execution
    """
    run_request = request.getfixturevalue(run_request_name)
    dynamic_quantum_architecture_result = request.getfixturevalue(quantum_architecture_name)
    run_request.circuits = [request.getfixturevalue(sample_circuit_name)]

    expect(iqm_client_mock._iqm_server_client, times=1).get_dynamic_quantum_architecture(sample_calset_id).thenReturn(
        dynamic_quantum_architecture_result
    )

    if (
        sample_circuit_name != "move_circuit_with_prx_in_the_sandwich"  # Valid circuit
        or run_request_name == "run_request_without_prx_move_validation"  # Validation is turned off
    ):
        expect(iqm_client_mock._iqm_server_client, times=1).submit_circuits(run_request, use_timeslot=False).thenReturn(
            sample_job_data(existing_job_id)
        )
        iqm_client_mock.submit_circuits(**submit_circuits_args(run_request))
    else:  # Invalid circuit and validation is turned on.
        with pytest.raises(CircuitValidationError):
            iqm_client_mock.submit_circuits(**submit_circuits_args(run_request))

    verifyNoUnwantedInteractions()


def test_get_dynamic_quantum_architecture_without_calset_id(
    iqm_client_mock, mock_default_dynamic_architecture_retrieval, sample_dynamic_architecture
):
    """Tests that the correct dynamic quantum architecture for the default calibration set is returned."""
    assert iqm_client_mock.get_dynamic_quantum_architecture() == sample_dynamic_architecture
    verifyNoUnwantedInteractions()


def test_get_dynamic_quantum_architecture_with_calset_id(
    iqm_client_mock, mock_dynamic_architecture_retrieval, sample_calset_id, sample_dynamic_architecture
):
    """Tests that the correct dynamic quantum architecture for the default calibration set is returned."""
    assert iqm_client_mock.get_dynamic_quantum_architecture(sample_calset_id) == sample_dynamic_architecture
    verifyNoUnwantedInteractions()


def test_get_dynamic_quantum_architecture_with_calset_id_caches(
    iqm_client_mock,
    sample_calset_id,
    sample_dynamic_architecture,
):
    """
    Tests that cached dynamic quantum architecture is returned when requesting it for the second time for
    a given calibration set id.
    """
    expect(iqm_client_mock._iqm_server_client, times=1).get_dynamic_quantum_architecture(sample_calset_id).thenReturn(
        sample_dynamic_architecture
    )

    # make sure cache is empty initially
    assert iqm_client_mock._dynamic_quantum_architectures == {}
    # first call to get_dynamic_quantum_architecture
    assert iqm_client_mock.get_dynamic_quantum_architecture(sample_calset_id) == sample_dynamic_architecture
    assert len(iqm_client_mock._dynamic_quantum_architectures) == 1
    # second call to get_dynamic_quantum_architecture
    assert iqm_client_mock.get_dynamic_quantum_architecture(sample_calset_id) == sample_dynamic_architecture
    # make sure cached architecture is correct
    assert iqm_client_mock._dynamic_quantum_architectures[sample_calset_id] == sample_dynamic_architecture
    # second call to get_dynamic_quantum_architecture
    assert iqm_client_mock.get_dynamic_quantum_architecture(sample_calset_id) == sample_dynamic_architecture
    assert len(iqm_client_mock._dynamic_quantum_architectures) == 1

    verifyNoUnwantedInteractions()


def test_get_dynamic_quantum_architecture_without_calset_id_does_not_cache(
    iqm_client_mock,
    sample_dynamic_architecture,
    sample_dynamic_architecture_2,
):
    """
    Tests that the correct dynamic quantum architecture is returned in the case where default calset
    changes between two invocations of get_dynamic_quantum_architecture().
    """
    expect(iqm_client_mock._iqm_server_client, times=2).get_dynamic_quantum_architecture("default").thenReturn(
        sample_dynamic_architecture
    ).thenReturn(sample_dynamic_architecture_2)

    dynamic_quantum_architecture_1 = iqm_client_mock.get_dynamic_quantum_architecture()
    dynamic_quantum_architecture_2 = iqm_client_mock.get_dynamic_quantum_architecture()

    assert dynamic_quantum_architecture_1 == sample_dynamic_architecture
    assert dynamic_quantum_architecture_2 == sample_dynamic_architecture_2

    verifyNoUnwantedInteractions()


def test_get_run_counts(iqm_client_mock, existing_job_id, base_url):
    """Test that the number of runs is returned."""
    expect(iqm_client_mock._iqm_server_client, times=1).get_job_artifact_measurement_counts(existing_job_id).thenReturn(
        CircuitMeasurementCountsBatch(
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
        )
    )
    counts = iqm_client_mock.get_job_measurement_counts(existing_job_id)
    assert counts == [
        CircuitMeasurementCounts(measurement_keys=["m1"], counts={"0": 5, "1": 5}),
        CircuitMeasurementCounts(measurement_keys=["m2"], counts={"0": 1, "1": 9}),
    ]
    verifyNoUnwantedInteractions()


def test_get_calibration_set_with_calset_id(
    iqm_client_mock, sample_calibration_set_iqm_server, sample_calset_id, mock_calset_retrieval
):
    """Tests that the correct calibration set for the given ``calibration_set_id`` is returned."""
    assert iqm_client_mock.get_calibration_set(sample_calset_id) == sample_calibration_set_iqm_server

    verifyNoUnwantedInteractions()


def test_get_calibration_set_without_calset_id(
    iqm_client_mock,
    mock_default_calset_retrieval,
    sample_calibration_set_iqm_server,
):
    """Tests that the correct calibration set for the default calibration set is returned."""
    assert iqm_client_mock.get_calibration_set() == sample_calibration_set_iqm_server
    verifyNoUnwantedInteractions()


def test_get_static_quantum_architecture(
    iqm_client_mock,
    mock_static_architecture_retrieval,
):
    client = iqm_client_mock
    static_quantum_architecture = client.get_static_quantum_architecture()

    assert static_quantum_architecture.qubits == ["QB1", "QB2", "QB3"]
    assert static_quantum_architecture.computational_resonators == ["CR1", "CR2"]
    assert static_quantum_architecture.connectivity == [
        ("QB1", "CR1"),
        ("QB2", "CR2"),
        ("QB1", "QB3"),
        ("QB3", "CR1"),
        ("QB3", "CR2"),
    ]

    verifyNoUnwantedInteractions()


def test_get_static_quantum_architecture_caches(
    iqm_client_mock,
    mock_static_architecture_retrieval,
    sample_static_architecture,
):
    """
    Tests that cached static quantum architecture is returned when requesting it for the second time for
    a given calibration set id.
    """
    assert iqm_client_mock.get_static_quantum_architecture() == sample_static_architecture
    assert iqm_client_mock.get_static_quantum_architecture() == sample_static_architecture

    verifyNoUnwantedInteractions()


def test_get_parsed_calibration_and_metrics(
    iqm_client_mock,
    crystal_5_calibration_set,
    crystal_5_quality_metrics,
):
    """Test client returns structured calibration and metric data in an ObservationFinder object."""
    iqm_server_client = iqm_client_mock._iqm_server_client
    calset_id = uuid.UUID("6a024885-40d8-48b2-bf9d-b52dddf6753b")

    when(iqm_server_client).get_calibration_set(calset_id).thenReturn(crystal_5_calibration_set)
    when(iqm_server_client).get_calibration_set_quality_metric_set(calset_id).thenReturn(crystal_5_quality_metrics)

    structured_data = iqm_client_mock._get_calibration_quality_metrics(calset_id)

    assert set(["gates", "characterization", "metrics", "controllers"]) <= set(structured_data.keys())

    verifyNoUnwantedInteractions()


def make_token(token_type: str, lifetime: int) -> str:
    """Encode given token type and expire time as a token.

    Args:
        token_type: 'Bearer' for access tokens, 'Refresh' for refresh tokens
        lifetime: seconds from current time to token's expire time

    Returns:
        Encoded token
    """
    empty = b64encode(b"{}").decode("utf-8")
    body = f'{{ "typ": "{token_type}", "exp": {int(time.time()) + lifetime} }}'
    body = b64encode(body.encode("utf-8")).decode("utf-8")
    return f"{empty}.{body}.{empty}"


@pytest.fixture(scope="function")
def iqm_client_mock_with_token(base_url) -> tuple[str, IQMClient]:
    token = make_token("Bearer", 300)
    client = IQMClient(base_url, token=token)
    return token, client


def test_submit_circuits_gets_token(
    monkeypatch,
    iqm_client_mock_with_token,
    jobs_url,
    sample_dynamic_architecture,
    sample_calibration_set_sc,
    sample_circuit,
):
    """Test that submit_circuits gets bearer token from TokenManager"""
    _patch_env(monkeypatch.setenv)
    token, iqm_client = iqm_client_mock_with_token

    # create mock response to get dynamic quantum architecture from IQM Server
    expect(iqm_client._iqm_server_client, times=1).get_dynamic_quantum_architecture("default").thenReturn(
        sample_dynamic_architecture
    )
    # create mock response for POST request to JOB_SUBMIT endpoint
    expect(requests, times=1).post(
        f"{jobs_url}/default/circuit",
        headers={
            "User-Agent": iqm_client._iqm_server_client._signature,
            "Content-Type": "application/json; charset=UTF-8",
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
        },
        params={"use_timeslot": "false"},
        data=ANY,
        timeout=REQUESTS_TIMEOUT,
    ).thenReturn(
        MockJsonResponse(
            HTTPStatus.OK, json_data=sample_job_data(uuid.uuid4(), JobStatus.WAITING).model_dump(mode="json")
        )
    )

    assert isinstance(iqm_client.submit_circuits(circuits=[sample_circuit], shots=10), CircuitJob)

    verifyNoUnwantedInteractions()
    unstub()


def test_get_job_gets_token(monkeypatch, iqm_client_mock_with_token, jobs_url):
    """Test that get_job gets bearer token from TokenManager"""
    _patch_env(monkeypatch.setenv)
    token, iqm_client = iqm_client_mock_with_token
    job_id = uuid.uuid4()

    expect(requests, times=1).get(
        f"{jobs_url}/{str(job_id)}",
        headers={
            "User-Agent": iqm_client._iqm_server_client._signature,
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
        },
        timeout=REQUESTS_TIMEOUT,
    ).thenReturn(
        MockJsonResponse(HTTPStatus.OK, json_data=sample_job_data(job_id, JobStatus.WAITING).model_dump(mode="json"))
    )

    assert isinstance(iqm_client.get_job(job_id), CircuitJob)

    verifyNoUnwantedInteractions()
    unstub()


def test_cancel_job_gets_token(monkeypatch, iqm_client_mock_with_token, jobs_url):
    """Test that cancel_job gets bearer token from TokenManager"""
    _patch_env(monkeypatch.setenv)
    token, iqm_client = iqm_client_mock_with_token
    job_id = uuid.uuid4()

    expect(requests, times=1).post(
        f"{jobs_url}/{str(job_id)}/cancel",
        headers={
            "User-Agent": iqm_client._iqm_server_client._signature,
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
        },
        timeout=REQUESTS_TIMEOUT,
    ).thenReturn(MockJsonResponse(HTTPStatus.NO_CONTENT, json_data={"id": str(uuid.uuid4())}))

    iqm_client.cancel_job(job_id)

    verifyNoUnwantedInteractions()
    unstub()


def test_debug_info(base_url, monkeypatch, capsys):
    """Test the debug_info method."""
    monkeypatch.setenv("IQM_TOKEN", "a" * 20)
    client = IQMClient(base_url)
    about = {
        "iqm_server": True,
        "qccsw_version": "x.y.z-abcd",
        "server_version": "000",
        "station_control_version": "111",
    }
    expect(client._iqm_server_client, times=1).get_about().thenReturn(about)
    expect(client._iqm_server_client, times=1).get_about_station().thenReturn(
        {
            "version": "111",
            "software_versions": {
                "iqm-xxx": "222",
                "other_pkg": "333",
            },
        }
    )

    client._debug_info()
    captured = capsys.readouterr()

    assert len(captured.out) > 200
    assert "platform.platform" in captured.out
    assert "root_url" in captured.out
    assert "quantum_computer" in captured.out
    assert "about" in captured.out
    assert "local packages" in captured.out
