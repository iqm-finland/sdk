# Copyright 2022 Qiskit on IQM developers
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

"""Testing IQMJob."""

from collections import Counter
from datetime import datetime
import uuid
from uuid import UUID

from iqm.iqm_client import (
    DEFAULT_TIMEOUT_SECONDS,
    APITimeoutError,
    CircuitJob,
    CircuitJobParameters,
    IQMClient,
    JobStatus,
)
from iqm.iqm_server_client.iqm_server_client import _IQMServerClient
from iqm.iqm_server_client.models import JobCompilation, JobData, JobError, TimelineEntry
from iqm.qiskit_iqm.iqm_job import IQMJob
from iqm.qiskit_iqm.iqm_provider import IQMBackend
from mockito import ANY, expect, mock, verifyNoUnwantedInteractions, when
import pytest
from qiskit import QuantumCircuit
from qiskit.providers import JobStatus as QJobStatus
from qiskit.result import Counts
from qiskit.result import Result as QiskitResult

from exa.common.errors.station_control_errors import InternalServerError
from iqm.pulse import Circuit, CircuitOperation
from iqm.station_control.interface.models import CircuitBatch, CircuitMeasurementResultsBatch, HeraldingMode

pytestmark = pytest.mark.usefixtures("unstub")


def sample_job_data(
    job_id: UUID,
    status: JobStatus = JobStatus.PROCESSING,
    calset_id: UUID | None = None,
    timeline: list[TimelineEntry] | None = None,
) -> JobData:
    return JobData(
        id=job_id,
        status=status,
        compilation=JobCompilation(calibration_set_id=calset_id) if calset_id else None,
        timeline=timeline or [],
    )


def mock_iqmjob_results(
    job: IQMJob,
    measurements: CircuitMeasurementResultsBatch,
    circuits: CircuitBatch | None = None,
    params: CircuitJobParameters | None = None,
    calset_id: UUID | None = None,
    timeline: list[TimelineEntry] | None = None,
    times_payload: int = 1,
    times_result: int = 1,
) -> None:
    """Mock the results and metadata of a finished IQMJob."""
    uuid = UUID(job.job_id())
    c_job = job._iqm_job
    expect(c_job, times=times_payload).payload().thenReturn((circuits, params))
    # this is the calset_id that was chosen by the compiler, "params" contains the requested one
    expect(c_job._iqm_client._iqm_server_client, times=1)._wait_job_completion(
        uuid, progress_callback=ANY, timeout_secs=DEFAULT_TIMEOUT_SECONDS
    ).thenReturn(sample_job_data(uuid, status=JobStatus.COMPLETED, calset_id=calset_id, timeline=timeline))
    expect(c_job, times=times_result).result().thenReturn(measurements)


@pytest.fixture()
def job(adonis_architecture) -> IQMJob:
    """Mocked IQMJob with an attached backend, in the WAITING state."""
    iqm_client = mock(IQMClient)
    iqm_client._iqm_server_client = mock(_IQMServerClient)
    when(iqm_client).get_dynamic_quantum_architecture(None).thenReturn(adonis_architecture)
    # when(iqm_client)._get_calibration_quality_metrics(ANY).thenReturn(None)
    backend = IQMBackend(iqm_client)
    return IQMJob(backend, CircuitJob(_iqm_client=iqm_client, data=sample_job_data(uuid.uuid4(), JobStatus.WAITING)))


@pytest.fixture()
def iqm_result_no_shots():
    return {"c_2_0_0": [], "c_2_0_1": []}


@pytest.fixture()
def iqm_result_two_registers():
    return {"c_2_0_0": [[1], [0], [1], [0]], "c_2_0_1": [[1], [1], [0], [1]], "d_4_1_2": [[1], [1], [1], [1]]}


@pytest.fixture()
def iqm_metadata() -> tuple[CircuitBatch, CircuitJobParameters]:
    return [
        Circuit(
            name="circuit_1",
            instructions=(CircuitOperation(name="measure", implementation=None, locus=("0",), args={"key": "m1"}),),
            metadata={"a": "b"},
        )
    ], CircuitJobParameters(
        shots=4,
        calibration_set_id=UUID("df124054-f6d8-41f9-b880-8487f90018f9"),
        qubit_mapping={
            "0": "QB1",
            "1": "QB2",
        },
    )


def test_submit_raises(job):
    with pytest.raises(NotImplementedError, match="You should never have to submit jobs by calling this method."):
        job.submit()


def test_cancel_successful(job, recwarn):
    expect(job._iqm_job._iqm_client._iqm_server_client, times=1).cancel_job(UUID(job.job_id())).thenReturn(None)
    assert job.cancel() is True
    assert len(recwarn) == 0
    verifyNoUnwantedInteractions()


def test_cancel_failed(job):
    expect(job._iqm_job._iqm_client._iqm_server_client, times=1).cancel_job(UUID(job.job_id())).thenRaise(
        InternalServerError("Internal Server Error.")
    )
    with pytest.warns(UserWarning, match="Failed to cancel job"):
        assert job.cancel() is False
    verifyNoUnwantedInteractions()


def test_status_for_completed_result(job):
    expect(job._iqm_job, times=1).payload().thenReturn((None, None))
    measurements = ["11", "10", "10"]
    job._iqm_result = [("circuit_1", measurements, Counts(Counter(measurements)))]
    job._iqm_job.data.status = JobStatus.COMPLETED

    assert job.status() == QJobStatus.DONE
    result = job.result()
    assert isinstance(result, QiskitResult)
    assert result.get_memory() == measurements
    verifyNoUnwantedInteractions()


@pytest.mark.parametrize(
    "job_status,qiskit_job_status",
    [
        (JobStatus.WAITING, QJobStatus.QUEUED),
        (JobStatus.PROCESSING, QJobStatus.RUNNING),
        (JobStatus.COMPLETED, QJobStatus.DONE),
        (JobStatus.FAILED, QJobStatus.ERROR),
        (JobStatus.CANCELLED, QJobStatus.CANCELLED),
    ],
)
def test_status(job, job_status: JobStatus, qiskit_job_status: JobStatus):
    """IQMJob.status correctly converts IQM job statuses to Qiskit job statuses."""
    when(job._iqm_job).update().thenReturn(job_status)
    assert job.status() == qiskit_job_status
    assert job._iqm_result is None
    verifyNoUnwantedInteractions()


def test_error_message(job):
    err_msg = "The job failed with this error message"
    when(job._iqm_job).update().thenReturn(JobStatus.FAILED)
    job._iqm_job.data.errors = [JobError(source="iqm-server", message=err_msg, error_code="xxx")]

    assert job.status() == QJobStatus.ERROR
    assert job.error_message() == f"  iqm-server: xxx: {err_msg}"
    verifyNoUnwantedInteractions()


def test_error_message_on_successful_job(job, iqm_metadata):
    when(job._iqm_job).update().thenReturn(JobStatus.COMPLETED)
    job._iqm_job.data.errors = []

    assert job.status() == QJobStatus.DONE
    assert job.error_message() is None
    verifyNoUnwantedInteractions()


def check_result(result: QiskitResult, calset_id: UUID, circuits: CircuitBatch, params: CircuitJobParameters) -> None:
    """Check that ``result`` is as expected."""
    assert isinstance(result, QiskitResult)
    assert result.get_memory() == ["0100 11", "0100 10", "0100 01", "0100 10"]
    assert result.get_counts() == Counts({"0100 11": 1, "0100 10": 2, "0100 01": 1})
    for r in result.results:
        assert r.calibration_set_id == calset_id
        assert r.data.metadata == {"a": "b"}
    assert result.circuits == circuits
    assert result.parameters == params


def test_result(job, iqm_result_two_registers, iqm_metadata):
    """Successful job returns a valid Result."""
    calset_id = UUID("df124054-f6d8-41f9-b880-8487f90018f9")
    mock_iqmjob_results(job, [iqm_result_two_registers], iqm_metadata[0], iqm_metadata[1], calset_id, times_payload=2)
    result = job.result()
    check_result(result, calset_id, *iqm_metadata)

    # Assert that repeated call does not query the client (i.e. works without calling the mocked wait_for_completion)
    # and call to status() does not call any functions from client.
    result = job.result()
    assert isinstance(result, QiskitResult)
    assert job.status() == QJobStatus.DONE
    verifyNoUnwantedInteractions()


def test_result_on_new_job_object(job, iqm_result_two_registers, iqm_metadata):
    """Multiple IQMJob objects referring to the same job return a valid Result."""
    calset_id = UUID("df124054-f6d8-41f9-b880-8487f90018f9")
    # CircuitJob.result is called twice, once for each IQMJob instance.
    mock_iqmjob_results(
        job,
        [iqm_result_two_registers],
        iqm_metadata[0],
        iqm_metadata[1],
        calset_id,
        times_payload=2,
        times_result=2,
    )

    # internal job object
    circuit_job = job._iqm_job
    assert circuit_job.status == JobStatus.WAITING

    # this will call _wait_job_completion on the circuit_job
    result = job.result()
    check_result(result, calset_id, *iqm_metadata)
    assert circuit_job.status == JobStatus.COMPLETED

    # create a new IQMJob with just the initial contents, nothing cached
    new_job = IQMJob(job.backend(), circuit_job)
    assert new_job._iqm_result is None

    # then retrieve the results again using the new object
    result = new_job.result()
    assert isinstance(new_job._iqm_result, list)
    check_result(result, calset_id, *iqm_metadata)
    verifyNoUnwantedInteractions()


def test_result_no_shots(job, iqm_result_no_shots, iqm_metadata):
    mock_iqmjob_results(
        job,
        [iqm_result_no_shots],
        iqm_metadata[0],
        params=CircuitJobParameters(
            shots=4,
            heralding_mode=HeraldingMode.ZEROS,
        ),
    )

    with pytest.warns(UserWarning, match="Received measurement results containing zero shots."):
        result = job.result()

    assert isinstance(result, QiskitResult)
    assert result.get_memory() == []
    assert result.get_counts() == Counts({})
    verifyNoUnwantedInteractions()


def test_result_multiple_circuits(job, iqm_result_two_registers):
    instruction = CircuitOperation(name="measure", locus=("0",), args={"key": "m1"})
    calset_id = UUID("9d75904b-0c93-461f-b1dc-bd200cfad1f1")
    mock_iqmjob_results(
        job,
        [iqm_result_two_registers] * 2,
        [
            Circuit(
                name="circuit_1",
                instructions=(instruction),
                metadata={"a": 0},
            ),
            Circuit(
                name="circuit_2",
                instructions=(instruction),
                metadata={"a": 1},
            ),
        ],
        CircuitJobParameters(
            shots=4,
            calibration_set_id=calset_id,
            qubit_mapping={
                "0": "QB1",
                "1": "QB2",
                "2": "QB3",
            },
        ),
        calset_id=calset_id,
    )

    result = job.result()
    assert isinstance(result, QiskitResult)
    for circuit_idx in range(2):
        assert result.get_memory(circuit_idx) == ["0100 11", "0100 10", "0100 01", "0100 10"]
        assert result.get_counts(circuit_idx) == Counts({"0100 11": 1, "0100 10": 2, "0100 01": 1})
    assert result.get_counts(QuantumCircuit(name="circuit_1")) == Counts({"0100 11": 1, "0100 10": 2, "0100 01": 1})
    assert result.get_counts(QuantumCircuit(name="circuit_2")) == Counts({"0100 11": 1, "0100 10": 2, "0100 01": 1})
    for i, r in enumerate(result.results):
        assert r.calibration_set_id == calset_id
        assert r.data.metadata == {"a": i}
    verifyNoUnwantedInteractions()


def test_result_with_timeline(job, iqm_result_two_registers, iqm_metadata):
    """IQMJob.result also provides metadata."""

    timeline = [
        TimelineEntry(
            source="iqm-server", status=JobStatus.WAITING, timestamp=datetime.fromisoformat("2023-01-02T12:34:56+00:00")
        ),
        TimelineEntry(
            source="iqm-server",
            status=JobStatus.PROCESSING,
            timestamp=datetime.fromisoformat("2023-01-02T12:34:57+00:00"),
        ),
        TimelineEntry(
            source="iqm-server",
            status=JobStatus.COMPLETED,
            timestamp=datetime.fromisoformat("2023-01-02T12:34:58+00:00"),
        ),
    ]
    mock_iqmjob_results(job, [iqm_result_two_registers], iqm_metadata[0], iqm_metadata[1], timeline=timeline)

    result = job.result()
    assert result.circuits == iqm_metadata[0]
    assert result.parameters == iqm_metadata[1]
    assert result.timeline == timeline
    verifyNoUnwantedInteractions()


def test_result_timeout_cancel(job):
    """Job is not ready after timeout cancels it and raises error."""
    when(job._iqm_job).payload().thenReturn((None, None))
    when(job._iqm_job).wait_for_completion(timeout_secs=1).thenReturn(JobStatus.PROCESSING)
    when(job._iqm_job).cancel().thenReturn(None)
    with pytest.raises(APITimeoutError, match="didn't finish in 1 seconds. Cancelled"):
        job.result(timeout=1, cancel_after_timeout=True)
    verifyNoUnwantedInteractions()


def test_result_timeout_cancel_failed(job):
    """Job is not ready after timeout, failing cancel raises error that propagates through."""
    when(job._iqm_job).payload().thenReturn((None, None))
    when(job._iqm_job).wait_for_completion(timeout_secs=1).thenReturn(JobStatus.PROCESSING)
    when(job._iqm_job).cancel().thenRaise(InternalServerError("Internal Server Error."))
    with pytest.raises(InternalServerError, match="Internal Server Error."):
        job.result(timeout=1, cancel_after_timeout=True)
    verifyNoUnwantedInteractions()


def test_result_timeout_error(job):
    """Job is not ready after timeout raises error."""
    when(job._iqm_job).payload().thenReturn((None, None))
    when(job._iqm_job).wait_for_completion(timeout_secs=1).thenReturn(JobStatus.PROCESSING)
    with pytest.raises(APITimeoutError):
        job.result(timeout=1)
    verifyNoUnwantedInteractions()
