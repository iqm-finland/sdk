# Copyright 2022-2024 Qiskit on IQM developers
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

"""Testing IQMBackend."""

from collections.abc import Sequence
import copy
import re
import uuid

from iqm.iqm_client import (
    CircuitCompilationOptions,
    CircuitJob,
    IQMClient,
)
from iqm.iqm_server_client.models import JobData, JobStatus
from iqm.qiskit_iqm.iqm_provider import IQMBackend, IQMJob
from mockito import ANY, expect, mock, unstub, verifyNoUnwantedInteractions, when
import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter

from iqm.station_control.interface.models import DynamicQuantumArchitecture, HeraldingMode, RunRequest

from .conftest import get_mocked_backend  # , create_metrics_from_dqa

pytestmark = pytest.mark.usefixtures("unstub")


@pytest.fixture
def circuit_job(job_id, backend):
    """A mock recently submitted job."""
    return CircuitJob(_iqm_client=backend.client, data=JobData(id=job_id, status=JobStatus.WAITING))


@pytest.fixture
def backend(linear_3q_architecture, request):
    return get_mocked_backend(linear_3q_architecture, request)


@pytest.fixture
def circuit():
    return QuantumCircuit(3, 3)


@pytest.fixture
def circuit_2() -> QuantumCircuit:
    circuit = QuantumCircuit(5)
    circuit.cz(0, 1)
    return circuit


@pytest.fixture
def create_run_request_default_kwargs(linear_3q_architecture) -> dict:
    return {
        "calibration_set_id": linear_3q_architecture.calibration_set_id,
        "shots": 1024,
        "options": ANY,
    }


@pytest.fixture
def job_id():
    return uuid.uuid4()


@pytest.fixture
def run_request():
    run_request = mock(RunRequest)
    run_request.circuits = []
    run_request.shots = 1
    return run_request


def test_default_options(backend):
    """Test that there are no default options set. The user specifies defaults through the function calls."""
    assert len(backend.options) == 0


def test_backend_name(backend):
    assert re.match(r"IQM(.*)Backend", backend.name)


def test_retrieve_job(backend, job_id, circuit_job):
    when(backend.client).get_job(...).thenReturn(circuit_job)
    job = backend.retrieve_job(str(job_id))
    assert job.backend() == backend
    assert job.job_id() == str(job_id)


def test_default_max_circuits(backend):
    assert backend.max_circuits is None


def test_set_max_circuits(backend):
    assert backend.max_circuits is None

    backend.max_circuits = 17
    assert backend.max_circuits == 17

    backend.max_circuits = 168
    assert backend.max_circuits == 168


def test_run_non_native_circuit(backend, circuit, job_id, run_request, circuit_job):
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(0, 2)

    when(backend.client).create_run_request(...).thenReturn(run_request)
    when(backend.client).submit_run_request(run_request, use_timeslot=False).thenReturn(circuit_job)
    transpiled_circuit = transpile(circuit, backend, optimization_level=0)
    job = backend.run(transpiled_circuit)
    assert isinstance(job, IQMJob)
    assert job.job_id() == str(job_id)


def test_run_single_circuit(backend, circuit, create_run_request_default_kwargs, job_id, run_request, circuit_job):
    circuit.measure(0, 0)
    circuit_ser = backend.serialize_circuit(circuit)
    kwargs = create_run_request_default_kwargs
    when(backend.client).create_run_request([circuit_ser], **kwargs).thenReturn(run_request)
    when(backend.client).submit_run_request(run_request, use_timeslot=False).thenReturn(circuit_job)
    job = backend.run(circuit)
    assert isinstance(job, IQMJob)
    assert job.job_id() == str(job_id)

    # Should also work if the circuit is passed inside a list
    job = backend.run([circuit])
    assert isinstance(job, IQMJob)
    assert job.job_id() == str(job_id)


def test_run_sets_circuit_metadata_to_the_job(backend, run_request, job_id, circuit_job):
    circuit_1 = QuantumCircuit(3)
    circuit_1.cz(0, 1)
    circuit_1.metadata = {"key1": "value1", "key2": "value2"}
    circuit_2 = QuantumCircuit(3)
    circuit_2.cz(0, 1)
    circuit_2.metadata = {"key1": "value2", "key2": "value1"}
    run_request.circuits = [backend.serialize_circuit(c) for c in [circuit_1, circuit_2]]
    when(backend.client).create_run_request(...).thenReturn(run_request)
    when(backend.client).submit_run_request(run_request, use_timeslot=False).thenReturn(circuit_job)
    job = backend.run([circuit_1, circuit_2], shots=10)
    assert isinstance(job, IQMJob)
    assert job.job_id() == str(job_id)
    assert job.circuit_metadata == [circuit_1.metadata, circuit_2.metadata]


@pytest.mark.parametrize("shots", [13, 978, 1137])
def test_run_with_custom_number_of_shots(
    backend, circuit, create_run_request_default_kwargs, job_id, shots, run_request, circuit_job
):
    circuit.measure(0, 0)
    kwargs = create_run_request_default_kwargs | {"shots": shots}
    when(backend.client).create_run_request(ANY, **kwargs).thenReturn(run_request)
    when(backend.client).submit_run_request(run_request, use_timeslot=False).thenReturn(circuit_job)
    backend.run(circuit, shots=shots)


@pytest.mark.parametrize(
    "calibration_set_id",
    [
        "67e77465-d90e-4839-986e-9270f952b743",
        uuid.UUID("67e77465-d90e-4839-986e-9270f952b743"),
    ],
)
def test_backend_run_with_custom_calibration_set_id(
    linear_3q_architecture,
    circuit,
    create_run_request_default_kwargs,
    job_id,
    calibration_set_id,
    run_request,
    circuit_job,
):
    if not isinstance(calibration_set_id, uuid.UUID):
        expected_id = uuid.UUID(calibration_set_id)
    else:
        expected_id = calibration_set_id

    architecture = linear_3q_architecture.model_copy(deep=True, update={"calibration_set_id": expected_id})
    client = mock(IQMClient)
    when(client).get_dynamic_quantum_architecture(expected_id).thenReturn(architecture)
    # metrics = create_metrics_from_dqa(linear_3q_architecture)
    # when(client)._get_calibration_quality_metrics(ANY).thenReturn(metrics)

    backend = IQMBackend(client, calibration_set_id=calibration_set_id)
    circuit.measure(0, 0)
    circuit_ser = backend.serialize_circuit(circuit)
    kwargs = create_run_request_default_kwargs | {
        "calibration_set_id": expected_id,
    }
    when(backend.client).create_run_request([circuit_ser], **kwargs).thenReturn(run_request)
    when(backend.client).submit_run_request(run_request, use_timeslot=False).thenReturn(circuit_job)

    backend.run([circuit])


def test_run_with_duration_check_disabled(
    backend, circuit, create_run_request_default_kwargs, job_id, run_request, circuit_job
):
    circuit.measure(0, 0)
    circuit_ser = backend.serialize_circuit(circuit)
    options = CircuitCompilationOptions(max_circuit_duration_over_t2=0.0)
    kwargs = create_run_request_default_kwargs | {"options": options}
    when(backend.client).create_run_request([circuit_ser], **kwargs).thenReturn(run_request)
    when(backend.client).submit_run_request(run_request, use_timeslot=False).thenReturn(circuit_job)

    backend.run([circuit], circuit_compilation_options=options)


def test_run_uses_heralding_mode_none_by_default(
    backend, circuit, create_run_request_default_kwargs, job_id, run_request, circuit_job
):
    circuit.measure(0, 0)
    circuit_ser = backend.serialize_circuit(circuit)
    default_compilation_options = CircuitCompilationOptions()
    kwargs = create_run_request_default_kwargs | {
        "options": default_compilation_options,
    }
    when(backend.client).create_run_request([circuit_ser], **kwargs).thenReturn(run_request)
    when(backend.client).submit_run_request(run_request, use_timeslot=False).thenReturn(circuit_job)
    backend.run([circuit])


def test_run_with_heralding_mode_zeros(
    backend, circuit, create_run_request_default_kwargs, job_id, run_request, circuit_job
):
    circuit.measure(0, 0)
    circuit_ser = backend.serialize_circuit(circuit)
    options = CircuitCompilationOptions(heralding_mode=HeraldingMode.ZEROS)
    kwargs = create_run_request_default_kwargs | {
        "options": options,
    }
    when(backend.client).create_run_request([circuit_ser], **kwargs).thenReturn(run_request)
    when(backend.client).submit_run_request(run_request, use_timeslot=False).thenReturn(circuit_job)
    backend.run([circuit], circuit_compilation_options=options)


# mypy: disable-error-code="attr-defined"
def test_run_with_circuit_callback(backend, job_id, create_run_request_default_kwargs, run_request, circuit_job):
    qc1 = QuantumCircuit(3)
    qc1.measure_all()
    qc2 = QuantumCircuit(3)
    qc2.r(np.pi, 0.3, 0)
    qc2.measure_all()

    def sample_callback(circuits) -> None:
        assert isinstance(circuits, Sequence)
        assert all(isinstance(c, QuantumCircuit) for c in circuits)
        assert len(circuits) == 2
        assert circuits[0].name == qc1.name
        assert circuits[1].name == qc2.name
        sample_callback.called = True

    sample_callback.called = False

    kwargs = create_run_request_default_kwargs
    when(backend.client).create_run_request(ANY, **kwargs).thenReturn(run_request)
    when(backend.client).submit_run_request(run_request, use_timeslot=False).thenReturn(circuit_job)
    backend.run([qc1, qc2], circuit_callback=sample_callback)
    assert sample_callback.called is True


def test_run_with_unknown_option(backend, circuit, job_id, run_request, circuit_job):
    circuit.measure_all()
    when(backend.client).create_run_request(...).thenReturn(run_request)
    when(backend.client).submit_run_request(run_request, use_timeslot=False).thenReturn(circuit_job)
    with pytest.warns(Warning, match=r"Unknown backend option\(s\)"):
        backend.run(circuit, to_option_or_not_to_option=17)


def test_run_batch_of_circuits(backend, circuit, create_run_request_default_kwargs, job_id, run_request, circuit_job):
    theta = Parameter("theta")
    theta_range = np.linspace(0, 2 * np.pi, 3)
    circuit.cz(0, 1)
    circuit.r(theta, 0, 0)
    circuit.cz(0, 1)
    circuits = [circuit.assign_parameters({theta: t}) for t in theta_range]
    circuits_serialized = [backend.serialize_circuit(circuit) for circuit in circuits]
    kwargs = create_run_request_default_kwargs
    when(backend.client).create_run_request(circuits_serialized, **kwargs).thenReturn(run_request)
    when(backend.client).submit_run_request(run_request, use_timeslot=False).thenReturn(circuit_job)

    job = backend.run(circuits)
    assert isinstance(job, IQMJob)
    assert job.job_id() == str(job_id)


def test_run_warns_if_default_calset_changed(
    adonis_architecture,
    circuit_2,
    job_id,
    run_request,
    circuit_job,
):
    client = mock(IQMClient)
    new_calset_id = uuid.uuid4()
    new_arch = adonis_architecture.model_copy(deep=True, update={"calibration_set_id": new_calset_id})

    when(client).get_dynamic_quantum_architecture(None).thenReturn(adonis_architecture).thenReturn(new_arch)
    # metrics = create_metrics_from_dqa(adonis_architecture)
    # when(client)._get_calibration_quality_metrics(ANY).thenReturn(metrics)
    when(client).create_run_request(...).thenReturn(run_request)
    when(client).submit_run_request(run_request, use_timeslot=False).thenReturn(circuit_job)

    backend = IQMBackend(client)
    with pytest.warns(
        UserWarning,
        match=f"default calibration set has changed from {adonis_architecture.calibration_set_id} to {new_calset_id}",
    ):
        backend.run(circuit_2)


def test_error_on_empty_circuit_list(backend):
    with pytest.raises(ValueError, match="Empty list of circuits submitted for execution."):
        backend.run([], shots=42)


def test_create_run_request(backend, circuit, create_run_request_default_kwargs, run_request, circuit_job):
    options = {"optimization_level": 0}

    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(0, 2)

    circuit_transpiled = transpile(circuit, backend, **options)
    circuit_serialized = backend.serialize_circuit(circuit_transpiled)
    kwargs = create_run_request_default_kwargs

    # verifies that backend.create_run_request() and backend.run() call client.create_run_request() with same arguments
    expect(backend.client, times=2).create_run_request(
        [circuit_serialized],
        **kwargs,
    ).thenReturn(run_request)
    when(backend.client).submit_run_request(run_request, use_timeslot=False).thenReturn(circuit_job)

    assert backend.create_run_request(circuit_transpiled) == run_request
    backend.run(circuit_transpiled)

    verifyNoUnwantedInteractions()
    unstub()


def test_filter_dqa(backend):
    dqa: DynamicQuantumArchitecture = copy.deepcopy(backend.architecture)
    # add a qubit with a PRX but no "measure"
    original_qubits = dqa.qubits.copy()
    dqa.qubits.append("X")
    original_loci = dqa.gates["prx"].implementations["drag_gaussian"].loci
    dqa.gates["prx"].implementations["drag_gaussian"].loci = original_loci + (("X",),)

    # the added qubit and gate locus are filtered out
    filtered_dqa = backend._filter_dqa(dqa)
    assert filtered_dqa.qubits == original_qubits
    assert filtered_dqa.gates["prx"].implementations["drag_gaussian"].loci == original_loci
