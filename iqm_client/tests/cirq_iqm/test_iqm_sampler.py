# Copyright 2020–2022 Cirq on IQM developers
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
import re
import sys
import uuid
import warnings

import cirq
import iqm.cirq_iqm as module_under_test
from iqm.cirq_iqm import Adonis, IQMDevice, IQMDeviceMetadata
from iqm.cirq_iqm.iqm_gates import IQMMoveGate
from iqm.cirq_iqm.iqm_sampler import IQMResult, IQMSampler, ResultMetadata, serialize_circuit
from iqm.iqm_client import (
    CircuitCompilationOptions,
    CircuitJob,
    CircuitJobParameters,
    CircuitValidationError,
    IQMClient,
)
from iqm.iqm_server_client.models import JobData, JobStatus
from mockito import ANY, expect, mock, verify, verifyNoUnwantedInteractions, when
import numpy as np
import pytest
import sympy  # type: ignore

from exa.common.errors.station_control_errors import InternalServerError
from iqm.pulse import Circuit, CircuitOperation
from iqm.station_control.interface.models import (
    CircuitBatch,
    DynamicQuantumArchitecture,
    GateImplementationInfo,
    GateInfo,
    HeraldingMode,
    RunRequest,
)

pytestmark = pytest.mark.usefixtures("unstub")


@pytest.fixture()
def circuit_physical(adonis_architecture):
    """Circuit with physical qubit names"""
    qubit_1 = cirq.NamedQubit("QB1")
    qubit_2 = cirq.NamedQubit("QB2")
    circuit = cirq.Circuit(cirq.measure(qubit_1, qubit_2, key="result"))
    circuit.iqm_calibration_set_id = adonis_architecture.calibration_set_id
    return circuit


@pytest.fixture()
def circuit_non_physical():
    """Circuit with non-physical qubit names"""
    qubit_1 = cirq.NamedQubit("Alice")
    qubit_2 = cirq.NamedQubit("Bob")
    return cirq.Circuit(cirq.measure(qubit_1, qubit_2, key="result"))


@pytest.fixture()
def iqm_metadata() -> tuple[CircuitBatch, CircuitJobParameters]:
    return [
        Circuit(
            name="circuit_1",
            instructions=(CircuitOperation(name="measure", implementation=None, locus=("QB1",), args={"key": "m1"}),),
        )
    ], CircuitJobParameters(
        shots=4,
    )


@pytest.fixture()
def adonis_sampler(base_url, iqm_client_mock_cirq, adonis_architecture) -> IQMSampler:
    """Gets its architecture from the default calset DQA."""
    when(IQMClient).get_dynamic_quantum_architecture(None).thenReturn(adonis_architecture)
    sampler = IQMSampler(base_url)
    sampler._client = iqm_client_mock_cirq
    return sampler


@pytest.fixture()
def adonis_architecture():
    return DynamicQuantumArchitecture(
        calibration_set_id=uuid.UUID("26c5e70f-bea0-43af-bd37-6212ec7d04cb"),
        qubits=["QB1", "QB2", "QB3", "QB4", "QB5"],
        computational_resonators=[],
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
                        loci=(("QB1", "QB3"), ("QB2", "QB3"), ("QB4", "QB3"), ("QB5", "QB3"))
                    ),
                },
                default_implementation="tgss",
                override_default_implementation={},
            ),
            "measure": GateInfo(
                implementations={
                    "constant": GateImplementationInfo(loci=(("QB1",), ("QB2",), ("QB3",), ("QB4",), ("QB5",))),
                },
                default_implementation="constant",
                override_default_implementation={},
            ),
        },
    )


@pytest.fixture()
def adonis_sampler_from_architecture(base_url, iqm_client_mock_cirq, adonis_architecture):
    when(IQMClient).get_dynamic_quantum_architecture(None).thenReturn(adonis_architecture)
    return IQMSampler(base_url, device=IQMDevice(IQMDeviceMetadata.from_architecture(adonis_architecture)))


@pytest.fixture
def create_run_request_default_kwargs(adonis_architecture) -> dict:
    return {
        "calibration_set_id": adonis_architecture.calibration_set_id,
        "shots": 1,
        "options": CircuitCompilationOptions(),
    }


@pytest.fixture
def job_id():
    return uuid.uuid4()


@pytest.fixture
def run_request():
    run_request = mock(RunRequest)
    return run_request


@pytest.fixture()
def circuit_job(job_id, iqm_client_mock_cirq) -> CircuitJob:
    """Mock a completed circuit job with the result already inside."""
    job = CircuitJob(
        data=JobData(id=job_id, status=JobStatus.COMPLETED),
        _iqm_client=iqm_client_mock_cirq,
        _result=[{"some stuff": [[0], [1]]}],
    )
    when(job).wait_for_completion().thenReturn(JobStatus.COMPLETED)
    return job


@pytest.fixture()
def mock_run_sweep(iqm_client_mock_cirq, create_run_request_default_kwargs, run_request, circuit_job) -> None:
    """Prepares a mocked pathway for adonis_sampler.run_sweep."""
    when(iqm_client_mock_cirq).create_run_request(ANY, **create_run_request_default_kwargs).thenReturn(run_request)
    when(iqm_client_mock_cirq).submit_run_request(run_request, use_timeslot=False).thenReturn(circuit_job)


@pytest.fixture()
def circuit_job_batch(job_id, iqm_client_mock_cirq) -> CircuitJob:
    """Mock a completed circuit job with the result already inside, 2-circuit batch."""
    job = CircuitJob(
        data=JobData(id=job_id, status=JobStatus.COMPLETED),
        _iqm_client=iqm_client_mock_cirq,
        _result=[{"some stuff": [[0]]}, {"some stuff": [[1]]}],
    )
    when(job).wait_for_completion().thenReturn(JobStatus.COMPLETED)
    return job


@pytest.fixture()
def mock_run_sweep_batch(
    iqm_client_mock_cirq, create_run_request_default_kwargs, run_request, circuit_job_batch
) -> None:
    """Prepares a mocked pathway for adonis_sampler.run_sweep."""
    when(iqm_client_mock_cirq).create_run_request(ANY, **create_run_request_default_kwargs).thenReturn(run_request)
    when(iqm_client_mock_cirq).submit_run_request(run_request, use_timeslot=False).thenReturn(circuit_job_batch)


def test_init_default(base_url, iqm_client_mock_cirq, adonis_architecture):
    when(IQMClient).get_dynamic_quantum_architecture(None).thenReturn(adonis_architecture)
    sampler = IQMSampler(base_url)
    assert sampler.device == IQMDevice(IQMDeviceMetadata.from_architecture(adonis_architecture))
    assert sampler._calibration_set_id == adonis_architecture.calibration_set_id


def test_init_with_calset_id(base_url, iqm_client_mock_cirq, adonis_architecture):
    calset_id = adonis_architecture.calibration_set_id
    when(IQMClient).get_dynamic_quantum_architecture(calset_id).thenReturn(adonis_architecture)
    sampler = IQMSampler(base_url, calibration_set_id=calset_id)
    assert sampler.device == IQMDevice(IQMDeviceMetadata.from_architecture(adonis_architecture))
    assert sampler._calibration_set_id == calset_id


def test_init_with_calset_id_and_device(base_url, iqm_client_mock_cirq, adonis_architecture):
    calset_id = adonis_architecture.calibration_set_id
    when(IQMClient).get_dynamic_quantum_architecture(calset_id).thenReturn(adonis_architecture)
    sampler = IQMSampler(base_url, device=Adonis(), calibration_set_id=calset_id)
    assert sampler.device.metadata == IQMDeviceMetadata.from_architecture(adonis_architecture)
    assert sampler._calibration_set_id == calset_id


def test_init_warns_if_device_not_compatible_with_default_calset(
    base_url, iqm_client_mock_cirq, fake_arch_with_resonator
):
    when(IQMClient).get_dynamic_quantum_architecture(None).thenReturn(fake_arch_with_resonator)
    with pytest.raises(
        ValueError,
        match="'device' is not compatible with the server default calibration set "
        f"{fake_arch_with_resonator.calibration_set_id}",
    ):
        IQMSampler(base_url, device=Adonis())


def test_init_warns_if_device_not_compatible_with_calset_id(base_url, iqm_client_mock_cirq, fake_arch_with_resonator):
    calset_id = uuid.uuid4()
    when(IQMClient).get_dynamic_quantum_architecture(calset_id).thenReturn(fake_arch_with_resonator)
    with pytest.raises(ValueError, match=f"'device' is not compatible with calibration set {calset_id}"):
        IQMSampler(base_url, device=Adonis(), calibration_set_id=calset_id)


def test_run_sweep_raises_with_non_physical_names(adonis_sampler_from_architecture, circuit_non_physical):
    sampler = adonis_sampler_from_architecture
    when(sampler._client).get_dynamic_quantum_architecture(ANY).thenReturn(sampler.device.metadata.architecture)
    # Note that validation is done in iqm_client, so this is now an integration test.
    with pytest.raises(CircuitValidationError, match="Alice is not allowed as locus for 'measure'"):
        sampler.run_sweep(circuit_non_physical, None)


def test_run_sweep_executes_circuit_with_physical_names(
    adonis_sampler,
    circuit_physical,
    mock_run_sweep,
):
    results = adonis_sampler.run_sweep(circuit_physical, None)
    assert isinstance(results[0], IQMResult)
    assert isinstance(results[0].metadata, ResultMetadata)
    np.testing.assert_array_equal(results[0].measurements["some stuff"], np.array([[0], [1]]))


def test_run_sweep_executes_circuit_with_calibration_set_id(
    base_url,
    iqm_client_mock_cirq,
    adonis_architecture,
    circuit_physical,
    mock_run_sweep,
):
    calibration_set_id = adonis_architecture.calibration_set_id
    when(IQMClient).get_dynamic_quantum_architecture(calibration_set_id).thenReturn(adonis_architecture)
    sampler = IQMSampler(base_url, calibration_set_id=calibration_set_id)
    sampler._client = iqm_client_mock_cirq

    results = sampler.run_sweep(circuit_physical, None)
    assert isinstance(results[0], IQMResult)
    assert isinstance(results[0].metadata, ResultMetadata)
    np.testing.assert_array_equal(results[0].measurements["some stuff"], np.array([[0], [1]]))


def test_run_sweep_has_duration_check_enabled_by_default(
    adonis_sampler,
):
    assert adonis_sampler._compiler_options.max_circuit_duration_over_t2 is None


def test_run_sweep_executes_circuit_with_duration_check_disabled(
    base_url,
    iqm_client_mock_cirq,
    adonis_architecture,
    circuit_physical,
    create_run_request_default_kwargs,
    circuit_job,
    run_request,
):
    when(IQMClient).get_dynamic_quantum_architecture(None).thenReturn(adonis_architecture)
    sampler = IQMSampler(
        base_url, device=Adonis(), compiler_options=CircuitCompilationOptions(max_circuit_duration_over_t2=0.0)
    )
    sampler._client = iqm_client_mock_cirq
    assert sampler._compiler_options.max_circuit_duration_over_t2 == 0.0

    kwargs = create_run_request_default_kwargs | {
        "options": CircuitCompilationOptions(max_circuit_duration_over_t2=0.0)
    }
    when(iqm_client_mock_cirq).create_run_request(ANY, **kwargs).thenReturn(run_request)
    when(iqm_client_mock_cirq).submit_run_request(run_request, use_timeslot=False).thenReturn(circuit_job)

    results = sampler.run_sweep(circuit_physical, None)
    assert isinstance(results[0], IQMResult)
    assert isinstance(results[0].metadata, ResultMetadata)
    np.testing.assert_array_equal(results[0].measurements["some stuff"], np.array([[0], [1]]))


def test_run_sweep_allows_to_override_polling_timeout(
    base_url,
    iqm_client_mock_cirq,
    adonis_architecture,
    circuit_physical,
    create_run_request_default_kwargs,
    job_id,
    run_request,
):
    timeout = 123
    when(IQMClient).get_dynamic_quantum_architecture(None).thenReturn(adonis_architecture)
    sampler = IQMSampler(base_url, device=Adonis(), run_sweep_timeout=timeout)
    sampler._client = iqm_client_mock_cirq

    when(iqm_client_mock_cirq).get_dynamic_quantum_architecture(None).thenReturn(adonis_architecture)
    when(iqm_client_mock_cirq).create_run_request(ANY, **create_run_request_default_kwargs).thenReturn(run_request)
    circuit_job = CircuitJob(
        data=JobData(id=job_id, status=JobStatus.COMPLETED),
        _iqm_client=iqm_client_mock_cirq,
        _result=[{"some stuff": [[0], [1]]}],
    )
    when(circuit_job).wait_for_completion(timeout_secs=timeout).thenReturn(JobStatus.COMPLETED)
    when(iqm_client_mock_cirq).submit_run_request(run_request, use_timeslot=False).thenReturn(circuit_job)

    results = sampler.run_sweep(circuit_physical, None)
    assert isinstance(results[0], IQMResult)
    assert isinstance(results[0].metadata, ResultMetadata)
    np.testing.assert_array_equal(results[0].measurements["some stuff"], np.array([[0], [1]]))


def test_run_sweep_has_heralding_mode_none_by_default(adonis_sampler):
    assert adonis_sampler._compiler_options.heralding_mode == HeraldingMode.NONE


def test_run_sweep_executes_circuit_with_heralding_mode_zeros(
    base_url,
    iqm_client_mock_cirq,
    adonis_architecture,
    circuit_physical,
    create_run_request_default_kwargs,
    run_request,
    circuit_job,
):
    when(IQMClient).get_dynamic_quantum_architecture(None).thenReturn(adonis_architecture)
    sampler = IQMSampler(
        base_url, device=Adonis(), compiler_options=CircuitCompilationOptions(heralding_mode=HeraldingMode.ZEROS)
    )
    sampler._client = iqm_client_mock_cirq
    assert sampler._compiler_options.heralding_mode == HeraldingMode.ZEROS

    kwargs = create_run_request_default_kwargs | {
        "options": CircuitCompilationOptions(heralding_mode=HeraldingMode.ZEROS)
    }
    when(iqm_client_mock_cirq).create_run_request(ANY, **kwargs).thenReturn(run_request)
    when(iqm_client_mock_cirq).submit_run_request(run_request, use_timeslot=False).thenReturn(circuit_job)

    results = sampler.run_sweep(circuit_physical, None)
    assert isinstance(results[0], IQMResult)
    assert isinstance(results[0].metadata, ResultMetadata)
    np.testing.assert_array_equal(results[0].measurements["some stuff"], np.array([[0], [1]]))


def test_run_sweep_with_parameter_sweep(
    adonis_sampler,
    mock_run_sweep_batch,
):
    qubit_1 = cirq.NamedQubit("QB1")
    qubit_2 = cirq.NamedQubit("QB2")
    circuit_sweep = cirq.Circuit(cirq.X(qubit_1) ** sympy.Symbol("t"), cirq.measure(qubit_1, qubit_2, key="result"))

    sweep_length = 2
    param_sweep = cirq.Linspace("t", start=0, stop=1, length=sweep_length)

    results = adonis_sampler.run_sweep(circuit_sweep, param_sweep)
    assert len(results) == sweep_length
    assert all(isinstance(result, IQMResult) for result in results)
    assert all(isinstance(result.metadata, ResultMetadata) for result in results)
    np.testing.assert_array_equal(results[0].measurements["some stuff"], np.array([[0]]))
    np.testing.assert_array_equal(results[1].measurements["some stuff"], np.array([[1]]))
    for idx, param in enumerate(param_sweep):
        assert results[idx].params == param


def test_run_sweep_cancel_job_successful(
    adonis_sampler,
    circuit_physical,
    job_id,
    run_request,
    recwarn,
):
    client = adonis_sampler._client
    circuit_job = CircuitJob(
        data=JobData(id=job_id, status=JobStatus.CANCELLED),
        _iqm_client=client,
        _result=[{"some stuff": [[0], [1]]}],
    )
    when(client).create_run_request(...).thenReturn(run_request)
    when(client).submit_run_request(run_request, use_timeslot=False).thenReturn(circuit_job)
    when(circuit_job).wait_for_completion().thenRaise(KeyboardInterrupt)
    when(circuit_job).cancel()
    when(sys).exit().thenRaise(NotImplementedError)  # just for testing without actually exiting python
    with pytest.raises(NotImplementedError):
        adonis_sampler.run_sweep(circuit_physical, None)

    assert len(recwarn) == 0
    verify(circuit_job, times=1).cancel()
    verify(sys, times=1).exit()


def test_run_sweep_cancel_job_failed(
    adonis_sampler,
    circuit_physical,
    job_id,
    run_request,
):
    client = adonis_sampler._client
    circuit_job = CircuitJob(
        data=JobData(id=job_id, status=JobStatus.PROCESSING),
        _iqm_client=client,
        _result=[{"some stuff": [[0], [1]]}],
    )
    when(client).create_run_request(...).thenReturn(run_request)
    when(client).submit_run_request(run_request, use_timeslot=False).thenReturn(circuit_job)
    when(circuit_job).wait_for_completion().thenRaise(KeyboardInterrupt)
    when(circuit_job).cancel().thenRaise(InternalServerError("Internal Server Error."))
    when(sys).exit().thenRaise(NotImplementedError)  # just for testing without actually exiting python

    with pytest.warns(UserWarning, match="Failed to cancel job"):
        with pytest.raises(NotImplementedError):
            adonis_sampler.run_sweep(circuit_physical, None)

    verify(circuit_job, times=1).cancel()
    verify(sys, times=1).exit()


def test_run_iqm_batch_raises_with_non_physical_names(adonis_sampler_from_architecture, circuit_non_physical):
    sampler = adonis_sampler_from_architecture
    when(sampler._client).get_dynamic_quantum_architecture(ANY).thenReturn(sampler.device.metadata.architecture)
    # Note that validation is done in iqm_client, so this is now an integration test.
    with pytest.raises(CircuitValidationError, match="Alice is not allowed as locus for 'measure'"):
        sampler.run_iqm_batch([circuit_non_physical])

    verifyNoUnwantedInteractions()


def test_run(
    adonis_sampler,
    create_run_request_default_kwargs,
    run_request,
    circuit_job_batch,
):
    client = adonis_sampler._client
    repetitions = 123
    kwargs = create_run_request_default_kwargs | {"shots": repetitions}
    when(client).create_run_request(ANY, **kwargs).thenReturn(run_request)
    when(client).submit_run_request(run_request, use_timeslot=False).thenReturn(circuit_job_batch)

    qubit_1 = cirq.NamedQubit("QB1")
    qubit_2 = cirq.NamedQubit("QB2")
    circuit1 = cirq.Circuit(cirq.X(qubit_1), cirq.measure(qubit_1, qubit_2, key="result"))

    result = adonis_sampler.run(circuit1, repetitions=repetitions)

    assert isinstance(result, IQMResult)
    assert isinstance(result.metadata, ResultMetadata)
    np.testing.assert_array_equal(result.measurements["some stuff"], np.array([[0]]))


def test_run_ndonis(
    iqm_client_mock_cirq,
    device_with_resonator,
    fake_arch_with_resonator,
    base_url,
    create_run_request_default_kwargs,
    circuit_job_batch,
    run_request,
):
    when(IQMClient).get_dynamic_quantum_architecture(None).thenReturn(fake_arch_with_resonator)
    sampler = IQMSampler(base_url, device=device_with_resonator)
    sampler._client = iqm_client_mock_cirq

    when(iqm_client_mock_cirq).get_dynamic_quantum_architecture(None).thenReturn(fake_arch_with_resonator)
    repetitions = 123
    kwargs = create_run_request_default_kwargs | {
        "shots": repetitions,
        "calibration_set_id": device_with_resonator.metadata.architecture.calibration_set_id,
    }
    when(iqm_client_mock_cirq).create_run_request(ANY, **kwargs).thenReturn(run_request)
    when(iqm_client_mock_cirq).submit_run_request(run_request, use_timeslot=False).thenReturn(circuit_job_batch)

    qubit_1, qubit_2 = device_with_resonator.qubits[:2]
    resonator = device_with_resonator.resonators[0]
    circuit = cirq.Circuit()
    circuit.append(device_with_resonator.decompose_operation(cirq.H(qubit_1)))
    circuit.append(IQMMoveGate().on(qubit_1, resonator))
    circuit.append(device_with_resonator.decompose_operation(cirq.H(qubit_2)))
    circuit.append(cirq.CZ(resonator, qubit_2))
    circuit.append(IQMMoveGate().on(qubit_1, resonator))
    circuit.append(device_with_resonator.decompose_operation(cirq.H(qubit_2)))
    circuit.append(cirq.MeasurementGate(2, key="result").on(qubit_1, qubit_2))

    result = sampler.run(circuit, repetitions=repetitions)

    assert isinstance(result, IQMResult)
    assert isinstance(result.metadata, ResultMetadata)
    np.testing.assert_array_equal(result.measurements["some stuff"], np.array([[0]]))


def test_run_does_not_warn(
    adonis_sampler,
    circuit_physical,
    mock_run_sweep,
):
    routed_circuit, _, _ = adonis_sampler.device.route_circuit(circuit_physical)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        adonis_sampler.run(routed_circuit)


def test_run_warns_if_default_calset_changed(
    base_url,
    iqm_client_mock_cirq,
    adonis_architecture,
    circuit_physical,
    mock_run_sweep,
):
    new_default_architecture = DynamicQuantumArchitecture(
        calibration_set_id=uuid.uuid4(),
        qubits=adonis_architecture.qubits,
        computational_resonators=adonis_architecture.computational_resonators,
        gates=adonis_architecture.gates,
    )
    when(IQMClient).get_dynamic_quantum_architecture(None).thenReturn(adonis_architecture).thenReturn(
        new_default_architecture
    )

    sampler = IQMSampler(base_url)
    sampler._client = iqm_client_mock_cirq
    routed_circuit, _, _ = sampler.device.route_circuit(circuit_physical)

    with pytest.warns(
        UserWarning,
        match=f"default calibration set has changed "
        f"from {adonis_architecture.calibration_set_id} to {new_default_architecture.calibration_set_id}",
    ):
        sampler.run(routed_circuit)


def test_run_warns_if_circuits_routed_with_different_calset_id(
    base_url,
    iqm_client_mock_cirq,
    adonis_architecture,
    circuit_physical,
    create_run_request_default_kwargs,
    run_request,
    circuit_job,
):
    other_calset_id = uuid.uuid4()
    other_architecture = DynamicQuantumArchitecture(
        calibration_set_id=other_calset_id,
        qubits=adonis_architecture.qubits,
        computational_resonators=adonis_architecture.computational_resonators,
        gates=adonis_architecture.gates,
    )
    when(IQMClient).get_dynamic_quantum_architecture(None).thenReturn(adonis_architecture)
    when(IQMClient).get_dynamic_quantum_architecture(other_calset_id).thenReturn(other_architecture)

    kwargs = create_run_request_default_kwargs | {"calibration_set_id": other_calset_id}
    when(iqm_client_mock_cirq).create_run_request(ANY, **kwargs).thenReturn(run_request)
    when(iqm_client_mock_cirq).submit_run_request(run_request, use_timeslot=False).thenReturn(circuit_job)

    sampler = IQMSampler(base_url)
    sampler._client = iqm_client_mock_cirq
    routed_circuit, _, _ = sampler.device.route_circuit(circuit_physical)

    other_sampler = IQMSampler(base_url, calibration_set_id=other_calset_id)
    other_sampler._client = iqm_client_mock_cirq

    with pytest.warns(
        UserWarning,
        match=re.escape(
            f"routed using calibration set(s) {set({adonis_architecture.calibration_set_id})}, "
            f"different than the current calibration set {other_calset_id}"
        ),
    ):
        other_sampler.run(routed_circuit)


def test_run_iqm_batch(
    iqm_client_mock_cirq,
    adonis_sampler,
    adonis_architecture,
    iqm_metadata,
    create_run_request_default_kwargs,
    circuit_job_batch,
    run_request,
):
    repetitions = 123
    kwargs = create_run_request_default_kwargs | {"shots": repetitions}
    when(iqm_client_mock_cirq).create_run_request(ANY, **kwargs).thenReturn(run_request)
    when(iqm_client_mock_cirq).submit_run_request(run_request, use_timeslot=False).thenReturn(circuit_job_batch)

    qubit_1 = cirq.NamedQubit("QB1")
    qubit_2 = cirq.NamedQubit("QB2")
    circuit1 = cirq.Circuit(cirq.X(qubit_1), cirq.measure(qubit_1, qubit_2, key="result"))
    circuit2 = cirq.Circuit(cirq.X(qubit_2), cirq.measure(qubit_1, qubit_2, key="result"))
    circuits = [circuit1, circuit2]

    results = adonis_sampler.run_iqm_batch(circuits, repetitions=repetitions)

    assert len(results) == len(circuits)
    assert all(isinstance(result, IQMResult) for result in results)
    assert all(isinstance(result.metadata, ResultMetadata) for result in results)
    np.testing.assert_array_equal(results[0].measurements["some stuff"], np.array([[0]]))
    np.testing.assert_array_equal(results[1].measurements["some stuff"], np.array([[1]]))


def test_run_iqm_batch_allows_to_override_polling_timeout(
    base_url,
    iqm_client_mock_cirq,
    adonis_architecture,
    job_id,
    create_run_request_default_kwargs,
    run_request,
):
    when(IQMClient).get_dynamic_quantum_architecture(None).thenReturn(adonis_architecture)
    timeout = 123
    sampler = IQMSampler(base_url, run_sweep_timeout=timeout)
    sampler._client = iqm_client_mock_cirq

    circuit_job_batch = CircuitJob(
        data=JobData(id=job_id, status=JobStatus.COMPLETED),
        _iqm_client=iqm_client_mock_cirq,
        _result=[{"some stuff": [[0]]}, {"some stuff": [[1]]}],
    )
    when(circuit_job_batch).wait_for_completion(timeout_secs=timeout).thenReturn(JobStatus.COMPLETED)
    when(iqm_client_mock_cirq).create_run_request(ANY, **create_run_request_default_kwargs).thenReturn(run_request)
    when(iqm_client_mock_cirq).submit_run_request(run_request, use_timeslot=False).thenReturn(circuit_job_batch)

    qubit_1 = cirq.NamedQubit("QB1")
    qubit_2 = cirq.NamedQubit("QB2")
    circuit1 = cirq.Circuit(cirq.X(qubit_1), cirq.measure(qubit_1, qubit_2, key="result"))
    circuit2 = cirq.Circuit(cirq.X(qubit_2), cirq.measure(qubit_1, qubit_2, key="result"))
    circuits = [circuit1, circuit2]

    results = sampler.run_iqm_batch(circuits)
    assert len(results) == len(circuits)
    assert all(isinstance(result, IQMResult) for result in results)
    np.testing.assert_array_equal(results[0].measurements["some stuff"], np.array([[0]]))
    np.testing.assert_array_equal(results[1].measurements["some stuff"], np.array([[1]]))


def test_credentials_are_passed_to_client(iqm_client_mock_cirq, adonis_architecture):
    user_auth_args = {"token": "fake-token"}
    expect(module_under_test.iqm_sampler, times=1).IQMClient(
        "http://url", quantum_computer=None, **user_auth_args
    ).thenReturn(iqm_client_mock_cirq)
    when(iqm_client_mock_cirq).get_dynamic_quantum_architecture(None).thenReturn(adonis_architecture)
    IQMSampler("http://url", device=Adonis(), **user_auth_args)


def test_create_run_request_for_run(
    iqm_client_mock_cirq,
    adonis_sampler,
    adonis_architecture,
    create_run_request_default_kwargs,
    run_request,
    circuit_job,
):
    repetitions = 123
    kwargs = create_run_request_default_kwargs | {"shots": repetitions}

    qubit_1 = cirq.NamedQubit("QB1")
    qubit_2 = cirq.NamedQubit("QB2")
    circuit = cirq.Circuit(cirq.X(qubit_1), cirq.measure(qubit_1, qubit_2, key="result"))

    # verifies that sampler.create_run_request() and sampler.run() call client.create_run_request() with same arguments
    expect(iqm_client_mock_cirq, times=2).create_run_request(
        [serialize_circuit(circuit)],
        **kwargs,
    ).thenReturn(run_request)
    when(iqm_client_mock_cirq).submit_run_request(run_request, use_timeslot=False).thenReturn(circuit_job)

    assert adonis_sampler.create_run_request(circuit, repetitions=repetitions) == run_request
    adonis_sampler.run(circuit, repetitions=repetitions)

    verifyNoUnwantedInteractions()


def test_create_run_request_for_run_iqm_batch(
    iqm_client_mock_cirq,
    adonis_sampler,
    adonis_architecture,
    create_run_request_default_kwargs,
    run_request,
    circuit_job_batch,
):
    repetitions = 123
    kwargs = create_run_request_default_kwargs | {"shots": repetitions}

    qubit_1 = cirq.NamedQubit("QB1")
    qubit_2 = cirq.NamedQubit("QB2")
    circuit1 = cirq.Circuit(cirq.X(qubit_1), cirq.measure(qubit_1, qubit_2, key="result"))
    circuit2 = cirq.Circuit(cirq.X(qubit_2), cirq.measure(qubit_1, qubit_2, key="result"))
    circuits = [circuit1, circuit2]

    # verifies that sampler.create_run_request() and sampler.run_iqm_batch() call client.create_run_request() with
    # same arguments
    expect(iqm_client_mock_cirq, times=2).create_run_request(
        [serialize_circuit(c) for c in circuits],
        **kwargs,
    ).thenReturn(run_request)
    when(iqm_client_mock_cirq).submit_run_request(run_request, use_timeslot=False).thenReturn(circuit_job_batch)

    assert adonis_sampler.create_run_request(circuits, repetitions=repetitions) == run_request
    adonis_sampler.run_iqm_batch(circuits, repetitions=repetitions)

    verifyNoUnwantedInteractions()
