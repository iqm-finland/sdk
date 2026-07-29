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

"""Testing IQMProvider."""

from importlib.metadata import version
import uuid

from iqm.iqm_client import CircuitJob, IQMClient
from iqm.iqm_server_client.models import JobData, JobStatus
from iqm.qiskit_iqm.iqm_provider import IQMBackend, IQMFacadeBackend, IQMProvider
from mockito import mock, verifyNoUnwantedInteractions, when
import pytest
from qiskit import QuantumCircuit

from iqm.station_control.interface.models import RunRequest

pytestmark = pytest.mark.usefixtures("unstub")


@pytest.fixture
def circuit() -> QuantumCircuit:
    circuit = QuantumCircuit(5)
    circuit.cz(0, 1)
    return circuit


@pytest.fixture
def run_request():
    run_request = mock(RunRequest)
    run_request.circuits = []
    run_request.shots = 1
    return run_request


def test_get_backend(base_url, linear_3q_architecture):
    calset_id = linear_3q_architecture.calibration_set_id

    when(IQMClient).get_dynamic_quantum_architecture(calset_id).thenReturn(linear_3q_architecture)

    provider = IQMProvider(base_url)
    backend = provider.get_backend(calibration_set_id=calset_id)

    assert isinstance(backend, IQMBackend)
    assert backend.client._iqm_server_client.root_url == base_url
    assert backend.num_qubits == 3
    assert set(backend.coupling_map.get_edges()) == {(0, 1), (1, 2)}
    assert backend._calibration_set_id == linear_3q_architecture.calibration_set_id

    verifyNoUnwantedInteractions()


def test_client_signature(base_url, adonis_architecture):
    provider = IQMProvider(base_url)

    calset_id = adonis_architecture.calibration_set_id

    when(IQMClient).get_dynamic_quantum_architecture(calset_id).thenReturn(adonis_architecture)

    backend = provider.get_backend(calibration_set_id=calset_id, use_metrics=False)

    version_string = version("iqm-client")

    assert f"iqm-client {version_string}" in backend.client._iqm_server_client._signature

    verifyNoUnwantedInteractions()


def test_get_facade_backend(
    base_url,
    adonis_architecture,
    adonis_static_architecture,
):
    calset_id = adonis_architecture.calibration_set_id

    when(IQMClient).get_dynamic_quantum_architecture(calset_id).thenReturn(adonis_architecture)
    when(IQMClient).get_static_quantum_architecture().thenReturn(adonis_static_architecture)

    provider = IQMProvider(base_url)

    backend = provider.get_backend(name="facade_adonis", calibration_set_id=calset_id, use_metrics=False)

    assert isinstance(backend, IQMFacadeBackend)
    assert backend.client._iqm_server_client.root_url == base_url
    assert backend.num_qubits == 5
    assert set(backend.coupling_map.get_edges()) == set(backend.target.build_coupling_map())

    verifyNoUnwantedInteractions()


def test_get_facade_backend_raises_error_non_matching_architecture(
    base_url,
    linear_3q_architecture,
    linear_3q_static_architecture,
):
    calset_id = linear_3q_architecture.calibration_set_id

    when(IQMClient).get_dynamic_quantum_architecture(calset_id).thenReturn(linear_3q_architecture)
    when(IQMClient).get_static_quantum_architecture().thenReturn(linear_3q_static_architecture)

    provider = IQMProvider(base_url)

    with pytest.raises(
        ValueError, match="Quantum architecture of the server does not match the requested IQMFakeBackend."
    ):
        provider.get_backend(name="facade_adonis", calibration_set_id=calset_id, use_metrics=False)

    verifyNoUnwantedInteractions()


def test_get_facade_backend_raises_error_unknown_name(
    base_url,
    linear_3q_architecture,
    linear_3q_static_architecture,
):
    calset_id = linear_3q_architecture.calibration_set_id

    when(IQMClient).get_dynamic_quantum_architecture(calset_id).thenReturn(linear_3q_architecture)
    when(IQMClient).get_static_quantum_architecture().thenReturn(linear_3q_static_architecture)

    provider = IQMProvider(base_url)

    with pytest.raises(ValueError, match="Unknown facade backend: facade_nope"):
        provider.get_backend(name="facade_nope", calibration_set_id=calset_id, use_metrics=False)

    verifyNoUnwantedInteractions()


def test_facade_backend_raises_error_on_remote_execution_fail(
    base_url, adonis_static_architecture, adonis_architecture, circuit, run_request
):
    job_id = uuid.uuid4()
    job = CircuitJob(_iqm_client=None, data=JobData(id=job_id, status=JobStatus.FAILED))
    when(IQMClient).get_dynamic_quantum_architecture(None).thenReturn(adonis_architecture)
    when(IQMClient).get_static_quantum_architecture().thenReturn(adonis_static_architecture)
    when(IQMClient).create_run_request(...).thenReturn(run_request)
    when(IQMClient).submit_run_request(...).thenReturn(job)

    # mock getting the job results
    when(job).payload().thenReturn((None, None))

    provider = IQMProvider(base_url)
    backend = provider.get_backend(name="facade_adonis", use_metrics=False)

    with pytest.raises(RuntimeError, match="Remote execution did not succeed"):
        backend.run(circuit)

    verifyNoUnwantedInteractions()
