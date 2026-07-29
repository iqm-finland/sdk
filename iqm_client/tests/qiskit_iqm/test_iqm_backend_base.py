# Copyright 2022-2023 Qiskit on IQM developers
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

"""Testing IQM backend."""

from iqm.qiskit_iqm.iqm_backend import IQMBackendBase
from iqm.qiskit_iqm.iqm_target import IQMTarget
import pytest
from qiskit.providers import Options

from iqm.station_control.client.qon import ObservationFinder
from iqm.station_control.interface.models import DynamicQuantumArchitecture

from .conftest import mock_iqm_target


class DummyIQMBackend(IQMBackendBase):
    """Dummy implementation for abstract methods of IQMBacked, so that instances can be created
    and the rest of functionality tested."""

    def __init__(
        self,
        architecture: DynamicQuantumArchitecture,
        metrics: ObservationFinder | None = None,
        **kwargs,
    ):
        super().__init__(architecture, metrics=metrics, **kwargs)

    @classmethod
    def _default_options(cls) -> Options:
        return Options()

    @property
    def max_circuits(self) -> int | None:
        return None

    def run(self, run_input, **options): ...


@pytest.fixture
def backend(linear_3q_architecture):
    return DummyIQMBackend(linear_3q_architecture)


def test_backendbase_initialization_linear_3q_architecture(linear_3q_architecture):
    """Test initialization of IQMBackendBase for linear dynamic architecture."""
    backend = DummyIQMBackend(linear_3q_architecture)

    assert backend.name == "IQMBackend"
    assert backend._target is not None
    assert isinstance(backend._target, IQMTarget)

    expected_qb_to_idx = {
        name: idx
        for idx, name in enumerate(linear_3q_architecture.qubits + linear_3q_architecture.computational_resonators)
    }
    assert backend._qb_to_idx == expected_qb_to_idx
    assert backend._coupling_map is not None


def test_backendbase_initialization_move_architecture(move_architecture, qb_to_idx_move_architecture):
    """Test initialization of IQMBackendBase for dynamic architecture that contains MOVE gate (nDonis)."""
    backend = DummyIQMBackend(move_architecture)
    target = backend._target_with_resonators

    assert backend.architecture == move_architecture
    expected_target = mock_iqm_target(move_architecture, include_resonators=True)
    assert target.num_qubits == expected_target.num_qubits
    assert target.operation_names == expected_target.operation_names
    assert backend._qb_to_idx == qb_to_idx_move_architecture


def test_qubit_name_to_index_to_qubit_name(adonis_architecture):
    backend = DummyIQMBackend(adonis_architecture)

    for idx, name in backend._idx_to_qb.items():
        assert backend.index_to_qubit_name(idx) == name
        assert backend.qubit_name_to_index(name) == idx

    with pytest.raises(ValueError, match="Qubit index 7 is not found on the backend."):
        backend.index_to_qubit_name(7)
    with pytest.raises(ValueError, match="Qubit 'Alice' is not found on the backend."):
        backend.qubit_name_to_index("Alice")
