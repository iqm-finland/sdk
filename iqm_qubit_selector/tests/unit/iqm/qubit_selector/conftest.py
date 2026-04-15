# Copyright 2022-2026 IQM
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

"""Qubit selector tests configuration."""

import json
from pathlib import Path

from iqm.qiskit_iqm.fake_backends import IQMFakeBackend
from iqm.qiskit_iqm.fake_backends.fake_apollo import IQMFakeApollo
from iqm.qiskit_iqm.fake_backends.fake_deneb import IQMFakeDeneb
from iqm.qubit_selector.qiskit_utils import CircuitType, get_circuit
import iqm.qubit_selector.qubit_selector as qs
import pytest
from pytest import MonkeyPatch
from qiskit import QuantumCircuit

from iqm.pulse.builder import CircuitOperation
from iqm.pulse.circuit_operations import Circuit
from iqm.station_control.interface.models import ObservationSetWithObservations

RESOURCES = Path(__file__).parents[3] / "resources"


@pytest.fixture
def fake_apollo_backend() -> IQMFakeBackend:
    """Return a fake IQM Apollo backend."""
    return IQMFakeApollo()


@pytest.fixture
def fake_deneb_backend() -> IQMFakeBackend:
    """Return a fake IQM Deneb backend."""
    return IQMFakeDeneb()


@pytest.fixture
def ghz_circuit_with_4_qubits() -> QuantumCircuit:
    """Return a quantum circuit of type "GHZ" with 4 qubits."""
    return get_circuit(CircuitType.GHZ, num_qubits=4)


@pytest.fixture
def cal_data_sorted_crystal() -> dict[str, dict[str, float]]:
    """Sorted Deneb calibration data, loaded from a JSON file."""
    path = RESOURCES / "calibration_sorted_crystal.json"
    with open(path, mode="r", encoding="utf-8") as f:
        cal_data = json.load(f)
    return cal_data


@pytest.fixture
def cal_data_sorted_star_and_crystal() -> list[dict[str, dict[str, float]]]:
    """Sorted Deneb and Garnet calibration data, loaded from a JSON files."""
    star_path = RESOURCES / "calibration_sorted_star.json"
    crystal_path = RESOURCES / "calibration_sorted_crystal.json"

    cal_data = []

    with open(star_path, mode="r", encoding="utf-8") as f:
        cal_data.append(json.load(f))

    with open(crystal_path, mode="r", encoding="utf-8") as f:
        cal_data.append(json.load(f))

    return cal_data


@pytest.fixture
def deneb_quality_metric_set() -> ObservationSetWithObservations:
    """Deneb quality metric set.

    TODO: ``observations`` is empty list; to be added?

    """
    path = RESOURCES / "calibration_sorted_deneb_with_extra_fields.json"

    with open(path, mode="r", encoding="utf-8") as f:
        cal_data = json.load(f)

    model = ObservationSetWithObservations(**cal_data)
    return model


@pytest.fixture
def mock_env_iqm_server_url(monkeypatch: MonkeyPatch) -> None:
    """Set environment variable IQM_SERVER_URL.

    The URL does not necessarily has to be of an existing server.

    """
    monkeypatch.setenv("IQM_SERVER_URL", "https://fake-iqm-server-url.com")


@pytest.fixture
def mock_circuit_compiler_data_crystal() -> tuple[list[Circuit], None]:
    """Mock return data of ``iqm.pulla.utils_qiskit.qiskit_to_pulla``.

    Set ``Compiler`` to None as it is irrelevant in the context where the fixture is supposed to be used.

    """
    path = RESOURCES / "serialized_circuits_crystal.json"

    with open(path, mode="r", encoding="utf-8") as f:
        serialized_circuits = json.load(f)

    deserialized_circuits = []  # Recreate ``tuple[list[Circuit], Compiler]`` from JSON (except for ``Compiler``)

    for circuit_data in serialized_circuits:
        circuit = Circuit(
            name=circuit_data["name"],
            instructions=[CircuitOperation(**instruction_data) for instruction_data in circuit_data["instructions"]],
            metadata=circuit_data["metadata"],
        )
        deserialized_circuits.append(circuit)

    return (deserialized_circuits, None)


@pytest.fixture
def patched_library_crystal(
    cal_data_sorted_crystal: dict[str, dict[str, float]], mock_circuit_compiler_data_crystal: tuple[list[Circuit], None]
) -> qs:
    """Patch select internal imports and classes of ``qubit_selector`` module."""
    with pytest.MonkeyPatch.context() as mp:
        from iqm.qubit_selector import qubit_selector  # noqa: PLC0415

        from iqm.pulla.pulla import Pulla  # noqa: PLC0415

        mp.setattr(Pulla, "__init__", lambda self, url: None)
        mp.setattr(
            qubit_selector, "qiskit_to_pulla", lambda pulla, backend, circuits: mock_circuit_compiler_data_crystal
        )
        mp.setattr(
            qubit_selector.CalibrationDataManager,
            "get_calibration_fidelities",
            lambda self, backend: cal_data_sorted_crystal,
        )

        yield qubit_selector


@pytest.fixture
def costevaluation_params() -> dict[str, list]:
    """Return a function that generates all combinations of given parameters."""
    params_general = {
        "readoutmode": [
            qs.ReadoutMode.NONE,
            qs.ReadoutMode.FIDELITY,
            qs.ReadoutMode.QNDNESS,
        ],
        "cost_function": [
            qs.CostFunction.GATE_COST_CZ,
            qs.CostFunction.GATE_COST_CLIFFORD,
        ],
        "remove_qubits": [
            None,
            [0, 1],
        ],
    }

    return params_general


@pytest.fixture
def fixed_gate_fidelities() -> dict[str, float]:
    """Return a function that generates all combinations of given parameters."""
    fixed_gate_fidelities = {
        "CZ": 0.997,
        "CLIFFORD": 0.99,
        "1Q": 0.999,
        "fidelity": 0.99,  # readout fidelity
        "qndness": 0.985,  # readout qndness
        "DOUBLE_MOVE": 0.995,
    }
    return fixed_gate_fidelities
