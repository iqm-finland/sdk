# Copyright 2024-2025 IQM
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
"""Test cocos.station_service module, mocking requests."""

import numpy as np
import pytest

from iqm.cpc.interface.circuit_execution import HeraldingMode
from iqm.pulla.utils import extract_readout_controller_result_names, map_sweep_results_to_logical_qubits
from iqm.pulse.playlist.playlist import Playlist

readout_mappings = (
    {"meas1": ("QB1__readout.result", "QB2__readout.result"), "meas2": ("QB3__readout.result",)},
    {"key_1": ("QB2__readout.result",), "key_007": ("QB3__readout.result", "QB1__readout.result")},
)

readout_mappings_complex = ({"meas1": ("QB1__readout.result",)},)

readout_mappings_float = ({"meas1": ("QB1__readout.result",), "meas2": ("QB2__readout.result",)},)

readout_mappings_complex_herald = ({"meas1": ("QB1__readout.result",), "__HERALD": ("QB1____HERALD",)},)

station_control_sweep_results = {
    "QB1__readout.result": [np.array([0, 1, 0, 0, 1, 0, 1, 1, 1, 1])],
    "QB2__readout.result": [np.array([1, 1, 0, 1, 1, 1, 0, 0, 1, 1])],
    "QB3__readout.result": [np.array([0, 1, 1, 0, 1, 1, 0, 1, 0, 1])],
}

station_control_sweep_results_complex = {
    "QB1__readout.result": [np.array([0.1 + 0.1j, 0.1 - 0.1j, 1.0 + 0.0j])],
}

station_control_sweep_results_float = {
    "QB1__readout.result": [np.array([0.1, 0.2, -0.3])],
    "QB2__readout.result": [np.array([0.3, 0.2, -0.1])],
}

station_control_sweep_results_complex_herald = {
    "QB1__readout.result": [np.array([0.1 + 0.1j, 0.1 - 0.1j, 1.0 + 0.0j])],
    "QB1____HERALD": [np.array([0.1 + 0.1j, 0.1 - 0.1j, 1.0 + 0.0j])],
}

client_measurements = [
    {"meas1": [[0, 1], [0, 0], [1, 1], [1, 0], [1, 1]], "meas2": [[0], [1], [1], [0], [0]]},
    {"key_1": [[1], [1], [1], [0], [1]], "key_007": [[1, 1], [0, 0], [1, 0], [1, 1], [1, 1]]},
]

client_measurements_complex = [
    {
        "meas1": [[0.1 + 0.1j], [0.1 - 0.1j], [1.0 + 0.0j]],
    }
]

client_measurements_float = [
    {
        "meas1": [[0.1], [0.2], [-0.3]],
        "meas2": [[0.3], [0.2], [-0.1]],
    }
]

client_measurements_complex_herald = [
    {
        "meas1": [[0.1 + 0.1j], [0.1 - 0.1j], [1.0 + 0.0j]],
        "__HERALD": [[0.1 + 0.1j], [0.1 - 0.1j], [1.0 + 0.0j]],
    }
]

bad_station_control_measurements: dict[str, str] = {
    "some bad measurement": "some bad measurement",
}

readout_mappings_mid_circuit = (
    {"c_3_0_0": ("QB1__c_3_0_0",), "c_3_1_1": ("QB2__c_3_1_1",), "key1": ("QB1__key1", "QB2__key1")},
)

station_control_sweep_results_mid_circuit = {
    "QB1__c_3_0_0": [np.array([0, 1, 0, 0, 1])],
    "QB2__c_3_1_1": [np.array([1, 1, 0, 1, 1])],
    "QB1__key1": [np.array([0, 1, 1, 0, 1])],
    "QB2__key1": [np.array([1, 0, 0, 0, 1])],
}

client_measurements_mid_circuit = [
    {
        "c_3_0_0": [[0], [1], [0], [0], [1]],
        "c_3_1_1": [[1], [1], [0], [1], [1]],
        "key1": [[0, 1], [1, 0], [1, 0], [0, 0], [1, 1]],
    },
]

readout_mappings_mid_circuit_herald = (
    {
        "c_3_0_0": ("QB1__c_3_0_0",),
        "c_3_1_1": ("QB2__c_3_1_1",),
        "key1": ("QB1__key1", "QB2__key1"),
        "__HERALD": ("QB1____HERALD", "QB2____HERALD"),
    },
)

station_control_sweep_results_mid_circuit_herald = {
    "QB1__c_3_0_0": [np.array([0, 1, 0, 0, 1])],
    "QB2__c_3_1_1": [np.array([1, 1, 0, 1, 1])],
    "QB1__key1": [np.array([0, 1, 1, 0, 1])],
    "QB2__key1": [np.array([1, 0, 0, 0, 1])],
    "QB1____HERALD": [np.array([0, 0, 1, 0, 0])],
    "QB2____HERALD": [np.array([0, 0, 0, 1, 0])],
}

client_measurements_mid_circuit_herald = [
    {"c_3_0_0": [[0], [1], [1]], "c_3_1_1": [[1], [1], [1]], "key1": [[0, 1], [1, 0], [1, 1]]},
]

station_control_sweep_results_ragged = {
    "QB1__circuit1": [np.array([0.0, 0.0])],
    "QB1__circuit1-2": [np.array([1.0, 0.0, 1.0, 0.0])],
    "QB2__circuit2-3": [np.array([0.0, 1.0, 0.0, 0.0])],
    "QB1__circuit1-2-3": [np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.0])],
}

client_measurements_ragged = [
    {"circuit1": [[0.0], [0.0]], "circuit1-2": [[1.0], [1.0]], "circuit1-2-3": [[0.0], [1.0]]},
    {"circuit1-2": [[0.0], [0.0]], "circuit2-3": [[0.0], [0.0]], "circuit1-2-3": [[0.0], [1.0]]},
    {"circuit2-3": [[1.0], [0.0]], "circuit1-2-3": [[0.0], [0.0]]},
]

readout_mappings_ragged = (
    {"circuit1": ("QB1__circuit1",), "circuit1-2": ("QB1__circuit1-2",), "circuit1-2-3": ("QB1__circuit1-2-3",)},
    {"circuit1-2": ("QB1__circuit1-2",), "circuit2-3": ("QB2__circuit2-3",), "circuit1-2-3": ("QB1__circuit1-2-3",)},
    {
        "circuit1-2-3": ("QB1__circuit1-2-3",),
        "circuit2-3": ("QB2__circuit2-3",),
    },
)

station_control_sweep_results_ragged_herald = {
    "QB1__circuit1": [np.array([0.0, 0.0])],
    "QB1__circuit1-2": [np.array([1.0, 0.0, 1.0, 0.0])],
    "QB2__circuit2-3": [np.array([0.0, 1.0, 0.0, 0.0])],
    "QB1__circuit1-2-3": [np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.0])],
    "QB4__circuit4": [np.array([1.0, 1.0])],
    "QB1____HERALD": [np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])],
    "QB2____HERALD": [np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])],
    "QB4____HERALD": [np.array([1.0, 0.0])],
}

client_measurements_ragged_herald = [
    {"circuit1": [[0.0], [0.0]], "circuit1-2": [[1.0], [1.0]], "circuit1-2-3": [[0.0], [1.0]]},
    {"circuit1-2": [[0.0]], "circuit2-3": [[0.0]], "circuit1-2-3": [[1.0]]},
    {"circuit2-3": [[1.0]], "circuit1-2-3": [[0.0]]},
    {"circuit4": [[1.0]]},
]

readout_mappings_ragged_herald = (
    {
        "circuit1": ("QB1__circuit1",),
        "circuit1-2": ("QB1__circuit1-2",),
        "circuit1-2-3": ("QB1__circuit1-2-3",),
        "__HERALD": ("QB1____HERALD", "QB2____HERALD"),
    },
    {
        "circuit1-2": ("QB1__circuit1-2",),
        "circuit2-3": ("QB2__circuit2-3",),
        "circuit1-2-3": ("QB1__circuit1-2-3",),
        "__HERALD": ("QB1____HERALD", "QB2____HERALD"),
    },
    {
        "circuit1-2-3": ("QB1__circuit1-2-3",),
        "circuit2-3": ("QB2__circuit2-3",),
        "__HERALD": ("QB1____HERALD", "QB2____HERALD"),
    },
    {
        "circuit4": ("QB4__circuit4",),
        "__HERALD": ("QB4____HERALD",),
    },
)


@pytest.fixture()
def playlist() -> Playlist:
    """Empty playlist."""
    return Playlist()


def test_extract_readout_result_parameter_names():
    """Test readout controllers mapping conversion."""
    some_readout_mapping = ({"1": ("a",), "2": ("b",)}, {"3": ("c",), "4": ("d",)})
    expected_readout_controllers_mapping = {"a", "b", "c", "d"}
    assert extract_readout_controller_result_names(some_readout_mapping) == expected_readout_controllers_mapping


def test_convert_sweep_to_client_representation():
    """Test measurements conversion."""
    assert (
        map_sweep_results_to_logical_qubits(station_control_sweep_results, readout_mappings, HeraldingMode.NONE)
        == client_measurements
    )


def test_convert_sweep_to_client_representation_complex():
    """Test measurements conversion."""
    assert (
        map_sweep_results_to_logical_qubits(
            station_control_sweep_results_complex, readout_mappings_complex, HeraldingMode.NONE
        )
        == client_measurements_complex
    )


def test_convert_sweep_to_client_representation_float():
    """Test conversion of measurements that were averaged on the hardware."""
    assert (
        map_sweep_results_to_logical_qubits(
            station_control_sweep_results_float, readout_mappings_float, HeraldingMode.NONE
        )
        == client_measurements_float
    )


def test_convert_sweep_to_client_representation_complex_herald():
    """Test measurements conversion."""
    assert (
        map_sweep_results_to_logical_qubits(
            station_control_sweep_results_complex_herald, readout_mappings_complex_herald, HeraldingMode.ZEROS
        )
        == client_measurements_complex_herald
    )


def test_convert_sweep_to_client_representation_with_mid_circuit_measurements():
    """Test measurements conversion with multiple measurements per qubit."""
    representation = map_sweep_results_to_logical_qubits(
        station_control_sweep_results_mid_circuit, readout_mappings_mid_circuit, HeraldingMode.NONE
    )
    assert representation == client_measurements_mid_circuit


def test_convert_sweep_to_client_representation_with_mid_circuit_measurements_and_herald():
    """Test measurements conversion with multiple measurements per qubit."""
    representation = map_sweep_results_to_logical_qubits(
        station_control_sweep_results_mid_circuit_herald, readout_mappings_mid_circuit_herald, HeraldingMode.ZEROS
    )
    assert representation == client_measurements_mid_circuit_herald


def test_convert_sweep_to_client_representation_ragged():
    """Test measurements conversion with multiple measurements per qubit."""
    representation = map_sweep_results_to_logical_qubits(
        station_control_sweep_results_ragged, readout_mappings_ragged, HeraldingMode.NONE
    )
    assert representation == client_measurements_ragged


def test_convert_sweep_to_client_representation_ragged_herald():
    """Test measurements conversion with multiple measurements per qubit."""
    representation = map_sweep_results_to_logical_qubits(
        station_control_sweep_results_ragged_herald, readout_mappings_ragged_herald, HeraldingMode.ZEROS
    )
    assert representation == client_measurements_ragged_herald
