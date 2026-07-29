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

"""E2E tests for ``qubit_selector`` module."""

import os

from iqm.qiskit_iqm import IQMProvider
from iqm.qiskit_iqm.iqm_backend import IQMBackendBase
from iqm.qubit_selector.qubit_selector import (
    CalibrationDataManager,
    CalibrationType,
    CostEvaluator,
    CostFunction,
    LayoutGenerator,
    ObservationType,
    ReadoutMode,
)
import pytest
from pytest import MonkeyPatch
from qiskit import QuantumCircuit

from tests.unit.iqm.qubit_selector.conftest import ghz_circuit_with_4_qubits

# E2E tests require the following environment variables to be set:
#  IQM_TOKEN - Resonance API token
#  RESONANCE_BASE_URL
#  RESONANCE_STATIONS - A CSV string with Resonance station names to run the tests on

RESONANCE_BASE_URL = os.environ.get("RESONANCE_BASE_URL")
RESONANCE_STATIONS = os.environ.get("RESONANCE_STATIONS").split(",")
NUM_QUBITS = 4

HTTP_GET_TIMEOUT = 60


class TestQubitSelector:
    """E2E tests for ``qubit_selector`` module."""

    def get_backend(self, iqm_server_url: str) -> IQMBackendBase:
        """Get IQM backend for testing.

        Args:
            iqm_server_url: The URL of the IQM server.

        Returns:
            The IQM backend instance.

        """
        provider = IQMProvider(iqm_server_url)
        backend = provider.get_backend()
        return backend

    @pytest.mark.parametrize("station", RESONANCE_STATIONS)
    def test_get_calibration_fidelities_from_resonance_returns_parsed_data(
        self, monkeypatch: MonkeyPatch, station: str
    ) -> None:
        """Test that ``get_calibration_data`` returns parsed calibration metrics data.

        The parsed data dictionary should contain certain keys.

        """
        iqm_server_url = RESONANCE_BASE_URL + station
        monkeypatch.setenv("IQM_SERVER_URL", iqm_server_url)

        backend = self.get_backend(iqm_server_url)

        cal_data = CalibrationDataManager().get_calibration_fidelities(backend=backend)

        assert "CZ" in cal_data
        assert "CLIFFORD" in cal_data
        assert "1Q" in cal_data
        assert "readout" in cal_data
        assert "readout_qndness" in cal_data
        assert "t1" in cal_data
        assert "t2" in cal_data

        if backend.has_resonators():
            assert "DOUBLE_MOVE" in cal_data

    @pytest.mark.parametrize("station", RESONANCE_STATIONS)
    def test_quality_metric_names(self, monkeypatch: MonkeyPatch, station: str) -> None:
        """Test that calibration data exists for each metric."""
        iqm_server_url = RESONANCE_BASE_URL + station
        monkeypatch.setenv("IQM_SERVER_URL", iqm_server_url)

        backend = self.get_backend(iqm_server_url)

        calibration_data = CalibrationDataManager().get_calibration_fidelities(backend=backend)

        gates_info = {}
        for gate in backend.architecture.gates.keys():
            gates_info[gate] = {
                x: backend.architecture.gates[gate].implementations[x].loci
                for x in backend.architecture.gates[gate].implementations.keys()
            }

        prx_key = ObservationType.SQG.value
        cz_key = ObservationType.CZ.value
        measure_key = ObservationType.READOUT.value[0]
        num_prx = sum(len(gates_info[prx_key][x]) for x in gates_info[prx_key].keys())
        num_cz = sum(len(gates_info[cz_key][x]) for x in gates_info[cz_key].keys())
        num_measure = sum(len(gates_info[measure_key][x]) for x in gates_info[measure_key].keys())

        assert len(calibration_data[CalibrationType.CZ.value]) == num_cz * 2
        assert len(calibration_data[CalibrationType.CLIFFORD.value]) == num_cz * 2
        assert len(calibration_data[CalibrationType.SQG.value]) == num_prx
        assert len(calibration_data[CalibrationType.READOUT.value]) == num_measure

        if backend.has_resonators():
            move_key = ObservationType.DOUBLE_MOVE.value
            num_move = sum(len(gates_info[move_key][x]) for x in gates_info[move_key].keys())
            assert len(calibration_data[CalibrationType.DOUBLE_MOVE.value]) == num_move * 2

    @pytest.mark.parametrize("station", RESONANCE_STATIONS, ids=lambda s: s)
    @pytest.mark.parametrize("remove_qubits", [None, [0, 1]])
    @pytest.mark.parametrize("readoutmode", list(ReadoutMode), ids=lambda rm: rm.name)
    @pytest.mark.parametrize("cost_function", list(CostFunction), ids=lambda cf: cf.name)
    def test_full_e2e_execution(
        self,
        ghz_circuit_with_4_qubits: QuantumCircuit,
        station: str,
        remove_qubits: list[int] | None,
        readoutmode: ReadoutMode,
        cost_function: CostFunction,
        monkeypatch: MonkeyPatch,
    ) -> None:
        """Test execution of the automatic layout selection on different QPU backends."""
        iqm_server_url = RESONANCE_BASE_URL + station
        monkeypatch.setenv("IQM_SERVER_URL", iqm_server_url)

        backend = self.get_backend(iqm_server_url)

        layouts = LayoutGenerator(
            backend=backend, quantum_circuit=ghz_circuit_with_4_qubits, remove_qubits=remove_qubits
        ).generate_unique_layouts()

        layouts, scores = CostEvaluator(
            backend,
            ghz_circuit_with_4_qubits,
            readoutmode=readoutmode,
            cost_function=cost_function,
            layouts=layouts,
        ).get_top_layouts(num_layouts=10)

        assert all(isinstance(layout, list) for layout in layouts), "Each layout should be a list"
        assert all(len(layout) == NUM_QUBITS for layout in layouts), "Each layout should have 4 qubits"
        assert len(layouts) > 0, "Layouts list should not be empty"
        assert isinstance(scores, list), "Scores should be a list"
        assert all(isinstance(score, float) for score in scores), "All scores should be floats"
        assert scores == sorted(scores), "Scores should be ordered from small to large"
        assert all(score >= 0 for score in scores), "All scores should be non-negative"
        if remove_qubits is not None:
            for layout in layouts:
                for rq in remove_qubits:
                    assert rq not in layout, f"Removed qubit {rq} should not be in any layout"
