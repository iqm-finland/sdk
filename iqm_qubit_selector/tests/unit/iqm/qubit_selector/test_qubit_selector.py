# Copyright 2025-2026 IQM
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
"""Unit tests for `qubit_selector` module."""

from iqm.iqm_client import IQMClient
from iqm.qiskit_iqm.fake_backends.fake_apollo import IQMFakeApollo
from iqm.qiskit_iqm.iqm_backend import IQMBackendBase
from iqm.qubit_selector.qubit_selector import (
    CalibrationDataManager,
    CostEvaluator,
    CostFunction,
    LayoutGenerator,
    ReadoutMode,
)
import pytest
from pytest import MonkeyPatch
from qiskit import QuantumCircuit
import rustworkx as rx


## Tests for ``LayoutGenerator`` class:
def test_generation_of_unique_layouts_for_cystal_topology(
    ghz_circuit_with_4_qubits: QuantumCircuit, fake_apollo_backend: IQMBackendBase
) -> None:
    """Test that the function returns a list of connected qubits for crystal topology."""
    full_graph = rx.PyGraph()
    full_graph.add_nodes_from(range(fake_apollo_backend.num_qubits))
    edge_list = list(fake_apollo_backend.coupling_map)
    full_graph.add_edges_from_no_data(edge_list)

    layouts = LayoutGenerator(
        fake_apollo_backend, ghz_circuit_with_4_qubits, num_trials=10
    )._generate_unique_layouts_for_crystal()
    num_random_elements = 5
    for layout in layouts[:num_random_elements]:
        components = rx.connected_components(full_graph)
        assert any(all(node in component for node in layout) for component in components), (
            f"Layout {layout} is not fully connected in the graph"
        )

    assert len(layouts) > 0, "No layouts generated for cyrstal topology"
    assert all(len(layout) == 4 for layout in layouts), "Generated layouts do not have the correct number of qubits"
    assert len(layouts) == len({tuple(layout) for layout in layouts}), "Generated layouts are not unique"


@pytest.mark.parametrize("fake_backend", [IQMFakeApollo()])
def test_layouts_do_not_contain_removed_qubits(
    deneb_quality_metric_set,
    fake_backend: IQMBackendBase,
    ghz_circuit_with_4_qubits: QuantumCircuit,
    mock_env_iqm_server_url: str,
    monkeypatch: MonkeyPatch,
) -> None:
    """Test that the correct qubits get removed from the calibration data when generating layouts."""
    # Crude way of mocking ``IQMClient``
    monkeypatch.setattr(IQMClient, "__init__", lambda self, url: None)
    monkeypatch.setattr(IQMClient, "get_quality_metric_set", lambda self: deneb_quality_metric_set)
    removed_qubits = [0]

    layouts = LayoutGenerator(
        backend=fake_backend,
        quantum_circuit=ghz_circuit_with_4_qubits,
        remove_qubits=removed_qubits,
    ).generate_unique_layouts()

    for layout in layouts:
        for qubit in removed_qubits:
            assert qubit not in layout, f"Removed qubit {qubit} is still in layout {layout}"


def test_calibration_data_manager_raises_exception_if_iqm_server_url_is_not_set(monkeypatch: MonkeyPatch) -> None:
    """Test that ``CalibrationDataManager`` raises an exception if environment variable IQM_SERVER_URL is not set."""
    monkeypatch.delenv("IQM_SERVER_URL", raising=False)

    with pytest.raises(ValueError, match="IQM_SERVER_URL environment variable not found. Please set it up."):
        CalibrationDataManager()


def test_cost_evaluator_raises_exception_if_iqm_server_url_is_not_set(
    monkeypatch: MonkeyPatch, fake_apollo_backend: IQMBackendBase, ghz_circuit_with_4_qubits: QuantumCircuit
) -> None:
    """Test that ``CostEvaluator`` raises an exception if environment variable IQM_SERVER_URL is not set."""
    monkeypatch.delenv("IQM_SERVER_URL", raising=False)

    with pytest.raises(ValueError, match="IQM_SERVER_URL environment variable not found. Please set it up."):
        CostEvaluator(backend=fake_apollo_backend, quantum_circuit=ghz_circuit_with_4_qubits)


## Tests for ``CostEvaluator`` class.
@pytest.mark.parametrize("fake_backend", [IQMFakeApollo()], ids=lambda fb: fb.name)
def test_cost_evaluation_for_hardcoded_calibration_values(
    fake_backend: IQMBackendBase,
    ghz_circuit_with_4_qubits: QuantumCircuit,
    mock_env_iqm_server_url: str,
    patched_library_crystal,
    costevaluation_params: dict,
    fixed_gate_fidelities: dict,
) -> None:
    """Test full cost function evaluation."""
    layout = [0, 1, 3, 4]
    transpiled_circuit = patched_library_crystal.perform_backend_transpilation(
        [ghz_circuit_with_4_qubits],
        fake_backend,
        qubits=layout,
        coupling_map=fake_backend.coupling_map.reduce(mapping=layout),
        qiskit_optim_level=3,
    )
    num_cz_gates = transpiled_circuit[0].count_ops().get("cz", 0)
    num_r_gates = transpiled_circuit[0].count_ops().get("r", 0)

    cost_ideal = {
        CostFunction.GATE_COST_CZ: 1
        - (fixed_gate_fidelities["1Q"] ** num_r_gates * fixed_gate_fidelities["CZ"] ** num_cz_gates),
        CostFunction.GATE_COST_CLIFFORD: 1
        - (fixed_gate_fidelities["1Q"] ** num_r_gates * fixed_gate_fidelities["CLIFFORD"] ** num_cz_gates),
    }
    all_cost_readout = []
    all_cost_tqg = []
    params = {
        "backend": fake_backend,
        "quantum_circuit": ghz_circuit_with_4_qubits,
        "layouts": [layout],
    }
    for cost_function in costevaluation_params["cost_function"]:
        for readoutmode in costevaluation_params["readoutmode"]:
            params.update({"readoutmode": readoutmode, "cost_function": cost_function})

            cost_evaluator = patched_library_crystal.CostEvaluator(**params)

            evaluated_cost = cost_evaluator.get_top_layouts()[1][0]
            if readoutmode == ReadoutMode.NONE:
                all_cost_tqg.append(evaluated_cost)
                assert pytest.approx(evaluated_cost) == cost_ideal[cost_function], (
                    f"Evaluated cost {evaluated_cost} does not match expected cost {cost_ideal[cost_function]}."
                )
            else:
                all_cost_readout.append(evaluated_cost)

        assert all(cost_readout > cost_ideal[cost_function] for cost_readout in all_cost_readout), (
            f"Cost evaluations with {ReadoutMode.NONE} should equal or lower than {readoutmode}."
        )
    assert all_cost_tqg[0] < all_cost_tqg[1], (
        f"Cost of {CostFunction.GATE_COST_CZ} should be lower than {CostFunction.GATE_COST_CLIFFORD}."
    )


@pytest.mark.skip("Additional non-mandatory unit test")
@pytest.mark.parametrize("fake_backend", [IQMFakeApollo()], ids=lambda fb: fb.name)
@pytest.mark.parametrize("cost_function", list(CostFunction), ids=lambda cf: cf.name)
@pytest.mark.parametrize("readoutmode", list(ReadoutMode), ids=lambda rm: rm.name)
def test_best_layout_evaluation_for_hardcoded_calibration_values(
    fake_backend: IQMBackendBase,
    ghz_circuit_with_4_qubits: QuantumCircuit,
    mock_env_iqm_server_url: str,
    patched_library_crystal,
    fixed_gate_fidelities: dict,
    cost_function: CostFunction,
    readoutmode: ReadoutMode,
) -> None:
    """Test full cost function evaluation."""
    best_layout = [0, 1, 3, 4]
    transpiled_circuit = patched_library_crystal.perform_backend_transpilation(
        [ghz_circuit_with_4_qubits],
        fake_backend,
        qubits=best_layout,
        coupling_map=fake_backend.coupling_map.reduce(mapping=best_layout),
        qiskit_optim_level=3,
    )
    num_cz_gates = transpiled_circuit[0].count_ops().get("cz", 0)
    num_r_gates = transpiled_circuit[0].count_ops().get("r", 0)
    num_qubits = ghz_circuit_with_4_qubits.num_qubits

    cost_ideal = {
        CostFunction.GATE_COST_CZ: 1
        - (fixed_gate_fidelities["1Q"] ** num_r_gates * fixed_gate_fidelities["CZ"] ** num_cz_gates),
        CostFunction.GATE_COST_CLIFFORD: 1
        - (fixed_gate_fidelities["1Q"] ** num_r_gates * fixed_gate_fidelities["CLIFFORD"] ** num_cz_gates),
    }
    params = {
        "backend": fake_backend,
        "quantum_circuit": ghz_circuit_with_4_qubits,
        "readoutmode": readoutmode,
        "cost_function": cost_function,
    }

    layout, costs = patched_library_crystal.CostEvaluator(**params)
    if readoutmode != ReadoutMode.NONE:
        evaluated_cost = 1 - (cost_ideal[cost_function] * fixed_gate_fidelities[readoutmode.values] ** num_qubits)
    else:
        evaluated_cost = 1 - cost_ideal[cost_function]
    print(layout, costs, evaluated_cost)
    assert len(layout) > 0, "No layouts returned from cost evaluator."
    assert len(costs) > 0, "No costs returned from cost evaluator."
    assert layout[0] == best_layout, f"Best layout {layout[0]} does not match expected layout {best_layout}."
    assert pytest.approx(costs[0]) == evaluated_cost, (
        f"Evaluated cost {costs[0]} does not match expected cost {evaluated_cost}."
    )
