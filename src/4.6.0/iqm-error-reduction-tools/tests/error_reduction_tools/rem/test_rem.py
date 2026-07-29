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

"""Unit tests for the REMWorkflow frontend-acceptance path.

Tests verify that REMWorkflow.submit() accepts Qiskit, Pulla, and Qrisp
circuits — exercising the full integration path down to the mocked client.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from iqm.error_reduction_tools.rem.rem_api import (
    REMWorkflow,
    WorkflowConfiguration,
)
import pytest
from qiskit.circuit import QuantumCircuit as QiskitQuantumCircuit

from iqm.pulse import Circuit, CircuitOperation

qrisp = pytest.importorskip("qrisp", reason="qrisp is not installed")
QrispQuantumCircuit = qrisp.QuantumCircuit

# ---------------------------------------------------------------------------
# Helpers — circuit factories
# ---------------------------------------------------------------------------


def _make_qiskit_circuit(qubits: list[str]) -> QiskitQuantumCircuit:
    """Build a Qiskit QuantumCircuit with PRX-equivalent gates and measurements."""
    n = len(qubits)
    qc = QiskitQuantumCircuit(n, n)
    for i in range(n):
        qc.rx(0.5, i)
    for i in range(n):
        qc.measure(i, i)
    return qc


def _make_immutable_pulse_circuit(qubits: list[str]) -> Circuit:
    """Build an immutable iqm.pulse.Circuit (the Pulla-native type)."""
    ops: list[CircuitOperation] = []
    for i, qb in enumerate(qubits):
        ops.append(CircuitOperation(name="prx", locus=(qb,), args={"angle": 0.5, "phase": 0.0}))
    for i, qb in enumerate(qubits):
        ops.append(CircuitOperation(name="measure", locus=(qb,), args={"key": f"m_{i}"}))
    return Circuit(name="test", instructions=tuple(ops))


def _make_qrisp_circuit(num_qubits: int) -> object:
    """Build a real Qrisp QuantumCircuit with gates and measurements."""
    qc = QrispQuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.rx(0.5, i)
    qc.measure(qc.qubits)
    return qc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_client() -> MagicMock:
    """Create a mock Pulla client that exposes a lattice topology."""
    client = MagicMock()
    chip_topo = MagicMock()
    chip_topo.qubits_sorted = ["QB1", "QB2", "QB3", "QB4"]
    chip_topo.cz_loci = [("QB1", "QB2"), ("QB1", "QB3"), ("QB2", "QB4"), ("QB3", "QB4")]
    client.get_chip_topology.return_value = chip_topo

    # Dynamic quantum architecture: all four qubits have a calibrated measure gate,
    # so they are reported as operational for readout characterization.
    dqa = MagicMock()
    measure_gate = MagicMock()
    measure_gate.loci = [("QB1",), ("QB2",), ("QB3",), ("QB4",)]
    dqa.gates = {"measure": measure_gate}
    dqa.qubits = ["QB1", "QB2", "QB3", "QB4"]
    client._iqm_server_client.get_dynamic_quantum_architecture.return_value = dqa

    # Compiler mock
    compiler = MagicMock()
    job_definition = MagicMock()
    context = {"shots": 0}
    compiler.compile.return_value = (job_definition, context)
    client.get_standard_compiler.return_value = compiler

    # Job mock
    job = MagicMock()
    client.submit_playlist.return_value = job

    return client


# ===================================================================
# REMWorkflow — acceptance of all three frontend types
# ===================================================================


class TestREMWorkflowAcceptsAllFrontends:
    """Verify that REMWorkflow.submit() accepts Qiskit, Pulla, and Qrisp circuits.

    REMWorkflow.submit() delegates to CircuitTwirler.twirl(), so these tests
    verify the integration path end-to-end (up to the mocked client).
    """

    def test_qiskit_via_workflow(self, mock_client: MagicMock) -> None:
        """REMWorkflow.submit() accepts Qiskit QuantumCircuit."""
        qc = _make_qiskit_circuit(["QB1", "QB2"])
        wf = REMWorkflow(mock_client, config=WorkflowConfiguration(shots=1000))
        wf.submit([qc])

    def test_pulse_circuit_via_workflow(self, mock_client: MagicMock) -> None:
        """REMWorkflow.submit() accepts iqm.pulse.Circuit (Pulla native)."""
        pc = _make_immutable_pulse_circuit(["QB1", "QB2"])
        wf = REMWorkflow(mock_client, config=WorkflowConfiguration(shots=1000))
        wf.submit([pc])

    def test_qrisp_via_workflow(self, mock_client: MagicMock) -> None:
        """REMWorkflow.submit() accepts a real Qrisp QuantumCircuit."""
        qrisp_circ = _make_qrisp_circuit(2)
        wf = REMWorkflow(mock_client, config=WorkflowConfiguration(shots=1000))
        wf.submit([qrisp_circ])

    def test_mixed_frontends_via_workflow(self, mock_client: MagicMock) -> None:
        """REMWorkflow.submit() accepts a heterogeneous list of all three types."""
        qiskit_circ = _make_qiskit_circuit(["QB1", "QB2"])
        pulse_circ = _make_immutable_pulse_circuit(["QB3", "QB4"])
        qrisp_circ = _make_qrisp_circuit(2)

        wf = REMWorkflow(mock_client, config=WorkflowConfiguration(shots=1000))
        wf.submit([qiskit_circ, pulse_circ, qrisp_circ])
