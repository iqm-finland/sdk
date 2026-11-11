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

Tests verify that REMWorkflow.submit() accepts IQM, Qiskit and Qrisp
circuits — exercising the full integration path down to the mocked client.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from unittest.mock import MagicMock, patch
from uuid import UUID

from iqm.error_reduction_tools.readout_characterization import RECConfiguration
from iqm.error_reduction_tools.readout_characterization.readout_error_characterization import (
    ReadoutErrorCharacterization,
)
from iqm.error_reduction_tools.rem.rem_api import (
    REMWorkflow,
    WorkflowConfiguration,
)
from iqm.error_reduction_tools.twirling.twirling_api import CircuitTwirler, TwirlingConfiguration
from iqm.iqm_client import RunJobTracker
import pytest
from qiskit.circuit import QuantumCircuit as QiskitQuantumCircuit

from iqm.pulse import Circuit, CircuitOperation
from iqm.pulse.playlist.instructions import ReadoutMetrics

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
    """Build an immutable iqm.pulse.Circuit."""
    ops: list[CircuitOperation] = [
        CircuitOperation(name="prx", locus=(qb,), args={"angle": 0.5, "phase": 0.0}) for qb in qubits
    ]
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
    """Create a mock IQMClient that exposes a lattice topology."""
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
    client.get_dynamic_quantum_architecture.return_value = dqa

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
    """Verify that REMWorkflow.submit() accepts IQM, Qiskit and Qrisp circuits.

    REMWorkflow.submit() delegates to CircuitTwirler.twirl(), so these tests
    verify the integration path end-to-end (up to the mocked client).
    """

    def test_qiskit_via_workflow(self, mock_client: MagicMock) -> None:
        """REMWorkflow.submit() accepts Qiskit QuantumCircuit."""
        qc = _make_qiskit_circuit(["QB1", "QB2"])
        wf = REMWorkflow(mock_client, config=WorkflowConfiguration(shots=1000))
        wf.submit([qc])

    def test_pulse_circuit_via_workflow(self, mock_client: MagicMock) -> None:
        """REMWorkflow.submit() accepts iqm.pulse.Circuit."""
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


# ===================================================================
# REMWorkflow — job id accessors
# ===================================================================


class TestREMWorkflowJobIds:
    """Verify job id retrieval from REMWorkflow."""

    def test_get_job_ids_before_submit_raises(self, mock_client: MagicMock) -> None:
        """get_job_ids() before submit() raises RuntimeError."""
        wf = REMWorkflow(mock_client, config=WorkflowConfiguration())
        with pytest.raises(RuntimeError, match="Call submit"):
            wf.get_job_ids()

    def test_get_job_ids_after_submit(self, mock_client: MagicMock) -> None:
        """After submit() both job ids are exposed (characterization not reused)."""
        qc = _make_qiskit_circuit(["QB1", "QB2"])
        wf = REMWorkflow(mock_client, config=WorkflowConfiguration(shots=1000))
        wf.submit([qc])

        job_ids = wf.get_job_ids()
        assert set(job_ids) == {"characterization", "twirling"}
        assert job_ids["twirling"] is not None
        assert job_ids["characterization"] is not None  # a REC job was submitted

    def test_get_job_ids_characterization_reused_is_none(self, mock_client: MagicMock) -> None:
        """When characterization is reused, no REC job is submitted, so its id is None."""
        qc = _make_qiskit_circuit(["QB1", "QB2"])
        reused = ReadoutErrorCharacterization(mock_client)
        wf = REMWorkflow(mock_client, config=WorkflowConfiguration(shots=1000), characterization=reused)
        wf.submit([qc])

        job_ids = wf.get_job_ids()
        assert job_ids["characterization"] is None
        assert job_ids["twirling"] is not None


# REMWorkflow: per-circuit observables


class _RecordingMitigation:
    """Stand-in for ReadoutErrorMitigation that records the observables it is handed."""

    calls: list = []

    def __init__(self, readout_errors) -> None:
        pass

    def mitigate_counts(self, experiment_counts, qubit_to_bit_mapping, observables=None, **kwargs):
        type(self).calls.append(observables)
        num_observables = len(observables) if observables else 1
        return {
            "mitigated_counts": [[{"00": 1.0} for _ in range(num_observables)]],
            "expectation_values": [[1.0] * num_observables],
            "characterization_performed": False,
        }


@pytest.fixture()
def recording_mitigation(monkeypatch: pytest.MonkeyPatch) -> type[_RecordingMitigation]:
    """Replace ReadoutErrorMitigation in rem_api with a recorder, and reset it per test."""
    _RecordingMitigation.calls = []
    monkeypatch.setattr("iqm.error_reduction_tools.rem.rem_api.ReadoutErrorMitigation", _RecordingMitigation)
    return _RecordingMitigation


def _workflow_ready_for_retrieval(num_circuits: int) -> REMWorkflow:
    """A workflow with submit() already faked out, ready for get_results()."""
    wf = REMWorkflow(MagicMock(), config=WorkflowConfiguration(shots=1000))

    twirler = MagicMock()
    twirler.retrieve_counts.return_value = [{"00": 500, "11": 500} for _ in range(num_circuits)]
    twirler.get_qubit_to_bit_mapping.return_value = [{"QB1": 0, "QB2": 1} for _ in range(num_circuits)]
    twirler.get_job_id.return_value = "job-id"
    wf._twirler = twirler

    rec = MagicMock()
    rec.get_readout_error_probabilities.return_value = {"charact_data": {}}
    wf._rec = rec
    wf._characterization_reused = True

    return wf


class TestREMWorkflowPerCircuitObservables:
    """Verify routing of shared vs per-circuit observables."""

    def test_shared_observables_go_to_every_circuit(self, recording_mitigation) -> None:
        wf = _workflow_ready_for_retrieval(2)
        wf._observables = ["ZZ"]

        wf.get_results()

        assert recording_mitigation.calls == [["ZZ"], ["ZZ"]]

    def test_per_circuit_observables_are_routed_by_index(self, recording_mitigation) -> None:
        wf = _workflow_ready_for_retrieval(2)
        wf._observables_per_circuit = [["ZZ"], ["ZI", "IZ"]]

        results = wf.get_results()

        assert recording_mitigation.calls == [["ZZ"], ["ZI", "IZ"]]
        # Each circuit gets as many expectation values as it has observables.
        assert [len(values) for values in results.expectation_values] == [1, 2]

    def test_per_circuit_observables_populate_expectation_values(self, recording_mitigation) -> None:
        wf = _workflow_ready_for_retrieval(1)
        wf._observables_per_circuit = [["ZZ"]]

        results = wf.get_results()

        assert results.expectation_values is not None

    def test_no_observables_leaves_expectation_values_none(self, recording_mitigation) -> None:
        wf = _workflow_ready_for_retrieval(1)

        results = wf.get_results()

        assert results.expectation_values is None

    def test_both_observable_arguments_raises(self, mock_client: MagicMock) -> None:
        wf = REMWorkflow(mock_client, config=WorkflowConfiguration(shots=1000))
        circuits = [_make_qiskit_circuit(["QB1", "QB2"])]

        with pytest.raises(ValueError, match="not both"):
            wf.submit(circuits, observables=["ZZ"], observables_per_circuit=[["ZZ"]])

    def test_length_mismatch_raises(self, mock_client: MagicMock) -> None:
        wf = REMWorkflow(mock_client, config=WorkflowConfiguration(shots=1000))
        circuits = [_make_qiskit_circuit(["QB1", "QB2"])]

        with pytest.raises(ValueError, match="match one-to-one"):
            wf.submit(circuits, observables_per_circuit=[["ZZ"], ["ZI"]])

    def test_submit_stores_per_circuit_observables(self, mock_client: MagicMock) -> None:
        wf = REMWorkflow(mock_client, config=WorkflowConfiguration(shots=1000))
        circuits = [_make_qiskit_circuit(["QB1", "QB2"])]

        wf.submit(circuits, observables_per_circuit=[["ZZ"]])

        assert wf._observables_per_circuit == [["ZZ"]]
        assert wf._observables is None


# REMWorkflow: the `twirled` flag handed to mitigation


class _TwirledFlagRecorder:
    """Stand-in for ReadoutErrorMitigation that records the ``twirled`` flag it receives."""

    calls: list = []

    def __init__(self, readout_errors) -> None:
        pass

    def mitigate_counts(self, experiment_counts, qubit_to_bit_mapping, observables=None, **kwargs):
        type(self).calls.append(kwargs.get("twirled"))
        return {
            "mitigated_counts": [[{"00": 1.0}]],
            "expectation_values": [[1.0]],
            "characterization_performed": False,
        }


@pytest.fixture()
def twirled_flag_recorder(monkeypatch: pytest.MonkeyPatch) -> type[_TwirledFlagRecorder]:
    """Replace ReadoutErrorMitigation in rem_api with a `twirled` recorder."""
    _TwirledFlagRecorder.calls = []
    monkeypatch.setattr("iqm.error_reduction_tools.rem.rem_api.ReadoutErrorMitigation", _TwirledFlagRecorder)
    return _TwirledFlagRecorder


class TestREMWorkflowTwirledFlag:
    """Tests that the confusion matrix is symmetrized only when readout twirling was applied (HCS-917)."""

    @pytest.mark.parametrize("applied", [True, False])
    def test_flag_follows_the_twirler(self, twirled_flag_recorder, applied: bool) -> None:
        """Test that mitigation gets the twirled flag the twirler reports."""
        wf = _workflow_ready_for_retrieval(2)
        wf._twirler.readout_twirling_applied = applied

        wf.get_results()

        assert twirled_flag_recorder.calls == [applied, applied]

    def test_not_hardcoded_true(self, twirled_flag_recorder) -> None:
        """The specific regression: an untwirled run must not be mitigated as if twirled."""
        wf = _workflow_ready_for_retrieval(1)
        wf._twirler.readout_twirling_applied = False

        wf.get_results()

        assert twirled_flag_recorder.calls == [False]


# REMWorkflow: persistence, for submit-now / retrieve-later


def _characterization_payload(qubits: list[str]) -> dict:
    """A retrieved-characterization payload, as ReadoutErrorCharacterization.to_dict emits."""
    matrix = [[0.97, 0.03], [0.03, 0.97]]
    return {
        "counts_by_prep": {},
        "measured_qubits": list(qubits),
        "config": asdict(RECConfiguration()),
        "charact_data": dict.fromkeys(qubits, matrix),
        "charact_data_std": {},
        "timestamp": "2026-08-28T00:00:00+00:00",
    }


def _retrieving_client() -> MagicMock:
    """A client whose `get_job(job_id)` returns a run-job tracker carrying that id.

    Specced as ``RunJobTracker`` because ``CircuitTwirler.from_dict`` checks the job type --
    a playlist run job is the only kind it could have submitted. The tracker's ``_context``
    is a real dict, because ``from_dict`` restores the compiler context into it.
    """
    client = MagicMock()

    def _job(job_id: UUID) -> MagicMock:
        tracker = MagicMock(spec=RunJobTracker, job_id=job_id)
        tracker._context = {}
        return tracker

    client.get_job.side_effect = _job
    return client


def _readout_metrics(qubits: list[str], num_segments: int) -> ReadoutMetrics:
    """The part of a compiler context a job tracker needs to post-process its results."""
    return ReadoutMetrics(
        num_segments=num_segments,
        integration_occurrences={
            f"{qubit}__m_{len(qubits)}_0_{index}": [1] * num_segments for index, qubit in enumerate(qubits)
        },
        implementations={f"{qubit}__m": {"measure.constant"} for qubit in qubits},
    )


def _pulse_circuit(qubits: list[str]) -> Circuit:
    """A pulse circuit measuring `qubits` in clbit order."""
    operations: list[CircuitOperation] = [
        CircuitOperation(name="prx", locus=(qubit,), args={"angle": 0.5, "phase": 0.0}) for qubit in qubits
    ]
    operations += [
        CircuitOperation(name="measure", locus=(qubit,), args={"key": f"m_{len(qubits)}_0_{index}"})
        for index, qubit in enumerate(qubits)
    ]
    return Circuit(name="c", instructions=tuple(operations))


def _submitted_workflow(qubits: list[str], job_id: UUID) -> REMWorkflow:
    """A workflow in the state submit() leaves it in, without touching a quantum computer."""
    twirling = TwirlingConfiguration(readout_twirl_strategy="HADAMARD", circuit_twirling=False, seed=11)
    workflow = REMWorkflow(MagicMock(), config=WorkflowConfiguration(shots=1000, twirling=twirling))

    twirler = CircuitTwirler(config=twirling)
    twirler.twirl([_pulse_circuit(qubits)])
    twirler._job = MagicMock(spec=RunJobTracker, job_id=job_id)
    twirler._shots_per_circuit = 250
    num_variants = len(twirler.get_twirled_circuits()[0])
    twirler._compiler_context = {
        "readout_metrics": _readout_metrics(qubits, num_variants),
        "mapped_readout_keys": {},
        "run_data": "rebuilt by the tracker, not serialized",
    }

    workflow._twirler = twirler
    workflow._rec = ReadoutErrorCharacterization.from_dict(_characterization_payload(qubits))
    workflow._characterization_reused = True
    workflow._observables = ["ZII"]
    return workflow


QUBITS = ["QB1", "QB2", "QB3"]
JOB_ID = UUID("12345678-1234-5678-1234-567812345678")


class TestREMWorkflowPersistence:
    """`submit()` and `get_results()` must be able to happen in different sessions."""

    def test_to_dict_is_json_serializable(self) -> None:
        payload = _submitted_workflow(QUBITS, JOB_ID).to_dict()
        assert json.loads(json.dumps(payload)) == payload

    def test_to_dict_carries_the_job_id(self) -> None:
        payload = _submitted_workflow(QUBITS, JOB_ID).to_dict()
        assert payload["twirler"]["job_id"] == str(JOB_ID)

    def test_to_dict_before_submit_raises(self) -> None:
        workflow = REMWorkflow(MagicMock(), config=WorkflowConfiguration())
        with pytest.raises(RuntimeError, match="Call submit"):
            workflow.to_dict()

    def test_to_dict_without_retrieved_characterization_raises(self) -> None:
        """Test that serializing with a pending characterization raises."""
        workflow = _submitted_workflow(QUBITS, JOB_ID)
        workflow._rec = ReadoutErrorCharacterization(MagicMock())
        with pytest.raises(RuntimeError, match="has not been retrieved"):
            workflow.to_dict()

    def test_from_dict_rejects_unknown_version(self) -> None:
        payload = _submitted_workflow(QUBITS, JOB_ID).to_dict()
        payload["version"] = 99
        with pytest.raises(ValueError, match="Unsupported REMWorkflow payload version"):
            REMWorkflow.from_dict(payload, MagicMock())

    def test_round_trip_preserves_configuration_and_observables(self) -> None:
        original = _submitted_workflow(QUBITS, JOB_ID)
        original._config.shots = 4321
        restored = REMWorkflow.from_dict(json.loads(json.dumps(original.to_dict())), _retrieving_client())

        assert restored._config.shots == 4321
        assert restored._config.twirling.readout_twirl_strategy == "HADAMARD"
        assert restored._observables == ["ZII"]

    def test_restored_workflow_rebuilds_the_job_from_its_id(self) -> None:
        payload = _submitted_workflow(QUBITS, JOB_ID).to_dict()
        client = _retrieving_client()
        REMWorkflow.from_dict(payload, client)
        client.get_job.assert_called_once_with(JOB_ID)

    def test_to_dict_carries_the_compiler_context(self) -> None:
        """Without it a restored tracker cannot post-process its own results."""
        payload = _submitted_workflow(QUBITS, JOB_ID).to_dict()
        stored = payload["twirler"]["compiler_context"]["readout_metrics"]
        assert stored["num_segments"] == 4
        assert set(stored["integration_occurrences"]) == {
            "QB1__m_3_0_0",
            "QB2__m_3_0_1",
            "QB3__m_3_0_2",
        }

    def test_from_dict_injects_the_context_into_the_rebuilt_tracker(self) -> None:
        """Regression for the hardware failure `KeyError: 'readout_metrics'`.

        `IQMClient.get_job()` returns a tracker with an empty context, because the compiler
        context is produced client-side at submission and cannot be fetched from the server.
        Restoring it is what makes the restored path identical to the live one.
        """
        original = _submitted_workflow(QUBITS, JOB_ID)
        restored = REMWorkflow.from_dict(json.loads(json.dumps(original.to_dict())), _retrieving_client())

        context = restored._twirler._job._context
        assert "readout_metrics" in context
        assert context["readout_metrics"] == original._twirler._compiler_context["readout_metrics"]

    def test_payload_without_a_context_is_rejected(self) -> None:
        """Test that a payload with a job id but no compiler context raises."""
        payload = _submitted_workflow(QUBITS, JOB_ID).to_dict()
        payload["twirler"]["compiler_context"] = None
        with pytest.raises(ValueError, match="no compiler context"):
            REMWorkflow.from_dict(payload, _retrieving_client())

    def test_restored_workflow_retrieves_the_same_counts(self) -> None:
        """The point of the feature: a new process gets what the original would have."""
        original = _submitted_workflow(QUBITS, JOB_ID)
        num_variants = len(original._twirler.get_twirled_circuits()[0])
        raw = [{"000": 100 + index, "111": 50} for index in range(num_variants)]

        def _results(*_args: object, **_kwargs: object) -> MagicMock:
            obj = MagicMock()
            obj.get_counts.side_effect = list(raw)
            return obj

        with patch("iqm.error_reduction_tools.utils.job_utils.sweep_job_to_qiskit", side_effect=_results):
            expected = original._twirler.retrieve_counts()

        restored = REMWorkflow.from_dict(json.loads(json.dumps(original.to_dict())), _retrieving_client())
        with patch("iqm.error_reduction_tools.utils.job_utils.sweep_job_to_qiskit", side_effect=_results):
            actual = restored._twirler.retrieve_counts()

        assert actual == expected

    def test_restored_workflow_produces_results(self) -> None:
        """End to end: the restored object can complete get_results()."""
        original = _submitted_workflow(QUBITS, JOB_ID)
        num_variants = len(original._twirler.get_twirled_circuits()[0])

        results_obj = MagicMock()
        results_obj.get_counts.side_effect = [{"000": 400, "100": 100} for _ in range(num_variants)]
        restored = REMWorkflow.from_dict(json.loads(json.dumps(original.to_dict())), _retrieving_client())

        with patch("iqm.error_reduction_tools.utils.job_utils.sweep_job_to_qiskit", return_value=results_obj):
            results = restored.get_results()

        assert len(results.raw_counts) == 1
        assert results.expectation_values is not None
        assert results.metadata.twirling_job_id == str(JOB_ID)

    def test_save_and_load(self, tmp_path) -> None:
        path = str(tmp_path / "workflow.json")
        original = _submitted_workflow(QUBITS, JOB_ID)
        original.save(path)

        restored = REMWorkflow.load(path, _retrieving_client())
        assert restored._observables == ["ZII"]
        assert restored._twirler.get_rot_strings() == original._twirler.get_rot_strings()


class TestCircuitTwirlerRestore:
    """The twirler half of the same story."""

    def test_round_trip_preserves_rot_strings_and_grouping(self) -> None:
        twirler = CircuitTwirler(
            config=TwirlingConfiguration(readout_twirl_strategy="HADAMARD", circuit_twirling=False, seed=11)
        )
        twirler.twirl([_pulse_circuit(QUBITS), _pulse_circuit(["QB4", "QB5"])])
        twirler._job = MagicMock(spec=RunJobTracker, job_id=JOB_ID)
        twirler._compiler_context = {"readout_metrics": _readout_metrics(QUBITS, 4), "mapped_readout_keys": {}}

        restored = CircuitTwirler.from_dict(json.loads(json.dumps(twirler.to_dict())), _retrieving_client())

        assert restored.get_rot_strings() == twirler.get_rot_strings()
        assert restored.get_qubit_to_bit_mapping() == twirler.get_qubit_to_bit_mapping()

    def test_restored_without_job_still_raises_on_retrieve(self) -> None:
        twirler = CircuitTwirler(
            config=TwirlingConfiguration(readout_twirl_strategy="MINIMAL", circuit_twirling=False, seed=1)
        )
        twirler.twirl([_pulse_circuit(QUBITS)])
        restored = CircuitTwirler.from_dict(twirler.to_dict(), MagicMock())
        with pytest.raises(RuntimeError, match="Call submit"):
            restored.retrieve_counts()
