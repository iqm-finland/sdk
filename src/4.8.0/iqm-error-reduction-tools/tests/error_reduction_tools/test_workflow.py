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

import warnings

from iqm.error_reduction_tools import workflow as workflow_module
from iqm.error_reduction_tools.cdr import CDR, CDREstimation
from iqm.error_reduction_tools.readout_characterization.readout_error_characterization import RECConfiguration
from iqm.error_reduction_tools.readout_characterization.topologies import QPUTopology
from iqm.error_reduction_tools.twirling.twirling_api import TwirlingConfiguration
from iqm.error_reduction_tools.workflow import ErrorReductionWorkflow
from iqm.error_reduction_tools.zne import ZNE, ZNEEstimation
from iqm.qiskit_iqm.fake_backends import IQMFakeApollo
from iqm.qrisp_iqm import transpile_to_iqm
from mockito import mock, unstub, verify, when
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qrisp import QuantumCircuit as QrispQuantumCircuit

from iqm.pulse import Circuit

from .builders import bell_circuit, rotated_circuit
from .exact import ExactExecutor, exact_distribution

NOISE = {"readout": {"QB1": (0.02, 0.1), "QB2": (0.15, 0.05)}, "depolarizing": 0.02}
"""Readout error per qubit as (P(0 -> 1), P(1 -> 0)), and global depolarizing per gate."""

CLIENT = mock()
"""Stands in for an IQMClient. The workflow only hands it to ResonanceExecutor, which the tests replace."""


@pytest.fixture(autouse=True)
def _restore_resonance_executor():
    yield
    unstub()


def _connect(executor):
    """Make the workflow run on ``executor`` instead of the ResonanceExecutor it builds from the client."""
    when(workflow_module).ResonanceExecutor(...).thenReturn(executor)


class _OtherComputer(ExactExecutor):
    quantum_computer = "elsewhere"


class _NoTwirling(ExactExecutor):
    supports_twirling = False


def _workflow(executor=None, rem=True, twirling=True, shots=2000):
    _connect(executor or ExactExecutor(**NOISE))
    return ErrorReductionWorkflow(
        [bell_circuit(), bell_circuit()],
        ["ZZ", [(0.5, "ZI"), (0.5, "IZ")]],
        ZNE(noise_factors=(1, 3)),
        client=CLIENT,
        twirling_configuration=TwirlingConfiguration(readout_twirl_strategy="MINIMAL", num_twirling_instances=2, seed=0)
        if twirling
        else None,
        rec_configuration=RECConfiguration(num_circuits=4, shots=400, seed=0) if rem is True else rem or None,
        shots=shots,
    )


class TestStages:
    def test_submit_without_plan_plans_with_a_warning(self):
        """Test that submit() before plan() plans the workflow and warns."""
        workflow = _workflow()
        with pytest.warns(UserWarning, match="planning it now"):
            submission = workflow.submit()
        assert workflow.execution_plan is not None
        assert len(submission.jobs) == len(workflow.execution_plan.batches)
        assert submission.calibration_job is not None

    def test_plan_cannot_change_after_submission(self):
        """Test that plan() and submit() raise after the workflow has been submitted."""
        workflow = _workflow()
        workflow.plan()
        workflow.submit()
        with pytest.raises(RuntimeError, match="can no longer change"):
            workflow.plan()
        with pytest.raises(RuntimeError, match="already been submitted"):
            workflow.submit()

    def test_collect_stores_counts_for_every_instance_with_its_shots(self):
        """Test that collect() stores the counts of every instance, with the shots of its batch."""
        workflow = _workflow()
        workflow.plan()
        workflow.submit()
        counts = workflow.collect()
        plan = workflow.execution_plan
        assert set(counts) == {key for batch in plan.batches for key in batch.keys}
        for batch in plan.batches:
            assert all(sum(counts[key].values()) == pytest.approx(batch.shots) for key in batch.keys)
        assert set(workflow.readout_errors) == {"QB1", "QB2"}
        # Column j is the prepared state j, row i the measured outcome i.
        for qubit, (to_one, to_zero) in NOISE["readout"].items():
            matrix = workflow.readout_errors[qubit]
            assert matrix.shape == (2, 2)
            np.testing.assert_allclose(matrix, [[1 - to_one, to_zero], [to_one, 1 - to_zero]], atol=1e-12)

    def test_client_and_compilation_options_reach_the_resonance_executor(self):
        """Test that the client and the compilation options are passed to the ResonanceExecutor."""
        options = {"DDStrategy": "placeholder"}
        executor = ExactExecutor()
        when(workflow_module).ResonanceExecutor(CLIENT, options).thenReturn(executor)
        workflow = ErrorReductionWorkflow([bell_circuit()], "ZZ", ZNE(), client=CLIENT, compilation_options=options)
        workflow.submit()
        verify(workflow_module, times=1).ResonanceExecutor(CLIENT, options)
        assert workflow.submission.quantum_computer == executor.quantum_computer

    def test_collect_with_a_client_on_another_quantum_computer_is_rejected(self, tmp_path):
        """Test that collecting with a client of another quantum computer is rejected."""
        workflow = _workflow()
        workflow.plan()
        workflow.submit()
        workflow.save(tmp_path / "submitted.json")
        _connect(_OtherComputer())
        restored = ErrorReductionWorkflow.load(tmp_path / "submitted.json", client=CLIENT)
        with pytest.raises(RuntimeError, match="elsewhere"):
            restored.collect()

    def test_local_readout_twirling_uses_the_topology_of_the_quantum_computer(self):
        """Test that LOCAL readout twirling uses the topology of the quantum computer."""
        executor = ExactExecutor()
        when(executor).topology().thenReturn(QPUTopology("line", 2, couplers=[("QB1", "QB2")]))
        _connect(executor)
        twirling = TwirlingConfiguration(readout_twirl_strategy="LOCAL", num_twirling_instances=4, seed=0)
        workflow = ErrorReductionWorkflow([bell_circuit()], "ZZ", ZNE(), client=CLIENT, twirling_configuration=twirling)
        plan = workflow.plan()
        verify(executor, times=1).topology()
        assert {instance.rot_string for instance in plan.targets[0].instances()} == {"II", "IX", "XI", "XX"}

    def test_twirling_on_an_executor_without_twirling_is_rejected(self):
        """Test that twirling is rejected when the quantum computer does not support it."""
        with pytest.raises(ValueError, match="does not support twirling"):
            _workflow(executor=_NoTwirling()).plan()

    def test_targets_of_an_unsupported_type_are_rejected(self):
        """Test that targets that cannot be converted to iqm.pulse circuits are rejected."""
        with pytest.raises(TypeError, match="Unsupported circuit type"):
            ErrorReductionWorkflow([bell_circuit(), 42], "ZZ", ZNE(), client=CLIENT)

    def test_observables_must_match_the_measured_width(self):
        """Test that observables of a different width than the measurements are rejected."""
        with pytest.raises(ValueError, match="measures 2 classical bits"):
            ErrorReductionWorkflow([bell_circuit()], "ZZZ", ZNE(), client=CLIENT)

    @pytest.mark.parametrize(
        ("targets", "shots", "message"),
        [([], None, "At least one target circuit"), ([bell_circuit()], 0, "shots must be at least 1, got 0")],
    )
    def test_empty_targets_and_shots_are_rejected(self, targets, shots, message):
        """Test that no targets and zero shots are rejected."""
        with pytest.raises(ValueError, match=message):
            ErrorReductionWorkflow(targets, "ZZ", ZNE(), client=CLIENT, shots=shots)

    def test_collect_needs_a_submission(self):
        """Test that collect() before submit() raises."""
        with pytest.raises(RuntimeError, match="Call submit"):
            _workflow().collect()

    def test_job_returning_the_wrong_number_of_results_is_rejected(self):
        """Test that a job returning a different number of results than circuits is rejected."""
        executor = ExactExecutor()
        workflow = _workflow(executor=executor, rem=False)
        workflow.plan()
        workflow.submit()
        when(executor).collect(...).thenReturn([])
        with pytest.raises(RuntimeError, match="returned 0 results"):
            workflow.collect()


class _Recording(ExactExecutor):
    """Executor that remembers every circuit it is asked to run, in submission order."""

    def __init__(self, **noise):
        super().__init__(**noise)
        self.received = []

    def submit(self, circuits, shots, *, calibration=False):
        self.received.extend(circuits)
        return super().submit(circuits, shots, calibration=calibration)


class TestInspection:
    def test_circuits_are_indexed_by_target_variant_and_instance(self):
        """Test that circuits[i][j][k] is instance k of variant j of target i."""
        workflow = _workflow()
        plan = workflow.plan()
        circuits = workflow.circuits
        assert len(circuits) == len(plan.targets)
        for i, target in enumerate(plan.targets):
            assert len(circuits[i]) == len(target.variants())
            for j, variant in enumerate(target.variants()):
                assert len(circuits[i][j]) == len(variant.instances())
                for k, circuit in enumerate(circuits[i][j]):
                    assert circuit is plan.instance((i, j, k)).circuit
                    assert circuit.name == f"t{i}_v{j}_i{k}"

    @pytest.mark.parametrize(("rem", "twirling"), [(True, True), (False, True), (True, False)])
    def test_circuits_and_calibration_are_exactly_what_submit_sends(self, rem, twirling):
        """Test that submit() sends exactly the circuits and the calibration circuits of the plan."""
        executor = _Recording()
        workflow = _workflow(executor=executor, rem=rem, twirling=twirling)
        plan = workflow.plan()
        nested = [circuit for target in workflow.circuits for variant in target for circuit in variant]
        calibration = [] if plan.calibration is None else list(plan.calibration.circuits)
        workflow.submit()
        assert all(isinstance(circuit, Circuit) for circuit in nested)
        assert len(nested) + len(calibration) == len(executor.received) == plan.num_circuits
        assert sorted(id(c) for c in nested + calibration) == sorted(id(c) for c in executor.received)
        assert [id(c) for c in executor.received[: len(calibration)]] == [id(c) for c in calibration]

    def test_circuits_need_a_plan(self):
        """Test that circuits raises before plan()."""
        with pytest.raises(RuntimeError, match="Call plan"):
            _ = _workflow().circuits


class TestEstimate:
    def _collected(self, **kwargs):
        workflow = _workflow(**kwargs)
        workflow.plan()
        workflow.submit()
        workflow.collect()
        return workflow

    def test_estimate_needs_counts(self):
        """Test that estimate() before collect() raises."""
        with pytest.raises(RuntimeError, match="collect"):
            _workflow().estimate(ZNEEstimation())

    def test_estimation_of_another_technique_is_rejected(self):
        """Test that an estimation for another technique is rejected."""
        with pytest.raises(ValueError, match="cannot process"):
            self._collected().estimate(CDREstimation())

    def test_rem_without_characterization_is_rejected(self):
        """Test that readout mitigation without a characterization is rejected, and is off by default."""
        workflow = self._collected(rem=False)
        with pytest.raises(ValueError, match="not characterized"):
            workflow.estimate(ZNEEstimation(), apply_rem=True)
        assert not workflow.estimate(ZNEEstimation()).readout_mitigated

    def test_reused_readout_errors_replace_the_calibration_job(self):
        """Test that given readout errors replace the calibration job and give the same estimate."""
        reference = self._collected()
        workflow = self._collected(rem=reference.readout_errors)
        assert workflow.submission.calibration_job is None
        result = workflow.estimate(ZNEEstimation(degree=1))
        assert result.readout_mitigated
        np.testing.assert_allclose(result.values, reference.estimate(ZNEEstimation(degree=1)).values)

    def test_readout_errors_missing_a_measured_qubit_are_rejected(self):
        """Test that readout errors missing a measured qubit are rejected."""
        with pytest.raises(ValueError, match=r"\['QB2'\]"):
            _workflow(rem={"QB1": np.eye(2)})

    def test_result_shapes_and_unmitigated_is_variant_zero(self):
        """Test the shapes of the result, and that the unmitigated values are those of variant 0."""
        workflow = self._collected()
        result = workflow.estimate(ZNEEstimation())
        assert result.values.shape == result.unmitigated.shape == (2, 2)
        assert result.readout_mitigated
        assert len(result.term_values) == 2
        assert result.term_values[0].shape == (2, 3)
        np.testing.assert_allclose(result.unmitigated[1], workflow.observables.combine(result.term_values[1][0]))
        assert result.sigma is None

    def test_bootstrap_is_reproducible_and_positive(self):
        """Test that the bootstrap is reproducible with a seed, gives positive sigmas and needs two samples."""
        workflow = self._collected(rem=False)
        first = workflow.estimate(ZNEEstimation(), bootstrap_samples=4, bootstrap_seed=2)
        second = workflow.estimate(ZNEEstimation(), bootstrap_samples=4, bootstrap_seed=2)
        assert first.sigma.shape == (2, 2)
        np.testing.assert_array_equal(first.sigma, second.sigma)
        assert np.all(first.sigma[:, 0] > 0)
        assert first.term_sigma[0].shape == (2, 3)
        with pytest.raises(ValueError, match="at least 2"):
            workflow.estimate(ZNEEstimation(), bootstrap_samples=1)

    def test_exponential_decay_extrapolates_to_the_exact_value(self):
        # Global depolarizing noise makes every expectation value decay exactly exponentially with the noise factor.
        """Test that exponential extrapolation of depolarizing noise recovers the exact value."""
        thetas = [0.5, 1.1, 2.0]
        _connect(ExactExecutor(depolarizing=0.05))
        workflow = ErrorReductionWorkflow(
            [rotated_circuit(theta) for theta in thetas],
            "Z",
            ZNE(noise_factors=(1, 3, 5)),
            client=CLIENT,
            shots=1000,
        )
        workflow.plan()
        workflow.submit()
        workflow.collect()
        result = workflow.estimate(ZNEEstimation(extrapolation_type="Exp"))
        np.testing.assert_allclose(result.values[:, 0], np.cos(thetas), atol=1e-9)
        np.testing.assert_allclose(result.unmitigated[:, 0], 0.95 * np.cos(thetas), atol=1e-9)


class TestRun:
    def test_run_gives_the_same_result_as_the_four_stages(self):
        """Test that run() gives the same plan and result as the four stages."""
        stepwise = _workflow()
        stepwise.plan()
        stepwise.submit()
        stepwise.collect()
        expected = stepwise.estimate(ZNEEstimation(degree=1))

        workflow = _workflow()
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            warnings.filterwarnings("ignore", message="These Pauli terms extrapolated outside")
            result = workflow.run(ZNEEstimation(degree=1))
        assert result is workflow.result
        assert workflow.execution_plan == stepwise.execution_plan
        np.testing.assert_array_equal(result.values, expected.values)

    def test_run_continues_without_resubmitting(self):
        """Test that run() reuses the submission and the counts of earlier stages."""
        workflow = _workflow()
        workflow.plan()
        workflow.submit()
        submission = workflow.submission
        workflow.run(ZNEEstimation(degree=1))
        assert workflow.submission is submission
        counts = workflow.counts
        workflow.run(ZNEEstimation())
        assert workflow.counts is counts

    @pytest.mark.parametrize(
        ("estimation", "kwargs", "match"),
        [
            (CDREstimation(), {}, "cannot process"),
            (ZNEEstimation(), {"bootstrap_samples": 1}, "at least 2"),
            (ZNEEstimation(), {"bootstrap_samples": 2, "bootstrap_seed": -1}, "negative"),
        ],
    )
    def test_invalid_arguments_are_rejected_before_anything_runs(self, estimation, kwargs, match):
        """Test that run() rejects invalid estimate() arguments before planning."""
        workflow = _workflow()
        with pytest.raises(ValueError, match=match):
            workflow.run(estimation, **kwargs)
        assert workflow.execution_plan is None
        assert workflow.submission is None

    def test_rem_without_characterization_is_rejected_before_anything_runs(self):
        """Test that run() rejects readout mitigation without a characterization before planning."""
        workflow = _workflow(rem=False)
        with pytest.raises(ValueError, match="not characterized"):
            workflow.run(ZNEEstimation(), apply_rem=True)
        assert workflow.execution_plan is None


class TestPersistence:
    def test_round_trip_reproduces_plan_counts_and_estimate(self, tmp_path):
        """Test that save() and load() reproduce the plan, submission, counts and estimate."""
        workflow = _workflow()
        workflow.plan()
        workflow.submit()
        workflow.collect()
        path = tmp_path / "workflow.json"
        workflow.save(path)

        restored = ErrorReductionWorkflow.load(path)
        assert restored.execution_plan == workflow.execution_plan
        assert restored.submission == workflow.submission
        assert restored.counts == workflow.counts
        assert restored.technique == workflow.technique
        expected = workflow.estimate(ZNEEstimation(degree=1))
        actual = restored.estimate(ZNEEstimation(degree=1))
        np.testing.assert_array_equal(actual.values, expected.values)

    def test_planned_workflow_can_be_submitted_after_reload(self, tmp_path):
        """Test that a planned workflow can be submitted and collected after load()."""
        workflow = _workflow(rem=False)
        workflow.plan()
        path = tmp_path / "planned.json"
        workflow.save(path)
        _connect(ExactExecutor())
        restored = ErrorReductionWorkflow.load(path, client=CLIENT)
        restored.submit()
        assert set(restored.collect()) == {key for batch in workflow.execution_plan.batches for key in batch.keys}

    def test_without_a_client_a_loaded_workflow_only_estimates(self, tmp_path):
        """Test that a workflow loaded without a client cannot be submitted."""
        workflow = _workflow(rem=False)
        workflow.plan()
        workflow.save(tmp_path / "planned.json")
        restored = ErrorReductionWorkflow.load(tmp_path / "planned.json")
        with pytest.raises(RuntimeError, match="Load the workflow with a client"):
            restored.submit()


THETAS = (0.7, 1.9)


def _qiskit_target():
    """Rotate qubit 0 by THETAS[0] and qubit 1 by THETAS[1], measuring qubit 0 into clbit 1 and qubit 1 into clbit 0."""
    qc = QuantumCircuit(2, 2)
    qc.r(THETAS[0], 0.0, 0)
    qc.r(THETAS[1], 0.0, 1)
    qc.measure(0, 1)
    qc.measure(1, 0)
    return qc


def _qrisp_target():
    """Rotate qubit i by THETAS[i] and measure it into clbit i."""
    qc = QrispQuantumCircuit(2)
    qc.rx(THETAS[0], 0)
    qc.rx(THETAS[1], 1)
    qc.measure([0, 1])
    return transpile_to_iqm(qc, [(0, 1)])


# Rotation angle that ends up in clbit 0 and clbit 1, for each kind of target.
TARGETS = [(_qiskit_target, (THETAS[1], THETAS[0])), (_qrisp_target, (THETAS[0], THETAS[1]))]


class TestTargetConversion:
    @pytest.mark.parametrize(("make_target", "clbit_thetas"), TARGETS)
    def test_qiskit_and_qrisp_targets_become_pulse_circuits_with_the_same_distribution(self, make_target, clbit_thetas):
        """Test that a qiskit or Qrisp target becomes a pulse circuit with the distribution of the original."""
        backend = IQMFakeApollo()
        workflow = ErrorReductionWorkflow([make_target()], "ZZ", ZNE(), client=CLIENT, backend=backend)
        circuit = workflow.targets[0].circuit
        assert isinstance(circuit, Circuit)
        assert set(workflow.targets[0].qubit_to_clbit) <= set(backend.architecture.qubits)
        p0, p1 = (np.sin(theta / 2) ** 2 for theta in clbit_thetas)
        expected = {"00": (1 - p0) * (1 - p1), "01": p0 * (1 - p1), "10": (1 - p0) * p1, "11": p0 * p1}
        assert exact_distribution(circuit) == pytest.approx(expected, abs=1e-12)


class TestEndToEnd:
    @pytest.mark.parametrize(("make_target", "clbit_thetas"), TARGETS)
    @pytest.mark.parametrize(
        ("technique", "estimation"),
        [
            (ZNE(noise_factors=(1, 3, 5)), ZNEEstimation(extrapolation_type="Exp")),
            (CDR(identity_as_delay=False, increase_combined_value_spread=False, seed=0), CDREstimation()),
        ],
    )
    def test_global_depolarizing_noise_is_removed_exactly(self, make_target, clbit_thetas, technique, estimation):
        """Test that ZNE and CDR recover the exact values of qiskit and Qrisp targets under depolarizing noise."""
        _connect(ExactExecutor(depolarizing=0.05))
        workflow = ErrorReductionWorkflow(
            [make_target()], ["IZ", "ZI"], technique, client=CLIENT, backend=IQMFakeApollo(), shots=1000
        )
        result = workflow.run(estimation)
        assert result.values.shape == (1, 2)
        np.testing.assert_allclose(result.values[0], np.cos(clbit_thetas), atol=1e-9)
        assert not np.allclose(result.unmitigated[0], np.cos(clbit_thetas), atol=1e-3)
