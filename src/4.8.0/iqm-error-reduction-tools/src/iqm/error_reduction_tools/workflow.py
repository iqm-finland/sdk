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

"""Error reduction workflow in four steps: plan, submit, collect and estimate."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any
import warnings

import numpy as np

from iqm.pulse import Circuit

from .circuits import CircuitKey, TargetCircuit
from .estimation import EstimationResult, measure_term_values, resample_counts
from .execution import Executor, ResonanceExecutor, Submission
from .observables import ObservableLike, ObservableSet
from .plan import CalibrationPlan, ExecutionPlan, build_plan
from .readout_characterization.data_processing import CalibrationData, compute_error_probabilities
from .readout_characterization.readout_error_characterization import RECConfiguration
from .rem.rem_processors import ReadoutErrorMitigation
from .technique import Estimation, TargetEstimate, Technique
from .twirling.twirling_api import TwirlingConfiguration
from .twirling.twirling_processors import sum_counts
from .utils.qiskit_utils import canonicalize_circuit
from .utils.serialization import decode, encode

if TYPE_CHECKING:
    from iqm.iqm_client import IQMClient
    from iqm.qiskit_iqm.iqm_backend import IQMBackendBase

MIN_BOOTSTRAP_SAMPLES = 2
"""Fewest bootstrap samples a standard deviation can be taken over."""


class ErrorReductionWorkflow:
    """Workflow to estimate observables of target circuits with an error reduction technique.

    The four stages are separate methods, and each stores its output on the workflow: :meth:`plan` sets
    :attr:`execution_plan`, :meth:`submit` sets :attr:`submission`, :meth:`collect` sets :attr:`counts`, and
    :meth:`estimate` sets :attr:`result`. :meth:`run` does all four in one call.

    :meth:`submit` and :meth:`collect` use the quantum computer. :meth:`plan` and :meth:`estimate` do not,
    and can be repeated with other settings.

    :meth:`save` and :meth:`load` can be used after the collection and before the estimation, respectively,
    to store and retrieve the workflow state from a file.

    Args:
        targets: Circuits whose observables are estimated, transpiled to the native gate set and the physical qubits
            of the quantum computer: :class:`iqm.pulse.Circuit`, qiskit or Qrisp circuits, or anything else
            :func:`~iqm.error_reduction_tools.utils.qiskit_utils.canonicalize_circuit` accepts. They are converted
            to :class:`iqm.pulse.Circuit`.
        observables: One observable or a sequence of observables, each a sum of I/Z Pauli terms, applied to every
            target. See :meth:`~iqm.error_reduction_tools.observables.ObservableSet.build`.
        technique: The error reduction technique, e.g. :class:`~iqm.error_reduction_tools.zne.ZNE` or
            :class:`~iqm.error_reduction_tools.cdr.CDR`.
        client: Client connected to the IQM quantum computer the circuits run on.
        compilation_options: Extra entries for the compiler context, e.g. a dynamical decoupling strategy. The
            readout calibration job is compiled without them.
        twirling_configuration: Twirling configuration, or ``None`` to execute every variant once as it is.
        rec_configuration: Readout error characterization configuration, or ``None`` for no readout error
            mitigation. Readout assignment matrices of an earlier characterization, e.g. the :attr:`readout_errors`
            of another workflow, are used instead of a new calibration job.
        shots: Total shots of each variant, split over its twirling instances. ``None`` uses 20 000. A technique can
            set its own shots for some variants, which override these with a warning.
        backend: IQM Qiskit backend the targets were transpiled for, used to name the qubits of qiskit and Qrisp
            targets. ``None`` names qubit ``i`` ``QB{i + 1}``.

    Raises:
        TypeError: If a target is of an unsupported type.
        ValueError: If the reused readout assignment matrices do not cover every measured qubit.

    """

    def __init__(  # noqa: PLR0913
        self,
        targets: Sequence[object],
        observables: ObservableLike | Sequence[ObservableLike],
        technique: Technique,
        *,
        client: IQMClient,
        compilation_options: dict[str, Any] | None = None,
        twirling_configuration: TwirlingConfiguration | None = None,
        rec_configuration: RECConfiguration | Mapping[str, np.ndarray] | None = None,
        shots: int | None = None,
        backend: IQMBackendBase | None = None,
    ) -> None:
        self._setup(
            [
                canonicalize_circuit(target, name=f"target_{index}", backend=backend)[0]
                for index, target in enumerate(targets)
            ],
            observables,
            technique,
            executor=ResonanceExecutor(client, compilation_options),
            twirling_configuration=twirling_configuration,
            rec_configuration=rec_configuration,
            shots=shots,
        )

    def _setup(
        self,
        targets: Sequence[Circuit],
        observables: ObservableLike | Sequence[ObservableLike],
        technique: Technique,
        *,
        executor: Executor | None,
        twirling_configuration: TwirlingConfiguration | None,
        rec_configuration: RECConfiguration | Mapping[str, np.ndarray] | None,
        shots: int | None,
    ) -> None:
        """Validate the inputs and set the initial state. Without an executor, only offline stages run."""
        if not targets:
            raise ValueError("At least one target circuit is required.")
        if shots is not None and shots < 1:
            raise ValueError(f"shots must be at least 1, got {shots}.")

        for index, circuit in enumerate(targets):
            if not isinstance(circuit, Circuit):
                raise TypeError(f"Target {index} is a {type(circuit).__name__}, but targets must be iqm.pulse.Circuit.")
        self.targets: tuple[TargetCircuit, ...] = tuple(
            TargetCircuit(target_index=index, circuit=circuit) for index, circuit in enumerate(targets)
        )
        self.observables = ObservableSet.build(observables)
        for target in self.targets:
            if target.num_clbits != self.observables.num_clbits:
                raise ValueError(
                    f"Target {target.target_index} measures {target.num_clbits} classical bits, but the observables "
                    f"act on {self.observables.num_clbits}."
                )
        self.technique = technique
        self._executor = executor
        self.twirling_configuration = twirling_configuration
        reused = None
        if isinstance(rec_configuration, Mapping):
            reused, rec_configuration = dict(rec_configuration), None
            missing = sorted({qubit for target in self.targets for qubit in target.qubit_to_clbit} - set(reused))
            if missing:
                raise ValueError(f"The readout assignment matrices do not cover the measured qubits {missing}.")
        self.rec_configuration = rec_configuration
        self.shots = shots

        self.execution_plan: ExecutionPlan | None = None
        """Every circuit to execute, set by :meth:`plan`."""
        self.submission: Submission | None = None
        """The submitted jobs, set by :meth:`submit`."""
        self.counts: dict[CircuitKey, dict[str, int]] | None = None
        """Counts of every twirling instance, set by :meth:`collect`."""
        self.readout_errors: dict[str, np.ndarray] | None = reused
        """Readout assignment matrix of each qubit, given at construction or set by :meth:`collect`."""
        self.result: EstimationResult | None = None
        """The latest estimate, set by :meth:`estimate`."""

    def plan(self) -> ExecutionPlan:
        """Build the variants of every target, twirl them and group them into jobs.

        Returns:
            The plan, also stored as :attr:`execution_plan`.

        Raises:
            RuntimeError: If the workflow has already been submitted.
            ValueError: If the executor cannot run twirled circuits, or the ``"LOCAL"`` strategy has no topology.

        """
        if self.submission is not None:
            raise RuntimeError("The workflow has already been submitted, so its plan can no longer change.")
        topology = None
        if self.twirling_configuration is not None and self._executor is not None:
            if not self._executor.supports_twirling:
                raise ValueError(
                    f"{self._executor.quantum_computer} does not support twirling. Set twirling_configuration=None."
                )
            if self.twirling_configuration.readout_twirl_strategy == "LOCAL":
                topology = self._executor.topology()
        self.execution_plan = build_plan(
            self.targets,
            self.observables,
            self.technique,
            shots=self.shots,
            twirling=self.twirling_configuration,
            rem=self.rec_configuration,
            topology=topology,
        )
        return self.execution_plan

    @property
    def circuits(self) -> list[list[list[Circuit]]]:
        """Return the circuit of every twirling instance, nested by target, variant and instance.

        ``circuits[i][j][k]`` is twirling instance ``k`` of variant ``j`` of target ``i``. The readout calibration
        circuits belong to no target and are not included, see :attr:`execution_plan`.

        Raises:
            RuntimeError: If the workflow has not been planned.

        """
        if self.execution_plan is None:
            raise RuntimeError("The workflow has not been planned. Call plan() first.")
        return [
            [[instance.circuit for instance in variant.instances()] for variant in target.variants()]
            for target in self.execution_plan.targets
        ]

    def submit(self) -> Submission:
        """Submit every job of the plan, planning first if needed.

        Returns:
            The submission, also stored as :attr:`submission`.

        Raises:
            RuntimeError: If the workflow was loaded without a client, or has already been submitted.

        """
        executor = self._require_executor()
        if self.submission is not None:
            raise RuntimeError("The workflow has already been submitted.")
        if self.execution_plan is None:
            warnings.warn("The workflow has not been planned yet, planning it now.", stacklevel=2)
            plan = self.plan()
        else:
            plan = self.execution_plan
        calibration_job = (
            None
            if plan.calibration is None
            else executor.submit(plan.calibration.circuits, plan.calibration.shots, calibration=True)
        )
        jobs = tuple(
            executor.submit([plan.instance(key).circuit for key in batch.keys], batch.shots) for batch in plan.batches
        )
        self.submission = Submission(
            quantum_computer=executor.quantum_computer, jobs=jobs, calibration_job=calibration_job
        )
        return self.submission

    def collect(self) -> dict[CircuitKey, dict[str, int]]:
        """Wait for every job and store the counts of every twirling instance.

        Returns:
            The counts, also stored as :attr:`counts`.

        Raises:
            RuntimeError: If the workflow was loaded without a client, has not been submitted, or the client is
                connected to another quantum computer than the one the jobs were submitted to.

        """
        executor = self._require_executor()
        if self.submission is None or self.execution_plan is None:
            raise RuntimeError("The workflow has not been submitted. Call submit() first.")
        plan = self.execution_plan
        if executor.quantum_computer != self.submission.quantum_computer:
            raise RuntimeError(
                f"The jobs were submitted to {self.submission.quantum_computer!r}, but the client is connected to "
                f"{executor.quantum_computer!r}."
            )
        counts: dict[CircuitKey, dict[str, int]] = {}
        for batch, job in zip(plan.batches, self.submission.jobs, strict=True):
            results = executor.collect(job)
            if len(results) != len(batch.keys):
                raise RuntimeError(f"Job {job.job_id} returned {len(results)} results for {len(batch.keys)} circuits.")
            counts.update(zip(batch.keys, results))
        if plan.calibration is not None and self.submission.calibration_job is not None:
            calibration_counts = executor.collect(self.submission.calibration_job)
            self.readout_errors = _readout_errors(plan.calibration, calibration_counts)
        self.counts = counts
        return counts

    def estimate(
        self,
        estimation: Estimation,
        *,
        apply_rem: bool | None = None,
        bootstrap_samples: int = 0,
        bootstrap_seed: int | None = 0,
    ) -> EstimationResult:
        """Estimate the observables of every target from the collected counts.

        Args:
            estimation: Post-processing settings of the technique, e.g.
                :class:`~iqm.error_reduction_tools.zne.ZNEEstimation`.
            apply_rem: Whether to mitigate readout errors. ``None`` mitigates them when they were characterized.
            bootstrap_samples: Number of bootstrap samples for the standard deviations, ``0`` for none. Each sample
                resamples the counts of every instance and repeats the whole estimation.
            bootstrap_seed: Seed of the resampling.

        Returns:
            The estimate, also stored as :attr:`result`.

        Raises:
            RuntimeError: If no counts have been collected.
            ValueError: If the estimation does not match the technique, readout errors are to be mitigated but were
                not characterized, or ``bootstrap_samples`` is 1.

        """
        self._check_estimate_arguments(estimation, apply_rem, bootstrap_samples, bootstrap_seed)
        if self.counts is None or self.execution_plan is None:
            raise RuntimeError("No counts have been collected. Call collect() first.")
        plan = self.execution_plan
        mitigate = self.readout_errors is not None if apply_rem is None else apply_rem
        if mitigate and self.readout_errors is None:
            raise ValueError("Readout errors were not characterized. Set rec_configuration, or set apply_rem=False.")
        mitigation = ReadoutErrorMitigation(dict(self.readout_errors)) if mitigate and self.readout_errors else None

        term_values, estimates = self._estimate_once(plan, self.counts, estimation, mitigation)
        sigma = term_sigma = None
        if bootstrap_samples:
            rng = np.random.default_rng(bootstrap_seed)
            samples_values, samples_terms = [], []
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for _ in range(bootstrap_samples):
                    terms, sample = self._estimate_once(plan, resample_counts(self.counts, rng), estimation, mitigation)
                    samples_terms.append(terms)
                    samples_values.append([estimate.values for estimate in sample])
            sigma = np.std(np.asarray(samples_values, dtype=float), axis=0, ddof=1)
            term_sigma = tuple(
                np.std(np.asarray([sample[index] for sample in samples_terms], dtype=float), axis=0, ddof=1)
                for index in range(len(plan.targets))
            )

        self.result = EstimationResult(
            values=np.asarray([estimate.values for estimate in estimates], dtype=float),
            unmitigated=np.asarray([self.observables.combine(values[0]) for values in term_values], dtype=float),
            sigma=sigma,
            term_values=term_values,
            term_sigma=term_sigma,
            targets=estimates,
            readout_mitigated=mitigation is not None,
        )
        return self.result

    def run(
        self,
        estimation: Estimation,
        *,
        apply_rem: bool | None = None,
        bootstrap_samples: int = 0,
        bootstrap_seed: int | None = 0,
    ) -> EstimationResult:
        """Run every remaining stage, from planning to estimation, and wait for the quantum computer.

        Stages that have already run are not repeated, so a loaded or partly run workflow continues where it
        stopped. The arguments are checked before anything is submitted.

        Args:
            estimation: Post-processing settings of the technique, e.g.
                :class:`~iqm.error_reduction_tools.zne.ZNEEstimation`.
            apply_rem: Passed to :meth:`estimate`.
            bootstrap_samples: Passed to :meth:`estimate`.
            bootstrap_seed: Passed to :meth:`estimate`.

        Returns:
            The estimate, also stored as :attr:`result`.

        Raises:
            ValueError: If an argument is invalid, see :meth:`estimate`.

        """
        self._check_estimate_arguments(estimation, apply_rem, bootstrap_samples, bootstrap_seed)
        if self.execution_plan is None:
            self.plan()
        if self.submission is None:
            self.submit()
        if self.counts is None:
            self.collect()
        return self.estimate(
            estimation, apply_rem=apply_rem, bootstrap_samples=bootstrap_samples, bootstrap_seed=bootstrap_seed
        )

    def save(self, path: str | Path) -> None:
        """Write the configuration, plan, submission, counts and readout errors to a JSON file.

        Args:
            path: Destination file.

        """
        data = {
            "targets": encode([target.circuit for target in self.targets]),
            "observables": [
                [[complex(c).real, label] for label, c in zip(op.paulis.to_labels(), op.coeffs)]
                for op in self.observables.observables
            ],
            "technique": encode(self.technique),
            "twirling_configuration": encode(self.twirling_configuration),
            "rec_configuration": encode(self.rec_configuration),
            "shots": self.shots,
            "execution_plan": encode(self.execution_plan),
            "submission": encode(self.submission),
            "counts": None
            if self.counts is None
            else [{"key": list(key), "counts": counts} for key, counts in self.counts.items()],
            "readout_errors": encode(self.readout_errors),
        }
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(
        cls, path: str | Path, client: IQMClient | None = None, compilation_options: dict[str, Any] | None = None
    ) -> ErrorReductionWorkflow:
        """Read a workflow written by :meth:`save`.

        Args:
            path: The file.
            client: Client connected to the quantum computer, needed to plan, submit or collect. ``None`` is enough to
                estimate from saved counts.
            compilation_options: Extra entries for the compiler context, e.g. a dynamical decoupling strategy.

        Returns:
            The workflow, in the stage it was saved in.

        """
        data = json.loads(Path(path).read_text())
        workflow = cls.__new__(cls)
        workflow._setup(
            decode(data["targets"]),
            [[(coefficient, label) for coefficient, label in observable] for observable in data["observables"]],
            decode(data["technique"]),
            executor=None if client is None else ResonanceExecutor(client, compilation_options),
            twirling_configuration=decode(data["twirling_configuration"]),
            rec_configuration=decode(data["rec_configuration"]),
            shots=data["shots"],
        )
        workflow.execution_plan = decode(data["execution_plan"])
        workflow.submission = decode(data["submission"])
        if data["counts"] is not None:
            workflow.counts = {CircuitKey(*entry["key"]): entry["counts"] for entry in data["counts"]}
        workflow.readout_errors = decode(data["readout_errors"])
        return workflow

    def _check_estimate_arguments(
        self, estimation: Estimation, apply_rem: bool | None, bootstrap_samples: int, bootstrap_seed: int | None
    ) -> None:
        """Reject estimate() arguments that would fail, before any circuits run."""
        if not estimation.accepts(self.technique):
            raise ValueError(f"{type(estimation).__name__} cannot process variants built by {self.technique!r}.")
        if bootstrap_samples and bootstrap_samples < MIN_BOOTSTRAP_SAMPLES:
            raise ValueError(
                f"bootstrap_samples must be 0 or at least {MIN_BOOTSTRAP_SAMPLES}, got {bootstrap_samples}."
            )
        if bootstrap_samples:
            np.random.default_rng(bootstrap_seed)
        if apply_rem and self.rec_configuration is None and self.readout_errors is None:
            raise ValueError("Readout errors were not characterized. Set rec_configuration, or set apply_rem=False.")

    def _estimate_once(
        self,
        plan: ExecutionPlan,
        counts: Mapping[CircuitKey, Mapping[str, float]],
        estimation: Estimation,
        mitigation: ReadoutErrorMitigation | None,
    ) -> tuple[tuple[np.ndarray, ...], tuple[TargetEstimate, ...]]:
        """Compute the term values of every variant and the estimate of every target from one set of counts."""
        readout_twirled = plan.readout_twirled
        term_values: list[np.ndarray] = []
        estimates: list[TargetEstimate] = []
        # A plain loop, so that warnings raised by the estimation point at the caller of estimate().
        for target in plan.targets:
            values = measure_term_values(target, counts, self.observables, mitigation, readout_twirled)
            term_values.append(values)
            estimates.append(estimation.estimate(target, values, self.observables))
        return tuple(term_values), tuple(estimates)

    def _require_executor(self) -> Executor:
        """Return the executor, or explain that the stage needs one."""
        if self._executor is None:
            raise RuntimeError("This stage runs circuits. Load the workflow with a client.")
        return self._executor


def _readout_errors(calibration: CalibrationPlan, counts: Sequence[Mapping[str, int]]) -> dict[str, np.ndarray]:
    """Compute the readout assignment matrix of each qubit from the counts of the calibration circuits."""
    grouped: dict[str, list[dict[str, float]]] = {}
    for prep_string, distribution in zip(calibration.prep_strings, counts, strict=True):
        grouped.setdefault(prep_string, []).append(dict(distribution))
    data: CalibrationData = {
        "counts_by_prep": {prep_string: sum_counts(group) for prep_string, group in grouped.items()},
        "measured_qubits": list(calibration.qubits),
    }
    return dict(compute_error_probabilities(data)["charact_data"])
