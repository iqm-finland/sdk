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

"""High level REM (Readout Error Mitigation) workflow API.

Combines readout error characterization, circuit twirling and
post-processing into a single lifecycle-oriented interface.

Typical usage::

    from iqm.error_reduction_tools.rem import REMWorkflow, WorkflowConfiguration
    from iqm.error_reduction_tools.twirling.twirling_api import TwirlingConfiguration

    config = WorkflowConfiguration(shots=20_000, twirling=TwirlingConfiguration(readout_twirl_strategy="LOCAL"))
    results = REMWorkflow(client, config=config).run(circuits, observables=["ZZII", "IIZZ"])

    print(results.mitigated_distributions)
    print(results.expectation_values)

"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
import json
import logging
from typing import TYPE_CHECKING, Any

from iqm.error_reduction_tools.readout_characterization import ReadoutErrorCharacterization, RECConfiguration

from .rem_processors import ReadoutErrorMitigation
from ..twirling.twirling_api import CircuitTwirler, TwirlingConfiguration

if TYPE_CHECKING:
    from iqm.iqm_client import IQMClient


logger = logging.getLogger(__name__)


ObservableSpecs = list[list[str] | str | list[int]]
"""Observable specifications for a single circuit.

Each element selects one observable, in any of the formats accepted by
:meth:`.ReadoutErrorMitigation.mitigate_counts`: qubit names (``["QB3", "QB5"]``),
a Z/I Pauli string (``"ZZII"``), or classical bit indices (``[0, 1]``).
"""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class WorkflowConfiguration:
    """Configuration for the full readout error mitigation (REM) workflow.

    Composes sub-configurations for readout error characterization and
    circuit twirling, plus workflow-level parameters.

    Example::

        config = WorkflowConfiguration(
            shots=40_000,
            twirling=TwirlingConfiguration(readout_twirl_strategy="LOCAL", seed=42),
        )

        # REM with custom compiler context (e.g. dynamical decoupling):
        config_with_dd = WorkflowConfiguration(
            shots=40_000,
            twirling=TwirlingConfiguration(readout_twirl_strategy="LOCAL"),
            compilation_options={"DDStrategy": my_dd_strategy},
        )

    """

    rec: RECConfiguration = field(default_factory=RECConfiguration)
    """Configuration for readout error characterization."""

    twirling: TwirlingConfiguration = field(default_factory=lambda: TwirlingConfiguration(circuit_twirling=False))
    """Configuration for circuit twirling."""

    shots: int = 20_000
    """Number of measurement shots per input circuit, distributed across that circuit's twirled instances."""

    nearest_probability: bool = True
    """Project quasi-probabilities to nearest valid distribution."""

    force_mitigation: bool = False
    """Apply mitigation even when complexity exceeds threshold."""

    compilation_options: dict[str, Any] | None = None
    """Extra key-value pairs passed to the compiler context during circuit compilation.
    
    Use this to activate compiler features such as dynamical decoupling.
    """


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class REMMetadata:
    """Metadata about a completed REM workflow run."""

    timestamp: datetime
    """ISO-8601 timestamp when results were retrieved, in UTC."""

    shots: int
    """Total shots used for circuit execution."""

    twirling_strategy: str
    """Twirling strategy name used."""

    characterization_reused: bool
    """``True`` if the user provided pre-existing characterization."""

    characterization_job_id: str | None
    """Id of the readout error characterization job, or ``None`` when a
    pre-existing characterization was reused (no job was submitted)."""

    twirling_job_id: str | None
    """Id of the twirled-circuit job submitted to the quantum computer."""

    rec_config: RECConfiguration
    """Configuration used for readout error characterization."""

    twirling_config: TwirlingConfiguration
    """Configuration used for circuit twirling."""


@dataclass
class REMResults:
    """Output of a completed REM workflow.

    Example::

        results = workflow.get_results()
        print(results.mitigated_distributions[0])
        results.characterization.save("charact.json")

    """

    mitigated_distributions: list[dict[str, float]]
    """One mitigated distribution per input circuit.

    Each distribution sums to 1. With ``nearest_probability``
    enabled (the default) the values form a valid probability distribution;
    otherwise they are quasi-probabilities that may be negative.
    """

    expectation_values: list[list[float]] | None
    """Observable expectation values per circuit (``None`` if no observables).

    ``expectation_values[i][j]`` is the value of observable ``j`` for circuit ``i``, following the
    order of the ``observables`` (or the circuit's entry of ``observables_per_circuit``) passed to
    :meth:`REMWorkflow.submit`.
    """

    raw_counts: list[dict[str, float]]
    """One untwirled (but unmitigated) count distribution per input circuit."""

    characterization: ReadoutErrorCharacterization
    """The :class:`ReadoutErrorCharacterization` object (saveable for reuse)."""

    metadata: REMMetadata
    """Workflow metadata."""


# ---------------------------------------------------------------------------
# REMWorkflow
# ---------------------------------------------------------------------------


class REMWorkflow:
    """End-to-end readout error mitigation workflow.

    Combines readout error characterization, circuit twirling, and
    postprocessing behind a clean, lifecycle-oriented API.

    **Two-step (async) usage** — submit jobs, do other things, then retrieve::

        workflow = REMWorkflow(client, config=config)
        workflow.submit(circuits, observables=["ZZII", "IIZZ"])
        # ... user can do other things while QPU jobs run ...
        results = workflow.get_results()

    **One-liner (blocking) usage** — for notebooks and demos::

        results = REMWorkflow(client).run(circuits)

    **Reusing previous characterization**::

        workflow = REMWorkflow(client, characterization="charact.json")
        results = workflow.run(circuits)

    Args:
        client: Client for connecting to the quantum computer.
        config: Workflow configuration.  Defaults are sensible for most cases.
        characterization: Pre-existing readout error characterization.
            Accepts a :class:`.ReadoutErrorCharacterization` object,
            a ``dict`` (from ``rec.to_dict()``), or a file path ``str``.
            When ``None``, characterization is run automatically during
            :meth:`submit`.

    """

    def __init__(
        self,
        client: IQMClient,
        *,
        config: WorkflowConfiguration | None = None,
        characterization: ReadoutErrorCharacterization | dict | str | None = None,
    ) -> None:
        self._client = client
        self._config = config or WorkflowConfiguration()
        self._rec: ReadoutErrorCharacterization | None = None
        self._twirler: CircuitTwirler | None = None
        self._observables: ObservableSpecs | None = None
        self._observables_per_circuit: list[ObservableSpecs] | None = None
        self._results: REMResults | None = None
        self._characterization_reused: bool = False
        self._restored_characterization_job_id: str | None = None

        if characterization is not None:
            self.load_characterization(characterization)

    # ------------------------------------------------------------------
    # Characterization reuse
    # ------------------------------------------------------------------

    def load_characterization(self, source: ReadoutErrorCharacterization | dict | str) -> REMWorkflow:
        """Load pre-existing readout error characterization.

        Args:
            source: A :class:`ReadoutErrorCharacterization` object, a dict
                (as returned by ``rec.to_dict()``), or a JSON file path.

        Returns:
            ``self`` for method chaining.

        Raises:
            TypeError: If the source type is not supported.

        """
        if isinstance(source, ReadoutErrorCharacterization):
            self._rec = source
        elif isinstance(source, dict):
            self._rec = ReadoutErrorCharacterization.from_dict(source)
        elif isinstance(source, str):
            self._rec = ReadoutErrorCharacterization.load(source)
        else:
            raise TypeError(
                f"Unsupported characterization source type: {type(source).__name__}. "
                "Expected ReadoutErrorCharacterization, dict, or str (file path)."
            )
        self._characterization_reused = True
        return self

    # ------------------------------------------------------------------
    # Two-step async execution
    # ------------------------------------------------------------------

    def submit(
        self,
        circuits: list,
        observables: ObservableSpecs | None = None,
        *,
        observables_per_circuit: list[ObservableSpecs] | None = None,
    ) -> REMWorkflow:
        """Submit characterization and twirled circuit jobs to the QPU.

        If no characterization was loaded, a REC job is submitted automatically.
        All input circuits are twirled and submitted as a separate job.

        Args:
            circuits: Quantum circuits to mitigate.  Accepts any type
                supported by
                :meth:`~iqm.error_reduction_tools.twirling.twirling_api.CircuitTwirler.twirl`:
                Qiskit :class:`~qiskit.circuit.QuantumCircuit`,
                :class:`~iqm.pulse.Circuit`,
                :class:`~iqm.error_reduction_tools.utils.circuit_utils.TwirledCircuit`,
                or any object with a ``to_qiskit()`` method (e.g.
                :class:`qrisp.QuantumCircuit`).
            observables: Observable specifications applied to every circuit.
                Supports qubit names (``[["QB3", "QB5"]]``), Pauli strings
                (``["ZZII"]``), or bit indices (``[[0, 1]]``).
                Pass ``None`` to mitigate the full distribution only.
            observables_per_circuit: Observable specifications per circuit, one
                :data:`ObservableSpecs` list per entry of ``circuits`` and in the same order.
                Use this when different circuits carry different observables, e.g. the
                Pauli groups of a Hamiltonian::

                    workflow.submit(circuits, observables_per_circuit=[["ZZII"], ["IIZZ", "ZIIZ"]])

                It is a separate argument because ``[["QB3", "QB5"], ["QB1"]]`` could mean two
                observables or one list per circuit. Cannot be combined with ``observables``.

        Returns:
            ``self`` for method chaining.

        Raises:
            ValueError: If both ``observables`` and ``observables_per_circuit`` are given, or if
                ``observables_per_circuit`` does not have exactly one entry per circuit.

        """
        if observables is not None and observables_per_circuit is not None:
            raise ValueError(
                "Provide either `observables` (shared by all circuits) or `observables_per_circuit` "
                "(one specification list per circuit), not both."
            )
        if observables_per_circuit is not None and len(observables_per_circuit) != len(circuits):
            raise ValueError(
                f"`observables_per_circuit` has {len(observables_per_circuit)} entries "
                f"but {len(circuits)} circuits were given. They must match one-to-one."
            )

        self._observables = observables
        self._observables_per_circuit = observables_per_circuit

        # Step 1: Twirl circuits.  This also extracts, per circuit, which qubits
        # are measured, which we reuse below to scope the characterization.
        self._twirler = CircuitTwirler(
            self._client,
            config=self._config.twirling,
            compilation_options=self._config.compilation_options,
        )
        self._twirler.twirl(circuits)

        # Step 2: Submit REC job if no characterization is available.
        if self._rec is None:
            # Copy so we never mutate the user's WorkflowConfiguration in place.
            rec_config = replace(self._config.rec)
            measured_qubits = {qubit for mapping in self._twirler.get_qubit_to_bit_mapping() for qubit in mapping}
            if rec_config.qubits is not None:
                # User specified an explicit qubit list; respect it.
                logger.info("Characterizing user-specified qubits: %s", rec_config.qubits)
            elif not measured_qubits:
                # Nothing to infer from; leave qubits=None so REC characterizes every
                # operational QPU qubit.
                logger.warning(
                    "Could not infer measured qubits from circuits; falling back to characterizing all "
                    "operational QPU qubits."
                )
            else:
                # Default: scope characterization to exactly the qubits the circuits measure.
                rec_config.qubits = sorted(measured_qubits)
                logger.info("Inferred qubits for REC from circuits: %s", rec_config.qubits)
            self._rec = ReadoutErrorCharacterization(self._client)
            self._rec.submit_job(rec_config)
            self._characterization_reused = False
            logger.info("Submitted readout error characterization job.")

        # Step 3: Submit twirled circuits.
        self._twirler.submit(shots=self._config.shots)
        logger.info("Submitted twirled circuit job.")

        return self

    def get_results(self) -> REMResults:
        """Wait for jobs, retrieve results, and apply readout error mitigation.

        Blocks until both the characterization and circuit jobs complete,
        then performs REM postprocessing and packages all outputs.

        Returns:
            :class:`REMResults` with mitigated counts, expectation values,
            raw counts, characterization data, and metadata.

        Raises:
            RuntimeError: If :meth:`submit` has not been called.

        """
        if self._twirler is None or self._rec is None:
            raise RuntimeError("No jobs to retrieve. Call submit() first.")

        # Step 1: Retrieve characterization results (if we submitted a job).
        if not self._characterization_reused:
            self._rec.retrieve_results()
        error_probs = self._rec.get_readout_error_probabilities()

        # Step 2: Retrieve and untwirl counts from CircuitTwirler.
        raw_counts = self._twirler.retrieve_counts()

        # Step 3: Apply readout error mitigation.
        #   - twirled tells mitigate_counts() whether to symmetrize the
        #     characterization data. This is only correct when the counts were
        #     actually readout-twirled; with strategy="NONE" or a skipped-twirling
        #     fallback the full asymmetric confusion matrix must be used instead.
        rem = ReadoutErrorMitigation(readout_errors=error_probs["charact_data"])
        qubit_to_bit_mappings = self._twirler.get_qubit_to_bit_mapping()
        readout_twirled = self._twirler.readout_twirling_applied

        all_mitigated_counts: list[dict[str, float]] = []
        all_expectation_values: list[list[float]] = []

        for index, (counts, mapping) in enumerate(zip(raw_counts, qubit_to_bit_mappings)):
            observables = (
                self._observables_per_circuit[index] if self._observables_per_circuit is not None else self._observables
            )
            result = rem.mitigate_counts(
                experiment_counts=[counts],  # type: ignore[list-item]
                qubit_to_bit_mapping=mapping,
                observables=observables,
                twirled=readout_twirled,
                force_mitigation=self._config.force_mitigation,
                nearest_probability=self._config.nearest_probability,
            )
            # mitigate_counts returns [circuit_i][observable_j]; we passed 1 circuit.
            all_mitigated_counts.append(result["mitigated_counts"][0][0])
            all_expectation_values.append(result["expectation_values"][0])

        metadata = REMMetadata(
            timestamp=datetime.now(tz=UTC),
            shots=self._config.shots,
            twirling_strategy=self._config.twirling.readout_twirl_strategy,
            characterization_reused=self._characterization_reused,
            characterization_job_id=(
                self._restored_characterization_job_id if self._characterization_reused else self._rec.get_job_id()
            ),
            twirling_job_id=self._twirler.get_job_id(),
            rec_config=self._config.rec,
            twirling_config=self._config.twirling,
        )

        self._results = REMResults(
            mitigated_distributions=all_mitigated_counts,
            expectation_values=(
                all_expectation_values if (self._observables or self._observables_per_circuit) else None
            ),
            raw_counts=raw_counts,
            characterization=self._rec,
            metadata=metadata,
        )

        return self._results

    # Persistence

    def to_dict(self) -> dict[str, Any]:
        """Serialize the workflow so its results can be retrieved in another session.

        Stores what :meth:`get_results` needs: the twirling job id, the rotation strings and mappings, the
        characterization, the observables and the configuration. The circuits are not stored.

        The characterization must already be retrieved or loaded, because a pending characterization job
        cannot be serialized. ``compilation_options`` is not stored, since it only affects submission.

        Returns:
            A JSON-serializable dictionary. Pass it to :meth:`from_dict`.

        Raises:
            RuntimeError: If :meth:`submit` has not been called, or if the characterization
                is not yet available.

        """
        if self._twirler is None or self._rec is None:
            raise RuntimeError("Nothing to serialize. Call submit() first.")

        try:
            characterization = self._rec.to_dict()
        except RuntimeError as error:
            raise RuntimeError(
                "The readout error characterization has not been retrieved yet, so it cannot be "
                "serialized. Call `get_characterization().retrieve_results()` first (it is fast), "
                "or construct the workflow with a pre-existing characterization."
            ) from error

        return {
            "version": 1,
            "config": {
                "rec": asdict(self._config.rec),
                "twirling": asdict(self._config.twirling),
                "shots": self._config.shots,
                "nearest_probability": self._config.nearest_probability,
                "force_mitigation": self._config.force_mitigation,
            },
            "twirler": self._twirler.to_dict(),
            "characterization": characterization,
            "characterization_job_id": self._rec.get_job_id(),
            "characterization_reused": self._characterization_reused,
            "observables": self._observables,
            "observables_per_circuit": self._observables_per_circuit,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], client: IQMClient) -> REMWorkflow:
        """Reconstruct a workflow from :meth:`to_dict`, ready for :meth:`get_results`.

        Args:
            data: Dictionary as returned by :meth:`to_dict`.
            client: Client for connecting to the quantum computer. It must point at the
                same quantum computer the job was submitted to.

        Returns:
            A :class:`REMWorkflow` on which :meth:`get_results` can be called.

        Raises:
            ValueError: If the payload version is not recognized.

        """
        version = data.get("version")
        if version != 1:
            raise ValueError(f"Unsupported REMWorkflow payload version {version!r}; expected 1.")

        stored = data["config"]
        config = WorkflowConfiguration(
            rec=RECConfiguration(**stored["rec"]),
            twirling=TwirlingConfiguration(**stored["twirling"]),
            shots=stored["shots"],
            nearest_probability=stored["nearest_probability"],
            force_mitigation=stored["force_mitigation"],
        )
        workflow = cls(client, config=config)
        workflow._twirler = CircuitTwirler.from_dict(data["twirler"], client)
        workflow._rec = ReadoutErrorCharacterization.from_dict(data["characterization"])
        workflow._observables = data.get("observables")
        workflow._observables_per_circuit = data.get("observables_per_circuit")
        # The characterization travelled with the payload, already retrieved, so
        # get_results() must not try to retrieve it again.
        workflow._characterization_reused = True
        workflow._restored_characterization_job_id = data.get("characterization_job_id")
        return workflow

    def save(self, path: str) -> None:
        """Write :meth:`to_dict` to a JSON file.

        Args:
            path: Destination file path.

        """
        with open(path, "w") as handle:
            json.dump(self.to_dict(), handle, indent=4)
        logger.info("Saved REM workflow to %s", path)

    @classmethod
    def load(cls, path: str, client: IQMClient) -> REMWorkflow:
        """Read a workflow back from a file written by :meth:`save`.

        Args:
            path: Source file path.
            client: Client for the same quantum computer the job was submitted to.

        Returns:
            A :class:`REMWorkflow` on which :meth:`get_results` can be called.

        """
        with open(path) as handle:
            data = json.load(handle)
        return cls.from_dict(data, client)

    # ------------------------------------------------------------------
    # Blocking convenience
    # ------------------------------------------------------------------

    def run(
        self,
        circuits: list,
        observables: ObservableSpecs | None = None,
        *,
        observables_per_circuit: list[ObservableSpecs] | None = None,
    ) -> REMResults:
        """Submit jobs and retrieve results in one blocking call.

        Convenience wrapper equivalent to::

            self.submit(circuits, observables=observables)
            return self.get_results()

        Args:
            circuits: Quantum circuits to mitigate.  Accepts any type
                supported by
                :meth:`~iqm.error_reduction_tools.twirling.twirling_api.CircuitTwirler.twirl`.
            observables: Observable specifications shared by all circuits
                (same format as :meth:`submit`).
            observables_per_circuit: Observable specifications per circuit
                (same format as :meth:`submit`). Mutually exclusive with ``observables``.

        Returns:
            :class:`REMResults` with all outputs.

        """
        self.submit(circuits, observables=observables, observables_per_circuit=observables_per_circuit)
        return self.get_results()

    # ------------------------------------------------------------------
    # Intermediate data accessors
    # ------------------------------------------------------------------

    def get_characterization(self) -> ReadoutErrorCharacterization:
        """Return the :class:`ReadoutErrorCharacterization` (saveable for reuse).

        Raises:
            RuntimeError: If characterization is not yet available.

        """
        if self._rec is None:
            raise RuntimeError("No characterization available. Call submit() and get_results() first.")
        return self._rec

    def get_job_ids(self) -> dict[str, str | None]:
        """Return the ids of the QPU jobs submitted by this workflow.

        Returns:
            Mapping with keys ``"characterization"`` and ``"twirling"``.
            ``"characterization"`` is ``None`` when a pre-existing
            characterization was reused (no characterization job was submitted).

        Raises:
            RuntimeError: If :meth:`submit` has not been called yet.

        """
        if self._twirler is None or self._rec is None:
            raise RuntimeError("No jobs available. Call submit() first.")
        return {
            "characterization": (
                self._restored_characterization_job_id if self._characterization_reused else self._rec.get_job_id()
            ),
            "twirling": self._twirler.get_job_id(),
        }

    def get_raw_counts(self) -> list[dict[str, float]]:
        """Return unmitigated (but untwirled) counts per input circuit.

        Raises:
            RuntimeError: If results are not yet available.

        """
        if self._results is None:
            raise RuntimeError("No results available. Call get_results() first.")
        return self._results.raw_counts

    def get_mitigated_distributions(self) -> list[dict[str, float]]:
        """Return mitigated distributions per input circuit.

        Raises:
            RuntimeError: If results are not yet available.

        """
        if self._results is None:
            raise RuntimeError("No results available. Call get_results() first.")
        return self._results.mitigated_distributions

    def get_expectation_values(self) -> list[list[float]] | None:
        """Return observable expectation values per circuit, or ``None``.

        Returns ``None`` if no observables were specified.

        Raises:
            RuntimeError: If results are not yet available.

        """
        if self._results is None:
            raise RuntimeError("No results available. Call get_results() first.")
        return self._results.expectation_values
