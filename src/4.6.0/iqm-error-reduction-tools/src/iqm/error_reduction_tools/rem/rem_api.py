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

    print(results.mitigated_counts)
    print(results.expectation_values)

"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
import logging
from typing import Any

from iqm.error_reduction_tools.readout_characterization import ReadoutErrorCharacterization, RECConfiguration

from iqm.pulla.pulla import Pulla

from .rem_processors import ReadoutErrorMitigation
from ..twirling.twirling_api import CircuitTwirler, TwirlingConfiguration

logger = logging.getLogger(__name__)


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

    rec_config: RECConfiguration
    """Configuration used for readout error characterization."""

    twirling_config: TwirlingConfiguration
    """Configuration used for circuit twirling."""


@dataclass
class REMResults:
    """Output of a completed REM workflow.

    Example::

        results = workflow.get_results()
        print(results.mitigated_counts[0])
        results.characterization.save("charact.json")

    """

    mitigated_counts: list[dict[str, float]]
    """One mitigated quasi-probability distribution per input circuit."""

    expectation_values: list[list[float]] | None
    """Observable expectation values per circuit (``None`` if no observables)."""

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
        client: Pulla,
        *,
        config: WorkflowConfiguration | None = None,
        characterization: ReadoutErrorCharacterization | dict | str | None = None,
    ) -> None:
        self._client = client
        self._config = config or WorkflowConfiguration()
        self._rec: ReadoutErrorCharacterization | None = None
        self._twirler: CircuitTwirler | None = None
        self._observables: list[list[str] | str | list[int]] | None = None
        self._results: REMResults | None = None
        self._characterization_reused: bool = False

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
        observables: list[list[str] | str | list[int]] | None = None,
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
            observables: Observable specifications for expectation value computation.
                Supports qubit names (``[["QB3", "QB5"]]``), Pauli strings
                (``["ZZII"]``), or bit indices (``[[0, 1]]``).
                Pass ``None`` to mitigate the full distribution only.

        Returns:
            ``self`` for method chaining.

        """
        self._observables = observables

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
        #   - twirled=True tells mitigate_counts() to symmetrize characterization
        #     data without mutating the original (appropriate for twirled counts).
        rem = ReadoutErrorMitigation(readout_errors=error_probs["charact_data"])
        qubit_to_bit_mappings = self._twirler.get_qubit_to_bit_mapping()

        all_mitigated_counts: list[dict[str, float]] = []
        all_expectation_values: list[list[float]] = []

        for counts, mapping in zip(raw_counts, qubit_to_bit_mappings):
            result = rem.mitigate_counts(
                experiment_counts=[counts],  # type: ignore[list-item]
                qubit_to_bit_mapping=mapping,
                observables=self._observables,
                twirled=True,
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
            rec_config=self._config.rec,
            twirling_config=self._config.twirling,
        )

        self._results = REMResults(
            mitigated_counts=all_mitigated_counts,
            expectation_values=all_expectation_values if self._observables else None,
            raw_counts=raw_counts,
            characterization=self._rec,
            metadata=metadata,
        )

        return self._results

    # ------------------------------------------------------------------
    # Blocking convenience
    # ------------------------------------------------------------------

    def run(
        self,
        circuits: list,
        observables: list[list[str] | str | list[int]] | None = None,
    ) -> REMResults:
        """Submit jobs and retrieve results in one blocking call.

        Convenience wrapper equivalent to::

            self.submit(circuits, observables=observables)
            return self.get_results()

        Args:
            circuits: Quantum circuits to mitigate.  Accepts any type
                supported by
                :meth:`~iqm.error_reduction_tools.twirling.twirling_api.CircuitTwirler.twirl`.
            observables: Observable specifications (same format as :meth:`submit`).

        Returns:
            :class:`REMResults` with all outputs.

        """
        self.submit(circuits, observables=observables)
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

    def get_raw_counts(self) -> list[dict[str, float]]:
        """Return unmitigated (but untwirled) counts per input circuit.

        Raises:
            RuntimeError: If results are not yet available.

        """
        if self._results is None:
            raise RuntimeError("No results available. Call get_results() first.")
        return self._results.raw_counts

    def get_mitigated_counts(self) -> list[dict[str, float]]:
        """Return mitigated quasi-probability distributions per input circuit.

        Raises:
            RuntimeError: If results are not yet available.

        """
        if self._results is None:
            raise RuntimeError("No results available. Call get_results() first.")
        return self._results.mitigated_counts

    def get_expectation_values(self) -> list[list[float]] | None:
        """Return observable expectation values per circuit, or ``None``.

        Returns ``None`` if no observables were specified.

        Raises:
            RuntimeError: If results are not yet available.

        """
        if self._results is None:
            raise RuntimeError("No results available. Call get_results() first.")
        return self._results.expectation_values
