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

"""Circuit twirling API for readout error mitigation.

Provides :class:`TwirlingConfiguration` for controlling twirling behaviour and
:class:`CircuitTwirler` for the full twirl → submit → retrieve lifecycle.

Typical usage::

    from iqm.error_reduction_tools.twirling.twirling_api import CircuitTwirler, TwirlingConfiguration

    config = TwirlingConfiguration(readout_twirl_strategy="LOCAL", seed=42)
    twirler = CircuitTwirler(client, config=config)
    twirler.twirl(circuits).submit(shots=20_000)
    counts = twirler.retrieve_counts()

"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
import math
from time import sleep
from typing import TYPE_CHECKING, Any, Literal
import warnings

from iqm.error_reduction_tools.twirling.twirling_processors import untwirl_and_sum_counts
import numpy as np
from qiskit.circuit import QuantumCircuit as QiskitQuantumCircuit
from qrisp import QuantumCircuit as QrispQuantumCircuit  # type: ignore[import-untyped]

from iqm.pulla.utils_qiskit import sweep_job_to_qiskit
from iqm.pulse import Circuit as PulseCircuit

from .twirling_modifiers import randomize_circuit
from ..utils.circuit_utils import TwirledCircuit
from ..utils.qiskit_utils import from_iqm_to_qiskit, from_qiskit_to_iqm
from ..utils.readout_twirling_strings import generate_rot_strings
from ..utils.topology_utils import topology_from_qc, uses_move_gates

if TYPE_CHECKING:
    from iqm.pulla.pulla import Pulla, PullaJob

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TwirlingConfiguration:
    """Configuration for circuit twirling."""

    readout_twirl_strategy: Literal["LOCAL", "MINIMAL", "HADAMARD", "NONE"] = "LOCAL"
    """Readout-twirling strategy.
    
    Supported values are ``"LOCAL"`` (4 strings, requires topology), ``"MINIMAL"`` (2 strings),
    ``"HADAMARD"`` (up to $2^n$ strings), and ``"NONE"`` to disable readout twirling entirely.
    When ``"NONE"`` is combined with ``circuit_twirling=True``, ``num_twirling_instances`` random
    circuit-twirled variants are generated instead.
    """

    seed: int | None = None
    """RNG seed for reproducibility. ``None`` means non-deterministic."""

    circuit_twirling: bool = True
    """Whether to apply Pauli twirling to two-qubit gates ("Poor Man's Pauli Twirling").
    
    When ``False``, the inserted Pauli rotations on single-qubit gates are suppressed;
    readout twirling (if any) still works.
    """

    num_twirling_instances: int = 20
    """Total number of randomized variants per input circuit when ``circuit_twirling=True``.
    
    This budget is split evenly across the readout-twirling rot-strings (whose count is
    fixed by ``readout_twirl_strategy``) so that every rot-string receives the same
    number of independent Pauli randomizations.

    Reconciliation rules (a warning is emitted whenever the effective count differs
    from the requested one):

    * If ``num_twirling_instances`` is a multiple of the number of rot-strings,
      it is used as-is.
    * If it is not a multiple, it is **rounded up** to the next multiple.
    * If it is **smaller** than the number of rot-strings, the number of rot-strings
      is used (one Pauli randomization per rot-string).

    When ``circuit_twirling=False`` this field is ignored: each rot-string is emitted
    exactly once. Default of 20 follows common Pauli-twirling practice (10–50 instances).
    """


# ---------------------------------------------------------------------------
# Circuit Twirler
# ---------------------------------------------------------------------------

_SUPPORTED_STRATEGIES = frozenset(["LOCAL", "MINIMAL", "HADAMARD", "NONE"])


class CircuitTwirler:
    """Manages the twirl → submit → retrieve lifecycle for readout-twirled circuits.

    Args:
        client: Client instance for connecting to an IQM quantum computer. Used for
            topology look-up (only required for the ``"LOCAL"`` readout-twirling
            strategy), compilation, and job submission.  May be ``None`` when
            the twirler is used purely as a circuit transformer — i.e. when
            calling only :meth:`twirl` (with a non-``LOCAL`` strategy) and
            :meth:`get_twirled_circuits` to drive submission externally.
            Required by :meth:`submit` and by the ``"LOCAL"`` strategy.
        config: Twirling configuration.  Defaults are sensible for most cases.
        compilation_options: Extra key-value pairs passed to the Pulla
            compiler context when :meth:`submit` compiles circuits.
            Use this to activate compiler features such as dynamical
            decoupling.  For example::

                compilation_options={"DDStrategy": my_dd_strategy}

    Example — full lifecycle (requires client)::

        twirler = CircuitTwirler(client, config=TwirlingConfiguration(readout_twirl_strategy="LOCAL", seed=42))
        twirler.twirl(circuits).submit(shots=20_000)
        mitigated_counts = twirler.retrieve_counts()

    Example — circuit-twirling-only, no client needed::

        config = TwirlingConfiguration(readout_twirl_strategy="NONE", circuit_twirling=True)
        twirler = CircuitTwirler(config=config)
        randomized = twirler.twirl(circuits).get_twirled_circuits()

    """

    def __init__(
        self,
        client: Pulla | None = None,
        config: TwirlingConfiguration | None = None,
        compilation_options: dict[str, Any] | None = None,
    ) -> None:
        self._client: Pulla | None = client
        self._config: TwirlingConfiguration = config or TwirlingConfiguration()
        self._compilation_options: dict[str, Any] | None = compilation_options

        strategy = self._config.readout_twirl_strategy
        if strategy not in _SUPPORTED_STRATEGIES:
            raise ValueError(f"Unsupported twirling strategy '{strategy}'. Supported: {sorted(_SUPPORTED_STRATEGIES)}")

        # Internal state — populated lazily by the lifecycle methods.
        self._input_circuits: list[TwirledCircuit] | None = None
        self._randomized_circuits_per_input: list[list[TwirledCircuit]] | None = None
        self._qubit_to_bit_mappings: list[dict[str, int]] | None = None
        self._qubit_index_to_name_per_input: list[dict[int, str]] | None = None
        self._job: PullaJob | None = None
        self._shots_per_circuit: int = 0
        self._untwirled_counts: list[dict[str, float]] | None = None
        self._twirling_skipped: bool = False
        self._twirl_strategy: str | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def twirl(  # noqa: PLR0912
        self, circuits: list[TwirledCircuit | QrispQuantumCircuit | QiskitQuantumCircuit | PulseCircuit]
    ) -> CircuitTwirler:
        """Convert input circuits to pulse form, generate rotation strings, and randomize.

        Args:
            circuits: Input circuits to twirl.

        Returns:
            ``self``, to allow method chaining.

        Raises:
            TypeError: If a circuit type is not supported.
            ValueError: If *circuits* is empty.

        """
        if not circuits:
            raise ValueError("At least one circuit must be provided.")

        # Readout twirling is enabled iff the user picked an actual strategy.
        strategy = self._config.readout_twirl_strategy
        readout_on = strategy != "NONE"
        circuit_on = self._config.circuit_twirling

        # Reject the no-op combination early instead of silently submitting
        # N identical untwirled circuits.
        if not readout_on and not circuit_on:
            warnings.warn(
                "readout_twirl_strategy='NONE' with circuit_twirling=False leaves nothing to do; "
                "you could submit the original circuits directly instead.",
                stacklevel=2,
            )

        if self._config.num_twirling_instances < 1:
            raise ValueError(f"num_twirling_instances must be >= 1, got {self._config.num_twirling_instances}.")

        self._twirl_strategy = strategy

        pulse_circuits, qubit_index_to_name_per_input = self._convert_circuits(circuits)
        self._input_circuits = pulse_circuits
        self._qubit_index_to_name_per_input = qubit_index_to_name_per_input

        # Extract qubit-to-bit mappings from each circuit.
        self._qubit_to_bit_mappings = [self._extract_qubit_to_bit_mapping(pc) for pc in pulse_circuits]

        # Get topology from the server (only needed for LOCAL).
        if strategy == "LOCAL":
            if self._client is None:
                raise ValueError(
                    "readout_twirl_strategy='LOCAL' requires a client to look up the QPU topology. "
                    "Either construct CircuitTwirler with a client, or pick a strategy that does not "
                    "need topology ('MINIMAL', 'HADAMARD', or 'NONE')."
                )
            topology = topology_from_qc(self._client)
        else:
            topology = None

        if strategy == "LOCAL":
            if topology is None:
                warnings.warn(
                    "Cannot determine QPU topology for LOCAL twirling strategy. "
                    "Falling back to standard REM without twirling.",
                    stacklevel=2,
                )
                self._twirling_skipped = True
                self._randomized_circuits_per_input = [[pc] for pc in pulse_circuits]
                return self

        self._twirling_skipped = False

        # Check for MOVE-gate-based QPUs — cannot twirl, fall back.
        if self._client is not None and uses_move_gates(self._client):
            warnings.warn(
                "QPU uses MOVE gates (computational resonators detected). "
                "Twirling is not supported for MOVE-based QPUs. "
                "Falling back to standard REM without twirling.",
                stacklevel=2,
            )
            self._twirling_skipped = True
            self._randomized_circuits_per_input = [[pc] for pc in pulse_circuits]
            return self

        rng = np.random.default_rng(self._config.seed)

        # When circuit_twirling is disabled, force the per-SQG insertion probability
        # to 0.0.  Readout twirling still works because it overrides this probability
        # on the *last* SQG of each measured qubit (see pmpt._determine_twirling_params).
        twirling_probabilities = None if circuit_on else 0.0

        all_randomized_per_input: list[list[TwirledCircuit]] = []

        requested_instances = self._config.num_twirling_instances

        for pc in pulse_circuits:
            qubits_for_readout_twirling = pc.measured_qubits

            if strategy == "NONE":
                # No readout-basis variation: we only differ via Pauli (circuit) twirling.
                # There is exactly one trivial rot-string, so the user's count is honoured as-is.
                base_rot_strings: list[dict[str, str] | bool] = [False]
            else:
                # generate_rot_strings returns list[dict[str, str]] mapping qubit -> 'I'/'X'.
                base_rot_strings = list(
                    generate_rot_strings(
                        qpu_topology=topology,
                        active_qubits=qubits_for_readout_twirling,
                        strategy=strategy,
                    )
                )

            # Reconcile the user-chosen number of (circuit-twirling) instances with the
            # fixed number of readout-twirling rot-strings, so each rot-string receives
            # the *same* number of independent Pauli randomizations (Option B).
            n_rot = len(base_rot_strings)
            if circuit_on:
                if requested_instances < n_rot:
                    warnings.warn(
                        f"num_twirling_instances={requested_instances} is smaller than the number of "
                        f"readout-twirling rot-strings ({n_rot}) for strategy '{strategy}'. "
                        f"Using {n_rot} instances instead so every rot-string is sampled at least once.",
                        stacklevel=2,
                    )
                    per_rot = 1
                elif requested_instances % n_rot != 0:
                    per_rot = math.ceil(requested_instances / n_rot)
                    rounded = per_rot * n_rot
                    warnings.warn(
                        f"num_twirling_instances={requested_instances} is not a multiple of the number "
                        f"of readout-twirling rot-strings ({n_rot}) for strategy '{strategy}'. "
                        f"Rounding up to {rounded} instances ({per_rot} per rot-string) "
                        "to keep readout-basis sampling balanced.",
                        stacklevel=2,
                    )
                else:
                    per_rot = requested_instances // n_rot
            else:
                # No Pauli randomization → repeating the same rot-string would produce
                # identical circuits.  Just emit one circuit per rot-string.
                per_rot = 1

            # Outer = rot-strings, inner = independent Pauli randomizations.
            randomized_circuits = [
                randomize_circuit(
                    circuit=pc,
                    rgen=rng,
                    readout_twirling=rs,
                    twirling_probabilities=twirling_probabilities,
                )
                for rs in base_rot_strings
                for _ in range(per_rot)
            ]
            all_randomized_per_input.append(randomized_circuits)

        self._randomized_circuits_per_input = all_randomized_per_input

        total = sum(len(group) for group in all_randomized_per_input)
        logger.info(
            "Twirled %d circuit(s) into %d randomized circuits.",
            len(pulse_circuits),
            total,
        )

        return self

    def submit(self, shots: int = 20_000, client: Pulla | None = None) -> CircuitTwirler:
        """Compile and submit all randomized circuits to the quantum computer for execution.

        ``shots`` refers to the number of shots *per input (target) circuit*. It is
        distributed evenly across that circuit's twirled instances, so the untwirled,
        aggregated counts for each input circuit are based on (approximately) ``shots``
        shots regardless of how many input circuits are submitted.

        Args:
            shots: Number of shots per input circuit, split across its twirled instances.
            client: Client for submitting the task to the quantum computer.
                If provided, it overrides the client
                supplied at construction time and is also stored on the
                instance for any subsequent calls (e.g. :meth:`retrieve_counts`).
                This makes it possible to construct a client-less
                ``CircuitTwirler`` purely as a circuit transformer and only
                bind a client at submission time.

                When the strategy used at :meth:`twirl` time was ``"LOCAL"``,
                the rotation strings are tied to the *exact* quantum computer that
                was used during :meth:`twirl`.  Using a different
                quantum computer here raises :class:`ValueError`; re-run
                :meth:`twirl` with the new QC first.

        Returns:
            ``self``, to allow method chaining.

        Raises:
            RuntimeError: If :meth:`twirl` has not been called yet, or if no
                client is available (neither at construction nor here).
            ValueError: If a different ``client`` instance is passed here while the
                ``"LOCAL"`` strategy was used at :meth:`twirl` time.

        """
        if self._randomized_circuits_per_input is None:
            raise RuntimeError("No circuits to submit. Call twirl() first.")

        if client is not None and client is not self._client:
            if self._twirl_strategy == "LOCAL" and self._client is not None:
                raise ValueError(
                    "Cannot change client at submit() time when twirl() used the 'LOCAL' "
                    "strategy: rotation strings are tied to the twirl-time QC's topology. "
                    "Re-run twirl() with the new client before submitting."
                )
            self._client = client

        if self._client is None:
            raise RuntimeError(
                "submit() requires a client, but CircuitTwirler was constructed without one "
                "and none was passed to submit(). Either pass `client=` to submit(), construct "
                "CircuitTwirler with a client, or use get_twirled_circuits() to drive "
                "submission externally."
            )

        # Flatten all randomized circuits into a single list for compilation,
        # mirroring the notebook pattern:
        #   compiler.compile(circuits=[c.to_circuit(f"twirled_{j}") for j, c in enumerate(all_circuits)])
        all_circuits = [circ for group in self._randomized_circuits_per_input for circ in group]
        immutable_circuits = [rc.to_circuit(name=f"twirled_{i}") for i, rc in enumerate(all_circuits)]

        compiler = self._client.get_standard_compiler()

        group_sizes = [len(group) for group in self._randomized_circuits_per_input]
        instances_per_circuit = min(group_sizes) if group_sizes else 1
        if len(set(group_sizes)) > 1:
            warnings.warn(
                f"Input circuits produced different numbers of twirled instances ({sorted(set(group_sizes))}); "
                "sizing shots from the smallest group so every input circuit receives at least the requested "
                "shots. Some circuits will receive more than the requested shots.",
                stacklevel=2,
            )
        self._shots_per_circuit = max(1, math.ceil(shots / instances_per_circuit))

        # Copy the user-supplied options so the compiler/runtime cannot mutate them
        # (we add `shots` to the returned context below; passing the original dict
        # in could leak that mutation back to the caller).
        compile_context = dict(self._compilation_options) if self._compilation_options else None

        # Inject shots into the compiler settings so that playlist_repeats in the
        # RunDefinition matches the intended per-circuit shot count.  Without this
        # the compiler falls back to its internal DEFAULT_REPETITIONS (1000), causing
        # the server to execute 1000 shots per twirled circuit regardless of the
        # user-requested total.
        settings = compiler.get_settings(circuits=immutable_circuits)
        settings.set_shots(self._shots_per_circuit)

        job_definition, context = compiler.compile(
            circuits=immutable_circuits,
            settings=settings,
            context=compile_context,
        )

        self._job = self._client.submit_playlist(job_definition, context=context)

        logger.info(
            "Submitted %d circuits (%d shots each, %d total, ~%d shots per input circuit).",
            len(immutable_circuits),
            self._shots_per_circuit,
            self._shots_per_circuit * len(immutable_circuits),
            self._shots_per_circuit * instances_per_circuit,
        )
        return self

    def retrieve_counts(self) -> list[dict[str, float]]:
        """Wait for job completion, untwirl, and sum counts per input circuit.

        Uses :func:`~iqm.error_reduction_tools.twirling.twirling_processors.untwirl_and_sum_counts`
        (the same function used in the tutorial notebooks) to untwirl and
        aggregate the raw counts.

        Returns:
            One count dictionary per original input circuit.

        Raises:
            RuntimeError: If :meth:`submit` has not been called yet.

        """
        if self._job is None:
            raise RuntimeError("No job to retrieve. Call submit() first.")
        if self._randomized_circuits_per_input is None or self._input_circuits is None:
            raise RuntimeError("Internal state is inconsistent. Call twirl() and submit() first.")

        max_retries = 10
        retry_delay_s = 30.0
        for attempt in range(max_retries):
            self._job.wait_for_completion()
            try:
                results = sweep_job_to_qiskit(self._job, shots=self._shots_per_circuit)
                break
            except ValueError as exc:
                if "WAITING" in str(exc) and attempt < max_retries - 1:
                    sleep(retry_delay_s)
                else:
                    raise
        else:
            raise RuntimeError("Job did not complete after multiple wait attempts.")

        untwirled_per_circuit: list[dict[str, float]] = []
        flat_idx = 0

        for group in self._randomized_circuits_per_input:
            raw_counts_list = []
            rot_string_list = []
            for circ in group:
                raw_counts_list.append(results.get_counts(flat_idx))
                rot_string_list.append(_rot_string_for(circ))
                flat_idx += 1

            untwirled = untwirl_and_sum_counts(raw_counts_list, rot_string_list)
            untwirled_per_circuit.append(untwirled)

        self._untwirled_counts = untwirled_per_circuit
        return untwirled_per_circuit

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    def get_twirled_circuits(
        self, return_qiskit: bool = False
    ) -> list[list[PulseCircuit]] | list[list[QiskitQuantumCircuit]]:
        """Return the randomized circuits produced by :meth:`twirl`.

        Args:
            return_qiskit: When ``True``, convert each circuit to a Qiskit
                :class:`~qiskit.circuit.QuantumCircuit` before returning.
                Defaults to ``False`` (returns :class:`~iqm.pulse.Circuit`).

        Returns:
            Nested list with one inner list of randomized variants per input
            circuit, in the same order as the circuits passed to :meth:`twirl`.

        Raises:
            RuntimeError: If :meth:`twirl` has not been called yet.

        """
        if self._randomized_circuits_per_input is None:
            raise RuntimeError("No twirled circuits available. Call twirl() first.")
        if return_qiskit:
            if self._qubit_index_to_name_per_input is None:
                raise RuntimeError("No twirled circuits available. Call twirl() first.")
            return [
                [from_iqm_to_qiskit(circ, self._qubit_index_to_name_per_input[i]) for circ in group]
                for i, group in enumerate(self._randomized_circuits_per_input)
            ]
        return [
            [circ.to_circuit(name=f"twirled_{i}_{j}") for j, circ in enumerate(group)]
            for i, group in enumerate(self._randomized_circuits_per_input)
        ]

    def get_twirled_circuits_flat(self, return_qiskit: bool = False) -> list[PulseCircuit] | list[QiskitQuantumCircuit]:
        """Return all randomized circuits as a single flat list.

        Args:
            return_qiskit: When ``True``, convert each circuit to a Qiskit
                :class:`~qiskit.circuit.QuantumCircuit` before returning.
                Defaults to ``False`` (returns :class:`~iqm.pulse.Circuit`).

        Returns:
            Flat list of all randomized circuits, matching the order used
            internally by :meth:`submit`.

        Raises:
            RuntimeError: If :meth:`twirl` has not been called yet.

        """
        return [circ for group in self.get_twirled_circuits(return_qiskit=return_qiskit) for circ in group]

    def get_rot_strings(self) -> list[list[str]]:
        """Return rotation strings produced by :meth:`twirl`.

        Returns:
            Nested list — one inner list per input circuit, each containing
            one rot string (e.g. ``"IXXI"``) per randomized variant.

        Raises:
            RuntimeError: If :meth:`twirl` has not been called yet.

        """
        if self._randomized_circuits_per_input is None:
            raise RuntimeError("No rotation strings available. Call twirl() first.")
        return [[_rot_string_for(circ) for circ in group] for group in self._randomized_circuits_per_input]

    def get_qubit_to_bit_mapping(self) -> list[dict[str, int]]:
        """Return qubit-to-classical-bit mappings extracted during :meth:`twirl`.

        Returns:
            One mapping per input circuit: ``{qubit_name: classical_bit_index}``.

        Raises:
            RuntimeError: If :meth:`twirl` has not been called yet.

        """
        if self._qubit_to_bit_mappings is None:
            raise RuntimeError("No mappings available. Call twirl() first.")
        return self._qubit_to_bit_mappings

    def get_job(self) -> object:
        """Return job from :meth:`submit`.

        Raises:
            RuntimeError: If :meth:`submit` has not been called yet.

        """
        if self._job is None:
            raise RuntimeError("No job available. Call submit() first.")
        return self._job

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize twirling state to a plain dictionary.

        Includes rotation strings, qubit-to-bit mappings, and the
        configuration — enough to reconstruct the twirler for
        post-processing without re-running on hardware.
        """
        return {
            "rot_strings": (self.get_rot_strings() if self._randomized_circuits_per_input else None),
            "qubit_to_bit_mappings": self._qubit_to_bit_mappings,
            "config": asdict(self._config),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], client: Pulla) -> CircuitTwirler:
        """Reconstruct a :class:`CircuitTwirler` from a dictionary.

        The restored instance has rotation strings and mappings populated
        but no circuits or job — it can be used for post-processing only.

        Args:
            data: Dictionary as returned by :meth:`to_dict`.
            client: Client for connecting to the quantum computer.

        Returns:
            A :class:`CircuitTwirler` with restored state.

        """
        twirler = cls(client=client, config=TwirlingConfiguration(**data["config"]))
        twirler._qubit_to_bit_mappings = data["qubit_to_bit_mappings"]
        return twirler

    def save_twirling_info(self, path: str) -> None:
        """Save twirling state to a JSON file.

        Args:
            path: Destination file path.

        """
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)
        logger.info("Saved twirling info to %s", path)

    @classmethod
    def load_twirling_info(cls, path: str, client: Pulla) -> CircuitTwirler:
        """Load twirling state from a JSON file.

        Args:
            path: Path to a JSON file previously written by
                :meth:`save_twirling_info`.
            client: Client for connecting to the quantum computer.

        Returns:
            A :class:`CircuitTwirler` with restored state.

        """
        with open(path) as f:
            return cls.from_dict(json.load(f), client=client)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _convert_circuits(self, circuits: list) -> tuple[list[TwirledCircuit], list[dict[int, str]]]:
        """Convert heterogeneous input circuits to :class:`TwirledCircuit`.

        Supports :class:`TwirledCircuit`, :class:`~iqm.pulse.Circuit`,
        :class:`~qiskit.circuit.QuantumCircuit`, and **any object that
        exposes a** ``to_qiskit()`` **method** returning a Qiskit
        :class:`~qiskit.circuit.QuantumCircuit` (e.g.
        :class:`qrisp.QuantumCircuit`).
        """
        converted: list[TwirledCircuit] = []
        mappings: list[dict[int, str]] = []
        for circ in circuits:
            if isinstance(circ, TwirledCircuit):
                converted.append(circ)
                sorted_qubits = sorted(circ.active_qubits)
                mappings.append({i: q for i, q in enumerate(sorted_qubits)})
            elif isinstance(circ, PulseCircuit):
                tpc = TwirledCircuit(list(circ.instructions))
                converted.append(tpc)
                sorted_qubits = sorted(tpc.active_qubits)
                mappings.append({i: q for i, q in enumerate(sorted_qubits)})
            elif isinstance(circ, QiskitQuantumCircuit):
                pulse_circ, qubit_index_to_name = from_qiskit_to_iqm(circ)
                converted.append(pulse_circ)
                mappings.append(qubit_index_to_name)
            elif isinstance(circ, QrispQuantumCircuit):
                pulse_circ, qubit_index_to_name = from_qiskit_to_iqm(circ.to_qiskit())
                converted.append(pulse_circ)
                mappings.append(qubit_index_to_name)
            else:
                raise TypeError(
                    f"Unsupported circuit type: {type(circ).__name__}. "
                    "Expected TwirledCircuit, iqm.pulse.Circuit, qiskit QuantumCircuit, "
                    "or any circuit with a to_qiskit() method (e.g. Qrisp QuantumCircuit)."
                )
        return converted, mappings

    @staticmethod
    def _extract_qubit_to_bit_mapping(circuit: TwirledCircuit) -> dict[str, int]:
        """Derive the qubit → classical-bit mapping from measurement operations."""
        mapping: dict[str, int] = {}
        bit_idx = 0
        for qubit in circuit.measured_qubits:
            mapping[qubit] = bit_idx
            bit_idx += 1
        return mapping


def _rot_string_for(circuit: TwirledCircuit) -> str:
    """Build a BIG-endian readout-twirling string aligned with ``circuit.measured_qubits``."""
    n = len(circuit.measured_qubits)
    if not circuit.rot_dict:
        return "I" * n
    return "".join(circuit.rot_dict.get(q, "I") for q in circuit.measured_qubits)
