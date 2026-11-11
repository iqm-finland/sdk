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

"""Generation of the Clifford training circuits of CDR."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import json
import os
from typing import Any
import warnings

from iqm.error_reduction_tools.cdr.cdr_clifford_utils import (
    _progress,
    _warn_if_identity_delay_unused,
    get_clifford_r_library,
    is_logically_single_qubit_clifford_r,
    stim_sequence_for_r_gate,
)
from iqm.error_reduction_tools.cdr.cdr_observables import diagonal_observable_to_terms, to_pauli_terms
from iqm.error_reduction_tools.cdr.cdr_training_circuits import (
    build_clbit_to_stim_qubit_map,
    exact_expectation_from_stim,
    qubit_order,
    training_circuit_to_stim,
    validate_training_circuit_is_clifford,
)
import numpy as np
import stim

from iqm.pulse import Circuit, CircuitOperation

_COEFFICIENT_ATOL = 1e-12
"""Tolerance below which a coefficient, or its distance to 1, counts as zero in observable labels."""

_VALUE_ATOL = 1e-9
"""Tolerance below which a difference between exact values counts as zero."""


@dataclass
class TrainingSet:
    """Clifford training circuits of one target, with their exact expectation values."""

    pulse_circuits: list
    """The Clifford training circuits, as pulse circuits."""

    stim_circuits: list
    """Stim circuit of each training circuit, or ``None`` if not built."""

    measurement_maps: list
    """Measurement map between the Stim and the pulse circuit of each training circuit, or ``None`` if not built."""

    exact_values: np.ndarray
    """Exact expectation values, shape ``(num_training, num_observables)``."""

    observable_labels: list[str]
    """Label of each observable, the columns of ``exact_values``."""

    target_circuit: Any = None
    """The target non-Clifford circuit, or ``None``."""

    metadata: dict = field(default_factory=dict)
    """Strategy, number of slots, library size, seed and number of evaluated candidates."""

    @property
    def num_training(self) -> int:
        """Return the number of training circuits."""
        return len(self.pulse_circuits)

    @property
    def num_observables(self) -> int:
        """Return the number of observables, the columns of ``exact_values``."""
        return self.exact_values.shape[1] if self.exact_values.ndim == 2 else 1  # noqa: PLR2004

    def spread(self) -> dict:
        """Return the spread statistics of the exact values, see :func:`training_set_spread`."""
        return training_set_spread(self.exact_values)


def _observable_label(observable: Any) -> str:
    """Return a readable label for an observable."""
    terms = to_pauli_terms(observable)

    if len(terms) == 1 and abs(terms[0][0] - 1.0) < _COEFFICIENT_ATOL:
        return terms[0][1]
    parts = []
    for coeff, label in terms:
        c = coeff.real if abs(coeff.imag) < _COEFFICIENT_ATOL else coeff
        parts.append(f"({c:.4g})*{label}")
    return " + ".join(parts)


def exact_expectation_vector(
    pulse_clifford_circuit: Circuit,
    observables: Sequence,
) -> tuple[np.ndarray, Any, list[dict]]:
    """Compute the exact expectation value of every observable for one Clifford circuit.

    Args:
        pulse_clifford_circuit: A Clifford pulse circuit.
        observables: Diagonal observables indexed by classical bit, in a format accepted by
            :func:`~iqm.error_reduction_tools.cdr.cdr_observables.to_pauli_terms`.

    Returns:
        The values, shape ``(num_observables,)``, the Stim circuit and its measurement map.

    """
    stim_circuit, measurement_map = training_circuit_to_stim(pulse_clifford_circuit, return_measurement_map=True)
    values = np.array(
        [
            float(
                np.real(
                    exact_expectation_from_stim(
                        stim_circuit=stim_circuit,
                        measurement_map=measurement_map,
                        observable=obs,
                    )
                )
            )
            for obs in observables
        ],
        dtype=float,
    )
    return values, stim_circuit, measurement_map


def _prepare_fast_evaluator(
    target_circuit: Circuit,
    observables: Sequence,
    slot_indices: Sequence[int],
    clifford_library: list,
    clifford_atol: float,
) -> dict:
    """Translate the circuit and the Clifford library to Stim once, so that every candidate reuses them.

    Args:
        target_circuit: The target circuit, in the native gate set.
        observables: Diagonal observables indexed by classical bit.
        slot_indices: Instruction indices of the non-Clifford prx gates to replace, from
            :func:`non_clifford_slot_indices`.
        clifford_library: The candidate gates, from :func:`get_clifford_r_library`.
        clifford_atol: Tolerance for Clifford detection.

    Returns:
        The state :func:`_eval_assignment_fast` reads: ``header`` (Stim identity gates that fix the qubit count),
        ``tokens`` (fixed Stim gate text or slot placeholders), ``lib_seq`` (Stim gates of each library candidate),
        ``obs_ps`` (each observable as ``(coefficient, stim.PauliString)`` pairs), ``num_qubits`` and
        ``measurement_map``.

    """
    qubits = qubit_order(target_circuit)
    qubit_index = {q: i for i, q in enumerate(qubits)}
    num_qubits = len(qubits)
    slot_pos = {idx: pos for pos, idx in enumerate(slot_indices)}

    # Stim gates of each library member.
    lib_seq: list[list[str]] = []
    for cand in clifford_library:
        if cand["is_delay"]:
            lib_seq.append([])  # The identity has no gates.
        else:
            lib_seq.append(list(stim_sequence_for_r_gate(cand["theta"], cand["phi"], atol=clifford_atol)))

    # Fixed Stim gate text, or a placeholder for each slot.
    header = "I " + " ".join(str(i) for i in range(num_qubits)) if num_qubits else ""
    tokens: list[tuple] = []
    for idx, operation in enumerate(target_circuit.instructions):
        name = operation.name
        if name == "prx":
            q = qubit_index[operation.locus[0]]
            if idx in slot_pos:
                tokens.append(("slot", q, slot_pos[idx]))
            else:  # A prx gate that is already Clifford.
                theta = float(operation.args["angle"])
                phi = float(operation.args["phase"])
                tokens.extend(("fix", f"{g} {q}") for g in stim_sequence_for_r_gate(theta, phi, atol=clifford_atol))
        elif name == "cz":
            q0, q1 = (qubit_index[q] for q in operation.locus)
            tokens.append(("fix", f"CZ {q0} {q1}"))
        elif name in ("measure", "barrier", "delay"):
            continue
        else:
            raise ValueError(
                f"Instruction {name!r} at index {idx} is not supported by the fast "
                "Clifford evaluator (expected prx/cz/measure/barrier/delay)."
            )

    # Translate one circuit to get the map from classical bits to Stim qubits.
    representative = build_clifford_circuit_from_assignment(
        target_circuit,
        slot_indices,
        tuple(0 for _ in slot_indices),
        clifford_library,
        validate=False,
        clifford_atol=clifford_atol,
    )
    _, meas_map = training_circuit_to_stim(representative, return_measurement_map=True)
    cmap = build_clbit_to_stim_qubit_map(meas_map)
    to_stim_qubit = lambda i: cmap[i]  # noqa: E731

    obs_ps = []
    for obs in observables:
        term_ps = []
        for coeff, z_bits in diagonal_observable_to_terms(obs):
            chars = ["_"] * num_qubits
            for b in z_bits:
                chars[to_stim_qubit(b)] = "Z"
            term_ps.append((complex(coeff), stim.PauliString("".join(chars))))
        obs_ps.append(term_ps)

    return {
        "header": header,
        "tokens": tokens,
        "lib_seq": lib_seq,
        "obs_ps": obs_ps,
        "num_qubits": num_qubits,
        "measurement_map": meas_map,
    }


def _eval_assignment_fast(assignment: Sequence[int], prep: dict) -> np.ndarray:
    """Compute the exact expectation vector of one library assignment from a Stim circuit string."""
    parts = [prep["header"]] if prep["header"] else []
    lib_seq = prep["lib_seq"]
    for token in prep["tokens"]:
        if token[0] == "fix":
            parts.append(token[1])
        else:
            _, qubit, pos = token
            seq = lib_seq[assignment[pos]]
            if seq:
                parts.append("\n".join(f"{g} {qubit}" for g in seq))

    sim = stim.TableauSimulator()
    sim.do_circuit(stim.Circuit("\n".join(parts)))

    values = np.empty(len(prep["obs_ps"]), dtype=float)
    for j, term_ps in enumerate(prep["obs_ps"]):
        acc = 0.0 + 0.0j
        for coeff, pauli_string in term_ps:
            acc += coeff * sim.peek_observable_expectation(pauli_string)
        values[j] = acc.real
    return values


def non_clifford_slot_indices(target_circuit: Circuit, clifford_atol: float = 1e-9) -> list[int]:
    """Find the instruction indices of the non-Clifford prx gates, the slots.

    Only these gates are replaced when building a training circuit. Every other operation is copied unchanged.

    Args:
        target_circuit: The target circuit.
        clifford_atol: Tolerance for Clifford detection.

    Returns:
        The instruction indices of the slots.

    """
    slots = []
    for idx, operation in enumerate(target_circuit.instructions):
        if operation.name != "prx":
            continue
        theta = float(operation.args["angle"])
        phi = float(operation.args["phase"])
        if not is_logically_single_qubit_clifford_r(theta, phi, atol=clifford_atol):
            slots.append(idx)
    return slots


def build_clifford_circuit_from_assignment(  # noqa: PLR0913
    target_circuit: Circuit,
    slot_indices: Sequence[int],
    assignment: Sequence[int],
    clifford_library: list,
    identity_delay: float = 20e-9,
    validate: bool = True,
    clifford_atol: float = 1e-9,
) -> Circuit:
    """Build a Clifford circuit by replacing each slot with a gate from the Clifford library.

    Args:
        target_circuit: The target circuit.
        slot_indices: Instruction indices of the non-Clifford prx gates, from :func:`non_clifford_slot_indices`.
        assignment: Index into ``clifford_library`` for each slot, in the order of ``slot_indices``.
        clifford_library: The candidate gates, from :func:`get_clifford_r_library`.
        identity_delay: Duration of the identity delay, in seconds.
        validate: Whether to check that the result is Clifford.
        clifford_atol: Tolerance for Clifford validation.

    Returns:
        A Clifford pulse circuit with the same structure as the target.

    Raises:
        ValueError: If ``slot_indices`` and ``assignment`` differ in length, or ``validate`` is True and the result is
            not Clifford.

    """
    if len(slot_indices) != len(assignment):
        raise ValueError("slot_indices and assignment must have equal length.")

    slot_to_choice = dict(zip(slot_indices, assignment))

    operations: list[CircuitOperation] = []
    for idx, operation in enumerate(target_circuit.instructions):
        if idx in slot_to_choice:
            q = operation.locus[0]
            candidate = clifford_library[slot_to_choice[idx]]
            if candidate["is_delay"]:
                operations.append(CircuitOperation(name="delay", locus=(q,), args={"duration": identity_delay}))
            else:
                operations.append(
                    CircuitOperation(
                        name="prx",
                        locus=(q,),
                        args={"angle": candidate["theta"], "phase": candidate["phi"]},
                    )
                )
        else:
            operations.append(operation)

    training_circuit = Circuit(name=f"{target_circuit.name}_clifford", instructions=tuple(operations))

    if validate:
        validate_training_circuit_is_clifford(training_circuit, atol=clifford_atol, raise_on_failure=True)
    return training_circuit


def random_assignment(num_slots: int, num_library: int, rng: np.random.Generator) -> tuple[int, ...]:
    """Draw a uniformly random library choice for every non-Clifford slot."""
    return tuple(int(x) for x in rng.integers(0, num_library, size=num_slots))


def farthest_point_selection(
    vectors: np.ndarray,
    num_select: int,
    rng: np.random.Generator | None = None,
    seed: str = "maxnorm",
    weights: Sequence[float] | None = None,
    allow_duplicate_noiseless_values: bool = True,
) -> list[int]:
    """Select rows greedily so that each new row is farthest from the rows already selected.

    This spreads the exact values of the training set instead of letting them cluster.

    Args:
        vectors: Exact values of each circuit, shape ``(n, d)``.
        num_select: Number of rows to select.
        rng: Random number generator, used only when ``seed="random"``.
        seed: First row: ``"maxnorm"`` takes the row with the largest norm, ``"random"`` a random row.
        weights: Coefficients of the columns, shape ``(d,)``. If given, distances are computed on
            ``vectors @ weights`` instead of on the rows.
        allow_duplicate_noiseless_values: Whether to keep selecting once every remaining row duplicates the exact
            values of a selected row. If True, the selection starts a new round over the remaining rows, so the
            exact values are repeated evenly. If False, it stops.

    Returns:
        The selected row indices, each at most once. Fewer than ``num_select`` are returned when there are fewer
        rows, or when ``allow_duplicate_noiseless_values`` is False and only duplicate rows remain.

    """
    vectors = np.atleast_2d(np.asarray(vectors, dtype=float))
    n = vectors.shape[0]
    if num_select >= n:
        return list(range(n))
    if rng is None:
        rng = np.random.default_rng()

    metric = vectors if weights is None else (vectors @ np.asarray(weights, dtype=float)).reshape(-1, 1)

    # "maxnorm": start from the row with the largest norm.
    first = int(rng.integers(0, n)) if seed == "random" else int(np.argmax(np.linalg.norm(metric, axis=1)))

    selected = [first]
    # Distance from each row to the closest selected row, -inf once the row is selected.
    min_dist = np.linalg.norm(metric - metric[first], axis=1)
    min_dist[first] = -np.inf

    while len(selected) < num_select:
        if min_dist.max() <= 0:  # Every remaining row duplicates a selected row.
            if not allow_duplicate_noiseless_values:
                break
            min_dist[min_dist == 0] = np.inf  # Start a new round over the remaining rows.
        nxt = int(np.argmax(min_dist))
        selected.append(nxt)
        new_dist = np.linalg.norm(metric - metric[nxt], axis=1)
        min_dist = np.minimum(min_dist, new_dist)
        min_dist[nxt] = -np.inf

    return selected


def training_set_spread(exact_values: np.ndarray) -> dict:
    """Compute spread statistics of the exact values of a training set.

    Args:
        exact_values: Exact values, shape ``(num_circuits, num_observables)``.

    Returns:
        The number of circuits and observables, the standard deviation, minimum, maximum and range of each
        observable, and the mean standard deviation.

    """
    exact_values = np.atleast_2d(np.asarray(exact_values, dtype=float))
    per_obs = {
        "std": np.std(exact_values, axis=0).tolist(),
        "min": np.min(exact_values, axis=0).tolist(),
        "max": np.max(exact_values, axis=0).tolist(),
        "range": (np.max(exact_values, axis=0) - np.min(exact_values, axis=0)).tolist(),
    }
    return {
        "num_circuits": int(exact_values.shape[0]),
        "num_observables": int(exact_values.shape[1]),
        "per_observable": per_obs,
        "mean_std_over_observables": float(np.mean(np.std(exact_values, axis=0))),
    }


def generate_clifford_training_set(  # noqa: PLR0913
    target_circuit: Circuit,
    observables: Sequence,
    num_training_circuits: int,
    *,
    strategy: str = "pool_fps",
    pool_size: int | None = None,
    coefficients: Sequence[float] | None = None,
    primary_index: int = 0,
    target_grid: Sequence[float] | None = None,
    max_descent_passes: int = 4,
    max_opt_slots: int = 60,
    dedup: bool = True,
    rng_seed: int | None = 0,
    identity_as_delay: bool = True,
    identity_delay: float = 20e-9,
    clifford_atol: float = 1e-9,
    progress: bool = True,
    self_check: bool = True,
    build_stim_circuits: bool = True,
    validate_selected: bool = True,
    verbose: bool = False,
) -> TrainingSet:
    """Generate Clifford training circuits for a target circuit.

    Args:
        target_circuit: The target circuit, with its final measurements.
        observables: Diagonal observables indexed by classical bit, in a format accepted by
            :func:`~iqm.error_reduction_tools.cdr.cdr_observables.to_pauli_terms`.
        num_training_circuits: Number of training circuits. Fewer are returned when there are not enough distinct
            library assignments.
        strategy: ``"pool_fps"`` evaluates a random pool of candidates and selects a spread subset with
            :func:`farthest_point_selection`. ``"optimize"`` searches, by coordinate descent over the gate choices,
            for one circuit per value of a uniform grid of target values.
        pool_size: Size of the candidate pool of ``"pool_fps"``. ``None`` uses
            ``max(200, 50 * num_training_circuits)``.
        coefficients: Coefficients of the terms of the fitted observable, shape ``(num_terms,)``. If given, both
            strategies spread the combined value instead of the values of single terms.
        primary_index: Observable that ``"optimize"`` spreads when ``coefficients`` is not given.
        target_grid: Target values of ``"optimize"``. ``None`` uses a uniform grid over ``[-1, 1]``, or over the
            reachable range of the combined value when ``coefficients`` is given.
        max_descent_passes: Largest number of coordinate-descent passes per target value of ``"optimize"``.
        max_opt_slots: Largest number of slots that ``"optimize"`` changes, chosen at random on deep circuits.
        dedup: Whether to drop duplicate library assignments before selection.
        rng_seed: Seed of the random number generator. ``None`` gives a different result on every call.
        identity_as_delay: Whether the identity in the library is a delay instead of ``prx(0, 0)``.
        identity_delay: Duration of the identity delay, in seconds.
        clifford_atol: Tolerance for Clifford detection.
        progress: Whether to show tqdm progress bars.
        self_check: Whether to compare the fast evaluation with the Stim simulation once.
        build_stim_circuits: Whether to also store the Stim circuits of the selected circuits.
        validate_selected: Whether to check that the selected circuits are Clifford.
        verbose: Whether to print the number of selected circuits and their spread.

    Returns:
        The training set.

    Raises:
        ValueError: If ``strategy`` is neither ``"pool_fps"`` nor ``"optimize"``.
        RuntimeError: If ``self_check`` is True and the two evaluations disagree.

    """
    rng = np.random.default_rng(rng_seed)
    observables = list(observables)
    labels = [_observable_label(o) for o in observables]
    _warn_if_identity_delay_unused(identity_as_delay, identity_delay)
    library = get_clifford_r_library(identity_as_delay=identity_as_delay)
    num_library = len(library)
    slots = non_clifford_slot_indices(target_circuit, clifford_atol=clifford_atol)

    if len(slots) == 0:
        warnings.warn(
            "The target circuit has no non-Clifford prx gates, so it is already Clifford and all training circuits "
            "would be identical. Returning one circuit.",
            UserWarning,
        )
        num_training_circuits = 1

    # Every candidate below reuses the fast evaluator.
    prep = _prepare_fast_evaluator(target_circuit, observables, slots, library, clifford_atol)

    # Check once that the fast evaluation agrees with the Stim simulation.
    if self_check and len(slots) > 0:
        probe = random_assignment(len(slots), num_library, np.random.default_rng(12345))
        fast_v = _eval_assignment_fast(probe, prep)
        probe_circ = build_clifford_circuit_from_assignment(
            target_circuit, slots, probe, library, validate=False, clifford_atol=clifford_atol
        )
        slow_v, _, _ = exact_expectation_vector(probe_circ, observables)
        if not np.allclose(fast_v, slow_v, atol=1e-9):
            raise RuntimeError(f"The fast evaluation {fast_v} disagrees with the Stim simulation {slow_v}.")

    n_evaluated = 0
    if strategy == "pool_fps":
        assignments = _generate_pool_assignments(len(slots), num_library, pool_size, num_training_circuits, dedup, rng)
        n_evaluated = len(assignments)
        vectors = np.array(
            [_eval_assignment_fast(a, prep) for a in _progress(assignments, "evaluating Clifford pool", progress)],
            dtype=float,
        )
        chosen = farthest_point_selection(vectors, num_training_circuits, rng=rng, weights=coefficients)
        sel_assignments = [assignments[i] for i in chosen]
        sel_vectors = vectors[chosen]

    elif strategy == "optimize":
        # With coefficients, spread the combined value.
        combine_weights = None if coefficients is None else np.asarray(coefficients, dtype=float)
        if target_grid is None:
            target_grid = _default_optimize_grid(
                prep, slots, num_library, num_training_circuits, combine_weights, dedup, rng
            )
        sel_assignments, sel_vectors_list = _optimize_to_targets(
            prep,
            slots,
            num_library,
            primary_index,
            list(target_grid),
            max_descent_passes,
            max_opt_slots,
            rng,
            progress,
            combine_weights=combine_weights,
        )
        sel_vectors = np.array(sel_vectors_list, dtype=float)

    else:
        raise ValueError(f"Unknown strategy {strategy!r}. Use 'pool_fps' or 'optimize'.")

    # Optional Stim translation and validation of the selected circuits.
    circuits, stims, maps = [], [], []
    for a in _progress(sel_assignments, "building selected circuits", progress):
        circ = build_clifford_circuit_from_assignment(
            target_circuit,
            slots,
            a,
            library,
            identity_delay=identity_delay,
            validate=validate_selected,
            clifford_atol=clifford_atol,
        )
        circuits.append(circ)
        if build_stim_circuits:
            stim_c, mmap = training_circuit_to_stim(circ, return_measurement_map=True)
            stims.append(stim_c)
            maps.append(mmap)
        else:
            stims.append(None)
            maps.append(None)

    ts = TrainingSet(
        pulse_circuits=circuits,
        stim_circuits=stims,
        measurement_maps=maps,
        exact_values=np.atleast_2d(sel_vectors),
        observable_labels=labels,
        target_circuit=target_circuit,
        metadata={
            "strategy": strategy,
            "num_slots": len(slots),
            "num_library": num_library,
            "rng_seed": rng_seed,
            "num_evaluated": n_evaluated,
        },
    )
    if verbose:
        print(f"Selected {ts.num_training} training circuits with spread {ts.spread()}.")
    return ts


def _generate_pool_assignments(
    num_slots: int,
    num_library: int,
    pool_size: int | None,
    num_training: int,
    dedup: bool,
    rng: np.random.Generator,
) -> list[tuple[int, ...]]:
    """Generate random library assignments for the candidate pool, deduplicated if ``dedup``."""
    if num_slots == 0:
        return [()]
    if pool_size is None:
        pool_size = max(200, 50 * num_training)
    seen = set()
    assignments: list[tuple[int, ...]] = []

    max_attempts = pool_size * 20  # Stops the loop when there are few distinct assignments.
    attempts = 0
    while len(assignments) < pool_size and attempts < max_attempts:
        a = random_assignment(num_slots, num_library, rng)
        attempts += 1
        if dedup:
            if a in seen:
                continue
            seen.add(a)
        assignments.append(a)
    return assignments


def _default_optimize_grid(  # noqa: PLR0913
    prep: dict,
    slots: Sequence[int],
    num_library: int,
    num_training: int,
    combine_weights: np.ndarray | None,
    dedup: bool,
    rng: np.random.Generator,
) -> list[float]:
    """Build a uniform grid of target values for ``"optimize"``.

    The range of the grid is the range of the combined value over a random sample of assignments.
    """
    if combine_weights is None or len(slots) == 0:
        return list(np.linspace(-1.0, 1.0, num_training))
    probe = _generate_pool_assignments(len(slots), num_library, max(200, 20 * num_training), num_training, dedup, rng)
    combined = np.array([float(_eval_assignment_fast(a, prep) @ combine_weights) for a in probe], dtype=float)
    lo, hi = float(combined.min()), float(combined.max())
    if hi - lo < _VALUE_ATOL:  # The reachable range is too small to use.
        span = max(abs(lo), abs(hi), 1e-3)
        lo, hi = -span, span
    return list(np.linspace(lo, hi, num_training))


def _optimize_to_targets(  # noqa: PLR0913
    prep: dict,
    slots: Sequence[int],
    num_library: int,
    primary_index: int,
    targets: Sequence[float],
    max_passes: int,
    max_opt_slots: int,
    rng: np.random.Generator,
    progress: bool,
    combine_weights: np.ndarray | None = None,
) -> tuple[list, list]:
    """Search, by coordinate descent, for a library assignment whose exact value matches each target value.

    The objective is the exact value of the observable at ``primary_index``, or ``vec @ combine_weights`` when
    ``combine_weights`` is given.

    Args:
        prep: Evaluator state, from :func:`_prepare_fast_evaluator`.
        slots: Instruction indices of the non-Clifford prx gates.
        num_library: Number of gates in the Clifford library.
        primary_index: Observable whose exact value is the objective when ``combine_weights`` is not given.
        targets: Target value of each training circuit.
        max_passes: Largest number of coordinate-descent passes per target value.
        max_opt_slots: Largest number of slots changed by the descent.
        rng: Random number generator for the initial assignment and the slot order.
        progress: Whether to show a tqdm progress bar.
        combine_weights: Coefficients of the terms. If given, the objective is their weighted sum.

    Returns:
        The library assignment of each target value, and its exact values.

    """

    def objective(vec: np.ndarray) -> float:
        if combine_weights is not None:
            return float(vec @ combine_weights)
        return float(vec[primary_index])

    num_slots = len(slots)
    assignments, vectors = [], []
    if num_slots == 0:
        vec = _eval_assignment_fast((), prep)
        return [()], [vec]

    for t in _progress(targets, "optimizing toward targets", progress):
        assignment = list(random_assignment(num_slots, num_library, rng))
        # On deep circuits, change only a random subset of the slots.
        if num_slots > max_opt_slots:
            opt_slots = [int(s) for s in rng.choice(num_slots, size=max_opt_slots, replace=False)]
        else:
            opt_slots = list(range(num_slots))

        vec = _eval_assignment_fast(tuple(assignment), prep)
        best = abs(objective(vec) - t)
        for _ in range(max_passes):
            improved = False
            for s in rng.permutation(opt_slots).tolist():
                current = assignment[s]
                best_choice, best_err, best_vec = current, best, vec
                for c in range(num_library):
                    if c == current:
                        continue
                    assignment[s] = c
                    v = _eval_assignment_fast(tuple(assignment), prep)
                    err = abs(objective(v) - t)
                    if err < best_err - 1e-12:
                        best_err, best_choice, best_vec = err, c, v
                assignment[s] = best_choice
                if best_choice != current:
                    best, vec, improved = best_err, best_vec, True
            if not improved or best < _VALUE_ATOL:
                break
        assignments.append(tuple(assignment))
        vectors.append(vec)
    return assignments, vectors


def _operation_to_dict(operation: CircuitOperation) -> dict:
    """Convert a CircuitOperation to a JSON-compatible dict."""
    return {
        "name": operation.name,
        "locus": list(operation.locus),
        "args": dict(operation.args),
        "implementation": operation.implementation,
    }


def _operation_from_dict(data: dict) -> CircuitOperation:
    """Convert a dict written by :func:`_operation_to_dict` back to a CircuitOperation."""
    return CircuitOperation(
        name=data["name"],
        locus=tuple(data["locus"]),
        args=dict(data["args"]),
        implementation=data.get("implementation"),
    )


def _circuit_to_dict(circuit: Circuit) -> dict:
    """Convert a pulse Circuit to a JSON-compatible dict."""
    return {
        "name": circuit.name,
        "instructions": [_operation_to_dict(op) for op in circuit.instructions],
        "metadata": circuit.metadata,
    }


def _circuit_from_dict(data: dict) -> Circuit:
    """Convert a dict written by :func:`_circuit_to_dict` back to a pulse Circuit."""
    return Circuit(
        name=data["name"],
        instructions=tuple(_operation_from_dict(op) for op in data["instructions"]),
        metadata=data.get("metadata"),
    )


def save_training_set(ts: TrainingSet, path_prefix: str) -> dict:
    """Save a training set to a JSON file.

    Args:
        ts: The training set.
        path_prefix: Path of the file without the ``.json`` extension. Missing directories are created.

    Returns:
        The written path, under the key ``"json"``.

    """
    directory = os.path.dirname(path_prefix)
    if directory:
        os.makedirs(directory, exist_ok=True)

    json_path = f"{path_prefix}.json"
    payload = {
        "pulse_circuits": [_circuit_to_dict(c) for c in ts.pulse_circuits],
        "target_circuit": None if ts.target_circuit is None else _circuit_to_dict(ts.target_circuit),
        "stim_circuits": [None if c is None else str(c) for c in ts.stim_circuits],
        "measurement_maps": ts.measurement_maps,
        "exact_values": np.asarray(ts.exact_values, dtype=float).tolist(),
        "observable_labels": ts.observable_labels,
        "metadata": ts.metadata,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    return {"json": json_path}


def load_training_set(path_prefix: str, load_target: bool = True) -> TrainingSet:
    """Load a training set written by :func:`save_training_set`.

    Args:
        path_prefix: The path prefix used when saving.
        load_target: Whether to load the target circuit, if it was saved.

    Returns:
        The training set.

    """
    with open(f"{path_prefix}.json") as f:
        payload = json.load(f)

    pulse_circuits = [_circuit_from_dict(c) for c in payload["pulse_circuits"]]

    target_circuit = None
    if load_target and payload.get("target_circuit") is not None:
        target_circuit = _circuit_from_dict(payload["target_circuit"])

    stim_circuits = [None if text is None else stim.Circuit(text) for text in payload["stim_circuits"]]

    return TrainingSet(
        pulse_circuits=pulse_circuits,
        stim_circuits=stim_circuits,
        measurement_maps=payload["measurement_maps"],
        exact_values=np.asarray(payload["exact_values"], dtype=float),
        observable_labels=payload["observable_labels"],
        target_circuit=target_circuit,
        metadata=payload.get("metadata", {}),
    )
