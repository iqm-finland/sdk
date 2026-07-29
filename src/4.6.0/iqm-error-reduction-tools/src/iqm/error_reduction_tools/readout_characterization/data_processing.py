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

"""Functions for computing readout error correlations from characterization data."""

from typing import TypeAlias, TypedDict

import numpy as np

CountsByPrep: TypeAlias = dict[str, dict[str, float]]
"""Mapping from preparation string to measurement outcome counts.

Outer key: preparation string where each character is ``'I'`` (prepare |0⟩) or
``'X'`` (prepare |1⟩). Inner key: measured bitstring (e.g. ``'0110'``).
Value: number of shots (or normalized probability) for that outcome.
"""

ErrorId: TypeAlias = int | tuple[int, int]
"""Error category identifier.

A single ``int`` (0–3) labels a single-qubit preparation/measurement outcome:

- 0: prepared |0⟩, measured 0 (correct)
- 1: prepared |1⟩, measured 1 (correct)
- 2: prepared |0⟩, measured 1 (0→1 flip)
- 3: prepared |1⟩, measured 0 (1→0 flip)

A ``tuple[int, int]`` labels a joint two-qubit error pair,
e.g. ``(2, 3)`` means qubit i had a 0→1 flip and qubit j had a 1→0 flip.
"""

AssignmentMatrix: TypeAlias = np.ndarray
"""2×2 column-stochastic assignment matrix ``[[P(0|0), P(0|1)], [P(1|0), P(1|1)]]`` characterizing
the readout errors for a qubit."""


class CalibrationData(TypedDict):
    """Calibration dataset container."""

    counts_by_prep: CountsByPrep
    """Measurement counts grouped by preparation string."""
    measured_qubits: list[str]
    """Ordered list of qubit labels corresponding to each bit position in the measurement bitstrings."""


class ErrorProbabilities(TypedDict):
    """Single-qubit characterization probabilities and standard deviations."""

    charact_data: dict[str, AssignmentMatrix]
    """Mapping from qubit name to its readout errors.

    This can be passed directly to :class:`iqm.error_reduction_tools.rem.mitigation.ReadoutErrorMitigation.__init__`.
    """
    charact_data_std: dict[str, np.ndarray]
    """Mapping from qubit name to 2×2 matrix of binomial standard deviations for the entries of :attr:`charact_data`."""


class SingleCovarianceData(TypedDict):
    """Single- and double-twirled covariance output container."""

    covariance_matrices: dict[int, np.ndarray]
    """NxN covariance matrices keyed by error ID (0–3 for single-twirled, 0–2 for double-twirled).
    Entry [i, j] is the covariance between the error/syndrome flip on qubit i and the syndrome flip on qubit j."""
    error_labels: list[tuple[int, str]]
    """List of ``(error_id, description)`` pairs providing human-readable labels for each error category."""
    measured_qubits: list[str]
    """Ordered list of qubit labels corresponding to the matrix row/column indices."""


class StateCovarianceData(TypedDict):
    """State-dependent covariance output container."""

    covariance_matrices: dict[tuple[int, int], np.ndarray]
    """NxN covariance matrices keyed by error ID pairs (error_i, error_j).
    Entry [m, n] is the covariance between error type error_i on qubit m and error type
    error_j on qubit n for a key (error_i, error_j), conditioned on the respective preparation states of both qubits."""
    error_labels: list[tuple[tuple[int, int], str]]
    """List of ``((error_i, error_j), description)`` pairs providing human-readable labels for each error pair."""
    measured_qubits: list[str]
    """Ordered list of qubit labels corresponding to the matrix row/column indices."""


def _get_error_ids_for_shot(bitstring: str, prep_str: str) -> list[int]:
    """Compute error IDs for each qubit in a single measurement shot.

    Maps preparation-measurement combinations to standardized error categories
    for correlation analysis. Uses little-endian bit ordering (rightmost bit = qubit 0).

    Error ID encoding:
        0: Prepared |0>, measured 0 (correct)
        1: Prepared |1>, measured 1 (correct)
        2: Prepared |0>, measured 1 (0→1 bit flip error)
        3: Prepared |1>, measured 0 (1→0 bit flip error)

    Args:
        bitstring: Measurement outcome string (e.g., '0101'). In little-endian format,
            the rightmost bit corresponds to qubit 0.
        prep_str: Preparation string with same length as bitstring.
            'I' = prepared |0>, 'X' = prepared |1>.

    Returns:
        List of error IDs (0-3), one per qubit, in qubit index order.


    """
    len_prep = len(prep_str)
    if len(bitstring) != len_prep:
        raise ValueError(f"Bitstring length ({len(bitstring)}) does not match preparation string length ({len_prep})")
    error_ids = []
    for bit_idx in range(len_prep):
        prep_bit = prep_str[bit_idx]
        meas_bit = bitstring[-1 - bit_idx]  # Little-endian indexing

        if prep_bit == "I" and meas_bit == "0":
            # Prepared |0>, measured 0 (correct)
            error_id = 0
        elif prep_bit == "X" and meas_bit == "1":
            # Prepared |1>, measured |1> (correct)
            error_id = 1
        elif prep_bit == "I" and meas_bit == "1":
            # Prepared |0>, measured |1> (0→1 error)
            error_id = 2
        elif prep_bit == "X" and meas_bit == "0":
            # Prepared |1>, measured |0> (1→0 error)
            error_id = 3
        else:
            raise ValueError(f"Invalid prep_bit '{prep_bit}' or meas_bit '{meas_bit}'")

        error_ids.append(error_id)
    return error_ids


def compute_error_probabilities(
    data: CalibrationData,
) -> ErrorProbabilities:
    """Compute single-qubit readout error probabilities from calibration data.

    Calculates conditional error probabilities P(measured | prepared) for each
    measured qubit, aggregating over all preparation strings. Returns per-qubit
    2×2 column-stochastic assignment matrices that are directly compatible with
    :class:`~iqm.error_reduction_tools.rem.mitigation.ReadoutErrorMitigation`.

    Args:
        data: Calibration data.

    Returns:
        Single-qubit error probabilities.

    Raises:
        ValueError: If calibration data is empty or ``measured_qubits`` is missing.

    Example:
        >>> result = compute_error_probabilities(data)
        >>> # Directly usable by REM
        >>> from src.iqm.error_reduction_tools.rem.mitigation import ReadoutErrorMitigation
        >>> mitigator = ReadoutErrorMitigation(result["charact_data"])
        >>> # Inspect individual qubit
        >>> print(result["charact_data"]["QB1"])

    """
    counts_by_prep = data.get("counts_by_prep", {})
    measured_qubits = data.get("measured_qubits", [])

    if not counts_by_prep:
        raise ValueError("'counts_by_prep' is empty. No calibration data to process.")

    if not measured_qubits:
        raise ValueError("'measured_qubits' is empty. Cannot compute error probabilities.")

    num_qubits = len(measured_qubits)

    # Count preparations and errors
    count_prep_0 = np.zeros(num_qubits)  # Times each qubit was prepared in |0>
    count_prep_1 = np.zeros(num_qubits)  # Times each qubit was prepared in |1>
    count_error_0to1 = np.zeros(num_qubits)  # Times prepared |0>, measured |1>
    count_error_1to0 = np.zeros(num_qubits)  # Times prepared |1>, measured |0>

    for prep_str, counts_dict in counts_by_prep.items():
        for bitstring, count in counts_dict.items():
            for bit_idx in range(num_qubits):
                prepared_1 = prep_str[bit_idx] == "X"
                measured_1 = bitstring[-1 - bit_idx] == "1"

                if prepared_1:
                    count_prep_1[bit_idx] += count
                    if not measured_1:
                        count_error_1to0[bit_idx] += count
                else:
                    count_prep_0[bit_idx] += count
                    if measured_1:
                        count_error_0to1[bit_idx] += count

    # Compute conditional probabilities
    p_0to1 = np.divide(count_error_0to1, count_prep_0, out=np.zeros(num_qubits), where=count_prep_0 > 0)
    p_1to0 = np.divide(count_error_1to0, count_prep_1, out=np.zeros(num_qubits), where=count_prep_1 > 0)

    # Compute standard deviations using binomial statistics: sqrt(p*(1-p)/n)
    std_0to1 = np.zeros(num_qubits)
    std_1to0 = np.zeros(num_qubits)

    for q in range(num_qubits):
        if count_prep_0[q] > 0:
            std_0to1[q] = np.sqrt(p_0to1[q] * (1 - p_0to1[q]) / count_prep_0[q])
        if count_prep_1[q] > 0:
            std_1to0[q] = np.sqrt(p_1to0[q] * (1 - p_1to0[q]) / count_prep_1[q])

    # Build per-qubit 2x2 assignment matrices and standard deviation matrices
    charact_data: dict[str, np.ndarray] = {}
    charact_data_std: dict[str, np.ndarray] = {}

    for i, qubit in enumerate(measured_qubits):
        charact_data[qubit] = np.array([[1.0 - p_0to1[i], p_1to0[i]], [p_0to1[i], 1.0 - p_1to0[i]]])
        charact_data_std[qubit] = np.array([[std_0to1[i], std_1to0[i]], [std_0to1[i], std_1to0[i]]])

    return {
        "charact_data": charact_data,
        "charact_data_std": charact_data_std,
    }


# Map 'error_id' to required preparation state (0=|0>, 1=|1>) for conditional probabilities
ERROR_TO_PREP: dict[int, int] = {0: 0, 2: 0, 1: 1, 3: 1}


def _compute_prep_counts(
    counts_by_prep: CountsByPrep,
    num_qubits: int,
) -> np.ndarray:
    """Compute total preparation state counts for each qubit.

    Aggregates how many times each qubit was prepared in |0> vs |1>
    across all circuits and measurement outcomes.

    Args:
        counts_by_prep: Measurement counts grouped by preparation string.
        num_qubits: Number of qubits in the system.

    Returns:
        Array of shape (2, ``num_qubits``) where:
            - prep_counts[0, q] = total |0> preparations for qubit q
            - prep_counts[1, q] = total |1> preparations for qubit q

    """
    prep_counts = np.zeros((2, num_qubits))
    for prep_str, counts_dict in counts_by_prep.items():
        n_shots = sum(counts_dict.values())
        for q in range(num_qubits):
            prep_state = 0 if prep_str[q] == "I" else 1
            prep_counts[prep_state, q] += n_shots
    return prep_counts


def compute_single_twirled_covariance(
    data: CalibrationData,
) -> SingleCovarianceData:
    """Compute single-twirled readout error correlations between qubit pairs.

    Calculates covariances where one qubit has a state-dependent error (conditioned
    on its preparation) and the other has a preparation-averaged (twirled) error.
    This reveals correlations that persist regardless of the second qubit's state.

    Args:
        data: Calibration data from :func:`run_calibration_circuits`.

    Returns:
        Single- and double-twirled covariance output.

    .. note::

        Error IDs match :func:`_get_error_ids_for_shot` encoding (0=00, 1=11, 2=01, 3=10).
        Correlation[i,j] = P(error_i, syndrome_j) - P(error_i)P(syndrome_j)

    """
    counts_by_prep = data.get("counts_by_prep", {})
    measured_qubits = data.get("measured_qubits", [])

    if not counts_by_prep:
        raise ValueError("'counts_by_prep' is empty. No calibration data to process.")

    if not measured_qubits:
        raise ValueError("'measured_qubits' is empty. Cannot compute covariances.")

    num_qubits = len(measured_qubits)

    error_1q_ids = [0, 1, 2, 3]
    error_2q_ids_set = {(0, 2), (0, 3), (1, 2), (1, 3), (2, 2), (3, 3), (2, 3), (3, 2)}

    # Compute prep counts efficiently
    prep_counts = _compute_prep_counts(counts_by_prep, num_qubits)
    total_shots = prep_counts[0] + prep_counts[1]

    # Track error counts and joint counts
    error_counts = np.zeros((4, num_qubits))  # indexed by error_id
    syndrome_flip_counts = np.zeros(num_qubits)  # counts of error_id in {2, 3}
    joint_counts = {eid: np.zeros((num_qubits, num_qubits)) for eid in error_1q_ids}

    for prep_str, counts_dict in counts_by_prep.items():
        for bitstring, count in counts_dict.items():
            error_shot = _get_error_ids_for_shot(bitstring, prep_str)
            for q in range(num_qubits):
                eid = error_shot[q]
                error_counts[eid, q] += count
                if eid in (2, 3):
                    syndrome_flip_counts[q] += count
            for q1 in range(num_qubits):
                for q2 in range(num_qubits):
                    if q1 != q2 and (error_shot[q1], error_shot[q2]) in error_2q_ids_set:
                        joint_counts[error_shot[q1]][q1, q2] += count

    # Conditional probabilities
    vec_probs = np.zeros((4, num_qubits))
    for eid in error_1q_ids:
        req_prep = ERROR_TO_PREP[eid]
        vec_probs[eid] = np.divide(
            error_counts[eid],
            prep_counts[req_prep],
            out=np.zeros(num_qubits),
            where=prep_counts[req_prep] > 0,
        )

    vec_syndrome_probs = np.divide(
        syndrome_flip_counts,
        total_shots,
        out=np.zeros(num_qubits),
        where=total_shots > 0,
    )

    # Compute correlation coefficients
    correlation_matrices = {eid: np.zeros((num_qubits, num_qubits)) for eid in error_1q_ids}
    for eid in error_1q_ids:
        req_prep_i = ERROR_TO_PREP[eid]
        norm = prep_counts[req_prep_i]
        for i in range(num_qubits):
            if norm[i] > 0:
                for j in range(num_qubits):
                    if i != j:
                        p_ij = joint_counts[eid][i, j] / norm[i]
                        correlation_matrices[eid][i, j] = p_ij - (vec_probs[eid, i] * vec_syndrome_probs[j])

    label_map = {
        0: "P(01|00) + P(00|01)",
        1: "P(10|11) + P(11|10)",
        2: "P(11|00) + P(10|01)",
        3: "P(01|10) + P(00|11)",
    }
    error_labels: list[tuple[int, str]] = [(k, f"{label_map[k]}") for k in correlation_matrices]

    covariance_data: SingleCovarianceData = {
        "covariance_matrices": correlation_matrices,
        "error_labels": error_labels,
        "measured_qubits": measured_qubits,
    }

    return covariance_data


def _compute_prep_joint_counts(
    counts_by_prep: CountsByPrep,
    num_qubits: int,
) -> np.ndarray:
    """Compute joint preparation state counts for all qubit pairs.

    For each qubit pair (i, j) and preparation state combination,
    counts total shots where qubit i was in one state and j in another.

    Args:
        counts_by_prep: Measurement counts grouped by preparation string.
        num_qubits: Number of qubits in the system.

    Returns:
        Array of shape (2, 2, ``num_qubits``, ``num_qubits``) where
            prep_joint_counts[prep_i, prep_j, i, j] = total shots with
            qubit i prepared in state prep_i (0=|0>, 1=|1>) and
            qubit j prepared in state prep_j.

    .. note::

        Diagonal (i==j) entries are set to 0 as they're not used in correlations.

    """
    prep_joint_counts = np.zeros((2, 2, num_qubits, num_qubits))
    for prep_str, counts_dict in counts_by_prep.items():
        n_shots = sum(counts_dict.values())
        for q1 in range(num_qubits):
            prep1 = 0 if prep_str[q1] == "I" else 1
            for q2 in range(num_qubits):
                if q1 != q2:
                    prep2 = 0 if prep_str[q2] == "I" else 1
                    prep_joint_counts[prep1, prep2, q1, q2] += n_shots
    return prep_joint_counts


def compute_state_dependent_covariance(
    data: CalibrationData,
) -> StateCovarianceData:
    """Compute fully state-dependent readout error correlations between qubit pairs.

    Calculates covariances conditioned on the preparation states of both qubits.
    This captures the full joint error distribution P(measure_i, measure_j | prep_i, prep_j).

    Args:
        data: Calibration data from :func:`run_calibration_circuits`.

    Returns:
        State-dependent covariance output.

    .. note::

        Computes 12 correlation types corresponding to all relevant error pair
        combinations: (2,0), (3,0), (2,1), (3,1), (0,2), (0,3), (1,2), (1,3),
        (2,2), (3,3), (2,3), (3,2). Excludes trivial (0,0), (0,1), (1,0), (1,1).

    """
    counts_by_prep = data.get("counts_by_prep", {})
    measured_qubits = data.get("measured_qubits", [])

    if not counts_by_prep:
        raise ValueError("'counts_by_prep' is empty. No calibration data to process.")

    if not measured_qubits:
        raise ValueError("'measured_qubits' is empty. Cannot compute covariances.")

    num_qubits = len(measured_qubits)

    error_2q_ids = [
        (2, 0),
        (3, 0),
        (2, 1),
        (3, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 2),
        (3, 3),
        (2, 3),
        (3, 2),
    ]
    error_2q_ids_set = set(error_2q_ids)

    # Compute prep counts efficiently
    prep_counts = _compute_prep_counts(counts_by_prep, num_qubits)
    prep_joint_counts = _compute_prep_joint_counts(counts_by_prep, num_qubits)

    # Track error counts
    error_counts = np.zeros((4, num_qubits))
    joint_counts = {eid: np.zeros((num_qubits, num_qubits)) for eid in error_2q_ids}

    for prep_str, counts_dict in counts_by_prep.items():
        for bitstring, count in counts_dict.items():
            error_shot = _get_error_ids_for_shot(bitstring, prep_str)
            for q in range(num_qubits):
                error_counts[error_shot[q], q] += count
            for q1 in range(num_qubits):
                for q2 in range(num_qubits):
                    if q1 != q2:
                        eid_pair = (error_shot[q1], error_shot[q2])
                        if eid_pair in error_2q_ids_set:
                            joint_counts[eid_pair][q1, q2] += count

    # Conditional single-qubit probabilities
    vec_probs = np.zeros((4, num_qubits))
    for error_id in range(4):
        req_prep = ERROR_TO_PREP[error_id]
        vec_probs[error_id] = np.divide(
            error_counts[error_id],
            prep_counts[req_prep],
            out=np.zeros(num_qubits),
            where=prep_counts[req_prep] > 0,
        )

    # Compute correlation coefficients
    correlation_matrices = {eid: np.zeros((num_qubits, num_qubits)) for eid in error_2q_ids}
    for error_pair in error_2q_ids:
        req_prep_i = ERROR_TO_PREP[error_pair[0]]
        req_prep_j = ERROR_TO_PREP[error_pair[1]]
        for i in range(num_qubits):
            for j in range(num_qubits):
                if i != j:
                    norm = prep_joint_counts[req_prep_i, req_prep_j, i, j]
                    if norm > 0:
                        p_ij = joint_counts[error_pair][i, j] / norm
                        correlation_matrices[error_pair][i, j] = p_ij - (
                            vec_probs[error_pair[0], i] * vec_probs[error_pair[1], j]
                        )

    label_map = {
        (0, 2): "P(01|00)",
        (0, 3): "P(00|01)",
        (1, 2): "P(11|10)",
        (1, 3): "P(10|11)",
        (2, 0): "P(10|00)",
        (3, 0): "P(00|10)",
        (2, 1): "P(11|01)",
        (3, 1): "P(01|11)",
        (2, 2): "P(11|00)",
        (3, 3): "P(00|11)",
        (2, 3): "P(10|01)",
        (3, 2): "P(01|10)",
    }
    error_labels: list[tuple[tuple[int, int], str]] = [(k, f"{label_map[k]}") for k in correlation_matrices]

    covariance_data: StateCovarianceData = {
        "covariance_matrices": correlation_matrices,
        "error_labels": error_labels,
        "measured_qubits": measured_qubits,
    }

    return covariance_data


def compute_double_twirled_covariance(  # noqa: PLR0912
    data: CalibrationData,
) -> SingleCovarianceData:
    """Compute double-twirled (syndrome-based) readout error correlations.

    Aggregates state-dependent errors into preparation-independent syndrome
    categories based on which qubits experience readout flips. This reveals
    correlated errors that persist regardless of preparation states.

    Syndrome categories:
        0: Qubit j flips, qubit i correct (errors: (0,2), (0,3), (1,2), (1,3))
        1: Qubit i flips, qubit j correct (errors: (2,0), (3,0), (2,1), (3,1))
        2: Both qubits flip (errors: (2,2), (3,3), (2,3), (3,2))

    Args:
        data: Calibration data from :func:`run_calibration_circuits`.

    Returns:
        Single- and double-twirled covariance output.

    """
    counts_by_prep = data.get("counts_by_prep", {})
    measured_qubits = data.get("measured_qubits", [])

    if not counts_by_prep:
        raise ValueError("'counts_by_prep' is empty. No calibration data to process.")

    if not measured_qubits:
        raise ValueError("'measured_qubits' is empty. Cannot compute covariances.")

    num_qubits = len(measured_qubits)

    error_2q_ids = [
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 0),
        (3, 0),
        (2, 1),
        (3, 1),
        (2, 2),
        (3, 3),
        (2, 3),
        (3, 2),
    ]
    error_2q_ids_set = set(error_2q_ids)

    # Compute prep counts efficiently
    prep_counts = _compute_prep_counts(counts_by_prep, num_qubits)
    prep_joint_counts = _compute_prep_joint_counts(counts_by_prep, num_qubits)

    # Track error counts
    error_counts = np.zeros((4, num_qubits))
    joint_counts = {eid: np.zeros((num_qubits, num_qubits)) for eid in error_2q_ids}

    for prep_str, counts_dict in counts_by_prep.items():
        for bitstring, count in counts_dict.items():
            error_shot = _get_error_ids_for_shot(bitstring, prep_str)
            for q in range(num_qubits):
                error_counts[error_shot[q], q] += count
            for q1 in range(num_qubits):
                for q2 in range(num_qubits):
                    if q1 != q2:
                        eid_pair = (error_shot[q1], error_shot[q2])
                        if eid_pair in error_2q_ids_set:
                            joint_counts[eid_pair][q1, q2] += count

    # Conditional single-qubit probabilities
    vec_probs = np.zeros((4, num_qubits))
    for error_id in range(4):
        req_prep = ERROR_TO_PREP[error_id]
        vec_probs[error_id] = np.divide(
            error_counts[error_id],
            prep_counts[req_prep],
            out=np.zeros(num_qubits),
            where=prep_counts[req_prep] > 0,
        )

    # Compute correlation coefficients aggregated by syndrome type
    correlation_matrices = {syndrome_id: np.zeros((num_qubits, num_qubits)) for syndrome_id in (0, 1, 2)}

    for error_pair in error_2q_ids:
        # Determine syndrome type
        if error_pair[0] in {0, 1} and error_pair[1] in {2, 3}:
            syndrome_error = 0  # First qubit correct, second flips
        elif error_pair[0] in {2, 3} and error_pair[1] in {0, 1}:
            syndrome_error = 1  # First qubit flips, second correct
        elif error_pair[0] in {2, 3} and error_pair[1] in {2, 3}:
            syndrome_error = 2  # Both qubits flip
        else:
            continue

        req_prep_i = ERROR_TO_PREP[error_pair[0]]
        req_prep_j = ERROR_TO_PREP[error_pair[1]]

        for i in range(num_qubits):
            for j in range(num_qubits):
                if i != j:
                    norm = prep_joint_counts[req_prep_i, req_prep_j, i, j]
                    if norm > 0:
                        p_ij = joint_counts[error_pair][i, j] / norm
                        correlation_matrices[syndrome_error][i, j] += p_ij - (
                            vec_probs[error_pair[0], i] * vec_probs[error_pair[1], j]
                        )

    label_map = {
        0: "P(j0|j1) + P(j1|j0)",
        1: "P(0j|1j) + P(1j|0j)",
        2: "P(11|00) + P(00|11)",
    }
    error_labels: list[tuple[int, str]] = [(k, f"{label_map[k]}") for k in correlation_matrices]
    covariance_data: SingleCovarianceData = {
        "covariance_matrices": correlation_matrices,
        "error_labels": error_labels,
        "measured_qubits": measured_qubits,
    }
    return covariance_data
