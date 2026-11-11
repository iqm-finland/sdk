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

"""Clifford properties of prx gates, and their conversion to Stim gates."""

from __future__ import annotations

from functools import cache
from typing import Any
import warnings

import numpy as np


def normalize_angle_2pi(angle: float, atol: float = 1e-12) -> float:
    """Map an angle to the interval [0, 2*pi).

    Args:
        angle: Angle in radians.
        atol: Values within this tolerance of 0 or 2*pi map to 0.

    Returns:
        Equivalent angle in [0, 2*pi).

    """
    x = float(angle) % (2.0 * np.pi)

    if np.isclose(x, 0.0, atol=atol) or np.isclose(x, 2.0 * np.pi, atol=atol):
        return 0.0

    return x


def r_gate_so3_matrix(theta: float, phi: float) -> np.ndarray:
    """Compute the SO(3) Bloch-sphere rotation matrix of prx(theta, phi).

    Rotates the Bloch vector by theta around the axis n = (cos(phi),
    sin(phi), 0). A single-qubit unitary is Clifford iff this SO(3) action is a
    signed permutation matrix.

    Args:
        theta: Rotation angle in radians.
        phi: Equatorial-axis angle in radians.

    Returns:
        3x3 real SO(3) rotation matrix.

    """
    theta = float(theta)
    phi = float(phi)

    nx = np.cos(phi)
    ny = np.sin(phi)
    nz = 0.0

    n = np.array([nx, ny, nz], dtype=float)

    # Cross-product matrix K such that K @ v = n x v.
    K = np.array(
        [
            [0.0, -nz, ny],
            [nz, 0.0, -nx],
            [-ny, nx, 0.0],
        ],
        dtype=float,
    )

    identity = np.eye(3)

    # Rodrigues rotation formula.
    R = np.cos(theta) * identity + (1.0 - np.cos(theta)) * np.outer(n, n) + np.sin(theta) * K

    return R


def is_signed_permutation_matrix(M: np.ndarray, atol: float = 1e-9) -> bool:
    """Check whether a matrix is a signed permutation matrix with determinant +1.

    Args:
        M: Candidate 3x3 matrix.
        atol: Numerical tolerance.

    Returns:
        True if M has one entry +1 or -1 per row and column, the rest 0, and determinant +1.

    """
    M_round = np.zeros_like(M)

    close_to_pos = np.isclose(M, 1.0, atol=atol)
    close_to_neg = np.isclose(M, -1.0, atol=atol)
    close_to_zero = np.isclose(M, 0.0, atol=atol)

    # Every entry must be approximately -1, 0, or +1.
    if not np.all(close_to_pos | close_to_neg | close_to_zero):
        return False

    M_round[close_to_pos] = 1.0
    M_round[close_to_neg] = -1.0

    # Each row must contain exactly one nonzero entry.
    row_counts = np.sum(np.abs(M_round), axis=1)
    if not np.all(row_counts == 1):
        return False

    # Each column must contain exactly one nonzero entry.
    col_counts = np.sum(np.abs(M_round), axis=0)
    if not np.all(col_counts == 1):
        return False

    # Clifford unitaries correspond to proper rotations in SO(3).
    return bool(np.isclose(np.linalg.det(M_round), 1.0, atol=atol))


def is_logically_single_qubit_clifford_r(
    theta: float,
    phi: float,
    atol: float = 1e-9,
) -> bool:
    """Check whether prx(theta, phi) is a Clifford gate.

    Args:
        theta: Rotation angle in radians.
        phi: Phase angle in radians.
        atol: Numerical tolerance.

    Returns:
        True if prx(theta, phi) is Clifford up to global phase.

    """
    theta = normalize_angle_2pi(theta)
    phi = normalize_angle_2pi(phi)

    R = r_gate_so3_matrix(theta, phi)

    return is_signed_permutation_matrix(R, atol=atol)


def get_clifford_r_library(identity_as_delay: bool = True) -> list:
    """Return the single-qubit Cliffords that a single prx gate implements.

    Args:
        identity_as_delay: If True, the identity operation is represented as a
            delay. If False, it is represented as prx(0, 0).

    Returns:
        A list of dictionaries, each with "label", "theta", "phi", "is_delay".

    """
    library = []

    if identity_as_delay:
        library.append(
            {
                "label": "DelayIdentity",
                "theta": 0.0,
                "phi": 0.0,
                "is_delay": True,
            }
        )
    else:
        library.append(
            {
                "label": "prx(0,0)",
                "theta": 0.0,
                "phi": 0.0,
                "is_delay": False,
            }
        )

    library.extend(
        [
            {
                "label": "RX(pi/2)",
                "theta": np.pi / 2.0,
                "phi": 0.0,
                "is_delay": False,
            },
            {
                "label": "RX(pi)",
                "theta": np.pi,
                "phi": 0.0,
                "is_delay": False,
            },
            {
                "label": "RX(3pi/2)",
                "theta": 3.0 * np.pi / 2.0,
                "phi": 0.0,
                "is_delay": False,
            },
            {
                "label": "RY(pi/2)",
                "theta": np.pi / 2.0,
                "phi": np.pi / 2.0,
                "is_delay": False,
            },
            {
                "label": "RY(pi)",
                "theta": np.pi,
                "phi": np.pi / 2.0,
                "is_delay": False,
            },
            {
                "label": "RY(3pi/2)",
                "theta": 3.0 * np.pi / 2.0,
                "phi": np.pi / 2.0,
                "is_delay": False,
            },
        ]
    )

    return library


def _warn_if_identity_delay_unused(identity_as_delay: bool, identity_delay: float, default: float = 20e-9) -> None:
    """Warn if identity_delay has no effect because identity_as_delay is False.

    Args:
        identity_as_delay: Whether the identity in the library is a delay.
        identity_delay: The requested delay duration, in seconds.
        default: The default of ``identity_delay``. Only a different value triggers the warning.

    """
    if not identity_as_delay and identity_delay != default:
        warnings.warn(
            "identity_delay has no effect when identity_as_delay=False: the identity candidate "
            "is a prx(0, 0) gate, not a delay.",
            UserWarning,
            stacklevel=3,
        )


def signed_permutation_key_from_so3(M: np.ndarray, atol: float = 1e-9) -> tuple:
    """Convert a Clifford in SO(3) representation into a hashable signed-permutation key.

    Rounds M to a signed permutation matrix (entries in {-1, 0, +1}, one nonzero
    entry per row and column, determinant +1) and flattens it into a tuple.

    Args:
        M: Candidate 3x3 SO(3) matrix.
        atol: Numerical tolerance.

    Returns:
        A tuple representation of the rounded signed-permutation matrix.

    Raises:
        ValueError: If M is not close to a valid Clifford SO(3) signed permutation.

    """
    M_round = np.zeros_like(M, dtype=int)

    close_to_pos = np.isclose(M, 1.0, atol=atol)
    close_to_neg = np.isclose(M, -1.0, atol=atol)
    close_to_zero = np.isclose(M, 0.0, atol=atol)

    # Every entry must be numerically close to -1, 0, or +1.
    if not np.all(close_to_pos | close_to_neg | close_to_zero):
        raise ValueError("Matrix is not close to a signed permutation matrix.")

    M_round[close_to_pos] = 1
    M_round[close_to_neg] = -1

    # Exactly one nonzero entry per row.
    if not np.all(np.sum(np.abs(M_round), axis=1) == 1):
        raise ValueError("Invalid signed permutation: row condition failed.")

    # Exactly one nonzero entry per column.
    if not np.all(np.sum(np.abs(M_round), axis=0) == 1):
        raise ValueError("Invalid signed permutation: column condition failed.")

    # Single-qubit Cliffords correspond to proper rotations with determinant +1.
    if not np.isclose(np.linalg.det(M_round), 1.0, atol=atol):
        raise ValueError("Invalid Clifford SO(3): determinant is not +1.")

    return tuple(M_round.reshape(-1).tolist())


def so3_h_matrix() -> np.ndarray:
    """Return the SO(3) action of the Hadamard gate H.

    Returns:
        3x3 real signed-permutation matrix.

    """
    return np.array(
        [
            [0, 0, 1],
            [0, -1, 0],
            [1, 0, 0],
        ],
        dtype=float,
    )


def so3_s_matrix() -> np.ndarray:
    """Return the SO(3) action of the phase gate S.

    Returns:
        3x3 real signed-permutation matrix.

    """
    return np.array(
        [
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ],
        dtype=float,
    )


@cache
def build_stim_single_qubit_clifford_table(atol: float = 1e-9) -> dict:
    """Build a lookup table from SO(3) Clifford actions to Stim gate sequences, cached per ``atol``.

    Generates the 24 single-qubit Cliffords by a breadth-first search over products of H and S, and stores one
    Stim gate sequence per Clifford.

    Args:
        atol: Numerical tolerance.

    Returns:
        Dictionary mapping a signed-permutation key to a list of Stim gate names,
        e.g. ["H", "S", "H"].

    """
    NUM_SINGLE_QUBIT_CLIFFORDS = 24
    H = so3_h_matrix()
    S = so3_s_matrix()

    generators = [
        ("H", H),
        ("S", S),
    ]

    identity = np.eye(3)

    table: dict[tuple[int, ...], list[str]] = {}
    queue: list[tuple[np.ndarray, list[str]]] = []

    identity_key = signed_permutation_key_from_so3(identity, atol=atol)
    table[identity_key] = []
    queue.append((identity, []))

    while queue:
        current_R, current_sequence = queue.pop(0)

        for gate_name, gate_R in generators:
            new_R = gate_R @ current_R
            new_sequence = current_sequence + [gate_name]

            key = signed_permutation_key_from_so3(new_R, atol=atol)

            if key not in table:
                table[key] = new_sequence
                queue.append((new_R, new_sequence))

        if len(table) == NUM_SINGLE_QUBIT_CLIFFORDS:
            break

    if len(table) != NUM_SINGLE_QUBIT_CLIFFORDS:
        raise RuntimeError(
            f"Failed to generate {NUM_SINGLE_QUBIT_CLIFFORDS} single-qubit Cliffords. Generated {len(table)}."
        )

    return table


def stim_sequence_for_r_gate(theta: float, phi: float, atol: float = 1e-9) -> list:
    """Convert a Clifford prx(theta, phi) gate into Stim gates.

    Args:
        theta: Rotation angle in radians.
        phi: Phase angle in radians.
        atol: Numerical tolerance.

    Returns:
        List of Stim gate names.

    Raises:
        ValueError: If prx(theta, phi) is not a Clifford gate.

    """
    if not is_logically_single_qubit_clifford_r(theta, phi, atol=atol):
        raise ValueError(f"prx(theta={theta}, phi={phi}) is not a Clifford gate.")

    theta = normalize_angle_2pi(theta)
    phi = normalize_angle_2pi(phi)

    R = r_gate_so3_matrix(theta, phi)

    key = signed_permutation_key_from_so3(R, atol=atol)
    table = build_stim_single_qubit_clifford_table(atol=atol)

    if key not in table:
        raise ValueError(f"prx(theta={theta}, phi={phi}) is not in the Stim Clifford table.")

    return table[key]


def append_stim_sequence(stim_circuit: Any, sequence: list, qubit: int) -> None:
    """Append a single-qubit Stim gate sequence to a Stim circuit.

    Args:
        stim_circuit: The Stim circuit to append to.
        sequence: List of Stim gate names.
        qubit: Stim index of the qubit.

    """
    for gate_name in sequence:
        stim_circuit.append(gate_name, [qubit])


try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # tqdm is optional

    def _tqdm(iterable: Any = None, **kwargs: Any) -> Any:  # type: ignore[no-redef]
        return iterable


def _progress(iterable: Any, desc: str, show: bool) -> Any:
    """Wrap an iterable in a tqdm progress bar if ``show`` is True and tqdm is installed."""
    if not show:
        return iterable
    wrapped = _tqdm(iterable, desc=desc)
    return wrapped if wrapped is not None else iterable
