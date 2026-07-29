# Copyright 2025 IQM Benchmarks developers
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

"""Shadow Tomography utility functions."""

from collections.abc import Sequence
import random
import secrets
from typing import Literal, cast

from iqm.benchmarks.utils import timeit
from iqm.qiskit_iqm import IQMCircuit
import numpy as np
from numpy.random import RandomState
from qiskit import ClassicalRegister, QuantumCircuit, quantum_info
from qiskit.circuit.library import UnitaryGate
import scipy.linalg as spl


# ruff: noqa: N802, N806
def CUE(random_gen: RandomState, n: int) -> np.ndarray:
    """Prepares single qubit Haar-random unitary (drawn from Circuilar Unitary Ensemble - CUE).

    Args:
        random_gen: A random generator.
        n: The size of the matrix.

    Returns:
        An n x n CUE matrix

    """
    # Generate an N × N complex matrix Z with standard normal entries
    Z = (random_gen.randn(n, n) + 1j * random_gen.randn(n, n)) / np.sqrt(2.0)

    # Perform QR decomposition, Z = QR
    Q, R = spl.qr(Z)

    # Create the diagonal matrix Lambda
    Lambda = np.diag(R.diagonal() / np.abs(R.diagonal()))

    # Compute U = Q * Lambda, which is Haar-distributed
    U = Q @ Lambda

    return U


def validate_shadow_tomography_params(
    clifford_or_haar: Literal["clifford", "haar"],
    cliffords_1q: dict[str, IQMCircuit] | None,
) -> None:
    """Validates the parameters for shadow tomography functions.

    Args:
        clifford_or_haar: Whether to use Clifford or Haar random 1Q gates.
        cliffords_1q: Dictionary of 1-qubit Cliffords in terms of IQM-native r and CZ gates.

    Raises:
        ValueError: If clifford_or_haar is not "clifford" or "haar".
        ValueError: If cliffords_1q is None and clifford_or_haar is "clifford".

    """
    if clifford_or_haar not in ["clifford", "haar"]:
        raise ValueError("clifford_or_haar must be either 'clifford' or 'haar'.")
    if clifford_or_haar == "clifford" and cliffords_1q is None:
        raise ValueError("cliffords_1q dictionary must be provided if clifford_or_haar is 'clifford'.")


@timeit
def local_shadow_tomography(
    qc: QuantumCircuit,
    n_unitaries: int,
    active_qubits: Sequence[int],
    measure_other: Sequence[int] | None = None,
    measure_other_name: str | None = None,
    clifford_or_haar: Literal["clifford", "haar"] = "clifford",
    cliffords_1q: dict[str, QuantumCircuit] | None = None,
) -> tuple[np.ndarray | dict[str, list[str]], list[QuantumCircuit]]:
    """Prepares the circuits to perform local Haar or Clifford shadow tomography.

    Args:
        qc: The quantum circuit to which random unitaries are appended.
        n_unitaries: Number of local random unitaries used.
        active_qubits: The Sequence of active qubits.
        measure_other: Whether to measure other qubits in the qc QuantumCircuit.
        measure_other_name: Name of the classical register to assign measure_other.
        clifford_or_haar: Whether to use Clifford or Haar random 1Q gates.
        cliffords_1q: Dictionary of 1-qubit Cliffords in terms of IQM-native r and CZ gates.

    Raises:
        ValueError: If clifford_or_haar is not "clifford" or "haar".
        ValueError: If cliffords_1q is None and clifford_or_haar is "clifford".

    Returns:
        Random unitaries, in either format:
            * Unitary gate (numpy ndarray),
                composed of local unitaries for each random initialisation and qubit, if clifford_or_haar == 'haar'.
            * Dictionary of lists of Clifford labels corresponding to each RM,
                keys being str(qubit), if clifford_or_haar == 'clifford'.
        List of tomography circuits.

    """
    validate_shadow_tomography_params(clifford_or_haar, cliffords_1q)
    if clifford_or_haar == "clifford":
        # Get the keys of the Clifford dictionaries
        clifford_1q_keys = list(cast(dict, cliffords_1q).keys())

    qclist = []
    seed = random.SystemRandom().randrange(2**32 - 1)  # Init Random Generator
    random_gen: RandomState = np.random.RandomState(seed)  # pylint: disable=no-member

    unitaries: dict[str, list[str]] | np.ndarray
    if clifford_or_haar == "haar":
        unitaries = np.zeros((n_unitaries, len(active_qubits), 2, 2), dtype=np.complex128)
    else:
        unitaries = {str(q): [] for q in active_qubits}

    for u in range(n_unitaries):
        qc_copy = qc.copy()
        for q_idx, qubit in enumerate(active_qubits):
            if clifford_or_haar == "haar":
                temp_u = CUE(random_gen, 2)
                qc_copy.append(UnitaryGate(temp_u), [qubit])
                cast(np.ndarray, unitaries)[u, q_idx, :, :] = np.array(temp_u)
            elif clifford_or_haar == "clifford":
                rand_key = secrets.choice(clifford_1q_keys)
                c_1q = cast(dict, cliffords_1q)[rand_key]
                qc_copy.compose(c_1q, qubits=[qubit], inplace=True)
                cast(dict, unitaries)[str(qubit)].append(rand_key)

        qc_copy.barrier()

        register_rm = ClassicalRegister(len(active_qubits), "RMs")
        qc_copy.add_register(register_rm)
        qc_copy.measure(active_qubits, register_rm)

        if measure_other is not None:
            if measure_other_name is None:
                measure_other_name = "non_RMs"
            register_neighbors = ClassicalRegister(len(measure_other), measure_other_name)
            qc_copy.add_register(register_neighbors)
            qc_copy.measure(measure_other, register_neighbors)

        qclist.append(qc_copy)

    return unitaries, qclist


def get_local_shadow(
    counts: dict[str, int],
    unitary_arg: np.ndarray | Sequence[str],
    subsystem_bit_indices: Sequence[int],
    clifford_or_haar: Literal["clifford", "haar"] = "clifford",
    cliffords_1q: dict[str, IQMCircuit] | None = None,
) -> np.ndarray:
    """Constructs shadows for each individual initialisation.

    Args:
        counts: A dictionary of bit-string counts.
        unitary_arg: Local random unitaries used for a given initialisation, either specified as
                - A numpy array, or
                - A Sequence of Clifford labels.
        subsystem_bit_indices: Bit indices in the counts of the subsystem to construct the shadow of.
        clifford_or_haar: Whether to use Clifford or Haar random 1Q gates.
        cliffords_1q: Dictionary of 1-qubit Cliffords in terms of IQM-native r and CZ gates

    Returns:
        Shadow of considered subsystem.

    """
    if clifford_or_haar not in ["clifford", "haar"]:
        raise ValueError("clifford_or_haar must be either 'clifford' or 'haar'.")
    if clifford_or_haar == "clifford" and cliffords_1q is None:
        raise ValueError("cliffords_1q dictionary must be provided if clifford_or_haar is 'clifford'.")
    if clifford_or_haar == "haar" and isinstance(unitary_arg, Sequence):
        raise ValueError("If clifford_or_haar is 'haar', the unitary operator must be a numpy array.")
    if clifford_or_haar == "clifford" and not isinstance(unitary_arg, Sequence):
        raise ValueError(
            "If clifford_or_haar is 'clifford', the unitary operator must be specified as a Sequence of strings."
        )

    nqubits = len(subsystem_bit_indices)
    rhoshadows = np.zeros([2**nqubits, 2**nqubits], dtype=complex)
    proj = np.stack((np.diag([1, 0]), np.diag([0, 1])), dtype=complex)
    shots = sum(list(counts.values()))

    unitary_op: np.ndarray
    if clifford_or_haar == "haar":
        unitary_op = cast(np.ndarray, unitary_arg)
    else:
        unitary_op = np.zeros((nqubits, 2, 2), dtype=complex)
        for qubit_idx, clif_label in enumerate(unitary_arg):
            unitary_op[qubit_idx, :, :] = quantum_info.Operator(cast(dict, cliffords_1q)[clif_label]).to_matrix()

    for bit_strings, values in counts.items():
        rho_j: int | np.ndarray = 1
        for j in subsystem_bit_indices:
            s_j = int(bit_strings[::-1][j])
            rho_j = np.kron(
                rho_j,
                3
                * np.einsum(
                    "ab,bc,cd",
                    np.transpose(np.conjugate(unitary_op[j, :, :])),
                    proj[s_j, :, :],
                    unitary_op[j, :, :],
                )
                - np.array([[1, 0], [0, 1]]),
            )

        rhoshadows += rho_j * values / shots

    return rhoshadows


def get_negativity(rho: np.ndarray, n_qubits_a: int, n_qubits_b: int) -> float:
    """Computes the negativity of a given density matrix.

    Note that a negativity >0 is only a necessary and sufficient condition
    for entanglement if n_qubits_a = n_qubits_b = 1.
    For more qubits per subsystems it is merely a necessary condition.

    Args:
        rho: Density matrix.
        n_qubits_a: Number of qubits for subsystem A.
        n_qubits_b: Number of qubits for subsystem B.

    Returns:
        The negativity of the input density matrix.

    """
    da = 2**n_qubits_a
    db = 2**n_qubits_b
    rho = rho.reshape(da, db, da, db)
    # TODO: # pylint: disable=fixme
    #  This is a one-liner here, but generally it would be nicer to have
    #  a partial transpose function w.r.t. any subsystem in the utils file!
    rho_t = np.einsum("ijkl -> kjil", rho)
    rho_t = rho_t.reshape(2 ** (n_qubits_a + n_qubits_b), 2 ** (n_qubits_a + n_qubits_b))
    evals, _ = np.linalg.eig(rho_t)
    neg = np.sum([np.abs(i) - i for i in np.real(evals)]) / 2

    return neg
