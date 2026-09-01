"""Utility functions for mGST."""
# ruff: noqa: N806, N803

import numpy as np
from qiskit.quantum_info import Pauli, pauli_basis


def basis_transform(matrix: np.ndarray, in_basis: str = "std", out_basis: str = "pp") -> np.ndarray:
    """Transform a matrix or vector between standard and Pauli bases.

    Args:
        matrix: Input matrix (superoperator) or vector (vectorized density matrix) to transform.
            For vectors: shape (pdim^2,) For matrices: shape (pdim^2, pdim^2)
        in_basis: Input basis, either "std" (standard) or "pp" (Pauli).
        out_basis: Output basis, either "std" (standard) or "pp" (Pauli).

    Returns:
        Transformed matrix or vector in the output basis. Returns real part for transformations involving Pauli basis.

    Raises:
        ValueError: If in_basis or out_basis is not "std" or "pp".

    """
    if in_basis not in ["std", "pp"] or out_basis not in ["std", "pp"]:
        raise ValueError(
            "Input and output basis must be either <std> or <pp>, other basis transforms are currently not implemented."
        )
    pdim = int(np.sqrt(matrix.shape[0]))
    n_qubits = int(np.log(pdim) / np.log(2))
    # Define a unitary whose columns are vectorized tensor products of n_qubits local Paulis
    U = np.array([Pauli(pauli_string).to_matrix().reshape(-1) for pauli_string in pauli_basis(n_qubits)]).T / np.sqrt(
        2**n_qubits
    )
    if in_basis == "std" and out_basis == "pp":
        if len(matrix.shape) == 1:  # vector (i.e. vectoriuzed density matrix)
            return ((matrix.reshape(pdim, pdim).T.reshape(-1)) @ U).real
        else:  # matrix (i.e. superoperator)
            return (U.T.conj() @ matrix @ U).real
    if in_basis == "pp" and out_basis == "std":
        if len(matrix.shape) == 1:  # vector
            return (matrix.reshape(pdim, pdim).T.reshape(-1)) @ U.T.conj()
        else:
            return U @ matrix @ U.T.conj()
    else:
        return matrix


def average_gate_fidelities(X1: np.ndarray, X2: np.ndarray) -> list[float]:
    """Return the average gate fidelities between gates of two pygsti models.

    Args:
        X1: Gate set tensor of shape (Number of Gates, Kraus rank, dimension^2, dimension^2)
        X2: Gate set tensor of shape (Number of Gates, Kraus rank, dimension^2, dimension^2)

    Returns:
        Average gate fidelities between all gate pairs gates of the input models

    """
    ent_fids = []
    dim_sq = X1[0].shape[0]
    pdim = int(np.sqrt(dim_sq))
    tol = 1e-10

    for gate1, gate2 in zip(X1, X2, strict=True):
        if (
            np.linalg.norm(np.eye(pdim**2) - gate1 @ gate1.T.conj()) < tol
            or np.linalg.norm(np.eye(pdim**2) - gate2 @ gate2.T.conj()) < tol
        ):
            ent_fids.append((np.trace(gate1 @ gate2.T.conj())).real / pdim**2)
        else:
            raise ValueError(
                "Average gate fidelity is currently only implemented for the case where at least one gate is unitary, "
                "the provided gate sets don't satisfy this condition."
            )
    fidelities = (np.array(ent_fids) * pdim + 1) / (pdim + 1)
    return list(fidelities)


def std2pp(X: np.ndarray, E: np.ndarray, rho: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Basis change of an mGST model from the standard basis to the Pauli basis.

    Args:
        X: Gate set tensor of shape (Number of Gates, Kraus rank, dimension^2, dimension^2) in standard basis
        E: POVM matrix of shape (#POVM elements, dimension^2) in standard basis
        rho: Initial state vector of shape (dimension^2) in standard basis

    Returns:
        Xpp: Gate set tensor of shape (Number of Gates, Kraus rank, dimension^2, dimension^2) in Pauli basis
        Epp: POVM matrix of shape (#POVM elements, dimension^2) in Pauli basis
        rhopp: Initial state vector of shape (dimension^2) in Pauli basis

    """
    Xpp = np.array([np.array(basis_transform(X[i], "std", "pp")) for i in range(X.shape[0])])
    Epp = np.array([np.array(basis_transform(E[i], "std", "pp")) for i in range(E.shape[0])])
    return (
        Xpp.astype(np.complex128),
        Epp.astype(np.complex128),
        basis_transform(rho, "std", "pp").astype(np.complex128),
    )


def pp2std(Xpp: np.ndarray, Epp: np.ndarray, rhopp: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Basis change of an mGST model from the Pauli basis to the standard basis.

    Args:
        Xpp: Gate set tensor of shape (Number of Gates, Kraus rank, dimension^2, dimension^2) in Pauli basis
        Epp: POVM matrix of shape (#POVM elements, dimension^2) in Pauli basis
        rhopp: Initial state vector of shape (dimension^2) in Pauli basis

    Returns:
        X: Gate set tensor of shape (Number of Gates, Kraus rank, dimension^2, dimension^2) in standard basis
        E: POVM matrix of shape (#POVM elements, dimension^2) in standard basis
        rho: Initial state vector of shape (dimension^2) in standard basis

    """
    X = np.array([np.array(basis_transform(Xpp[i], "pp", "std")) for i in range(Xpp.shape[0])])
    E = np.array([np.array(basis_transform(Epp[i], "pp", "std")) for i in range(Epp.shape[0])])
    return (
        X.astype(np.complex128),
        E.astype(np.complex128),
        basis_transform(rhopp, "pp", "std").astype(np.complex128),
    )
