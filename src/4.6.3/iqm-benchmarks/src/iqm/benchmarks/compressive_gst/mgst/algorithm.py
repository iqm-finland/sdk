"""The main algorithm and functions that perform iteration steps."""
# ruff: noqa: N802, N803, N806

from decimal import Decimal
import time
from typing import Any
from warnings import warn

from iqm.benchmarks.compressive_gst.mgst.additional_fns import batch, random_gs, transp
from iqm.benchmarks.compressive_gst.mgst.low_level_jit import ddA_derivs, ddB_derivs, ddM, dK, dK_dMdM, objf
from iqm.benchmarks.compressive_gst.mgst.optimization import (
    lineobjf_A_geodesic,
    lineobjf_B_geodesic,
    lineobjf_isom_geodesic,
    tangent_proj,
    update_A_geodesic,
    update_B_geodesic,
    update_K_geodesic,
)
from iqm.benchmarks.compressive_gst.mgst.reporting.figure_gen import plot_objf
from iqm.benchmarks.logging_config import qcvv_logger
import numpy as np
import numpy.linalg as la
from scipy.linalg import eig, eigh
from scipy.optimize import minimize
from tqdm import trange
from tqdm.contrib.logging import logging_redirect_tqdm


def A_SFN_riem_Hess(
    K: np.ndarray, A: np.ndarray, B: np.ndarray, y: np.ndarray, J: np.ndarray, lam: float = 1e-3, mle: bool = False
) -> np.ndarray:
    """Riemannian saddle free Newton step on the POVM parametrization.

    Args:
        K: Each subarray along the first axis contains a set of Kraus operators. The second axis enumerates Kraus
            operators for a gate specified by the first axis.
        A: Current POVM parametrization
        B: Current initial state parametrization
        y: 2D array of measurement outcomes for sequences in J; Each column contains the outcome probabilities for a
            fixed sequence
        J: 2D array where each row contains the gate indices of a gate sequence
        lam: Damping parameter for dampled Newton method
        mle: Whether to use MLE objective function or least squares objective function

    Returns:
        A_new: Updated POVM parametrization

    """
    d, rK, pdim = K.shape[:3]
    r = pdim**2
    n_povm = A.shape[0]
    n = n_povm * pdim
    nt = n_povm * r
    rho = (B @ B.T.conj()).reshape(-1)
    H = np.zeros((2, nt, 2, nt)).astype(np.complex128)
    P_T = np.zeros((2, nt, 2, nt)).astype(np.complex128)
    Fyconjy = np.zeros((n_povm, r, n_povm, r)).astype(np.complex128)
    Fyy = np.zeros((n_povm, r, n_povm, r)).astype(np.complex128)

    X = np.einsum("ijkl,ijnm -> iknlm", K, K.conj()).reshape((d, r, r))
    dA_, dMdM, dMconjdM, dconjdA = ddA_derivs(X, A, B, J, y, n_povm, mle=mle)

    # Second derivatives
    for i in range(n_povm):
        Fyconjy[i, :, i, :] = dMconjdM[i] + dconjdA[i]
        Fyy[i, :, i, :] = dMdM[i]

    # derivative
    Fy = dA_.reshape(n, pdim)
    Y = A.reshape(n, pdim)
    rGrad = 2 * (Fy.conj() - Y @ Fy.T @ Y)
    G = np.array([rGrad, rGrad.conj()]).reshape(-1)

    P = np.eye(n) - Y @ Y.T.conj()
    T = transp(n, pdim)

    # Hessian assembly
    H00 = (
        -(np.kron(Y, Y.T)) @ T @ Fyy.reshape(nt, nt).T
        + Fyconjy.reshape(nt, nt).T.conj()
        - (np.kron(np.eye(n), Y.T @ Fy)) / 2
        - (np.kron(Y @ Fy.T, np.eye(pdim))) / 2
        - (np.kron(P, Fy.T.conj() @ Y.conj())) / 2
    )
    H01 = (
        Fyy.reshape(nt, nt).T.conj()
        - np.kron(Y, Y.T) @ T @ Fyconjy.reshape(nt, nt).T
        + (np.kron(Fy.conj(), Y.T) @ T) / 2
        + (np.kron(Y, Fy.T.conj()) @ T) / 2
    )

    H[0, :, 0, :] = H00
    H[0, :, 1, :] = H01
    H[1, :, 0, :] = H01.conj()
    H[1, :, 1, :] = H00.conj()

    P_T[0, :, 0, :] = np.eye(nt) - np.kron(Y @ Y.T.conj(), np.eye(pdim)) / 2
    P_T[0, :, 1, :] = -np.kron(Y, Y.T) @ T / 2
    P_T[1, :, 0, :] = P_T[0, :, 1, :].conj()
    P_T[1, :, 1, :] = P_T[0, :, 0, :].conj()

    H = H.reshape(2 * nt, 2 * nt) @ P_T.reshape(2 * nt, 2 * nt)

    # saddle free newton method
    H = (H + H.T.conj()) / 2
    evals, U = eigh(H)

    # Damping all eigenvalues
    H_abs_inv = U @ np.diag(1 / (np.abs(evals) + lam)) @ U.T.conj()

    Delta_A = ((H_abs_inv @ G)[:nt]).reshape(n, pdim)

    Delta = tangent_proj(A, Delta_A, 1, n_povm)[0]

    a = minimize(lineobjf_A_geodesic, 1e-9, args=(Delta, X, A, rho, J, y, mle), method="COBYLA").x  # type: ignore[call-overload]
    A_new = update_A_geodesic(A, Delta, a)
    return A_new


def B_SFN_riem_Hess(
    K: np.ndarray, A: np.ndarray, B: np.ndarray, y: np.ndarray, J: np.ndarray, lam: float = 1e-3, mle: bool = False
) -> np.ndarray:
    """Riemannian saddle free Newton step on the initial state parametrization.

    Args:
        K: Each subarray along the first axis contains a set of Kraus operators. The second axis enumerates Kraus
            operators for a gate specified by the first axis.
        A: Current POVM parametrization
        B: Current initial state parametrization
        y: 2D array of measurement outcomes for sequences in J; Each column contains the outcome probabilities for a
            fixed sequence
        J: 2D array where each row contains the gate indices of a gate sequence
        lam: Damping parameter for dampled Newton method
        mle: Whether to use MLE objective function or least squares objective function

    Returns:
        B_new: Updated initial state parametrization

    """
    d, rK, pdim = K.shape[:3]
    r = pdim**2
    n_povm = A.shape[0]
    n = r
    nt = r
    E = np.array([(A[i].T.conj() @ A[i]).reshape(-1) for i in range(n_povm)])
    H = np.zeros((2, nt, 2, nt)).astype(np.complex128)
    P_T = np.zeros((2, nt, 2, nt)).astype(np.complex128)

    X = np.einsum("ijkl,ijnm -> iknlm", K, K.conj()).reshape((d, r, r))
    dB_, dMdM, dMconjdM, dconjdB = ddB_derivs(X, A, B, J, y, r, pdim, mle=mle)

    # Second derivatives
    Fyconjy = dMconjdM + dconjdB
    Fyy = dMdM

    # derivative
    Fy = dB_.reshape(n)
    Y = B.reshape(n)
    rGrad = 2 * (Fy.conj() - Y * (Fy.T @ Y))
    G = np.array([rGrad, rGrad.conj()]).reshape(-1)

    P = np.eye(n) - np.outer(Y, Y.T.conj())

    # Hessian assembly
    H00 = (
        -(np.outer(Y, Y.T)) @ Fyy.reshape(nt, nt).T
        + Fyconjy.reshape(nt, nt).T.conj()
        - np.eye(n) * (Y.T @ Fy) / 2
        - np.outer(Y, Fy.T) / 2
        - P * (Fy.T.conj() @ Y.conj()) / 2
    )
    H01 = (
        Fyy.reshape(nt, nt).T.conj()
        - np.outer(Y, Y.T) @ Fyconjy.reshape(nt, nt).T
        + np.outer(Fy.conj(), Y.T) / 2
        + np.outer(Y, Fy.T.conj()) / 2
    )

    H[0, :, 0, :] = H00
    H[0, :, 1, :] = H01
    H[1, :, 0, :] = H01.conj()
    H[1, :, 1, :] = H00.conj()

    P_T[0, :, 0, :] = np.eye(nt) - np.outer(Y, Y.T.conj()) / 2
    P_T[0, :, 1, :] = -np.outer(Y, Y.T) / 2
    P_T[1, :, 0, :] = P_T[0, :, 1, :].conj()
    P_T[1, :, 1, :] = P_T[0, :, 0, :].conj()

    H = H.reshape(2 * nt, 2 * nt) @ P_T.reshape(2 * nt, 2 * nt)

    # saddle free newton method
    H = (H + H.T.conj()) / 2
    evals, U = eigh(H)

    # Damping all eigenvalues
    H_abs_inv = U @ np.diag(1 / (np.abs(evals) + lam)) @ U.T.conj()

    Delta = (H_abs_inv @ G)[:nt]
    # Projection onto tangent space
    Delta = Delta - Y * (Y.T.conj() @ Delta + Delta.T.conj() @ Y) / 2
    res = minimize(  # type: ignore[call-overload]
        lineobjf_B_geodesic,
        1e-9,
        args=(Delta, X, E, B, J, y, mle),
        method="COBYLA",
        options={"maxiter": 20},
    )
    a = res.x
    B_new = update_B_geodesic(B, Delta, a)
    return B_new


def gd(
    K: np.ndarray,
    E: np.ndarray,
    rho: np.ndarray,
    y: np.ndarray,
    J: np.ndarray,
    fixed_gates: np.ndarray | None = None,
    mle: bool = False,
) -> np.ndarray:
    """Do Riemannian gradient descent optimization step on gates.

    Args:
        K: Each subarray along the first axis contains a set of Kraus operators. The second axis enumerates Kraus
            operators for a gate specified by the first axis.
        E: Current POVM estimate
        rho: Current initial state estimate
        y: 2D array of measurement outcomes for sequences in J; Each column contains the outcome probabilities for a
            fixed sequence
        J: 2D array where each row contains the gate indices of a gate sequence
        fixed_gates: List of gate indices which are not optimized over and assumed as fixed
        mle: Whether to use MLE objective function or least squares objective function

    Returns:
        K_new: Updated Kraus parametrizations

    Notes:
        Gradient descent using the Riemannian gradient and updating along the geodesic. The step size is determined
        by minimizing the objective function in the step size parameter.

    """
    # setup
    d, rK, pdim = K.shape[:3]
    r = pdim**2
    n = rK * pdim
    Delta = np.zeros((d, n, pdim)).astype(np.complex128)
    X = np.einsum("ijkl,ijnm -> iknlm", K, K.conj()).reshape((d, r, r))

    dK_ = dK(X, K, E, rho, J, y, mle=mle)
    if fixed_gates is None:
        fixed_gates = np.zeros(d, dtype=bool)  # all gates are optimized over by default
    for k in np.where(~fixed_gates)[0]:
        # derivative
        Fy = dK_[k].reshape((n, pdim))
        Y = K[k].reshape(n, pdim)
        # Riem. gradient taken from conjugate derivative
        rGrad = 2 * (Fy.conj() - Y @ Fy.T @ Y)
        Delta[k] = rGrad

    # Additional projection onto tangent space to avoid numerical instability
    Delta = tangent_proj(K, Delta, d, rK)

    res = minimize(  # type: ignore[call-overload]
        lineobjf_isom_geodesic,
        1e-8,
        args=(Delta, K, E, rho, J, y, mle),
        options={"maxiter": 200},
    )
    a = res.x
    K_new = update_K_geodesic(K, Delta, a)

    return K_new


def SFN_riem_Hess(
    K: np.ndarray,
    E: np.ndarray,
    rho: np.ndarray,
    y: np.ndarray,
    J: np.ndarray,
    lam: float = 1e-3,
    ls: str = "COBYLA",
    fixed_gates: np.ndarray | None = None,
    mle: bool = False,
) -> np.ndarray:
    """Riemannian saddle free Newton step on each gate individually.

    Args:
        K: Each subarray along the first axis contains a set of Kraus operators. The second axis enumerates Kraus
            operators for a gate specified by the first axis.
        E: Current POVM estimate
        rho: Current initial state estimate
        y: 2D array of measurement outcomes for sequences in J; Each column contains the outcome probabilities for a
            fixed sequence
        J: 2D array where each row contains the gate indices of a gate sequence
        lam: Damping parameter for dampled Newton method
        ls: Line search method, takes "method" arguments of scipy.optimize.minimize
        fixed_gates: List of gate indices which are not optimized over and assumed as fixed
        mle: Whether to use MLE objective function or least squares objective function

    Returns:
        K_new: Updated Kraus parametrizations

    """
    # setup
    d, rK, pdim = K.shape[:3]
    r = pdim**2
    n = rK * pdim
    nt = rK * r
    H = np.zeros((2 * nt, 2 * nt)).astype(np.complex128)
    P_T = np.zeros((2 * nt, 2 * nt)).astype(np.complex128)
    Delta_K = np.zeros((d, rK, pdim, pdim)).astype(np.complex128)
    X = np.einsum("ijkl,ijnm -> iknlm", K, K.conj()).reshape((d, r, r))

    # compute derivatives
    dK_, dM10, dM11 = dK_dMdM(X, K, E, rho, J, y, mle=mle)
    dd, dconjd = ddM(X, K, E, rho, J, y, mle=mle)

    # Second derivatives
    Fyconjy = dM11.reshape(d, nt, d, nt) + np.einsum("ijklmnop->ikmojlnp", dconjd).reshape((d, nt, d, nt))
    Fyy = dM10.reshape(d, nt, d, nt) + np.einsum("ijklmnop->ikmojlnp", dd).reshape((d, nt, d, nt))

    if fixed_gates is None:
        fixed_gates = np.zeros(d, dtype=bool)  # all gates are optimized over by default
    for k in np.where(~fixed_gates)[0]:
        Fy = dK_[k].reshape(n, pdim)
        Y = K[k].reshape(n, pdim)
        # riemannian gradient, taken from conjugate derivative
        rGrad = 2 * (Fy.conj() - Y @ Fy.T @ Y)
        G = np.array([rGrad, rGrad.conj()]).reshape(-1)

        P = np.eye(n) - Y @ Y.T.conj()
        T = transp(n, pdim)

        # Riemannian Hessian with correction terms
        H00 = (
            -(np.kron(Y, Y.T)) @ T @ Fyy[k, :, k, :].T
            + Fyconjy[k, :, k, :].T.conj()
            - (np.kron(np.eye(n), Y.T @ Fy)) / 2
            - (np.kron(Y @ Fy.T, np.eye(pdim))) / 2
            - (np.kron(P, Fy.T.conj() @ Y.conj())) / 2
        )
        H01 = (
            Fyy[k, :, k, :].T.conj()
            - np.kron(Y, Y.T) @ T @ Fyconjy[k, :, k, :].T
            + (np.kron(Fy.conj(), Y.T) @ T) / 2
            + (np.kron(Y, Fy.T.conj()) @ T) / 2
        )

        H[:nt, :nt] = H00
        H[:nt, nt:] = H01
        H[nt:, :nt] = H[:nt, nt:].conj()
        H[nt:, nt:] = H[:nt, :nt].conj()

        # Tangent space projection
        P_T[:nt, :nt] = np.eye(nt) - np.kron(Y @ Y.T.conj(), np.eye(pdim)) / 2
        P_T[:nt, nt:] = -np.kron(Y, Y.T) @ T / 2
        P_T[nt:, :nt] = P_T[:nt, nt:].conj()
        P_T[nt:, nt:] = P_T[:nt, :nt].conj()

        H = H @ P_T

        # saddle free newton method
        evals, S = eig(H)

        H_abs_inv = S @ np.diag(1 / (np.abs(evals) + lam)) @ la.inv(S)
        Delta_K[k] = ((H_abs_inv @ G)[:nt]).reshape(rK, pdim, pdim)

    Delta = tangent_proj(K, Delta_K, d, rK)

    res = minimize(  # type: ignore[call-overload]
        lineobjf_isom_geodesic,
        1e-8,
        args=(Delta, K, E, rho, J, y, mle),
        method=ls,
        options={"maxiter": 200},
    )
    a = res.x
    K_new = update_K_geodesic(K, Delta, a)

    return K_new


def SFN_riem_Hess_full(
    K: np.ndarray,
    E: np.ndarray,
    rho: np.ndarray,
    y: np.ndarray,
    J: np.ndarray,
    lam: float = 1e-3,
    ls: str = "COBYLA",
    mle: bool = False,
) -> np.ndarray:
    """Riemannian saddle free Newton step on product manifold of all gates.

    Args:
        K: Each subarray along the first axis contains a set of Kraus operators. The second axis enumerates Kraus
            operators for a gate specified by the first axis.
        E: Current POVM estimate
        rho: Current initial state estimate
        y: 2D array of measurement outcomes for sequences in J; Each column contains the outcome probabilities for a
            fixed sequence
        J: 2D array where each row contains the gate indices of a gate sequence
        lam: Damping parameter for dampled Newton method
        ls: Line search method, takes "method" arguments of scipy.optimize.minimize
        mle: Whether to use MLE objective function or least squares objective function

    Returns:
        K_new: Updated Kraus parametrizations

    """
    d, rK, pdim = K.shape[:3]
    r = pdim**2
    n = rK * pdim
    nt = rK * r
    H = np.zeros((2, d, nt, 2, d, nt)).astype(np.complex128)
    P_T = np.zeros((2, d, nt, 2, d, nt)).astype(np.complex128)
    G = np.zeros((2, d, nt)).astype(np.complex128)
    X = np.einsum("ijkl,ijnm -> iknlm", K, K.conj()).reshape((d, r, r))

    # compute derivatives
    dK_, dM10, dM11 = dK_dMdM(X, K, E, rho, J, y, mle=mle)
    dd, dconjd = ddM(X, K, E, rho, J, y, mle=mle)

    # Second derivatives
    Fyconjy = dM11.reshape(d, nt, d, nt) + np.einsum("ijklmnop->ikmojlnp", dconjd).reshape((d, nt, d, nt))
    Fyy = dM10.reshape(d, nt, d, nt) + np.einsum("ijklmnop->ikmojlnp", dd).reshape((d, nt, d, nt))

    for k in range(d):
        Fy = dK_[k].reshape((n, pdim))
        Y = K[k].reshape((n, pdim))
        rGrad = 2 * (Fy.conj() - Y @ Fy.T @ Y)

        G[0, k, :] = rGrad.reshape(-1)
        G[1, k, :] = rGrad.conj().reshape(-1)

        P = np.eye(n) - Y @ Y.T.conj()
        T = transp(n, pdim)
        H00 = (
            -(np.kron(Y, Y.T)) @ T @ Fyy[k, :, k, :].T
            + Fyconjy[k, :, k, :].T.conj()
            - (np.kron(np.eye(n), Y.T @ Fy)) / 2
            - (np.kron(Y @ Fy.T, np.eye(pdim))) / 2
            - (np.kron(P, Fy.T.conj() @ Y.conj())) / 2
        )
        H01 = (
            Fyy[k, :, k, :].T.conj()
            - np.kron(Y, Y.T) @ T @ Fyconjy[k, :, k, :].T
            + (np.kron(Fy.conj(), Y.T) @ T) / 2
            + (np.kron(Y, Fy.T.conj()) @ T) / 2
        )

        # Riemannian Hessian with correction terms
        H[0, k, :, 0, k, :] = H00
        H[0, k, :, 1, k, :] = H01
        H[1, k, :, 0, k, :] = H01.conj()
        H[1, k, :, 1, k, :] = H00.conj()

        # Tangent space projection
        P_T[0, k, :, 0, k, :] = np.eye(nt) - np.kron(Y @ Y.T.conj(), np.eye(pdim)) / 2
        P_T[0, k, :, 1, k, :] = -np.kron(Y, Y.T) @ T / 2
        P_T[1, k, :, 0, k, :] = P_T[0, k, :, 1, k, :].conj()
        P_T[1, k, :, 1, k, :] = P_T[0, k, :, 0, k, :].conj()

        for k2 in range(d):
            if k2 != k:
                Yk2 = K[k2].reshape(n, pdim)
                H[0, k2, :, 0, k, :] = Fyconjy[k, :, k2, :].T.conj() - np.kron(Yk2, Yk2.T) @ T @ Fyy[k, :, k2, :].T
                H[0, k2, :, 1, k, :] = Fyy[k, :, k2, :].T.conj() - np.kron(Yk2, Yk2.T) @ T @ Fyconjy[k, :, k2, :].T
                H[1, k2, :, 0, k, :] = H[0, k2, :, 1, k, :].conj()
                H[1, k2, :, 1, k, :] = H[0, k2, :, 0, k, :].conj()

    H = H.reshape(2 * d * nt, -1) @ P_T.reshape((2 * d * nt, -1))

    # application of saddle free newton method
    H = (H + H.T.conj()) / 2
    evals, U = eigh(H)

    # Damping all eigenvalues
    H_abs_inv = U @ np.diag(1 / (np.abs(evals) + lam)) @ U.T.conj()
    Delta_K = ((H_abs_inv @ G.reshape(-1))[: d * nt]).reshape((d, rK, pdim, pdim))

    # Delta_K is already in tangent space but not to sufficient numerical accuracy
    Delta = tangent_proj(K, Delta_K, d, rK)
    res = minimize(  # type: ignore[call-overload]
        lineobjf_isom_geodesic,
        1e-8,
        args=(Delta, K, E, rho, J, y, mle),
        method=ls,
        options={"maxiter": 20},
    )
    a = res.x
    K_new = update_K_geodesic(K, Delta, a)
    return K_new


def optimize(
    y: np.ndarray,
    J: np.ndarray,
    method: str,
    K: np.ndarray,
    rho: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    fixed_elements: list[str],
    mle: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Full gate set optimization update alternating on E, K and rho.

    Args:
        y: 2D array of measurement outcomes for sequences in J; Each column contains the outcome probabilities for a
            fixed sequence
        J: 2D array where each row contains the gate indices of a gate sequence
        method: Optimization method
        K: Current estimates of Kraus operators
        rho: Current initial state estimate
        A: Current POVM parametrization
        B: Current initial state parametrization
        fixed_elements: List of element names to keep fixed during optimization (e.g., ["G1", "E", "rho"])
        mle: Whether to use maximum likelihood estimation or least squares estimation

    Returns:
        K_new: Updated estimates of Kraus operators
        X_new: Updated estimates of superoperatos corresponding to K_new
        E_new: Updated POVM estimate
        rho_new: Updated initial state estimate
        A_new: Updated POVM parametrization
        B_new: Updated initial state parametrization

    """
    d, rK, pdim = K.shape[:3]
    r = pdim**2
    n_povm = A.shape[0]
    if "E" in fixed_elements:
        A_new = A
        E_new = np.array([(A_new[i].T.conj() @ A_new[i]).reshape(-1) for i in range(n_povm)])
    else:
        A_new = A_SFN_riem_Hess(K, A, B, y, J, mle=mle)
        E_new = np.array([(A_new[i].T.conj() @ A_new[i]).reshape(-1) for i in range(n_povm)])
    if any((("G%i" % i in fixed_elements) for i in range(d))):
        fixed_gates = np.array([("G%i" % i in fixed_elements) for i in range(d)])
        if method == "SFN":
            K_new = SFN_riem_Hess(
                K,
                E_new,
                rho,
                y,
                J,
                lam=1e-3,
                ls="COBYLA",
                fixed_gates=fixed_gates,
                mle=mle,
            )
        else:
            K_new = gd(
                K,
                E_new,
                rho,
                y,
                J,
                fixed_gates=fixed_gates,
                mle=mle,
            )
    elif method == "SFN":
        K_new = SFN_riem_Hess_full(K, E_new, rho, y, J, lam=1e-3, ls="COBYLA", mle=mle)
    else:
        fixed_gates = np.array([("G%i" % i in fixed_elements) for i in range(d)])
        K_new = gd(
            K,
            E_new,
            rho,
            y,
            J,
            fixed_gates=fixed_gates,
            mle=mle,
        )
    if "rho" in fixed_elements:
        rho_new = rho
        B_new = B
    else:
        B_new = B_SFN_riem_Hess(K_new, A_new, B, y, J, lam=1e-3, mle=mle)
        rho_new = (B_new @ B_new.T.conj()).reshape(-1)
    X_new = np.einsum("ijkl,ijnm -> iknlm", K_new, K_new.conj()).reshape((d, r, r))
    return K_new, X_new, E_new, rho_new, A_new, B_new


def _initialize_from_provided(
    init: list[np.ndarray], d: int, r: int, n_povm: int, J: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """Initialize parameters from provided initial values."""
    pdim = int(np.sqrt(r))
    K, E = (init[0], init[1])
    # offset small negative eigenvalues for stability
    rho = init[2] + 1e-14 * np.eye(pdim).reshape(-1)
    A = np.array([la.cholesky(E[k].reshape(pdim, pdim) + 1e-14 * np.eye(pdim)).T.conj() for k in range(n_povm)])
    B = la.cholesky(rho.reshape(pdim, pdim))
    X = np.einsum("ijkl,ijnm -> iknlm", K, K.conj()).reshape((d, r, r))
    res_list = [objf(X, E, rho, J, y)]
    return K, X, E, rho, A, B, res_list


def _run_batch_optimization(
    y: np.ndarray,
    J: np.ndarray,
    dimensions: list[int],
    bsize: int,
    method: str,
    max_inits: int,
    max_iter: int,
    delta: float,
    fixed_elements: list[str],
    verbose_level: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[float], bool]:
    """Run batch optimization with random initializations."""
    max_verbose_level = 2
    d, r, rK, n_povm = dimensions
    pdim = int(np.sqrt(r))
    success = False

    for i in range(max_inits):
        K, X, E, rho = random_gs(d, r, rK, n_povm)
        A = np.array([la.cholesky(E[k].reshape(pdim, pdim) + 1e-14 * np.eye(pdim)).T.conj() for k in range(n_povm)])
        B = la.cholesky(rho.reshape(pdim, pdim))
        res_list = [objf(X, E, rho, J, y)]

        with logging_redirect_tqdm(loggers=[qcvv_logger] if verbose_level > 0 else None):
            for _ in trange(max_iter, disable=verbose_level == 0):
                yb, Jb = batch(y, J, bsize)
                K, X, E, rho, A, B = optimize(yb, Jb, method, K, rho, A, B, fixed_elements)
                res_list.append(objf(X, E, rho, J, y))
                if res_list[-1] < delta:
                    qcvv_logger.info("Batch optimization successful, improving estimate over full data....")
                    success = True
                    break

        if verbose_level == max_verbose_level:
            plot_objf(res_list, "Objective function for batch optimization", delta=delta)
        if success:
            break
        if verbose_level > 0:
            qcvv_logger.info(f"Run {i + 1}/{max_inits} failed, trying new initialization...")

    return K, X, E, rho, A, B, res_list, success


def _run_full_optimization(
    y: np.ndarray,
    J: np.ndarray,
    method: str,
    K: np.ndarray,
    rho: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    fixed_elements: list[str],
    final_iter: int,
    target_rel_prec: float,
    verbose_level: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """Run optimization on full dataset with least squares."""
    with logging_redirect_tqdm(loggers=[qcvv_logger] if verbose_level > 0 else None):
        res_list = []
        for _ in trange(final_iter, disable=verbose_level == 0):
            K, X, E, rho, A, B = optimize(y, J, method, K, rho, A, B, fixed_elements, mle=False)
            res_list.append(objf(X, E, rho, J, y))
            if len(res_list) >= 2 and np.abs(res_list[-2] - res_list[-1]) < res_list[-1] * target_rel_prec:
                break
    return K, X, E, rho, A, B, res_list


def _run_mle_optimization(
    y: np.ndarray,
    J: np.ndarray,
    method: str,
    K: np.ndarray,
    rho: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    fixed_elements: list[str],
    final_iter: int,
    target_rel_prec: float,
    verbose_level: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[float], list[float]]:
    """Run MLE optimization on full dataset."""
    qcvv_logger.info("Improving estimate with MLE...")
    with logging_redirect_tqdm(loggers=[qcvv_logger] if verbose_level > 0 else None):
        res_list = []
        res_list_mle = []
        for _ in trange(final_iter, disable=verbose_level == 0):
            # Currently only GD is supported for MLE
            K, X, E, rho, A, B = optimize(y, J, "GD", K, rho, A, B, fixed_elements, mle=True)
            res_list.append(objf(X, E, rho, J, y))
            res_list_mle.append(objf(X, E, rho, J, y, mle=True))
            if (
                len(res_list_mle) >= 2
                and np.abs(res_list_mle[-2] - res_list_mle[-1]) < res_list_mle[-1] * target_rel_prec
            ):
                break
    return K, X, E, rho, A, B, res_list, res_list_mle


def _log_final_results(
    success: bool, delta: float, res_list: list[float], res_list_mle: list[float], t0: float, verbose_level: int
) -> None:
    """Log final optimization results."""
    # ruff: noqa: PLR2004
    max_verbose_level = 2

    if verbose_level == max_verbose_level:
        plot_objf(res_list, "Least squares error over batches and full data", delta=delta)
        plot_objf(res_list_mle, "Negative log-likelihood over full data")

    if verbose_level > 0:
        if success or (res_list[-1] < delta):
            qcvv_logger.info("Convergence criterion satisfied")
        else:
            qcvv_logger.warning(
                "Convergence criterion not satisfied. Potential causes include too low max_iterations, "
                "bad initialization or model mismatch."
            )
        qcvv_logger.info(
            f"Final objective {Decimal(res_list[-1]):.2e} in time {(time.monotonic() - t0):.2f}s",
        )


def run_mGST(
    args: tuple[np.ndarray, np.ndarray, None, int, int, int, int, int, int],
    method: str = "SFN",
    max_inits: int = 10,
    max_iter: int = 200,
    final_iter: int = 120,
    target_rel_prec: float = 1e-5,
    threshold_multiplier: float = 5,
    fixed_elements: list | None = None,
    init: list[np.ndarray[Any, Any]] | None = None,
    verbose_level: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """Main mGST routine.

    Args:
        args: Tuple containing (y, J, _, d, r, rK, n_povm, bsize, meas_samples) where:
            - y: 2D array of measurement outcomes for sequences in J
            - J: 2D array where each row contains gate indices of a gate sequence
            - _: unused parameter
            - d: Number of different gates in the gate set
            - r: Superoperator dimension (square of physical dimension)
            - rK: Target Kraus rank
            - n_povm: Number of POVM elements
            - bsize: Size of the batch (number of sequences)
            - meas_samples: Number of samples per gate sequence for measurement array y
        method: Optimization method: "SFN" or "GD"
        max_inits: Maximum number of random initializations to try
        max_iter: Maximum number of iterations on batches
        final_iter: Maximum number of iterations on full data set
        target_rel_prec: Target relative precision at which the optimization loop breaks
        threshold_multiplier: Multiplier for the stopping criterion threshold
        fixed_elements: List of element names to keep fixed during optimization (e.g., ["E", "rho", "G0"])
        init: List of 3 numpy arrays [K, E, rho] for initialization; If not provided, random initialization is used
        verbose_level: Verbosity level: 0 (silent), 1 (info), 2 (info + plots)

    Returns:
        K: Updated estimates of Kraus operators
        X: Updated estimates of superoperators corresponding to K
        E: Updated POVM estimate
        rho: Updated initial state estimate
        res_list: Collected objective function values after each iteration

    """
    # ruff: noqa: PLR0913
    y, J, _, d, r, rK, n_povm, bsize, meas_samples = args
    t0 = time.monotonic()
    # stopping criterion (Factor 3 can be increased if model mismatch is high)
    delta = threshold_multiplier * (1 - y.reshape(-1)) @ y.reshape(-1) / len(J) / n_povm / meas_samples

    if not fixed_elements:
        fixed_elements = []

    if any((("G%i" % i in fixed_elements) for i in range(d))) and method == "SFN":
        warn(
            "The SFN method with fixed gates is currently only implemented via \n"
            "iterative updates over individual gates and might lead to a slower converges \n"
            "compared to the default SFN method.",
            stacklevel=2,
        )

    if verbose_level > 0:
        qcvv_logger.info("Starting mGST optimization...")

    # Initialization phase
    if init:
        K, X, E, rho, A, B, res_list_init = _initialize_from_provided(init, d, r, n_povm, J, y)
        success = False
    else:
        K, X, E, rho, A, B, res_list_init, success = _run_batch_optimization(
            y, J, [d, r, rK, n_povm], bsize, method, max_inits, max_iter, delta, fixed_elements, verbose_level
        )

    # Full dataset optimization
    if not success and init is None and verbose_level > 0:
        qcvv_logger.info("Success threshold not reached, attempting optimization over full data set...")

    K, X, E, rho, A, B, res_list_full = _run_full_optimization(
        y, J, method, K, rho, A, B, fixed_elements, final_iter, target_rel_prec, verbose_level
    )

    # MLE optimization
    K, X, E, rho, A, B, res_list_mle_ls, res_list_mle = _run_mle_optimization(
        y, J, method, K, rho, A, B, fixed_elements, final_iter, target_rel_prec, verbose_level
    )

    # Combine all results
    res_list = res_list_init + res_list_full + res_list_mle_ls

    # Logging
    _log_final_results(success, delta, res_list, res_list_mle, t0, verbose_level)

    return K, X, E, rho, res_list
