"""Functions related to optimization on manifolds."""
# ruff: noqa: N802, N803, N806

from copy import deepcopy

from iqm.benchmarks.compressive_gst.mgst.additional_fns import randU
from iqm.benchmarks.compressive_gst.mgst.low_level_jit import ddM, dK_dMdM, objf, objf_gauge
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize


def eigy_expm(A: np.ndarray) -> np.ndarray:
    """Custom Matrix exponential using the eigendecomposition of numpy's linalg.

    Args:
        A: Matrix to be exponentiated

    Returns:
        Matrix exponential of A

    """
    vals, vects = np.linalg.eig(A)
    return np.einsum("...ik, ...k, ...kj -> ...ij", vects, np.exp(vals), np.linalg.inv(vects))


def tangent_proj(K: np.ndarray, Z: np.ndarray, d: int, rK: int) -> np.ndarray:
    """Projection onto the local tangent space.

    Args:
        K: Current position
        Z: Element of the ambient space to be projected onto the tangent space at K
        d: Number of matrices which are projected (i.e. number of gates in the gate set)
        rK: Kraus rank of the gate set

    Returns:
        Projection of Z onto the tangent space of the Stiefel manifold at position K

    Notes:
        The projection is done with respect to the canonical metrik.

    """
    pdim = K.shape[2]
    n = pdim * rK
    K = K.reshape(d, n, pdim)
    Z = Z.reshape(d, n, pdim)
    G = np.ascontiguousarray(np.zeros((d, n, pdim)).astype(np.complex128))
    for i in range(d):
        G[i] += Z[i] - (K[i] @ K[i].T.conj() @ Z[i] + K[i] @ Z[i].T.conj() @ K[i]) / 2
    return G


def update_K_geodesic(K: np.ndarray, H: np.ndarray, a: float) -> np.ndarray:
    """Compute a new point on the geodesic.

    Args:
        K: Current position
        H: Element of the tangent space at K and local direction of the geodesic
        a: Geodesic curve parameter

    Returns:
        New position given by K_new = K(a) with K(a) being a geodesic with K(0) = K, [d/dt K](0) = H

    """
    d = K.shape[0]
    rK = K.shape[1]
    pdim = K.shape[2]
    n = pdim * rK
    K = K.reshape(d, n, pdim)
    K_new = K.copy()
    AR_mat = np.zeros((2 * pdim, 2 * pdim)).astype(np.complex128)
    for i in range(d):
        Q, R = np.linalg.qr((np.eye(n) - K[i] @ K[i].T.conj()) @ H[i])
        AR_mat[:pdim, :pdim] = K[i].T.conj() @ H[i]
        AR_mat[pdim:, :pdim] = R
        AR_mat[:pdim, pdim:] = -R.T.conj()
        MN = eigy_expm(-a * AR_mat) @ np.eye(2 * pdim, pdim)
        K_new[i] = K[i] @ MN[:pdim, :] + Q @ MN[pdim:, :]
    return K_new.reshape(d, rK, pdim, pdim)


def lineobjf_isom_geodesic(
    a: float,
    H: np.ndarray,
    K: np.ndarray,
    E: np.ndarray,
    rho: np.ndarray,
    J: np.ndarray,
    y: np.ndarray,
    mle: bool,
) -> float:
    """Compute objective function at position on geodesic.

    Args:
        a: Geodesic curve parameter
        H: Element of the tangent space at K and local direction of the geodesic
        K: Current position
        E: Current POVM estimate
        rho: Current initial state estimate
        J: 2D array where each row contains the gate indices of a gate sequence
        y: 2D array of measurement outcomes for sequences in J; The columns contain the outcome probabilities for
            different povm elements
        mle: If True, the log-likelihood objective function is used, otherwise the least squares objective function
            is used

    Returns:
        Objective function value at new position along the geodesic

    """
    d = K.shape[0]
    pdim = K.shape[2]
    r = pdim**2
    K_test = update_K_geodesic(K, H, a)
    X_test = np.einsum("ijkl,ijnm -> iknlm", K_test, K_test.conj()).reshape((d, r, r))
    return objf(X_test, E, rho, J, y, mle=mle)


def lineobjf_gauge_geodesic(
    a: float,
    Delta: np.ndarray,
    U: np.ndarray,
    gates: list[np.ndarray],
    target_gates: list[np.ndarray],
    weights: list[float] | None,
) -> float:
    """Compute objective function at position on geodesic for gauge optimization Parameters.

    Args:
        a: Geodesic curve parameter
        Delta: Element of the tangent space at U and local direction of the geodesic
        U: Current gauge transformation matrix
        gates: Current gate, POVM, and state estimates [X, E, rho]
        target_gates: Target gates, POVM, and state [X_target, E_target, rho_target]
        weights: Weights for gates and SPAM

    Returns:
        Objective function value at new position along the geodesic

    """
    d = 1
    rK = 1
    pdim = U.shape[1]
    U_test = update_K_geodesic(U.reshape(d, rK, pdim, pdim), Delta, a)
    return objf_gauge(*gates, *target_gates, U_test.reshape(pdim, pdim), weights)


def update_A_geodesic(A: np.ndarray, H: np.ndarray, a: float) -> np.ndarray:
    """Compute a new point on the geodesic for the POVM parametrization.

    Args:
        A: Current position
        H: Element of the tangent space at A and local direction of the geodesic
        a: Geodesic curve parameter

    Returns:
        New position given by A_new = A(a) with A(a) being a geodesic with A(0) = A, [d/dt A](0) = H

    """
    n_povm = A.shape[0]
    pdim = A.shape[1]
    n = pdim * n_povm
    A = A.reshape(n, pdim)
    H = H.reshape(n, pdim)
    A_new = A.copy()
    AR_mat = np.zeros((2 * pdim, 2 * pdim)).astype(np.complex128)
    Q, R = np.linalg.qr((np.eye(n) - A @ A.T.conj()) @ H)
    AR_mat[:pdim, :pdim] = A.T.conj() @ H
    AR_mat[pdim:, :pdim] = R
    AR_mat[:pdim, pdim:] = -R.T.conj()
    MN = eigy_expm(-a * AR_mat) @ np.eye(2 * pdim, pdim)
    A_new = A @ MN[:pdim, :] + Q @ MN[pdim:, :]
    return A_new.reshape(n_povm, pdim, pdim)


def update_B_geodesic(B: np.ndarray, H: np.ndarray, a: float) -> np.ndarray:
    """Compute a new point on the geodesic for the initial state parametrization.

    Args:
        B: Current initial state parametrization
        H: Element of the tangent space at B and local direction of the geodesic
        a: Geodesic curve parameter

    Returns:
        New position given by B_new = B(a) with B(a) being a geodesic with B(0) = B, [d/dt B](t=0) = H

    """
    pdim = B.shape[0]
    n = pdim**2
    B = B.reshape(n)
    H = H.reshape(n)
    B_temp = B.copy()
    AR_mat = np.zeros((2, 2)).astype(np.complex128)
    R = np.linalg.norm((np.eye(n) - np.outer(B, B.T.conj())) @ H)
    Q = ((np.eye(n) - np.outer(B, B.T.conj())) @ H) / R
    AR_mat[0, 0] = B.T.conj() @ H
    AR_mat[1, 0] = R
    AR_mat[0, 1] = -R.T.conj()
    MN = eigy_expm(-a * AR_mat) @ np.array([1, 0])
    B_temp = B * MN[0] + Q * MN[1]
    return B_temp.reshape(pdim, pdim)


def lineobjf_A_geodesic(
    a: float,
    H: np.ndarray,
    X: np.ndarray,
    A: np.ndarray,
    rho: np.ndarray,
    J: np.ndarray,
    y: np.ndarray,
    mle: bool = False,
) -> float:
    """Compute objective function at position on geodesic for POVM parametrization.

    Args:
        a: Geodesic curve parameter
        H: Element of the tangent space at A and local direction of the geodesic
        X: Current gate estimate
        A: Current position
        rho: Current initial state estimate
        J: 2D array where each row contains the gate indices of a gate sequence
        y: 2D array of measurement outcomes for sequences in J; The columns contain the outcome probabilities for
            different povm elements
        mle: If True, the log-likelihood objective function is used, otherwise the least squares objective function
            is used

    Returns:
        Objective function value at new position along the geodesic

    """
    n_povm = A.shape[0]
    A_test = update_A_geodesic(A, H, a)
    E_test = np.array([(A_test[i].T.conj() @ A_test[i]).reshape(-1) for i in range(n_povm)])
    return objf(X, E_test, rho, J, y, mle=mle)


def lineobjf_B_geodesic(
    a: float,
    H: np.ndarray,
    X: np.ndarray,
    E: np.ndarray,
    B: np.ndarray,
    J: np.ndarray,
    y: np.ndarray,
    mle: bool = False,
) -> float:
    """Compute objective function at position on geodesic for the initial state parametrization.

    Args:
        a: Geodesic curve parameter
        H: Element of the tangent space at B and local direction of the geodesic
        X: Current gate estimate
        E: Current POVM estimate
        B: Current initial state parametrization
        J: 2D array where each row contains the gate indices of a gate sequence
        y: 2D array of measurement outcomes for sequences in J; The columns contain the outcome probabilities
            for different povm elements
        mle: If True, the log-likelihood objective function is used, otherwise the least squares
            objective function is used

    Returns:
        Objective function value at new position along the geodesic

    """
    B_test = update_B_geodesic(B, H, a)
    rho_test = (B_test @ B_test.T.conj()).reshape(-1)
    return objf(X, E, rho_test, J, y, mle=mle)


def lineobjf_A_B(
    a: float,
    v: np.ndarray,
    delta_v: np.ndarray,
    X: np.ndarray,
    C: np.ndarray,
    y: np.ndarray,
    J: np.ndarray,
    argument: str,
) -> float:
    """Compute objective function at translated position.

    Args:
        a: Step size
        v: Current position
        delta_v: Step direction
        X: Current gate estimate
        C: Current initial state/POVM estimate that is not updated
        y: 2D array of measurement outcomes for sequences in J; The columns contain the outcome probabilities
            for different povm elements
        J: 2D array where each row contains the gate indices of a gate sequence
        argument: Takes the following options: "E" or "rho", and indicates which gate set component is optimized over

    Returns:
        Objective function value at new position v + a*delta_v

    Notes:
        This function is used for the line search with linear updates v_new = v + a*delta_v,
        where v can be either the POVM estimate or the state estimate.

    """
    v_test = v - a * delta_v
    if argument == "rho":
        rho_test = (v_test @ v_test.T.conj()).reshape(-1)
        return objf(X, C, rho_test, J, y)
    if argument == "E":
        E_test = (v_test @ v_test.T.conj()).reshape(-1)
        return objf(X, E_test, C, J, y)
    raise ValueError("The <argument> variable takes either E or rho")


def Hess_evals(K: np.ndarray, E: np.ndarray, rho: np.ndarray, y: np.ndarray, J: np.ndarray) -> np.ndarray:
    """Compute eigenvalues of the Euclidean Hessian.

    Args:
        K: Current position
        E: Current POVM estimate
        rho: Current initial state estimate
        y: 2D array of measurement outcomes for sequences in J; The columns contain the outcome probabilities
            for different povm elements
        J: 2D array where each row contains the gate indices of a gate sequence

    Returns:
        Eigenvalues of the Euclidean Hessian for the Kraus operators at position (K,E,rho)

    """
    d = K.shape[0]
    rK = K.shape[1]
    pdim = K.shape[2]
    r = pdim**2
    n = d * rK * r
    H = np.zeros((2 * n, 2 * n)).astype(np.complex128)
    X = np.einsum("ijkl,ijnm -> iknlm", K, K.conj()).reshape((d, r, r))
    _, dM10, dM11 = dK_dMdM(X, K, E, rho, J, y)
    dd, dconjd = ddM(X, K, E, rho, J, y)

    A00 = dM11.reshape(n, n) + np.einsum("ijklmnop->ikmojlnp", dconjd).reshape(n, n)
    A10 = dM10.reshape(n, n) + np.einsum("ijklmnop->ikmojlnp", dd).reshape(n, n)
    A11 = A00.conj()
    A01 = A10.conj()

    H[:n, :n] = A00
    H[:n, n:] = A01
    H[n:, :n] = A10
    H[n:, n:] = A11

    evals, _ = eigh(H)
    return evals


def dU_gauge(
    X: np.ndarray,
    E: np.ndarray,
    rho: np.ndarray,
    X_target: np.ndarray,
    E_target: np.ndarray,
    rho_target: np.ndarray,
    U: np.ndarray,
    weights: list[float] | None,
) -> np.ndarray:
    """Compute the derivative of the gauge transformation objective function with respect to given unitary U.

    Args:
        X: Current gate estimates with shape (n_gates, r, r)
        E: Current POVM estimates with shape (n_povm, r)
        rho: Current initial state estimate with shape (r,)
        X_target: Target gate estimates with shape (n_gates, r, r)
        E_target: Target POVM estimates with shape (n_povm, r)
        rho_target: Target initial state estimate with shape (r,)
        U: Current gauge transformation matrix with shape (pdim, pdim)
        weights: Weights for gates and SPAM. If None or incorrect length, defaults to all ones.
            Expected length is n_gates + 1

    Returns:
        Derivative of the gauge objective function with respect to U, shape (pdim, pdim)

    """
    n_gates = X.shape[0]
    n_povm = E.shape[0]
    r = X.shape[1]
    pdim = int(np.sqrt(r))
    X_r = X.reshape((n_gates, pdim, pdim, pdim, pdim))
    X_t_r = X_target.reshape((n_gates, pdim, pdim, pdim, pdim))
    E_r = E.reshape((n_povm, pdim, pdim))
    E_target_r = E_target.reshape((n_povm, pdim, pdim))
    rho_r = rho.reshape((pdim, pdim))
    rho_target_r = rho_target.reshape((pdim, pdim))
    dU_ = np.zeros((pdim, pdim)).astype(np.complex128)

    # Create weights for gates and SPAM (state and measurement share the same weight)
    if weights is None or len(weights) != n_gates + 1:
        weights = [1] * (n_gates + 1)

    # Derivative of the gate terms
    for i in range(n_gates):
        dU_ += -weights[i] * np.einsum(
            "ij,kl,np,jlmn,ikop -> mo",
            U.T.conj(),
            U.T,
            U.conj(),
            X_r[i],
            X_t_r[i].conj(),
        )
        dU_ += -weights[i] * np.einsum("ij,mo,np,jlmn,ikop -> lk", U.T.conj(), U, U.conj(), X_r[i], X_t_r[i].conj())
        dU_ += -weights[i] * np.einsum(
            "ij,kl,np,mnjl,opik -> mo",
            U.T.conj(),
            U.T,
            U.conj(),
            X_r[i].conj(),
            X_t_r[i],
        )
        dU_ += -weights[i] * np.einsum("ij,mo,np,mnjl,opik -> lk", U.T.conj(), U, U.conj(), X_r[i].conj(), X_t_r[i])
    # Derivative of the POVM terms
    for i in range(n_povm):
        dU_ += -weights[-1] * np.einsum("kl,ik,jl -> ij", U.conj(), E_r[i].conj(), E_target_r[i])
        dU_ += -weights[-1] * np.einsum("ij,ik,jl -> lk", U.T.conj(), E_target_r[i].conj(), E_r[i])
    # Derivative of the state terms
    dU_ += -weights[-1] * np.einsum("ij,ik,jl -> lk", U.T.conj(), rho_target_r.conj(), rho_r)
    dU_ += -weights[-1] * np.einsum("kl,ik,jl -> ij", U.conj(), rho_r.conj(), rho_target_r)
    return dU_ / pdim**2


def gauge_opt(
    X: np.ndarray,
    E: np.ndarray,
    rho: np.ndarray,
    X_target: np.ndarray,
    E_target: np.ndarray,
    rho_target: np.ndarray,
    gauge_precision: float = 1e-6,
    weights: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optimize gauge transformation to align estimates with target values.

    Args:
        X: Current gate estimates with shape (n_gates, r, r)
        E: Current POVM estimates with shape (n_povm, r)
        rho: Current initial state estimate with shape (r,)
        X_target: Target gate estimates with shape (n_gates, r, r)
        E_target: Target POVM estimates with shape (n_povm, r)
        rho_target: Target initial state estimate with shape (r,)
        gauge_precision: Convergence threshold for objective function
        weights: Weights for gate, POVM, and state terms

    Returns:
        X_opt: Gauge-transformed gate estimates
        E_opt: Gauge-transformed POVM estimates
        rho_opt: Gauge-transformed initial state estimate

    """
    d = 1
    rK = 1
    r = X.shape[1]
    max_iter = 300
    pdim = int(np.sqrt(r))
    U = deepcopy(randU(pdim, a=0.1))
    for _ in range(max_iter):
        dU = dU_gauge(X, E, rho, X_target, E_target, rho_target, U, weights)
        # derivative
        # Riem. gradient taken from conjugate derivative
        rGrad = 2 * (dU.conj() - U @ dU.T @ U)
        Delta = rGrad.reshape((d, pdim, pdim))

        # Additional projection onto tangent space to avoid numerical instability
        Delta = tangent_proj(U.reshape(d, rK, pdim, pdim), Delta, d, rK)

        res = minimize(  # type: ignore[call-overload]
            lineobjf_gauge_geodesic,
            1e-8,
            args=(Delta, U, [X, E, rho], [X_target, E_target, rho_target], weights),
            options={"maxiter": 200},
            method="COBYLA",
        )
        a = res.x
        if a * np.linalg.norm(Delta) < gauge_precision:
            break
        U = update_K_geodesic(U.reshape(d, rK, pdim, pdim), Delta, a).reshape(pdim, pdim)

    U_channel = np.kron(U, U.conj())
    U_channel_dag = U_channel.T.conj()
    X_opt = np.array([U_channel_dag @ X[i] @ U_channel for i in range(X.shape[0])])
    E_opt = np.array([(E[i].conj() @ U_channel).conj() for i in range(E.shape[0])])
    rho_opt = U_channel_dag @ rho

    return X_opt, E_opt, rho_opt
