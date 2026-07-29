"""All functions compiled with numba, such as tensor contractions and derivatives."""
# ruff: noqa: N802, N803, N806

from numba import njit, prange  # type: ignore[import-untyped]
import numpy as np


@njit(cache=True)
def local_basis(x: int, b: int, length: int) -> np.ndarray:
    """Convert a base-10 integer to an integer in a specified base with a fixed number of digits.

    This function takes an integer `x` in base-10 and converts it to base `b`.
    The result is returned as an array of length `length` with leading zeros.

    Args:
        x: The input number in base-10 to be converted.
        b: The target base to convert the input number to.
        length: The number of output digits in the target base representation.

    Returns:
        A numpy array of integers representing the base-`b` digits of the converted number, with leading zeros if
        necessary. The length of the array is `length`.

    """
    r = np.zeros(length).astype(np.int32)
    k = 1
    while x > 0:
        r[-k] = x % b
        x //= b
        k += 1
    return r


@njit(cache=True)
def contract(X: np.ndarray, j_vec: np.ndarray) -> np.ndarray:
    """Contract a sequence of matrices in the given order.

    This function computes the product of a sequence of matrices specified by
    the indices in `j_vec`. The result is the contracted product of the matrices
    in the given order.

    Args:
        X: A 3D array containing the input matrices, of shape (n_matrices, n_rows, n_columns).
        j_vec: A 1D array of indices specifying the order in which to contract the matrices in X.

    Returns:
        The contracted product of the matrices specified by the indices in `j_vec`.

    """
    j_vec = j_vec[j_vec >= 0]
    res = np.eye(X[0].shape[0])
    res = res.astype(np.complex128)
    for j in j_vec:
        res = res.dot(X[j])
    return res


@njit(cache=True, fastmath=True)
def objf(
    X: np.ndarray,
    E: np.ndarray,
    rho: np.ndarray,
    J: np.ndarray,
    y: np.ndarray,
    mle: bool = False,
) -> float:
    """Calculate the objective function value for matrices, POVM elements, and target values.

    This function computes the objective function value based on input matrices X, POVM elements E, density matrix
    rho, and target values y.

    Args:
        X: A 3D array containing the input matrices, of shape (n_matrices, n_rows, n_columns).
        E: A 2D array representing the POVM elements, of shape (n_povm, r).
        rho: A 1D array representing the density matrix.
        J: A 2D array representing the indices for which the objective function will be evaluated.
        y: A 2D array of shape (n_povm, len(J)) containing the target values.
        mle: If True, the log-likelihood objective function is used, otherwise the least squares objective function
            is used

    Returns:
        The objective function value for the given set of matrices, POVM elements, and target values, normalized by m
        and n_povm.

    """
    m = len(J)
    n_povm = y.shape[0]
    objf_: float = 0
    for i in prange(m):
        j = J[i][J[i] >= 0]
        state = rho
        for ind in j[::-1]:
            state = X[ind] @ state
        for o in range(n_povm):
            if mle:
                objf_ -= np.log(abs(E[o].conj() @ state)) * y[o, i]
            else:
                objf_ += abs(E[o].conj() @ state - y[o, i]) ** 2 / m / n_povm
    return objf_


@njit(cache=True, fastmath=True)
def objf_gauge(
    X: np.ndarray,
    E: np.ndarray,
    rho: np.ndarray,
    X_target: np.ndarray,
    E_target: np.ndarray,
    rho_target: np.ndarray,
    U: np.ndarray,
    weights: list[float] | None = None,
) -> float:
    """Calculate the objective function with respect to a unitary gauge transformation.

    This function computes the weighted sum of squared Frobenius norm differences between
    the gauge-transformed gate set and the target gate set, including gates, POVM elements,
    and the initial state.

    Args:
        X: A 3D array containing the input matrices, of shape (n_gates, pdim**2, pdim**2).
        E: A 2D array representing the POVM elements, of shape (n_povm, pdim**2).
        rho: A 1D array representing the density matrix, of shape (pdim**2,).
        X_target: A 3D array containing the target gate matrices, of shape (n_gates, pdim**2, pdim**2).
        E_target: A 2D array representing the target POVM elements, of shape (n_povm, pdim**2).
        rho_target: A 1D array representing the target density matrix, of shape (pdim**2,).
        U: A 2D unitary matrix representing the gauge transformation, of shape (pdim, pdim).
        weights: A list of weights for the different components. Length must be n_gates + 2. If None or incorrect
            length, defaults to all ones.

    Returns:
        The normalized objective function value, scaled by pdim**2.

    """
    n_gates = X.shape[0]
    n_povm = E.shape[0]
    pdim = U.shape[0]
    U_channel = np.kron(U, U.conj())
    U_channel_dag = U_channel.T.conj()
    if weights is None or len(weights) != n_gates + 1:
        weights = [1.0] * (n_gates + 1)

    objf_: complex = 0
    for i in range(n_gates):
        objf_ += -2 * weights[i] * np.trace(U_channel_dag @ X[i] @ U_channel @ X_target[i].conj().T).real
        objf_ += weights[i] * np.trace(X[i] @ X[i].conj().T).real
        objf_ += weights[i] * np.trace(X_target[i] @ X_target[i].conj().T).real
    for i in range(n_povm):
        objf_ += -2 * weights[-1] * ((E[i].conj() @ U_channel).conj() @ E_target[i].conj()).real
        objf_ += weights[-1] * (E[i].conj() @ E[i]).real
        objf_ += weights[-1] * (E_target[i].conj() @ E_target[i]).real
    objf_ += -2 * weights[-1] * (rho_target.conj() @ U_channel_dag @ rho).real
    objf_ += weights[-1] * (rho.conj() @ rho).real
    objf_ += weights[-1] * (rho_target.conj() @ rho_target).real
    return np.real(objf_) / pdim**2


@njit(cache=True)
def MVE_lower(
    X_true: np.ndarray,
    E_true: np.ndarray,
    rho_true: np.ndarray,
    X: np.ndarray,
    E: np.ndarray,
    rho: np.ndarray,
    J: np.ndarray,
    n_povm: int,
) -> tuple[float, float]:
    """Compute the lower bound of the mean value error (MVE) between true and estimated parameters.

    This function calculates the lower bound of the MVE between the true parameters (X_true,
    E_true, rho_true) and the estimated parameters (X, E, rho) based on the provided J indices.

    Args:
        X_true: A 3D array containing true input matrices, of shape (n_matrices, n_rows, n_columns).
        E_true: A 2D array representing the true POVM elements, of shape (n_povm, r).
        rho_true: A 1D array representing the true density matrix.
        X: A 3D array containing estimated input matrices, of shape (n_matrices, n_rows, n_columns).
        E: A 2D array representing the estimated POVM elements, of shape (n_povm, r).
        rho: A 1D array representing the estimated density matrix.
        J: A 2D array representing indices for which the objective function will be evaluated.
        n_povm: The number of POVM elements.

    Returns:
        A tuple containing the lower bound of the mean value error and the maximum distance.

    """
    m = len(J)
    dist: float = 0
    max_dist: float = 0
    for i in range(m):
        j = J[i]
        C_t = contract(X_true, j)
        C = contract(X, j)
        curr: float = 0
        for k in range(n_povm):
            y_t = E_true[k].conj() @ C_t @ rho_true
            y = E[k].conj() @ C @ rho
            curr += np.abs(y_t - y)
        curr = curr / 2
        dist += curr
        max_dist = max(max_dist, curr)
    return dist / m, max_dist


@njit(cache=True)
def Mp_norm_lower(
    X_true: np.ndarray,
    E_true: np.ndarray,
    rho_true: np.ndarray,
    X: np.ndarray,
    E: np.ndarray,
    rho: np.ndarray,
    J: np.ndarray,
    p: float,
) -> tuple[float, float]:
    """Compute the Mp-norm lower bound of the distance between true and estimated parameters.

    This function calculates the lower bound of the Mp-norm between the true parameters (X_true,
    E_true, rho_true) and the estimated parameters (X, E, rho) based on the provided J indices.

    Args:
        X_true: A 3D array containing true input matrices, of shape (n_matrices, n_rows, n_columns).
        E_true: A 2D array representing the true POVM elements, of shape (n_povm, r).
        rho_true: A 1D array representing the true density matrix.
        X: A 3D array containing estimated input matrices, of shape (n_matrices, n_rows, n_columns).
        E: A 2D array representing the estimated POVM elements, of shape (n_povm, r).
        rho: A 1D array representing the estimated density matrix.
        J: A 2D array representing indices for which the objective function will be evaluated.
        p: The order of the Mp-norm (p > 0).

    Returns:
        A tuple containing the Mp-norm lower bound and the maximum distance.

    """
    m = len(J)
    n_povm = E.shape[0]
    dist: float = 0
    max_dist: float = 0
    curr: float = 0
    for i in range(m):
        j = J[i]
        C_t = contract(X_true, j)
        C = contract(X, j)
        for k in range(n_povm):
            y_t = E_true[k].conj() @ C_t @ rho_true
            y = E[k].conj() @ C @ rho
            dist += np.abs(y_t - y) ** p
        max_dist = max(max_dist, curr)
    return dist ** (1 / p) / m / n_povm, max_dist ** (1 / p)


@njit(cache=True)
def dK(
    X: np.ndarray,
    K: np.ndarray,
    E: np.ndarray,
    rho: np.ndarray,
    J: np.ndarray,
    y: np.ndarray,
    mle: bool = False,
) -> np.ndarray:
    """Compute the derivative of the objective function with respect to the Kraus tensor K.

    This function calculates the derivative of the Kraus operator K, based on the
    input matrices X, E, and rho, as well as the isometry condition.

    Args:
    X : The input matrix X, of shape (pdim, pdim).
    K : The Kraus operator K, reshaped to (d, rK, -1).
    E : A 2D array representing the POVM elements, of shape (n_povm, r).
    rho : A 1D array representing the density matrix.
    J : A 2D array representing the indices for which the derivatives will be computed.
    y : A 2D array of shape (n_povm, len(J)) containing the target values.
    mle : If True, the log-likelihood objective function is used, otherwise the least squares objective function is used

    Returns:
        The derivative objective function with respect to the Kraus tensor K,
        reshaped to (d, rK, pdim, pdim), and scaled by 2/m/n_povm.

    """
    d = K.shape[0]
    rK = K.shape[1]
    pdim = K.shape[2]
    r = pdim**2
    K = K.reshape(d, rK, -1)
    n_povm = y.shape[0]
    dK_ = np.zeros((d, rK, r))
    dK_ = np.ascontiguousarray(dK_.astype(np.complex128))
    m = len(J)

    for k in prange(d):
        for n in range(m):
            j = J[n][J[n] >= 0]
            for i, j_curr in enumerate(j):
                if j_curr == k:
                    R = rho.copy()
                    for ind in j[i + 1 :][::-1]:
                        R = X[ind] @ R
                    for o in range(n_povm):
                        L = E[o].conj()
                        for ind in j[:i]:
                            L = L @ X[ind]
                        if mle:
                            p_ind = L @ X[k] @ R
                            dK_[k] -= (
                                K[k].conj()
                                @ np.kron(L.reshape(pdim, pdim).T, R.reshape(pdim, pdim).T)
                                * y[o, n]
                                / p_ind
                            )
                        else:
                            D_ind = L @ X[k] @ R - y[o, n]
                            dK_[k] += (
                                D_ind
                                * K[k].conj()
                                @ np.kron(L.reshape(pdim, pdim).T, R.reshape(pdim, pdim).T)
                                * 2
                                / m
                                / n_povm
                            )
    return dK_.reshape(d, rK, pdim, pdim)


@njit(cache=True)
def dK_dMdM(
    X: np.ndarray,
    K: np.ndarray,
    E: np.ndarray,
    rho: np.ndarray,
    J: np.ndarray,
    y: np.ndarray,
    mle: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute first order derivative and products.

    Compute the derivatives of the objective function with respect to K and the
    product of derivatives of the measurement map with respect to K.

    Args:
        X: A 3D array containing input matrices, of shape (n_matrices, n_rows, n_columns).
        K: A 1D array representing the matrix K.
        E: A 2D array representing the POVM elements, of shape (n_povm, r).
        rho: A 1D array representing the density matrix.
        J: A 2D array representing indices for which the objective function will be evaluated.
        y: A 2D array of shape (n_povm, len(J)) containing target values.
        mle: If True, the log-likelihood objective function is used, otherwise the least squares
            objective function is used

    Returns:
        A tuple containing the derivatives of K, dM10, and dM11, each of which is a numpy.ndarray.

    """
    d = K.shape[0]
    rK = K.shape[1]
    pdim = K.shape[2]
    r = pdim**2
    K = K.reshape(d, rK, -1)
    n = d * rK * r
    n_povm = y.shape[0]
    dK_ = np.zeros((d, rK, r)).astype(np.complex128)
    dM11 = np.zeros(n**2).astype(np.complex128)
    dM10 = np.zeros(n**2).astype(np.complex128)
    m = len(J)
    for n in range(m):
        j = J[n][J[n] >= 0]
        dM = np.ascontiguousarray(np.zeros((n_povm, d, rK, r)).astype(np.complex128))
        p_ind_array = np.zeros(n_povm).astype(np.complex128)
        for o in range(n_povm):
            for i, k in enumerate(j):
                R = rho.copy()
                for ind in j[i + 1 :][::-1]:
                    R = X[ind] @ R
                L = E[o].conj().copy()
                for ind in j[:i]:
                    L = L @ X[ind]
                dM_loc = K[k].conj() @ np.kron(L.reshape((pdim, pdim)).T, R.reshape((pdim, pdim)).T)
                p_ind = L @ X[k] @ R
                if mle:
                    dM[o, k] += dM_loc
                    dK_[k] -= dM_loc * y[o, n] / p_ind
                else:
                    dM[o, k] += dM_loc
                    D_ind = p_ind - y[o, n]
                    dK_[k] += D_ind * dM_loc * 2 / m / n_povm
            if len(j) == 0:
                p_ind_array[o] = E[o].conj() @ rho
            else:
                p_ind_array[o] = p_ind
        for o in range(n_povm):
            if mle:
                dM11 += np.kron(dM[o].conj().reshape(-1), dM[o].reshape(-1)) * y[o, n] / p_ind_array[o] ** 2
                dM10 += np.kron(dM[o].reshape(-1), dM[o].reshape(-1)) * y[o, n] / p_ind_array[o] ** 2
            else:
                dM11 += np.kron(dM[o].conj().reshape(-1), dM[o].reshape(-1)) * 2 / m / n_povm
                dM10 += np.kron(dM[o].reshape(-1), dM[o].reshape(-1)) * 2 / m / n_povm
    return (dK_.reshape((d, rK, pdim, pdim)), dM10, dM11)


@njit(cache=True)
def ddM(  # noqa: PLR0912
    X: np.ndarray,
    K: np.ndarray,
    E: np.ndarray,
    rho: np.ndarray,
    J: np.ndarray,
    y: np.ndarray,
    mle: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the second derivative of the objective function with respect to the Kraus tensor K.

    This function calculates the second derivative of the objective function for a given
    set of input parameters.

    Args:
        X: Array of input matrices.
        K: Array of Kraus operators.
        E: Array of measurement operators.
        rho: Array of quantum states.
        J: Array of indices corresponding to the sequence of operations.
        y: Array of observed probabilities.
        mle: If True, the log-likelihood objective function is used, otherwise the least squares
            objective function is used

    Returns:
        ddK: Array of shape (d, d, rK, rK, pdim, pdim, pdim, pdim)
            Second derivative of the objective function with respect to matrix elements, reshaped
            for easier manipulation.
        dconjdK: Array of shape (d, d, rK, rK, pdim, pdim, pdim, pdim)
            Conjugate of the second derivative of the objective function with respect to matrix
            elements, reshaped for easier manipulation.

    """
    d = K.shape[0]
    rK = K.shape[1]
    pdim = K.shape[2]
    r = pdim**2
    n_povm = y.shape[0]
    ddK = np.zeros((d**2, rK**2, r, r))
    ddK = np.ascontiguousarray(ddK.astype(np.complex128))
    dconjdK = np.zeros((d**2, rK**2, r, r))
    dconjdK = np.ascontiguousarray(dconjdK.astype(np.complex128))
    m = len(J)
    for k in range(d**2):
        k1, k2 = local_basis(k, d, 2)
        for n in range(m):
            j = J[n][J[n] >= 0]
            for i1, j_1 in enumerate(j):
                if j_1 == k1:
                    for i2, j_2 in enumerate(j):
                        if j_2 == k2:
                            L0 = contract(X, j[: min(i1, i2)])
                            C = contract(X, j[min(i1, i2) + 1 : max(i1, i2)]).reshape(pdim, pdim, pdim, pdim)
                            R = contract(X, j[max(i1, i2) + 1 :]) @ rho
                            for o in range(n_povm):
                                L = E[o].conj() @ L0
                                if i1 == i2:
                                    p_ind = L @ X[k1] @ R
                                elif i1 < i2:
                                    p_ind = L @ X[k1] @ C.reshape(r, r) @ X[k2] @ R
                                else:
                                    p_ind = L @ X[k2] @ C.reshape(r, r) @ X[k1] @ R
                                D_ind = p_ind - y[o, n]

                                ddK_loc = np.zeros((rK**2, r, r)).astype(np.complex128)
                                dconjdK_loc = np.zeros((rK**2, r, r)).astype(np.complex128)
                                for rk1 in range(rK):
                                    for rk2 in range(rK):
                                        if i1 < i2:
                                            ddK_loc[rk1 * rK + rk2] = np.kron(
                                                L.reshape(pdim, pdim) @ K[k1, rk1].conj(),
                                                R.reshape(pdim, pdim) @ K[k2, rk2].T.conj(),
                                            ) @ np.ascontiguousarray(C.transpose(1, 3, 0, 2)).reshape(r, r)

                                            ddK_loc[rk1 * rK + rk2] = np.ascontiguousarray(
                                                ddK_loc[rk1 * rK + rk2]
                                                .reshape(pdim, pdim, pdim, pdim)
                                                .transpose(0, 3, 2, 1)
                                            ).reshape(r, r)

                                            dconjdK_loc[rk1 * rK + rk2] = np.kron(
                                                L.reshape(pdim, pdim) @ K[k1, rk1].conj(),
                                                R.reshape(pdim, pdim).T @ K[k2, rk2].T,
                                            ) @ np.ascontiguousarray(C.transpose(1, 2, 3, 0)).reshape(r, r)

                                            dconjdK_loc[rk1 * rK + rk2] = np.ascontiguousarray(
                                                dconjdK_loc[rk1 * rK + rk2]
                                                .reshape(pdim, pdim, pdim, pdim)
                                                .transpose(0, 2, 3, 1)
                                            ).reshape(r, r)

                                        elif i1 == i2:
                                            dconjdK_loc[rk1 * rK + rk2] = np.outer(L, R)

                                        elif i1 > i2:
                                            ddK_loc[rk1 * rK + rk2] = np.kron(
                                                L.reshape(pdim, pdim) @ K[k2, rk2].conj(),
                                                R.reshape(pdim, pdim) @ K[k1, rk1].T.conj(),
                                            ) @ np.ascontiguousarray(C.transpose(1, 3, 0, 2)).reshape(r, r)

                                            ddK_loc[rk1 * rK + rk2] = np.ascontiguousarray(
                                                ddK_loc[rk1 * rK + rk2]
                                                .reshape(pdim, pdim, pdim, pdim)
                                                .transpose(3, 0, 1, 2)
                                            ).reshape(r, r)

                                            dconjdK_loc[rk1 * rK + rk2] = np.kron(
                                                L.reshape(pdim, pdim).T @ K[k2, rk2],
                                                R.reshape(pdim, pdim) @ K[k1, rk1].T.conj(),
                                            ) @ np.ascontiguousarray(C.transpose((0, 3, 2, 1))).reshape(r, r)

                                            dconjdK_loc[rk1 * rK + rk2] = np.ascontiguousarray(
                                                dconjdK_loc[rk1 * rK + rk2]
                                                .reshape(pdim, pdim, pdim, pdim)
                                                .transpose(2, 0, 1, 3)
                                            ).reshape(r, r)
                                if mle:
                                    ddK[k1 * d + k2] -= ddK_loc * y[o, n] / p_ind
                                    dconjdK[k1 * d + k2] -= dconjdK_loc * y[o, n] / p_ind
                                else:
                                    ddK[k1 * d + k2] += D_ind * ddK_loc * 2 / m / n_povm
                                    dconjdK[k1 * d + k2] += D_ind * dconjdK_loc * 2 / m / n_povm
    return (
        ddK.reshape(d, d, rK, rK, pdim, pdim, pdim, pdim),
        dconjdK.reshape(d, d, rK, rK, pdim, pdim, pdim, pdim),
    )


@njit(cache=True)
def dA(
    X: np.ndarray, A: np.ndarray, B: np.ndarray, J: np.ndarray, y: np.ndarray, r: int, pdim: int, n_povm: int
) -> np.ndarray:
    """Compute the derivative of to the objective function with respect to the POVM tensor A.

    Args:
        X: Array of input matrices.
        A: Array of measurement operators.
        B: Array of quantum states.
        J: Array of indices corresponding to the sequence of operations.
        y: Array of observed probabilities.
        r: Number of elements in each measurement operator.
        pdim: Dimension of the density matrices.
        n_povm: Number of measurement operators.

    Returns:
        Derivative of the objective function with respect to A.

    """
    A = np.ascontiguousarray(A)
    B = np.ascontiguousarray(B)
    E = np.zeros((n_povm, r)).astype(np.complex128)
    for k in range(n_povm):
        E[k] = (A[k].T.conj() @ A[k]).reshape(-1)
    rho = (B @ B.T.conj()).reshape(-1)
    dA_ = np.zeros((n_povm, pdim, pdim)).astype(np.complex128)
    m = len(J)
    for n in prange(m):
        j = J[n][J[n] >= 0]
        inner_deriv = contract(X, j) @ rho
        dA_step = np.zeros((n_povm, pdim, pdim)).astype(np.complex128)
        for o in range(n_povm):
            D_ind = E[o].conj() @ inner_deriv - y[o, n]
            dA_step[o] += D_ind * A[o].conj() @ inner_deriv.reshape(pdim, pdim).T
        dA_ += dA_step
    return dA_ * 2 / m / n_povm


@njit(cache=True)
def dB(X: np.ndarray, A: np.ndarray, B: np.ndarray, J: np.ndarray, y: np.ndarray, pdim: int) -> np.ndarray:
    """Compute the derivative of the objective function with respect to the state tensor B.

    Args:
        X: Array of input matrices.
        A: Array of measurement operators.
        B: Array of quantum states.
        J: Array of indices corresponding to the sequence of operations.
        y: Array of observed probabilities.
        pdim: Dimension of the density matrices.

    Returns:
        Derivative of the objective function with respect to the state tensor B.

    """
    A = np.ascontiguousarray(A)
    B = np.ascontiguousarray(B)
    E = (A.T.conj() @ A).reshape(-1)
    rho = (B @ B.T.conj()).reshape(-1)
    dB_ = np.zeros((pdim, pdim))
    dB_ = dB_.astype(np.complex128)
    m = len(J)
    for n in prange(m):
        jE = J[n][J[n] >= 0][0]
        j = J[n][J[n] >= 0][1:]
        inner_deriv = E[jE].conj().dot(contract(X, j))
        D_ind = inner_deriv.dot(rho) - y[n]
        dB_ += D_ind * inner_deriv.reshape(pdim, pdim).conj() @ B
    return dB_


@njit(cache=True)
def ddA_derivs(
    X: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    J: np.ndarray,
    y: np.ndarray,
    n_povm: int,
    mle: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate all nonzero terms of the second derivatives with respect to the POVM tensor A.

    Args:
        X: The input matrix X, of shape (pdim, pdim).
        A: A 3D array of shape (n_povm, pdim, pdim) representing the POVM elements.
        B: A 2D array of shape (pdim, pdim) representing the isometry matrix.
        J: A 2D array representing the indices for which the derivatives will be computed.
        y: A 2D array of shape (n_povm, len(J)) containing the target values.
        n_povm: The number of POVM elements.
        mle: If True, the log-likelihood objective function is used, otherwise the least squares objective function
            is used

    Returns:
        A tuple containing the computed derivatives:
        - dA: The derivative w.r.t. A
        of shape (n_povm, pdim, pdim).
        - dMdM: The product of the measurement map derivatives dM and dM, of shape (n_povm, r, r).
        - dMconjdM: The product of the conjugate of dM and dM, of shape (n_povm, r, r).
        - dconjdA: The product of the conjugate of dA, of shape (n_povm, r, r).

    """
    pdim = A.shape[1]
    r = pdim**2
    A = np.ascontiguousarray(A)
    B = np.ascontiguousarray(B)
    E = np.zeros((n_povm, r)).astype(np.complex128)
    for k in range(n_povm):
        E[k] = (A[k].T.conj() @ A[k]).reshape(-1)
    rho = (B @ B.T.conj()).reshape(-1)
    dA_ = np.zeros((n_povm, pdim, pdim)).astype(np.complex128)
    dMdM = np.zeros((n_povm, r, r)).astype(np.complex128)
    dMconjdM = np.zeros((n_povm, r, r)).astype(np.complex128)
    dconjdA = np.zeros((n_povm, r, r)).astype(np.complex128)
    m = len(J)
    for n in prange(m):
        j = J[n][J[n] >= 0]
        R = contract(X, j) @ rho
        dA_step = np.zeros((n_povm, pdim, pdim)).astype(np.complex128)
        dMdM_step = np.zeros((n_povm, r, r)).astype(np.complex128)
        dMconjdM_step = np.zeros((n_povm, r, r)).astype(np.complex128)
        dconjdA_step = np.zeros((n_povm, r, r)).astype(np.complex128)
        for o in range(n_povm):
            dM = A[o].conj() @ R.reshape(pdim, pdim).T
            if mle:
                p_ind = E[o].conj() @ R
                dMdM_step[o] += np.outer(dM, dM) * y[o, n] / p_ind**2
                dMconjdM_step[o] += np.outer(dM.conj(), dM) * y[o, n] / p_ind**2
                dA_step[o] -= dM * y[o, n] / p_ind
                dconjdA_step[o] -= (
                    np.kron(np.eye(pdim).astype(np.complex128), R.reshape(pdim, pdim).T) * y[o, n] / p_ind
                )
            else:
                D_ind = E[o].conj() @ R - y[o, n]
                dMdM_step[o] += np.outer(dM, dM) * 2 / m / n_povm
                dMconjdM_step[o] += np.outer(dM.conj(), dM) * 2 / m / n_povm
                dA_step[o] += D_ind * dM * 2 / m / n_povm
                dconjdA_step[o] += (
                    D_ind * np.kron(np.eye(pdim).astype(np.complex128), R.reshape(pdim, pdim).T) * 2 / m / n_povm
                )
        dA_ += dA_step
        dMdM += dMdM_step
        dMconjdM += dMconjdM_step
        dconjdA += dconjdA_step
    return dA_, dMdM, dMconjdM, dconjdA


@njit(cache=True)
def ddB_derivs(
    X: np.ndarray, A: np.ndarray, B: np.ndarray, J: np.ndarray, y: np.ndarray, r: int, pdim: int, mle: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate all nonzero terms of the second derivative with respect to the state tensor B.

    Args:
        X: The input matrix X, of shape (pdim, pdim).
        A: A 3D array of shape (n_povm, pdim, pdim) representing the POVM elements.
        B: A 2D array of shape (pdim, pdim) representing the isometry matrix.
        J: A 2D array representing the indices for which the derivatives will be computed.
        y: A 2D array of shape (n_povm, len(J)) containing the target values.
        r: The rank of the problem.
        pdim: The dimension of the input matrices A and B.
        mle: If True, the log-likelihood objective function is used instead of the least squares objective function

    Returns:
        A tuple containing the computed derivatives:
        - dB: The derivative w.r.t. B, of shape (pdim, pdim).
        - dMdM: The product of the derivatives dM and dM, of shape (r, r).
        - dMconjdM: The product of the conjugate of dM and dM, of shape (r, r).
        - dconjdB: The mixed second derivative of by dB and dB*, of shape (r, r).

    """
    n_povm = A.shape[0]
    A = np.ascontiguousarray(A)
    B = np.ascontiguousarray(B)
    E = np.zeros((n_povm, r)).astype(np.complex128)
    for k in range(n_povm):
        E[k] = (A[k].T.conj() @ A[k]).reshape(-1)
    rho = (B @ B.T.conj()).reshape(-1)
    dB_ = np.zeros((pdim, pdim)).astype(np.complex128)
    dM = np.zeros((pdim, pdim)).astype(np.complex128)
    dMdM = np.zeros((r, r)).astype(np.complex128)
    dMconjdM = np.zeros((r, r)).astype(np.complex128)
    dconjdB = np.zeros((r, r)).astype(np.complex128)
    m = len(J)
    for n in prange(m):
        j = J[n][J[n] >= 0]
        C = contract(X, j)
        for o in range(n_povm):
            L = E[o].conj() @ C
            dM = L.reshape(pdim, pdim) @ B.conj()
            if mle:
                p_ind = L @ rho
                dMdM += np.outer(dM, dM) * y[o, n] / p_ind**2
                dMconjdM += np.outer(dM.conj(), dM) * y[o, n] / p_ind**2
                dB_ -= dM * y[o, n] / p_ind
                dconjdB -= np.kron(L.reshape(pdim, pdim), np.eye(pdim).astype(np.complex128)) * y[o, n] / p_ind
            else:
                D_ind = L @ rho - y[o, n]
                dMdM += np.outer(dM, dM) * 2 / m / n_povm
                dMconjdM += np.outer(dM.conj(), dM) * 2 / m / n_povm
                dB_ += D_ind * dM * 2 / m / n_povm
                dconjdB += D_ind * np.kron(L.reshape(pdim, pdim), np.eye(pdim).astype(np.complex128)) * 2 / m / n_povm
    return dB_, dMdM, dMconjdM, dconjdB.T
