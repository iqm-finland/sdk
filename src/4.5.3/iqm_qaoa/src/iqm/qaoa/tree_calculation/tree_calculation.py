# Copyright (c) 2024-2025 IQM Quantum Computers
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted (subject to the
# limitations in the disclaimer below) provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this list of conditions and the following
#   disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
#   disclaimer in the documentation and/or other materials provided with the distribution.
# * Neither the name of IQM Quantum Computers nor the names of its contributors may be used to endorse or promote
#   products derived from this software without specific prior written permission.
#
# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY
# THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""The main module for the tree angle calculation."""

from iqm.qaoa.tree_calculation.generate_basis import get_z_basis_m, get_z_basis_m_t
from numba import jit  # type: ignore[attr-defined]
import numpy as np
from numpy.typing import NDArray


@jit(nopython=True)
def pair(ai: int, aj: int, beta_i: np.float64) -> np.complex128:
    r"""Represents the equation (A12) in :cite:`Wybo_2024`.

    Returns the expectation value of :math:`\langle z^{[m]}_k | e^{i\beta_i X_k} | z^{[n]}_k \rangle}` where the ket
    and the bra are basis states and the operator is the exponentiation of the mixer.

    Args:
        ai: The basis state of the bra vector, must be either +1 or -1.
        aj: The basis state of the ket vector, must be either +1 or -1.
        beta_i: The angle by which the :math:`X_k` operator gets exponentiated.

    Returns:
        The right-hand side of the equation (A12). In other words, the cosine of `beta_i` if `aj` and `ai` are equal
        and the imaginary sine of `beta_i` otherwise.

    Raises:
        ValueError: If the input variables `ai` or `aj` aren't either +1 or -1.

    """
    if (ai not in {-1, 1}) or (aj not in {-1, 1}):
        raise ValueError(
            f"The variables representing the bra and ket vectors (`ai` and `aj`) have to be either +1 or"
            f" -1. They are `ai = {ai}` and `aj = {aj}`."
        )

    if ai == aj:
        res = np.cos(beta_i)
    else:
        res = 1j * np.sin(beta_i)
    return res


# This function is not used anywhere.
@jit(nopython=True)
def get_f(p: int, a: NDArray[np.int8], big_beta: NDArray[np.float64]) -> np.complex128:
    r"""Represents the function :math:`f(\mathbb{a})` defined in (9) and (10) in :cite:`Wybo_2024`.

    For a given basis state :math:`\mathbb{a}` (i.e., an array of +1 and -1), this calculates the function :math:`f`.
    The function is one half of a product of expectation values of the form
    :math:`\langle a_k | e^{i\beta_k X} | a_{k+1} \rangle}` for variable :math:`k`.

    Args:
        p: The number of QAOA layers.
        a: The array of basis states of the ket and bra vector which appear in the definition of :math`f(\mathbb{a})`.
        big_beta: The array containing the beta angles of the QAOA, sorted ascendingly and descendingly in an array of
            length `2*p` as :math:`(\beta_1, \beta_2, ... , \beta_p, -\beta_p, -\beta_{p-1}, ..., -\beta_2, -\beta_1)`.

    Returns:
        The value of the :math:`f(\mathbb{a})` function.

    Raises:
        ValueError: If `a` is not a 1D array.
        ValueError: If `a` does not have length `2*p+1`.
        ValueError: If `big_beta` is not a 1D array.
        ValueError: If `big_beta` does not have length `2*p`.
        ValueError: If `a` contains other entries than +1 or -1.
        ValueError: If `big_beta` is not antisymmetric.

    """
    if a.ndim != 1:
        raise ValueError(f"The input array `a` must be 1D numpy array. It has {a.ndim} dimensions.")
    if a.shape[0] != 2 * p + 1:
        raise ValueError(f"The input `a` must have length `2*p+1`, i.e., {2 * p + 1}. It has length {a.shape[0]}.")
    if big_beta.ndim != 1:
        raise ValueError(f"The input array `big_beta` must be 1D numpy array. It has {big_beta.ndim} dimensions.")
    if big_beta.shape[0] != 2 * p:
        raise ValueError(
            f"The input `big_beta` must have length `2*p`, i.e., {2 * p}. It has length {big_beta.shape[0]}."
        )
    if not np.all(np.isin(a, [-1, 1])):
        raise ValueError("The input array `a` must only contain values +1 or -1.")
    if not np.all(big_beta == -big_beta[::-1]):
        raise ValueError("The `big_beta` array is not antisymmetric")

    phases = np.array([pair(a[i], a[i + 1], big_beta[i]) for i in range(2 * p)])
    return np.prod(phases) * 0.5


@jit(nopython=True)
def small_g(m_minus_1: int, b: NDArray[np.int8], big_beta: NDArray[np.float64]) -> np.complex128:
    r"""Represents the function :math:`g(b^{(m-1)})` defined in (A26) in :cite:`Wybo_2024`.

    For a given basis state `b` (i.e., an array of +1 and -1), this calculates the function :math:`g`. The function is
    a product of expectation values.

    Args:
        m_minus_1: The parameter :math:`m-1` from the text.
        b: The array of basis states of the ket and bra vector which appear in the definition of :math`g(b^{(m-1)})`.
            It is expected to have length `2*m_minus_1`, but it can be longer because it's accessed from left and right.
        big_beta: The array containing the beta angles of the QAOA, sorted ascendingly and descendingly in an array like
            :math:`(\beta_1, \beta_2, ... , \beta_{p-1}, -\beta_{p-1}, -\beta_{p-2}, ..., -\beta_2, -\beta_1)`. Here
            :math:`p \geq m-1` (but not necessarily equal).

    Returns:
        The value of the :math`g(b^{(m-1)})` function.

    """
    if m_minus_1 <= 1:
        return np.complex128(1.0)

    phases = np.array(
        # In the first half of (A26), b's go from b_1 to b_{m-1} (in Python notation that's from 0 to m-2)
        # At the same time, betas go from beta_1 to beta_{m-2} (in Python that's from 0 to m-3)
        [
            pair(b[i], b[i + 1], big_beta[i]) for i in range(m_minus_1 - 1)
        ]  # Note that range(m_minus_1 - 1) iterates from 0 to m-3
        # In the 2nd half of (A26), b's go from b_-{m-1} to b_-1 (in Python notation that's from -(m-1) to -1)
        # At the same time, betas go from -beta_{m-2} to -beta_1 (in Python that's from -(m-2) to -1)
        + [
            pair(b[neg_i - 1], b[neg_i], big_beta[neg_i]) for neg_i in range(-(m_minus_1 - 1), 0)
        ]  # Note that range(-(m_minus_1 - 1), 0) iterates from -(m-2) to -1
    )
    return np.prod(phases)


@jit(nopython=True)
def prune_angles(
    layer: int, big_gamma: NDArray[np.float64], big_beta: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""The function to take the full array of Gamma / Beta angles and shorten them to those relevant for layer `layer`.

    The arrays of Gamma / Beta angles are assumed to have the form described in equation (A8) in :cite:`Wybo_2024`,
    that is :math:`\Gamma = (\gamma_1, ..., \gamma_p, 0, -\gamma_p, ..., -\gamma_1)`, where `p` is the total number of
    QAOA layers. This is pruned to only include angles relevant for QAOA layer `layer` and to remove the central 0. Note
    that only `layer-1` Beta angles are relevant for layer `layer`.

    Args:
        layer: Specifying up to which layer should the arrays of angles be trimmed.
        big_gamma: An array of the gamma QAOA angles (sorted ascedingly and descendingly in an array).
        big_beta: An array of the beta QAOA angles (sorted ascedingly and descendingly in an array).

    Returns:
        A tuple containing the pruned arrays of angles `big_gamma` and `big_beta`.

    Raises:
        ValueError: If the input `layer` is more than the number of QAOA layers.
        ValueError: If the input `layer` is less than 1.

    """
    p = (len(big_gamma) - 1) // 2  # This is the number of QAOA layers.
    if layer > p:
        raise ValueError(
            f"The `layer` must not be greater than the original number of QAOA layers, i.e., {p}. "
            f"The provided `layer` is {layer}."
        )
    if layer < 1:
        raise ValueError(f"The `layer` must be a positive integer. The provided `layer` is {layer}.")

    # Note the difference in lengths between ``big_gamma_pruned`` and ``big_beta_pruned``.
    big_gamma_pruned = np.concatenate((big_gamma[:layer], big_gamma[-layer:]))

    if layer == 1:
        big_beta_pruned = np.zeros(0, dtype=np.float64)  # Empty array
    else:
        big_beta_pruned = np.concatenate((big_beta[: layer - 1], big_beta[-(layer - 1) :]))

    return big_gamma_pruned, big_beta_pruned


@jit(nopython=True, fastmath=True)
def big_g(a_t: int, beta_t: NDArray[np.float64]) -> np.complex128:
    r"""Calculates the function :math:`G(a_t)` from equations (A45) and (A46) from :cite:`Wybo_2024`.

    Args:
        a_t: The variable :math:`a_t`.
        beta_t: An array containing the QAOA beta angles from :math:`\beta_t` to :math:`\beta_p`, where :math:`p` is
            the number of QAOA layers.

    Returns:
        The result of the calculation.

    """
    factors = np.sin(2 * beta_t[0])
    coefs = np.cos(2 * beta_t[1:])
    factors *= np.prod(coefs)
    return -factors * 0.5j * a_t


@jit(nopython=True, fastmath=True)
def big_g_tilde(a_t: int, beta_t: NDArray[np.float64]) -> np.complex128:
    r"""Calculates the function :math:`\tilde{G}(a_t)` from equations (A47) and (A48) from :cite:`Wybo_2024`.

    Args:
        a_t: The variable :math:`a_t`.
        beta_t: An array containing the QAOA beta angles from :math:`\beta_t` to :math:`\beta_p`, where :math:`p` is
            the number of QAOA layers.

    Returns:
        The result of the calculation.

    """
    coefs = np.cos(2 * beta_t)
    factors = np.prod(coefs)
    return np.complex128(factors * a_t * 0.5)  # This is actually a real number, but we need to respect the output type.


@jit(nopython=True, fastmath=True)
def a24_a25(
    a_m: int, a_minus_m: int, b_m_minus_1: int, b_1_minus_m: int, beta_m_1: np.float64, gamma_m: np.float64, big_d: int
) -> np.complex128:
    r"""Calculates the term surrounded in the big parentheses in equations A24 and A25.

    Args:
        a_m: The variable :math:`a_{m}`.
        a_minus_m: The variable :math:`a_{-m}`.
        b_m_minus_1: The :math:`b_{m-1}` variable.
        b_1_minus_m: The :math:`b_{-m+1}` variable.
        beta_m_1: The beta angle :math:`\beta_{m-1}`.
        gamma_m: The gamma angle :math:`\gamma_m`.
        big_d: The regularity of the problem graph minus one.

    """
    a24 = (
        pair(b_m_minus_1, 1, beta_m_1)
        * pair(1, b_1_minus_m, -beta_m_1)
        * np.exp(1j * gamma_m / np.sqrt(big_d) * (a_m - a_minus_m))
    )
    a25 = (
        pair(b_m_minus_1, -1, beta_m_1)
        * pair(-1, b_1_minus_m, -beta_m_1)
        * np.exp(-1j * gamma_m / np.sqrt(big_d) * (a_m - a_minus_m))
    )
    return a24 + a25


@jit(nopython=True, fastmath=True)
def exp_gamma(
    big_d: int, h: float, big_gamma_pruned: NDArray[np.float64], a: NDArray[np.float64], b: NDArray[np.float64]
) -> np.complex128:
    r"""Calculates the complex exponential in the 2nd half of (A23).

    Args:
        big_d: Graph regularity minus one. It appears in denominators as :math:`\sqrt{D}` instead of
            :math:`\sqrt{D+1} = \sqrt{d}` which appears in the formula in the paper.
        h: The local field of the problem.
        big_gamma_pruned: The array of gamma angles :math:`\mathbb{\Gamma}^{(m-1)}`. sorted ascendingly and descendingly
            in an array like :math:`(\gamma_1, \gamma_2, ... , \gamma_{p-1}, -\gamma_{p-1}, ..., -\gamma_2, -\gamma_1)`.
        a: The vector :math:`\mathbb{a}^{(m-1)}`.
        b: The vector :math:`\mathbb{b}^{(m-1)}`

    Returns:
        The product of the two exponential terms in (A23).

    """
    c = a * b
    # These need to be cast as arrays of floats, so that `numba` can `np.dot` them with `big_gamma_pruned`.
    c_float = c.astype(np.float64)
    b_float = b.astype(np.float64)

    return np.exp(1j * 1 / np.sqrt(big_d) * (np.dot(big_gamma_pruned, c_float) + h * np.dot(big_gamma_pruned, b_float)))


@jit(nopython=True, fastmath=True)
def get_big_h_fixed_points_from_scratch(
    p: int,
    big_d: int,
    h: float,
    big_gamma: NDArray[np.float64],
    big_beta: NDArray[np.float64],
    basis_list_t: list[list[NDArray[np.int8]]] | None = None,
) -> tuple[list[NDArray[np.complex128]], list[NDArray[np.complex128]]]:
    r"""Recursively implements equations (A23), (A24) and (A25) to obtain the fixed points of the calculation.

    Args:
        p: The number of QAOA layers, corresponding to the largest order of the fixed points that we want to calculate.
        big_d: Graph regularity minus one. Note that in expressions with :math:`Gamma`, we use :math:`\sqrt{D}` in
            the denominator instead of :math:`\sqrt{d} = \sqrt{D+1}`, but this just introduces a small factor to
            the gamma angles.
        h: The local field of the problem.
        big_gamma: The array of the :math:`\gamma` angles of the QAOA. The full array, with a 0 in the middle (see A8).
        big_beta: The array of the :math:`\beta` angles of the QAOA. The full array, with a 0 in the middle (see A8).
        basis_list_t: For some compatibility, the basis list sorted by the symmetry sectors can be pregenerated and
            given as an input. Otherwise it's generated internally.

    Returns:
        A list of lists of the fixed points. The outer index labels the order of the fixed point and the inner index
        labels its input. Therefore taking `[3][0]` of the output returns :math:`H^{(3)}_{d-1}([-1, -1, -1, 1, -1, -1])`
        because `[-1, -1, -1, 1, -1, -1]` is presumably the first basis state of 3 variables in symmetry sector 3.

    """
    fixed_points = []  # List()  # This will be outputted
    fixed_points_onsite = []  # List()  # This will be outputted

    # We calculate `fixed_points` by going through all values for `m`.
    # For `m` equal to 0, the fixed point is just 1, so we don't calculate anything.
    fixed_points.append(np.array([1], dtype=np.complex128))
    fixed_points_onsite.append(np.array([1], dtype=np.complex128))

    # Here is `m` equal to 1:
    fixed_points.append(
        np.array([np.cos(2 * big_gamma[0] / np.sqrt(big_d)) ** big_d], dtype=np.complex128)
    )  # First element is :math:`H^{(1)}_{d-1}(a_1^(1))`
    fixed_points_onsite.append(
        np.array([np.cos(2 * big_gamma[0] / np.sqrt(big_d)) ** (big_d + 1)], dtype=np.complex128)
    )

    if basis_list_t is None:
        # List of 2D numpy arrays. Each 2D array contains a full basis of a given size ranging from 0 to `p`.
        basis_list = [get_z_basis_m(m) for m in range(0, p + 1)]
        # Each inner 2D array is split into a list of 2D arrays (sorting the basis based on its symmetry sectors).
        # Thus, this is a list of lists of arrays. The outer list labels the number of variables in the basis vectors.
        # The middle list labels the symmetry sectors.
        # The arrays contains the numpy vectors of that size and symmetry, ordered.
        # E.g., `basis_list_t[4][2][0, :]` should return the first basis vector of 4 variables and symmetry sector 2,
        # i.e., `[-1, +1, -1, -1, -1, -1, -1, -1]`. Remember, the `t`th element is fixes to +1 (see `get_z_basis_m_t`).
        basis_list_t = [get_z_basis_m_t(basis) for basis in basis_list]

    for m in range(2, p + 1):  # Here `m` goes from 2 to p (p+1 is excluded).
        # The basis vectors in the symmetry sector :math:`t=m` have `2*m-2` unique elements.
        # Two of the elements are fixed (see `get_z_basis_m_t`).

        size = 2 ** (2 * m - 2)
        fixed_points_m = np.zeros(size, dtype=np.complex128)
        fixed_points_m_onsite = np.zeros(size, dtype=np.complex128)

        # Angle dependency for this iteration
        big_gamma_pruned, big_beta_pruned = prune_angles(m - 1, big_gamma, big_beta)

        # Indices to prune away the middle two elements of `am` (those are the fixed values).
        idx_select = np.arange(2 * m)
        idx_select = np.delete(idx_select, [m - 1, m])

        # Consider possible inputs :math:`\mathbb{a}^{(m)}`, only those with symmetry sector :math:`t = m`.
        # That's because of how the fixed points work, see text between (A37) and (A38).
        for i, am in enumerate(basis_list_t[m][m]):
            sum_over_b = 0.0 + 0.0j  # This will be the sum over :math:\mathbb{b}^{(m-1)}: in (A23)

            # However, the sum over bm_1 sums over all symmetry sectors.
            # First, iterate over bm_1 vectors with t = 0.
            for bm_1 in basis_list_t[m - 1][0]:
                # For t = 0, the fixed point is just 1.
                a23 = small_g(m - 1, bm_1, big_beta_pruned) * exp_gamma(
                    big_d, h, big_gamma_pruned, am[idx_select], bm_1
                )

                sum_over_b += (
                    a23
                    * 0.5
                    * a24_a25(am[m - 1], am[-m], bm_1[m - 2], bm_1[1 - m], big_beta[m - 2], big_gamma[m - 1], big_d)
                )

            # Now, iterate over bm_1 vectors with t > 0
            # Note that this only iterates over the elements of the basis up to a reflection (flip)
            # Therefore this flip needs to be explicitly added!
            for tm in range(1, m):
                # We create a list of indices covering the non-symmetric elements of bm_1
                # This does not depend on bm_1, so it can be done here
                idx_prune = np.array(list(range(tm - 1)) + list(range(-tm + 1, 0)), dtype=np.int64)

                for bm_1 in basis_list_t[m - 1][tm]:
                    # Same length of basis vectors, but now a different symmetry sector
                    # The summands contain a term :math:`H_{d-1}^{(m-1)(b^{(m-1)})}`. For bm_1 in a symmetry sector
                    # tm<m-1, this will change to :math:`H_{d-1}^{(tm)(b^{(tm)})}`.
                    # In code, this will be `fixed_points[tm][j]` where `j` is the index of the pruned bm_1 in
                    # the arrays `basis_list_t[tm][tm]` and `fixed_points[tm]`. We thus have to find this index!

                    # Extracts the non-symmetric part of `bm_1` and maps the -1 entries to 0.
                    pruned_bm_1 = (bm_1[idx_prune] + 1) // 2
                    # This is literally just an array of the powers of two.
                    powers_of_two = 1 << np.arange(len(pruned_bm_1) - 1, -1, -1)
                    # We assume that the fixed points are sorted ascedingly, so this will be the position of the correct
                    # fixed point. That's what the scalar product with the powers of two does.
                    j = np.sum(pruned_bm_1 * powers_of_two)

                    a23 = (
                        fixed_points[tm][j]
                        * small_g(m - 1, bm_1, big_beta_pruned)
                        * exp_gamma(big_d, h, big_gamma_pruned, am[idx_select], bm_1)
                    )

                    sum_over_b += (
                        a23
                        * 0.5
                        * a24_a25(am[m - 1], am[-m], bm_1[m - 2], bm_1[1 - m], big_beta[m - 2], big_gamma[m - 1], big_d)
                    )

                    flip_bm_1 = np.flip(bm_1)  # Now do the same as above, except for flipped `bm_1`

                    a23 = (
                        np.conj(fixed_points[tm][j])
                        * small_g(m - 1, flip_bm_1, big_beta_pruned)
                        * exp_gamma(big_d, h, big_gamma_pruned, am[idx_select], flip_bm_1)
                    )

                    sum_over_b += (
                        a23
                        * 0.5
                        * a24_a25(
                            am[m - 1],
                            am[-m],
                            flip_bm_1[m - 2],
                            flip_bm_1[1 - m],
                            big_beta[m - 2],
                            big_gamma[m - 1],
                            big_d,
                        )
                    )

            fixed_points_m[i] = sum_over_b**big_d
            fixed_points_m_onsite[i] = sum_over_b ** (big_d + 1)

        fixed_points.append(fixed_points_m)
        fixed_points_onsite.append(fixed_points_m_onsite)

    return fixed_points, fixed_points_onsite


@jit(nopython=True, fastmath=True)
def get_exp_vals(  # noqa: PLR0912
    p: int,
    big_d: int,
    h: float,
    big_gamma: np.ndarray,
    big_beta: np.ndarray,
    basis_list_t: None | list[list[np.ndarray]] = None,
) -> tuple[np.complex128, np.complex128]:
    r"""Calculates the expectation values of <Z> and <ZZ> terms.

    Basically, this reproduces the full calculation in equations A41-A44 and A49. The sum in equation A43 over
    :math:`t_a>t_b` is split in two cases, :math:`t_b = 0` and :math:`t_b > 0`.

    Args:
        p: The number of QAOA layers, corresponding to the largest order of the fixed points that we want to calculate.
        big_d: Graph regularity minus one. Note that in expressions with :math:`\gamma`, we use :math:`\sqrt{D}` in
            the denominator instead of :math:`\sqrt{d} = \sqrt{D+1}`, but this just introduces a small factor to
            the gamma angles.
        h: The local field of the problem.
        big_gamma: The array of the :math:`\gamma` angles of the QAOA. The full array, with a 0 in the middle (see A8).
        big_beta: The array of the :math:`\beta` angles of the QAOA. The full array, with a 0 in the middle (see A8).
        basis_list_t: For some compatibility, the basis list sorted by the symmetry sectors can be pregenerated and
            given as an input. Otherwise it's generated internally.

    Returns:
        A tuple of two floats representing the expectation values of <Z> and <ZZ> respectively. The output values are
        typed as `np.complex128`, but they are in fact real values (so their imaginary component is minimal).

    """
    if basis_list_t is None:
        # List of 2D numpy arrays. Each 2D array contains a full basis of a given size ranging from 0 to `p`.
        basis_list = [get_z_basis_m(m) for m in range(0, p + 1)]
        # Each inner 2D array is split into a list of 2D arrays (sorting the basis based on its symmetry sectors).
        # Thus, this is a list of lists of arrays. The outer list labels the number of variables in the basis vectors.
        # The middle list labels the symmetry sectors.
        # The arrays contains the numpy vectors of that size and symmetry, ordered.
        # E.g., `basis_list_t[4][2][0, :]` should return the first basis vector of 4 variables and symmetry sector 2,
        # i.e., `[-1, +1, -1, -1, -1, -1, -1, -1]`. Remember, the `t`th element is fixes to +1 (see `get_z_basis_m_t`).
        basis_list_t = [get_z_basis_m_t(basis) for basis in basis_list]

    # Start by calculating all of the necessary fixed points.
    fixed_points, fixed_points_onsite = get_big_h_fixed_points_from_scratch(
        p=p, big_d=big_d, h=h, big_gamma=big_gamma, big_beta=big_beta, basis_list_t=basis_list_t
    )

    # This will be returned
    sum_zz_term = np.complex128(0.0 + 0.0j)
    sum_z_term = np.complex128(0.0 + 0.0j)

    # <Z>
    for ta in range(1, p + 1):
        big_gamma_pruned, big_beta_pruned = prune_angles(ta, big_gamma, big_beta)
        for k, a in enumerate(basis_list_t[ta][ta]):
            for loop_nr in range(2):
                aa = (
                    a if loop_nr & 1 == 0 else np.flip(a)
                )  # This is bitwise AND operation, checks last bit of `loop_nr`.

                a49 = (
                    big_g(aa[ta - 1], big_beta[ta - 1 : p])
                    * small_g(ta, aa, big_beta_pruned)
                    * (fixed_points_onsite[ta][k] if loop_nr & 1 == 0 else np.conj(fixed_points_onsite[ta][k]))
                    * exp_gamma(big_d, h, big_gamma_pruned, np.zeros(len(big_gamma_pruned)), aa)
                )  # We use `exp_gamma` with zeros, so that we get only the desired product.

                sum_z_term += a49

    # Now we head into <ZZ>.
    # ta == tb, i.e., (A41) and (A42)
    for ta in range(1, p + 1):
        big_gamma_pruned, big_beta_pruned = prune_angles(ta, big_gamma, big_beta)

        # For a fixed symmetry sector `ta`, we iterate over all suitable a's and b's
        for k, a in enumerate(basis_list_t[ta][ta]):
            for i, b in enumerate(basis_list_t[ta][ta]):
                for loop_nr in range(4):
                    aa = (
                        a if loop_nr & 1 == 0 else np.flip(a)
                    )  # This is bitwise AND operation, checks last bit of `loop_nr`.
                    bb = b if loop_nr & 2 == 0 else np.flip(b)  # This is bitwise AND operation.

                    c = aa * bb

                    c_float = c.astype(np.float64)
                    a_float = aa.astype(np.float64)
                    b_float = bb.astype(np.float64)

                    a41 = (
                        big_g(aa[ta - 1], big_beta[ta - 1 : p])
                        * big_g(bb[ta - 1], big_beta[ta - 1 : p])
                        * small_g(ta, aa, big_beta_pruned)
                        * small_g(ta, bb, big_beta_pruned)
                        * (fixed_points[ta][k] if loop_nr & 1 == 0 else np.conj(fixed_points[ta][k]))
                        * (fixed_points[ta][i] if loop_nr & 2 == 0 else np.conj(fixed_points[ta][i]))
                    )

                    a42 = np.exp(
                        1j
                        * 1
                        / np.sqrt(big_d)
                        * (
                            np.dot(big_gamma_pruned, c_float)
                            + h * (np.sum(big_gamma_pruned * a_float) + np.sum(big_gamma_pruned * b_float))
                        )
                    )

                    sum_zz_term += a41 * a42

    # ta > tb = 0 (i.e., a special case of A43 and A44)
    for ta in range(1, p + 1):
        big_gamma_pruned, big_beta_pruned = prune_angles(ta, big_gamma, big_beta)
        for k, a in enumerate(basis_list_t[ta][ta]):
            for b in basis_list_t[ta][0]:
                # basis_list_t[ta][0] already contains the negative of all vectors, so we don't have to flip `b`
                for loop_nr in range(2):
                    aa = (
                        a if loop_nr & 1 == 0 else np.flip(a)
                    )  # This is bitwise AND operation, checks last bit of `loop_nr`.

                    c = aa * b
                    c_float = c.astype(np.float64)
                    b_float = b.astype(np.float64)
                    a_float = aa.astype(np.float64)

                    a43 = (
                        2
                        * big_g(aa[ta - 1], big_beta[ta - 1 : p])
                        * big_g_tilde(b[ta - 1], big_beta[ta - 1 : p])
                        * small_g(ta, aa, big_beta_pruned)
                        * small_g(ta, b, big_beta_pruned)
                        * (fixed_points[ta][k] if loop_nr & 1 == 0 else np.conj(fixed_points[ta][k]))
                    )  # The other fixed point is 1
                    a44 = np.exp(
                        1j
                        * 1
                        / np.sqrt(big_d)
                        * (np.sum(big_gamma_pruned * c_float) + h * np.sum(big_gamma_pruned * a_float))
                    )  # np.dot(big_gamma_pruned, b_float) is 0

                    sum_zz_term += a43 * a44

    # ta > tb > 0 (i.e., a more generic case of A43 and A44)
    for tb in range(1, p):
        for ta in range(tb + 1, p + 1):
            big_gamma_pruned, big_beta_pruned = prune_angles(ta, big_gamma, big_beta)
            for k, a in enumerate(basis_list_t[ta][ta]):
                for b in basis_list_t[ta][tb]:
                    # The summands contain a term :math:`H_{d-1}^{(tb)(b^{(tb)_tb})}`. But our `b` here has length `ta`,
                    # so we cannot simply use its index `i` to access the correct fixed point. We have to calculate
                    # the new index of this `b` when it's pruned only its first and last `tb` elements.
                    if tb == 1:
                        m = 0
                    else:
                        # Extracts the first and last `tb` entries of `b` and maps the -1 entries to 0.
                        pruned_b = (np.concatenate((b[: tb - 1], b[-(tb - 1) :])) + 1) // 2
                        # This is literally just an array of the powers of two.
                        powers_of_two = 1 << np.arange(len(pruned_b) - 1, -1, -1)
                        # We assume that the fixed points are sorted ascedingly, so this will be the position of
                        # the correct fixed point. That's what the scalar product with the powers of two does.
                        m = np.sum(pruned_b * powers_of_two)

                    for loop_nr in range(4):
                        aa = (
                            a if loop_nr & 1 == 0 else np.flip(a)
                        )  # This is bitwise AND operation, checks last bit of `loop_nr`.
                        bb = b if loop_nr & 2 == 0 else np.flip(b)  # This is bitwise AND operation.

                        c = aa * bb
                        c_float = c.astype(np.float64)
                        b_float = bb.astype(np.float64)
                        a_float = aa.astype(np.float64)

                        a43 = (
                            2
                            * big_g(aa[ta - 1], big_beta[ta - 1 : p])  # :math:`G_t(a_{t_a})`
                            * big_g_tilde(bb[ta - 1], big_beta[ta - 1 : p])  # :math:`\tilde{G}_t(b_{t_b})`
                            * small_g(ta, aa, big_beta_pruned)
                            * small_g(ta, bb, big_beta_pruned)
                            * (fixed_points[ta][k] if loop_nr & 1 == 0 else np.conj(fixed_points[ta][k]))
                            * (fixed_points[tb][m] if loop_nr & 2 == 0 else np.conj(fixed_points[tb][m]))
                        )
                        a44 = np.exp(
                            1j
                            * 1
                            / np.sqrt(big_d)
                            * (
                                np.sum(big_gamma_pruned * c_float)
                                + h * np.sum(big_gamma_pruned * a_float)
                                + h * np.sum(big_gamma_pruned * b_float)
                            )
                        )

                        sum_zz_term += a43 * a44

    return sum_z_term, sum_zz_term
