"""Generation of error measures and result tables."""
# ruff: noqa: N802, N803, N806

from argparse import Namespace
import csv
from itertools import product
import os

from iqm.benchmarks.benchmark_definition import BenchmarkObservationIdentifier
from iqm.benchmarks.compressive_gst.mgst import (
    additional_fns,
    algorithm,
    low_level_jit,
    optimization,
    qiskit_interface,
    utils_gst,
)
import numpy as np
import numpy.linalg as la
import pandas as pd
from qiskit.quantum_info import SuperOp
from qiskit.quantum_info.operators.measures import diamond_norm
from scipy.linalg import expm, logm, schur
from scipy.optimize import linear_sum_assignment, minimize
import xarray as xr


def min_spectral_distance(X1: np.ndarray, X2: np.ndarray) -> float:
    """Computes the average absolute distance between the eigenvlues of two matrices.

    The matrices are first diagonalized, then the eigenvalues are matched such that the average
    distance of matched eigenvalue pairs is minimal.
    The big advantage of this distance metric is that it is gauge invariant and it can thus be used
    to if the reconstructed gates are similar to the target gates before any gauge optimization.

    Args:
        X1: The first matrix
        X2: The second matrix

    Returns:
        The minimal distance

    """
    eigs = la.eig(X1)[0]
    eigs_t = la.eig(X2)[0]
    cost_matrix = np.abs(eigs[None, :] - eigs_t[:, None])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    normalization = np.abs(eigs).sum()
    dist = cost_matrix[row_ind, col_ind].sum() / normalization
    return dist


def MVE_data(X: np.ndarray, E: np.ndarray, rho: np.ndarray, J: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Mean varation error between measured outcomes and predicted outcomes from a gate set.

    Args:
        X: Gate set
        E: POVM
        rho: Initial state
        J: 2D array where each row contains the gate indices of a gate sequence
        y: 2D array of measurement outcomes for sequences in J; Each column contains the outcome probabilities
            for a fixed sequence

    Returns:
        dist: Mean variation error
        max_dist: Maximal varaition error

    Notes:
        For each sequence the total variation error of the two probability distribution over the POVM elements
            is computed. Afterwards the meean over these total variation errors is returned.

    """
    m = y.shape[1]
    dist: float = 0
    max_dist: float = 0
    for i in range(m):
        j = J[i]
        C = low_level_jit.contract(X, j)
        curr = np.sum(np.abs(E.conj() @ C @ rho - y[:, i])) / 2.0
        dist += curr
        max_dist = max(max_dist, curr)
    return dist / m, max_dist


def generate_basis_labels(pdim: int, basis: str | None = None) -> list[str]:
    """Generate a list of labels for the Pauli basis or the standard basis.

    Args:
        pdim: Physical dimension
        basis: Which basis the labels correspond to, currently default is standard basis and "Pauli" can be chosen
            for Pauli basis labels like "II", "IX", "XX", ...

    Returns:
        labels: A list of all string combinations for the given dimension and basis

    """
    separator = ""
    if basis == "Pauli":
        pauli_labels_loc = ["I", "X", "Y", "Z"]
        pauli_labels_rep = [pauli_labels_loc for _ in range(int(np.log2(pdim)))]
        labels = [separator.join(map(str, x)) for x in product(*pauli_labels_rep)]
    else:
        std_labels_loc = ["0", "1"]
        std_labels_rep = [std_labels_loc for _ in range(int(np.log2(pdim)))]
        labels = [separator.join(map(str, x)) for x in product(*std_labels_rep)]

    return labels


def report(
    X: np.ndarray,
    E: np.ndarray,
    rho: np.ndarray,
    J: np.ndarray,
    y: np.ndarray,
    X_t: np.ndarray,
    E_t: np.ndarray,
    rho_t: np.ndarray,
    gate_labels: dict[int, str],
    weights: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generation of pandas dataframes with gate and SPAM quality measures.

    The resutls can be converted to .tex tables or other formats to be used for GST reports

    Args:
        X: Gate set
        E: POVM
        rho: Initial state
        J: 2D array where each row contains the gate indices of a gate sequence
        y: 2D array of measurement outcomes for sequences in J; Each column contains the outcome probabilities
            for a fixed sequence
        X_t: Target gate set
        E_t: Target POVM
        rho_t: Target initial state
        gate_labels: A list of names for the gates in X
        weights: A list of weights used for gauge optimization

    Returns:
        df_g: DataFrame of gate quality measures
        df_o: DataFrame of all other quality/error measures

    """
    pdim = int(np.sqrt(rho.shape[0]))

    X_opt, E_opt, rho_opt = optimization.gauge_opt(X, E, rho, X_t, E_t, rho_t, weights=weights)
    E_map = np.array(
        [
            np.outer(
                np.kron(additional_fns.basis(pdim, i), additional_fns.basis(pdim, i)),
                E_opt[i].conj(),
            )
            for i in range(E_opt.shape[0])
        ]
    ).sum(axis=0)
    E_map_t = np.array(
        [
            np.outer(
                np.kron(additional_fns.basis(pdim, i), additional_fns.basis(pdim, i)),
                E_t[i].conj(),
            )
            for i in range(E_opt.shape[0])
        ]
    ).sum(axis=0)

    final_objf = low_level_jit.objf(X, E, rho, J, y)
    MVE, _ = MVE_data(X, E, rho, J, y)
    MVE_target, _ = MVE_data(X_t, E_t, rho_t, J, y)

    povm_dd = diamond_norm(SuperOp(E_map) - SuperOp(E_map_t)) / 2
    rho_td = la.norm(rho_opt.reshape((pdim, pdim)) - rho_t.reshape((pdim, pdim)), ord="nuc") / 2
    F_avg = utils_gst.average_gate_fidelities(X_opt, X_t)
    DD = [diamond_norm(SuperOp(X_opt[i]) - SuperOp(X_t[i])) / 2 for i in range(len(X))]
    min_spectral_dists = [min_spectral_distance(X[i], X_t[i]) for i in range(X.shape[0])]

    df_g = pd.DataFrame(
        {
            "F_avg": F_avg,
            "Diamond distances": DD,
            "Min. Spectral distances": min_spectral_dists,
        }
    )
    df_o = pd.DataFrame(
        {
            "Final cost function value": final_objf,
            "Mean total variation dist. to data": MVE,
            "Mean total variation dist. target to data": MVE_target,
            "POVM - Meas. map diamond dist.": povm_dd,
            "State - Trace dist.": rho_td,
        },
        index=[0],
    )
    df_g.rename(index=gate_labels, inplace=True)
    df_o.rename(index={0: ""}, inplace=True)

    return df_g, df_o


def quick_report(
    X: np.ndarray,
    E: np.ndarray,
    rho: np.ndarray,
    J: np.ndarray,
    y: np.ndarray,
    X_t: np.ndarray,
    E_t: np.ndarray,
    rho_t: np.ndarray,
    gate_labels: dict[int, str] | None = None,
    weights: list[float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generation of pandas dataframes with gate and SPAM quality measures.

    The quick report is intended to check on a GST estimate with fast to compute measures
    (no diamond distance) to get a first picture and check whether mGST and the gauge optimization
    produce meaningful results.

    Args:
        X: Gate set
        E: POVM
        rho: Initial state
        J: 2D array where each row contains the gate indices of a gate sequence
        y: 2D array of measurement outcomes for sequences in J; Each column contains the outcome probabilities
            for a fixed sequence
        X_t: Target gate set
        E_t: Target POVM
        rho_t: Target initial state
        gate_labels: A list of names for the gates in X
        weights: A list of weights used for gauge optimization

    Returns:
        df_g: DataFrame of gate quality measures
        df_o: DataFrame of all other quality/error measures

    """
    pdim = int(np.sqrt(rho.shape[0]))
    d = X.shape[0]
    if not gate_labels:
        gate_labels = {i: "Gate %i" % i for i in range(d)}

    X_opt, E_opt, rho_opt = optimization.gauge_opt(X, E, rho, X_t, E_t, rho_t, weights=weights)

    final_objf = low_level_jit.objf(X, E, rho, J, y)
    MVE, _ = MVE_data(X, E, rho, J, y)
    MVE_target, _ = MVE_data(X_t, E_t, rho_t, J, y)

    E_map = np.array(
        [
            np.outer(
                np.kron(additional_fns.basis(pdim, i), additional_fns.basis(pdim, i)),
                E_opt[i].conj(),
            )
            for i in range(E_opt.shape[0])
        ]
    ).sum(axis=0)
    E_map_t = np.array(
        [
            np.outer(
                np.kron(additional_fns.basis(pdim, i), additional_fns.basis(pdim, i)),
                E_t[i].conj(),
            )
            for i in range(E_opt.shape[0])
        ]
    ).sum(axis=0)

    povm_dd = diamond_norm(SuperOp(E_map) - SuperOp(E_map_t)) / 2

    rho_td = la.norm(rho.reshape((pdim, pdim)) - rho_t.reshape((pdim, pdim)), ord="nuc") / 2
    F_avg = utils_gst.average_gate_fidelities(X_opt, X_t)
    min_spectral_dists = [min_spectral_distance(X[i], X_t[i]) for i in range(X.shape[0])]

    df_g = pd.DataFrame({"F_avg": F_avg, "Min. Spectral distances": min_spectral_dists})
    df_o = pd.DataFrame(
        {
            "Final cost function": final_objf,
            "Mean TVD estimate-data": MVE,
            "Mean TVD target-data": MVE_target,
            "SPAM error:": rho_td + povm_dd,
        },
        index=[0],
    )
    df_g.rename(index=gate_labels, inplace=True)
    df_o.rename(index={0: ""}, inplace=True)

    return df_g, df_o


def compute_angles_axes(
    U_set: np.ndarray, alternative_phase: bool = False
) -> tuple[list[float], list[np.ndarray], np.ndarray]:
    """Takes the matrix logarithm of the given unitaries and returns the Hamiltonian parameters.

    The parametrization is U = exp(-i pi H/2), i.e. H = i log(U)*2/pi

    Args:
        U_set: A list contining unitary matrices
        alternative_phase: Whether an attempt should be made to report more intuitive rotations,
            for example a rotation of 3 pi/2 around the -X axis would be turned into a pi/2 rotation around the X-axis.

    Returns:
        angles: The rotation angle on the Bloch sphere for all gates
        axes: The normalized rotation axes in the Pauli basis for all gates
        pauli_coeffs: The full list of Pauli basis coefficients of the Hamiltonian for all gates

    Notes:
        sqrt(pdim) factor is due to Pauli basis normalization

    """
    d = U_set.shape[0]
    pdim = U_set.shape[1]
    angles = []
    axes = []
    pp_vecs = []
    for i in range(d):
        H = 1j * logm(U_set[i])
        pp_vec = utils_gst.basis_transform(H.reshape(-1), "std", "pp")
        original_phase = la.norm(pp_vec[1:]) * 2 / np.sqrt(pdim)
        if alternative_phase and (-np.min(pp_vec) > np.max(pp_vec)) and original_phase > np.pi:
            alt_phase = (-original_phase + 2 * np.pi) % (2 * np.pi)
            pp_vec = -pp_vec
        else:
            alt_phase = original_phase
        angles.append(alt_phase / np.pi)
        axes.append(pp_vec[1:] / la.norm(pp_vec[1:]))
        pp_vecs.append(pp_vec)
    pauli_coeffs = np.array(pp_vecs) / np.sqrt(pdim) / np.pi * 2
    return angles, axes, pauli_coeffs


def compute_sparsest_Pauli_Hamiltonian(U_set: np.ndarray) -> np.ndarray:
    """Takes the matrix logarithms of the given unitaries and returns sparsest Hamiltonian parameters in Pauli basis.

    The parametrization is U = exp(-i pi H/2), i.e. H = i log(U)*2/pi.
    Different branches in the matrix logarithm lead to different Hamiltonians. This function optimizes over
    combinations of adding 2*pi to different eigenvalues, in order arrive at the branch with the Hamiltonian
    whose Pauli basis representation is the most sparse.

    Args:
        U_set: A list contining unitary matrices

    Returns:
        The full list of Pauli basis coefficients of the Hamiltonian for all gates

    Notes:
        sqrt(pdim) factor is due to Pauli basis normalization

    """
    pdim = U_set.shape[1]
    pp_vecs = []

    for _, U in enumerate(U_set):
        # Schur decomposition finds the unitary diagonalization of a unitary matrix,
        # which is not always returned by np.linalg.eig
        T, evecs = schur(U)
        evals = np.diag(T)
        Pauli_norms = []
        log_evals = np.log(evals)
        for i in range(2**pdim):
            bits = low_level_jit.local_basis(i, 2, pdim)
            evals_new = 1j * log_evals + 2 * np.pi * bits
            H_new = evecs @ np.diag(evals_new) @ evecs.T.conj()
            pp_vec = utils_gst.basis_transform(H_new.reshape(-1), "std", "pp")
            Pauli_norms.append(np.linalg.norm(pp_vec, ord=1))
        opt_bits = low_level_jit.local_basis(np.argsort(Pauli_norms)[0], 2, pdim)
        evals_opt = 1j * log_evals + 2 * np.pi * opt_bits
        H_opt = evecs @ np.diag(evals_opt) @ evecs.T.conj()
        pp_vecs.append(utils_gst.basis_transform(H_opt.reshape(-1), "std", "pp"))
    pauli_coeffs = np.array(pp_vecs) / np.sqrt(pdim) / np.pi * 2
    return pauli_coeffs


def match_hamiltonian_phase(pauli_coeffs: np.ndarray, pauli_coeffs_target: np.ndarray) -> np.ndarray:
    """Matches the sign of the phase of a target Hamiltonian represented by Pauli coeffs to a measured Hamiltonian.

    Args:
        pauli_coeffs: Pauli coefficients of the measured Hamiltonian.
        pauli_coeffs_target: Pauli coefficients of the target Hamiltonian.

    Returns:
        Pauli coefficients of the target Hamiltonian with matched phase.

    """
    dim = int(np.sqrt(len(pauli_coeffs)))
    H_target = utils_gst.basis_transform(pauli_coeffs_target, "pp", "std").reshape((dim, dim))
    U_target = expm(-1j * H_target * np.pi * np.sqrt(dim) / 2)
    U_target_alt = expm(1j * H_target * np.pi * np.sqrt(dim) / 2)

    tolerance = 1e-6

    # Check if sign flip leads to same unitary
    if np.linalg.norm(U_target - U_target_alt) < tolerance:
        # Check if sign flip leads to better match to target Hamiltonian in Pauli basis
        if np.linalg.norm(pauli_coeffs_target - pauli_coeffs) > np.linalg.norm(pauli_coeffs_target + pauli_coeffs):
            return -pauli_coeffs_target

    return pauli_coeffs_target


def generate_rotation_param_results(
    dataset: xr.Dataset,
    qubit_layout: list[int],
    X_opt: np.ndarray,
    K_target: np.ndarray,
    X_array: np.ndarray | None = None,
    E_array: np.ndarray | None = None,
    rho_array: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Produces result tables and data for Kraus rank 1 estimates.

    This includes parameters of the Hamiltonian generators in the Pauli basis for all gates.
    If bootstrapping data is available, error bars will also be generated.

    Args:
        dataset: xarray.Dataset
            A dataset containing counts from the experiment and configurations
        qubit_layout: List[int]
            The list of qubits for the current GST experiment
        X_opt: 3D numpy array
            The gate set after gauge optimization
        K_target: 4D numpy array
            The Kraus operators of all target gates, used to compute distance measures.
        X_array: ndarray, optional
            Array of bootstrap gate estimates, used for error bars
        E_array: ndarray, optional
            Array of bootstrap POVM estimates, used for error bars
        rho_array: ndarray, optional
            Array of bootstrap state estimates, used for error bars

    Returns:
        Tuple[DataFrame, dict]:
            - df_g_rotation: Pandas DataFrame containing Hamiltonian (rotation) parameters with formatted values
            - hamiltonian_params: Dictionary with raw values and uncertainty bounds if bootstrapping was used

    """
    identifier = BenchmarkObservationIdentifier(qubit_layout).string_identifier
    pauli_labels = generate_basis_labels(dataset.attrs["pdim"], basis="Pauli")

    U_opt = phase_opt(X_opt, K_target)
    pauli_coeffs = compute_sparsest_Pauli_Hamiltonian(U_opt)

    bootstrap = X_array is not None and E_array is not None and rho_array is not None

    if X_array is not None and E_array is not None and rho_array is not None:
        bootstrap_pauli_coeffs = np.zeros((len(X_array), dataset.attrs["num_gates"], dataset.attrs["pdim"] ** 2))
        for i in range(X_array.shape[0]):
            X_std, _, _ = utils_gst.pp2std(X_array[i], E_array[i], rho_array[i])
            U_opt_ = phase_opt(X_std, K_target)
            pauli_coeffs_ = compute_sparsest_Pauli_Hamiltonian(U_opt_)
            bootstrap_pauli_coeffs[i, :, :] = pauli_coeffs_
        pauli_coeffs_low, pauli_coeffs_high = np.nanpercentile(bootstrap_pauli_coeffs, [2.5, 97.5], axis=0)

    # Pandas dataframe with formated confidence intervals
    df_g_rotation = pd.DataFrame(
        np.array(
            [
                [
                    number_to_str(
                        pauli_coeffs[i, j],
                        ((pauli_coeffs_high[i, j], pauli_coeffs_low[i, j]) if bootstrap else None),
                        precision=5,
                    )
                    for i in range(dataset.attrs["num_gates"])
                ]
                for j in range(dataset.attrs["pdim"] ** 2)
            ]
        ).T
    )
    df_g_rotation.columns = ["h_%s" % label for label in pauli_labels]
    df_g_rotation.rename(index=dataset.attrs["gate_labels"][identifier], inplace=True)

    hamiltonian_params = {
        "values": pauli_coeffs,
        "uncertainties": (pauli_coeffs_low, pauli_coeffs_high) if bootstrap else None,
    }

    return df_g_rotation, hamiltonian_params


def compute_matched_ideal_hamiltonian_params(dataset: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Computes the Hamiltonian parameters and matches the ideal Hamiltonian parameters to the measured ones.

     This does not change the unitary generated by the Hamiltonian

    Args:
        dataset (xarray.Dataset): A dataset containing counts from the experiment and configurations.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: A tuple containing:
            - hamiltonian_params: Hamiltonian parameters of the measured gates.
            - hamiltonian_params_ideal_matched: Ideal Hamiltonian parameters matched to the measured ones.

    """
    qubit_layouts = dataset.attrs["qubit_layouts"]
    param_list_layouts = [
        dataset.attrs[f"results_layout_{BenchmarkObservationIdentifier(layout).string_identifier}"][
            "hamiltonian_params"
        ]["values"]
        for layout in qubit_layouts
    ]
    hamiltonian_params = np.array(param_list_layouts)

    K_target = qiskit_interface.qiskit_gate_to_operator(dataset.attrs["gate_set"])
    X_target = np.einsum("ijkl,ijnm -> iknlm", K_target, K_target.conj()).reshape(
        (
            dataset.attrs["num_gates"],
            dataset.attrs["pdim"] ** 2,
            dataset.attrs["pdim"] ** 2,
        )
    )

    param_list_layouts = []
    for qubit_layout in qubit_layouts:
        _, ideal_params = generate_rotation_param_results(dataset, qubit_layout, X_target, K_target)
        ideal_params = ideal_params["values"]
        param_list_layouts.append(ideal_params)
    hamiltonian_params_ideal = np.array(param_list_layouts)

    hamiltonian_params_ideal_matched = np.empty(hamiltonian_params_ideal.shape)
    for i, _ in enumerate(qubit_layouts):
        for j in range(hamiltonian_params_ideal.shape[1]):
            hamiltonian_params_ideal_matched[i, j, :] = match_hamiltonian_phase(
                hamiltonian_params[i, j, :], hamiltonian_params_ideal[i, j, :]
            )

    return hamiltonian_params, hamiltonian_params_ideal_matched


def phase_err(angle: float, U: np.ndarray, U_t: np.ndarray) -> float:
    """Computes norm between two input unitaries after a global phase is added to one of them.

    Args:
        angle: The global phase angle (in rad)
        U: The first unitary matrix
        U_t: The second unitary matrix (typically the a target gate)

    Returns:
        The norm of the difference

    """
    return float(la.norm(np.exp(1j * angle) * U - U_t))


def phase_opt(X: np.ndarray, K_t: np.ndarray) -> np.ndarray:
    """Return rK = 1 gate set with global phase fitting matching to target gate set.

    Args:
        X: Array where CPT superoperators are stacked along the first axis. These should correspond to rK = 1 gates for
            the outputs to be meaningful.
        K_t: Array of target gate Kraus operators Each subarray along the first axis contains a set of Kraus operators.
            The second axis enumerates Kraus operators for a gate specified by the first axis.

    Returns:
        Array of Kraus operators with mathed global phase

    """
    d = X.shape[0]
    r = X.shape[1]
    pdim = int(np.sqrt(r))
    K = additional_fns.Kraus_rep(X, d, pdim, 1).reshape((d, pdim, pdim))
    K_t = K_t.reshape(d, pdim, pdim)
    K_opt = np.zeros(K.shape).astype(complex)
    for i in range(d):
        angle_opt = minimize(phase_err, 1, bounds=[[-np.pi, np.pi]], args=(K[i], K_t[i])).x  # type: ignore
        K_opt[i] = K[i] * np.exp(1j * angle_opt)
    return K_opt


def unitarities(X_opt_pp: np.ndarray) -> np.ndarray:
    """Computes the unitarities of all gates in the gate set.

    Args:
        X_opt_pp: Array where CPT superoperators in Pauli basis are stacked along the first axis.

    Returns:
        List of unitarities for the gates in X_opt_pp.

    """
    # Definition: Proposition 1 of https://arxiv.org/pdf/1503.07865.pdf
    pdim = int(np.sqrt(X_opt_pp.shape[1]))
    E_u = X_opt_pp[:, 1:, 1:]  # unital submatrices
    unitarity_list = np.real(np.einsum("ijk, ijk -> i", E_u.conj(), E_u) / (pdim**2 - 1))
    return unitarity_list


# Spectrum of the Choi matrix
def generate_Choi_EV_table(
    X_opt: np.ndarray, n_evals: int, gate_labels: dict[int, str], filename: str | None = None
) -> pd.DataFrame:
    """Outputs a .tex document containing a table with the larges eigenvalues of the Choi matrix for each gate.

    Args:
        X_opt: Array where CPT superoperators in standard basis are stacked along the first axis.
        n_evals: Number of eigenvalues to be returned
        gate_labels: The names of gates in the gate set
        filename: The file name of the output .tex file

    """
    d, r, _ = X_opt.shape
    pdim = int(np.sqrt(r))
    Choi_evals_result = np.zeros((d, r))
    X_choi = X_opt.reshape(d, pdim, pdim, pdim, pdim)
    X_choi = np.einsum("ijklm->iljmk", X_choi).reshape((d, pdim**2, pdim**2))
    for j in range(d):
        Choi_evals_result[j, :] = np.sort(np.abs(la.eig(X_choi[j])[0]))[::-1]
    Choi_evals_normalized = np.einsum("ij,i -> ij", Choi_evals_result, 1 / la.norm(Choi_evals_result, axis=1, ord=1))

    df_g_evals = pd.DataFrame(Choi_evals_normalized)
    df_g_evals.rename(index=gate_labels, inplace=True)
    if filename:
        df_g_evals.style.to_latex(
            filename + ".tex",
            column_format="c|*{%i}{c}" % n_evals,
            position_float="centering",
            hrules=True,
            caption="Eigenvalues of the Choi state",
            position="h!",
        )
    return df_g_evals


def dephasing_dist(prob_vec: list[float], X_pp: np.ndarray) -> float:
    """Returns the distance between a given channel and a local dephasing channel with given probabilities.

    Args:
        prob_vec: A list of dephasing probabilities
        X_pp: Array where CPT superoperators in Pauli basis are stacked along the first axis.

    Returns:
        The norm between the channel difference

    """
    X_deph = additional_fns.local_dephasing_pp(prob_vec)
    return float(la.norm(X_pp - X_deph))


def dephasing_probabilities_2q(X_opt_pp: np.ndarray, X_ideal_pp: np.ndarray) -> list[np.ndarray]:
    """Determines the local dephasing channel parameters which best describe the noise model.

    Works for two qubit gates only

    Args:
        X_opt_pp: Array where reconstructed CPT superoperators in Pauli basis are stacked along the first axis.
        X_ideal_pp: Array where target gate CPT superoperators in Pauli basis are stacked along the first axis.

    Returns:
        The two best fit dephasing probabilities

    """
    dephasing_probs = []
    for i in range(X_opt_pp.shape[0]):
        dephasing_probs.append(minimize(dephasing_dist, [0.1, 0.1], args=X_opt_pp[i] @ la.inv(X_ideal_pp[i])).x)  # type: ignore[arg-type]
    return dephasing_probs


def bootstrap_errors(
    K: np.ndarray,
    X: np.ndarray,
    E: np.ndarray,
    rho: np.ndarray,
    mGST_args: dict,
    bootstrap_samples: int,
    weights: list[float],
    gate_labels: dict[int, str],
    X_t: np.ndarray,
    E_t: np.ndarray,
    rho_t: np.ndarray,
    parametric: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resamples circuit outcomes a number of times and computes GST estimates for each repetition.

    All results are then returned in order to compute bootstrap-error bars for GST estimates.
    Parametric bootstrapping uses the estimated gate set to create a newly sampled data set.
    Non-parametric bootstrapping uses the initial dataset and resamples according to the
    corresp. outcome probabilities.
    Each bootstrap run is initialized with the estimated gate set in order to save processing time.

    Args:
        K: Each subarray along the first axis contains a set of Kraus operators. The second axis enumerates
            Kraus operators for a gate specified by the first axis.
        X: Array where reconstructed CPT superoperators in standard basis are stacked along the first axis.
        E: Current POVM estimate
        rho: Current initial state estimate
        mGST_args: Arguments with which the run_mGST function was called
        bootstrap_samples: Number of bootstrapping repretitions
        weights: Gate weights used for gauge optimization
        gate_labels: The names of gates in the gate set
        X_t: Target gate set
        E_t: Target POVM
        rho_t: Target initial state
        parametric: If set to True, parametric bootstrapping is used, else non-parametric bootstrapping.

    Returns:
        X_array: Array containing all estimated gate tensors of different bootstrapping repretitions along first axis
        E_array: Array containing all estimated POVM tensors of different bootstrapping repretitions along first axis
        rho_array: Array containing all estimated initial states of different bootstrapping repretitions along
            first axis
        df_g_array: Contains gate quality measures of bootstrapping repetitions
        df_o_array: Contains SPAM and other quality measures of bootstrapping repetitions

    """
    # ruff: noqa: PLR0913
    ns = Namespace(**mGST_args)
    if parametric:
        y = np.real(
            np.array([[E[i].conj() @ low_level_jit.contract(X, j) @ rho for j in ns.J] for i in range(ns.n_povm)])
        )
    else:
        y = ns.y
    X_array = np.zeros((bootstrap_samples, *X.shape)).astype(complex)
    E_array = np.zeros((bootstrap_samples, *E.shape)).astype(complex)
    rho_array = np.zeros((bootstrap_samples, *rho.shape)).astype(complex)
    df_g_list = []
    df_o_list = []

    for i in range(bootstrap_samples):
        y_sampled = additional_fns.sampled_measurements(y, ns.meas_samples).copy()
        main_args = (y_sampled, ns.J, ns.l, ns.d, ns.r, ns.rK, ns.n_povm, ns.bsize, ns.meas_samples)
        _, X_, E_, rho_, _ = algorithm.run_mGST(
            main_args,
            method=ns.method,
            max_inits=ns.max_inits,
            max_iter=0,
            final_iter=ns.final_iter,
            threshold_multiplier=ns.threshold_multiplier,
            target_rel_prec=ns.target_rel_prec,
            init=[K, E, rho],
        )

        X_opt, E_opt, rho_opt = optimization.gauge_opt(X_, E_, rho_, X_t, E_t, rho_t, weights=weights)
        df_g, df_o = report(X_opt, E_opt, rho_opt, ns.J, y_sampled, X_t, E_t, rho_t, gate_labels, weights)
        df_g_list.append(df_g.values)
        df_o_list.append(df_o.values)

        X_opt_pp, E_opt_pp, rho_opt_pp = utils_gst.std2pp(X_opt, E_opt, rho_opt)

        X_array[i, :] = X_opt_pp
        E_array[i, :] = E_opt_pp
        rho_array[i, :] = rho_opt_pp

    return (X_array, E_array, rho_array, np.array(df_g_list), np.array(df_o_list))


def job_counts_to_mGST_format(qubits: list[int], num_povm: int, result_dict: list[dict[str, int]]) -> np.ndarray:
    """Turns the dictionary of outcomes obtained from qiskit backend into the format which is used in mGST.

    Args:
        qubits: The list of active qubits in the experiment
        num_povm: The number of POVM elements in the experiment
        result_dict: The counts from the qiskit backend for each circuit

    Returns:
        2D array of measurement outcomes for sequences in J; Each column contains the outcome probabilities
        for a fixed sequence

    """
    basis_dict_list = []
    for result in result_dict:
        # Translate dictionary entries of bitstring on the full system to the decimal
        # representation of bitstrings on the active qubits
        basis_dict = {entry: int("".join([entry[::-1][i] for i in qubits][::-1]), 2) for entry in result}
        # Sort by index:
        basis_dict = dict(sorted(basis_dict.items(), key=lambda item: item[1]))
        basis_dict_list.append(basis_dict)
    y_temp = []
    for i in range(len(result_dict)):
        row = [result_dict[i][key] for key in basis_dict_list[i]]
        if len(row) < num_povm:
            missing_entries = list(range(num_povm))
            for given_entry in basis_dict_list[i].values():
                missing_entries.remove(given_entry)
            for missing_entry in missing_entries:
                row.insert(missing_entry, 0)  # 0 measurement outcomes in not recorded entry
        y_temp.append(np.array(row) / np.sum(row))
    y = np.array(y_temp).T
    return y


def save_var_latex(key: str, value: str | int | float) -> None:
    """Saves variables in data file to be read in latex document.

    Credit to https://stackoverflow.com/a/66620671

    Args:
        key: The name of the variable
        value: The variable content, could be a string as in the name of the experiment, or the number of circuits run,
            or any other python variable that needs to be transfered

    """
    dict_var: dict[str, str | int | float] = {}

    file_path = os.path.join(os.getcwd(), "report/latex_vars.dat")

    try:
        with open(file_path, newline="", encoding="ASCII") as file:
            reader = csv.reader(file)
            for row in reader:
                dict_var[row[0]] = row[1]
    except FileNotFoundError:
        pass

    dict_var[key] = value

    with open(file_path, "w", encoding="ASCII") as f:
        for key_, value_ in dict_var.items():
            f.write(f"{key_},{value_}\n")


def number_to_str(number: float, uncertainty: tuple[float, float] | None = None, precision: int = 3) -> str:
    """Formats a floating point number to a string with the given precision.

    Args:
        number: The floating point number to format.
        uncertainty: The upper and lower values of the confidence interval
        precision: The number of decimal places to include in the formatted string.

    Returns:
        The formatted floating point number as a string.

    """
    if uncertainty is None:
        return f"{number:.{precision}f}"

    return f"{number:.{precision}f} [{uncertainty[1]:.{precision}f},{uncertainty[0]:.{precision}f}]"


def result_str_to_floats(result_str: str, err: str) -> tuple[float, float]:
    """Converts formated string results from mgst to float (value, uncertainty) pairs.

    Args:
        result_str: str
            The value of a result parameter formated as str
        err: str
            The error interval of the parameters

    Returns:
        value: float
            The parameter value as float
        uncertainty: float
            A single uncertainty value

    """
    if err:
        value = float(result_str.split("[", maxsplit=1)[0])
        rest = result_str.split("[")[1].split(",")
        uncertainty = float(rest[1][:-1]) - float(rest[0])
        return value, uncertainty
    return float(result_str), np.nan
