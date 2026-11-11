#  ********************************************************************************
#    _____  ____ _
#   / _ \ \/ / _` |  Framework for control
#  |  __/>  < (_| |  and measurement of
#   \___/_/\_\__,_|  superconducting qubits
#
#  Copyright 2019-2026 IQM Finland Oy
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ********************************************************************************
"""Utility functions for the model selection fitter."""

import copy
import itertools
import logging
from typing import Any

import lmfit
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import xarray as xr

from exa.core.util.formatting import format_unit_string, get_prefix, get_scaling

logger = logging.getLogger(__name__)


def calculate_aicc(aic: float, n: int, k: int, extra_parameter_penalty_factor: float = 1) -> float:
    r"""Akaike information criterion for small sample size.

    From wikipedia (https://en.wikipedia.org/wiki/Akaike_information_criterion):
    The formula for AICc depends upon the statistical model.
    Assuming that the model is univariate, is linear in its parameters,
    and has normally-distributed residuals (conditional upon regressors),
    then the formula for AICc is :math:`AICc = AIC + \frac{2k^{2}+2k}{n-k-1}`

    Args:
        aic: Akaike information
        n: sample size
        k: number of free parameters
        extra_parameter_penalty_factor: extra factor penalizing additional parameters
    Returns:
        AIC corrected for small sample sizes if condition n-k-1 is met, else returns ``np.nan``.

    """
    return aic + extra_parameter_penalty_factor * (2 * k**2 + 2 * k) / (n - k - 1) if (n - k - 1) > 0 else np.nan


def calculate_model_probability(aics: np.ndarray) -> np.ndarray:
    """Converts the AIC values to model probabilities. Works also for BIC.

    Args:
        aics: The AIC values for different models.

    Returns:
        Probabilities for each of the model to give the best fit to the data.

    """
    aic_min = aics.min()
    probs = np.exp((aic_min - aics) / 2)
    return probs


def outlier_detection(  # noqa: PLR0913, PLR0915, PLR0912
    y: np.ndarray | xr.DataArray,
    model: lmfit.Model,
    independent_vars: dict[str, np.ndarray],
    plot_fit: bool = False,
    fitter_options: dict[str, Any] | None = None,
    n_iterations: int = 3,
    n_picks: int = 2,
    drop_samples_with_large_difference: bool = False,
    max_outlier_fraction: float = 0.6,
    override_model_initial_guess: bool = False,
) -> np.ndarray:
    """Finds outlier indices using leave-one-out.

    The first independent variable is used as the outlier search dimension. Only supports 1-d fits.

    Args:
        y: 1-d data variable
        model: Model that is fit to the data.
        plot_fit: boolean indicating if the fit for outlier detection should be plotted
        fitter_options: extra options passed to lmfit.fit
        n_iterations:
            Number of iterations in outlier detection. For example, n_iterations=1 would lead to only
            n_picks outliers with the largest deviation to be selected. A new reference fit is done between
            each iteration.
        n_picks:
            Maximum number of outliers selected at each iteration.
        drop_samples_with_large_difference:
            Selects sample as outliers if their value differs from next the point by a significant value.
            The current implementation is not good, and therefore this feature should not be used.
        max_outlier_fraction:
            Maximum fraction of outliers from the data. When the maximum is reached, returns the outliers
            found.
        override_model_initial_guess:
            If True, initial guess is done using parameter values. If False model.guess function is used instead.
        independent_vars:
            kwargs for the names of the independent variables in the fit.

    Returns:
        outlier indices

    """
    fitter_options = fitter_options or {}

    def _to_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    independent_var = list(independent_vars.keys())[0]

    y_values = np.asarray(y.values if isinstance(y, xr.DataArray) else y)
    x = np.asarray(list(independent_vars.values())[0])

    deleted_mask: np.ndarray
    deleted_mask = np.zeros(y_values.shape, dtype=bool)
    deleted_indices: list[int] = []
    if drop_samples_with_large_difference:  # TODO: does not work correctly at the moment.
        diff_data = np.abs(np.diff(y_values, prepend=0))
        diff_data[0] = diff_data[1]
        diff_data_reverse = np.abs(np.diff(y_values[::-1], prepend=0))
        diff_data_reverse[0] = diff_data_reverse[1]
        diff_data_reverse = diff_data_reverse[::-1]
        median_diff = np.median(np.abs(diff_data))

        median_diff = np.median(diff_data)
        bad_data = np.atleast_1d(np.squeeze(np.argwhere(diff_data > median_diff * 3)))
        bad_data_reversed = np.atleast_1d(np.squeeze(np.argwhere(diff_data_reverse > median_diff * 3)))
        if len(bad_data) > 0 and len(bad_data_reversed) > 0:
            bad_data_combined = list(set(bad_data).union(set(bad_data_reversed)))
            for idx in bad_data_combined:
                idx_i = int(idx)
                if not deleted_mask[idx_i]:
                    deleted_mask[idx_i] = True
                    deleted_indices.append(idx_i)

    norm_chi_sq_max = -np.inf * np.ones(y_values.shape)
    N = int(y_values.size)
    chi_sq_limit: float = 3.0
    normalization: float = 1.0
    for jj in range(n_iterations):
        idx_fit_mask = ~deleted_mask
        tmp_x = x[idx_fit_mask]
        try:
            if override_model_initial_guess:
                pfit = model.make_params()
            else:
                pfit = model.guess(np.asarray(y_values[idx_fit_mask]), **{independent_var: tmp_x})
        except (NotImplementedError, TypeError):
            pfit = model.make_params()
        fit_result = model.fit(y_values[idx_fit_mask], **{independent_var: tmp_x}, params=pfit, **fitter_options)
        n = len(tmp_x)
        redchi = _to_float(getattr(fit_result, "redchi", 0.0))
        chisqr = _to_float(getattr(fit_result, "chisqr", 0.0))
        aicc_original = chisqr + redchi * (N - n)
        chi_sq = np.zeros_like(y_values)
        if jj == 0:
            normalization = float(aicc_original) if aicc_original else 1.0
        if np.std(chi_sq) < 100 * normalization:
            normalization = float(aicc_original) if aicc_original else 1.0
        for i in range(N):
            if deleted_mask[i]:
                continue

            # Leave-one-out: fit all non-deleted samples except i.
            idx2_mask = idx_fit_mask.copy()
            idx2_mask[i] = False
            tmp_x = x[idx2_mask]
            tmp = model.fit(y_values[idx2_mask], **{independent_var: tmp_x}, params=pfit, **fitter_options)
            n_tmp = len(tmp_x)
            tmp_chisqr = _to_float(getattr(tmp, "chisqr", 0.0))
            chi_sq[i] = tmp_chisqr + redchi * (N - n_tmp)

        norm_chi_sq = (aicc_original - chi_sq) / normalization * N
        norm_chi_sq_max = np.maximum(norm_chi_sq_max, norm_chi_sq)
        n_picks_current = np.min([n_picks, len(norm_chi_sq)])
        deleted_indices_temp = np.argpartition(norm_chi_sq, -n_picks_current)[-n_picks_current:]
        accepted_for_deletion = norm_chi_sq[deleted_indices_temp] > chi_sq_limit
        if not np.any(accepted_for_deletion):
            break
        for idx in deleted_indices_temp[accepted_for_deletion]:
            idx_i = int(idx)
            if not deleted_mask[idx_i]:
                deleted_mask[idx_i] = True
                deleted_indices.append(idx_i)

        if plot_fit:
            outlier_ind_tmp = np.array([False] * int(N))
            outlier_ind_tmp[deleted_indices] = True
            _, ax = plt.subplots()
            ax.plot(x, norm_chi_sq)
            ax.axhline(chi_sq_limit, color="k", linestyle="--")
            ax.scatter(
                x[outlier_ind_tmp],
                norm_chi_sq[outlier_ind_tmp],
                color="r",
                label="outlier",
            )
            ax.set_ylabel(r"Relative red. $\chi^2$ change")
            ax.set_xlabel("x")
            ax.legend()
            plt.title(f"Iteration {jj + 1}")

        if len(deleted_indices) > max_outlier_fraction * len(y):
            break

    # Plot the influence on the red. aic of each point:
    outlier_ind = np.array([False] * int(N))
    outlier_ind[deleted_indices] = True
    if plot_fit:
        _, ax = plt.subplots()
        ax.plot(x, norm_chi_sq_max)
        ax.axhline(chi_sq_limit, color="k", linestyle="--")
        ax.scatter(x[outlier_ind], norm_chi_sq_max[outlier_ind], color="r", label="outlier")
        ax.set_ylabel(r"Relative red. $\chi^2$ change")
        ax.set_xlabel("x")
        ax.legend()
    return outlier_ind


def plot_conf_intervals(
    result: lmfit.model.ModelResult,
    parameter_pairs: list[tuple[str, str]] | None = None,
    N: int = 30,
) -> None:
    """Plots the confindence intervals for correlations between model parameters.

    By default, confidence intervals are plotted only for pairs which magnitued of correlation is more than 0.1.

    Args:
        result: Fitting result.
        parameter_pairs: List of parameter pairs for which the confidence interavals are calculated.
                        If None, pairs with high correlations are chosen.
        N: Number of points in the plot for both dimensions.

    """
    result = copy.deepcopy(result)
    if parameter_pairs is None:
        parameter_pairs = []
        all_pairs = list(itertools.combinations(result.params.keys(), 2))
        for pair in all_pairs:
            try:
                if np.abs(result.params[pair[0]].correl[pair[1]]) > 0.1:
                    parameter_pairs.append(pair)
            except (
                KeyError,
                TypeError,
            ):  # pair is not found if it is fixed. Gives TypeError
                pass

    fig, axs = plt.subplots(1, len(parameter_pairs), figsize=(12.8, 4.8))
    for ii, pair in enumerate(parameter_pairs):
        limits = None
        cx, cy, grid = lmfit.conf_interval2d(result, result, pair[0], pair[1], N, N, limits=limits)
        try:
            ax = axs[ii]
        except TypeError:
            ax = axs
        ctp = ax.contourf(cx, cy, grid, np.linspace(0, 1, N))
        fig.colorbar(ctp, ax=ax)
        ax.set_xlabel(pair[0])
        ax.set_ylabel(pair[1])
    plt.tight_layout()


def independent_vars_to_list_of_coords(independent_vars: dict[str, str | list[str]]) -> list[str]:
    """Flattens the independent variables to a list of coordinates.

    Args:
        independent_vars: The independent variables to flatten.

    Returns:
        The flattened coordinate names corresponding to independent variables in the model.

    """
    independent_var_values = independent_vars.values()
    list_of_coorindates: list[str] = []
    for independent_var_coordinate in independent_var_values:
        if isinstance(independent_var_coordinate, list):
            list_of_coorindates.extend(independent_var_coordinate)
        else:
            list_of_coorindates.append(independent_var_coordinate)
    return list_of_coorindates


def auto_rotate(vec: np.ndarray) -> np.ndarray:
    """Rotates a list of complex numbers to the real quadrature that contains the most amount of information.

    Args:
        vec: A NumPy array containing the complex numbers. If real, original value is returned.

    Returns: A NumPy array with real entries that contains the rotated complex numbers.

    """
    if np.all(np.isreal(vec)):
        return vec
    shape = vec.shape
    vec = np.reshape(vec, (-1, 1))

    pca = PCA(n_components=1)
    X = np.concatenate([np.real(vec), np.imag(vec)], axis=-1)
    pca.fit(X)
    X_pca = np.mean(pca.transform(X), axis=-1)
    X_pca = np.reshape(X_pca, shape)

    return X_pca


def fft_unit_conversion(unit: str, label: str | None = None) -> tuple[str, str]:
    """Converts seconds to frequency and frequency to seconds."""
    unit_conversions = {
        "s": ("frequency", "Hz"),
        "Hz": ("time", "s"),
    }
    if unit in unit_conversions:
        return unit_conversions[unit]
    return f"F({label})", ""


def get_unit(data_array: xr.DataArray) -> str | list[str]:
    """Return unit or units of the data array."""
    return getattr(data_array, "units", getattr(data_array, "unit", ""))


def create_unit_string(data_array: xr.DataArray, enclosure: str = "(") -> tuple[float, str]:
    """Creates a string that contains the units for that coordinates in the data_array.

    Args:
        data_array: Array with all the data and the coordinate information.
        enclosure: enclosing bracket

    Returns: a tuple where the first element is the scaling of the data (float) and the second element
    is the SI unit with a prefix (str).

    """
    scaling = 1
    unit = get_unit(data_array)
    if isinstance(unit, list):
        unit = unit[0] if unit else ""
    value = np.array(data_array.value) if hasattr(data_array, "value") else np.array(data_array)
    try:
        scaling = get_scaling(value)  # type: ignore[assignment]
        unit_prefix = get_prefix(value)
    except Exception:
        scaling = 1
        unit_prefix = ""

    normalized_unit = ""
    unit = unit or (normalized_unit if scaling != 1 else "")
    if isinstance(unit, list):
        unit = unit[0] if unit else ""
    if not unit and scaling != 1:
        unit_string = f"x {scaling:.0e}"
    else:
        unit_string = format_unit_string(unit_prefix, unit, enclosure=enclosure if unit else "")  # type: ignore[arg-type]
    return (scaling, unit_string)


def get_formatted_label(data_array: xr.DataArray) -> tuple[float, str]:
    """Gets the formatted label of an `xarray.DataArray`.

    Args:
        data_array: The data array

    Returns:
        Tuple containing the scaling and the quantity + unit as a string.

    """
    try:
        scaling, unit = create_unit_string(data_array)
        quantity = str(data_array.name)
    except Exception:
        scaling = 1
        unit = ""
        quantity = ""
    return scaling, " ".join(filter(None, [quantity, unit]))


def get_matching_indices(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Gets the indices for which values in target match the source."""
    xsorted = np.argsort(source)
    ypos = np.searchsorted(source[xsorted], target)
    indices = xsorted[ypos]
    return indices
