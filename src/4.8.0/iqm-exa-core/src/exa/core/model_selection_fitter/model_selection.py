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
"""Model-selection statistics for the model-selection fitter."""

from dataclasses import dataclass
import logging

import numpy as np

from .utils import calculate_aicc, calculate_model_probability

logger = logging.getLogger(__name__)


@dataclass
class ModelSelectionStats:
    """Dataclass to hold model selection statistics for each coordinate slice."""

    best_model: np.ndarray
    best_model_index: np.ndarray
    model_probabilities: np.ndarray
    AICs: np.ndarray
    AICcs: np.ndarray
    BICs: np.ndarray
    redchis: np.ndarray
    r2s: np.ndarray


def _compute_scalar_stats_into(
    model_results: np.ndarray,
    extra_parameter_penalty_factor: float,
    *,
    out_aics: np.ndarray,
    out_aiccs: np.ndarray,
    out_bics: np.ndarray,
    out_redchis: np.ndarray,
    out_r2s: np.ndarray,
) -> int:
    """Compute per-model scalar statistics into preallocated output arrays.

    Args:
        model_results: Object array of :class:`lmfit.ModelResult` for one coordinate slice.
        extra_parameter_penalty_factor: Extra penalty on parameter count for AICc.
        out_aics: Output buffer for AIC values.
        out_aiccs: Output buffer for AICc values.
        out_bics: Output buffer for BIC values.
        out_redchis: Output buffer for reduced chi-square values.
        out_r2s: Output buffer for R2 values.

    Returns:
        Index of the model with the lowest AICc (falling back to AIC when AICc is unavailable).

    """
    n_models = len(model_results)
    for model_index in range(n_models):
        model = model_results[model_index]
        out_aics[model_index] = model.aic
        out_bics[model_index] = model.bic
        sample_size = len(model.data)
        k = sample_size - model.nfree
        out_aiccs[model_index] = calculate_aicc(
            model.aic, sample_size, k, extra_parameter_penalty_factor=extra_parameter_penalty_factor
        )
        out_redchis[model_index] = model.redchi
        out_r2s[model_index] = 1 - model.redchi / np.var(model.data, ddof=1)

    if np.all(~np.isnan(out_aiccs)):
        return int(np.argmin(out_aiccs))
    logger.debug("Insufficient sample size for AICc calculation in some models.")
    min_aic = np.min(out_aics)
    is_significantly_better_with_aic = all(aic - min_aic > 5 or aic == min_aic for aic in out_aics)
    if is_significantly_better_with_aic or np.all(np.isnan(out_aiccs)):
        logger.debug(
            "Selecting best fit model based on AIC, because either there is a clear optimal model based "
            "on AIC or because AICc could not be calculated for any model."
        )
        return int(out_aics.argmin())
    logger.debug("Selecting best fit model based on AICc, excluding models where AICc calculation failed.")
    return int(np.nanargmin(out_aiccs))


def _select_best_models_values(
    fit_result_values: np.ndarray,
    extra_parameter_penalty_factor: float,
    select_by_bic: bool = False,
) -> ModelSelectionStats:
    """Select best models and stats using pure NumPy, avoiding any xarray operations.

    Args:
        fit_result_values: Object array of shape ``(n_models, *remaining_shape)`` containing
            :class:`lmfit.ModelResult` instances.
        extra_parameter_penalty_factor: Extra penalty on parameter count forwarded to AICc.
        select_by_bic: If ``True``, use BIC instead of AICc for best-model selection.

    Returns:
        :class:`_ModelSelectionStats` with arrays shaped to ``(n_models, *remaining_shape)``
        (or ``(*remaining_shape,)`` for scalar outputs).

    """
    # fit_result_values shape: (n_models, *remaining_shape)
    n_models = fit_result_values.shape[0]
    remaining_shape = fit_result_values.shape[1:]
    n_remaining = int(np.prod(remaining_shape)) if remaining_shape else 1

    fit_result_flat = fit_result_values.reshape(n_models, n_remaining)

    # Preallocate output arrays once
    best_model_flat = np.empty(n_remaining, dtype=object)
    best_ind_flat = np.empty(n_remaining, dtype=int)
    model_probs_flat = np.empty((n_models, n_remaining), dtype=float)
    aics_flat = np.empty((n_models, n_remaining), dtype=float)
    aiccs_flat = np.empty((n_models, n_remaining), dtype=float)
    bics_flat = np.empty((n_models, n_remaining), dtype=float)
    redchis_flat = np.empty((n_models, n_remaining), dtype=float)
    r2s_flat = np.empty((n_models, n_remaining), dtype=float)

    for i in range(n_remaining):
        best_ind = _compute_scalar_stats_into(
            fit_result_flat[:, i],
            extra_parameter_penalty_factor=extra_parameter_penalty_factor,
            out_aics=aics_flat[:, i],
            out_aiccs=aiccs_flat[:, i],
            out_bics=bics_flat[:, i],
            out_redchis=redchis_flat[:, i],
            out_r2s=r2s_flat[:, i],
        )
        if select_by_bic:
            # BIC selection overrides AIC
            best_ind = int(np.argmin(bics_flat[:, i]))
        best_ind_flat[i] = best_ind
        best_model_flat[i] = fit_result_flat[best_ind, i]

    if select_by_bic:
        model_probs_flat[:, :] = calculate_model_probability(bics_flat)
    elif np.all(~np.isnan(aiccs_flat)):
        model_probs_flat[:, :] = calculate_model_probability(aiccs_flat)
    else:
        aics_min = np.min(aics_flat, axis=0)  # (n_remaining,)
        is_significantly_better = np.all((aics_flat - aics_min > 5) | (aics_flat == aics_min), axis=0)
        all_nan_aiccs = np.all(np.isnan(aiccs_flat), axis=0)
        use_aics_mask = is_significantly_better | all_nan_aiccs  # (n_remaining,)
        if np.any(~use_aics_mask):
            model_probs_flat[:, ~use_aics_mask] = calculate_model_probability(aiccs_flat[:, ~use_aics_mask])
        if np.any(use_aics_mask):
            model_probs_flat[:, use_aics_mask] = calculate_model_probability(aics_flat[:, use_aics_mask])

    # Reshape to the original shapes for each output
    best_model = best_model_flat.reshape(remaining_shape)
    best_model_index = best_ind_flat.reshape(remaining_shape)
    model_probabilities = model_probs_flat.reshape((n_models,) + remaining_shape)
    AICs = aics_flat.reshape((n_models,) + remaining_shape)
    AICcs = aiccs_flat.reshape((n_models,) + remaining_shape)
    BICs = bics_flat.reshape((n_models,) + remaining_shape)
    redchis = redchis_flat.reshape((n_models,) + remaining_shape)
    r2s = r2s_flat.reshape((n_models,) + remaining_shape)

    return ModelSelectionStats(
        best_model=best_model,
        best_model_index=best_model_index,
        model_probabilities=model_probabilities,
        AICs=AICs,
        AICcs=AICcs,
        BICs=BICs,
        redchis=redchis,
        r2s=r2s,
    )
