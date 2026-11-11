#  ********************************************************************************
#    _____  ____ _
#   / _ \ \/ / _` |  Framework for control
#  |  __/>  < (_| |  and measurement of
#   \___/_/\_\__,_|  superconducting qubits
#
#  Copyright 2023-2026 IQM Finland Oy
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
"""Helpers for creating phase and conditional phase datasets."""

from collections.abc import Callable

import numpy as np
import xarray as xr

from exa.common.data import Parameter
from iqm.cpc.core.dataset import FitResults, add_many_data_arrays, annotate, apply_along_coordinate


def unwrap_with_nans(x: np.ndarray, y: np.ndarray):  # noqa: ANN201
    """Unwrap y along x, skipping NaNs."""
    unwrapped = np.copy(y)
    unwrapped[~np.isnan(unwrapped)] = np.unwrap(unwrapped[~np.isnan(unwrapped)])
    return {}, unwrapped


def get_unwrapped_phase_array(
    array: xr.DataArray,
    dimension: str | None = None,
) -> xr.DataArray:
    """Produce an unwrapped array skipping NaNs."""
    dimension = dimension or str(array.dims[-1])

    fit = apply_along_coordinate(
        data_array=array,
        coord=dimension,
        func=unwrap_with_nans,
    )
    unwrapped = annotate(fit[0], array.parameter.name + "_unwrapped")

    return unwrapped


def get_phase_2_points(phases: np.ndarray, data: np.ndarray) -> FitResults:
    """Calculate the qubit phase, assuming two points in the phase sweep were used, and the excited state probability is
    well-calibrated, so the data offset is 0.5.
    Find optimal parameters (A, delta) for the least-squares approximation of the model.

    data[k] = A*cos(phases[k] + delta) + 0.5

    Args:
        phases: an array of size 2, phases used
        data: experimental data, assumed to be excited state probability

    Returns:
        a dict containing the phase and amplitude of the oscillation, and the data back. This would be a fit, but
    there are no free parameters in this model, so the fit would be equal to the data anyway

    """
    model_mat = np.vstack([np.cos(phases), np.sin(phases)])

    (amplitude_1, amplitude_2), _chi2, _norm, _svdvals = np.linalg.lstsq(model_mat.T, data - 0.5, rcond=None)

    amplitude = np.hypot(amplitude_1, amplitude_2)
    delta = -np.arctan2(amplitude_2, amplitude_1)
    return {"phase": delta, "amplitude": amplitude, "offset": 0.5}, data


def get_phase_more_points(phases: np.ndarray, data: np.ndarray) -> FitResults:
    """Calculate the qubit phase, assuming more than two points in the phase sweep were used.
    Find optimal parameters (A, B, delta) for the least-squares approximation of the model.

    data[k] = A*cos(phases[k] + delta) + B

    Note that if the number of phases is exactly 3, the calculated fit is equal to the data.

    Args:
        phases: an array of size >=3, phases used
        data: experimental data, assumed to be excited state probability

    Returns:
        a dict containing the phase and amplitude of the oscillation, and the fit.

    """
    model_mat = np.vstack([np.cos(phases), np.sin(phases), np.ones_like(phases)])

    (amplitude_1, amplitude_2, offset), _chi2, _norm, _svdvals = np.linalg.lstsq(model_mat.T, data, rcond=None)

    amplitude = np.hypot(amplitude_1, amplitude_2)
    delta = -np.arctan2(amplitude_2, amplitude_1)

    fit = amplitude * np.cos(phases + delta) + offset

    return {"phase": delta, "amplitude": amplitude, "offset": offset}, fit


def get_phase_fitter(
    n_phases: int,
) -> Callable:
    r"""Generates a fitter function for cosine oscillation with frequency 1, used to determine the phase of the data.

    The fitter uses :func:`numpy.linalg.lstsq` to calculate the phase of the cosine oscillation. Its behaviour depends
    on the number of phases in the phase sweep. With 2 phases, the fitter will calculate the amplitude and phase
    corresponding to the model:
    :math:`y(x) = A * \cos(x+phi) + 0.5`
    where :math:`A>0` is amplitude, :math:`phi>0` is phase. Thus, this should only be used if excited state probability
    is calibrated correctly, and thus the model's assumption is satisfied.
    With 3 phases or more, the fitter will do a least-squares fit to the model
    :math:`y(x) = A * \cos(x+phi) + B`,
    where :math:`B` is offset. This three-parameter model cannot be used if there are only 2 phases, and
    if there are exactly 3, it is fully determined, so the resulting fit will be equal to the provided data.

    Args:
        n_phases: size of the phases array

    Returns:
        Function for fitting data to a cosine oscillation with frequency 1 model.

    """
    if n_phases == 1:
        raise ValueError("Number of phases needs to be larger than 1")

    if n_phases == 2:

        def _phase_fitter(x: np.ndarray, y: np.ndarray) -> tuple[dict, np.ndarray]:
            return get_phase_2_points(x, y)

    else:

        def _phase_fitter(x: np.ndarray, y: np.ndarray) -> tuple[dict, np.ndarray]:
            return get_phase_more_points(x, y)

    return _phase_fitter


def get_phase_array(
    data_array: xr.DataArray,
    phase_coord: str,
    prefix: str = "",
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Calculates the fit, phase and amplitude arrays, by collapsing the phase dimension on the given data. The fit
    data array has the same dimensions as the input data array, and the phase and amplitude arrays have the 'phase'
    dimension collapsed.

    Args:
        data_array: array containing the data, one dimension of which is phase
        phase_coord: name of the phase coordinate in the data array
        prefix: name to attach to the phase and amplitude arrays

    Returns:
        tuple of three data arrays, containing the fit, the qubit phase and amplitude respectively

    """
    phases_array = data_array[phase_coord].values
    n_phases = len(phases_array)
    phase_fitter = get_phase_fitter(n_phases)
    fit, fitted_phase, fitted_amplitude = apply_along_coordinate(
        data_array=data_array,
        coord=phase_coord,
        func=phase_fitter,
        returns=["phase", "amplitude"],
        prefix=prefix,
    )
    return fit, fitted_phase, fitted_amplitude


def add_phase_amplitude(  # noqa: ANN201
    data_array: xr.DataArray,
    target_dataset: xr.Dataset,
    phase_coord: str,
    prefix: str = "",
    add_fit: bool = False,
):
    """Adds the phase and amplitude calculated by :func:`.get_phase_array` to the target dataset.

    Args:
        data_array: array containing the data, one dimension of which is phase
        target_dataset: dataset to add the phase and amplitude arrays to
        phase_coord: name of the phase coordinate in the data array
        prefix: name to attach to the phase and amplitude arrays
        add_fit: whether to also add the fit to the dataset

    Returns:
        nothing, modifies the target_dataset

    """
    fit, phase, amplitude = get_phase_array(
        data_array,
        phase_coord,
        prefix=prefix,
    )
    phase_param = Parameter("phase", "Phase", "rad")
    phase = annotate(phase, phase_param, prefix=prefix)
    if add_fit:
        fit = annotate(fit, f"{prefix}_{fit.name}")
        add_many_data_arrays(target_dataset, (phase, amplitude, fit))
    else:
        add_many_data_arrays(target_dataset, (phase, amplitude))


def add_conditional_phase_amplitude(  # noqa: ANN201
    data_array: xr.DataArray,
    target_dataset: xr.Dataset,
    phase_coord: str,
    conditioned_coord: str,
    only_positive_cphase: bool | None = False,
    prefix: str = "",
    add_fit: bool = False,
):
    """Uses :func:`.get_phase_array` to calculate phase and amplitude, and then calculates conditional phase by
    comparing phase values for conditioned=True and conditioned=False. Afterwards, adds them both to the dataset.

    Args:
        data_array: array containing the data, one dimension of which is phase
        target_dataset: dataset to add the phase and amplitude arrays to
        phase_coord: name of the phase coordinate in the data array
        conditioned_coord: name of the conditioned coordinate in the data array
        only_positive_cphase: if False, cphase in range (-pi, pi]. If True, cphase in range
            [0, 2pi). False by default.
        prefix: name to attach to the phase and amplitude arrays
        add_fit: whether to also add the fit to the dataset


    Returns:
        nothing, modifies the target_dataset

    """
    fit, phase, amplitude = get_phase_array(
        data_array,
        phase_coord,
        prefix=prefix,
    )

    amplitude = amplitude.sel({conditioned_coord: True}, drop=True)
    if only_positive_cphase:
        cphase = (
            phase.sel({conditioned_coord: True}, drop=True) - phase.sel({conditioned_coord: False}, drop=True)
        ) % (2 * np.pi)
    else:
        cphase = (
            phase.sel({conditioned_coord: True}, drop=True) - phase.sel({conditioned_coord: False}, drop=True) + np.pi
        ) % (2 * np.pi) - np.pi

    cphase_param = Parameter("cphase", "Controlled Phase", "rad")
    target_amp_param = Parameter("amplitude_target", "Amplitude of Target")
    cphase = annotate(cphase, cphase_param, prefix=prefix)
    amplitude = annotate(amplitude, target_amp_param, prefix=prefix)

    if add_fit:
        fit = annotate(fit, f"{prefix}_{fit.name}")
        add_many_data_arrays(target_dataset, (cphase, amplitude, fit))
    else:
        add_many_data_arrays(target_dataset, (cphase, amplitude))
