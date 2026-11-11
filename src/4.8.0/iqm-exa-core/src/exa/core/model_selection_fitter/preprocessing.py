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
"""Data preprocessing utilities for the model-selection fitter."""

from collections.abc import Iterable
import copy
from itertools import zip_longest
import logging
from typing import Any, Literal

import lmfit
import numpy as np
import xarray as xr

from .structures import FitContext
from .utils import (
    auto_rotate,
    independent_vars_to_list_of_coords,
    outlier_detection,
)

logger = logging.getLogger(__name__)


def _process_dataset(
    data: xr.Dataset | xr.DataArray,
    datavar: str = "",
    take_indices: Iterable | None = None,
    reduction_action: Literal["mean", "median"] | None = None,
    complex_action: Literal["real", "imag", "rotate_to_real"] | None = None,
    **independent_vars,
) -> tuple[xr.DataArray, list, dict]:
    """Processes the given `xarray.Dataset` or `xarray.DataArray`.

    Args:
        data: Dataset with the data to be fitted. Can be also DataArray, in which case datavar is ignored.
        datavar: Data variable name. Defaults to None.
        take_indices: The indices to take. Defaults to None.
        reduction_action: The function for reduce. Can be a Literal "mean" or "median". Defaults to ``None``.
        complex_action: Literal ``"real"`` , ``"imag"`` , ``"rotate_to_real"`` . Defaults to ``None``.
        independent_vars: The independent variables.

    Raises:
        ValueError: If the reduction action is not implemented

    Returns:
        Processed data, remaining coords after taking, coords for mapping as a tuple

    """
    data_array: xr.DataArray
    if isinstance(data, xr.Dataset):
        if not datavar:
            logger.debug("Datavar is None, trying to infer its value.")
            datavar = list(data.data_vars)[0]
            logger_message = f"Found datavar={datavar} from dataset."
            logger.debug(logger_message)
        logger_message = f"Using datavar={datavar}"
        logger.debug(logger_message)
        data_array = data[datavar]
    else:
        data_array = data
    coords = independent_vars_to_list_of_coords(independent_vars)

    set_coords = set(coords)

    missing_coords = [x for x in set_coords if x not in data_array.dims]
    logger.debug("Missing coordinates %s", missing_coords)

    found_in_coords = [x for x in missing_coords if x in data_array.coords]
    logger.debug("Searching missing coordinate from .coords. Found %s", found_in_coords)

    # WARNING: this will not work if fitting two variables from a tuple of sweeps
    data_array, coord_to_main_mapping = _swap_coordinates(data_array=data_array, found_in_coords=found_in_coords)

    # This keeps remaining dims
    remaining_dims = [x for x in data_array.dims if x not in set_coords]
    data_array = data_array.transpose(*coords, ...)

    if take_indices is None:
        take_indices = [None]
    elif not isinstance(take_indices, Iterable):
        take_indices = np.unravel_index(take_indices, data_array.shape)

    slice_dict: dict[str, int | slice] = {str(x): slice(None) for x in coords}
    remaining_dims_after_taking = copy.copy(remaining_dims)

    for coord, idx in zip_longest(remaining_dims, take_indices):
        if idx is None or coord is None:
            continue
        remaining_dims_after_taking.remove(coord)
        slice_dict[str(coord)] = int(idx)  # type: ignore[arg-type]

    data_array = data_array[slice_dict]

    if complex_action == "real":
        data_array = data_array.real
    elif complex_action == "imag":
        data_array = data_array.imag
    elif complex_action == "rotate_to_real":
        data_array = data_array.copy(data=auto_rotate(data_array.values))

    if reduction_action is not None:
        if reduction_action == "mean":
            data_array = data_array.mean(dim=remaining_dims_after_taking)
        elif reduction_action == "median":
            data_array = data_array.median(dim=remaining_dims_after_taking)
        else:
            raise ValueError(f"Requested reduction action '{reduction_action}' is not implemented.")
        remaining_dims_after_taking = []

    return data_array, remaining_dims_after_taking, coord_to_main_mapping


def _swap_coordinates(data_array: xr.DataArray, found_in_coords: list[str]) -> tuple[xr.DataArray, dict]:
    """Swap coordinates if independent variables are only in the coordinates of the data array."""
    coord_to_main_mapping = {}
    data_array_coords = data_array.coords
    for found in found_in_coords:
        target = data_array_coords[found].dims[0]
        data_array = data_array.swap_dims({target: found})
        coord_to_main_mapping[found] = target
    return data_array, coord_to_main_mapping


def _detect_outliers_sliced_data(
    sliced_data: xr.DataArray,
    sliced_weights: xr.DataArray | None,
    model: lmfit.Model,
    independent_vars_current: dict,
    fitter_options: dict,
    override_model_initial_guess: bool,
    outlier_detection_options: dict,
) -> tuple[xr.DataArray, xr.DataArray | None, dict]:
    """Detect outliers for a single slice.

    Notes:
        This function must not mutate its input independent_vars_current, because callers
        may reuse the same mapping across many slices for performance.

    """
    outlier_inds = outlier_detection(
        sliced_data,
        model,
        independent_vars=independent_vars_current,
        fitter_options=fitter_options,
        override_model_initial_guess=override_model_initial_guess,
        **outlier_detection_options,
    )
    independent_vars_filtered = {
        independent_value_name: independent_values[np.invert(outlier_inds)]
        for independent_value_name, independent_values in independent_vars_current.items()
    }
    sliced_data = sliced_data[np.invert(outlier_inds)]
    sliced_weights = sliced_weights[np.invert(outlier_inds)] if sliced_weights is not None else None
    return sliced_data, sliced_weights, independent_vars_filtered


def _construct_coordinates(fit_data_coords: list, coord_to_main_mapping: dict, data: xr.DataArray) -> tuple[dict, list]:
    coordinate_dict = {}
    fit_data_coords_adjusted = []

    for coord_string in fit_data_coords:
        if coord_string in coord_to_main_mapping:
            coordinate_dict[coord_to_main_mapping[coord_string]] = (
                coord_to_main_mapping[coord_string],
                data.coords[coord_to_main_mapping[coord_string]].data,
            )
            coordinate_dict[coord_string] = (
                coord_to_main_mapping[coord_string],
                data.coords[coord_string].data,
            )
            fit_data_coords_adjusted.append(coord_to_main_mapping[coord_string])
        else:
            coordinate_dict[coord_string] = (
                coord_string,
                data.coords[coord_string].data,
            )
            fit_data_coords_adjusted.append(coord_string)

    return coordinate_dict, fit_data_coords_adjusted


def prepare_fit_context(
    processed_dataset: tuple[xr.DataArray, list[str], dict],
    independent_vars: dict[str, Any],
) -> FitContext:
    """Compute and bundle shared fit metadata, producing a FitContext.

    Args:
        processed_dataset: Output of _process_dataset: (xr.DataArray, list of remaining dims, coord_to_main_mapping dict).
        independent_vars: Mapping from model variable names to dataset coordinate names.

    Returns:
        FitContext: Fully assembled fit context holding all metadata for model fit.

    """  # noqa: E501
    data, remaining_dims, coord_to_main_mapping = processed_dataset

    coords = {key: np.asarray(data.coords[value]) for key, value in independent_vars.items()}
    independent_coord_names = list(independent_vars.values())

    remaining_shape = tuple(len(data.coords[dim]) for dim in remaining_dims)
    fit_data_dims = remaining_dims + independent_coord_names
    data_dimensions = tuple(len(data.coords[dim]) for dim in fit_data_dims)
    independent_coord_dims = tuple(len(coord) for coord in coords.values())

    coordinate_dict, fit_data_coords_adjusted = _construct_coordinates(
        fit_data_coords=fit_data_dims,
        coord_to_main_mapping=coord_to_main_mapping,
        data=data,
    )
    remaining_coords = {
        coord: (dim, data.coords[coord].values)
        for dim in remaining_dims
        for coord in data.coords[dim].coords
        if coord not in remaining_dims and data.coords[dim].coords[coord].variable.ndim and coord != "models"
    }
    coordinate_dict.update(remaining_coords)

    coords_keys = tuple(coords.keys())
    coords_values = tuple(coords[key] for key in coords_keys)

    return FitContext(
        processed_dataset=processed_dataset,
        coords=coords,
        coords_keys=coords_keys,
        coords_values=coords_values,
        independent_coord_names=independent_coord_names,
        independent_coord_dims=independent_coord_dims,
        remaining_dims=remaining_dims,
        remaining_shape=remaining_shape,
        data_dimensions=data_dimensions,
        coordinate_dict=coordinate_dict,
        fit_data_coords_adjusted=fit_data_coords_adjusted,
    )

    remaining_coords = {
        coord: (dim, data.coords[coord].values)
        for dim in remaining_dims
        for coord in data.coords[dim].coords
        if coord not in remaining_dims and data.coords[dim].coords[coord].variable.ndim and coord != "models"
    }
    coordinate_dict.update(remaining_coords)

    coords_keys = tuple(coords.keys())
    coords_values = tuple(coords[key] for key in coords_keys)

    return (
        remaining_dims,
        coords,
        independent_coord_names,
        remaining_shape,
        data_dimensions,
        coordinate_dict,
        fit_data_coords_adjusted,
        coords_keys,
        coords_values,
        independent_coord_dims,
    )
