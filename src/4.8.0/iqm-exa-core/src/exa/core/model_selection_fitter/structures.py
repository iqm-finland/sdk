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
"""Internal request/context objects for model-selection fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import xarray as xr


@dataclass(frozen=True, slots=True)
class FitterSettings:
    """Settings for the fit method of the model selection fitter."""

    plot_fit: bool = False
    add_plot_to_dataset: bool = False
    use_outlier_detection: bool = False
    complex_action: Literal["real", "imag", "rotate_to_real"] = "rotate_to_real"
    reduction_action: Literal["mean", "median"] | None = None
    override_model_initial_guess: bool = False
    extra_parameter_penalty_factor: float = 1
    select_by_bic: bool = False


@dataclass(frozen=True, slots=True)
class FitRequest:
    """Immutable inputs for a single fit call."""

    data: xr.Dataset | xr.DataArray
    datavar: str
    independent_vars: dict[str, Any]
    take_indices: tuple[int] | None
    settings: FitterSettings
    fitter_options: dict[str, Any]
    outlier_detection_options: dict[str, Any]
    weights: xr.DataArray | None


@dataclass(frozen=True, slots=True)
class FitContext:
    """Reusable metadata derived from processed inputs."""

    processed_dataset: tuple[xr.DataArray, list[str], dict]
    coords: dict[str, np.ndarray]
    coords_keys: tuple[str, ...]
    coords_values: tuple[np.ndarray, ...]
    independent_coord_names: list[str]
    independent_coord_dims: tuple[int, ...]
    remaining_dims: list[str]
    remaining_shape: tuple[int, ...]
    data_dimensions: tuple[int, ...]
    coordinate_dict: dict
    fit_data_coords_adjusted: list[str]
