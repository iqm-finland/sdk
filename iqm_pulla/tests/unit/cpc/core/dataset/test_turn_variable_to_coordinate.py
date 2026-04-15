# Copyright 2024-2025 IQM
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest
import xarray as xr

from iqm.cpc.core.dataset import turn_variable_to_coordinate


@pytest.fixture
def dataset() -> xr.Dataset:
    x = np.full((10,), 1)
    y = np.full((10,), 2)
    z = np.full((10, 10), 3)
    return xr.Dataset(
        {
            "z": (["x", "y"], z),
            "w": (["x"], y),
        },
        coords={"x": x, "y": y},
    )


def test_1d_variable_to_coordinate_conversion(dataset):
    turn_variable_to_coordinate(dataset, variable_name_to_convert="w")
    assert "w" in dataset.coords
    assert "w" not in dataset.data_vars


def test_2d_variable_to_coordinate_conversion(dataset):
    turn_variable_to_coordinate(dataset, variable_name_to_convert="z")
    assert "z" in dataset.coords
    assert "z" not in dataset.data_vars
