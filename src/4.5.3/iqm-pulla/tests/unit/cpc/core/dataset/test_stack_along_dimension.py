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

from iqm.cpc.core.dataset import stack_along_dimension


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


def test_stack_along_dimension(dataset):
    stack = stack_along_dimension(dataset.z, dimension="x")
    assert "stack" in stack.dims
    assert "x" in stack.dims


def test_stack_along_dimension_single_fails(dataset):
    with pytest.raises(RuntimeError, match="Xarray does not allow stacking data along x"):
        stack_along_dimension(dataset.w, dimension="x")


def test_stack_along_dimension_no_dimension(dataset):
    for data in [dataset.z, dataset.w]:
        stack = stack_along_dimension(data, stack="some_name")
        assert stack.dims == ("some_name",)
