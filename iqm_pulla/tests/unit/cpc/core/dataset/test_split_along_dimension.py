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

from iqm.cpc.core.dataset import split_along_dimension


@pytest.fixture
def data() -> xr.DataArray:
    x = np.linspace(1, 10, 10)
    y = np.linspace(1, 10, 10)
    z = np.empty((10, 10))
    for i in range(10):
        for j in range(10):
            z[i, j] = i + 1
    return xr.DataArray(z, coords=[("x", x), ("y", y)])


def test_split_along_dimension(data):
    split_data = split_along_dimension(data, "y")
    assert len(split_data) == 10
    assert len(split_data[3]) == 10
    assert split_data[3][3] == 4
