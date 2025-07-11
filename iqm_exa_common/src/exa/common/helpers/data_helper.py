# Copyright 2024 IQM
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


from collections.abc import Hashable

import xarray as xr

"""Helper methods for data manipulation.
"""


def add_data_array(ds: xr.Dataset, da: xr.DataArray, name: Hashable | None = None) -> xr.Dataset:
    """Add data array `da` to dataset `ds`.

    Unlike the default xarray command, preserves metadata of the dataset.

    Args:
        ds: Dataset to add to.
        da: DataArray to add
        name: name under which `da` can be accessed inside `ds`.
            By default, uses the `name` property of `da`.

    Returns:
        The updated dataset.

    """
    if name is None:
        if da.name is not None:
            name = da.name  # type: ignore[assignment]
        else:
            raise ValueError("No name was given to the dataArray.")
    # Attributes of Dataset coordinates are dropped/replaced when adding a DataArray
    # https://github.com/pydata/xarray/issues/2245
    # So we need to temporarily store all coord attrs, and then add them back
    attributes = {}
    for key in ds.coords:
        attributes[key] = ds.coords[key].attrs
    for key in ds.data_vars:
        attributes[key] = ds.data_vars[key].attrs
    ds[name] = da
    for key in ds.coords:
        if attributes.get(key):
            ds.coords[key].attrs = attributes.get(key)  # type:ignore[assignment]
    for key in ds.data_vars:
        if attributes.get(key):
            ds.data_vars[key].attrs = attributes[key]
    return ds
