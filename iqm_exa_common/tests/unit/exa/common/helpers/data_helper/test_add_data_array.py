#  ********************************************************************************
#  Copyright (c) 2019-2020 IQM Finland Oy.
#  All rights reserved. Confidential and proprietary.
#
#  Distribution or reproduction of any information contained herein
#  is prohibited without IQM Finland Oy’s prior written permission.
#  ********************************************************************************

import numpy as np
import pytest
import xarray as xr

from exa.common.data.parameter import DataType, Parameter
from exa.common.helpers.data_helper import add_data_array

frequency = Parameter(
    name="sweeper::frequency",
    label="Frequency",
    unit="Hz",
    data_type=DataType.COMPLEX,
)
voltage = Parameter(
    name="bias::voltage",
    label="Voltage",
    unit="V",
    data_type=DataType.COMPLEX,
)


@pytest.fixture
def data_set():
    ds = xr.Dataset(
        coords={
            "x": ("x", np.arange(10, 20), {"meta": "foo"}),
            "y": ("y", np.arange(20, 30), {"meta": "bar"}),
            "z": ("z", np.arange(30, 40), {"meta": "baz"}),
        },
        data_vars={
            "v1": frequency.build_data_array(np.arange(10, 20)),
            "v2": voltage.build_data_array(np.arange(10, 20)),
        },
    )
    return ds


@pytest.fixture
def data_array():
    da = xr.DataArray(np.asarray(np.arange(10, 20)), dims="x", name="test")
    return da


def test_add_data_array_to_dataset(data_set, data_array):
    # example taken directly from https://github.com/pydata/xarray/issues/2245
    with pytest.warns(DeprecationWarning, match="`add_data_array` is deprecated since 2025-12-09"):
        ds = add_data_array(data_set, data_array)
    assert "test" in ds.data_vars
    assert ds.coords["x"].attrs["meta"] == "foo"


def test_add_data_var_attrs(data_set, data_array):
    with pytest.warns(DeprecationWarning, match="`add_data_array` is deprecated since 2025-12-09"):
        ds = add_data_array(data_set, data_array)
    assert ds.data_vars["v1"].attrs["standard_name"] == frequency.name
    assert ds.data_vars["v1"].attrs["units"] == frequency.unit
    assert ds.data_vars["v2"].attrs["standard_name"] == voltage.name
    assert ds.data_vars["v2"].attrs["units"] == voltage.unit


def test_assigns_custom_name(data_set, data_array):
    with pytest.warns(DeprecationWarning, match="`add_data_array` is deprecated since 2025-12-09"):
        ds = add_data_array(data_set, data_array, "custom_name")
    assert (ds["custom_name"] == data_array).all()


def test_data_variable_attribute_overwrite(data_set):
    # Create two DataArrays with the same name but different attributes
    da1 = xr.DataArray([1, 2, 3], attrs={"param1": "a", "param2": "b"})
    da2 = xr.DataArray([4, 5, 6], attrs={"param2": "c", "param3": "d"})

    # Add da1 to the dataset
    with pytest.warns(DeprecationWarning, match="`add_data_array` is deprecated since 2025-12-09"):
        add_data_array(data_set, da1, "my_data_array")

    # Add da2 to the dataset, overwriting the existing DataArray
    with pytest.warns(DeprecationWarning, match="`add_data_array` is deprecated since 2025-12-09"):
        add_data_array(data_set, da2, "my_data_array")

    # Check that the attributes were overwritten correctly
    assert data_set["my_data_array"].attrs == da2.attrs


def test_data_variable_attribute_not_lost_when_adding_different_array(data_set):
    # Create two DataArrays with the same name but different attributes
    da1 = xr.DataArray([1, 2, 3], attrs={"param1": "a", "param2": "b"})
    da2 = xr.DataArray([4, 5, 6], attrs={"param2": "c", "param3": "d"})

    with pytest.warns(DeprecationWarning, match="`add_data_array` is deprecated since 2025-12-09"):
        add_data_array(data_set, da1, "my_data_array")
        add_data_array(data_set, da2, "my_data_array_2")

    assert "my_data_array" in data_set.data_vars
    assert "my_data_array_2" in data_set.data_vars
    assert data_set.coords["x"].attrs["meta"] == "foo"

    # Check that the attributes are correct
    assert data_set["my_data_array"].attrs == da1.attrs
    assert data_set["my_data_array_2"].attrs == da2.attrs


def test_add_data_array_loop_identical_to_update_dataset(data_set):
    ds1 = xr.Dataset().update(data_set.coords)
    ds2 = xr.Dataset().update(data_set.coords)
    ds1.update(data_set)
    for data_var in data_set.data_vars:
        with pytest.warns(DeprecationWarning, match="`add_data_array` is deprecated since 2025-12-09"):
            add_data_array(ds2, data_set[data_var])
    xr.testing.assert_identical(ds1, ds2)
