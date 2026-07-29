#  ********************************************************************************
#  Copyright (c) 2019-2024 IQM Finland Oy.
#  All rights reserved. Confidential and proprietary.
#
#  Distribution or reproduction of any information contained herein
#  is prohibited without IQM Finland Oy’s prior written permission.
#  ********************************************************************************
import numpy as np
import pytest
import xarray as xr

from exa.common.data.parameter import DataType, Parameter


def test_simple_usage() -> None:
    short_name = "readout::sweeper::frequency"
    long_name = "Frequency"
    units = "Hz"
    data_dimensions = [6, 8, 2]
    parameter = Parameter(short_name, label=long_name, unit=units)
    some_data = parameter.build_data_array(np.asarray(data_dimensions))
    assert short_name in some_data.dims
    assert long_name == some_data.long_name
    assert short_name == some_data.standard_name
    assert units == some_data.units
    assert short_name == some_data.attrs["standard_name"]
    assert long_name == some_data.attrs["long_name"]
    assert units == some_data.attrs["units"]
    assert np.min(data_dimensions) == some_data.values.min()
    assert np.max(data_dimensions) == some_data.values.max()
    assert some_data.values.size == len(data_dimensions)


def test_empty_data_array():
    parameter = Parameter("test", "Test", "Smoots", DataType.INT)
    data_array = parameter.build_data_array(np.zeros(8))
    assert isinstance(data_array, xr.DataArray)
    assert np.allclose(data_array.data, np.zeros(8))
    assert data_array.name == "test"
    assert data_array.dims == ("test",)
    assert data_array.coords == {}
    assert data_array.attrs["parameter"] is parameter
    assert data_array.attrs["units"] == "Smoots"


def test_valid_one_dimensional_data_array():
    parameter = Parameter("test", "Test", "", DataType.INT)
    data_array = parameter.build_data_array(np.zeros(8), dimensions=["a"])
    assert data_array.dims == ("a",)


def test_integer_dimension_type():
    parameter = Parameter("test", "Test", "", DataType.INT)
    parameter.build_data_array(np.zeros(8), dimensions=[1])


def test_invalid_dimension_type():
    parameter = Parameter("test", "Test", "", DataType.INT)
    with pytest.raises(TypeError):
        parameter.build_data_array(np.zeros(8), dimensions=[["foo"]])


def test_invalid_dimension_amount():
    parameter = Parameter("test", "Test", "", DataType.INT)
    with pytest.raises(ValueError):
        parameter.build_data_array(np.zeros(8), dimensions=["a", "b"])


def test_valid_higher_dimensions():
    parameter = Parameter("test", "Test", "", DataType.INT)
    data_array = parameter.build_data_array(np.zeros((5, 6, 8)))
    assert data_array.dims == ("test_0", "test_1", "test_2")
    data_array = parameter.build_data_array(np.zeros((5, 6, 8)), dimensions=["a", "b", "c"])
    assert data_array.dims == ("a", "b", "c")


def test_invalid_higher_dimensions():
    parameter = Parameter("test", "Test", "", DataType.INT)
    with pytest.raises(ValueError):
        parameter.build_data_array(np.zeros((5, 6, 8)), dimensions=["a"])
    with pytest.raises(ValueError):
        parameter.build_data_array(np.zeros((5, 6, 8)), dimensions=["a", "b"])


def test_valid_0_dimensional_data_array():
    parameter = Parameter("test", "Test", "", DataType.INT)
    data_array = parameter.build_data_array(np.array(0))
    assert data_array.dims == ()


def test_data_type_coords():
    parameter = Parameter("test", "Test", "", DataType.INT)
    data_array = parameter.build_data_array(np.zeros(8), coords={"a": 1, "b": 4})
    assert data_array.coords == {"a": 1, "b": 4}


def test_valid_data_type_metadata():
    parameter = Parameter("test", "Test", "", DataType.INT)
    data_array = parameter.build_data_array(np.zeros(8), metadata={"a": 1, "b": 4})
    assert data_array.attrs["a"] is 1
    assert data_array.attrs["b"] is 4
    assert data_array.attrs["parameter"] is parameter


def test_duplicate_metadata():
    parameter = Parameter("test", "Test", "", DataType.INT)
    with pytest.raises(ValueError):
        parameter.build_data_array(np.zeros(8), metadata={"units": 1, "b": 3, "parameter": 4})
