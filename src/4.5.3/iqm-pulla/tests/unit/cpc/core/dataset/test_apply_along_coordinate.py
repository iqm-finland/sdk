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
import xarray

from exa.common.data.parameter import Parameter
from iqm.cpc.core.dataset import FitResults, apply_along_coordinate

frequency = Parameter("sweeper.frequency", "Frequency", "Hz")
voltage = Parameter("bias.voltage", "Voltage", "V")
amplitude = Parameter("readout.amplitude", "Readout amplitude", "V")


def sum_func(x: np.ndarray, y: np.ndarray) -> FitResults:
    return {"test": True, "ex": x[0]}, x + y


@pytest.fixture
def data() -> xarray.DataArray:
    data_1 = np.asarray(list(range(1, 11)))
    data_2 = np.asarray(list(range(11, 13)))
    data_3 = np.asarray([list(range(21, 31)), list(range(31, 41))]).T
    voltage_data = voltage.build_data_array(data_1, dimensions=[voltage.name])
    frequency_data = frequency.build_data_array(data_2, dimensions=[frequency.name])
    amplitude_data = amplitude.build_data_array(
        data_3,
        dimensions=[voltage.name, frequency.name],
        coords={voltage.name: voltage_data, frequency.name: frequency_data},
    )
    return amplitude_data


@pytest.mark.parametrize(["coord", "other"], [(voltage.name, frequency.name), (frequency.name, voltage.name)])
def test_result_shape_content_and_attrs(data, coord, other):
    (sum_result,) = apply_along_coordinate(data, coord, sum_func)

    assert sum_result.sizes == data.sizes == {frequency.name: 2, voltage.name: 10}
    assert sum_result.shape == data.shape
    assert sum_result.attrs["stacked_dimensions"] == [other]

    expected = data + data[coord]
    np.testing.assert_array_equal(expected, sum_result)

    assert sum_result.name == "readout.amplitude_fit"
    assert sum_result.attrs["parameter"].name == "readout.amplitude_fit"
    assert sum_result.attrs["parameter"].unit == amplitude.unit
    assert sum_result.attrs["parameter"].data_type == amplitude.data_type


@pytest.mark.parametrize(["prefix", "expected_prefix"], [("", ""), ("pre", "pre_")])
@pytest.mark.parametrize(["coord", "other"], [(voltage.name, frequency.name), (frequency.name, voltage.name)])
def test_adds_returns_to_dataset(data, coord, other, prefix, expected_prefix):
    dataset = xarray.Dataset()
    sum_result, test, ex = apply_along_coordinate(
        data, coord, sum_func, returns=["test", "ex"], add_to=dataset, prefix=prefix
    )

    assert sum_result.sizes == data.sizes == {frequency.name: 2, voltage.name: 10}
    assert sum_result.shape == data.shape

    assert test.dims == ex.dims == (other,)
    np.testing.assert_array_equal(ex, data[coord].isel({coord: 0}))

    assert expected_prefix + "ex" in dataset.data_vars.keys()
    assert expected_prefix + "test" in dataset.data_vars.keys()
    expected = data + data[coord]
    np.testing.assert_array_equal(expected, sum_result)


@pytest.fixture
def dataset() -> xarray.Dataset:
    x = np.linspace(1, 2, 10)
    y = np.linspace(2, 4, 10)
    z = np.full((10, 10), 3)
    data = xarray.Dataset(
        {
            "z": (["x", "y"], z),
            "w": (["x"], y),
        },
        coords={"x": x, "y": y},
    )
    return data


def test_apply_along_coordinate_no_stacking_needed(dataset):
    (result,) = apply_along_coordinate(dataset["w"], "x", sum_func)
    assert result.data[0] == 3
    assert result.attrs["stacked_dimensions"] == []


def test_apply_along_coordinate_order_check(dataset):
    (fitted,) = apply_along_coordinate(dataset["z"], "x", sum_func)
    assert fitted.data[0][0] == 4
    assert fitted.attrs["stacked_dimensions"] == ["y"]


@pytest.mark.parametrize(
    "shape",
    [
        (6,),  # 1-d case
        (3, 4),  # typical case
        (5, 2, 4),  # rare case
        (1, 1, 1, 4, 5),  # case with singular dimensions
        (2, 2, 3, 4, 5),  # case with duplicates
    ],
)
def test_fit_values_are_in_same_order_as_in_original_for_any_chosen_fit_dimension(shape: tuple[int, ...]):
    original = xarray.DataArray(np.reshape(np.arange(0, np.prod(shape)), shape))
    for dim in original.dims:
        # the lambda function just returns the values so that the order can be checked:
        (result, extra) = apply_along_coordinate(
            original, str(dim), lambda x, y: ({"extra": 0.6}, y), returns=["extra"]
        )
        assert (result == original).all(), dim
        assert len(extra.dims) < len(original.dims), dim
        assert dim not in extra.dims, dim


def test_return_dtype_is_correct_for_complex_results():
    original = xarray.DataArray(np.ones(5) * (2 + 1j))
    (result,) = apply_along_coordinate(original, "dim_0", lambda x, y: ({}, y))
    assert result.dtype == complex
    assert (result == original).all()


def test_default_dtype_is_correct_for_real_results():
    original = xarray.DataArray(np.ones(5) * (2 + 1j))
    (result,) = apply_along_coordinate(original, "dim_0", lambda x, y: ({}, y.real))
    assert result.dtype == float
    assert (result == original.real).all()
