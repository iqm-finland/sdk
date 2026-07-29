#  ********************************************************************************
#  Copyright (c) 2019-2024 IQM Finland Oy.
#  All rights reserved. Confidential and proprietary.
#
#  Distribution or reproduction of any information contained herein
#  is prohibited without IQM Finland Oy’s prior written permission.
#  ********************************************************************************
import numpy as np

from exa.common.data.parameter import CollectionType, DataType, Parameter


def test_list_with_valid_tuple_data_type() -> None:
    parameter = Parameter("foo", "Foo", "UNIT", (DataType.FLOAT, DataType.STRING), CollectionType.LIST)
    assert parameter.validate(["test", "test"]) is True


def test_list_with_invalid_tuple_data_type() -> None:
    parameter = Parameter("foo", "Foo", "UNIT", (DataType.FLOAT, DataType.COMPLEX), CollectionType.LIST)
    assert parameter.validate([True, 50]) is False


def test_list_as_invalid_ndarray() -> None:
    parameter = Parameter("foo", "Foo", "UNIT", DataType.INT, CollectionType.LIST)
    assert parameter.validate(np.zeros((2, 2))) is False


def test_list_as_invalid_scalar() -> None:
    parameter = Parameter("foo", "Foo", "UNIT", DataType.FLOAT, CollectionType.LIST)
    assert parameter.validate(0.0) is False


def test_list_as_invalid_nested_list() -> None:
    parameter = Parameter("foo", "Foo", "UNIT", DataType.FLOAT, CollectionType.LIST)
    assert parameter.validate([[0.0, 0.0], [0.0, 0.0]]) is False


def test_list_with_invalid_values() -> None:
    parameter = Parameter("foo", "Foo", "UNIT", DataType.INT, CollectionType.LIST)
    assert parameter.validate([3, 5 + 5j, True]) is False


def test_list_with_valid_numbers() -> None:
    parameter = Parameter("foo", "Foo", "UNIT", DataType.INT, CollectionType.LIST)
    assert parameter.validate([1, 2, 3]) is True


def test_list_with_valid_complex_numbers() -> None:
    parameter = Parameter("foo", "Foo", "UNIT", DataType.COMPLEX, CollectionType.LIST)
    assert parameter.validate([0.0 + 0.0j, 12.0 + 4.0j]) is True


def test_list_with_valid_strings() -> None:
    parameter = Parameter("foo", "Foo", "UNIT", DataType.STRING, CollectionType.LIST)
    assert parameter.validate(["text", "text"]) is True


def test_list_with_valid_booleans() -> None:
    parameter = Parameter("foo", "Foo", "UNIT", DataType.BOOLEAN, CollectionType.LIST)
    assert parameter.validate([False, True]) is True


def test_array_with_valid_tuple_data_type() -> None:
    parameter = Parameter("foo", "Foo", "UNIT", (DataType.INT, DataType.BOOLEAN), CollectionType.NDARRAY)
    assert parameter.validate(np.full((2, 2), False)) is True


def test_array_with_invalid_tuple_data_type() -> None:
    parameter = Parameter("foo", "Foo", "UNIT", (DataType.INT, DataType.STRING), CollectionType.NDARRAY)
    assert parameter.validate(np.full((2, 2), False)) is False


def test_array_as_invalid_scalar() -> None:
    parameter = Parameter("foo", "Foo", "UNIT", DataType.FLOAT, CollectionType.NDARRAY)
    assert parameter.validate(0.0) is False


def test_array_as_invalid_list() -> None:
    parameter = Parameter("foo", "Foo", "UNIT", DataType.FLOAT, CollectionType.NDARRAY)
    assert parameter.validate([0.0, 0.0]) is False


def test_array_with_invalid_values() -> None:
    parameter = Parameter("foo", "Foo", "UNIT", DataType.INT, CollectionType.NDARRAY)
    assert parameter.validate(np.full((2, 2), "text")) is False


def test_array_with_valid_numbers() -> None:
    parameter = Parameter("foo", "Foo", "UNIT", DataType.INT, CollectionType.NDARRAY)
    assert parameter.validate(np.zeros((2, 2))) is True


def test_array_with_valid_complex_numbers() -> None:
    parameter = Parameter("foo", "Foo", "UNIT", DataType.COMPLEX, CollectionType.NDARRAY)
    assert parameter.validate(np.full((2, 2), 0.0 + 0.0j)) is True


def test_array_with_valid_strings() -> None:
    parameter = Parameter("foo", "Foo", "UNIT", DataType.STRING, CollectionType.NDARRAY)
    assert parameter.validate(np.full((2, 2), "text")) is True


def test_array_with_valid_booleans() -> None:
    parameter = Parameter("foo", "Foo", "UNIT", DataType.BOOLEAN, CollectionType.NDARRAY)
    assert parameter.validate(np.full((2, 2), False)) is True


def test_scalar_with_valid_tuple_data_type() -> None:
    parameter = Parameter("foo", "Foo", "UNIT", (DataType.INT, DataType.STRING), CollectionType.SCALAR)
    assert parameter.validate(50) is True


def test_scalar_with_invalid_tuple_data_type() -> None:
    parameter = Parameter("foo", "Foo", "UNIT", (DataType.INT, DataType.BOOLEAN), CollectionType.SCALAR)
    assert parameter.validate("test") is False


def test_scalar_as_invalid_list() -> None:
    parameter = Parameter("foo", "Foo", "UNIT", DataType.FLOAT)
    assert parameter.validate([0.0, 0.0]) is False


def test_scalar_as_invalid_ndarray() -> None:
    parameter = Parameter("foo", "Foo", "UNIT", DataType.INT)
    assert parameter.validate(np.zeros((2, 2))) is False


def test_scalar_with_invalid_value() -> None:
    parameter = Parameter("foo", "Foo", "UNIT", DataType.INT)
    assert parameter.validate("test") is False


def test_scalar_with_valid_number() -> None:
    parameter = Parameter("foo", "Foo", "UNIT", DataType.FLOAT)
    assert parameter.validate(0.0) is True


def test_scalar_with_valid_complex() -> None:
    parameter = Parameter("foo", "Foo", "UNIT", DataType.COMPLEX)
    assert parameter.validate(0.0 + 0.0j) is True


def test_scalar_with_valid_string() -> None:
    parameter = Parameter("foo", "Foo", "UNIT", DataType.STRING)
    assert parameter.validate("test") is True


def test_scalar_with_valid_boolean() -> None:
    parameter = Parameter("foo", "Foo", "UNIT", DataType.BOOLEAN)
    assert parameter.validate(False) is True
