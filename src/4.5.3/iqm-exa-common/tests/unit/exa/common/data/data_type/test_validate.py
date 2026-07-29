#  ********************************************************************************
#  Copyright (c) 2019-2020 IQM Finland Oy.
#  All rights reserved. Confidential and proprietary.
#
#  Distribution or reproduction of any information contained herein
#  is prohibited without IQM Finland Oy’s prior written permission.
#  ********************************************************************************
import numpy as np

from exa.common.data.parameter import DataType


def test_valid_none_value():
    data_type = DataType.ANYTHING
    assert data_type.validate(None) is True


def test_valid_any_value():
    data_type = DataType.ANYTHING
    assert data_type.validate(54) is True


def test_valid_np_generic():
    data_type = DataType.INT
    assert data_type.validate(np.int16(3).dtype.type(3)) is True


def test_valid_int_number():
    data_type = DataType.INT
    assert data_type.validate(54) is True


def test_valid_float_number():
    data_type = DataType.FLOAT
    assert data_type.validate(54.4) is True


def test_valid_int_number_as_complex():
    data_type = DataType.COMPLEX
    assert data_type.validate(54) is True


def test_valid_float_number_as_complex():
    data_type = DataType.COMPLEX
    assert data_type.validate(54.4) is True


def test_valid_complex_number():
    data_type = DataType.COMPLEX
    assert data_type.validate(5 + 5j) is True


def test_valid_string():
    data_type = DataType.STRING
    assert data_type.validate("test") is True


def test_valid_boolean():
    data_type = DataType.BOOLEAN
    assert data_type.validate(False) is True


def test_invalid_number_as_string():
    data_type = DataType.STRING
    assert data_type.validate(54) is False


def test_invalid_number_as_boolean():
    data_type = DataType.BOOLEAN
    assert data_type.validate(54) is False


def test_invalid_complex_number_as_number():
    data_type = DataType.INT
    assert data_type.validate(5 + 5j) is False


def test_invalid_complex_number_as_string():
    data_type = DataType.STRING
    assert data_type.validate(5 + 5j) is False


def test_invalid_complex_number_as_boolean():
    data_type = DataType.BOOLEAN
    assert data_type.validate(5 + 5j) is False


def test_invalid_string_as_number():
    data_type = DataType.INT
    assert data_type.validate("test") is False


def test_invalid_string_as_complex():
    data_type = DataType.COMPLEX
    assert data_type.validate("test") is False


def test_invalid_string_as_boolean():
    data_type = DataType.BOOLEAN
    assert data_type.validate("test") is False


def test_invalid_boolean_as_number():
    data_type = DataType.INT
    assert data_type.validate(False) is False


def test_invalid_boolean_as_complex():
    data_type = DataType.COMPLEX
    assert data_type.validate(False) is False


def test_invalid_boolean_as_string():
    data_type = DataType.STRING
    assert data_type.validate(False) is False


def test_invalid_np_generic():
    data_type = DataType.STRING
    assert data_type.validate(np.int16(3).dtype.type(3)) is False
