#  ********************************************************************************
#  Copyright (c) 2019-2020 IQM Finland Oy.
#  All rights reserved. Confidential and proprietary.
#
#  Distribution or reproduction of any information contained herein
#  is prohibited without IQM Finland Oy’s prior written permission.
#  ********************************************************************************
import pytest

from exa.common.data.parameter import DataType


def test_cast_none_to_anything():
    data_type = DataType.ANYTHING
    assert data_type.cast(None) is None
    assert data_type.cast("None") is None


def test_cast_none_to_int():
    data_type = DataType.INT
    assert data_type.cast(None) is None


def test_cast_none_to_float():
    data_type = DataType.FLOAT
    assert data_type.cast(None) is None


def test_cast_none_to_complex():
    data_type = DataType.COMPLEX
    assert data_type.cast(None) is None


def test_cast_none_to_string():
    data_type = DataType.STRING
    assert data_type.cast(None) is None


def test_cast_none_to_boolean():
    data_type = DataType.BOOLEAN
    assert data_type.cast(None) is None


def test_cast_int_to_anything():
    data_type = DataType.ANYTHING
    cast = data_type.cast("42")
    assert cast == 42
    assert isinstance(cast, int)


def test_cast_int_to_int():
    data_type = DataType.INT
    cast = data_type.cast("42")
    assert cast == 42
    assert isinstance(cast, int)


def test_cast_int_to_float():
    data_type = DataType.FLOAT
    cast = data_type.cast("42")
    assert cast == 42.0
    assert isinstance(cast, float)


def test_cast_int_to_complex():
    data_type = DataType.COMPLEX
    cast = data_type.cast("42")
    assert cast == 42 + 0j
    assert isinstance(cast, complex)


def test_cast_int_to_string():
    data_type = DataType.STRING
    cast = data_type.cast("42")
    assert cast == "42"
    assert isinstance(cast, str)


def test_cast_valid_int_to_boolean():
    data_type = DataType.BOOLEAN
    cast = data_type.cast("1")
    assert cast is True
    assert isinstance(cast, bool)
    cast = data_type.cast("0")
    assert cast is False
    assert isinstance(cast, bool)


def test_cast_invalid_int_to_boolean():
    data_type = DataType.BOOLEAN
    with pytest.raises(TypeError):
        data_type.cast("42")


def test_cast_int_float_to_anything():
    data_type = DataType.ANYTHING
    cast = data_type.cast("42.0")
    assert cast == 42
    assert isinstance(cast, float)


def test_cast_int_float_to_int():
    data_type = DataType.INT
    with pytest.raises(ValueError, match="invalid literal for int"):
        data_type.cast("42.0")


def test_cast_int_float_to_complex():
    data_type = DataType.COMPLEX
    cast = data_type.cast("42.0")
    assert cast == 42 + 0j
    assert isinstance(cast, complex)


def test_cast_int_float_to_string():
    data_type = DataType.STRING
    cast = data_type.cast("42.0")
    assert cast == "42.0"
    assert isinstance(cast, str)


def test_cast_invalid_float_to_bool():
    data_type = DataType.BOOLEAN
    with pytest.raises(TypeError):
        data_type.cast("0.0")


def test_cast_float_to_anything():
    data_type = DataType.ANYTHING
    cast = data_type.cast("42.3")
    assert cast == 42.3
    assert isinstance(cast, float)


def test_cast_float_to_float():
    data_type = DataType.FLOAT
    cast = data_type.cast("42.3")
    assert cast == 42.3
    assert isinstance(cast, float)


def test_cast_float_to_complex():
    data_type = DataType.COMPLEX
    cast = data_type.cast("42.3")
    assert cast == 42.3
    assert isinstance(cast, complex)


def test_cast_float_to_string():
    data_type = DataType.STRING
    cast = data_type.cast("42.3")
    assert cast == "42.3"
    assert isinstance(cast, str)


def test_cast_complex_to_anything():
    data_type = DataType.ANYTHING
    cast = data_type.cast("42.3 +      34j")
    assert cast == 42.3 + 34j
    assert isinstance(cast, complex)


def test_cast_complex_to_int():
    data_type = DataType.INT
    with pytest.raises(ValueError):
        data_type.cast("42.3 +      34j")


def test_cast_complex_to_float():
    data_type = DataType.FLOAT
    with pytest.raises(ValueError):
        data_type.cast("42.3 +      34j")


def test_cast_complex_to_complex():
    data_type = DataType.COMPLEX
    cast = data_type.cast("42.3 +      34j")
    assert cast == 42.3 + 34j
    assert isinstance(cast, complex)


def test_cast_complex_to_string():
    data_type = DataType.STRING
    cast = data_type.cast("42.3 +      34j")
    assert cast == "42.3 +      34j"
    assert isinstance(cast, str)


def test_cast_complex_to_bool():
    data_type = DataType.BOOLEAN
    with pytest.raises(TypeError):
        data_type.cast("42.3 +      34")


def test_cast_complex_real_to_anything():
    data_type = DataType.ANYTHING
    cast = data_type.cast("42.3 +      0j")
    assert cast == 42.3
    assert isinstance(cast, complex)


def test_cast_complex_real_to_int():
    data_type = DataType.INT
    with pytest.raises(ValueError):
        data_type.cast("42.3 +      0j")


def test_cast_complex_real_to_float():
    data_type = DataType.FLOAT
    with pytest.raises(ValueError):
        data_type.cast("42.3 +      0j")


def test_cast_complex_real_to_complex():
    data_type = DataType.COMPLEX
    cast = data_type.cast("42.3 +      0j")
    assert cast == 42.3
    assert isinstance(cast, complex)


def test_cast_complex_real_to_string():
    data_type = DataType.STRING
    cast = data_type.cast("42.3 +      0j")
    assert cast == "42.3 +      0j"
    assert isinstance(cast, str)


def test_cast_complex_real_to_bool():
    data_type = DataType.BOOLEAN
    with pytest.raises(TypeError):
        data_type.cast("42.3 +      0j")


def test_cast_string_to_anything():
    data_type = DataType.ANYTHING
    cast = data_type.cast("some random string")
    assert cast == "some random string"
    assert isinstance(cast, str)
    cast = data_type.cast("1 + 3")
    assert cast == "1 + 3"
    assert isinstance(cast, str)


def test_cast_string_to_int():
    data_type = DataType.INT
    with pytest.raises(ValueError):
        data_type.cast("some random string")


def test_cast_string_to_float():
    data_type = DataType.FLOAT
    with pytest.raises(ValueError):
        data_type.cast("some random string")


def test_cast_string_to_complex():
    data_type = DataType.COMPLEX
    with pytest.raises(ValueError):
        data_type.cast("some random string")


def test_cast_string_to_string():
    data_type = DataType.STRING
    cast = data_type.cast("some random string")
    assert cast == "some random string"
    assert isinstance(cast, str)


def test_cast_string_to_bool():
    data_type = DataType.BOOLEAN
    with pytest.raises(TypeError):
        data_type.cast("some random string")


def test_cast_boolean_to_anything():
    data_type = DataType.ANYTHING
    cast = data_type.cast("True")
    assert cast is True
    assert isinstance(cast, bool)
    cast = data_type.cast("False")
    assert cast is False
    assert isinstance(cast, bool)


def test_cast_boolean_to_int():
    data_type = DataType.INT
    with pytest.raises(ValueError):
        data_type.cast("True")
    with pytest.raises(ValueError):
        data_type.cast("False")


def test_cast_boolean_to_float():
    data_type = DataType.FLOAT
    with pytest.raises(ValueError):
        data_type.cast("True")
    with pytest.raises(ValueError):
        data_type.cast("False")


def test_cast_boolean_to_complex():
    data_type = DataType.COMPLEX
    with pytest.raises(ValueError):
        data_type.cast("True")


def test_cast_boolean_to_string():
    data_type = DataType.STRING
    cast = data_type.cast("True")
    assert cast == "True"
    assert isinstance(cast, str)
    cast = data_type.cast("False")
    assert cast is "False"
    assert isinstance(cast, str)


def test_cast_python_like_boolean_to_bool():
    data_type = DataType.BOOLEAN
    cast = data_type.cast("True")
    assert cast is True
    assert isinstance(cast, bool)
    cast = data_type.cast("False")
    assert cast is False
    assert isinstance(cast, bool)


def test_cast_lower_case_boolean_to_bool():
    data_type = DataType.BOOLEAN
    cast = data_type.cast("true")
    assert cast is True
    assert isinstance(cast, bool)
    cast = data_type.cast("false")
    assert cast is False
    assert isinstance(cast, bool)


def test_list_to_anything():
    data_type = DataType.ANYTHING
    test_list = ["0", "1", "1.0", "1.5", "3 + 3j", "string", "{'a': 4}", "False", "True"]
    expected_list = [0, 1, 1.0, 1.5, 3 + 3j, "string", {"a": 4}, False, True]
    cast = data_type.cast(test_list)
    assert cast == expected_list
    for x, y in zip(expected_list, cast):
        assert type(x) is type(y)


def test_list_to_int():
    data_type = DataType.INT
    test_list = ["0", "1", "2"]
    expected_list = [0, 1, 2]
    cast = data_type.cast(test_list)
    assert cast == expected_list
    for x, y in zip(expected_list, cast):
        assert type(x) is type(y)


def test_list_to_float():
    data_type = DataType.FLOAT
    test_list = ["0", "1", "1.0", "1.5"]
    expected_list = [0.0, 1.0, 1.0, 1.5]
    cast = data_type.cast(test_list)
    assert cast == expected_list
    for x, y in zip(expected_list, cast):
        assert type(x) is type(y)


def test_list_to_complex():
    data_type = DataType.COMPLEX
    test_list = ["0", "1", "1.0", "1.5", "3 + 3j"]
    expected_list = [0 + 0j, 1 + 0j, 1 + 0j, 1.5 + 0j, 3 + 3j]
    cast = data_type.cast(test_list)
    assert cast == expected_list
    for x, y in zip(expected_list, cast):
        assert type(x) is type(y)


def test_list_to_string():
    data_type = DataType.STRING
    test_list = ["0", "1", "1.0", "1.5", "3 + 3j", "string", "{'a': 4}", "False", "True"]
    expected_list = test_list
    cast = data_type.cast(test_list)
    assert cast == expected_list
    for x, y in zip(expected_list, cast):
        assert type(x) is type(y)


def test_list_to_boolean():
    data_type = DataType.BOOLEAN
    test_list = ["0", "1", "True", "False"]
    expected_list = [False, True, True, False]
    cast = data_type.cast(test_list)
    assert cast == expected_list
    for x, y in zip(expected_list, cast):
        assert type(x) is type(y)


def test_nested_list_to_anything():
    data_type = DataType.ANYTHING
    test_list = [[], "0", "1", [["1.0", "1.5"], "3 + 3j", "string", "{'a': 4}"], "False", "True"]
    expected_list = [[], 0, 1, [[1.0, 1.5], 3 + 3j, "string", {"a": 4}], False, True]
    cast = data_type.cast(test_list)
    assert cast == expected_list
    for x, y in zip(expected_list, cast):
        assert type(x) is type(y)
