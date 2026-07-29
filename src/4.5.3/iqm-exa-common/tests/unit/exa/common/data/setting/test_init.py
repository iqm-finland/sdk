#  ********************************************************************************
#  Copyright (c) 2019-2024 IQM Finland Oy.
#  All rights reserved. Confidential and proprietary.
#
#  Distribution or reproduction of any information contained herein
#  is prohibited without IQM Finland Oy’s prior written permission.
#  ********************************************************************************
import numpy as np
import pytest

from exa.common.data.parameter import CollectionType, DataType, Parameter, Setting

array1 = Setting(Parameter(name="a", data_type=DataType.INT, collection_type=CollectionType.NDARRAY), np.arange(6))
array2 = Setting(
    Parameter(name="a", data_type=DataType.INT, collection_type=CollectionType.NDARRAY), np.arange(6).reshape((2, 3))
)
array3 = Setting(
    Parameter(name="a", data_type=DataType.INT, collection_type=CollectionType.NDARRAY), np.array([0, 1, 2, 3, 4, 10])
)
float1 = Setting(Parameter(name="f1"), 1.0)
float2 = Setting(Parameter(name="f1"), 2.0)
float3 = Setting(Parameter(name="f2"), 1.0)


@pytest.mark.parametrize(
    "a,b,res",
    [
        (float1, 1.0, False),  # wrong class
        (float1, float1, True),
        (float1, float2, False),  # wrong value
        (float1, float3, False),  # wrong param
        (array1, array1, True),
        (float1, array1, False),  # wrong type
        (array1, float1, False),  # wrong type
        (array1, array2, False),  # wrong shape
        (array1, array3, False),  # wrong contents
    ],
)
def test_setting_eq(a, b, res):
    if res:
        assert a == b
    else:
        assert a != b


def test_setting_diff_sets():
    p1 = Parameter("only_in_1", "")
    p2 = Parameter("only_in_2", "")
    p3 = Parameter("in_both", "")
    p4 = Parameter("equal", "")

    set1 = {Setting(p1, 1), Setting(p3, 55), Setting(p4, 8)}
    set2 = {Setting(p2, 2), Setting(p3, 123), Setting(p4, 8)}

    diff = Setting.diff_sets(set1, set2)
    reverse_diff = Setting.diff_sets(set2, set1)

    assert len(diff) == 2
    assert len(reverse_diff) == 2
    assert Setting.get_by_name(p1.name, diff)
    assert Setting.get_by_name(p3.name, diff)
    assert Setting.get_by_name(p1.name, diff).value == 1
    assert Setting.get_by_name(p3.name, diff).value == 55

    assert Setting.get_by_name(p2.name, reverse_diff)
    assert Setting.get_by_name(p3.name, reverse_diff)
    assert Setting.get_by_name(p2.name, reverse_diff).value == 2
    assert Setting.get_by_name(p3.name, reverse_diff).value == 123


def test_setting_properties():
    s = Setting(Parameter("foo", "Label", "kg"), 1000)
    assert s.name == "foo"
    assert s.label == "Label"
    assert s.unit == "kg"


@pytest.mark.parametrize(
    "parameter, value, expected_type",
    [
        (Parameter("nn", "ll", "uu", DataType.FLOAT, CollectionType.SCALAR), 1.0, float),
        (Parameter("nn", "ll", "uu", DataType.INT, CollectionType.LIST), [1, 2, 3], list),
        (Parameter("nn", "ll", "uu", DataType.FLOAT, CollectionType.NDARRAY), [1.0], np.ndarray),
        (Parameter("nn", "ll", "uu", DataType.FLOAT, CollectionType.NDARRAY), [1.0, 2.0], np.ndarray),
        (Parameter("nn", "ll", "uu", DataType.INT, CollectionType.NDARRAY), [[1], [2]], np.ndarray),
    ],
)
def test_validate_parameter_value_after(parameter, value, expected_type):
    setting = Setting(parameter=parameter, value=value)
    assert type(setting.value) is expected_type
