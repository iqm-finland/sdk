#  ********************************************************************************
#  Copyright (c) 2019-2024 IQM Finland Oy.
#  All rights reserved. Confidential and proprietary.
#
#  Distribution or reproduction of any information contained herein
#  is prohibited without IQM Finland Oy’s prior written permission.
#  ********************************************************************************
import pydantic
import pytest

from exa.common.data.parameter import CollectionType, DataType, Parameter
from exa.common.errors.station_control_errors import ValidationError


def test_simple_init():
    parameter = Parameter("test", "TEST", "some_unit", DataType.STRING, CollectionType.LIST)
    assert parameter.name == "test"
    assert parameter.label == "TEST"
    assert parameter.unit == "some_unit"
    assert parameter.data_type is DataType.STRING
    assert parameter.collection_type is CollectionType.LIST


def test_init_with_default_values():
    parameter = Parameter("test")
    assert parameter.name == "test"
    assert parameter.label == "test"
    assert parameter.unit == ""
    assert parameter.data_type is DataType.FLOAT
    assert parameter.collection_type is CollectionType.SCALAR


def test_elemental_wise_init():
    parameter = Parameter("test", "test label", element_indices=0)
    assert parameter.name == "test__0"
    assert parameter.label == "test label [0]"
    assert parameter.parent_name == "test"
    assert parameter.element_indices == [0]

    parameter = Parameter("test", "test label", element_indices=[0, 1, 2])
    assert parameter.name == "test__0__1__2"
    assert parameter.label == "test label [0, 1, 2]"
    assert parameter.parent_name == "test"
    assert parameter.element_indices == [0, 1, 2]

    parameter = Parameter("test", "test label", element_indices=[0, 1, 2])
    assert parameter.name == "test__0__1__2"
    assert parameter.label == "test label [0, 1, 2]"
    assert parameter.parent_name == "test"
    assert parameter.element_indices == [0, 1, 2]

    with pytest.raises(
        ValidationError, match="Element-wise parameter must have 'CollectionType.SCALAR' collection type."
    ):
        Parameter("this_is_collection", collection_type=CollectionType.LIST, element_indices=0)

    with pytest.raises(pydantic.ValidationError, match="2 validation errors for Parameter"):
        Parameter("this_is_scalar", element_indices={"this_cant_have_dicts": 2})


def test_create_elemental_parameter_for_collection_parameter():
    parameter = Parameter("test", "TEST", "some_unit", DataType.STRING, CollectionType.LIST)
    elem_param = parameter.create_element_parameter_for([0, 1])
    assert elem_param.name == "test__0__1"
    assert elem_param.label == "TEST [0, 1]"
    assert elem_param.element_indices == [0, 1]
    assert elem_param.parent_name == "test"
    assert elem_param.unit == "some_unit"
    assert elem_param.data_type is DataType.STRING
    assert elem_param.collection_type is CollectionType.SCALAR

    with pytest.raises(ValidationError, match="Cannot create an element-wise parameter"):
        scalar = Parameter("this_is_scalar")
        scalar.create_element_parameter_for(0)


def test_data_type_number_is_deprecated():
    with pytest.warns(
        DeprecationWarning, match="data_type 'DataType.NUMBER' is deprecated, use 'DataType.FLOAT' instead."
    ):
        parameter = Parameter("test", "TEST", "some_unit", DataType.NUMBER, CollectionType.LIST)
    assert parameter.data_type == DataType.FLOAT
