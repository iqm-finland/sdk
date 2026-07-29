#  ********************************************************************************
#  Copyright (c) 2019-2024 IQM Finland Oy.
#  All rights reserved. Confidential and proprietary.
#
#  Distribution or reproduction of any information contained herein
#  is prohibited without IQM Finland Oy’s prior written permission.
#  ********************************************************************************
from exa.common.data.parameter import CollectionType, DataType, Parameter, Setting


def test_equal_settings():
    parameter1 = Parameter(
        name="parameter", label="parameter", unit="", data_type=DataType.INT, collection_type=CollectionType.SCALAR
    )
    setting1 = Setting(parameter=parameter1, value=10)
    parameter2 = Parameter(
        name="parameter", label="parameter", unit="", data_type=DataType.INT, collection_type=CollectionType.SCALAR
    )
    setting2 = Setting(parameter=parameter2, value=10)
    assert setting1 == setting2


def test_not_equal_parameters():
    parameter1 = Parameter(
        name="parameter", label="parameter", unit="", data_type=DataType.INT, collection_type=CollectionType.SCALAR
    )
    setting1 = Setting(parameter=parameter1, value=10)
    parameter2 = Parameter(
        name="parameter_test",
        label="parameter",
        unit="",
        data_type=DataType.INT,
        collection_type=CollectionType.SCALAR,
    )
    setting2 = Setting(parameter=parameter2, value=10)
    assert setting1 != setting2


def test_not_equal_values():
    parameter1 = Parameter(
        name="parameter", label="parameter", unit="", data_type=DataType.INT, collection_type=CollectionType.SCALAR
    )
    setting1 = Setting(parameter=parameter1, value=1)
    parameter2 = Parameter(
        name="parameter", label="parameter", unit="", data_type=DataType.INT, collection_type=CollectionType.SCALAR
    )
    setting2 = Setting(parameter=parameter2, value=10)
    assert setting1 != setting2


def test_source_not_part_of_model():
    setting1 = Setting(Parameter("foo"), 1.0)
    setting2 = Setting(Parameter("foo"), 1.0, source={"type": "poopy", "poop": True})
    assert setting1 == setting2
