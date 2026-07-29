#  ********************************************************************************
#  Copyright (c) 2019-2024 IQM Finland Oy.
#  All rights reserved. Confidential and proprietary.
#
#  Distribution or reproduction of any information contained herein
#  is prohibited without IQM Finland Oy’s prior written permission.
#  ********************************************************************************
from exa.common.data.parameter import CollectionType, DataType, Parameter


def test_equal_parameters():
    parameter1 = Parameter(
        name="parameter",
        label="parameter",
        unit="",
        data_type=DataType.INT,
        collection_type=CollectionType.SCALAR,
    )
    parameter2 = Parameter(
        name="parameter",
        label="parameter",
        unit="",
        data_type=DataType.INT,
        collection_type=CollectionType.SCALAR,
    )
    assert parameter1 == parameter2


def test_not_equal_names():
    parameter1 = Parameter(
        name="parameter",
        label="parameter",
        unit="",
        data_type=DataType.INT,
        collection_type=CollectionType.SCALAR,
    )
    parameter2 = Parameter(
        name="parameter_test",
        label="parameter",
        unit="",
        data_type=DataType.INT,
        collection_type=CollectionType.SCALAR,
    )
    assert parameter1 != parameter2


def test_not_equal_data_types():
    parameter1 = Parameter(
        name="parameter",
        label="parameter",
        unit="",
        data_type=DataType.INT,
        collection_type=CollectionType.SCALAR,
    )
    parameter2 = Parameter(
        name="parameter",
        label="parameter",
        unit="",
        data_type=DataType.STRING,
        collection_type=CollectionType.SCALAR,
    )
    assert parameter1 != parameter2
