#  ********************************************************************************
#  Copyright (c) 2019-2024 IQM Finland Oy.
#  All rights reserved. Confidential and proprietary.
#
#  Distribution or reproduction of any information contained herein
#  is prohibited without IQM Finland Oy’s prior written permission.
#  ********************************************************************************
from exa.common.data.parameter import CollectionType, DataType, Parameter, Sweep


def test_equal_sweeps():
    parameter1 = Parameter(
        name="parameter", label="parameter", unit="", data_type=DataType.INT, collection_type=CollectionType.SCALAR
    )
    sweep1 = Sweep(parameter=parameter1, data=[1, 2])
    parameter2 = Parameter(
        name="parameter", label="parameter", unit="", data_type=DataType.INT, collection_type=CollectionType.SCALAR
    )
    sweep2 = Sweep(parameter=parameter2, data=[1, 2])
    assert sweep1 == sweep2


def test_not_equal_parameters():
    parameter1 = Parameter(
        name="parameter", label="parameter", unit="", data_type=DataType.INT, collection_type=CollectionType.SCALAR
    )
    sweep1 = Sweep(parameter=parameter1, data=[1, 2])
    parameter2 = Parameter(
        name="parameter_test",
        label="parameter",
        unit="",
        data_type=DataType.INT,
        collection_type=CollectionType.SCALAR,
    )
    sweep2 = Sweep(parameter=parameter2, data=[1, 2])
    assert sweep1 != sweep2


def test_not_equal_data():
    parameter1 = Parameter(
        name="parameter", label="parameter", unit="", data_type=DataType.INT, collection_type=CollectionType.SCALAR
    )
    sweep1 = Sweep(parameter=parameter1, data=[1, 2])
    parameter2 = Parameter(
        name="parameter", label="parameter", unit="", data_type=DataType.INT, collection_type=CollectionType.SCALAR
    )
    sweep2 = Sweep(parameter=parameter2, data=[1, 2, 3])
    assert sweep1 != sweep2
