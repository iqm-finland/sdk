#  ********************************************************************************
#  Copyright (c) 2019-2020 IQM Finland Oy.
#  All rights reserved. Confidential and proprietary.
#
#  Distribution or reproduction of any information contained herein
#  is prohibited without IQM Finland Oy’s prior written permission.
#  ********************************************************************************
from exa.common.data.parameter import CollectionType


def test_scalar_value():
    collection_type = CollectionType.SCALAR
    assert collection_type.value == 0


def test_list_value():
    collection_type = CollectionType.LIST
    assert collection_type.value == 1


def test_array_value():
    collection_type = CollectionType.NDARRAY
    assert collection_type.value == 2
