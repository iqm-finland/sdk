#  ********************************************************************************
#    _____  ____ _
#   / _ \ \/ / _` |  Framework for control
#  |  __/>  < (_| |  and measurement of
#   \___/_/\_\__,_|  superconducting qubits
#
#  Copyright (c) 2019-2021 IQM Finland Oy.
#  All rights reserved. Confidential and proprietary.
#
#  Distribution or reproduction of any information contained herein
#  is prohibited without IQM Finland Oy’s prior written permission.
#  ********************************************************************************
import numpy as np

from exa.common.data.parameter import CollectionType


def test_list_to_array():
    assert isinstance(CollectionType.NDARRAY.cast([1, 2, 3]), np.ndarray)


def test_array_to_list():
    assert CollectionType.LIST.cast(np.array([1, 2, 3])) == [1, 2, 3]


def test_scalar_to_list():
    assert CollectionType.LIST.cast(5) == [5]


def test_scalar_remains_scalar():
    assert CollectionType.SCALAR.cast(5) == 5
