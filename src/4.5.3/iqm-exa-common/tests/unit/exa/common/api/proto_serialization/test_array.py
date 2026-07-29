#  ********************************************************************************
#  Copyright (c) 2019-2023 IQM Finland Oy.
#  All rights reserved. Confidential and proprietary.
#
#  Distribution or reproduction of any information contained herein
#  is prohibited without IQM Finland Oy’s prior written permission.
#  ********************************************************************************
import numpy as np
import pytest

import exa.common.api.proto_serialization as proto_serialization

TEST_ARRAYS = [
    ("int64_array", np.array([[1, 2, 3, 4], [1, 6, 3, 4], [1, 2, 7, 4]])),
    ("float64_array", np.array([[1.6, 2.7, 3.8, 4.4], [-1.36, 2, 7, 4]])),
    ("complex128_array", np.array([[1.6j, 0.0, -1 + 5j, 4.4], [-1.36j, 2, 7j, 4]])),
    ("bool_array", np.array([[True, False], [False, True]])),
    ("empty_array", np.array([])),
    ("single_array", np.array([1])),
]


@pytest.mark.parametrize(["name", "array"], TEST_ARRAYS)
def test_array_values_are_preserved_in_round_trip(name, array):
    packed = proto_serialization.array.pack(array)
    unpacked = proto_serialization.array.unpack(packed)
    assert np.all(unpacked == array)
    assert unpacked.shape == array.shape
