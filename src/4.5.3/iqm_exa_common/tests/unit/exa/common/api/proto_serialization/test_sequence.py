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

TEST_LISTS: list[tuple[str, list]] = [
    ("string_array", ["str0", "str1", "str2"]),
    ("int_list", [1, 2, 3]),
    ("float_list", [1.1]),
    ("complex_list", [1.1j, -0, 1 - 0.11j]),
    ("bool_list", [True, False]),
    ("empty_list", []),
    ("numpy_int", [np.int64(1), np.int64(2), np.int64(3)]),
    ("numpy_float", [np.float64(1.0), np.float64(2.0), np.float64(3.0)]),
    ("numpy_complex", [np.complex128(1 + 1j), np.complex128(2 + 2j), np.complex128(3 + 3j)]),
]


@pytest.mark.parametrize(["name", "original"], TEST_LISTS)
def test_array_values_are_preserved_in_round_trip(name, original):
    packed = proto_serialization.sequence.pack(original)
    unpacked = proto_serialization.sequence.unpack(packed)
    assert unpacked == original
