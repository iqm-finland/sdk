#  ********************************************************************************
#  Copyright (c) 2019-2023 IQM Finland Oy.
#  All rights reserved. Confidential and proprietary.
#
#  Distribution or reproduction of any information contained herein
#  is prohibited without IQM Finland Oy’s prior written permission.
#  ********************************************************************************
from typing import Any

import numpy as np
import pytest

import exa.common.api.proto_serialization as proto_serialization
from exa.common.helpers.numpy_helper import coerce_numpy_type_to_native

TEST_ARRAYS = [
    ("int64_array", np.array([[1, 2, 3, 4], [1, 6, 3, 4], [1, 2, 7, 4]])),
    ("float64_array", np.array([[1.6, 2.7, 3.8, 4.4], [-1.36, 2, 7, 4]])),
    ("complex128_array", np.array([[1.6j, 0.0, -1 + 5j, 4.4], [-1.36j, 2, 7j, 4]])),
    ("bool_array", np.array([[True, False], [False, True]])),
    ("empty_array", np.array([])),
    ("single_array", np.array([1])),
]


TEST_DATA: list[tuple[str, Any]] = [
    ("none_value", None),
    ("int64_value", 46687),
    ("float64_value", -1.3224),
    ("bool_value", False),
    ("complex128_value", -88.9 + 1j),
    ("string_value", "some stringy"),
    ("string_array", ["str0", "str1", "str2"]),
    ("int_list", [1, 2, 3]),
    ("float_list", [1.1]),
    ("complex_list", [1.1j, -0, 1 - 0.11j]),
    ("bool_list", [True, False]),
    ("empty_list", []),
]


@pytest.mark.parametrize(["name", "array"], TEST_ARRAYS)
def test_array_values_are_preserved_in_round_trip(name, array):
    serialized = proto_serialization.datum.serialize(array)
    deserialized = proto_serialization.datum.deserialize(serialized)
    assert np.all(deserialized == array)
    assert deserialized.shape == array.shape

    packed = proto_serialization.datum.pack(array)
    unpacked = proto_serialization.datum.unpack(packed)
    assert np.all(unpacked == array)
    assert unpacked.shape == array.shape


@pytest.mark.parametrize(["name", "datum"], TEST_DATA)
def test_scalar_values_are_preserved_in_round_trip(name, datum):
    serialized = proto_serialization.datum.serialize(datum)
    deserialized = proto_serialization.datum.deserialize(serialized)
    assert deserialized == datum
    assert type(deserialized) is type(datum)

    packed = proto_serialization.datum.pack(datum)
    unpacked = proto_serialization.datum.unpack(packed)
    assert unpacked == datum
    assert type(unpacked) is type(datum)


rejected_cases = [
    np.bool_(True),
    np.int64(13241),
    np.float64(3.1212),
    np.complex128(3.12j + 55),
]


@pytest.mark.parametrize("datum", rejected_cases)
def test_numpy_types_are_rejected_but_not_after_coercion(datum):
    with pytest.raises(TypeError, match="not supported"):
        proto_serialization.datum.serialize(datum)

    proto_serialization.datum.serialize(coerce_numpy_type_to_native(datum))


def test_ragged_arrays_are_rejected():
    with pytest.raises(TypeError, match="Unsupported numpy array type"):
        proto_serialization.datum.serialize(np.array([1, [2, 3]], dtype=object))
