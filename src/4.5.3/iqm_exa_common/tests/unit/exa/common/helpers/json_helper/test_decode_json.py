#  ********************************************************************************
#  Copyright (c) 2019-2020 IQM Finland Oy.
#  All rights reserved. Confidential and proprietary.
#
#  Distribution or reproduction of any information contained herein
#  is prohibited without IQM Finland Oy’s prior written permission.
#  ********************************************************************************
import base64

import numpy as np

from exa.common.helpers.json_helper import decode_json


def test_not_dict():
    value = 50
    assert decode_json(value) == value


def test_dict_without_decoding():
    dct = {"test": "test"}
    assert decode_json(dct) == dct


def test_dict_with_complex_number():
    complex = 3 + 4j
    dct = {"__complex__": "true", "real": 3, "imag": 4}
    assert decode_json(dct) == complex


def test_dict_with_ndarray():
    array = np.arange(10).reshape(2, 5)
    array_b64 = base64.b64encode(array)
    dct = {"__ndarray__": "true", "data": array_b64, "dtype": str(array.dtype), "shape": [2, 5]}
    assert (decode_json(dct) == array).all()


def test_dict_with_ndarray_of_complex_numbers():
    array = np.array([[1 + 2j, 1 + 3j], [5 + 6j, 3 + 8j]])
    array_b64 = base64.b64encode(array)
    dct = {"__ndarray__": "true", "data": array_b64, "dtype": "complex128", "shape": [2, 2]}
    assert (decode_json(dct) == array).all()


def test_dict_with_tuple():
    tuple = (1, 2, "test")
    dict = {"__tuple__": "true", "data": [1, 2, "test"]}
    assert decode_json(dict) == tuple
