#  ********************************************************************************
#  Copyright (c) 2019-2024 IQM Finland Oy.
#  All rights reserved. Confidential and proprietary.
#
#  Distribution or reproduction of any information contained herein
#  is prohibited without IQM Finland Oy’s prior written permission.
#  ********************************************************************************
import base64
from json import loads

import numpy as np

from exa.common.data.parameter import CollectionType, DataType, Parameter, Setting


def test_json_with_complex_number():
    parameter = Parameter(name="test", label="test", unit="", data_type=DataType.COMPLEX)
    setting = Setting(parameter=parameter, value=0.5 + 0.5j)
    assert loads(setting.model_dump_json())["value"] == {"__complex__": "true", "real": 0.5, "imag": 0.5}


def test_json_with_ndarray():
    parameter = Parameter(
        name="test", label="test", unit="", data_type=DataType.INT, collection_type=CollectionType.NDARRAY
    )
    array = np.arange(6).reshape(2, 3)
    array_b64 = base64.b64encode(array)
    setting = Setting(parameter=parameter, value=array)
    expected_json = {
        "__ndarray__": "true",
        "data": array_b64.decode("utf-8"),
        "dtype": str(array.dtype),
        "shape": [2, 3],
    }
    assert loads(setting.model_dump_json())["value"] == expected_json


def test_json_with_ndarray_of_complex_numbers():
    parameter = Parameter(
        name="test", label="test", unit="", data_type=DataType.COMPLEX, collection_type=CollectionType.NDARRAY
    )
    array = np.array([[1 + 2j, 1 + 3j], [5 + 6j, 3 + 8j]])
    array_b64 = base64.b64encode(array)
    setting = Setting(parameter=parameter, value=array)
    expected_json = {"__ndarray__": "true", "data": array_b64.decode("utf-8"), "dtype": "complex128", "shape": [2, 2]}
    assert loads(setting.model_dump_json())["value"] == expected_json


def test_dump_ignores_source():
    # make a non-serializable source
    setting = Setting(Parameter("foo"), 1.0, source=lambda x: x + 1)
    # source not in dump, so it is not a problem
    assert set(setting.model_dump().keys()) == {"parameter", "value", "path", "read_only"}
