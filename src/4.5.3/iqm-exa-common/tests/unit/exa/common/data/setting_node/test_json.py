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
from exa.common.data.setting_node import SettingNode


def test_json_with_complex_number():
    parameter = Parameter(name="test", label="test", unit="", data_type=DataType.COMPLEX)
    setting = Setting(parameter=parameter, value=0.5 + 0.6j)
    node = SettingNode(name="test", settings={"setting_key": setting}, subtrees={})
    expected_value = {"__complex__": "true", "real": 0.5, "imag": 0.6}
    assert loads(node.model_dump_json())["settings"]["setting_key"]["value"] == expected_value


def test_json_with_ndarray():
    parameter = Parameter(
        name="test", label="test", unit="", data_type=DataType.INT, collection_type=CollectionType.NDARRAY
    )
    array = np.arange(6).reshape(2, 3)
    array_b64 = base64.b64encode(array)
    setting = Setting(parameter=parameter, value=array)
    node = SettingNode(name="test", settings={"setting_key": setting}, subtrees={})
    expected_json = {
        "__ndarray__": "true",
        "data": array_b64.decode("utf-8"),
        "dtype": str(array.dtype),
        "shape": [2, 3],
    }
    assert loads(node.model_dump_json())["settings"]["setting_key"]["value"] == expected_json


def test_json_with_ndarray_of_complex_numbers():
    parameter = Parameter(
        name="test", label="test", unit="", data_type=DataType.COMPLEX, collection_type=CollectionType.NDARRAY
    )
    array = np.array([[1 + 2j, 1 + 3j], [5 + 6j, 3 + 8j]])
    array_b64 = base64.b64encode(array)
    setting = Setting(parameter=parameter, value=array)
    node = SettingNode(name="test", settings={"setting_key": setting}, subtrees={})
    expected_json = {"__ndarray__": "true", "data": array_b64.decode("utf-8"), "dtype": "complex128", "shape": [2, 2]}
    assert loads(node.model_dump_json())["settings"]["setting_key"]["value"] == expected_json


def test_serialize_and_deserialize_with_mixed_node_types():
    root = SettingNode("root")
    root["named_node"] = SettingNode("named_node", align_name=False)
    root["named_node.not_named"] = SettingNode("not_named")
    root["named_node.not_named.foo.bar"] = Setting(Parameter("bar"), 1.0)
    root["named_node.not_named.foo.named_again"] = SettingNode("named_again", align_name=False)
    root["named_node.not_named.foo.named_again.fuu"] = Setting(Parameter("fuu"), 1.0)

    deserialized_root = SettingNode(**root.model_dump())

    assert type(deserialized_root) is SettingNode
    assert type(deserialized_root.named_node) is SettingNode
    assert deserialized_root.named_node.name == "named_node"

    assert type(deserialized_root.named_node.not_named) is SettingNode
    assert deserialized_root.named_node.not_named.name == "named_node.not_named"

    assert type(deserialized_root.named_node.not_named.foo) is SettingNode
    assert type(deserialized_root.named_node.not_named.foo.bar) is Setting
    assert deserialized_root.named_node.not_named.foo.bar.name == "named_node.not_named.foo.bar"

    assert type(deserialized_root.named_node.not_named.foo.named_again) is SettingNode
    assert deserialized_root.named_node.not_named.foo.named_again.name == "named_again"
    assert deserialized_root.named_node.not_named.foo.named_again.fuu.name == "fuu"
