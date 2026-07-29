#  ********************************************************************************
#  Copyright (c) 2019-2023 IQM Finland Oy.
#  All rights reserved. Confidential and proprietary.
#
#  Distribution or reproduction of any information contained herein
#  is prohibited without IQM Finland Oy’s prior written permission.
#  ********************************************************************************
import numpy as np

import exa.common.api.proto_serialization as protos
from exa.common.data.parameter import CollectionType, DataType, Parameter, Setting
from exa.common.data.setting_node import SettingNode

"""The structure of the tested tree:
    root
    ├── child1
    │   ├── child1_1
    │   ├── child1_2
    │       └── child1_2_1
    ├── child2
    │   ├── child2_setting1
    │   ├── child2_setting2
    │   ...
    ├── setting1
    ├── setting2
    ...
    """
root = SettingNode(
    name="root",
    align_name=False,
    child1_key=SettingNode(
        align_name=False,
        name="child1",
        child1_1_key=SettingNode("child1_1"),
        child1_2_key=SettingNode("child1_2", child1_2_1_key=SettingNode("child1_2_1")),
    ),
    child2_key=SettingNode(
        name="child2",
        align_name=False,
        child2_setting1=Setting(Parameter(name="parameter str", data_type=DataType.STRING), value="gaah"),
        child2_setting2=Setting(Parameter(name="parameter bool true", data_type=DataType.BOOLEAN), value=True),
        child2_setting3=Setting(Parameter(name="parameter bool false", data_type=DataType.BOOLEAN), value=False),
        child2_setting4=Setting(Parameter(name="parameter int", data_type=DataType.INT), value=-1),
        child2_setting5=Setting(Parameter(name="parameter float", data_type=DataType.FLOAT), value=10.0),
        child2_setting6=Setting(Parameter(name="parameter complex", data_type=DataType.COMPLEX), value=10j),
        child2_setting7=Setting(Parameter(name="parameter anything", data_type=DataType.ANYTHING), value=0),
        child2_setting8=Setting(
            Parameter(
                name="parameter complex array", data_type=DataType.COMPLEX, collection_type=CollectionType.NDARRAY
            ),
            value=np.array([1j, 2j]),
        ),
        child2_setting9=Setting(
            Parameter(name="parameter complex element", data_type=DataType.COMPLEX, element_indices=0),
            value=1j,
        ),
    ),
    setting1=Setting(Parameter(name="parameter float none", label="label", unit="unit"), value=None),
    setting2=Setting(Parameter(name="parameter ro float", label="label", unit="unit"), value=None, read_only=True),
)


def test_reversibility():
    packed = protos.setting_node.pack(root, minimal=False)
    unpacked = protos.setting_node.unpack(packed)
    assert unpacked == root

    root_dict = {s.name: s for s in root.all_settings}
    unpacked_dict = {s.name: s for s in unpacked.all_settings}

    # read_only flag was serialized
    assert not root_dict["parameter float"].read_only
    assert root_dict["parameter ro float"].read_only
    assert not unpacked_dict["parameter float"].read_only
    assert unpacked_dict["parameter ro float"].read_only


def test_optimized_version():
    packed = protos.setting_node.pack(root, minimal=True)
    unpacked = protos.setting_node.unpack(packed)

    root_dict = {s.name: s for s in root.all_settings}
    unpacked_dict = {s.name: s for s in unpacked.all_settings}

    # setting (name, value) pairs must match
    assert root_dict.keys() == unpacked_dict.keys()
    for k, v in root_dict.items():
        assert np.all(v.value == unpacked_dict[k].value)

    # read_only flag was not serialized, and False is the default
    assert not root_dict["parameter float"].read_only
    assert root_dict["parameter ro float"].read_only
    assert not unpacked_dict["parameter float"].read_only
    assert not unpacked_dict["parameter ro float"].read_only
