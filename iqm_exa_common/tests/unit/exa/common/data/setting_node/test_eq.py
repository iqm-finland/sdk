#  ********************************************************************************
#  Copyright (c) 2019-2024 IQM Finland Oy.
#  All rights reserved. Confidential and proprietary.
#
#  Distribution or reproduction of any information contained herein
#  is prohibited without IQM Finland Oy’s prior written permission.
#  ********************************************************************************
from exa.common.data.parameter import CollectionType, DataType, Parameter, Setting
from exa.common.data.setting_node import SettingNode


def test_equal_with_empty_node():
    node1 = SettingNode(name="root", settings={}, subtrees={}, align_name=False)
    node2 = SettingNode(name="root", align_name=False)
    assert node1 == node2


def test_equal_with_name_only():
    node1 = SettingNode(name="root", settings={}, subtrees={}, align_name=False)
    node2 = SettingNode(name="root", settings={}, subtrees={}, align_name=False)
    assert node1 == node2


def test_equal_nested_nodes():
    parameter_setting_child_root1 = Parameter(
        name="parameter", label="parameter", unit="", data_type=DataType.INT, collection_type=CollectionType.SCALAR
    )
    setting_child_root1 = Setting(parameter=parameter_setting_child_root1, value=10)
    child_root1 = SettingNode(
        name="child", settings={"setting_child_key": setting_child_root1}, subtrees={}, align_name=False
    )
    node1 = SettingNode(name="root", settings={}, subtrees={"child_key": child_root1}, align_name=False)
    parameter_setting_child_root2 = Parameter(
        name="parameter", label="parameter", unit="", data_type=DataType.INT, collection_type=CollectionType.SCALAR
    )
    setting_child_root2 = Setting(parameter=parameter_setting_child_root2, value=10)
    child_root2 = SettingNode(
        name="child", settings={"setting_child_key": setting_child_root2}, subtrees={}, align_name=False
    )
    node2 = SettingNode(name="root", settings={}, subtrees={"child_key": child_root2}, align_name=False)
    assert node1 == node2


def test_not_equal_names():
    node1 = SettingNode(name="node1", settings={}, subtrees={}, align_name=False)
    node2 = SettingNode(name="node2", settings={}, subtrees={}, align_name=False)
    assert node1 != node2


def test_not_equal_children():
    child_root1 = SettingNode(name="child", settings={}, subtrees={}, align_name=False)
    node1 = SettingNode(name="root", settings={}, subtrees={"child_key": child_root1}, align_name=False)
    node2 = SettingNode(name="root", settings={}, subtrees={}, align_name=False)
    assert node1 != node2


def test_equal_with_root_and_settings_only():
    parameter1 = Parameter(
        name="parameter1",
        label="parameter1",
        unit="",
        data_type=DataType.COMPLEX,
        collection_type=CollectionType.SCALAR,
    )
    parameter2 = Parameter(name="parameter2")
    setting1_model = Setting(parameter=parameter1, value=3 + 4j)
    setting2_model = Setting(parameter=parameter2, value=50)
    node1 = SettingNode(
        name="root",
        settings={"setting1_key": setting1_model, "setting2_key": setting2_model},
        subtrees={},
        align_name=False,
    )
    setting1 = Setting(parameter=parameter1, value=3 + 4j)
    setting2 = Setting(parameter=parameter2, value=50)
    node2 = SettingNode(
        "root",
        setting1_key=setting1,
        setting2_key=setting2,
        align_name=False,
    )
    assert node1 == node2


def test_equal_with_root_and_child_nodes_only():
    child1_model = SettingNode(name="child1", settings={}, subtrees={}, align_name=False)
    child2_model = SettingNode(name="child2", settings={}, subtrees={}, align_name=False)
    root_model = SettingNode(
        name="root", settings={}, subtrees={"child1_key": child1_model, "child2_key": child2_model}, align_name=False
    )
    child1 = SettingNode("child1", align_name=False)
    child2 = SettingNode("child2", align_name=False)
    node1 = SettingNode("root", child1_key=child1, child2_key=child2, align_name=False)
    node2 = root_model
    assert node2 == node1
    assert node2["child1_key"].name == "child1"
    assert node2["child2_key"].name == "child2"


def test_equal_with_complex_tree():
    """The structure of the tested tree:
    root
    ├── child1
    │   ├── child1_1
    │   ├── child1_2
    │       └── child1_2_1
    ├── child2
    │   └── child2_settings
    ├── child3
    │   ├── child3_1
    │
    └── settings
    """
    child1_2_1_model = SettingNode(name="child1_2_1", settings={}, subtrees={}, align_name=False)
    child1_1_model = SettingNode(name="child1_1", settings={}, subtrees={}, align_name=False)
    child1_2_model = SettingNode(
        name="child1_2", settings={}, subtrees={"child1_2_1_key": child1_2_1_model}, align_name=False
    )
    parameter1_model = Parameter(
        name="parameter1", label="parameter1", unit="", data_type=DataType.FLOAT, collection_type=CollectionType.SCALAR
    )
    child2_settings_model = Setting(parameter=parameter1_model, value=10)
    child1_model = SettingNode(
        name="child1",
        settings={},
        subtrees={"child1_1_key": child1_1_model, "child1_2_key": child1_2_model},
        align_name=False,
    )
    child2_model = SettingNode(
        name="child2", settings={"child2_settings_key": child2_settings_model}, subtrees={}, align_name=False
    )
    parameter2_model = Parameter(
        name="parameter2", label="parameter2", unit="", data_type=DataType.FLOAT, collection_type=CollectionType.SCALAR
    )
    settings_model = Setting(parameter=parameter2_model, value=20)
    child3_1_model = SettingNode(name="child3_1", settings={}, subtrees={}, align_name=False)
    child3_model = SettingNode(name="child3", settings={}, subtrees={"child3_1_key": child3_1_model}, align_name=False)
    node1 = SettingNode(
        name="root",
        settings={"settings_key": settings_model},
        subtrees={"child1_key": child1_model, "child2_key": child2_model, "child3_key": child3_model},
        align_name=False,
    )
    child1_2_1 = SettingNode("child1_2_1", align_name=False)
    child1_1 = SettingNode("child1_1", align_name=False)
    child1_2 = SettingNode("child1_2", child1_2_1_key=child1_2_1, align_name=False)
    parameter1 = Parameter(name="parameter1")
    child2_settings = Setting(parameter=parameter1, value=10)
    child1 = SettingNode("child1", child1_1_key=child1_1, child1_2_key=child1_2, align_name=False)
    child2 = SettingNode("child2", child2_settings_key=child2_settings, align_name=False)
    parameter2 = Parameter(name="parameter2")
    settings = Setting(parameter=parameter2, value=20)
    child3_1 = SettingNode("child3_1", align_name=False)
    child3 = SettingNode("child3", child3_1_key=child3_1, align_name=False)
    node2 = SettingNode(
        "root", child1_key=child1, child2_key=child2, settings_key=settings, child3_key=child3, align_name=False
    )
    assert node1 == node2


def test_equal_with_inherited_model():
    # Make sure that both setting nodes are considered equal if the content is the same,
    # even though they are not instances of the same classes.
    # This is special logic overwriting Pydantic's default logic.
    setting_node = SettingNode(
        "root",
        freq=Parameter("frequency", "Frequency", "Hz"),
        align_name=False,
    )

    class ChildSettingNode(SettingNode):
        pass

    child_setting_node = ChildSettingNode(
        "root",
        freq=Parameter("frequency", "Frequency", "Hz"),
        align_name=False,
    )
    assert child_setting_node == setting_node
