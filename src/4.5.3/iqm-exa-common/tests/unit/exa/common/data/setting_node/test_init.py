#  ********************************************************************************
#  Copyright (c) 2019-2024 IQM Finland Oy.
#  All rights reserved. Confidential and proprietary.
#
#  Distribution or reproduction of any information contained herein
#  is prohibited without IQM Finland Oy’s prior written permission.
#  ********************************************************************************
import copy
import logging

import numpy as np
import pytest

from exa.common.data.parameter import CollectionType, DataType, Parameter, Setting
from exa.common.data.setting_node import SettingNode
from exa.common.errors.exa_error import UnknownSettingError


def test_init_without_fixture():
    # For easier debugging
    SettingNode(
        "root",
        freq=Parameter("frequency", "Frequency", "Hz"),
        power=Parameter("power", "Power", "dBm").set(15),
        volt=Parameter("volt", "Voltage", "V").set(-0.01),
    )


def test_init(node_1, node_2, node_3):
    assert node_2["freq"] is not node_1["freq"]
    assert node_1.freq.value is None
    assert node_1.power.value is 15
    assert isinstance(node_3["qubit"], Setting)
    assert isinstance(node_3["rest"], SettingNode)
    assert node_3["rest"] == node_2
    assert node_3["rest"]["subtree"] == node_1
    assert node_3["rest"]["subtree"].path == "rest.subtree"
    assert node_3.rest.subtree.volt.value == -0.01
    assert node_3.rest.subtree.volt.parameter.unit == "V"
    for key in ["rest", "horse", "qubit"]:
        assert key in node_3.children
    # normal SettingNode aligns name and path:
    new = SettingNode("root", child=SettingNode("some name", thing=Setting(Parameter("foo"), 1.0)))
    assert new == SettingNode(
        "root",
        subtrees={
            "child": SettingNode("child", settings={"thing": Setting(Parameter("child.thing", label="foo"), 1.0)})
        },
    )


def test_delete_setting_member(node_1):
    assert "power" in node_1.children
    del node_1.power
    assert "power" not in node_1.children
    with pytest.raises(AttributeError):
        node_1.power


def test_delete_setting_item(node_1):
    assert "power" in node_1.children
    del node_1["power"]
    assert "power" not in node_1.children
    with pytest.raises(AttributeError):
        node_1.power


def test_iteration(node_1, node_3):
    all_nodes = [n for n in node_3]
    assert len(all_nodes) == 9
    assert node_1 in all_nodes
    assert node_1.power in all_nodes

    branches = list(node_3.nodes_by_type(SettingNode, recursive=True))
    assert node_3 in branches
    assert node_1 in branches
    assert len(branches) == 3
    assert len(list(node_3.nodes_by_type(Setting))) == 2
    assert len(list(node_3.all_settings)) == 6
    assert node_1.volt in node_3.all_settings


def test_get_parent_of(node_3):
    assert node_3.get_parent_of("qubit") == node_3
    assert node_3.get_parent_of("freq") == node_3.rest
    assert node_3.get_parent_of("power") == node_3.rest.subtree

    with pytest.raises(AttributeError) as err:
        node_3.get_parent_of("not there")
    assert "not there" in str(err.value)


def test_update_setting(node_3):
    s = node_3.rest.subtree.power
    assert s.value == 15
    node_3.update_setting(s.update(88))
    assert node_3.rest.subtree.power.value == 88
    assert s.value == 15  # maybe bad design, but this is how Setting.update works. It creates a new instance.


class SomeClass(SettingNode):
    def __init__(self, name, some_attribute=None, **kwargs):
        super().__init__(name, **kwargs)
        self.some_attribute = some_attribute


def test_transform_into_setting_nodes(node_3):
    some = SomeClass("name", some_attribute=4, sub=node_3)
    some.sub.subsub = SomeClass("subsub", some_attribute=6)
    assert list(some.nodes_by_type(SomeClass, recursive=True))
    assert isinstance(some.sub.subsub, SomeClass)
    assert some.some_attribute == 4

    transformed = SettingNode.transform_node_types(some)
    assert not list(transformed.nodes_by_type(SomeClass, recursive=True))
    assert isinstance(transformed.sub.subsub, SettingNode)
    assert not hasattr(transformed, "some_attribute")


def test_transform_into_other_class(node_3):
    some = SettingNode("name", sub=node_3)
    some.sub.subsub = SettingNode("subsub")

    transformed = SomeClass.transform_node_types(some)
    assert all(isinstance(node, SomeClass) for node in transformed.nodes_by_type(SettingNode, recursive=True))


def test_set_from_dict(node_4):
    new_values = {
        "rest": {
            "subtree": {
                "freq": 44,
                "volt": -1,
            }
        }
    }
    node_4.set_from_dict(new_values)
    assert node_4.rest.subtree.freq.value == 44
    assert node_4.rest.subtree.volt.value == -1


def test_set_from_dict_logs_error(node_4, caplog):
    new_values = {
        "something_else": {
            "foo": {
                "freq": 44,
                "volt": -1,
            }
        }
    }

    with caplog.at_level(logging.DEBUG):
        node_4.set_from_dict(new_values, strict=False)
    expected_message = "Tried to set something_else to"
    assert any(expected_message in rec.message for rec in caplog.records)


def test_set_from_dict_raises_error(node_4):
    new_values = {
        "something_else": {
            "foo": {
                "freq": 44,
                "volt": -1,
            }
        }
    }
    with pytest.raises(UnknownSettingError) as err:
        node_4.set_from_dict(new_values, strict=True)
    assert "something_else" in str(err.value)


def test_setting_with_path_name(node_5):
    with_path = node_5.setting_with_path_name(Setting(Parameter("fish1"), None))
    assert with_path.name == "fish1"
    assert with_path.parameter.collection_type == CollectionType.SCALAR
    assert with_path.value == 4.0
    with_path = node_5.setting_with_path_name(Setting(Parameter("other_horses"), None))
    assert with_path.name == "horses.other_horses"
    assert with_path.parameter.collection_type == CollectionType.LIST
    assert with_path.value == [2.0, 3.0]


def test_diff():
    """diff of two SettingNodes provides correct results"""
    a = SettingNode(
        "node_a",
        freq=Parameter("frequency", "Frequency", "Hz").set(1.0),
        power=Parameter("power", "Power", "dBm").set(15),
        voltages=Parameter("voltages", "Voltages", "V", collection_type=CollectionType.LIST).set([-0.01, 0.2]),
        sub_x=SettingNode(
            "subnode_x",
            ampl=Parameter("amplitude", "Amplitude", "").set(0.5),
            sub_y=SettingNode(
                "subnode_y",
            ),
        ),
        align_name=False,
    )
    assert not a.diff(a)

    b = copy.deepcopy(a)
    b.name = "node_b"
    assert a.diff(b) == [": node name: node_a/node_b"]

    b = copy.deepcopy(a)
    del b.freq
    assert a.diff(b) == [": -setting: freq"]

    b = copy.deepcopy(a)
    b.qqq = Setting(Parameter("qqq"), 1.0)
    assert a.diff(b) == [": +setting: qqq"]

    b = copy.deepcopy(a)
    b.sub_x.www = Setting(Parameter("www"), 1.0)
    assert a.diff(b) == ["sub_x: +setting: www"]

    b = copy.deepcopy(a)
    b.sub_x.sub_y.www = Setting(Parameter("www"), 1.0)
    assert a.diff(b) == ["sub_x.sub_y: +setting: www"]

    b = copy.deepcopy(a)
    b.sub_x.ampl = 0.8
    assert a.diff(b) == ["sub_x: ampl: v: 0.5/0.8"]

    b = copy.deepcopy(a)
    b.sub_y = SettingNode("subnode_y")
    assert a.diff(b) == [": +subnode: sub_y"]

    b = copy.deepcopy(a)
    del b.sub_x
    assert a.diff(b) == [": -subnode: sub_x"]

    b = copy.deepcopy(a)
    b.freq = Parameter("xxx", "Frequency", "Hz").set(1.0)
    assert a.diff(b) == [": freq: n: frequency/xxx"]

    b = copy.deepcopy(a)
    b.freq = Parameter("frequency", "xxx", "Hz").set(1.0)
    assert a.diff(b) == [": freq: l: Frequency/xxx"]

    b = copy.deepcopy(a)
    b.freq = Parameter("frequency", "Frequency", "xxx").set(1.0)
    assert a.diff(b) == [": freq: u: Hz/xxx"]

    b = copy.deepcopy(a)
    b.freq = Parameter("frequency", "Frequency", "Hz", data_type=DataType.COMPLEX).set(1.0)
    assert a.diff(b) == [": freq: dt: 1/2"]

    b = copy.deepcopy(a)
    b.voltages = Parameter("voltages", "Voltages", "V", collection_type=CollectionType.NDARRAY).set(
        np.array([-0.01, 0.2])
    )
    assert a.diff(b) == [": voltages: ct: 1/2"]

    b = copy.deepcopy(a)
    b.freq = Parameter("frequency", "Frequency", "Hz").set(2.0)
    assert a.diff(b) == [": freq: v: 1.0/2.0"]

    b = copy.deepcopy(a)
    b.voltages = Parameter("voltages", "Voltages", "V", collection_type=CollectionType.LIST).set([1.0, 2.0, 3.1])
    assert a.diff(b) == [": voltages: v: [-0.01, 0.2]/[1.0, 2.0, 3.1]"]

    # multiple changes
    b = copy.deepcopy(a)
    del b.sub_x
    b.qqq = Setting(Parameter("qqq"), 1.0)
    b.freq = Parameter("frequency", "Frequency", "Hz").set(2.0)
    assert a.diff(b) == [
        ": freq: v: 1.0/2.0",
        ": +setting: qqq",
        ": -subnode: sub_x",
    ]


def test_add_for_path_and_get_node_for_path():
    root = SettingNode("root")
    subnode = SettingNode("duck")
    subnode.add_for_path([Setting(Parameter("goose"), 2.0)], "moose")
    nodes_to_add = [Parameter("rat"), Setting(Parameter("horse"), 1.0), subnode]
    root.add_for_path(nodes_to_add, "all_animals.stupid_animals", override_values={"horse": 100.0})
    assert root.all_animals.stupid_animals.horse.value == 100.0
    assert root.all_animals.stupid_animals.rat.value is None
    assert root.all_animals.stupid_animals.duck.moose.goose.value == 2.0
    assert root.all_animals.stupid_animals.duck.moose.goose.name == "all_animals.stupid_animals.duck.moose.goose"
    assert root.get_node_for_path("all_animals.stupid_animals.horse").value == 100.0
    assert isinstance(root.get_node_for_path("all_animals.stupid_animals.duck"), SettingNode)
    assert root.get_node_for_path("all_animals.stupid_animals.duck").moose.goose.value == 2.0
    with pytest.raises(KeyError, match="key 'smart_animal' is not found in the preceding node"):
        root.get_node_for_path("all_animals.stupid_animals.smart_animal")


def test_get_node_for_path_if_setting_in_the_middle_should_raise_key_error():
    root = SettingNode("root")
    root.add_for_path([Setting(Parameter("stupid_animals"), 1.0)], "all_animals")
    with pytest.raises(ValueError, match="'stupid_animals' is a setting, not a node"):
        root.get_node_for_path("all_animals.stupid_animals.horse")


def test_model_dump_and_model_validate_reversibility():
    # This test is mainly testing that Pydantic does what it should,
    # but still good to have since we are using our own custom BaseModel,
    # and changes in shared "model_config" might change the Pydantic behaviour.
    child1 = SettingNode("child1")
    parameter = Parameter(name="parameter")
    setting = Setting(parameter=parameter, value=50)
    root = SettingNode("root", child1_key=child1, setting_key=setting)
    assert SettingNode.model_validate(root.model_dump()) == root
