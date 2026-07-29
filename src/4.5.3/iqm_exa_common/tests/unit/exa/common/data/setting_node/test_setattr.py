#  ********************************************************************************
#  Copyright (c) 2019-2024 IQM Finland Oy.
#  All rights reserved. Confidential and proprietary.
#
#  Distribution or reproduction of any information contained herein
#  is prohibited without IQM Finland Oy’s prior written permission.
#  ********************************************************************************
from pytest import fixture

from exa.common.data.parameter import Parameter, Setting
from exa.common.data.setting_node import SettingNode


@fixture
def node_1():
    return SettingNode(
        "node1",
        freq=Parameter("frequency", "Frequency", "Hz"),
        volt=Parameter("volt", "Voltage", "V").set(-0.01),
        align_name=False,
    )


@fixture
def node_2():
    return SettingNode(
        "node2",
        freq=Parameter("frequency", "Frequency", "Hz"),
        volt=Parameter("volt", "Voltage", "V").set(-0.01),
    )


def test_value_of_existing_setting_is_changed(node_1, node_2):
    # Test looks funny but better to check anyway, since there is special logic
    node_1.volt = 13
    assert node_1.volt.value == 13
    node_2.volt = 13
    assert node_2.volt.value == 13


def test_can_override_with_new_setting(node_1, node_2):
    new = Setting(Parameter("power", "Power", "dBm"), 15)
    node_1.volt = new
    assert node_1.volt == new
    node_2.volt = new
    assert node_2.volt == Setting(Parameter("volt", "Power", "dBm"), 15)


def test_can_override_with_new_parameter(node_1, node_2):
    param = Parameter("power", "Power", "dBm")
    node_1.volt = param
    assert node_1.volt == Setting(param, None)
    node_2.volt = param
    assert node_2.volt == Setting(Parameter("volt", "Power", "dBm"), None)


def test_can_override_with_new_node(node_1, node_2):
    node_1.volt = SettingNode("new", align_name=False)
    assert node_1.volt == SettingNode("new", align_name=False)
    assert "volt" not in node_1.settings
    node_2.volt = SettingNode("new")
    assert node_2.volt == SettingNode("volt")
    assert "volt" not in node_2.settings


def test_set_item_and_get_item(node_1, node_2):
    # align_name=False retains the original names, but still fixes the paths
    node_1["duck"] = Setting(Parameter("duckie"), 1.0, source={"type": "duck_source", "ducking": True})
    node_1["horse"] = SettingNode("horse", fish=Setting(Parameter("fishy"), 1.0), align_name=False)
    node_1["cat.dog.goose"] = Setting(Parameter("goosey"), 1.0)
    assert node_1["duck"] == Setting(Parameter("duckie"), 1.0)
    assert node_1["duck"].source == {"type": "duck_source", "ducking": True}
    assert node_1["duck"].path == "duck"
    assert node_1["horse"] == SettingNode("horse", fish=Setting(Parameter("fishy"), 1.0), align_name=False)
    assert node_1["horse"]["fish"].path == "horse.fish"
    assert node_1["cat"].name == "cat"
    assert node_1["cat"].path == "cat"
    assert node_1["cat.dog"].name == "dog"
    assert node_1["cat.dog"].path == "cat.dog"
    assert node_1["cat.dog.goose"] == Setting(Parameter("goosey"), 1.0)
    assert node_1["cat.dog.goose"].path == "cat.dog.goose"

    # SettingNode automatically aligns path and name
    node_2["duck"] = Setting(Parameter("duckie", label="duckie"), 1.0, source={"type": "duck_source", "ducking": True})
    node_2["horse"] = SettingNode("horse", fish=Setting(Parameter("fishy", label="fishy"), 1.0))
    node_2["cat.dog.goose"] = Setting(Parameter("goosey", label="goosey"), 1.0)
    assert node_2["duck"] == Setting(Parameter("duck", label="duckie"), 1.0)
    assert node_2["duck"].source == {"type": "duck_source", "ducking": True}
    assert node_2["duck"].path == "duck"
    assert node_2["horse"].name == "horse"
    assert node_2["horse"]["fish"] == Setting(Parameter("horse.fish", label="fishy"), 1.0)
    assert node_2["horse"]["fish"].path == "horse.fish"
    assert node_2["cat"].name == "cat"
    assert node_2["cat"].path == "cat"
    assert node_2["cat.dog"].name == "cat.dog"
    assert node_2["cat.dog"].path == "cat.dog"
    assert node_2["cat.dog.goose"] == Setting(Parameter("cat.dog.goose", label="goosey"), 1.0)
    assert node_2["cat.dog.goose"].path == "cat.dog.goose"
    # mix and match
    node_2["things.animals"] = SettingNode(  # this level does not align paths
        "animals",
        birds=SettingNode("birds", bird1=Setting(Parameter("bird1"), 1.0)),  # this level aligns
        align_name=False,
    )
    assert node_2["things.animals"].name == "animals"  # did not align
    assert node_2["things.animals.birds"].name == "things.animals.birds"  # aligns
    assert node_2["things.animals.birds.bird1"] == Setting(Parameter("things.animals.birds.bird1", label="bird1"), 1.0)


def test_set_source(node_2):
    tree = SettingNode("root")
    tree["node2"] = node_2
    tree["plop"] = Setting(Parameter("plop"), 1.0)
    tree["plip"] = Setting(Parameter("plip"), None)

    tree.set_source({"type": "idk"})
    assert tree["plip"].source is None
    assert tree["plop"].source == {"type": "idk"}
    assert tree.node2.freq.source is None
    assert tree.node2.volt.source == {"type": "idk"}

    tree.set_source({"type": "idk"}, ignore_nones=False)
    assert tree["plip"].source == {"type": "idk"}
    assert tree.node2.freq.source == {"type": "idk"}
