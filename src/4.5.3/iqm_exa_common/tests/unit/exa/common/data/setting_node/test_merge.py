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
def awg_node() -> SettingNode:
    return SettingNode(
        "awg_node",
        freq=Parameter("frequency", "Frequency", "Hz"),
        power=Parameter("power", "Power", "dBm").set(15),
        voltage=Parameter("voltage", "Voltage", "V").set(-0.01),
        align_name=False,
    )


@fixture
def qubit1_node(awg_node) -> SettingNode:
    return SettingNode("qubit1", wires=Parameter("wires", "Number of wires").set(1001), awg=awg_node, align_name=False)


@fixture
def qubit2_node(awg_node) -> SettingNode:
    return SettingNode("qubit2", wires=Parameter("wires", "Number of wires").set(2), awg=awg_node, align_name=False)


@fixture
def chip(qubit1_node, qubit2_node) -> SettingNode:
    return SettingNode("chip", q1=qubit1_node, q2=qubit2_node, align_name=False)


def test_merging_node_with_itself_does_not_change_it(chip):
    assert chip == SettingNode.merge(chip, chip)


def test_preserves_all_attributes(qubit1_node, qubit2_node):
    qubit1_node.hello = "hello"
    qubit2_node.something_else = "something else"
    merged = SettingNode.merge(qubit1_node, qubit2_node)
    assert merged.hello == "hello"
    assert merged.something_else == "something else"


def test_prioritizes_values_from_first_argument(qubit1_node, qubit2_node):
    qubit1_node.wires = 101
    qubit2_node.wires = 123
    assert SettingNode.merge(qubit1_node, qubit2_node).wires.value == 101


def test_does_not_prioritize_none_value(qubit1_node, qubit2_node):
    qubit1_node.wires = None
    qubit2_node.wires = 123
    assert SettingNode.merge(qubit1_node, qubit2_node).wires.value == 123


def test_prioritizes_structure_of_first_argument_in_case_of_conflict(qubit1_node, qubit2_node, awg_node):
    qubit1_node.wires = awg_node.model_copy(deep=False)
    qubit2_node.wires = 123
    merged = SettingNode.merge(qubit1_node, qubit2_node)
    assert merged.wires == awg_node


def test_merges_recursively(chip):
    chip2 = chip.model_copy(deep=False)
    chip.q1.awg.voltage = 555
    chip.q1.wires = 666
    merged = SettingNode.merge(chip, chip2)
    assert merged.q1.awg.voltage.value == 555
    assert merged.q1.wires.value == 666


def test_result_is_a_copy_not_a_reference(qubit1_node, qubit2_node):
    merged = SettingNode.merge(qubit1_node, qubit2_node)
    assert merged is not qubit1_node
    assert merged is not qubit2_node
    assert merged.wires is not qubit1_node.wires
    assert merged.awg is not qubit1_node.awg


def test_adds_all_nodes_that_are_not_common_to_both_inputs(qubit1_node, qubit2_node):
    n1 = SettingNode("n1", q1=qubit1_node, align_name=False)
    n2 = SettingNode("n2", q2=qubit2_node, align_name=False)
    merged = SettingNode.merge(n1, n2)
    assert merged.q1 == qubit1_node
    assert merged.q2 == qubit2_node


def test_align_name_and_path_when_merging():
    first = SettingNode(
        "first", more_stuff=SettingNode("more_stuff", other_thingy=Setting(Parameter("other_thingy"), 1.0))
    )
    second = SettingNode("second", stuff=SettingNode("stuff", thingy=Setting(Parameter("thingy"), 1.0)))
    merged = SettingNode.merge(first, second)
    assert merged["more_stuff.other_thingy"] == Setting(Parameter("more_stuff.other_thingy", label="other_thingy"), 1.0)


def test_advanced_arguments():
    first = SettingNode(
        "first",
        more_stuff=SettingNode("more_stuff", other_thingy=Setting(Parameter("other_thingy"), None)),
    )
    second = SettingNode("second", stuff=SettingNode("stuff", thingy=Setting(Parameter("thingy"), 1.0)))
    merged = SettingNode.merge(first, second, merge_nones=True, align_name=False, deep_copy=False)
    assert merged["more_stuff.other_thingy"] == Setting(
        Parameter("more_stuff.other_thingy", label="other_thingy"), None
    )
    merged["more_stuff.inserted_to_original"] = Setting(Parameter("inserted_to_original", label="inserted_to_original"))
    # without deepcopying, this was now inserted also to the original one
    assert first["more_stuff.inserted_to_original"] == Setting(
        Parameter("more_stuff.inserted_to_original", label="inserted_to_original")
    )
    # test name/path prefixes get preserved correctly when merging
    wrapped_first = SettingNode("first_wrap", content=first)
    wrapped_second = SettingNode("second_wrap", content=second)
    wrapped_merged = SettingNode.merge(
        wrapped_first.content, wrapped_second.content, merge_nones=True, align_name=False, deep_copy=False
    )
    assert wrapped_merged.more_stuff.name == "content.more_stuff"
