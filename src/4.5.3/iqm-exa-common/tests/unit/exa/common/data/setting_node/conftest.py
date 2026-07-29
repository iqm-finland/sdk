#  ********************************************************************************
#  Copyright (c) 2019-2024 IQM Finland Oy.
#  All rights reserved. Confidential and proprietary.
#
#  Distribution or reproduction of any information contained herein
#  is prohibited without IQM Finland Oy’s prior written permission.
#  ********************************************************************************
import numpy as np
import pytest

from exa.common.data.parameter import CollectionType, Parameter, Setting
from exa.common.data.setting_node import SettingNode


@pytest.fixture
def node_1():
    return SettingNode(
        "node1",
        freq=Parameter("freq", "Frequency", "Hz"),
        power=Parameter("power", "Power", "dBm").set(15),
        volt=Parameter("volt", "Voltage", "V").set(-0.01),
        align_name=False,
    )


@pytest.fixture
def node_2(node_1):
    return SettingNode(
        "node2",
        freq=Parameter("freq", "Frequency", "Hz").set(6),
        subtree=node_1,
        align_name=False,
    )


@pytest.fixture
def node_3(node_2):
    ret = SettingNode(
        "node3",
        qubit=Parameter("qubit", "Number of qubit").set(1001),
        horse=Parameter("horse", "Number of horses"),
        rest=node_2,
        align_name=False,
    )
    ret.non_tree_attribute = "hello"
    return ret


@pytest.fixture
def node_4(node_1):
    ret = SettingNode(
        "node4",
        qubit=SettingNode("qubitnode", some=Parameter("fish", "Fish").set(4.4), align_name=False),
        rest=SettingNode(
            "somenode",
            fish=Parameter("fish", "Fish").set(4.4),
            subtree=node_1.model_copy(deep=False),
            align_name=False,
        ),
        align_name=False,
    )
    ret.non_tree_attribute = "hello"
    ret.rest.non_tree_attribute = "hi"
    ret.rest.subtree.volt = 1.111  # value changed
    ret.rest.subtree.power = None  # value reset
    return ret


@pytest.fixture
def node_5():
    ret = SettingNode(
        "fish_and_horses",
        horses=SettingNode(
            "horses",
            horse1=Parameter("horse1").set(1.0),
            other_horses=Parameter("other_horses", collection_type=CollectionType.LIST).set([2.0, 3.0]),
            align_name=False,
        ),
        fish1=Setting(Parameter("fish1"), 4.0),
        other_fish=Setting(
            Parameter("other_fish", collection_type=CollectionType.NDARRAY), np.array([[5.0, 6.0], [7.0, 8.0]])
        ),
        align_name=False,
    )
    return ret
