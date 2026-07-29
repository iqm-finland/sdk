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


def test_model_copy(node_3):
    model_copy = node_3.model_copy(deep=True)
    assert model_copy is not node_3
    assert node_3 == model_copy
    # Setting and Parameter are immutable, i.e. we reuse the same instance
    assert model_copy["qubit"] is node_3["qubit"]
    assert model_copy["horse"] is node_3["horse"]
    assert model_copy["qubit"].parameter is node_3["qubit"].parameter
    assert model_copy.non_tree_attribute == "hello"


def test_copy_works_but_is_deprecated(node_3):
    with pytest.warns(DeprecationWarning, match="`copy` method is deprecated since 2025-03-28"):
        model_copy = node_3.copy()

    assert model_copy is not node_3
    assert node_3 == model_copy
    # Setting and Parameter are immutable, i.e. we reuse the same instance
    assert model_copy["qubit"] is node_3["qubit"]
    assert model_copy["horse"] is node_3["horse"]
    assert model_copy["qubit"].parameter is node_3["qubit"].parameter
    assert model_copy.non_tree_attribute == "hello"


def test_merge_values(node_3, node_4):
    original = node_3.model_copy(deep=False)
    node_3.merge_values(node_3)
    assert original == node_3

    original = node_4.model_copy(deep=True)
    node_4.merge_values(node_3)
    assert not hasattr(node_4, "horse")
    assert not hasattr(node_4.rest, "freq")
    assert hasattr(node_4.rest, "fish")
    assert node_4["rest"] is not original["rest"]
    assert node_4.rest.subtree.volt.value == 1.111
    assert node_4.rest.subtree.power.value == 15
    assert node_4.rest.fish.value == 4.4
    assert node_4.non_tree_attribute == "hello"
    assert node_4.rest.non_tree_attribute == "hi"


def test_merge_values_does_not_change_anything_else_than_value(node_1):
    other = node_1.model_copy(deep=True)
    node_1.freq = 6
    other.freq = Setting(Parameter("other_name"), 3)
    other.merge_values(node_1, prioritize_other=True)
    assert other.freq == Setting(Parameter("other_name"), 6)


def test_update_setting_with_elemental_parameters(node_5):
    tree = node_5.model_copy(deep=False)
    setting = Setting(Parameter("fish1"), 10.0)
    tree.update_setting(setting)
    assert tree["fish1"].value == 10.0

    tree = node_5.model_copy(deep=False)
    setting = Setting(Parameter("horse1"), 10.0)
    tree.update_setting(setting)
    assert tree["horses"]["horse1"].value == 10.0

    tree = node_5.model_copy(deep=False)
    setting = Parameter("other_horses", collection_type=CollectionType.LIST).create_element_parameter_for(1).set(10.0)
    tree.update_setting(setting)
    assert tree["horses"]["other_horses"].value == [2.0, 10.0]

    tree = node_5.model_copy(deep=False)
    setting = (
        Parameter("other_fish", collection_type=CollectionType.NDARRAY).create_element_parameter_for([1, 0]).set(10.0)
    )
    tree.update_setting(setting)
    assert np.allclose(tree["other_fish"].value, np.array([[5.0, 6.0], [10.0, 8.0]]))


def test_prune(node_4):
    smaller = node_4.model_copy(deep=False)
    del smaller.qubit
    del smaller.rest.subtree
    node_4.prune(smaller)

    assert not hasattr(node_4, "qubit")
    assert not hasattr(node_4.rest, "subtree")
    assert hasattr(node_4.rest, "fish")


def test_set_from_dict_ignores_other_keys(node_4):
    new_values = {
        "something_else": {
            "foo": {
                "freq": 44,
                "volt": -1,
            }
        }
    }
    not_changed = node_4.model_copy(deep=False)
    not_changed.set_from_dict(new_values)
    assert node_4 == not_changed
