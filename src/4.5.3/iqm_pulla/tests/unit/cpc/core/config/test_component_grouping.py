# Copyright 2024-2025 IQM
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest

from iqm.cpc.core.config import ComponentGrouping, ComponentGroupingMode


def test_init():
    components = ComponentGrouping(["a", "b"])
    assert components == ["a", "b"]
    assert components.grouping_mode == ComponentGroupingMode.LIST

    components = ComponentGrouping([("a", "b"), ("c", "d")])
    assert components == [("a", "b"), ("c", "d")]
    assert components.grouping_mode == ComponentGroupingMode.GROUP

    components = ComponentGrouping([[("a", "b"), ("c", "d")], [("a", "c"), ("b", "d")]])
    assert components == [[("a", "b"), ("c", "d")], [("a", "c"), ("b", "d")]]
    assert components.grouping_mode == ComponentGroupingMode.COLOUR_GROUP

    component_grouping = ComponentGrouping(["a", "b"])
    components = ComponentGrouping(component_grouping)
    assert components == ["a", "b"]
    assert components.grouping_mode == ComponentGroupingMode.LIST


@pytest.mark.parametrize(
    "components",
    [
        ["a", 1],
        [("a", "b"), "c"],
        [("a", "b"), ("c", 2)],
        [[("a", "b"), ("c", "d")], "e"],
        [[("a", "b"), ("c", "d")], ("e", "f")],
        [[[("a", "b"), ("c", "d")]]],
    ],
)
def test_init_with_malformed_input(components):
    match = "Provided components are malformed"
    with pytest.raises(ValueError, match=match):
        ComponentGrouping(components)


def test_flatten():
    assert ComponentGrouping(["a", "b", "a", "c", "b", "a", "d", "c"]).flatten() == ["a", "b", "c", "d"]
    assert ComponentGrouping([("a", "b"), ("b", "c"), ("a", "c")]).flatten() == ["a", "b", "c"]
    assert ComponentGrouping([[("a", "b"), ("c", "d")], [("a", "c"), ("b", "d")]]).flatten() == ["a", "b", "c", "d"]


def test_contains_with_strings():
    components = ComponentGrouping(["a", "b", "c", "d"])
    assert components.contains("a")
    assert not components.contains("e")

    components = ComponentGrouping([("a", "b"), ("c", "d")])
    assert components.contains("a")
    assert not components.contains("e")

    components = ComponentGrouping([[("a", "b")], [("c", "d")]])
    assert components.contains("a")
    assert not components.contains("e")


def test_contains_with_groups():
    components = ComponentGrouping(["a", "b", "c", "d"])
    assert components.contains(("a", "b"))
    assert not components.contains(("e", "f"))
    assert not components.contains(("a", "b", "c", "e"))

    components = ComponentGrouping([("a", "b"), ("c", "d")])
    assert components.contains(("a", "b"))
    assert not components.contains(("a", "e"))
    assert not components.contains(("b", "c"))

    components = ComponentGrouping([[("a", "b")], [("c", "d")]])
    assert components.contains(("a", "b"))
    assert not components.contains(("a", "e"))
    assert not components.contains(("b", "c"))


def test_contains_with_colour_groups():
    components = ComponentGrouping(["a", "b", "c", "d"])
    assert components.contains([("a", "b"), ("c", "d")])
    assert not components.contains([("a", "b"), ("c", "e")])

    components = ComponentGrouping([("a", "b"), ("c", "d")])
    assert components.contains([("a", "b"), ("c", "d")])
    assert components.contains([("a", "b")])
    assert not components.contains([("a", "b"), ("a", "c")])

    components = ComponentGrouping([[("a", "b"), ("c", "d")], [("e", "f"), ("h", "i")]])
    assert components.contains([("a", "b"), ("c", "d")])
    assert not components.contains([("a", "b")])


def test_limit_by_components():
    components = ComponentGrouping(["a", "b", "a", "c", "a", "b", "d"])
    assert components.limit_by_components(["a", "b"]) == ["a", "b", "a", "a", "b"]

    components = ComponentGrouping([("a", "b"), ("a", "c"), ("b", "a"), ("d", "e")])
    assert components.limit_by_components(["a", "b"]) == [("a", "b"), ("b", "a")]

    components = ComponentGrouping([[("a", "b"), ("a", "c")], [("b", "a"), ("d", "e")], [("a", "c")]])
    assert components.limit_by_components(["a", "b"]) == [[("a", "b")], [("b", "a")]]

    with pytest.raises(ValueError, match="Cannot limit a flat list"):
        ComponentGrouping(["a", "b", "a", "c", "a", "b", "d"]).limit_by_components([("a", "b")])

    components = ComponentGrouping([("a", "b"), ("a", "c"), ("b", "a"), ("d", "e")])
    assert components.limit_by_components([("a", "b"), ("a", "c"), ("a", "e")]) == [("a", "b"), ("a", "c")]

    components = ComponentGrouping([[("a", "b"), ("a", "c")], [("b", "a"), ("d", "e")], [("a", "c")]])
    assert components.limit_by_components([("a", "b"), ("a", "c"), ("a", "e")]) == [
        [("a", "b"), ("a", "c")],
        [("a", "c")],
    ]


def test_to_json_serializable():
    # We need to "pre-serialize" component grouping before sending it to the station control.
    # In monolithic mode, this would not be necessary since serialization/deserialization is not used.
    # But in remote mode, "additional_run_properties" has to be serialized to be sent over HTTP.
    # Serialization is done with protobuf, which doesn't support tuples out-of-the-box,
    # but instead expects a JSON serializable dict. Thus, tuples has to be converted to lists.
    # We do it in exa-experiment since our API can't take responsibility to support freeform data in dicts.
    # However, we could consider adding support for generic tuple support, and then this could be removed.

    component_grouping = ComponentGrouping(["a", "b"])
    components = ComponentGrouping.to_json_serializable(component_grouping)
    assert components == ["a", "b"]

    component_grouping = ComponentGrouping([("a", "b"), ("c", "d")])
    components = ComponentGrouping.to_json_serializable(component_grouping)
    assert components == [["a", "b"], ["c", "d"]]

    component_grouping = ComponentGrouping([[("a", "b"), ("c", "d")], [("a", "c"), ("b", "d")]])
    components = ComponentGrouping.to_json_serializable(component_grouping)
    assert components == [[["a", "b"], ["c", "d"]], [["a", "c"], ["b", "d"]]]
