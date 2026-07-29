from exa.common.data.parameter import Parameter, Setting
from exa.common.data.setting_node import SettingNode


def test_index_dirty_after_construction(node_1) -> None:
    # Index is initially dirty, we build it lazily on the first find_by_name call
    assert hasattr(node_1, "_index_dirty")
    assert node_1._index_dirty is True


def test_rebuild_index_after_find_by_name(node_1) -> None:
    # Index is initially dirty, we build it lazily on the first find_by_name call
    assert node_1._index_dirty is True
    assert len(node_1._index_by_name) == 0
    assert len(node_1._index_nodes_by_type) == 0
    # find_by_name must trigger _rebuild_index and clean the flag
    freq = node_1.find_by_name("freq")
    assert freq.parameter == Parameter("freq", "Frequency", "Hz")
    assert node_1._index_dirty is False
    assert len(node_1._index_by_name) == 4
    assert len(node_1._index_nodes_by_type) == 0


def test_mark_index_dirty_after_setattr(node_1) -> None:
    node_1._rebuild_index()
    assert node_1.freq.value is None
    assert node_1._index_dirty is False

    node_1.freq = Setting(parameter=Parameter(name="freq", label="Frequency", unit="Hz"), value=1.0)
    assert node_1.freq.value == 1.0
    assert node_1._index_dirty is True


def test_mark_index_dirty_after_delattr(node_1) -> None:
    node_1._rebuild_index()
    assert hasattr(node_1, "freq")
    assert node_1._index_dirty is False

    del node_1.freq
    assert not hasattr(node_1, "freq")
    assert node_1._index_dirty is True


def test_mark_index_dirty_after_setitem(node_1) -> None:
    node_1._rebuild_index()
    assert node_1["freq"].value is None
    assert node_1._index_dirty is False

    node_1["freq"] = Setting(parameter=Parameter(name="freq", label="Frequency", unit="Hz"), value=1.0)
    assert node_1["freq"].value == 1.0
    assert node_1._index_dirty is True


def test_mark_index_dirty_after_delitem(node_1) -> None:
    node_1._rebuild_index()
    assert hasattr(node_1, "freq")
    assert node_1._index_dirty is False

    del node_1["freq"]
    assert not hasattr(node_1, "freq")
    assert node_1._index_dirty is True


def test_mark_index_dirty_after_update_setting(node_1) -> None:
    node_1._rebuild_index()
    assert node_1.freq.value is None
    assert node_1._index_dirty is False

    setting = Setting(parameter=Parameter(name="freq", label="Frequency", unit="Hz"), value=1.0)
    node_1.update_setting(setting)
    assert node_1.freq.value == 1.0
    assert node_1._index_dirty is True


def test_mark_index_dirty_after_merge_values(node_1) -> None:
    node_1._rebuild_index()
    assert node_1.freq.value is None
    assert node_1._index_dirty is False

    merge_node = node_1.model_copy(deep=True)
    merge_node.freq = Setting(parameter=Parameter(name="freq", label="Frequency", unit="Hz"), value=1.0)
    assert node_1._index_dirty is False

    node_1.merge_values(merge_node)
    assert node_1.freq.value == 1.0
    assert node_1._index_dirty is True


def test_mark_index_dirty_after_prune(node_4) -> None:
    node_4._rebuild_index()
    assert hasattr(node_4, "qubit")
    assert hasattr(node_4, "rest")
    assert node_4._index_dirty is False

    prune_node = SettingNode("prune_node")

    node_4.prune(prune_node)
    assert not hasattr(node_4, "qubit")
    assert not hasattr(node_4, "rest")
    assert node_4._index_dirty is True


def test_mark_index_dirty_after_set_from_dict(node_4) -> None:
    node_4._rebuild_index()
    assert node_4.rest.subtree.freq.value is None
    assert node_4._index_dirty is False

    new_values = {
        "rest": {
            "subtree": {
                "freq": 44,
            }
        }
    }
    node_4.set_from_dict(new_values)
    assert node_4.rest.subtree.freq.value == 44
    assert node_4._index_dirty is True


def test_mark_index_dirty_after_add_for_path() -> None:
    root = SettingNode("root")
    root._rebuild_index()
    assert root._index_dirty is False
    subnode = SettingNode("duck")
    subnode.add_for_path([Setting(Parameter("goose"), 2.0)], "moose")
    root.add_for_path(subnode, "all_animals.stupid_animals")

    assert root.all_animals.stupid_animals.duck.moose.goose.value == 2.0
    assert root._index_dirty is True


def test_mark_index_dirty_after_set_source() -> None:
    root = SettingNode("root")
    root["plop"] = Setting(Parameter("plop"), 1.0)
    root._rebuild_index()
    assert root._index_dirty is False

    root.set_source({"type": "idk"})
    assert root["plop"].source == {"type": "idk"}
    assert root._index_dirty is True


def test_nodes_by_type_caches_index_nodes_by_type(node_4) -> None:
    assert len(node_4._index_nodes_by_type) == 0

    settings = list(node_4.nodes_by_type(Setting, recursive=True))
    assert len(settings) == 5
    assert len(node_4._index_nodes_by_type) == 1
    assert (Setting,) in node_4._index_nodes_by_type.keys()
    assert len(node_4._index_nodes_by_type[(Setting,)]) == 5

    setting_nodes = list(node_4.nodes_by_type(SettingNode, recursive=True))
    assert len(setting_nodes) == 4
    assert len(node_4._index_nodes_by_type) == 2
    assert (SettingNode,) in node_4._index_nodes_by_type.keys()
    assert len(node_4._index_nodes_by_type[(SettingNode,)]) == 4

    all_nodes = list(node_4.nodes_by_type((Setting, SettingNode), recursive=True))
    assert len(all_nodes) == 9
    assert len(node_4._index_nodes_by_type) == 3
    assert (Setting, SettingNode) in node_4._index_nodes_by_type.keys()
    assert len(node_4._index_nodes_by_type[(Setting, SettingNode)]) == 9


def test_root_index_rebuild_leaves_subnodes_lazy():
    """Confirms that rebuilding the root index does not prematurely initialize sub-node indices.

    This ensures memory efficiency by only building the local index for the node being accessed.
    Sub-nodes remain 'dirty' until they are directly queried.
    """
    root = SettingNode("root", sub=SettingNode("sub", leaf=Parameter("p", "P")))

    assert root._index_dirty is True
    assert root.sub._index_dirty is True

    _ = list(root.all_settings)

    assert root._index_dirty is False, "Root index should be clean after lazy build."
    assert root.sub._index_dirty is True, "Sub-node index should remain lazy/dirty."


def test_manual_invalidation_propagates_dirty_signal():
    """Confirms that the dirty signal propagates up the tree, and the root rebuilds lazily."""
    root = SettingNode("root", sub=SettingNode("sub", leaf=Parameter("p", "P")))
    _ = list(root.all_settings)
    assert root._index_dirty is False

    root.sub._mark_index_dirty()

    assert root.sub._index_dirty is True
    assert root._index_dirty is True, "Dirty signal failed to propagate up to the root."


def test_subtree_replacement_links_parent_and_propagates_dirty_signal():
    """Confirms that replacing an entire child node/setting establishes the parent link
    and propagates the dirty signal to the root.
    """
    root = SettingNode("root", sub=SettingNode("sub", leaf=Parameter("p", "P")))
    _ = list(root.all_settings)
    assert root._index_dirty is False

    # Action: Replace a subtree with a new instance
    new_sub = SettingNode("new_sub", leaf=Parameter("p2", "P2"))
    root.sub = new_sub

    # Verification
    assert root._index_dirty is True, "Root failed to detect subtree replacement."
    parent = root.sub._parent
    assert parent is not None and parent.name == "root", "Parent link was lost or incorrect."

    # Recovery check
    _ = list(root.all_settings)
    assert root._index_dirty is False


def test_leaf_value_assignment_links_parent_and_propagates_dirty_signal():
    """Confirms that assigning a value directly to a node's attribute establishes the parent link
    and propagates the dirty signal to the root.
    """
    root = SettingNode("root", sub=SettingNode("sub", leaf=Parameter("p", "P")))
    _ = list(root.all_settings)
    assert root._index_dirty is False

    # Action: SettingNode convenience syntax (calls Setting.update)
    root.sub.leaf = 2.0

    # Verification
    assert root.sub.leaf.value == 2.0
    assert root._index_dirty is True, "Dirty signal failed to propagate after value assignment."
    parent = root.sub.leaf._parent
    assert parent is not None and parent.name == "sub", "Parent link was lost or incorrect."


def test_leaf_private_attribute_mutation_links_parent_and_propagates_dirty_signal():
    """Confirms that mutating a leaf's private attribute (e.g., _source) establishes the parent link
    and propagates the dirty signal to the root.
    """
    root = SettingNode("root", sub=SettingNode("sub", leaf=Parameter("p", "P")))
    _ = list(root.all_settings)
    assert root._index_dirty is False

    # Action: Direct private attribute mutation
    root.sub.leaf._source = {"type": "manual", "user": "admin"}

    # Verification
    assert root._index_dirty is True, "Dirty signal failed to propagate after private attribute mutation."
    parent = root.sub.leaf._parent
    assert parent is not None and parent.name == "sub", "Parent link was lost or incorrect."


def test_index_dicts_not_shared():
    a = SettingNode("a", generate_paths=False)
    b = SettingNode("b", generate_paths=False)
    assert a._index_by_name is not b._index_by_name
    assert a._index_nodes_by_type is not b._index_nodes_by_type


def test_parent_links_survive_ensure_index_attributes():
    root = SettingNode("root", sub=SettingNode("sub", leaf=Parameter("leaf")))
    sub = root.sub
    leaf = sub.leaf

    assert sub._parent is root
    assert leaf._parent is sub

    sub._ensure_index_attributes()

    assert sub._parent is root, "Sub-node lost its parent reference after indexing"
    assert leaf._parent is sub, "Leaf-node lost its parent reference after indexing"


def test_link_children_recursive_repairs_subtree_parent_links():
    # Build a subtree with nested nodes and a leaf Setting
    subtree = SettingNode(
        "sub",
        inner=SettingNode(
            "inner",
            leaf=Parameter("leaf", value=None),
        ),
    )
    root = SettingNode("root")

    # Attach subtree without going through __setattr__/__setitem__ (simulates copy/update paths
    # where dicts get swapped in directly)
    root.subtrees["sub"] = subtree

    # Intentionally break an internal parent link:
    # leaf Setting should have parent "inner", but we point it to "sub" to simulate stale pointers
    leaf_setting = subtree.inner.leaf
    leaf_setting._parent = subtree  # wrong on purpose

    # Now repair: recursive relinking should fix the full subtree consistently
    root._link_children_recursive()

    assert root.sub._parent is root
    assert root.sub.inner._parent is root.sub
    assert root.sub.inner.leaf._parent is root.sub.inner
