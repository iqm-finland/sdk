# Copyright 2024 IQM
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

# mypy: ignore-errors

"""A tree-structured container for :class:`Settings <exa.common.data.parameter.Setting>`.

The :class:`.SettingNode` class combines a bunch of Settings together.
It may also contain other SettingNodes.
Together, the contents form a tree structure that provides a useful way of grouping Settings.

As an example, we manually construct a tree of SettingNodes with some dummy Settings, but it is usually not necessary.
The root node in the following examples is called ``'node'``.

.. testsetup:: pulse

    from exa.common.data.parameter import Setting, Parameter
    from exa.common.data.setting_node import SettingNode
    node = SettingNode('root',
        flux=SettingNode('root.flux',
            voltage=Setting(Parameter('root.flux.voltage', 'Voltage', 'V'), 1.5),
            resistance=Setting(Parameter('root.flux.resistance', 'Resistance', 'Ohm'), None),
        ),
        pulse=SettingNode('root.pulse',
            amplitude=Setting(Parameter('root.pulse.amplitude', 'Amplitude'), 1.0),
            duration=Setting(Parameter('root.pulse.duration', 'Duration', 's'), 100e-9),
       )
    )


What's inside?
--------------

The easiest way to see the content of the node is the :meth:`.SettingNode.print_tree` method:

.. doctest:: pulse
   :options: +NORMALIZE_WHITESPACE

    >>> node.print_tree(levels=1)
     "root"
     ║
     ╚═ flux: "root.flux"
     ╚═ pulse: "root.pulse"


We see that the ``'root'`` node has two children, named ``'root.flux'`` and ``'root.pulse'``, which
themselves are also SettingNodes.
This follows the typical naming convention in EXA: Subnodes include the names of their parents, separated by a dot.

.. doctest:: pulse
   :options: +NORMALIZE_WHITESPACE

    >>> node.print_tree()
     "root"
     ║
     ╠═ flux: "root.flux"
     ║   ╠─ voltage: Voltage = 1.5 V
     ║   ╚─ resistance: Resistance = None (automatic/unspecified)
     ╚═ pulse: "root.pulse"
         ╠─ amplitude: Amplitude = 1.0
         ╚─ duration: Duration = 1e-07 s


The children contain some dummy Settings, showing the keys, labels and current values.

For other ways to access the content of the node, see also :attr:`.SettingNode.children`,
:attr:`.SettingNode.all_settings`, and :meth:`.SettingNode.nodes_by_type`.


Get and set values
------------------

The values within the nodes can be accessed using the attribute or dictionary syntax:


.. doctest:: pulse

    >>> node.pulse.amplitude.value
    1.0
    >>> node['flux']['voltage'].value
    1.5

The values can be changed with a simple ``=`` syntax:

.. doctest:: pulse

    >>> node.pulse.amplitude = 1.4
    >>> node.pulse.amplitude.value
    1.4

.. note::

    ``node.setting`` refers to the Setting object. ``node.setting.value`` syntax refers to the data stored inside.

``SettingNode`` also supports "the path notation" by default (but not if ``align_name`` is set to ``False``,
since it cannot be made to work consistently if nodes are allowed to be named differently from their paths):

.. doctest:: pulse

    >>> node['flux.voltage']

is the same as ``node['flux']['voltage']``.

Basic manipulation
------------------

Adding and deleting new Settings and nodes is simple:

.. doctest:: pulse

    >>> modified = node.copy()
    >>> del modified.flux # removes the node
    >>> del modified.pulse.amplitude # removes the Setting
    >>> modified.pulse.my_new_setting = Setting(Parameter('my name'), 33)

It is usually a good idea to make a copy of the original node, so that it won't be modified accidentally.

The path notation of ``SettingNode`` also works when inserting:

.. doctest:: pulse

    >>> node['flux.my.new.path.foo'] = Setting(Parameter('foo'), 1.0)

Any nodes that did not already exist under ``node`` will be inserted (in this case ``flux`` already existed, but
the rest not, so under ``flux`` the nodes ``my``, ``new``, and ``path`` would be added), and then finally the
value is added as child to the final node. Note: ``SettingNode`` always alings the path and name of any nodes under it,
so this would result in the new setting being renamed as "flux.my.new.path.foo":

.. doctest:: pulse

    >>> node['flux.my.new.path.foo'] = Setting(Parameter('bar'), 1.0)

If ``align_name`` is set to ``False", the name and path of nodes are not automatically aligned, but otherwise the above
path notation will still work. The added nodes will be named by just their path fragments ("my", "new", "path", and
so on), and the Setting will be added under the key "foo", but it will still retain its name "bar". Note: the root node
name will always be excluded from the paths (and names when they are aligned with the path), so that the path of
``root.foo.bar`` is ``"foo.bar"``.

To merge values of two SettingNodes, there are helpers :meth:`.SettingNode.merge` and
:meth:`.SettingNode.merge_values`.

The first one merges the tree structure and values of two nodes and outputs a third one as a result.
``None`` values are always replaced by a proper value if such exists. In case of conflicting nodes or values,
the content of the first argument takes priority.

.. doctest:: pulse
    :options: +NORMALIZE_WHITESPACE

    >>> result = SettingNode.merge(node.flux, node.pulse)
    >>> result.print_tree()
     "root.flux"
     ╠─ amplitude: Amplitude = 1.4
     ╠─ duration: Duration = 1e-07 s
     ╚─ voltage: Voltage = 1.5 V


Note how the result has values from ``node.flux``, but also settings ``node.pulse`` that do not exist in ``node.flux``.

The :meth:`.SettingNode.merge_values` method is an in-place operation that only changes
the values of Settings that already exist in the node, if possible:

.. doctest:: pulse
    :options: +NORMALIZE_WHITESPACE

    >>> modified = node.copy()
    >>> modified.flux.voltage = 222
    >>> modified.flux.resistance = 333
    >>> node.merge_values(modified, prioritize_other=True)
    >>> node.print_tree()
     "root"
     ║
     ╠═ flux: "root.flux"
     ║   ╠─ voltage: Voltage = 222 V
     ║   ╚─ resistance: Resistance = 333 Ohm
     ╚═ pulse: "root.pulse"
         ╠─ amplitude: Amplitude = 1.4
         ╚─ duration: Duration = 1e-07 s

Sometimes, it is easier to collect values in a dictionary and set them all at once by using
:meth:`.SettingNode.set_from_dict`. The nested structure of the dictionary should match
the structure of the SettingNode. Keys that are not found in the tree are silently ignored, unless the ``strict``
flag is used.

.. doctest:: pulse
    :options: +NORMALIZE_WHITESPACE

    >>> values_to_set = {'flux': {'resistance': 0.001}, 'ignored_entry': 234}
    >>> node.set_from_dict(values_to_set)
    >>> node.flux.print_tree()
     "root.flux"
     ╠─ voltage: Voltage = 222 V
     ╚─ resistance: Resistance = 0.001 Ohm


"""

from __future__ import annotations

from collections.abc import Generator, ItemsView, Iterable, Iterator, Mapping
from copy import copy
from itertools import permutations
import logging
import numbers
import pathlib
from typing import Any, Self

import jinja2
import numpy as np

from exa.common.data.base_model import BaseModel
from exa.common.data.parameter import CollectionType, Parameter, Setting, SourceType
from exa.common.errors.exa_error import UnknownSettingError
from exa.common.helpers import json_helper
from exa.common.qcm_data.chip_topology import sort_components

logger = logging.getLogger(__name__)


def _fix_path_recursive(node: SettingNode, path: str) -> SettingNode:
    """Recursively rebuilds a SettingNode tree with updated hierarchical paths.

    Instead of deep-copying the entire tree, this method performs a high-speed
    reconstruction using model_construct and fast_construct. This bypasses
    Pydantic's validation and setter overhead, which is safe because the
    source node is already validated.

    Args:
        node: The source SettingNode to replicate.
        path: The new hierarchical path prefix for this node.

    Returns:
        A new SettingNode instance with corrected paths and established parent links.

    """
    align_name = node.align_name
    settings: dict[str, Setting] = {}

    for key, setting in node.settings.items():
        child_path = f"{path}.{key}"
        old_param = setting.parameter

        if align_name:
            new_param = Parameter.model_construct(
                name=child_path,
                label=old_param.label,
                unit=old_param.unit,
                data_type=old_param.data_type,
                collection_type=old_param.collection_type,
                element_indices=old_param.element_indices,
            )
            object.__setattr__(new_param, "_parent_name", old_param._parent_name)
            object.__setattr__(new_param, "_parent_label", old_param._parent_label)
            if old_param.element_indices is not None:
                new_param._finalize_element_indices()
        else:
            new_param = old_param

        settings[key] = Setting.fast_construct(
            parameter=new_param,
            value=setting.value,
            read_only=setting.read_only,
            path=child_path,
            source=setting._source,
        )

    # Recurse first: children return already-linked sub-subtrees
    subtrees = {key: _fix_path_recursive(subnode, f"{path}.{key}") for key, subnode in node.subtrees.items()}

    new_node = node.model_copy(
        update={"path": path, "settings": settings, "subtrees": subtrees, "name": path if align_name else node.name},
        deep=False,
    )
    # Link children to the new parent instance so that cache invalidation works correctly
    new_node._link_children()

    return new_node


_INDEX_ATTRIBUTES = {
    "_parent": lambda: None,
    "_index_by_name": dict,
    "_index_nodes_by_type": dict,
    "_index_dirty": lambda: True,
}
"""Names and default value factories of private index attributes on :class:`.SettingNode`.

These attributes store cached indices used when querying nodes
from the tree and are treated as internal implementation details.
"""
# - ``_parent``: A back-pointer to the parent node, used for dirty-flag bubbling.
# - ``_index_by_name``: Maps names to their corresponding nodes.
# - ``_index_nodes_by_type``: Groups nodes based on their type.
# - ``_index_dirty``: Flags whether the index structures require rebuilding.
#
# We keep this variable to explicitly distinguish these index-related
# privates from other private attributes on a node. Normal private
# attributes may themselves hold :class:`.SettingNode`, :class:`.Setting`,
# or :class:`.Parameter` objects and are therefore processed like regular
# child attributes, whereas the attributes listed here must always behave as
# plain Python objects and never as part of the setting-tree structure.


class SettingNode(BaseModel):  # noqa: PLW1641 (Object does not implement `__hash__` method)
    """A tree-structured :class:`.Setting` container.

    Each child of the node is a :class:`.Setting`, or another :class:`SettingNode`.
    Iterating over the node returns all children, recursively.
    Settings can be accessed by dictionary syntax or attribute syntax:

    .. doctest::

        >>> from exa.common.data.parameter import Parameter
        >>> from exa.common.data.setting_node import SettingNode
        >>> p1 = Parameter("voltage", "Voltage")
        >>> f1 = Parameter("frequency", "Frequency")
        >>> sub = SettingNode("sub", frequency=f1)
        >>> settings = SettingNode('name', voltage=p1)
        >>> settings.voltage.parameter is p1
        True
        >>> settings['voltage'].parameter is p1
        True
        >>> settings.voltage.value is None
        True
        >>> settings.voltage = 7  # updates to Setting(p1, 7)
        >>> settings.voltage.value
        7
        >>> settings["sub.frequency"] = 8
        >>> settings["sub.frequency"].value
        8

    Args:
        name: Name of the node.
        settings: Dict of setting path fraqment names (usually the same as the setting name) to the settings. Mostly
            used when deserialising and otherwise left empty.
        subtrees: Dict of child node path fraqment names (usually the same as the child node name) to the settings.
            Mostly used when deserialising and otherwise left empty.
        path: Optionally give a path for the node, by default empty.
        align_name: Whether to align `node.name = node.path` in this node and all of its child nodes, by default True.
        generate_paths: If set ``True``, all subnodes will get their paths autogenerated correctly. Only set to
            ``False`` if the subnodes already have correct paths set (e.g. when deserialising).
        kwargs: The children given as keyword arguments. Each argument must be a :class:`.Setting`,
            :class:`.Parameter`, or a :class:`SettingNode`. The keywords are used as the names of the nodes.
            Parameters will be cast into Settings with the value ``None``.

    """

    name: str
    settings: dict[str, Setting] = {}
    subtrees: dict[str, SettingNode] = {}
    path: str = ""

    align_name: bool = True

    def __init__(
        self,
        name: str,
        settings: dict[str, Any] | None = None,
        subtrees: dict[str, Any] | None = None,
        *,
        path: str = "",
        align_name: bool = True,
        generate_paths: bool | None = None,
        **kwargs: Any,
    ):
        # Manually initialise internal attributes. This bypasses Pydantic’s
        # costly `init_private_attributes` step, since these fields are not
        # part of the validation logic and don’t require Pydantic handling.
        self._ensure_index_attributes()

        settings = settings or {}
        subtrees = subtrees or {}

        self._populate_children_from_kwargs(settings, subtrees, kwargs)

        super().__init__(
            name=name,
            settings=settings,
            subtrees=subtrees,
            path=path,
            align_name=align_name,
            **kwargs,
        )

        # Decide default behaviour for generate_paths only if not explicitly given
        if generate_paths is None:
            # If path is empty, we’re likely creating a fresh tree in Python code:
            # generate paths once for this subtree.
            # If path is non-empty (e.g. from serialized data), assume paths are correct already.
            generate_paths = self.path == ""

        if generate_paths:
            # Single pass: _generate_paths_and_names handles both path reconstruction and parent linking.
            # We fuse these operations here to avoid two separate tree traversals.
            self._generate_paths_and_names()
        else:
            # If paths are not being generated, we must link parents separately to ensure cache consistency
            self._link_children_recursive()

    @classmethod
    def fast_construct(
        cls,
        name: str,
        settings: dict[str, Any] | None = None,
        subtrees: dict[str, Any] | None = None,
        *,
        path: str = "",
        align_name: bool = True,
        generate_paths: bool = False,
        **kwargs: Any,
    ) -> SettingNode:
        """Fast construction alternative to __init__ from trusted serialized dict  using model_construct.

        This is the "middle ground" between ``__init__`` (full validation and normalization) and
        ``BaseModel.model_construct`` (no normalization).

        The key purpose is performance: when loading large trees (e.g. tens of thousands of settings)
        that were already validated before being written to the database, re-running Pydantic validation
        and index construction in ``__init__`` is unnecessary and expensive. ``fast_construct`` lets us
        reuse that trusted data as-is and skip the extra work.

        This must only be used with already validated / trusted data (DB rows, internal API payloads, etc.).
        It intentionally skips Pydantic validation and coercion, so untrusted input should always go through
        the normal constructor instead.
        """
        settings = settings or {}
        subtrees = subtrees or {}

        cls._populate_children_from_kwargs(settings, subtrees, kwargs)

        # Fast path: normalise settings/subtrees and skip validation
        norm_settings: dict[str, Setting] = {}
        for key, child in settings.items():
            if isinstance(child, Setting):
                norm_settings[key] = child
            elif isinstance(child, Parameter):
                norm_settings[key] = Setting.fast_construct(parameter=child, value=None, path=key)
            elif isinstance(child, dict):
                # TRUSTED DB / API payload: skip validation
                raw_value = child.get("value")
                if "value" in child:
                    value = json_helper.decode_json(raw_value)
                else:
                    value = None

                param_data = child["parameter"]
                parameter = Parameter.fast_construct(**param_data)
                norm_settings[key] = Setting.fast_construct(
                    parameter=parameter,
                    value=value,
                    read_only=child.get("read_only", False),
                    path=child.get("path", ""),
                )
            else:
                raise TypeError(f"Unsupported type for settings['{key}']: {type(child)}")

        norm_subtrees: dict[str, SettingNode] = {}
        for key, child in subtrees.items():
            if isinstance(child, SettingNode):
                norm_subtrees[key] = child
            elif isinstance(child, dict):
                norm_subtrees[key] = cls.fast_construct(**child)
            else:
                raise TypeError(f"Unsupported type for subtrees['{key}']: {type(child)}")

        setting_node = cls.model_construct(
            name=name,
            settings=norm_settings,
            subtrees=norm_subtrees,
            path=path,
            align_name=align_name,
        )

        # Manually initialise internal attributes
        setting_node._ensure_index_attributes()

        if generate_paths:
            # Single pass: _generate_paths_and_names handles both path reconstruction and parent linking.
            # We fuse these operations here to avoid two separate tree traversals.
            setting_node._generate_paths_and_names()
        else:
            # If paths are not being generated, we must link parents separately to ensure cache consistency
            setting_node._link_children_recursive()
        return setting_node

    def __repr__(self):
        return f"{self.__class__.__name__}{self.children}"

    def __str__(self):
        content = ", ".join(f"{key}={node.__class__.__name__}" for key, node in self.children.items())
        return f"{self.__class__.__name__}({content})"

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, SettingNode) and (
            (self.name, self.settings, self.subtrees) == (other.name, other.settings, other.subtrees)
        )

    def __getattr__(self, key):  # noqa: ANN001
        if key in _INDEX_ATTRIBUTES:
            self._ensure_index_attributes()
            # Private / internal attributes: behave like normal Python objects
            return object.__getattribute__(self, key)
        if key == "settings":
            # Prevent infinite recursion. If settings actually exists, this method is not called anyway
            raise AttributeError
        if key in self.settings:
            return self.settings[key]
        if key in self.subtrees:
            return self.subtrees[key]
        raise UnknownSettingError(
            f'{self.__class__.__name__} "{self.name}" has no attribute {key}. Children: {self.children.keys()}.'
        )

    def __setattr__(self, key: str, value: Any) -> None:
        """Overrides default attribute assignment to allow the following syntax: ``self.foo = 3`` which is
        equivalent to ``self.foo.value.update(3)`` (if ``foo`` is a :class:`.Setting`).
        """
        if key in _INDEX_ATTRIBUTES:
            # Private / internal attributes: behave like normal Python objects
            object.__setattr__(self, key, value)
            return

        self._mark_index_dirty()
        path = self._get_path(key)
        if isinstance(value, (Setting, Parameter)):
            if isinstance(value, Parameter):
                if self.align_name:
                    value = value.model_copy(update={"name": path})
                value = Setting(parameter=value, value=None, path=path)
            else:
                update_dict = {"path": path}
                if self.align_name:
                    update_dict["parameter"] = value.parameter.model_copy(update={"name": path})
                value = value.model_copy(update=update_dict)

            # Explicitly link the new setting instance back to this node
            value._parent = self
            self.settings[key] = value
            self.subtrees.pop(key, None)

        elif isinstance(value, SettingNode):
            # Recursively fix paths and internal parent links of the new subtree
            fixed_node = _fix_path_recursive(value, path)
            # Link the top of the fixed subtree back to this node
            fixed_node._parent = self
            self.subtrees[key] = fixed_node
            self.settings.pop(key, None)

        elif key != "settings" and key in self.settings:
            # Handle the self.key = value syntax (updating a setting's value)
            updated_setting = self.settings[key].update(value)
            updated_setting._parent = self
            self.settings[key] = updated_setting
        else:
            self.__dict__[key] = value

    def __delattr__(self, key):  # noqa: ANN001
        self._mark_index_dirty()
        if key in self.settings:
            del self.settings[key]
        elif key in self.subtrees:
            del self.subtrees[key]
        else:
            del self.__dict__[key]

    def __dir__(self):
        """List settings and subtree names, so they occur in IPython autocomplete after ``node.<TAB>``."""
        return [name for name in list(self.settings) + list(self.subtrees) if name.isidentifier()] + super().__dir__()

    def __iter__(self):
        """Breadth-first iteration through the tree."""
        yield self
        yield from iter(self.settings.values())
        for subtree in self.subtrees.values():
            yield from iter(subtree)

    def __getitem__(self, item: str) -> Setting | SettingNode:
        """Allows dictionary syntax."""
        if len(item.split(".")) == 1:
            return self.__getattr__(item)
        return self.get_node_for_path(item)

    def __setitem__(self, key: str, value: Any) -> None:
        """Allows dictionary syntax."""
        path_fragments = key.split(".")
        node_key = path_fragments[0]
        remaining_path = ".".join(path_fragments[1:])

        if not remaining_path:
            self.__setattr__(key, value)
        elif node_key in self.children:
            try:
                self[node_key][remaining_path] = value
            except TypeError:
                raise ValueError(f"Path '{key}' is invalid: '{node_key}' is a setting, not a node.")
        else:
            if not isinstance(value, (SettingNode, Setting, Parameter)):
                raise ValueError(
                    f"Assigning value {value} to path {key} cannot be done when this path is not found in self."
                )
            self.add_for_path({path_fragments[-1]: value}, path=".".join(path_fragments[:-1]))

    def __delitem__(self, key):  # noqa: ANN001
        """Allows dictionary syntax."""
        self.__delattr__(key)

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> SettingNode:
        """Compatibility for copy.deepcopy; delegate to _fast_deep_copy.

        The ``memo`` argument is ignored on purpose; this keeps the implementation simple
        and reuses the same fast path as model_copy(deep=True). Python’s copy.deepcopy normally
        uses the memo dict to keep track of objects it has already copied, so it can handle cycles
        and shared references in arbitrary object graphs. For SettingNode we know the structure is
        always a tree (no cycles), so we don’t need that machinery.
        """
        return self._fast_deep_copy()

    def _fast_deep_copy(self) -> SettingNode:
        """Fast deep copy of this SettingNode tree.

        - Preserves the dynamic class of each node (subclasses stay subclasses).
        - Reuses Setting and Parameter instances (assumed immutable).
        - Shallow-copies non-tree attributes (those not in the core fields and
          not starting with '_index').

        This function underpins both ``__deepcopy__`` and the optimized path in
        :meth:`model_copy(deep=True)`. It is intended specifically for fast cloning of
        large ``SettingNode`` trees, avoiding the work that a generic deep copy would do.

        Instead of recursively copying every object, we only duplicate the tree
        structure (the ``SettingNode`` instances and their subtrees) and reuse
        structure (the ``SettingNode`` instances and their subtrees) and reuse
        underlying ``Setting`` and ``Parameter`` objects, which are treated as
        immutable. This makes the operation much faster while still producing a
        structurally identical tree.

        Note: if you attach extra mutable attributes to a ``SettingNode`` instance,
        they are shallow-copied here. If those attributes contain shared mutable state,
        the cloned node will reference the same objects.
        """

        def clone(node: SettingNode) -> SettingNode:
            cls = node.__class__  # Preserve subclass type

            # Reuse Setting instances – they are assumed immutable
            new_settings: dict[str, Setting] = dict(node.settings)

            # Recursively clone subtrees (pure tree, no cycles)
            new_subtrees: dict[str, SettingNode] = {name: clone(child) for name, child in node.subtrees.items()}

            # Use the subclass' fast_construct to avoid __init__ overhead
            new_node = cls.fast_construct(  # type: ignore[attr-defined]
                name=node.name,
                settings=new_settings,
                subtrees=new_subtrees,
                path=node.path,
                align_name=node.align_name,
            )

            # Copy non-tree attributes (e.g. controller-specific stuff)
            for attr, value in node.__dict__.items():
                # Skip core fields – already set by fast_construct
                if attr in ("name", "settings", "subtrees", "path", "align_name"):
                    continue

                # Skip indices – let them be rebuilt lazily if needed
                if attr in _INDEX_ATTRIBUTES:
                    continue

                # If fast_construct / BaseModel already set this, don't overwrite
                if attr in new_node.__dict__:
                    continue

                # IMPORTANT: shallow copy here to avoid recursion via SettingNode attributes
                setattr(new_node, attr, value)

            return new_node

        return clone(self)

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = True) -> Self:
        """Copy the model.

        - ``deep=True`` and no ``update`` → use fast tree clone (``_fast_deep_copy``).
        - ``deep=False`` and no ``update`` → return ``self`` (no copy).
        - Any ``update`` → fall back to Pydantic's ``model_copy`` for correct update handling.
        """
        # Note: default deep=True so that `model_copy()` produces a real clone for SettingNode.
        if deep and update is None:
            # Fast path: pure deep tree clone, no update
            return self._fast_deep_copy()

        return super().model_copy(deep=deep, update=update)

    def nodes_by_type(
        self,
        node_types: type | tuple[type, ...] | None = None,
        recursive: bool = False,
    ) -> Iterator:
        """Yields all nodes, filtered by given `node_types`.

        Used to find and iterate over nodes of specific types.

        Args:
            node_types: when iterating over the tree, yields only instances that match this type
                or any of the types in the tuple. By default, yields Settings and SettingNodes.
            recursive: If True, the search is carried recursively. If False, the search is limited to
                immediate child nodes.

        Returns:
             Iterator that yields the filtered nodes.

        """
        if self._index_dirty:
            self._rebuild_index()
        node_types = node_types or (Setting, SettingNode)

        # Non-recursive: behave exactly as before
        if not recursive:
            for child in self.children.values():
                if isinstance(child, node_types):
                    yield child
            return

        # Recursive: cache by node_types, but preserve original iteration semantics
        key = node_types if isinstance(node_types, tuple) else (node_types,)

        # If the tree changed, you should clear this cache in _mark_index_dirty
        cached = self._index_nodes_by_type.get(key)
        if cached is None:
            # IMPORTANT: iterate over `self`, not over _index_by_name
            # This preserves duplicates and order exactly as before.
            cached = [node for node in self if isinstance(node, node_types)]
            self._index_nodes_by_type[key] = cached

        yield from cached

    def update_setting(self, setting: Setting) -> None:
        """Update an existing `Setting` in this tree.

        Args:
            setting: Setting that will replace an existing Setting with the same name. Or if the setting is an
                element-wise setting (i.e. it has a non-empty value of ``setting.element_indices``), the corresponding
                element will be updated in the collection.

        Raises:
            UnknownSettingError: If no setting is found in the children of this tree.

        """

        def list_assign(value, array, indices_list) -> None:  # noqa: ANN001
            sub_array = array
            for index in indices_list[:-1]:
                sub_array = sub_array[index]
            sub_array[indices_list[-1]] = value

        for branch in self.nodes_by_type(SettingNode, True):
            for key, item in branch.children.items():
                if setting.element_indices is None:
                    if item.name == setting.name:
                        branch[key] = setting.value
                        return
                elif isinstance(item, Setting) and item.name == setting.parent_name and item.element_indices is None:
                    parent_value = item.value.copy()
                    list_assign(setting.value, parent_value, setting.element_indices)
                    branch[key] = parent_value
                    return
        raise UnknownSettingError(
            f'No Setting with name {setting.name} was found in {self.__class__.__name__} "{self.name}".'
        )

    @property
    def all_settings(self) -> Generator[Setting]:
        """Yields all :class:`.Setting` instances inside this node, recursively."""
        yield from self.nodes_by_type(Setting, recursive=True)

    @property
    def children(self) -> dict[str, Setting | SettingNode]:
        """Dictionary of immediate child nodes of this node."""
        return {**self.settings, **self.subtrees}

    @property
    def child_settings(self) -> ItemsView[str, Setting]:
        """ItemsView of settings of this node."""
        return self.settings.items()

    @property
    def child_nodes(self) -> ItemsView[str, SettingNode]:
        """ItemsView of immediate child nodes of this node."""
        return self.subtrees.items()

    def get_parent_of(self, name: str) -> SettingNode:
        """Get the first SettingNode that has a Setting named `name`.

        Args:
            name: Name of the setting to look for.

        Returns:
            A SettingNode that has a child `name`.

        """
        for branch in self.nodes_by_type(SettingNode, recursive=True):
            for setting in branch.children.values():
                if setting.name == name:
                    return branch
        raise UnknownSettingError(f'{name} not found inside {self.__class__.__name__} "{self.name}".')

    def find_by_name(self, name: str) -> SettingNode | Setting | None:
        """Find first occurrence of Setting or SettingNode by name, by iterating recursively through all children.

        Args:
            name: Name of the Setting or SettingNode to look for.

        Returns:
            First found item, or None if nothing is found.

        """
        if self._index_dirty:
            self._rebuild_index()
        return self._index_by_name.get(name, None)

    @staticmethod
    def merge(
        first: SettingNode,
        second: SettingNode,
        merge_nones: bool = False,
        align_name: bool = True,
        deep_copy: bool = True,
    ) -> SettingNode:
        """Recursively combine the tree structures and values of two SettingNodes.

        In case of conflicting nodes,values in `first` take priority regardless of the replaced content in `second`.
        `None` values are not prioritized unless ``merge_nones`` is set to ``True``.

        Args:
            first: SettingNode to merge, whose values and structure take priority
            second: SettingNode to merge.
            merge_nones: Whether to merge also ``None`` values from ``first`` to ``second``.
            align_name: Whether to align the paths (and also names if ``second`` does not use ``align_name==False``)
                when merging the nodes. Should never be set ``False`` unless the paths in ``first`` already align with
                what they should be in ``second`` (setting it ``False`` in such cases can improve performance).
            deep_copy: Whether to deepcopy or just shallow copy all the sub-nodes. Set to ``False`` with high caution
                and understand the consequences.

        Returns:
            A new SettingNode constructed from arguments.

        """
        new = second.model_copy(deep=deep_copy)
        for key, item in first.settings.items():
            if merge_nones or item.value is not None:
                if align_name:
                    new[key] = item.model_copy(update={"name": item.name.replace(f"{first.name}.", "")}, deep=True)
                else:
                    new.settings[key] = item.model_copy(deep=True)
        for key, item in first.subtrees.items():
            update = {"name": item.name.replace(f"{first.name}.", "")} if align_name else {}
            item_copy = item.model_copy(update=update, deep=deep_copy)
            subs = new if align_name else new.subtrees
            if key in new.subtrees:
                subs[key] = SettingNode.merge(item_copy, new[key])
            else:
                subs[key] = item_copy

        for key, item in first.__dict__.items():
            if key not in ["settings", "subtrees"]:
                new[key] = copy(item)
        return new

    def merge_values(self, other: SettingNode, prioritize_other: bool = False):  # noqa: ANN201
        """Recursively combine the values from another :class:`SettingNode` to this one.

        The resulting tree structure the same as that of self.

        Args:
            other: SettingNode to merge.
            prioritize_other: If True, will prioritize values in other. If False (default), only None values in self
                will be replaced.

        """
        for key, item in other.settings.items():
            if key in self.settings and (prioritize_other or (self[key].value is None)):
                self[key] = Setting(self.settings[key].parameter, item.value, source=self.settings[key].source)
        for key, item in other.subtrees.items():
            if key in self.subtrees:
                self.subtrees[key].merge_values(copy(item), prioritize_other)

    def prune(self, other: SettingNode) -> None:
        """Recursively delete all branches from this SettingNode that are not found in ``other``."""
        for key, node in self.subtrees.copy().items():
            if key not in other.subtrees:
                del self[key]
            else:
                self[key].prune(other[key])

    def print_tree(self, levels: int = 5) -> None:
        """Print a tree representation of the contents of this node.

        Args:
            levels: display this many levels, starting from the root.

        """

        def append_lines(node: SettingNode, lines: list[str], indents: list[bool]):  # noqa: ANN202
            indent = "".join([" ║  " if i else "    " for i in indents])
            if len(indents) < levels:
                for key, setting in node.settings.items():
                    if setting.value is None:
                        value = "None (automatic/unspecified)"
                    elif setting.parameter.collection_type == CollectionType.NDARRAY:
                        value = str(setting.value).replace("\n", "") + f" {setting.unit}"
                    else:
                        value = f"{setting.value} {setting.unit}"
                    lines.append(indent + f" ╠─ {key}: {setting.label} = {value}")
                if node.subtrees:
                    lines.append(indent + " ║ ")
                subtrees_written = 0
                for key, subtree in node.subtrees.items():
                    lines.append(indent + f' ╠═ {key}: "{subtree.name}: align_name={subtree.align_name}"')
                    if subtrees_written == len(node.subtrees) - 1:
                        lines[-1] = lines[-1].replace("╠", "╚")
                    append_lines(subtree, lines, indents + [subtrees_written < len(node.subtrees) - 1])
                    subtrees_written += 1
            lines[-1] = lines[-1].replace("╠", "╚")

        lines = [f'"{self.name}: align_name={self.align_name}"']
        append_lines(self, lines, [])
        print("\n", "\n".join(lines))

    @classmethod
    def transform_node_types(cls, node: SettingNode) -> SettingNode:
        """Reduce any subclass of SettingNode and it's contents into instances of `cls`.

        Args:
            node: node to transform.

        Return:
            A new SettingNode with the same structure as the original, but where node instances are of type `cls`.

        """
        new = cls.fast_construct(name=node.name, align_name=node.align_name, **node.settings)
        for key, subnode in node.subtrees.items():
            new.subtrees[key] = cls.transform_node_types(subnode)
        return new

    def set_from_dict(
        self,
        dct: dict[str, Any],
        strict: bool = False,
        source: SourceType = None,
    ) -> None:
        """Recursively set values to Settings, taking values from a dictionary that has similar tree structure.
        Keys that are not found in self are ignored, unless ``strict`` is True.

        Args:
            dct: Dictionary containing the new values to use.
            strict: If True, will raise error if ``dct`` contains a setting that is not found in ``self``.
            source: Source for the settings (this same source is applied to all settings from the dict).

        Raises:
            UnknownSettingError: If the condition of ``strict`` happens.

        """
        self._mark_index_dirty()
        for key, value in dct.items():
            if key not in self.children:
                error = UnknownSettingError(f"Tried to set {key} to {value}, but no such node exists in {self.name}.")
                logger.debug(error.message)
                if strict:
                    raise error
                continue
            if isinstance(value, dict) and isinstance(self[key], SettingNode):
                self[key].set_from_dict(value, source=source)
            elif type(value) is str:
                self.settings[key] = Setting(
                    self.settings[key].parameter,
                    self.settings[key].parameter.data_type.cast(value),
                    path=self.settings[key].path,
                    source=source,
                )
            else:
                self.settings[key] = self.settings[key].update(value, source=source)

    def setting_with_path_name(self, setting: Setting) -> Setting | None:
        """Get a copy of a setting with its name replaced with the path name."""
        first_item = self.find_by_name(setting.name)
        if isinstance(first_item, Setting):
            return first_item.with_path_name()
        return None

    def diff(self, other: SettingNode, *, path: str = "") -> list[str]:
        """Recursive diff between two SettingNodes.

        This function is meant to produce human-readable output, e.g. for debugging purposes.
        It returns the differences in a list of strings, each string detailing
        one specific difference. The diff is non-symmetric.

        Args:
            other: second node to compare ``self`` to
            path: node path to the currently compared nodes (used in printing the results)

        Returns:
            differences from ``self`` to ``other``, in depth-first order

        """

        def diff_settings(x: Setting, y: Setting, key: str) -> str:
            """Compare two settings, return the differences."""
            line = f"{key}:"

            def compare(x: Any, y: Any, tag: str) -> None:
                """Compare the given properties, add a note to the current line if there is a difference."""
                nonlocal line
                if isinstance(x, np.ndarray):
                    if np.any(x != y):
                        line += f" {tag}: differs"
                    return
                if x != y:
                    line += f" {tag}: {x}/{y}"

            compare(x.name, y.name, "n")
            compare(x.label, y.label, "l")
            compare(x.unit, y.unit, "u")
            compare(x.parameter.data_type, y.parameter.data_type, "dt")
            a_ct = x.parameter.collection_type
            b_ct = y.parameter.collection_type
            compare(a_ct, b_ct, "ct")
            if a_ct == b_ct:
                compare(x.value, y.value, "v")
            return line

        node_diff: list[str] = []
        # compare node names
        if self.name != other.name:
            node_diff.append(f"node name: {self.name}/{other.name}")

        # compare settings
        b_keys = set(other.settings)
        for key, a_setting in self.settings.items():
            b_setting = other.settings.get(key)
            if b_setting is None:
                node_diff.append(f"-setting: {key}")
                continue
            b_keys.remove(key)
            if a_setting != b_setting:
                node_diff.append(diff_settings(a_setting, b_setting, key))
        for key in b_keys:
            if key not in self.settings:
                node_diff.append(f"+setting: {key}")

        # compare subnodes
        diff_subnodes: list[tuple[SettingNode, SettingNode, str]] = []
        b_keys = set(other.subtrees)
        for key, a_sub in self.subtrees.items():
            b_sub = other.subtrees.get(key)
            if b_sub is None:
                node_diff.append(f"-subnode: {key}")
            else:
                b_keys.remove(key)
                diff_subnodes.append((a_sub, b_sub, key))
        for key in b_keys:
            if key not in self.subtrees:
                node_diff.append(f"+subnode: {key}")

        # add path prefixes
        diff = [f"{path}: {d}" for d in node_diff]

        # recurse into subnodes, depth first
        for a_sub, b_sub, key in diff_subnodes:
            diff.extend(a_sub.diff(b_sub, path=f"{path}.{key}" if path else key))

        return diff

    def get_node_for_path(self, path: str) -> Setting | SettingNode:
        """Return the node corresponding to the given path.

        Args:
            path: The path.

        Returns:
            The node at ``path`` in self.

        Raises:
            ValueError: If the given path cannot be found in self.

        """
        keys = path.split(".")
        node = self
        for index, key in enumerate(keys, start=1):
            if index < len(keys) and key in node.settings:
                raise ValueError(f"Path '{path}' is invalid: '{key}' is a setting, not a node.")
            if key not in node.children:
                raise KeyError(f"Path '{path}' is invalid: key '{key}' is not found in the preceding node {node}.")
            node = node[key]
        return node

    def add_for_path(
        self,
        nodes: Iterable[Setting | Parameter | SettingNode] | dict[str, Setting | Parameter | SettingNode],
        path: str,
        override_values: dict[str, Any] | None = None,
        override_source: SourceType = None,
    ) -> None:
        """Add nodes to ``self`` while creating the missing nodes in-between.

        Whether the names and paths are aligned is determined by the attribute ``align_name`` of the current node
        (``self``). All the created missing nodes will use this same ``align_name`` value,
        which determines whether their names will align with their paths.

        Args:
            nodes: Nodes to add as new leaves/branches of ``path``. If of type ``dict``, maps the keys used in
                ``self.settings`` or ``self.subtrees`` to the nodes themselves. If ``align_name=False``, the key and
                the node name can differ, but otherwise the names will be replaced by the path anyways).
            path: Path in ``self`` to which ``nodes`` will be added. If the path or any part (suffix) of it is not
                found in self, the associated nodes will be created automatically.
            override_values: Optionally override the values for the `Settings` corresponding to ``nodes``. This dict
                should have the same structure as ``nodes``, including matching names.
                NOTE: If an override's value is None, it is ignored.
            override_source: Optionally override the source for the ``Settings`` corresponding to ``nodes``. All the
                settings will then have this same source.

        """
        override_values = override_values or {}
        path_split = path.split(".")
        # find the depth already found in self
        latest_node = self
        levels_to_add = []
        for idx, fragment in enumerate(path_split):
            if fragment in latest_node.children:
                latest_node = latest_node[fragment]
            else:
                levels_to_add = path_split[idx:]
                break
        # add the missing levels
        for fragment in levels_to_add:
            latest_node[fragment] = SettingNode(name=fragment, align_name=latest_node.align_name)
            latest_node = latest_node[fragment]
        # finally add the nodes
        nodes_to_add: Iterable[Setting | Parameter | SettingNode] = nodes.values() if isinstance(nodes, dict) else nodes
        nodes_keys = list(nodes.keys()) if isinstance(nodes, dict) else []
        for idx, node in enumerate(nodes_to_add):
            key = nodes_keys[idx] if isinstance(nodes, dict) else node.name.split(".")[-1]
            if isinstance(node, SettingNode):
                latest_node[key] = node
            else:
                default_value = node.value if isinstance(node, Setting) else None
                default_source = node.source if isinstance(node, Setting) else None
                parameter = node.parameter if isinstance(node, Setting) else node
                override_value = override_values.get(node.name)
                if override_value is None:
                    latest_node[key] = Setting(parameter, default_value, source=default_source)
                else:
                    latest_node[key] = Setting(parameter, override_value, source=override_source)

    def get_default_implementation_name(self, gate: str, locus: str | Iterable[str]) -> str:
        """Get the default implementation name for a given gate and locus.

        Takes into account the global default implementation and a possible locus specific implementation and also
        the symmetry properties of the gate.

        NOTE: using this method requires the standard EXA settings tree structure.

        Args:
            gate: The name of the gate.
            locus: Individual qubits, couplers, or combinations.

        Returns:
            The default implementation name.

        """
        gate_settings = self.gate_definitions[gate]
        for impl in gate_settings.children:
            if (
                isinstance(gate_settings[impl], SettingNode)
                and impl not in ("symmetric", "default_implementation")  # sanity check
                and "override_default_for_loci" in gate_settings[impl].children  # backwards compatibility
                and gate_settings[impl].override_default_for_loci.value
            ):
                if isinstance(locus, str):
                    locus = locus.split("__")
                loci = list(permutations(locus)) if gate_settings.symmetric.value else [tuple(locus)]
                for permuted_locus in loci:
                    locus_str = "__".join(permuted_locus)
                    if locus_str in gate_settings[impl].override_default_for_loci.value:
                        return impl
        return gate_settings.default_implementation.value

    def get_gate_node_for_locus(
        self,
        gate: str,
        locus: str | Iterable[str],
        implementation: str | None = None,
    ) -> SettingNode:
        """Get the gate calibration sub-node for the locus given as a parameter if it exists in the settings tree.

        NOTE: using this method requires the standard EXA settings tree structure.

        Args:
            gate: The gate to retrieve the settings for.
            locus: Individual qubits, couplers, or combinations.
            implementation: Using a custom rather than the default gate implementation.

        Returns:
            The settings of the specified locus and gate.

        """
        pulse_settings = self["gates"] if "gates" in self.children else self

        if gate not in pulse_settings.children:
            raise ValueError(f"Gate {gate} cannot be found in the pulse settings.")

        if not implementation:
            implementation = self.get_default_implementation_name(gate, locus)

        if implementation not in pulse_settings[gate].children:
            raise ValueError(f"Gate implementation {implementation} cannot be found in the pulse settings.")

        str_loci = self._get_symmetric_loci(gate, implementation, locus)

        for str_locus in str_loci:
            if str_locus in pulse_settings[gate][implementation].children:
                return pulse_settings[gate][implementation][str_locus]

        raise ValueError(f"Locus {locus} cannot be found in the pulse settings.")

    def get_locus_node_paths_for(self, gate: str, implementations: list[str] | None = None) -> list[str]:
        """Get all the gate locus node paths for a given ``gate``.

        NOTE: using this method requires the standard EXA settings tree structure.

        Args:
            gate: Gate name.
            implementations: optionally limit the paths by these gate implementations.

        Returns:
            The locus node (string) paths corresponding to this gate.

        """
        node_paths: list[str] = []
        if "gates" not in self.children or gate not in self.gates.children:
            return node_paths
        if implementations is not None:
            impls = [i for i in self.gates[gate].children if i in implementations]
        else:
            impls = self.gates[gate].children
        for impl_name in impls:
            if isinstance(self.gates[gate][impl_name], SettingNode):
                for locus in self.gates[gate][impl_name].children:
                    if isinstance(self.gates[gate][impl_name][locus], SettingNode):
                        node_paths.append(f"{gate}.{impl_name}.{locus}")
        return node_paths

    def get_gate_properties_for_locus(
        self, gate: str, locus: str | Iterable[str], implementation: str | None = None
    ) -> SettingNode:
        """Get the gate characterization sub-node for the locus given as a parameter if it exists in the settings tree.

        NOTE: using this method requires the standard EXA settings tree structure.

        Args:
            gate: The gate to retrieve the settings for.
            locus: Individual qubits, couplers, or combinations.
            implementation: Using a custom rather than the default gate implementation.

        Returns:
            The settings of the specified locus and gate.

        """
        if not implementation:
            implementation = self.get_default_implementation_name(gate, locus)
        gate_properties = self.characterization.gate_properties
        if gate not in gate_properties.children:
            raise ValueError(f"Gate {gate} cannot be found in the characterization settings.")
        if implementation not in gate_properties[gate].children:
            raise ValueError(
                f"Implementation {implementation} of Gate {gate} cannot be found in the characterization settings."
            )

        str_loci = self._get_symmetric_loci(gate, implementation, locus)
        for str_locus in str_loci:
            if str_locus in gate_properties[gate][implementation].children:
                return gate_properties[gate][implementation][str_locus]

        raise ValueError(f"Locus {locus} cannot be found in the gate properties characterization settings.")

    def set_shots(self, shots: int, averaging_bins: int | None = None) -> None:
        """Set ``playlist_repeats`` and ``averaging_bins`` under ``self.controllers.options``.

        Args:
            shots: The number of shots.
            averaging_bins: The number of averaging bins. If ``None``, the same value as in ``shots`` will be used,
                i.e. no averaging is done.

        """
        self._mark_index_dirty()
        controller_settings = self["controllers"] if "controllers" in self.children else self
        if "options" not in controller_settings.subtrees:
            raise KeyError("CommonOptionsControllers cannot be found in the controller settings.")
        controller_settings.options.playlist_repeats = shots
        controller_settings.options.averaging_bins = averaging_bins if averaging_bins is not None else shots

    def set_source(self, source: SourceType, ignore_nones: bool = True, ignore_populated: bool = False) -> None:
        """Set source recursively to all Settings in ``self``.

        Args:
            source: The source to set.
            ignore_nones: If ``True``, the source will not be set for Settings with ``None`` value.
            ignore_populated: If ``True``, the source will not be set for Settings that already have a source.

        """
        self._mark_index_dirty()
        for setting in self.settings.values():
            if (not ignore_nones or setting.value is not None) and (not ignore_populated or setting._source is None):
                setting._source = source
        for subtree in self.subtrees.values():
            subtree.set_source(source, ignore_nones=ignore_nones)

    def _repr_html_(self) -> str:
        tmpl_path = pathlib.Path(__file__).parent
        jenv = jinja2.Environment(loader=jinja2.FileSystemLoader(tmpl_path), auto_reload=True)

        return jenv.get_template("settingnode_v2.html.jinja2").render(s=self, withsi=self._withsiprefix, startopen=0)

    def _get_path(self, key: str) -> str:
        """Return the full path for a key.

        If ``self.path`` is empty, the key is returned as-is. Otherwise,
        the key is appended to ``self.path`` using dot notation (e.g., ``"parent.child"``).

        Args:
            key: Name of the entry.

        Returns:
            The full dotted path for the given key.

        """
        if not self.path:
            return key
        return f"{self.path}.{key}"

    def _generate_paths_and_names(self) -> None:
        """In-place tree reconstruction to align node paths and parameter names.

        This method performs a single-pass reconstruction of the subtree. It replaces
        existing children with new instances that carry the correct hierarchical path.
        Children are linked to their parents during this process so that updates can
        propagate upward, allowing parents to invalidate their caches when a child changes.
        """
        path = self.path
        align_name = self.align_name

        for key, child in self.subtrees.items():
            update_path = f"{path}.{key}" if path else key
            new_node = _fix_path_recursive(child, update_path)
            new_node._parent = self
            self.subtrees[key] = new_node
        for key, child in self.settings.items():
            update_path = f"{path}.{key}" if path else key
            if isinstance(child, Setting):
                new_param = child.parameter
                if align_name:
                    # Use model_construct to bypass validation during the rename
                    new_param = Parameter.fast_construct(
                        name=update_path,
                        label=new_param.label,
                        unit=new_param.unit,
                        data_type=new_param.data_type,
                        collection_type=new_param.collection_type,
                        element_indices=new_param.element_indices,
                    )
                self.settings[key] = Setting.fast_construct(
                    parameter=new_param,
                    value=child.value,
                    read_only=child.read_only,
                    path=update_path,
                    source=child._source,
                )
            elif align_name:
                new_param = Parameter.fast_construct(
                    name=update_path,
                    label=child.label,
                    unit=child.unit,
                    data_type=child.data_type,
                    collection_type=child.collection_type,
                    element_indices=child.element_indices,
                )
                self.settings[key] = Setting.fast_construct(parameter=new_param, value=None, path=update_path)
        if path and align_name:
            self.name = path

    def _withsiprefix(self, val: Any, unit: str) -> tuple[Any, str]:
        """Turn a numerical value and unit, and return rescaled value and SI prefixed unit.

        Unit must be a whitelisted SI base unit.
        """
        if not isinstance(val, numbers.Real):
            return val, unit
        if unit not in {"Hz", "rad", "s", "V"}:
            return val, unit

        val = float(val)

        pfx = ""
        for p in "kMGP":
            if abs(val) <= 10e3:
                break
            val *= 1e-3
            pfx = p
        for p in "mμnp":
            if not 1 > abs(val) > 0:
                break
            val *= 1e3
            pfx = p

        return val, f"{pfx}{unit}"

    def _get_symmetric_loci(self, gate: str, implementation: str, locus: str | Iterable[str]) -> list[str]:
        if not isinstance(locus, str):
            if self.gate_definitions[gate][implementation].symmetric.value:
                str_loci = ["__".join(sort_components(locus))]
            elif self.gate_definitions[gate].symmetric.value:
                str_loci = ["__".join(item) for item in list(permutations(locus))]
            else:
                str_loci = ["__".join(locus)]
        else:
            str_loci = [locus]
        return str_loci

    def _ipython_key_completions_(self) -> list[str]:
        """List items and subtree names, so they occur in IPython autocomplete after ``node[<TAB>``"""
        return [*self.settings, *self.subtrees]

    @staticmethod
    def _populate_children_from_kwargs(
        settings: dict[str, Any],
        subtrees: dict[str, Any],
        kwargs: dict[str, Any],
    ) -> None:
        """Move children from kwargs into settings and subtrees in-place, and remove them from kwargs."""
        to_delete: list[str] = []

        for key, child in kwargs.items():
            if isinstance(child, Setting):
                settings[key] = child
                to_delete.append(key)
            elif isinstance(child, Parameter):
                settings[key] = Setting(parameter=child, value=None, path=key)
                to_delete.append(key)
            elif isinstance(child, SettingNode):
                subtrees[key] = child
                to_delete.append(key)
            # else: leave it in kwargs for BaseModel / callers to see

        for key in to_delete:
            kwargs.pop(key)

    def _ensure_index_attributes(self) -> None:
        """Make sure index-related attributes exist.

        This is defensive: some code paths may touch the indices
        before __init__/fast_construct has fully initialised them.
        """
        for index_attr, factory in _INDEX_ATTRIBUTES.items():
            if index_attr not in self.__dict__:
                object.__setattr__(self, index_attr, factory())

    def _mark_index_dirty(self) -> None:
        """Mark this node and all ancestors as needing index rebuild."""
        self._ensure_index_attributes()
        self._index_by_name.clear()
        self._index_nodes_by_type.clear()
        self._index_dirty = True

        # Bubble up: if the parent exists, it is also now dirty
        if self._parent:
            self._parent._mark_index_dirty()

    def _rebuild_index(self) -> None:
        """Rebuild name -> [Setting|SettingNode] index for this subtree."""
        index: dict[str, Setting | SettingNode] = {}
        stack: list[SettingNode] = [self]

        while stack:
            node = stack.pop()

            if node.name in index:
                # This is strange, but the current logic expects that duplicates are simply ignored
                continue
            index[node.name] = node

            for setting in node.settings.values():
                if setting.name in index:
                    # This is strange, but the current logic expects that duplicates are simply ignored
                    continue
                index[setting.name] = setting

            stack.extend(node.subtrees.values())

        self._index_by_name = {key: value for key, value in index.items()}
        self._index_dirty = False
        self._index_nodes_by_type.clear()

    def _link_children_recursive(self) -> None:
        """Iteratively establish parent links to minimize function call overhead.

        We use an iterative stack-based approach instead of a recursive call to
        avoid Python's function call overhead. On large trees with millions of nodes,
        the overhead of creating a new  function scope/frame for millions of calls
        adds a significant delay to execution time.
        """
        stack = [self]
        # Local cache of object.__setattr__ to shave off lookup time in the loop
        set_attr = object.__setattr__

        while stack:
            current = stack.pop()

            # Link immediate settings
            for setting in current.settings.values():
                set_attr(setting, "_parent", current)
            # Link subtrees and add them to the stack for processing
            for subtree in current.subtrees.values():
                set_attr(subtree, "_parent", current)
                stack.append(subtree)

    def _link_children(self) -> None:
        """Establish parent links for immediate children only."""
        # Link immediate settings
        for child in self.settings.values():
            child._parent = self
            object.__setattr__(child, "_parent", self)
        for child in self.subtrees.values():
            object.__setattr__(child, "_parent", self)
