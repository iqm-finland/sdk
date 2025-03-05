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


Basic manipulation
------------------

Adding and deleting new Settings and nodes is simple:

.. doctest:: pulse

    >>> modified = node.copy()
    >>> del modified.flux # removes the node
    >>> del modified.pulse.amplitude # removes the Setting
    >>> modified.pulse.my_new_setting = Setting(Parameter('my name'), 33)

It is usually a good idea to make a copy of the original node, so that it won't be modified accidentally.

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

from collections.abc import Generator, ItemsView, Iterator
from copy import copy, deepcopy
import logging
import numbers
import pathlib
from typing import Any

import jinja2
import numpy as np

from exa.common.data.parameter import CollectionType, Parameter, Setting
from exa.common.errors.exa_error import UnknownSettingError


class SettingNode:
    """A tree-structured :class:`.Setting` container.

    Each child of the node is a :class:`.Setting`, or another :class:`SettingNode`.
    Iterating over the node returns all children, recursively.
    Settings can be accessed by dictionary syntax or attribute syntax:

    .. doctest::

        >>> from exa.common.data.parameter import Parameter
        >>> from exa.common.data.setting_node import SettingNode
        >>> p1 = Parameter("voltage", "Voltage")
        >>> settings = SettingNode('name', volt=p1)
        >>> settings.volt.parameter is p1
        True
        >>> settings['volt'].parameter is p1
        True
        >>> settings.volt.value is None
        True
        >>> settings.volt = 7  # updates to Setting(p1, 7)
        >>> settings.volt.value
        7

    Args:
        name: Name of the node.
        children: The children given as keyword arguments. Each argument must be a :class:`.Setting`,
            :class:`.Parameter`, or a :class:`SettingNode`. The keywords are used as the names of the nodes.
            Parameters will be cast into Settings with the value ``None``.

    """

    def __init__(self, name: str, **children):
        self._settings: dict[str, Setting] = {}
        self._subtrees: dict[str, SettingNode] = {}
        self.name = name
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        for key, child in children.items():
            if isinstance(child, Setting):
                self._settings[key] = child
            elif isinstance(child, Parameter):
                self._settings[key] = Setting(child, None)
            elif isinstance(child, SettingNode):
                self._subtrees[key] = child
            else:
                raise ValueError(f"{key} should be a Parameter, Setting or a SettingNode, not {type(child)}.")

    def __getattr__(self, key):
        if key == "_settings":
            # Prevent infinite recursion. If _settings actually exists, this method is not called anyway
            raise AttributeError
        if key in self._settings:
            return self._settings[key]
        if key in self._subtrees:
            return self._subtrees[key]
        raise UnknownSettingError(f'{self.__class__.__name__} "{self.name}" has no attribute {key}.')

    def __dir__(self):
        """List settings and subtree names, so they occur in IPython autocomplete after ``node.<TAB>``."""
        return [name for name in list(self._settings) + list(self._subtrees) if name.isidentifier()] + super().__dir__()

    def _ipython_key_completions_(self):
        """List items and subtree names, so they occur in IPython autocomplete after ``node[<TAB>``"""
        return [*self._settings, *self._subtrees]

    def __setattr__(self, key, value):
        """Overrides default attribute assignment to allow the following syntax: ``self.foo = 3`` which is
        equivalent to ``self.foo.value.update(3)`` (if ``foo`` is a :class:`.Setting`).
        """
        if isinstance(value, Parameter):
            value = value.set(None)
        if isinstance(value, Setting):
            self._settings[key] = value
            self._subtrees.pop(key, None)
        elif isinstance(value, SettingNode):
            self._subtrees[key] = value
            self._settings.pop(key, None)
        elif key != "_settings" and key in self._settings:  # != prevents infinite recursion
            self._settings[key] = self._settings[key].update(value)
        else:
            self.__dict__[key] = value

    def __delattr__(self, key):
        if key in self._settings:
            del self._settings[key]
        elif key in self._subtrees:
            del self._subtrees[key]
        else:
            del self.__dict__[key]

    def __getitem__(self, item: str) -> Setting | SettingNode:
        """Allows dictionary syntax."""
        return self.__getattr__(item)

    def __setitem__(self, key: str, value: Any) -> None:
        """Allows dictionary syntax."""
        self.__setattr__(key, value)

    def __delitem__(self, key):
        """Allows dictionary syntax."""
        self.__delattr__(key)

    def __iter__(self):
        """Allows breadth-first iteration through the tree."""
        yield self
        yield from iter(self._settings.values())
        for subtree in self._subtrees.values():
            yield from iter(subtree)

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
        node_types = node_types or (Setting, SettingNode)
        iterable = self if recursive else self.children.values()
        return filter(lambda node: isinstance(node, node_types), iterable)

    def update_setting(self, setting: Setting) -> None:
        """Update an existing `Setting` in this tree.

        Args:
            setting: Setting that will replace an existing Setting with the same name. Or if the setting is an
                element-wise setting (i.e. it has a non-empty value of ``setting.element_indices``), the corresponding
                element will be updated in the collection.

        Raises:
            UnknownSettingError: If no setting is found in the children of this tree.

        """

        def list_assign(value, array, indices_list) -> None:
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
        return {**self._settings, **self._subtrees}

    @property
    def child_settings(self) -> ItemsView[str, Setting]:
        """ItemsView of settings of this node."""
        return self._settings.items()

    @property
    def child_nodes(self) -> ItemsView[str, SettingNode]:
        """ItemsView of immediate child nodes of this node."""
        return self._subtrees.items()

    def copy(self) -> SettingNode:
        """Return a deepcopy of this SettingNode."""
        return deepcopy(self)

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
        return next((item for item in self if item.name == name), None)

    @staticmethod
    def merge(first: SettingNode, second: SettingNode) -> SettingNode:
        """Recursively combine the tree structures and values of two SettingNodes.

        In case of conflicting nodes,values in `first` take priority regardless of the replaced content in `second`.
        `None` values are never prioritized.

        Args:
            first: SettingNode to merge, whose values and structure take priority
            second: SettingNode to merge.

        Returns:
            A new SettingNode constructed from arguments.

        """
        new = second.copy()
        for key, item in first._settings.items():
            if item.value is not None:
                new[key] = copy(item)
        for key, item in first._subtrees.items():
            if key in new._subtrees:
                new[key] = SettingNode.merge(item, new[key])
            else:
                new[key] = copy(item)

        for key, item in first.__dict__.items():
            if key not in ["_settings", "_subtrees"]:
                new[key] = copy(item)
        return new

    def merge_values(self, other: SettingNode, prioritize_other: bool = False):
        """Recursively combine the values from another :class:`SettingNode` to this one.

        The resulting tree structure the same as that of self.

        Args:
            other: SettingNode to merge.
            prioritize_other: If True, will prioritize values in other. If False (default), only None values in self
                will be replaced.

        """
        for key, item in other._settings.items():
            if key in self._settings and (prioritize_other or (self[key].value is None)):
                self[key] = item.value
        for key, item in other._subtrees.items():
            if key in self._subtrees:
                self[key].merge_values(copy(item), prioritize_other)

    def prune(self, other: SettingNode) -> None:
        """Recursively delete all branches from this SettingNode that are not found in ``other``."""
        for key, node in self._subtrees.copy().items():
            if key not in other._subtrees:
                del self[key]
            else:
                self[key].prune(other[key])

    def print_tree(self, levels: int = 5) -> None:
        """Print a tree representation of the contents of this node.

        Args:
            levels: display this many levels, starting from the root.

        """

        def append_lines(key: str, node: SettingNode, lines: List[str], indents: List[bool]):  # noqa: F821
            indent = "".join([" ║  " if i else "    " for i in indents])
            if len(indents) < levels:
                for key, setting in node._settings.items():  # noqa: PLR1704
                    if setting.value is None:
                        value = "None (automatic/unspecified)"
                    elif setting.parameter.collection_type == CollectionType.NDARRAY:
                        value = str(setting.value).replace("\n", "") + f" {setting.unit}"
                    else:
                        value = f"{setting.value} {setting.unit}"
                    lines.append(indent + f" ╠─ {key}: {setting.label} = {value}")
                if node._subtrees:
                    lines.append(indent + " ║ ")
                subtrees_written = 0
                for key, subtree in node._subtrees.items():
                    lines.append(indent + f' ╠═ {key}: "{subtree.name}"')
                    if subtrees_written == len(node._subtrees) - 1:
                        lines[-1] = lines[-1].replace("╠", "╚")
                    append_lines(key, subtree, lines, indents + [subtrees_written < len(node._subtrees) - 1])
                    subtrees_written += 1
            lines[-1] = lines[-1].replace("╠", "╚")

        lines = [f'"{self.name}"']
        append_lines("", self, lines, [])
        print("\n", "\n".join(lines))

    def __eq__(self, other: SettingNode):
        return isinstance(other, SettingNode) and self.children == other.children and self.name == other.name

    def __repr__(self):
        return f"{self.__class__.__name__}{self.children}"

    def __str__(self):
        content = ", ".join(f"{key}={node.__class__.__name__}" for key, node in self.children.items())
        return f"{self.__class__.__name__}({content})"

    @classmethod
    def transform_node_types(cls, node: SettingNode) -> SettingNode:
        """Reduce any subclass of SettingNode and it's contents into instances of `cls`.

        Args:
            node: node to transform.

        Return:
            A new SettingNode with the same structure as the original, but where node instances are of type `cls`.

        """
        new = cls(name=node.name, **node._settings)
        for key, subnode in node._subtrees.items():
            new[key] = cls.transform_node_types(subnode)
        return new

    def set_from_dict(self, dct: dict[str, Any], strict: bool = False) -> None:
        """Recursively set values to Settings, taking values from a dictionary that has similar tree structure.
        Keys that are not found in self are ignored, unless `strict` is True.

        Args:
            dct: Dictionary containing the new values to use.
            strict: If True, will raise error if `dct` contains a setting that is not found in `self`.

        Raises:
            UnknownSettingError: If the condition of `strict` happens.

        """
        for key, value in dct.items():
            if key not in self.children:
                error = UnknownSettingError(f"Tried to set {key} to {value}, but no such node exists in {self.name}.")
                self.logger.debug(error.message)
                if strict:
                    raise error
                continue
            if isinstance(value, dict) and isinstance(self[key], SettingNode):
                self[key].set_from_dict(value)
            else:
                self[key] = self[key].parameter.data_type.cast(value) if type(value) == str else value  # noqa: E721

    def setting_with_path_name(self, setting: Setting) -> Setting:
        """Get Setting from ``self`` where the name of the parameter is replaced with its path in ``self``.

        The path is defined as the sequence of attribute keys leading to the node with ``setting`` in it.
        The method is used for conveniently saving settings tree observations with their settings tree path
        as the ``dut_field``.

        Args:
            setting: Setting to search for.

        Returns:
            Copy of Setting ``setting`` where the parameter name is replaced with its path in ``self``.

        Raises:
            UnknownSettingError: If ``name`` is not found within ``self``.

        """

        def _search(settings: SettingNode, name: str, prefix: str) -> tuple[str, Setting] | tuple[None, None]:
            for key, value in settings.children.items():
                path_prefix = f"{prefix}.{key}" if prefix else key
                if isinstance(value, Setting) and value.name == setting.name:
                    return path_prefix, value
                if isinstance(value, SettingNode) and value.find_by_name(setting.name):
                    return _search(value, setting, path_prefix)
            return None, None

        path, found_setting = _search(self, setting, "")
        if path is None:
            raise UnknownSettingError(f'{name} not found inside {self.__class__.__name__} "{self.name}".')  # noqa: F821
        param = found_setting.parameter
        return Setting(
            Parameter(
                name=path,
                label=param.label,
                unit=param.unit,
                data_type=param.data_type,
                collection_type=param.collection_type,
                element_indices=param.element_indices,
            ),
            found_setting.value,
        )

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
        b_keys = set(other._settings)
        for key, a_setting in self._settings.items():
            b_setting = other._settings.get(key)
            if b_setting is None:
                node_diff.append(f"-setting: {key}")
                continue
            b_keys.remove(key)
            if a_setting != b_setting:
                node_diff.append(diff_settings(a_setting, b_setting, key))
        for key in b_keys:
            if key not in self._settings:
                node_diff.append(f"+setting: {key}")

        # compare subnodes
        diff_subnodes: list[tuple[SettingNode, SettingNode, str]] = []
        b_keys = set(other._subtrees)
        for key, a_sub in self._subtrees.items():
            b_sub = other._subtrees.get(key)
            if b_sub is None:
                node_diff.append(f"-subnode: {key}")
            else:
                b_keys.remove(key)
                diff_subnodes.append((a_sub, b_sub, key))
        for key in b_keys:
            if key not in self._subtrees:
                node_diff.append(f"+subnode: {key}")

        # add path prefixes
        diff = [f"{path}: {d}" for d in node_diff]

        # recurse into subnodes, depth first
        for a_sub, b_sub, key in diff_subnodes:
            diff.extend(a_sub.diff(b_sub, path=f"{path}.{key}" if path else key))

        return diff

    def _withsiprefix(self, val, unit):
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

    def _repr_html_(self):
        tmpl_path = pathlib.Path(__file__).parent
        jenv = jinja2.Environment(loader=jinja2.FileSystemLoader(tmpl_path), auto_reload=True)

        return jenv.get_template("settingnode_v2.html.jinja2").render(s=self, withsi=self._withsiprefix, startopen=0)
