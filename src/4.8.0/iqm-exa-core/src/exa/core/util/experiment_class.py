# Copyright 2019-2026 IQM Finland Oy
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
"""Utility functions for creating instances based on subclasses of provided class."""

from typing import TypeVar


def _get_name_to_class(cls: type) -> dict[str, type]:
    """Map from name attribute to class for all subclasses of ``cls`` which have a ``name`` attribute."""
    name_to_class: dict[str, type] = {}

    def _get_all_subclasses(cls: type) -> set[type]:
        direct_subclasses = cls.__subclasses__()
        return set(direct_subclasses).union(s for c in direct_subclasses for s in _get_all_subclasses(c))

    for _cls in _get_all_subclasses(cls):
        if (name := getattr(_cls, "name", None)) is not None:
            name_to_class[name] = _cls
    return name_to_class


T = TypeVar("T")


def get_class_by_name(cls: type[T], name: str) -> type[T] | None:
    """Try to find a subclass of ``cls`` with the given name.

    Args:
        cls: Class whose subclasses to search.
        name: Name of the subclass.

    Returns:
        Subclass, or None if not found.

    """
    return _get_name_to_class(cls).get(name, None)
