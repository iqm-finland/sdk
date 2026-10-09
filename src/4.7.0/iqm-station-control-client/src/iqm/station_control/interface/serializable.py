# Copyright 2026 IQM
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
"""Provides a mixin interface for objects that can be serialized to and from dictionary payloads."""

from abc import ABC, abstractmethod
from typing import Any, TypeVar

T = TypeVar("T", bound="Serializable")


class Serializable(ABC):
    """A mixin class adding uniform dictionary serialization capabilities to subclasses."""

    @abstractmethod
    def serialize(self) -> dict[str, Any]:
        """Serialize the object instance into a dictionary payload."""

    @classmethod
    @abstractmethod
    def deserialize(cls: type[T], data: dict[str, Any]) -> T:
        """Reconstruct an instance from a dictionary payload."""
