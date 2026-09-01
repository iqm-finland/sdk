# Copyright 2025 IQM
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
"""Models to extend a standard list with metadata."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Generic, TypeVar


@dataclass(kw_only=True)
class Meta:
    """Class holding metadata for list return values, including pagination and query statistics."""

    count: int | None = None
    """Total number of items existing in the storage that match the query criteria."""

    order_by: str | None = None
    """Ordering rule applied to the queried items (e.g., "-created_timestamp" for descending)."""

    limit: int | None = None
    """Maximum number of items returned in this query response (page size).

    If set to 0 or a negative integer, pagination is disabled and the query attempts to
    return all available records (matching ``count - offset``).
    Note: Disabling the limit may result in extremely heavy database and network load."""

    offset: int | None = None
    """Number of items to skip from the beginning of the result sequence.

    This value can be incremented in successive queries to loop through pages
    until ``count`` is reached."""

    errors: list[str] | None = None
    """List of error messages encountered during the execution of the query."""


T = TypeVar("T")


class ListWithMeta(list, Generic[T]):
    """Standard list extension holding optional metadata as well."""

    meta: Meta | None

    def __init__(self, items: Iterable[T], *, meta: Meta | None = None):
        super().__init__(items)
        self.meta = meta
