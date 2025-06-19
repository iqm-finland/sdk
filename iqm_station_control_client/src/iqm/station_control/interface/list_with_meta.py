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
"""Models to extend standard list with metadata."""

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(kw_only=True)
class Meta:
    """Class holding metadata for list return values, like pagination related data."""

    count: int | None = None
    order_by: str | None = None
    limit: int | None = None
    offset: int | None = None
    errors: list[str] | None = None


class ListWithMeta(list):
    """Standard list extension holding optional metadata as well."""

    def __init__(self, items: Iterable, *, meta: Meta):
        super().__init__(items)
        self.meta = meta
