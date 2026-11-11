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
"""Helper for lazily paginating limit/offset-based storage queries."""

from collections.abc import Callable, Iterator
from typing import TypeVar
import warnings

from exa.common.helpers import format_deprecated
from iqm.station_control.interface.list_with_meta import ListWithMeta

T = TypeVar("T")

_DEFAULT_PAGE_SIZE = 50


def paginate(fetch_page: Callable[[int, int], ListWithMeta[T]], *, limit: int | None = None) -> Iterator[T]:
    """Lazily yield items from a limit/offset-paginated backend, fetching pages on demand.

    The caller doesn't have to think about pagination: not passing ``limit`` yields every
    matching item, fetched lazily one page at a time, exactly like passing ``limit=0``.

    Args:
        fetch_page: Callable taking ``(limit, offset)`` for a single page and returning it.
        limit: Maximum number of items to yield in total. Not given (or 0, or a negative
            number) means unlimited: every matching item is yielded.

    """
    if limit is None:
        limit = 0
    offset = 0
    yielded = 0
    unlimited = limit <= 0
    while unlimited or yielded < limit:
        page_limit = _DEFAULT_PAGE_SIZE if unlimited else min(_DEFAULT_PAGE_SIZE, limit - yielded)
        page = fetch_page(page_limit, offset)
        items = list(page)
        for item in items:
            yield item
            yielded += 1
            if not unlimited and yielded >= limit:
                return

        offset += len(items)
        total = page.meta.count if page.meta is not None else None
        if len(items) < page_limit or (total is not None and offset >= total):
            return


def pop_deprecated_offset(kwargs: dict) -> None:
    """Remove ``offset`` from ``kwargs`` in place, warning if it was given.

    Pagination is handled internally by the storage implementation now, so ``offset`` no
    longer has any effect; it is still accepted, so old callers passing it don't crash.
    """
    if "offset" in kwargs:
        warnings.warn(
            DeprecationWarning(
                format_deprecated(old="the 'offset' query parameter", new=None, since="2026-10-17")
                + " Pagination is handled internally by the storage implementation; 'offset' has no effect."
            ),
            stacklevel=2,
        )
        kwargs.pop("offset")
