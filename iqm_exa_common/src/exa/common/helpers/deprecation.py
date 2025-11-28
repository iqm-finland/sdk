from datetime import date
from typing import LiteralString, cast


def _assert_iso_8601_format(since: str) -> None:
    try:
        # This is strict about the ISO calendar date form.
        # '2025-9-3' will fail; '2025-09-03' passes.
        date.fromisoformat(since)
    except ValueError:
        raise ValueError("Invalid date format. Use 'YYYY-MM-DD'.")


def format_deprecated(old: str, new: str | None, since: str) -> LiteralString:  # noqa: D103
    _assert_iso_8601_format(since)
    message: str = (
        f"{old} is deprecated since {since}, it can be be removed from the codebase in the next major release."
    )
    if new:
        message += f" Use {new} instead."
    return cast(LiteralString, message)
