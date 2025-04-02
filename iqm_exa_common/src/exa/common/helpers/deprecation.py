from datetime import datetime
from typing import LiteralString


def format_deprecated(old: str, new: str | None, since: str) -> LiteralString:
    datetime.strptime(since, "%d.%m.%Y")
    message: str = (
        f"{old} is deprecated since {since}, it can be be removed from the codebase in the next major release."
    )
    if new:
        message += f" Use {new} instead."
    return message
