import pytest

from exa.common.helpers.deprecation import format_deprecated


def test_format_deprecated_all_valid_positional_parameters():
    message = format_deprecated("old", "new", "2025-03-28")
    assert message == (
        "old is deprecated since 2025-03-28, "
        "it can be be removed from the codebase in the next major release. Use new instead."
    )


def test_format_deprecated_all_valid_keyword_parameters():
    message = format_deprecated(old="old", new="new", since="2025-03-28")
    assert message == (
        "old is deprecated since 2025-03-28, "
        "it can be be removed from the codebase in the next major release. Use new instead."
    )


def test_format_deprecated_new_is_none():
    # This can be used when there is no new replacing functionality for the deprecated feature.
    # "new" is still required to be given explicitly as "None" for clarity.
    message = format_deprecated(old="old", new=None, since="2025-03-28")
    assert message == (
        "old is deprecated since 2025-03-28, it can be be removed from the codebase in the next major release."
    )


def test_invalid_date_format():
    # To enforce consistency, format_deprecated accepts only one date format as an input.

    with pytest.raises(ValueError, match="Invalid date format. Use 'YYYY-MM-DD'."):
        format_deprecated(old="old", new="new", since="2025-9-3")

    with pytest.raises(ValueError, match="Invalid date format. Use 'YYYY-MM-DD'."):
        format_deprecated(old="old", new="new", since="28.03.2025")

    with pytest.raises(ValueError, match="Invalid date format. Use 'YYYY-MM-DD'."):
        format_deprecated(old="old", new="new", since="28/3/2025")

    with pytest.raises(ValueError, match="Invalid date format. Use 'YYYY-MM-DD'."):
        format_deprecated(old="old", new="new", since="2025/3/28")

    with pytest.raises(ValueError, match="Invalid date format. Use 'YYYY-MM-DD'."):
        format_deprecated(old="old", new="new", since="latest")

    with pytest.raises(ValueError, match="Invalid date format. Use 'YYYY-MM-DD'."):
        format_deprecated(old="old", new="new", since="2025-02-31")
