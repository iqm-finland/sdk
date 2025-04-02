from typing_extensions import deprecated

from exa.common.helpers.deprecation import format_deprecated


class ExaError(Exception):
    """Base class for exa errors.

    Attributes:
        message: Error message.

    """

    def __init__(self, message: str, *args):
        super().__init__(message, *args)
        self.message = message

    def __str__(self):
        return self.message


class UnknownSettingError(ExaError, AttributeError):
    """This SettingNode does not have a given key."""


class EmptyComponentListError(ExaError, ValueError):
    """Error raised when an empty list is given as components for running an experiment."""


@deprecated(format_deprecated(old="`InvalidSweepOptionsTypeError`", new="`Sweep.data`", since="28.3.2025"))
class InvalidSweepOptionsTypeError(ExaError, TypeError):
    """The type of sweep options is invalid."""

    def __init__(self, options: str, *args):
        super().__init__(f"Options have unsupported type of {options}", *args)
