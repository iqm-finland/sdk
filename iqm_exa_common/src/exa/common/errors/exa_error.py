import warnings


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


class InvalidSweepOptionsTypeError(ExaError, TypeError):
    """The type of sweep options is invalid."""

    def __init__(self, options: str, *args):
        warnings.warn("InvalidSweepOptionsTypeError is deprecated.", DeprecationWarning)
        super().__init__(f"Options have unsupported type of {options}", *args)
