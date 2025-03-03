class ExaError(Exception):
    """Base class for exa errors.

    Attributes:
        message: Error message.

    """

    def __init__(self, message: str, *args):
        super().__init__(message, *args)
        self.message = message


class UnknownSettingError(ExaError, AttributeError):
    """This SettingNode does not have a given key."""


class InvalidParameterValueError(ExaError, ValueError):
    """The value set does not conform to the parameter restrictions."""


class InvalidSweepOptionsTypeError(ExaError, TypeError):
    """The type of sweep options is invalid."""

    def __init__(self, options: str, *args):
        super().__init__(f"Options have unsupported type of {options}", *args)


class RequestError(ExaError):
    """Error raised when something went wrong on the server after sending a request."""


class EmptyComponentListError(ExaError, ValueError):
    """Error raised when an empty list is given as components for running an experiment."""
