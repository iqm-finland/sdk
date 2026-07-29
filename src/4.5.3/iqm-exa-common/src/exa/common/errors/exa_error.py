class ExaError(Exception):
    """Base class for exa errors.

    Attributes:
        message: Error message.

    """

    def __init__(self, message: str, *args) -> None:
        super().__init__(message, *args)
        self.message = message

    def __str__(self) -> str:
        return self.message


class UnknownSettingError(ExaError, AttributeError):
    """This SettingNode does not have a given key."""


class EmptyComponentListError(ExaError, ValueError):
    """Error raised when an empty list is given as components for running an experiment."""
