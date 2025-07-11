from collections.abc import Mapping
from typing import Any, Self

import pydantic
from pydantic import ConfigDict
from typing_extensions import deprecated

from exa.common.helpers.deprecation import format_deprecated


class BaseModel(pydantic.BaseModel):
    """Pydantic base model to change the behaviour of pydantic globally.
    Note that setting model_config in child classes will merge the configs rather than override this one.
    https://docs.pydantic.dev/latest/concepts/config/#change-behaviour-globally
    """

    model_config = ConfigDict(
        # TODO: Consider setting extra="forbid", extra="ignore" is needed only if new clients use older servers
        #  After station control API versioning, backwards compatibility works, i.e. old clients can use newer servers.
        #  Reverse shouldn't be needed, since we don't promise any forwards compatibility.
        extra="ignore",  # Ignore any extra attributes
        validate_assignment=True,  # Validate the data when the model is changed
        validate_default=True,  # Validate default values during validation
        ser_json_inf_nan="constants",  # Will serialize infinity and NaN values as Infinity and NaN
        frozen=True,  # This makes instances of the model potentially hashable if all the attributes are hashable
    )

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = True) -> Self:
        """Returns a copy of the model.

        Overrides the Pydantic default 'model_copy' to set 'deep=True' by default.
        """
        return super().model_copy(update=update, deep=deep)

    @deprecated(format_deprecated(old="`copy` method", new="`model_copy`", since="28.3.2025"))
    def copy(self, **kwargs) -> Self:
        """Returns a copy of the model."""
        return super().copy(update=kwargs, deep=True)
