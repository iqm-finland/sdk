#  ********************************************************************************
#  Copyright (c) 2019-2025 IQM Finland Oy.
#  All rights reserved. Confidential and proprietary.
#
#  Distribution or reproduction of any information contained herein
#  is prohibited without IQM Finland Oy’s prior written permission.
#  ********************************************************************************
"""Base model."""

from collections.abc import Mapping
from typing import Any, Self

import pydantic
from pydantic import ConfigDict


class BaseModel(pydantic.BaseModel):
    """Pydantic base model to change the behavior of pydantic globally.
    Note that setting model_config in child classes will merge the configs rather than override this one.
    https://docs.pydantic.dev/latest/concepts/config/#change-behaviour-globally
    """

    model_config = ConfigDict(
        extra="ignore",  # Ignore any extra attributes
        validate_assignment=True,  # Validate the data when the model is changed
        validate_default=False,  # Don't validate default values during validation
        ser_json_inf_nan="constants",  # Will serialize infinity and NaN values as Infinity and NaN
        frozen=True,  # This makes instances of the model potentially hashable if all the attributes are hashable
    )

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> Self:
        # # Safe for immutable models
        return self

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        """Copy the model.

        Without ``deep`` or ``update``, return ``self`` (immutable fast path);
        otherwise defer to Pydantic's standard ``model_copy``.
        """
        if not deep and update is None:
            return self  # Safe for immutable models
        # Fallback to normal Pydantic behaviour
        return super().model_copy(update=update, deep=deep)
