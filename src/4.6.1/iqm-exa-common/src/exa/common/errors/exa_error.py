#  ********************************************************************************
#  Copyright (c) 2019-2020 IQM Finland Oy.
#  All rights reserved. Confidential and proprietary.
#
#  Distribution or reproduction of any information contained herein
#  is prohibited without IQM Finland Oy’s prior written permission.
#  ********************************************************************************
"""EXA errors."""

import warnings

from typing_extensions import deprecated

from exa.common.errors.iqm_error import ValidationError
from exa.common.helpers.deprecation import format_deprecated

DEPRECATION_MSG = (
    format_deprecated(old="`ExaError`", new=None, since="2026-03-10")
    + " Please use the most relevant subclass of `IQMError` (e.g., `NotFoundError`, `ConflictError`) "
    "instead of the base `ValidationError` where possible."
)


@deprecated(DEPRECATION_MSG)
class ExaError(ValidationError):
    """Deprecated alias for ValidationError.

    This class remains for backward compatibility but will be removed in
    a future major release. Users should migrate to a specific subclass
    of the abstract `IQMError` to ensure correct error mapping.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(DEPRECATION_MSG, category=DeprecationWarning, stacklevel=2)
        super().__init__(*args, **kwargs)
