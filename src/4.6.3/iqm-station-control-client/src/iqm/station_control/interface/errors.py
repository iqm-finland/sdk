# Copyright 2026 IQM
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Data models and mapping utilities for IQM REST API error handling.

This module provides the :class:`IQMServerError` DTO, which defines the standard
JSON structure for error responses across the IQM ecosystem. It also contains
the logic to bridge the gap between HTTP status codes and the :class:`IQMError`
exception hierarchy.

The mapping logic supports:

1. **Serialization & Encoding:** Transforming server-side exceptions into
       :class:`IQMServerError` payloads while determining the appropriate
       HTTP status code and preserving error metadata (codes, messages, and sources).
2. **Reconstruction & Decoding:** Converting received :class:`IQMServerError`
       data back into the :class:`IQMError` hierarchy on the client side, ensuring
       failure data remains actionable in application logic.
"""

from __future__ import annotations

from http import HTTPStatus
from typing import Any, NoReturn

from pydantic import computed_field, model_validator

from exa.common.errors.iqm_error import (
    AuthenticationError,
    BadGatewayError,
    ConflictError,
    DataSizeError,
    ForbiddenError,
    InternalSystemError,
    InvalidOperationError,
    IQMError,
    NotFoundError,
    OperationTimeoutError,
    RateLimitError,
    SystemUnavailableError,
    ValidationError,
)
from exa.common.helpers.deprecation import format_deprecated
from iqm.station_control.interface.models.type_aliases import Source
from iqm.station_control.interface.pydantic_base import PydanticBase


class IQMServerError(PydanticBase):
    """Data transfer model for communicating error details over the REST API.

    This model defines the structured JSON schema used to transport error details
    between components. It is used in both standard error response bodies (4xx/5xx)
    and within the 'errors' field of successful (200 OK) asynchronous job results.

    This model acts as a data-carrying DTO (Data Transfer Object). High-level
    application code should map this model back into a raisable IQMError to
    represent the failure as a standard Python exception:

    * **Domain Logic:** Errors carry a unique :attr:`error_code` (a stable machine-readable
      string) and a :attr:`message` (a human-readable description).
    * **Network Transmission:** When crossing a network boundary (e.g., via REST),
      the exception is mapped to an HTTP status code. **Only the base category
      subclass (like :class:`ValidationError`) is recoverable on the receiving end.**
    * **Information Preservation:** While the specific server-side class type
      (e.g., :class:`UnknownSettingError`) is lost during transmission, the
      :attr:`error_code` and :attr:`message` attributes remain intact. This ensures
      the client receives full diagnostic information and can react to the specific
      failure code.
    """

    message: str
    """Human-readable description of the error."""
    error_code: str | None = None
    """Optional machine-readable identifier for programmatic error handling."""
    source: Source
    """Originating component or service of the error."""

    def __str__(self) -> str:
        """Pretty printing."""
        return f"{self.source}: {self.error_code}: {self.message}"

    @computed_field(
        json_schema_extra={
            "deprecated": True,
            "description": format_deprecated(old="`detail`", new="`message`", since="2025-09-16"),
        },
    )
    def detail(self) -> str:
        """Human-readable description of the error."""
        # "detail" is deprecated to unify the format with IQM Server which uses "message"
        return self.message

    @model_validator(mode="before")
    @classmethod
    def ensure_compatibility(cls, data: Any) -> Any:
        """Harmonizes differences between various service response formats.

        This handles both legacy fields (detail -> message) and transitional
        requirements (defaulting missing 'source' for older server versions).
        """
        if not isinstance(data, dict):
            return data

        # Backward compatibility: map 'detail' to 'message'
        if "detail" in data and "message" not in data:
            data["message"] = data["detail"]

        # Forward compatibility: default missing 'source' for older servers
        if "source" not in data:
            # TODO: Remove this default when all IQM services provide 'source'
            data["source"] = "iqm-server"

        return data

    def raise_exception(self, status_code: int) -> NoReturn:
        """Maps the error data to the appropriate IQMError and raises it."""
        error_class = map_from_status_code_to_error(status_code)
        raise error_class(message=self.message, error_code=self.error_code)

    @classmethod
    def from_exception(cls, exception: Exception, *, source: Source) -> IQMServerError:
        """Factory method to create a DTO from an Exception."""
        return cls(message=str(exception), error_code=getattr(exception, "error_code", None), source=source)


_ERROR_TO_STATUS_CODE_MAPPING = {
    InvalidOperationError: HTTPStatus.BAD_REQUEST,  # 400
    AuthenticationError: HTTPStatus.UNAUTHORIZED,  # 401
    ForbiddenError: HTTPStatus.FORBIDDEN,  # 403
    NotFoundError: HTTPStatus.NOT_FOUND,  # 404
    ConflictError: HTTPStatus.CONFLICT,  # 409
    DataSizeError: HTTPStatus.REQUEST_ENTITY_TOO_LARGE,  # 413
    ValidationError: HTTPStatus.UNPROCESSABLE_ENTITY,  # 422
    RateLimitError: HTTPStatus.TOO_MANY_REQUESTS,  # 429
    InternalSystemError: HTTPStatus.INTERNAL_SERVER_ERROR,  # 500
    BadGatewayError: HTTPStatus.BAD_GATEWAY,  # 502
    SystemUnavailableError: HTTPStatus.SERVICE_UNAVAILABLE,  # 503
    OperationTimeoutError: HTTPStatus.GATEWAY_TIMEOUT,  # 504
}
_STATUS_CODE_TO_ERROR_MAPPING = {value: key for key, value in _ERROR_TO_STATUS_CODE_MAPPING.items()}


def map_from_error_to_status_code(error: IQMError) -> HTTPStatus:
    """Map a IQMError to an HTTPStatus code."""
    # Try direct lookup (fastest)
    if type(error) in _ERROR_TO_STATUS_CODE_MAPPING:
        return _ERROR_TO_STATUS_CODE_MAPPING[type(error)]

    # Fallback to inheritance check (handles UnknownSettingError -> ValidationError)
    for error_type, status_code in _ERROR_TO_STATUS_CODE_MAPPING.items():
        if isinstance(error, error_type):
            return status_code

    return HTTPStatus.INTERNAL_SERVER_ERROR


def map_from_status_code_to_error(status_code: HTTPStatus | int) -> type[IQMError]:
    """Map an HTTPStatus code to a IQMError."""
    if isinstance(status_code, int):
        status_code = HTTPStatus(status_code)
    return _STATUS_CODE_TO_ERROR_MAPPING.get(status_code, InternalSystemError)
