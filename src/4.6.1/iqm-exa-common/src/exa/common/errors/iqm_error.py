# Copyright 2024 IQM
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
"""Shared error definitions for the IQM ecosystem.

This module provides the unified exception hierarchy used by both client and
server components. By centralizing these definitions in a shared library,
we ensure that error states are communicated and handled consistently
across the network.

Architectural Context
---------------------

1. Symmetric Logic: Components like :class:`.SettingNode` or :class:`.Experiment` use
   these errors for local validation. Sharing them ensures that a
   Validation failure on the client feels identical to one on the server.
2. Type Preservation: Server-side errors (e.g., :class:`InternalServerError`)
   are included here so the client can deserialize them into specific
   Python types rather than generic catch-all exceptions.
3. Wire Compatibility: These exceptions map directly to HTTP status
   codes and the :class:`IQMServerError` Pydantic data model.

Client-Relevant Errors
----------------------

While all errors can be received from the server, the following are
most commonly raised or handled specifically in client-side logic:

- :class:`ValidationError`: Local input/parameter checks before transmission.
- :class:`NotFoundError`: Handling missing resources in lookups.
- :class:`ConflictError`: Managing state-machine or concurrency issues.
"""

from abc import ABC
import logging
from typing import ClassVar


class IQMError(Exception, ABC):
    """Base exception for the IQM ecosystem logic.

    This class serves as a universal functional representation of a failure
    within application logic. It is defined in 'exa-common' to provide a
    consistent error contract for all services, libraries, and clients,
    regardless of whether they communicate over a network or run locally.

    IQMError is abstract and cannot be instantiated directly. Instead,
    specialized subclasses are used to categorize the failure (e.g.,
    :class:`NotFoundError` or :class:`ValidationError`).

    While these exceptions are context-agnostic, the core subclasses are loosely
    equivalent to standard HTTP status codes. This enables a consistent
    transmission contract.
    """

    message: str
    """Human-readable description of the error."""
    error_code: str | None
    """Optional machine-readable identifier for programmatic error handling."""
    log_level: int
    """Optional logging severity (e.g., logging.ERROR) used to control logging noise."""

    _ERROR_CODE: ClassVar[str | None] = None
    """Optional class-level default error identifier.

    Subclasses can override this to provide a specific machine-readable
    string (e.g., 'unknown_setting_error') that applies to all instances
    of that subclass by default.
    """

    def __init__(self, message: str, *, error_code: str | None = None, log_level: int = logging.ERROR):
        if type(self) is IQMError:
            raise TypeError("IQMError is abstract and cannot be instantiated directly.")

        self.message = message
        # Use provided error_code, otherwise fall back to the class-level _ERROR_CODE
        self.error_code = error_code or self._ERROR_CODE
        self.log_level = log_level
        super().__init__(message)


class InvalidOperationError(IQMError):
    """Raised when the operation parameters are invalid or the action is unsupported."""


class AuthenticationError(IQMError):
    """Raised when the caller is not authenticated or credentials are missing.

    The client must provide valid credentials or log in to access the resource.
    """


class ForbiddenError(IQMError):
    """Raised when the user is authenticated but lacks permission for the operation.

    Logging in again will not help; the user's account does not have the required scope.
    """


class NotFoundError(IQMError):
    """Raised when a requested resource or data point was not found.

    Used when a resource is expected to exist, such as a lookup by a specific ID.
    """


class ConflictError(IQMError):
    """Raised when an operation conflicts with the current state of the system.

    Typically used for duplicate submissions of unique data or state-machine violations.
    """


class DataSizeError(IQMError):
    """Raised when the provided data exceeds the allowed size limits."""


class ValidationError(IQMError):
    """Raised when data is well-formed but semantically invalid.

    Used for invalid input values, range violations, or business logic failures.
    """


class RateLimitError(IQMError):
    """Raised when the operation frequency exceeds the allowed limit in a given timeframe."""


class InternalSystemError(IQMError):
    """Raised when an unexpected internal failure occurs.

    This indicates a system-level malfunction or logic error rather than
    an expected or recoverable failure mode.
    """


class BadGatewayError(IQMError):
    """Raised when an invalid or unintelligible response is received from a dependency.

    This dependency could be an upstream service, a database, or an external
    hardware controller.
    """


class SystemUnavailableError(IQMError):
    """Raised when a required system component is temporarily unable to operate.

    This usually indicates the component is overloaded, initializing, or
    undergoing maintenance.
    """


class OperationTimeoutError(IQMError):
    """Raised when an operation or a call to a dependency fails to complete in time."""


# =============================================================================
# SPECIALIZED DOMAIN ERRORS
# =============================================================================
# The classes defined below are provided for convenience to capture specific
# failure modes and provide standardized error messages and codes.
#
# Unlike the "Base Subclasses" defined above, these are not mapped directly
# to HTTP status codes. Note that during client-side reconstruction,
# the specific type (e.g., UnknownSettingError) is lost and replaced by its
# parent Base Subclass (e.g., NotFoundError).
# =============================================================================


class EmptyComponentListError(ValidationError, ValueError):
    """Raised when an empty list is given as components for running an experiment."""

    _ERROR_CODE = "empty_component_list_error"


class UnknownSettingError(NotFoundError, AttributeError):
    """Raised when a SettingNode attribute is not found."""

    _ERROR_CODE = "unknown_setting_error"


class EngineInitialisationError(SystemUnavailableError):
    """Raised when the station control engine fails to initialize."""

    def __init__(
        self,
        message: str = (
            "Station-control engine appears not initialized. A probable cause is that the "
            "job executor worker process failed to start. To find out what went wrong, "
            "please check station-control logs from the very beginning of the service startup. "
            "If station-control loggers are configured with 'verbose=True', each log message "
            "contains the module path; filtering by 'job_executor' will show all messages "
            "from the worker process, including possible errors. It is highly recommended to "
            "inspect the entire log for any preceding warnings or errors."
        ),
        log_level: int = logging.WARNING,
    ):
        super().__init__(
            message=message,
            error_code="engine_initialization_error",
            log_level=log_level,
        )


class CircuitValidationError(ValidationError):
    """Raised when a circuit is well-formed but violates hardware or logical constraints."""

    _ERROR_CODE = "circuit_validation_error"


class CircuitTranspilationError(ValidationError):
    """Raised when a circuit cannot be transpiled, typically due to hardware constraints.

    This occurs if the circuit contains gates, connectivity, or operations (like invalid
    MOVE loci) that are not supported by the target architecture.
    """

    _ERROR_CODE = "circuit_transpilation_error"


class CircuitExecutionError(InternalSystemError):
    """Raised when a failure occurs during the physical execution of a circuit.

    In synchronous flows, this error is raised directly to the caller.
    In asynchronous flows, this error is only raised if the execution fails
    before a job can be successfully created; otherwise, the failure is
    recorded within the job's terminal status.
    """

    _ERROR_CODE = "circuit_execution_error"


class UnknownObservationError(ValidationError):
    """Raised when an observation name is syntactically correct but contains unknown elements.

    This indicates that while the format of the name is valid, the specific
    identifiers (e.g. keys or indices) do not exist in the current context.
    """

    _ERROR_CODE = "unknown_observation_error"


# --- Backwards Compatibility Aliases ---
# Legacy aliases maintained to prevent breaking changes for existing consumers.
# These will be formally deprecated in a future release.
# RATIONALE:
# We have moved away from transport-specific naming (e.g., 'BadRequest', 'InternalServer')
# to transport-agnostic, domain-driven naming (e.g., 'InvalidOperation', 'InternalSystem').
# As these errors are used across  the entire stack, names should reflect the nature of
# the failure rather than a specific protocol like HTTP.
BadRequestError = InvalidOperationError
UnauthorizedError = AuthenticationError
PayloadTooLargeError = DataSizeError
TooManyRequestsError = RateLimitError
InternalServerError = InternalSystemError
ServiceUnavailableError = SystemUnavailableError
GatewayTimeoutError = OperationTimeoutError
