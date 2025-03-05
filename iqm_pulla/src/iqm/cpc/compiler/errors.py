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
"""Definitions of error classes used to raise issues during circuit compilation."""


class ClientError(RuntimeError):
    """Client submitted a bad request, and needs to be notified.

    Errors inheriting this class will be reported to the client in the
    ``message`` field of the failed job.
    """


class CircuitError(ClientError):
    """There is something wrong with the quantum circuit."""


class UnknownLogicalQubitError(CircuitError):
    """A logical qubit in the circuit has not been defined in the qubit mapping."""


class UnknownHardwareComponentError(CircuitError):
    """Circuit contains a reference to an unknown hardware component."""


class SettingsConventionError(ClientError):
    """While parsing Station Control settings, something breaks a structural or naming convention."""


class CalibrationError(ClientError):
    """A required calibration observation is missing from the calibration set, or an unknown
    gate calibration observation is encountered.
    """


class CompilationPassError(ValueError):
    """There is something wrong with the compilation pass."""


class InsufficientContextError(CompilationPassError):
    """The context provided to the compilation pass does not contain all necessary fields."""
