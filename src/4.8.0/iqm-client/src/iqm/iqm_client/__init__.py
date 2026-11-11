# Copyright 2021 IQM client developers
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
"""Client-side library for connecting to and executing quantum circuits on IQM quantum computers."""

__all__ = [
    "STANDARD_DD_STRATEGY",
    "AWGProperties",
    "AuthenticationError",
    "BadGatewayError",
    "CalibrationSet",
    "ChipTopology",
    "Circuit",
    "CircuitBatch",
    "CircuitCompilationOptions",
    "CircuitExecutionError",
    "CircuitJobDefinition",
    "CircuitJobParameters",
    "CircuitJobTracker",
    "CircuitMeasurementCounts",
    "CircuitMeasurementResults",
    "CircuitOperation",
    "CircuitTranspilationError",
    "CircuitValidationError",
    "Compiler",
    "ConflictError",
    "DDMode",
    "DDStrategy",
    "DataSizeError",
    "DynamicQuantumArchitecture",
    "EmptyComponentListError",
    "EngineInitialisationError",
    "ExistingMoveHandlingOptions",
    "ForbiddenError",
    "GateImplementationInfo",
    "GateInfo",
    "HeraldingMode",
    "IQMClient",
    "IQMError",
    "IQMServerClient",
    "IQMServerExecutor",
    "IQMServerStorage",
    "InternalSystemError",
    "InvalidOperationError",
    "IterationTracker",
    "JobTracker",
    "LatestFromStash",
    "Locus",
    "MoveGateFrameTrackingMode",
    "MoveGateValidationMode",
    "NotFoundError",
    "ObservationHandler",
    "OperationTimeoutError",
    "QIRCode",
    "QualityMetricSet",
    "QubitMapping",
    "RateLimitError",
    "Resolution",
    "RunJobTracker",
    "StaticQuantumArchitecture",
    "StrUUIDOrDefault",
    "SystemUnavailableError",
    "UnknownObservationError",
    "UnknownSettingError",
    "ValidationError",
    "build_quantum_ops",
    "simplify_architecture",
    "transpile_insert_moves",
    "transpile_remove_moves",
    "validate_circuit",
    "validate_circuit_instructions",
    "validate_circuit_moves",
    "validate_instruction",
    "validate_qubit_mapping",
]

from importlib.metadata import PackageNotFoundError, version
import sys
import warnings

from iqm.iqm_server_client import IQMServerClient, IQMServerExecutor, IQMServerStorage
from iqm.models.channel_properties import AWGProperties

from exa.common.errors import (
    AuthenticationError,
    BadGatewayError,
    CircuitExecutionError,
    CircuitTranspilationError,
    CircuitValidationError,
    ConflictError,
    DataSizeError,
    EmptyComponentListError,
    EngineInitialisationError,
    ForbiddenError,
    InternalSystemError,
    InvalidOperationError,
    IQMError,
    NotFoundError,
    OperationTimeoutError,
    RateLimitError,
    SystemUnavailableError,
    UnknownObservationError,
    UnknownSettingError,
    ValidationError,
)
from exa.common.qcm_data import ChipTopology
from iqm.cpc.compiler.compiler import Compiler
from iqm.cpc.core.observation.observation_handler import ObservationHandler
from iqm.cpc.core.observation.observation_loading_rules import LatestFromStash
from iqm.pulse import Circuit, CircuitOperation
from iqm.pulse.builder import build_quantum_ops
from iqm.station_control.client.job_trackers import CircuitJobTracker, JobTracker, RunJobTracker
from iqm.station_control.interface.models import (
    CircuitBatch,
    DDMode,
    DDStrategy,
    DynamicQuantumArchitecture,
    GateImplementationInfo,
    GateInfo,
    HeraldingMode,
    Locus,
    MoveGateFrameTrackingMode,
    MoveGateValidationMode,
    QIRCode,
    QubitMapping,
    StaticQuantumArchitecture,
)
from iqm.station_control.interface.models.circuit import (
    CircuitJobDefinition,
    CircuitMeasurementCounts,
    CircuitMeasurementResults,
)
from iqm.station_control.interface.models.observation_set import CalibrationSet, QualityMetricSet
from iqm.station_control.interface.models.type_aliases import StrUUIDOrDefault

from .iqm_client import IQMClient, IterationTracker
from .models import STANDARD_DD_STRATEGY, CircuitCompilationOptions, CircuitJobParameters, validate_circuit
from .transpile import (
    ExistingMoveHandlingOptions,
    Resolution,
    simplify_architecture,
    transpile_insert_moves,
    transpile_remove_moves,
)
from .util import print_env_vars
from .validation import (
    validate_circuit_instructions,
    validate_circuit_moves,
    validate_instruction,
    validate_qubit_mapping,
)

try:
    DIST_NAME = "iqm-client"
    __version__ = version(DIST_NAME)
except PackageNotFoundError:
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError
