# Copyright 2024-2025 IQM
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

"""Common data types and exceptions for the IQM Pulla interface.

Many of these must be identical to those in iqm-client.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias
from uuid import UUID

from exa.common.data.value import ObservationValue
from iqm.cpc.interface.compiler import Locus


class CHADRetrievalException(Exception):
    """Exception for CHAD retrieval failures."""


class SettingsRetrievalException(Exception):
    """Exception for Station Control settings retrieval failures."""


class ChipLabelRetrievalException(Exception):
    """Exception for chip label retrieval failures."""


# Map from OIL tuple to a CalibrationError.
CalibrationErrors: TypeAlias = dict[tuple[str, str, Locus], str]


class TaskStatus(StrEnum):
    """Status of a Station Control task."""

    READY = "READY"
    """Task has completed successfully"""

    FAILED = "FAILED"
    """Task has failed"""

    PROGRESS = "PROGRESS"
    """Task is being executed"""

    PENDING = "PENDING"
    """Task is waiting to be executed"""


CalibrationSet = dict[str, ObservationValue]
CalibrationSetId = UUID

CircuitMeasurementResults = dict[str, list[list[int]]]
"""Measurement results from a single circuit/schedule. For each measurement operation in the circuit,
maps the measurement key to the corresponding results. The outer list elements correspond to shots,
and the inner list elements to the qubits measured in the measurement operation."""

CircuitMeasurementResultsBatch = list[CircuitMeasurementResults]
"""Type that represents measurement results for a batch of circuits."""


@dataclass
class StationControlResult:
    """Result of a station control task"""

    sweep_id: UUID
    """ID of the executed sweep"""
    task_id: UUID  # TODO? Rename to job_id
    """ID of the station control task"""
    status: TaskStatus
    """Status of the station control task"""
    start_time: str | None = None
    """Time when the sweep began in the station control"""
    end_time: str | None = None
    """Time when the sweep ended in the station control"""
    result: CircuitMeasurementResultsBatch | None = None
    """Sweep results converted to the circuit measurement results expected by the client"""
    message: str | None = None
    """Information about task failure"""


ACQUISITION_LABEL_KEY = "m{idx}"
ACQUISITION_LABEL = "{qubit}__{key}"
MEASUREMENT_MODE_KEY = "__MEASUREMENT_MODE"
HERALDING_KEY = "__HERALD"
RESTRICTED_MEASUREMENT_KEYS = [MEASUREMENT_MODE_KEY, HERALDING_KEY]

# NOTE the buffer duration needs to match all instrument granularities!
# Integer multiples of 80 ns work with 1.8 GHz, 2.0 GHz and 2.4 GHz sample rates and 16 sample granularity,
# which should cover all instruments currently in use. In s.
_BUFFER_GRANULARITY = 80e-9
BUFFER_AFTER_MEASUREMENT_PROBE = 4 * _BUFFER_GRANULARITY
"""Buffer that allows the readout resonator and qubit state to stabilize after a probe pulse, in s.
TODO: not needed after EXA-2089 is done."""
