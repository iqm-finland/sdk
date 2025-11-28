# Copyright 2025 IQM
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
"""Models related to quantum circuit execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Any, TypeAlias
from uuid import UUID

from pydantic import AliasChoices, BeforeValidator, Field, PlainSerializer, WithJsonSchema, computed_field

from exa.common.helpers.deprecation import format_deprecated
from iqm.pulse import Circuit
from iqm.station_control.interface.pydantic_base import PydanticBase

if TYPE_CHECKING:
    from iqm.pulse import Circuit, CircuitOperation


PRXSequence: TypeAlias = list[tuple[float, float]]
"""Sequence of PRX gates. A generic PRX gate is defined by rotation angle and phase angle, theta and phi,
respectively."""

QIRCode: TypeAlias = str
"""QIR program code in string representation."""


# TODO: remove when CUDA-Q supports the new circuit format
@dataclass
class _Instruction:
    """An instruction in a quantum circuit. Old format."""

    name: str
    implementation: str | None = None
    qubits: tuple[str, ...] = field(default_factory=tuple)
    args: dict[str, Any] = field(default_factory=dict)

    def to_cpc_type(self) -> CircuitOperation:
        """Convert the model to a dataclass."""
        return CircuitOperation(
            name=self.name,
            implementation=self.implementation,
            locus=self.qubits,
            args=self.args,
        )


# TODO: remove when CUDA-Q supports the new circuit format
@dataclass
class _Circuit:
    """Quantum circuit to be executed. Old format."""

    name: str
    instructions: tuple[_Instruction, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] | None = None

    def to_cpc_type(self) -> Circuit:
        """Convert the model to a dataclass."""
        return Circuit(
            name=self.name,
            instructions=tuple(instruction.to_cpc_type() for instruction in self.instructions),
        )


CircuitBatch: TypeAlias = list[Circuit | _Circuit | QIRCode]
"""Sequence of quantum circuits to be executed together in a single batch."""


CircuitMeasurementResults: TypeAlias = dict[str, list[list[int]]]
"""Measurement results from a single circuit.

For each measurement operation in the circuit, maps the measurement key to the corresponding results.
``results[key][shot][qubit_index]`` is the result of measuring the
``qubit_index``'th qubit in measurement operation ``key`` in the shot ``shot``.
The results are non-negative integers representing the computational basis state (for qubits, 0 or 1)
that was the measurement outcome.
"""

CircuitMeasurementResultsBatch: TypeAlias = list[CircuitMeasurementResults]
"""Type that represents measurement results for a batch of circuits."""


class HeraldingMode(StrEnum):
    """Heralding mode for circuit execution.

    Heralding is the practice of generating data about the state of qubits prior to execution of a circuit.
    This can be achieved by measuring the qubits immediately before executing each shot for a circuit.
    """

    NONE = "none"
    """Do not do any heralding."""
    ZEROS = "zeros"
    """For each circuit, perform a heralding measurement after the initial reset on all the QPU components
    used in the circuit that have the "measure" operation available in the calset.
    Only retain shots where all the components are measured to be in the zero state.

    Note: in this mode, the number of shots returned after execution will be <= the requested amount
    due to the post-selection based on heralding data.
    If zero shots would be returned, the job will have the FAILED status."""


class MoveGateValidationMode(StrEnum):
    """MOVE gate validation mode for circuit compilation. This options is meant for advanced users."""

    STRICT = "strict"
    """Perform standard MOVE gate validation: MOVE(qubit, resonator) gates must only
    appear in sandwiches (pairs). Inside a sandwich there must be no gates acting on the
    MOVE qubit, and no other MOVE gates acting on the resonator."""
    ALLOW_PRX = "allow_prx"
    """Allow PRX gates on the MOVE qubit inside MOVE sandwiches during validation."""
    NONE = "none"
    """Do not perform any MOVE gate validation."""


class MoveGateFrameTrackingMode(StrEnum):
    """MOVE gate reference frame tracking mode for circuit compilation. This option is meant for advanced users."""

    FULL = "full"
    """Apply both explicit z rotations on the resonator, and a dynamic phase correction
    due to qubit-resonator detuning, to the qubit at the end of a MOVE sandwich."""
    NO_DETUNING_CORRECTION = "no_detuning_correction"
    """Only apply explicit z rotations on the resonator to the qubit at the end of the sandwich.
    Do not apply a detuning correction, the user is expected to do this manually."""
    NONE = "none"
    """Do not perform any MOVE gate frame tracking. The user is expected to do this manually."""


class DDMode(StrEnum):
    """Dynamical Decoupling (DD) mode for circuit execution."""

    DISABLED = "disabled"
    """Do not apply dynamical decoupling."""
    ENABLED = "enabled"
    """Apply dynamical decoupling."""


class DDStrategy(PydanticBase):
    """Describes a particular dynamical decoupling strategy.

    The current standard DD stategy can be found in :attr:`~iqm.cpc.compiler.dd.STANDARD_DD_STRATEGY`,
    but users can use this class to provide their own dynamical decoupling strategies.

    See Ezzell et al., Phys. Rev. Appl. 20, 064027 (2022) for information on DD sequences.
    """

    # TODO station-control-client docs need to support bibtex citations.
    # TODO :cite:`Ezzell_2022`

    merge_contiguous_waits: bool = Field(default=True)
    """Merge contiguous ``Wait`` instructions into one if they are separated only by ``Block`` instructions."""

    target_qubits: frozenset[str] | None = Field(default=None)
    """Qubits on which dynamical decoupling should be applied. If ``None``, all qubits are targeted."""

    skip_leading_wait: bool = Field(default=True)
    """Skip processing leading ``Wait`` instructions."""

    skip_trailing_wait: bool = Field(default=True)
    """Skip processing trailing ``Wait`` instructions."""

    gate_sequences: list[tuple[int, str | PRXSequence, str]] = Field(default_factory=list)
    """Available decoupling gate sequences to choose from in this strategy.

    Each sequence is defined by a tuple of ``(ratio, gate pattern, align)``:

        * ratio: Minimal duration for the sequence (in PRX gate durations).

        * gate pattern: Gate pattern can be defined in two ways. It can be a string containing "X" and "Y" characters,
          encoding a PRX gate sequence. For example, "YXYX" corresponds to the
          XY4 sequence, "XYXYYXYX" to the EDD sequence, etc. If more flexibility is needed, a gate pattern can be
          defined as a sequence of PRX gate argument tuples (that contain a rotation angle and a phase angle). For
          example, sequence "YX" could be written as ``[(math.pi, math.pi / 2), (math.pi, 0)]``.

        * align: Controls the alignment of the sequence within the time window it is inserted in. Supported values:

          - "asap": Corresponds to a ASAP-aligned sequence with no waiting time before the first pulse.
          - "center": Corresponds to a symmetric sequence.
          - "alap": Corresponds to a ALAP-aligned sequence.

    The Dynamical Decoupling algorithm uses the best fitting gate sequence by first sorting them
    by ``ratio`` in descending order. Then the longest fitting pattern is determined by comparing ``ratio``
    with the duration of the time window divided by the PRX gate duration.
    """


def _parse_legacy_qubit_mapping(value: Any) -> dict[str, str] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): str(value) for key, value in value.items()}
    # Deprecated since 2025-11-03.
    # `qubit_mapping` currently uses the legacy list format on the wire,
    # but it should eventually be replaced with the new `QubitMapping` (dict-based) format.
    # The switch can only be made once all legacy clients have been updated,
    # as they still depend on receiving the old response format.
    return {item["logical_name"]: item["physical_name"] for item in value}


def _serialize_as_legacy_qubit_mapping(mapping: dict[str, str] | None) -> list[dict[str, str]] | None:
    if mapping is None:
        return None
    # Deprecated since 2025-11-03.
    # `qubit_mapping` currently uses the legacy list format on the wire,
    # but it should eventually be replaced with the new `QubitMapping` (dict-based) format.
    # The switch can only be made once all legacy clients have been updated,
    # as they still depend on receiving the old response format.
    return [{"logical_name": k, "physical_name": v} for k, v in mapping.items()]


LEGACY_QUBIT_MAPPING_SCHEMA = {
    "anyOf": [
        {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["logical_name", "physical_name"],
                "properties": {
                    "logical_name": {"type": "string"},
                    "physical_name": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
        {"type": "null"},
    ]
}


QubitMapping = Annotated[
    dict[str, str],
    BeforeValidator(_parse_legacy_qubit_mapping),
    PlainSerializer(_serialize_as_legacy_qubit_mapping),
    WithJsonSchema(LEGACY_QUBIT_MAPPING_SCHEMA),
]
"""Mapping from logical qubit names to physical qubit names."""


# ATTENTION: Do **not** rename RunRequest model!
# IQM Server implements circuit validation by loading the StationControl OpenAPI specification
# and using this (JSON) schema for the validation. OpenAPI specification contains schemas for
# all station control models so the name of the schema (= name of this dataclass) is used to
# select the correct schema. If you intend to rename or move this model, consult IQM Server
# team to ensure that nothing gets broken!! Sub-schemas (e.g. Circuit) can be renamed freely
# - FastAPI uses local schema references so the renamed references are resolved correctly,
# as long as all the referenced schemas are added to the OpenAPI specification
# (which is done automatically by FastAPI/Pydantic,
# unless there are some really weird stuff in the dataclas definition).
#
# In addition to the schema name, IQM Server depends on the following features:
#
#  * RunRequest has "shots" integer property
#  * RunRequest has "circuits" array property
#
# If you change those properties, coordinate the changes with IQM Server team!
class PostJobsRequest(PydanticBase):
    """Request to Station Control run a job that executes a batch of quantum circuits."""

    circuits: CircuitBatch = Field(...)
    """Batch of quantum circuit(s) to execute."""
    calibration_set_id: UUID | None = Field(None)
    """ID of the calibration set to use, or None to use the current default calibration set."""
    qubit_mapping: QubitMapping | None = Field(None)
    """Mapping from logical qubit names to physical qubit names, or None if ``circuits`` use physical qubit names."""
    shots: int = Field(1, gt=0)
    """How many times to execute each circuit in the batch, must be greater than zero."""
    max_circuit_duration_over_t2: float | None = Field(None)
    """Circuits are disqualified on the server if they are longer than this fraction
       of the T2 time of the worst qubit used.
       If set to 0.0, no circuits are disqualified. If set to None the server default value is used."""
    heralding_mode: HeraldingMode = Field(HeraldingMode.NONE)
    """Which heralding mode to use during the execution of circuits in this request."""
    move_gate_validation: MoveGateValidationMode = Field(
        MoveGateValidationMode.STRICT,
        validation_alias=AliasChoices("move_gate_validation", "move_validation_mode"),
    )
    """Which method of MOVE gate validation to use in circuit compilation."""
    move_gate_frame_tracking: MoveGateFrameTrackingMode = Field(
        MoveGateFrameTrackingMode.FULL,
        validation_alias=AliasChoices("move_gate_frame_tracking", "move_gate_frame_tracking_mode"),
    )
    """Which method of MOVE gate frame tracking to use for circuit compilation."""
    active_reset_cycles: int | None = Field(None)
    """Number of active ``reset`` operations inserted at the beginning of each circuit for each active qubit.
    ``None`` means active reset is not used but instead reset is done by waiting (relaxation). Integer values smaller
    than 1 result in neither active nor reset by wait being used, in which case any reset operations must be explicitly
    added in the circuit."""
    dd_mode: DDMode = Field(DDMode.DISABLED)
    """Whether dynamical decoupling is enabled or disabled during the execution."""
    dd_strategy: DDStrategy | None = Field(None)
    """Dynamical decoupling strategy to be used during the execution, if DD is enabled.
    If None, use the server default strategy."""

    @computed_field(
        json_schema_extra={
            "deprecated": True,
            "description": format_deprecated(
                old="`move_validation_mode`", new="`move_gate_validation`", since="2025-10-17"
            ),
        },
    )
    def move_validation_mode(self) -> MoveGateValidationMode:
        return self.move_gate_validation

    @computed_field(
        json_schema_extra={
            "deprecated": True,
            "description": format_deprecated(
                old="`move_gate_frame_tracking_mode`", new="`move_gate_frame_tracking`", since="2025-10-17"
            ),
        },
    )
    def move_gate_frame_tracking_mode(self) -> MoveGateFrameTrackingMode:
        return self.move_gate_frame_tracking


RunRequest: TypeAlias = PostJobsRequest


class CircuitMeasurementCounts(PydanticBase):
    """Circuit measurement counts in histogram representation."""

    measurement_keys: list[str]
    """Measurement keys in the order they are concatenated to form the state bitstrings in :attr:`counts`.

    For example, if :attr:`measurement_keys` is ``['mk_1', 'mk2']`` and ``'mk_1'`` measures ``QB1``
    and ``'mk_2'`` measures ``QB3`` and ``QB5``, then :attr:`counts` could contains keys such as ``'010'`` representing
    shots where ``QB1`, ``QB3`` and ``QB5`` were observed to be in the state :math:`|010\rangle`.
    """
    counts: dict[str, int]
    """Mapping from computational basis states, represented as bitstrings, to the number of times they were observed
    when executing the circuit."""


CircuitMeasurementCountsBatch: TypeAlias = list[CircuitMeasurementCounts]
"""Measurement results in histogram representation for each circuit in the batch."""
