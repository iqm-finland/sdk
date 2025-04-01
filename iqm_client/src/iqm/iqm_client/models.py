# Copyright 2024 IQM client developers
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
"""This module contains the data models used by IQMClient."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
import re
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, StrictStr, TypeAdapter, field_validator
from pydantic_core.core_schema import ValidationInfo


@dataclass(frozen=True)
class NativeOperation:
    """Describes a native operation on the quantum computer."""

    name: str
    """Name of the operation."""
    arity: int
    """Number of locus components (usually qubits) the operation acts on.
    Zero means the operation can be applied on any number of locus components."""
    args_required: dict[str, tuple[type, ...]] = field(default_factory=dict)
    """Maps names of required operation parameters to their allowed types."""
    args_not_required: dict[str, tuple[type, ...]] = field(default_factory=dict)
    """Maps names of optional operation parameters to their allowed types."""
    symmetric: bool = False
    """True iff the effect of operation is symmetric in the locus components it acts on.
    Only meaningful if :attr:`arity` != 1."""
    renamed_to: str = ""
    """If nonempty, indicates that this operation name is deprecated, and IQM client will
    auto-rename it to the new name."""
    factorizable: bool = False
    """Iff True, any multi-component instance of this operation can be broken down to
    single-component instances, and calibration data is specific to single-component loci."""
    no_calibration_needed: bool = False
    """Iff true, the operation is always allowed on all QPU loci regardless of calibration state.
    Typically a metaoperation like barrier."""


_SUPPORTED_OPERATIONS: dict[str, NativeOperation] = {
    op.name: op
    for op in [
        NativeOperation("barrier", 0, symmetric=True, no_calibration_needed=True),
        NativeOperation("delay", 0, {"duration": (float,)}, symmetric=True, no_calibration_needed=True),
        NativeOperation("measure", 0, {"key": (str,)}, args_not_required={"feedback_key": (str,)}, factorizable=True),
        NativeOperation(
            "prx",
            1,
            {
                "angle_t": (float, int),
                "phase_t": (float, int),
            },
        ),
        NativeOperation(
            "cc_prx",
            1,
            {
                "angle_t": (float, int),
                "phase_t": (float, int),
                "feedback_key": (str,),
                "feedback_qubit": (str,),
            },
        ),
        # TODO reset does need calibration, but it inherits it from cc_prx and does not yet appear in the DQA itself.
        NativeOperation("reset", 0, symmetric=True, factorizable=True, no_calibration_needed=True),
        NativeOperation("cz", 2, symmetric=True),
        NativeOperation("move", 2),
    ]
}

Locus = tuple[StrictStr, ...]
"""Names of the QPU components (typically qubits) a quantum operation instance is acting on, e.g. `("QB1", "QB2")`."""


class Instruction(BaseModel):
    r"""Native quantum operation instance with particular arguments and locus.

    This class represents a native quantum operation
    acting on :attr:`qubits`, with the arguments :attr:`args`.
    The operation is determined by :attr:`name`.

    We currently support the following native operations:

    ================ =========== ======================================= ===========
    name             # of qubits args                                    description
    ================ =========== ======================================= ===========
    measure          >= 1        ``key: str``, ``feedback_key: str``     Measurement in the Z basis.
    prx              1           ``angle_t: float``, ``phase_t: float``  Phased x-rotation gate.
    cc_prx           1           ``angle_t: float``, ``phase_t: float``,
                                 ``feedback_qubit: str``,
                                 ``feedback_key: str``                   Classically controlled PRX gate.
    reset            >= 1                                                Reset the qubit(s) to :math:`|0\rangle`.
    cz               2                                                   Controlled-Z gate.
    move             2                                                   Move a qubit state between a qubit and a
                                                                         computational resonator, as long as
                                                                         at least one of the components is
                                                                         in the :math:`|0\rangle` state.
    barrier          >= 1                                                Execution barrier.
    delay            >= 1        ``duration: float``                     Force a delay between circuit operations.
    ================ =========== ======================================= ===========

    For each Instruction you may also optionally specify :attr:`~Instruction.implementation`,
    which contains the name of an implementation of the operation to use.
    Support for multiple implementations is currently experimental and in normal use the
    field should be omitted, this selects the default implementation for the operation for that locus.

    Measure
    -------

    Measurement in the computational (Z) basis. The measurement results are the output of the circuit.
    Takes two string arguments: ``key``, denoting the measurement key the returned results are labeled with,
    and ``feedback_key``, which is only needed if the measurement result is used for classical control
    within the circuit.
    All the measurement keys and feedback keys used in a circuit must be unique (but the two groups of
    keys are independent namespaces).
    Each qubit may be measured multiple times, i.e. mid-circuit measurements are allowed.

    .. code-block:: python
        :caption: Example

        Instruction(name='measure', qubits=('alice', 'bob', 'charlie'), args={'key': 'm1'})

    PRX
    ---

    Phased x-rotation gate, i.e. an x-rotation conjugated by a z-rotation.
    Takes two arguments, the rotation angle ``angle_t`` and the phase angle ``phase_t``,
    both measured in units of full turns (:math:`2\pi` radians).
    The gate is represented in the standard computational basis by the matrix

    .. math::
        \text{PRX}(\theta, \phi) = \exp(-i (X \cos (2 \pi \; \phi) + Y \sin (2 \pi \; \phi)) \: \pi \; \theta)
        = \text{RZ}(\phi) \: \text{RX}(\theta) \: \text{RZ}^\dagger(\phi),

    where :math:`\theta` = ``angle_t``, :math:`\phi` = ``phase_t``,
    and :math:`X` and :math:`Y` are Pauli matrices.

    .. code-block:: python
        :caption: Example

        Instruction(name='prx', qubits=('bob',), args={'angle_t': 0.7, 'phase_t': 0.25})

    CC_PRX
    ------

    Classically controlled PRX gate. Takes four arguments. ``angle_t`` and ``phase_t`` are exactly as in PRX.
    ``feedback_key`` is a string that identifies the ``measure`` instruction whose result controls
    the gate (the one that shares the feedback key).
    ``feedback_qubit`` is the name of the physical qubit within the ``measure`` instruction that produces the feedback.
    If the measurement result is 1, the PRX gate is applied. If it is 0, an identity gate of similar time
    duration gate is applied instead.
    The measurement instruction must precede the classically controlled gate instruction in the quantum circuit.

    Reset
    -----

    Resets the qubit(s) non-unitarily to the :math:`|0\rangle` state.

    .. code-block:: python
        :caption: Example

        Instruction(name='reset', qubits=('alice', 'bob'), args={})

    .. note:: Currently inherits its calibration from ``cc_prx`` and is only available when ``cc_prx`` is.

    CZ
    --

    Controlled-Z gate. Represented in the standard computational basis by the matrix

    .. math:: \text{CZ} = \text{diag}(1, 1, 1, -1).

    It is symmetric wrt. the qubits it's acting on, and takes no arguments.

    .. code-block:: python
        :caption: Example

        Instruction(name='cz', qubits=('alice', 'bob'), args={})

    MOVE
    ----

    The MOVE operation is a unitary population exchange operation between a qubit and a resonator.
    Its effect is only defined in the invariant subspace :math:`S = \text{span}\{|00\rangle, |01\rangle, |10\rangle\}`,
    where it swaps the populations of the states :math:`|01\rangle` and :math:`|10\rangle`.
    Its effect on the orthogonal subspace is undefined.

    MOVE has the following presentation in the subspace :math:`S`:

    .. math:: \text{MOVE}_S = |00\rangle \langle 00| + a |10\rangle \langle 01| + a^{-1} |01\rangle \langle 10|,

    where :math:`a` is an undefined complex phase that is canceled when the MOVE gate is applied a second time.

    To ensure that the state of the qubit and resonator has no overlap with :math:`|11\rangle`, it is
    recommended that no single qubit gates are applied to the qubit in between a
    pair of MOVE operations.

    .. code-block:: python
        :caption: Example

        Instruction(name='move', qubits=('alice', 'resonator'), args={})

    .. note:: MOVE is only available in quantum computers with the IQM Star architecture.

    Barrier
    -------

    Affects the physical execution order of the instructions elsewhere in the
    circuit that act on qubits spanned by the barrier.
    It ensures that any such instructions that succeed the barrier are only executed after
    all such instructions that precede the barrier have been completed.
    Hence it can be used to guarantee a specific causal order for the other instructions.
    It takes no arguments, and has no other effect.

    .. code-block:: python
        :caption: Example

        Instruction(name='barrier', qubits=('alice', 'bob'), args={})

    .. note::

       One-qubit barriers will not have any effect on circuit's compilation and execution. Higher layers
       that sit on top of IQM Client can make actual use of one-qubit barriers (e.g. during circuit optimization),
       therefore having them is allowed.

    Delay
    -----

    Forces a delay between the preceding and following circuit operations.
    It can be applied to any number of qubits. Takes one argument, ``duration``, which is the minimum
    duration of the delay in seconds. It will be rounded up to the nearest possible duration the
    hardware can handle.

    .. code-block:: python
        :caption: Example

        Instruction(name='delay', qubits=('alice', 'bob'), args={'duration': 80e-9})


    .. note::

       We can only guarantee that the delay is *at least* of the requested duration, due to both
       hardware and practical constraints, but could be much more depending on the other operations
       in the circuit. To see why, consider e.g. the circuit

       .. code-block:: python

          [
              Instruction(name='cz', qubits=('alice', 'bob'), args={}),
              Instruction(name='delay', qubits=('alice',), args={'duration': 1e-9}),
              Instruction(name='delay', qubits=('bob',), args={'duration': 100e-9}),
              Instruction(name='cz', qubits=('alice', 'bob'), args={}),
          ]

       In this case the actual delay between the two CZ gates will be 100 ns rounded up to
       hardware granularity, even though only 1 ns was requested for `alice`.
    """

    name: str = Field(examples=["measure"])
    """name of the quantum operation"""
    implementation: StrictStr | None = Field(None)
    """name of the implementation, for experimental use only"""
    qubits: Locus = Field(examples=[("alice",)])
    """names of the locus components (typically qubits) the operation acts on"""
    args: dict[str, Any] = Field(default_factory=dict, examples=[{"key": "m"}])
    """arguments for the operation"""

    def __init__(self, **data):
        super().__init__(**data)
        # Auto-convert name if a deprecated name is used
        self.name = _op_current_name(self.name)

    @field_validator("name")
    @classmethod
    def name_validator(cls, value):
        """Check if the name of instruction is set to one of the supported quantum operations."""
        name = value
        if name not in _SUPPORTED_OPERATIONS:
            message = ", ".join(_SUPPORTED_OPERATIONS)
            raise ValueError(f'Unknown operation "{name}". Supported operations are "{message}"')
        return name

    @field_validator("implementation")
    @classmethod
    def implementation_validator(cls, value):
        """Check if the implementation of the instruction is set to a non-empty string."""
        implementation = value
        if isinstance(implementation, str):
            if not implementation:
                raise ValueError("Implementation of the instruction should be None, or a non-empty string")
        return implementation

    @field_validator("qubits")
    @classmethod
    def qubits_validator(cls, value, info: ValidationInfo):
        """Check if the instruction has the correct number of qubits for its operation."""
        qubits = value
        name = info.data.get("name")
        if not name:
            raise ValueError("Could not validate qubits because the name of the instruction did not pass validation")
        arity = _SUPPORTED_OPERATIONS[name].arity
        if (0 < arity) and (arity != len(qubits)):
            raise ValueError(f'The "{name}" operation acts on {arity} qubit(s), but {len(qubits)} were given: {qubits}')
        return qubits

    @field_validator("args")
    @classmethod
    def args_validator(cls, value, info: ValidationInfo):
        """Check argument names and types for a given instruction"""
        args = value
        name = info.data.get("name")
        if not name:
            raise ValueError("Could not validate args because the name of the instruction did not pass validation")

        # Check argument names
        submitted_arg_names = set(args)
        required_arg_names = set(_SUPPORTED_OPERATIONS[name].args_required)
        allowed_arg_types = _SUPPORTED_OPERATIONS[name].args_required | _SUPPORTED_OPERATIONS[name].args_not_required
        allowed_arg_names = set(allowed_arg_types)
        if not required_arg_names <= submitted_arg_names:
            raise ValueError(
                f'The operation "{name}" requires '
                f"{tuple(required_arg_names)} argument(s), "
                f"but {tuple(submitted_arg_names)} were given"
            )
        if not submitted_arg_names <= allowed_arg_names:
            message = tuple(allowed_arg_names) if allowed_arg_names else "no"
            raise ValueError(
                f'The operation "{name}" allows {message} argument(s), but {tuple(submitted_arg_names)} were given'
            )

        # Check argument types
        for arg_name, arg_value in args.items():
            allowed_types = allowed_arg_types[arg_name]
            if not isinstance(arg_value, allowed_types):
                raise TypeError(
                    f'The argument "{arg_name}" should be of one of the following supported types'
                    f" {allowed_types}, but ({type(arg_value)}) was given"
                )

        return value


def _op_is_symmetric(name: str) -> bool:
    """Returns True iff the given native operation is symmetric, i.e. the order of the
    locus components does not matter.

    Args:
        name: name of the operation
    Returns:
        True iff the locus order does not matter
    Raises:
        KeyError: ``name`` is unknown

    """
    return _SUPPORTED_OPERATIONS[name].symmetric


def _op_arity(name: str) -> int:
    """Returns the arity of the given native operation, i.e. the number of locus components it acts on.

    Zero means any number of locus components is OK.

    Args:
        name: name of the operation
    Returns:
        arity of the operation
    Raises:
        KeyError: ``name`` is unknown

    """
    return _SUPPORTED_OPERATIONS[name].arity


def _op_current_name(name: str) -> str:
    """Checks if the operation name has been deprecated and returns the new name if it is;
    otherwise, just returns the name as-is.

    Args:
        name: name of the operation

    Returns:
        current name of the operation
    Raises:
        KeyError: ``name`` is unknown

    """
    return _SUPPORTED_OPERATIONS[name].renamed_to or name


class Circuit(BaseModel):
    """Quantum circuit to be executed.

    Consists of native quantum operations, each represented by an instance of the :class:`Instruction` class.
    """

    name: str = Field(..., examples=["test circuit"])
    """name of the circuit"""
    instructions: tuple[Instruction, ...] = Field(...)
    """instructions comprising the circuit"""
    metadata: dict[str, Any] | None = Field(None)
    """arbitrary metadata associated with the circuit"""

    def all_qubits(self) -> set[str]:
        """Return the names of all qubits in the circuit."""
        qubits: set[str] = set()
        for instruction in self.instructions:
            qubits.update(instruction.qubits)
        return qubits

    @field_validator("name")
    @classmethod
    def name_validator(cls, value):
        """Check if the circuit name is a non-empty string"""
        name = value
        if len(name) == 0:
            raise ValueError("A circuit should have a non-empty string for a name.")
        return name

    @field_validator("instructions")
    @classmethod
    def instructions_validator(cls, value):
        """Check the container of instructions and each instruction within"""
        instructions = value

        # Check container type
        if not isinstance(instructions, (list, tuple)):
            raise ValueError("Instructions of a circuit should be packed in a tuple")

        # Check if any instructions are present
        if len(value) == 0:
            raise ValueError("Each circuit should have at least one instruction.")

        # Check each instruction explicitly, because automatic validation for Instruction
        # is only called when we create a new instance of Instruction, but not if we modify
        # an existing instance.
        for instruction in instructions:
            if isinstance(instruction, Instruction):
                Instruction.model_validate(instruction.__dict__)
            else:
                raise ValueError("Every instruction in a circuit should be of type <Instruction>")

        return instructions


CircuitBatch = list[Circuit]
"""Type that represents a list of quantum circuits to be executed together in a single batch."""


def validate_circuit(circuit: Circuit) -> None:
    """Validates a submitted quantum circuit using Pydantic tooling.

    Args:
        circuit: a circuit that needs validation

    Raises:
        pydantic.error_wrappers.ValidationError: validation failed

    """
    Circuit.model_validate(circuit.__dict__)


class SingleQubitMapping(BaseModel):
    """Mapping of a logical qubit name to a physical qubit name."""

    logical_name: str = Field(..., examples=["alice"])
    """logical qubit name"""
    physical_name: str = Field(..., examples=["QB1"])
    """physical qubit name"""


QubitMapping = list[SingleQubitMapping]
"""Type that represents a qubit mapping for a circuit, i.e. a list of single qubit mappings
for all qubits in the circuit."""


def serialize_qubit_mapping(qubit_mapping: dict[str, str]) -> list[SingleQubitMapping]:
    """Serializes a qubit mapping dict into the corresponding IQM data transfer format.

    Args:
        qubit_mapping: mapping from logical to physical qubit names

    Returns:
        data transfer object representing the mapping

    """
    return [SingleQubitMapping(logical_name=k, physical_name=v) for k, v in qubit_mapping.items()]


class QuantumArchitectureSpecification(BaseModel):
    """Quantum architecture specification."""

    name: str = Field(...)
    """Name of the quantum architecture."""
    operations: dict[str, list[list[str]]] = Field(...)
    """Operations supported by this quantum architecture, mapped to the allowed loci."""
    qubits: list[str] = Field(...)
    """List of qubits of this quantum architecture."""
    qubit_connectivity: list[list[str]] = Field(...)
    """Qubit connectivity of this quantum architecture."""

    def __init__(self, **data):
        operations = data.get("operations")
        if isinstance(operations, list):
            # backwards compatibility for the old quantum architecture format
            qubits = data.get("qubits")
            qubit_connectivity = data.get("qubit_connectivity")
            # add all possible loci for the ops
            data["operations"] = {
                _op_current_name(op): (
                    qubit_connectivity if _op_arity(_op_current_name(op)) == 2 else [[qb] for qb in qubits]
                )
                for op in operations
            }

        super().__init__(**data)
        self.operations = {_op_current_name(k): v for k, v in self.operations.items()}

    def has_equivalent_operations(self, other: QuantumArchitectureSpecification) -> bool:
        """Compares the given operation sets defined by the quantum architecture against
        another architecture specification.

        Returns:
            True if the operation and the loci are equivalent.

        """
        return QuantumArchitectureSpecification.compare_operations(self.operations, other.operations)

    @staticmethod
    def compare_operations(ops1: dict[str, list[list[str]]], ops2: dict[str, list[list[str]]]) -> bool:
        """Compares the given operation sets.

        Returns:
            True if the operation and the loci are equivalent.

        """
        if set(ops1) != set(ops2):
            return False
        for op, loci1 in ops1.items():
            loci2 = ops2[op]
            if _op_is_symmetric(op):
                # for comparing symmetric instruction loci, sorting order does not matter as long as it's consistent
                l1 = [tuple(sorted(locus)) for locus in loci1]
                l2 = [tuple(sorted(locus)) for locus in loci2]
            else:
                l1 = [tuple(locus) for locus in loci1]
                l2 = [tuple(locus) for locus in loci2]

            if set(l1) != set(l2):
                return False
        return True


class QuantumArchitecture(BaseModel):
    """Quantum architecture as returned by server."""

    quantum_architecture: QuantumArchitectureSpecification = Field(...)
    """Details about the quantum architecture."""


def _component_sort_key(component_name: str) -> tuple[str, int, str]:
    """Sorting key for QPU component names."""

    def get_numeric_id(name: str) -> int:
        match = re.search(r"(\d+)", name)
        return int(match.group(1)) if match else 0

    return re.sub(r"[^a-zA-Z]", "", component_name), get_numeric_id(component_name), component_name


class StaticQuantumArchitecture(BaseModel):
    """Static quantum architecture of the server.

    The static quantum architecture (SQA) provides information about the QPU,
    including the names of its computational components and the connections between them.
    """

    qubits: list[str] = Field(...)
    """Names of the physical qubits on the QPU, sorted."""
    computational_resonators: list[str] = Field(...)
    """Names of the physical computational resonators on the QPU, sorted."""
    connectivity: list[Locus] = Field(...)
    """Groups of components (qubits and computational resonators) that are connected by a coupler on the QPU, sorted."""


class QualityMetricSet(BaseModel):
    """Quality metrics for a calibration set."""

    calibration_set_id: UUID | None = Field(...)
    """ID of the calibration set."""
    calibration_set_dut_label: str = Field(...)
    """Chip Label of the calibration set."""
    calibration_set_number_of_observations: int = Field(...)
    """Number of observations in the calibration set."""
    calibration_set_created_timestamp: str = Field(...)
    """Timestamp when the calibration set was created."""
    calibration_set_end_timestamp: str = Field(...)
    """Timestamp when the calibration set was finalized."""
    calibration_set_is_invalid: bool = Field(...)
    """Whether the calibration set is invalid."""
    quality_metric_set_id: UUID | None = Field(...)
    """ID of the quality metric set."""
    quality_metric_set_dut_label: str | None = Field(...)
    """Chip label of the quality metric set."""
    quality_metric_set_created_timestamp: str | None = Field(...)
    """Timestamp when the quality metric set was created."""
    quality_metric_set_end_timestamp: str | None = Field(...)
    """Timestamp when the quality metric set was finalized."""
    quality_metric_set_is_invalid: bool = Field(...)
    """Whether the quality metric set is invalid."""
    metrics: dict[str, dict[str, Any]] | None = Field(...)
    """Quality metrics."""


class CalibrationSet(BaseModel):
    """Metadata and observations of a calibration set."""

    calibration_set_id: UUID = Field(...)
    """ID of the calibration set."""
    calibration_set_dut_label: str = Field(...)
    """Chip Label of the calibration set."""
    calibration_set_is_invalid: bool = Field(...)
    """Whether the calibration set is invalid."""
    calibration_set_created_timestamp: str = Field(...)
    """Timestamp when the calibration set was created."""
    calibration_set_end_timestamp: str = Field(...)
    """Timestamp when the calibration set was finalized."""
    observations: dict[str, Any] = Field(...)
    """Calibration data."""


class GateImplementationInfo(BaseModel):
    """Information about an implementation of a quantum gate/operation."""

    loci: tuple[Locus, ...] = Field(...)
    """loci for which this gate implementation has been calibrated"""


class GateInfo(BaseModel):
    """Information about a quantum gate/operation."""

    implementations: dict[str, GateImplementationInfo] = Field(...)
    """mapping of available implementation names to information about the implementations"""
    default_implementation: str = Field(...)
    """default implementation for the gate, used unless overridden by :attr:`override_default_implementation`
    or unless the user requests a specific implementation for a particular gate in the circuit using
    :attr:`.Instruction.implementation`"""
    override_default_implementation: dict[Locus, str] = Field(...)
    """mapping of loci to implementation names that override ``default_implementation`` for those loci"""

    @field_validator("override_default_implementation", mode="before")
    @classmethod
    def override_default_implementation_validator(cls, value: Any) -> dict[Locus, str]:
        """Converts locus keys to tuples if they are encoded as strings."""
        new_value = {}
        if isinstance(value, dict):
            for k, v in value.items():
                if isinstance(k, tuple):
                    new_value[k] = v
                # When Pydantic serializes a dict with tuple keys into JSON, the keys are turned into
                # comma-separated strings (because JSON only supports string keys). Here we convert
                # them back into tuples.
                elif isinstance(k, str):
                    new_k = tuple(k.split(","))
                    new_value[new_k] = v
                else:
                    raise ValueError("'override_default_implementation' keys must be strings or tuples.")
            return new_value
        raise ValueError("'override_default_implementation' must be a dict.")

    @cached_property
    def loci(self) -> tuple[Locus, ...]:
        """Returns all loci which are available for at least one of the implementations.

        The loci are sorted first based on the first locus component, then the second, etc.
        The sorting of individual locus components is first done alphabetically based on their
        non-numeric part, and then components with the same non-numeric part are sorted numerically.
        An example of loci sorted this way would be:

            ('QB1', 'QB2'),
            ('QB2', 'COMPR1'),
            ('QB2', 'QB3'),
            ('QB3', 'COMPR1'),
            ('QB3', 'COMPR2'),
            ('QB3', 'QB1'),
            ('QB10', 'QB2'),

        """
        loci_set = {locus for impl in self.implementations.values() for locus in impl.loci}
        loci_sorted = sorted(loci_set, key=lambda locus: tuple(map(_component_sort_key, locus)))
        return tuple(loci_sorted)


class DynamicQuantumArchitecture(BaseModel):
    """Dynamic quantum architecture as returned by server.

    The dynamic quantum architecture (DQA) describes gates/operations for which calibration data
    exists in the calibration set.
    """

    calibration_set_id: UUID = Field(...)
    """id of the calibration set from which this DQA was generated"""
    qubits: list[str] = Field(...)
    """qubits that appear in at least one gate locus in the calibration set"""
    computational_resonators: list[str] = Field(...)
    """computational resonators that appear in at least one gate locus in the calibration set"""
    gates: dict[str, GateInfo] = Field(...)
    """mapping of gate names to information about the gates"""

    @cached_property
    def components(self) -> tuple[str, ...]:
        """All locus components (qubits and computational resonators) sorted.

        The components are first sorted alphabetically based on their non-numeric part, and then
        components with the same non-numeric part are sorted numerically. An example of components
        sorted this way would be: ('COMPR1', 'COMPR2', 'QB1', 'QB2', 'QB3', 'QB10', 'QB11', 'QB20').
        """
        return tuple(sorted(self.qubits + self.computational_resonators, key=_component_sort_key))


class HeraldingMode(str, Enum):
    """Heralding mode for circuit execution.

    Heralding is the practice of generating data about the state of qubits prior to execution of a circuit.
    This can be achieved by measuring the qubits immediately before executing each shot for a circuit.
    """

    NONE = "none"
    """Do not do any heralding."""
    ZEROS = "zeros"
    """Perform a heralding measurement after qubit initialization, only retain shots with an all-zeros result.

    Note: in this mode, the number of shots returned after execution will be less or equal to the requested amount
    due to the post-selection based on heralding data."""


class MoveGateValidationMode(str, Enum):
    """MOVE gate validation mode for circuit compilation. This options is meant for advanced users."""

    STRICT = "strict"
    """Perform standard MOVE gate validation: MOVE gates must only appear in sandwiches, with no gates acting on the
    MOVE qubit inside the sandwich."""
    ALLOW_PRX = "allow_prx"
    """Allow PRX gates on the MOVE qubit inside MOVE sandwiches during validation."""
    NONE = "none"
    """Do not perform any MOVE gate validation."""


class MoveGateFrameTrackingMode(str, Enum):
    """MOVE gate frame tracking mode for circuit compilation. This option is meant for advanced users."""

    FULL = "full"
    """Perform complete MOVE gate frame tracking."""
    NO_DETUNING_CORRECTION = "no_detuning_correction"
    """Do not add the phase detuning corrections to the pulse schedule for the MOVE gate. The user is expected to do
    these manually."""
    NONE = "none"
    """Do not perform any MOVE gate frame tracking. The user is expected to do these manually."""


class DDMode(str, Enum):
    """Dynamical Decoupling (DD) mode for circuit execution."""

    DISABLED = "disabled"
    """Do not apply dynamical decoupling."""
    ENABLED = "enabled"
    """Apply dynamical decoupling."""


PRXSequence = list[tuple[float, float]]
"""A sequence of PRX gates. A generic PRX gate is defined by rotation angle and phase angle, Theta and Phi,
respectively."""


class DDStrategy(BaseModel):
    """Describes a particular dynamical decoupling strategy.

    The current standard DD stategy can be found in :attr:`.STANDARD_DD_STRATEGY`,
    but users can use this class to provide their own dynamical decoupling strategies.

    See Ezzell et al., Phys. Rev. Appl. 20, 064027 (2022) for information on DD sequences.
    """

    merge_contiguous_waits: bool = Field(default=True)
    """Merge contiguous ``Wait`` instructions into one if they are separated only by ``Block`` instructions."""

    target_qubits: frozenset[str] | None = Field(default=None)
    """Qubits on which dynamical decoupling should be applied. If ``None``, all qubits are targeted."""

    skip_leading_wait: bool = Field(default=True)
    """Skip processing leading ``Wait`` instructions."""

    skip_trailing_wait: bool = Field(default=True)
    """Skip processing trailing ``Wait`` instructions."""

    gate_sequences: list[tuple[int, str | PRXSequence, str]] = Field(default_factory=list)
    """Available decoupling gate sequences to chose from in this strategy.

    Each sequence is defined by a tuple of ``(ratio, gate pattern, align)``:

        * ratio: Minimal duration for the sequence (in PRX gate durations) as integer.

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


STANDARD_DD_STRATEGY = DDStrategy(gate_sequences=[(9, "XYXYYXYX", "asap"), (5, "YXYX", "asap"), (2, "XX", "center")])
"""The default DD strategy uses the following gate sequences:

* Simple symmetric CPMG sequence for short idling times.
* Asymmetric (left-aligned) universal XY4 sequence for medium idling times.
* Asymmetric (left-aligned) universal EDD sequence for longer idling times.
"""


@dataclass(frozen=True)
class CircuitCompilationOptions:
    """Various discrete options for quantum circuit compilation to pulse schedule."""

    max_circuit_duration_over_t2: float | None = None
    """Server-side circuit disqualification threshold.
    The job is rejected on the server if any circuit in it is estimated to take longer than
    the shortest T2 time of any qubit used in the circuit, multiplied by this value.
    Setting this value to ``0.0`` turns off circuit duration checking.
    ``None`` tells the server to use its default value in the check."""
    heralding_mode: HeraldingMode = HeraldingMode.NONE
    """Heralding mode to use during the execution."""
    move_gate_validation: MoveGateValidationMode = MoveGateValidationMode.STRICT
    """MOVE gate validation mode for circuit compilation. This options is ignored on devices that do not support MOVE
        and for circuits that do not contain MOVE gates."""
    move_gate_frame_tracking: MoveGateFrameTrackingMode = MoveGateFrameTrackingMode.FULL
    """MOVE gate frame tracking mode for circuit compilation. This options is ignored on devices that do not support
        MOVE and for circuits that do not contain MOVE gates."""
    active_reset_cycles: int | None = None
    """Number of active ``reset`` operations inserted at the beginning of each circuit for each active qubit.
    ``None`` means active reset is not used but instead reset is done by waiting (relaxation). Integer values smaller
    than 1 result in neither active nor reset by wait being used, in which case any reset operations must be explicitly
    added in the circuit."""
    dd_mode: DDMode = DDMode.DISABLED
    """Control whether dynamical decoupling should be enabled or disabled during the execution."""
    dd_strategy: DDStrategy | None = None
    """A particular dynamical decoupling strategy to be used during the execution."""

    def __post_init__(self):
        """Validate the options."""
        if self.move_gate_frame_tracking == MoveGateFrameTrackingMode.FULL and self.move_gate_validation not in [
            MoveGateValidationMode.STRICT,
            MoveGateValidationMode.ALLOW_PRX,
            None,
        ]:
            raise ValueError(
                "Unable to perform full MOVE gate frame tracking if MOVE gate validation is not"
                + ' "strict" or "allow_prx".'
            )


class RunRequest(BaseModel):
    """Request for an IQM quantum computer to run a job that executes a batch of quantum circuits.

    Note: all circuits in a batch must measure the same qubits otherwise batch execution fails.
    """

    circuits: CircuitBatch = Field(...)
    """batch of quantum circuit(s) to execute"""
    custom_settings: dict[str, Any] | None = Field(None)
    """Custom settings to override default IQM hardware settings and calibration data.
    Note: This field should be always None in normal use."""
    calibration_set_id: UUID | None = Field(None)
    """ID of the calibration set to use, or None to use the latest calibration set"""
    qubit_mapping: list[SingleQubitMapping] | None = Field(None)
    """mapping of logical qubit names to physical qubit names, or None if using physical qubit names"""
    shots: int = Field(..., gt=0)
    """how many times to execute each circuit in the batch, must be greater than zero"""
    max_circuit_duration_over_t2: float | None = Field(None)
    """Circuits are disqualified on the server if they are longer than this ratio
        of the T2 time of the qubits.
        If set to 0.0, no circuits are disqualified. If set to None the server default value is used."""
    heralding_mode: HeraldingMode = Field(HeraldingMode.NONE)
    """which heralding mode to use during the execution of circuits in this request."""

    move_validation_mode: MoveGateValidationMode = Field(MoveGateValidationMode.STRICT)
    """Which method of MOVE gate validation to use for circuit compilation."""
    move_gate_frame_tracking_mode: MoveGateFrameTrackingMode = Field(MoveGateFrameTrackingMode.FULL)
    """Which method of MOVE gate frame tracking to use for circuit compilation."""
    active_reset_cycles: int | None = Field(None)
    """Number of active ``reset`` operations inserted at the beginning of each circuit for each active qubit.
    ``None`` means active reset is not used but instead reset is done by waiting (relaxation). Integer values smaller
    than 1 result in neither active nor reset by wait being used, in which case any reset operations must be explicitly
    added in the circuit."""
    dd_mode: DDMode = Field(DDMode.DISABLED)
    """Control whether dynamical decoupling should be enabled or disabled during the execution."""
    dd_strategy: DDStrategy | None = Field(None)
    """A particular dynamical decoupling strategy to be used during the execution."""


CircuitMeasurementResults = dict[str, list[list[int]]]
"""Measurement results from a single circuit. For each measurement operation in the circuit,
maps the measurement key to the corresponding results. The outer list elements correspond to shots,
and the inner list elements to the qubits measured in the measurement operation."""

CircuitMeasurementResultsBatch = list[CircuitMeasurementResults]
"""Type that represents measurement results for a batch of circuits."""


class JobParameters(BaseModel):
    """Job-specific parameters extracted from the original RunRequest."""

    shots: int = Field(...)
    max_circuit_duration_over_t2: float | None = Field(None)
    heralding_mode: HeraldingMode = Field(HeraldingMode.NONE)
    move_validation_mode: MoveGateValidationMode = Field(MoveGateValidationMode.STRICT)
    move_gate_frame_tracking_mode: MoveGateFrameTrackingMode = Field(MoveGateFrameTrackingMode.FULL)
    dd_mode: DDMode = Field(DDMode.DISABLED)
    dd_strategy: DDStrategy | None = Field(None)


class Metadata(BaseModel):
    """Metadata describing a circuit execution job."""

    calibration_set_id: UUID | None = Field(None)
    """ID of the calibration set used"""
    request: RunRequest | None = Field(None)
    """optional copy of the original RunRequest sent to the server"""
    parameters: JobParameters | None = Field(None)
    """job-specific parameters extracted from the original request"""
    circuits_batch: CircuitBatch | None = Field(None)
    """circuits batch submitted for execution"""
    cocos_version: str | None = Field(None)
    """CoCoS version used to execute the job"""
    timestamps: dict[str, str] | None = Field(None)
    """Timestamps of execution progress"""

    @property
    def shots(self) -> int:
        """Return the number of shots in the job."""
        if self.parameters is not None:
            return self.parameters.shots
        if self.request is not None:
            return self.request.shots
        raise ValueError("No shots information available in the metadata")

    @property
    def circuits(self) -> CircuitBatch:
        """Return the circuits in the job."""
        if self.circuits_batch is not None:
            return self.circuits_batch
        if self.request is not None:
            return self.request.circuits
        raise ValueError("No circuits information available in the metadata")

    @property
    def heralding_mode(self) -> HeraldingMode:
        """Return the heralding mode requested with the job."""
        if self.parameters is not None:
            return self.parameters.heralding_mode
        if self.request is not None:
            return self.request.heralding_mode
        raise ValueError("No heralding mode information available in the metadata")

    @property
    def dd_mode(self) -> DDMode:
        """Return the dynamical decoupling mode requested with the job."""
        if self.parameters is not None:
            return self.parameters.dd_mode
        if self.request is not None:
            return self.request.dd_mode
        raise ValueError("No dynamical decoupling mode information available in the metadata")

    @property
    def dd_strategy(self) -> DDStrategy | None:
        """Return the dynamical decoupling strategy used with the job."""
        if self.parameters is not None:
            return self.parameters.dd_strategy
        if self.request is not None:
            return self.request.dd_strategy
        raise ValueError("No dynamical decoupling strategy information available in the metadata")


class Status(str, Enum):
    """Status of a job."""

    RECEIVED = "received"
    """Job has been received but nothing has been done about it."""
    PROCESSING = "processing"
    ACCEPTED = "accepted"
    """Job has passed initial checks and will proceed to compilation."""

    PENDING_COMPILATION = "pending compilation"
    """Job has been queued for compilation."""
    PENDING_EXECUTION = "pending execution"
    """Job has been compiled and is queued for execution."""
    COMPILED = "compiled"
    """Job has been compiled to a low-level representation."""
    READY = "ready"
    """Job has been executed and results are available."""
    FAILED = "failed"
    """Execution or compilation failed."""
    ABORTED = "aborted"
    """User caneceled the execution."""
    PENDING_DELETION = "pending deletion"
    """Job is set to be deleted."""
    DELETION_FAILED = "deletion failed"
    """Job was supposed to be deleted but deletion failed."""
    DELETED = "deleted"
    """Job deleted from the database."""
    UNKNOWN = "unknown"
    """Job is in a state not recognized by this version of the client."""

    @classmethod
    def terminal_statuses(cls) -> set[Status]:
        """Statuses from which the execution can't continue."""
        return {cls.READY, cls.FAILED, cls.ABORTED, cls.DELETED, cls.DELETION_FAILED}


class RunResult(BaseModel):
    """Results of the quantum circuit execution job.
    If the job succeeded, :attr:`measurements` contains the output of the batch of circuits,
    consisting of the results of the measurement operations in each circuit.
    It is a list of dictionaries, where each dict maps each measurement key to a 2D array of measurement
    results, represented as a nested list.
    ``RunResult.measurements[circuit_index][key][shot][qubit_index]`` is the result of measuring the
    ``qubit_index``'th qubit in measurement operation ``key`` in the shot ``shot`` in the
    ``circuit_index``'th circuit of the batch.
    :attr:`measurements` is present iff the status is ``'ready'``.
    :attr:`message` carries additional information for the ``'failed'`` status.
    If the status is ``'pending compilation'`` or ``'pending execution'``,
    :attr:`measurements` and :attr:`message` are ``None``.

    The results are non-negative integers representing the computational basis state (for qubits, 0 or 1)
    that was the measurement outcome.

    ----
    """

    status: Status = Field(...)
    """current status of the job, in ``{'pending compilation', 'pending execution', 'ready', 'failed', 'aborted'}``"""
    measurements: CircuitMeasurementResultsBatch | None = Field(None)
    """if the job has finished successfully, the measurement results for the circuit(s)"""
    message: str | None = Field(None)
    """if the job failed, an error message"""
    metadata: Metadata = Field(...)
    """metadata about the job"""
    warnings: list[str] | None = Field(None)
    """list of warning messages"""

    @staticmethod
    def from_dict(inp: dict[str, str | dict | list | None]) -> RunResult:
        """Parses the result from a dict.

        Args:
            inp: value to parse, has to map to RunResult

        Returns:
            parsed job result

        """
        input_copy = inp.copy()
        try:
            status = Status(input_copy.pop("status"))
        except ValueError:
            status = Status.UNKNOWN
        return RunResult(status=status, **input_copy)


class RunStatus(BaseModel):
    """Status of a circuit execution job."""

    status: Status = Field(...)
    """current status of the job, in ``{'pending compilation', 'pending execution', 'ready', 'failed', 'aborted'}``"""
    message: str | None = Field(None)
    """if the job failed, an error message"""
    warnings: list[str] | None = Field(None)
    """list of warning messages"""


class Counts(BaseModel):
    """Circuit measurement results in histogram representation."""

    measurement_keys: list[str]
    """Measurement keys in the order they are concatenated to form the state bitstrings in :attr:`counts`.

    For example, if :attr:`measurement_keys` is ``['mk_1', 'mk2']`` and ``'mk_1'`` measures ``QB1``
    and ``'mk_2'`` measures ``QB3`` and ``QB5``, then :attr:`counts` could contains keys such as ``'010'`` representing
    shots where ``QB1`, ``QB3`` and ``QB5`` were observed to be in the state :math:`|010\rangle`.
    """
    counts: dict[str, int]
    """mapping from computational basis states, represented as bitstrings, to the number of times they were observed
    when executing the circuit"""


class RunCounts(BaseModel):
    """Measurement results of a circuit execution job in histogram representation."""

    status: Status = Field(...)
    """current status of the job, in ``{'pending compilation', 'pending execution', 'ready', 'failed', 'aborted'}``"""
    counts_batch: list[Counts] | None = Field(None)
    """measurement results in histogram representation for each circuit in the batch"""


class ClientLibrary(BaseModel):
    """Represents a client library with its metadata.

    Args:
        name: display name of the client library.
        package_name: name of the package as published in package repositories.
        repo_url: URL to the source code repository.
        package_url: URL to the package in the package repository.
        min: minimum supported version.
        max: maximum supported version.

    """

    name: str
    package_name: str | None = Field(None)
    repo_url: str | None = Field(None)
    package_url: str | None = Field(None)
    min: str
    max: str


ClientLibraryDict = TypeAdapter(dict[str, ClientLibrary])
DictDict = TypeAdapter(dict[str, dict])
