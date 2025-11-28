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
r"""Data models used by IQMClient.

We currently support the following native operations for circuit execution,
represented by :class:`iqm.pulse.CircuitOperation`:

================ =========== ======================================= ===========
name             # of qubits args                                    description
================ =========== ======================================= ===========
measure          >= 1        ``key: str``, ``feedback_key: str``     Measurement in the Z basis.
prx              1           ``angle: float``, ``phase: float``      Phased x-rotation gate.
cc_prx           1           ``angle: float``, ``phase: float``,
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

Measure
-------

:mod:`iqm.pulse.gates.measure`

Measurement in the computational (Z) basis. The measurement results are the output of the circuit.
Takes two string arguments: ``key``, denoting the measurement key the returned results are labeled with,
and ``feedback_key``, which is only needed if the measurement result is used for classical control
within the circuit.
All the measurement keys and feedback keys used in a circuit must be unique (but the two groups of
keys are independent namespaces).
Each qubit may be measured multiple times, i.e. mid-circuit measurements are allowed.

.. code-block:: python
    :caption: Example

    CircuitOperation(name='measure', locus=('alice', 'bob', 'charlie'), args={'key': 'm1'})

PRX
---

:mod:`iqm.pulse.gates.prx`

Phased x-rotation gate, i.e. an x-rotation conjugated by a z-rotation.
Takes two arguments, the rotation angle ``angle`` and the phase angle ``phase``,
both measured in units of radians.
The gate is represented in the standard computational basis by the matrix

.. math::
    \text{PRX}(\theta, \phi) = \exp(-i (X \cos (\phi) + Y \sin (\phi)) \: \theta/2)
    = \text{RZ}(\phi) \: \text{RX}(\theta) \: \text{RZ}^\dagger(\phi),

where :math:`\theta` = ``angle``, :math:`\phi` = ``phase``,
and :math:`X` and :math:`Y` are Pauli matrices.

.. code-block:: python
    :caption: Example

    CircuitOperation(name='prx', locus=('bob',), args={'angle': 1.4 * pi, 'phase': 0.5 * pi})

CC_PRX
------

:mod:`iqm.pulse.gates.conditional`

Classically controlled PRX gate. Takes four arguments. ``angle`` and ``phase`` are exactly as in PRX.
``feedback_key`` is a string that identifies the ``measure`` instruction whose result controls
the gate (the one that shares the feedback key).
``feedback_qubit`` is the name of the physical qubit within the ``measure`` instruction that produces the feedback.
If the measurement result is 1, the PRX gate is applied. If it is 0, an identity gate of similar time
duration gate is applied instead.
The measurement instruction must precede the classically controlled gate instruction in the quantum circuit.

Reset
-----

:mod:`iqm.pulse.gates.reset`

Resets the qubit(s) non-unitarily to the :math:`|0\rangle` state.

.. code-block:: python
    :caption: Example

    CircuitOperation(name='reset', locus=('alice', 'bob'), args={})

.. note:: Currently inherits its calibration from ``cc_prx`` and is only available when ``cc_prx`` is.

CZ
--

:mod:`iqm.pulse.gates.cz`

Controlled-Z gate. Represented in the standard computational basis by the matrix

.. math:: \text{CZ} = \text{diag}(1, 1, 1, -1).

It is symmetric wrt. the qubits it's acting on, and takes no arguments.

.. code-block:: python
    :caption: Example

    CircuitOperation(name='cz', locus=('alice', 'bob'), args={})

MOVE
----

:mod:`iqm.pulse.gates.move`

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

    CircuitOperation(name='move', locus=('alice', 'resonator'), args={})

.. note:: MOVE is only available in quantum computers with the IQM Star architecture.

Barrier
-------

:mod:`iqm.pulse.gates.barrier`

Affects the physical execution order of the instructions elsewhere in the
circuit that act on qubits spanned by the barrier.
It ensures that any such instructions that succeed the barrier are only executed after
all such instructions that precede the barrier have been completed.
Hence it can be used to guarantee a specific causal order for the other instructions.
It takes no arguments, and has no other effect.

.. code-block:: python
    :caption: Example

    CircuitOperation(name='barrier', locus=('alice', 'bob'), args={})

.. note::

   One-qubit barriers will not have any effect on circuit's compilation and execution. Higher layers
   that sit on top of IQM Client can make actual use of one-qubit barriers (e.g. during circuit optimization),
   therefore having them is allowed.

Delay
-----

:mod:`iqm.pulse.gates.delay`

Forces a delay between the preceding and following circuit operations.
It can be applied to any number of qubits. Takes one argument, ``duration``, which is the minimum
duration of the delay in seconds. It will be rounded up to the nearest possible duration the
hardware can handle.

.. code-block:: python
    :caption: Example

    CircuitOperation(name='delay', locus=('alice', 'bob'), args={'duration': 80e-9})


.. note::

   We can only guarantee that the delay is *at least* of the requested duration, due to both
   hardware and practical constraints, but could be much more depending on the other operations
   in the circuit. To see why, consider e.g. the circuit

   .. code-block:: python

      (
          CircuitOperation(name='cz', locus=('alice', 'bob'), args={}),
          CircuitOperation(name='delay', locus=('alice',), args={'duration': 1e-9}),
          CircuitOperation(name='delay', locus=('bob',), args={'duration': 100e-9}),
          CircuitOperation(name='cz', locus=('alice', 'bob'), args={}),
      )

   In this case the actual delay between the two CZ gates will be 100 ns rounded up to
   hardware granularity, even though only 1 ns was requested for `alice`.

"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from pydantic import BaseModel, Field

from iqm.pulse import Circuit
from iqm.pulse.builder import build_quantum_ops
from iqm.station_control.interface.models import (
    DDMode,
    DDStrategy,
    HeraldingMode,
    MoveGateFrameTrackingMode,
    MoveGateValidationMode,
    QIRCode,
    QubitMapping,
)

_SUPPORTED_OPERATIONS = build_quantum_ops({})


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


def validate_circuit(circuit: Circuit) -> None:
    """Validates a submitted quantum circuit.

    Args:
        circuit: a circuit that needs validation

    Raises:
        ValueError: validation failed

    """
    if isinstance(circuit, Circuit):
        circuit.validate(_SUPPORTED_OPERATIONS)
    elif isinstance(circuit, QIRCode):
        pass
    else:
        raise ValueError("Every circuit in a batch should be of type <Circuit> or <QIRCode>")


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
                op: (qubit_connectivity if _op_arity(op) == 2 else [[qb] for qb in qubits]) for op in operations
            }

        super().__init__(**data)

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


@dataclass(frozen=True, kw_only=True)
class CircuitJobParameters(CircuitCompilationOptions):
    """Parameters for a circuit execution job (see :class:`RunRequest` for definitions)."""

    shots: int
    """How many times to execute each circuit in the batch, must be greater than zero."""
    calibration_set_id: UUID | None = None
    """ID of the calibration set to use, or None to use the current default calibration set."""
    qubit_mapping: QubitMapping | None = None
    """Mapping of logical qubit names to physical qubit names, or None if ``circuits`` use physical qubit names."""
