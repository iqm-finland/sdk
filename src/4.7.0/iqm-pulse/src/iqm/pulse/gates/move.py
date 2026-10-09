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
r"""Two-qubit MOVE gate.

The MOVE gate is a population exchange operation between two components,
mediated by a coupler, that has the following properties:

* MOVE is unitary.
* The effect of MOVE is only defined in the invariant
  subspace :math:`S = \text{span}\{|00\rangle, |01\rangle, |10\rangle\}`, where it swaps the populations of the states
  :math:`|01\rangle` and :math:`|10\rangle`. Anything may happen in the orthogonal subspace as long as it is unitary and
  invariant.
* In the subspace where it is defined, MOVE is an involution: :math:`\text{MOVE}_S^2 = I_S`.

Thus MOVE has the following presentation in the subspace :math:`S`:

.. math:: \text{MOVE}_S = |00\rangle \langle 00| + a |10\rangle \langle 01| + a^{-1} |01\rangle \langle 10|,

where :math:`a` is an undefined complex phase. This degree of freedom (in addition to the undefined effect of the gate
in the orthogonal subspace) means there is a continuum of different MOVE gates, all equally valid.
The phase :math:`a` is canceled when the MOVE gate is applied a second time due to the involution property.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from exa.common.data.parameter import Parameter, Setting
from iqm.pulse.gates.cz import FluxPulseGate
from iqm.pulse.playlist.instructions import Block, Instruction, VirtualRZ, Wait
from iqm.pulse.playlist.schedule import Schedule
from iqm.pulse.playlist.waveforms import CosineRiseFall, Slepian, TruncatedGaussianSmoothedSquare
from iqm.pulse.timebox import TimeBox
from iqm.pulse.utils import normalize_angle

if TYPE_CHECKING:  # pragma: no cover
    from iqm.pulse.builder import CircuitOperation, ScheduleBuilder


@dataclass(frozen=True, kw_only=True)
class MoveMarker(Wait):
    """Special annotation instruction to indicate the beginning and ending of MOVE gates.

    The *same instance* of this instruction will be inserted into the drive channels of both MOVE components, right
    before the beginning MOVE VirtualRZ instructions,
    to link the channels together (otherwise, there would be nothing explicit in the Schedule
    indicating that there is a MOVE gate happening between two components).

    Another shared instance will be inserted to the aforementioned channels right before the ending
    MOVE VirtualRZ instruction. The VirtualRZ instructions between the markers on the second MOVE component drive
    channel will be applied to the first MOVE component instead in a post-compilation pass.
    """

    duration: int = 0
    first_move_component: str
    second_move_component: str
    detuning: float


class MOVE_CustomWaveforms(FluxPulseGate):
    """Qubit-resonator or qubit-qubit MOVE gate using flux pulses both on a qubit and the coupler.

    This class implements the extra phase bookkeeping logic required to make the MOVE
    gates work as intended. Due to the unknown phase in the MOVE gate definition, the MOVEs
    need to be applied in pairs, i.e. the state is always moved back to the qubit
    it came from. Between a pair of MOVE gates, you can apply any number of other two-component
    gates (CZs for example) to entangle the component where the state was moved to with other qubits.
    This sequence of gates enclosed by two MOVE operations is called a *MOVE sandwich*. At the end of a sandwich we have
    to apply a local phase correction (z rotation) on the state that was moved back to the qubit.

    The :meth:`__call__` method of this class uses the :class:`.MoveMarker` annotation instruction
    to mark the beginning and end of each MOVE sandwich, in order to enable the calculation of the
    angle of the z rotation to be applied on the moved qubit at the end of the sandwich to
    counteract the phase accumulation during the sandwich relative to the computational frame of
    the qubit.
    The phase accumulation has two sources:

    * Phase due to the frequency detuning between the two MOVE components,
      proportional to the time duration of the MOVE sandwich.

    * Phase due to the virtual z rotations applied on the central components as
      gates are applied between it and another qubit, which need to be summed up.
      By convention the central component VirtualRZ angle of the MOVE implementation itself is currently
      always zero (since only the sum of the central component and qubit z rotation angles matters for MOVE),
      but we also include it in the sum for completeness.

    The phases are calculated and applied on the qubits using :func:`.apply_move_gate_phase_corrections`.
    """

    root_parameters: dict[str, Parameter | Setting | dict] = {
        "duration": Parameter("", "Gate duration", "s"),
        "rz": {
            "*": Parameter("", "Z rotation angle", "rad"),  # wildcard parameter
        },
        "detuning": Parameter("", "Detuning of the computational components at parking", "Hz"),
    }
    """Include the detuning between the two MOVE components in the gate parameters for phase tracking."""

    def _call(self) -> TimeBox:
        first_move_component, second_move_component = self.locus
        first_drive_channel = self.builder.get_drive_channel(first_move_component)
        second_drive_channel = self.builder.get_drive_channel(second_move_component)
        detuning = self.calibration_data["detuning"]

        # special annotated zero-duration Wait instruction appearing in both drive channels
        marker = MoveMarker(
            first_move_component=first_move_component, second_move_component=second_move_component, detuning=detuning
        )
        marker_box = self.to_timebox(
            Schedule(
                {
                    first_drive_channel: [marker],
                    second_drive_channel: [marker],
                }
            )
        )
        move_box = super()._call()  # box implementing MOVE
        # first the marker, then the MOVE gate
        return TimeBox.composite([marker_box, move_box], label=move_box.label)


class MOVE_CRF_CRF(MOVE_CustomWaveforms, coupler_wave=CosineRiseFall, qubit_wave=CosineRiseFall):
    """MOVE gate using CRF waveform for the coupler and the qubit flux pulses."""


class MOVE_SLEPIAN_CRF(MOVE_CustomWaveforms, coupler_wave=Slepian, qubit_wave=CosineRiseFall):
    """MOVE gate using Slepian waveform for the coupler flux pulse and CRF waveform for the qubit flux pulse."""


class MOVE_TGSS_CRF(MOVE_CustomWaveforms, coupler_wave=TruncatedGaussianSmoothedSquare, qubit_wave=CosineRiseFall):
    """MOVE gate using TGSS waveform for the coupler flux pulse and CRF waveform for the qubit flux pulse."""


def apply_move_gate_phase_corrections(  # noqa: PLR0915
    schedule: Schedule,
    builder: ScheduleBuilder,
    apply_detuning_corrections: bool = True,
) -> Schedule:
    """Schedule-level pass applying phase corrections for MOVE sandwiches to the first MOVE component.

    .. note:: Assumes the MOVE gate implementation is based on :class:`.MOVE_CustomWaveforms`.

    Processes all the MOVE sandwiches in ``schedule``, summing up the :class:`.VirtualRZ` instructions
    on the (virtual) drive channels of the second MOVE component (either qubit or resonator), adding the phase
    difference resulting from detuning of the two MOVE components to the total, and applying it on the first MOVE
    component (has to be a qubit) at the end of each sandwich.

    Args:
        schedule: instruction schedule to process
        builder: schedule builder that was used to build ``schedule``
        apply_detuning_corrections: if True, also apply detuning phase corrections
    Returns:
        copy of ``schedule`` with the phase corrections applied

    """

    def consume_move_rz(iterator: Iterator[Instruction], component: str) -> VirtualRZ:
        """Consumes the next instruction from the drive channel iterator, ensuring it is a VirtualRZ."""
        inst = next(iterator)
        if not isinstance(inst, VirtualRZ):
            raise ValueError(f"{type(inst)} following MoveMarker on {component}, expected VirtualRZ.")
        return inst

    def find_phase_corrections() -> dict[MoveMarker, deque[float]]:
        """Determine both fixed and dynamic phase corrections for the second component of each MOVE sandwich.

        Loop over drive channels of all computational components (qubits and resonators) to find the phase
        corrections of only the second MOVE locus component for all the MOVE sandwiches.

        Returns:
            A mapping from ``MoveMarker`` marking the end of a MOVE sandwich to a list of total phase
            corrections, in radians, for the MOVE sandwiches ended by that ``MoveMarker``.
            The same ``MoveMarker`` instance appears on the drive channels of both MOVE components, and is re-used
            every time a MOVE is applied between the same two components (because of gate TimeBox caching).
            Hence it may end multiple MOVE sandwiches.
            The list contains one phase for each such sandwich.

        Raises:
            ValueError: If the MOVE sandwich is not properly formed.

        """
        phase_correction: dict[MoveMarker, deque[float]] = defaultdict(deque)

        for component in builder.chip_topology.computational_resonators | builder.chip_topology.qubits:
            drive_channel_name = builder.get_drive_channel(component)
            if drive_channel_name not in schedule:
                continue

            active_move_partner: str | None = None
            accumulated_phase: float = 0.0  # accumulated fixed phase corrections
            accumulated_samples: int = 0  # duration of the MOVE sandwich in samples for dynamic phase correction

            drive_channel = iter(schedule[drive_channel_name])
            for inst in drive_channel:
                if active_move_partner and not isinstance(inst, (Wait, Block, VirtualRZ, MoveMarker)):
                    raise ValueError(f"{type(inst)} is not currently supported in MOVE sandwiches on.")

                if isinstance(inst, MoveMarker):
                    if active_move_partner is None:
                        # --- Start of a MOVE Sandwich ---
                        if component == inst.second_move_component:
                            active_move_partner = inst.first_move_component
                        else:
                            # The MOVE sandwiches are identified from the second MOVE component drive channel.
                            break

                        # Handle the RZ from the MOVE gate following the MOVE marker.
                        rz = consume_move_rz(drive_channel, component)
                        accumulated_phase = rz.phase_increment
                        accumulated_samples = 0

                    else:
                        # --- End of a MOVE Sandwich ---
                        # NOTE: we are now guaranteed to be in the second MOVE component drive channel, since we only
                        # set active_move_partner in that case, and we break out of the loop if we encounter a
                        # MOVE marker in the first MOVE component drive channel.

                        # Handle the RZ from the MOVE gate itself.
                        rz = consume_move_rz(drive_channel, component)
                        accumulated_phase += rz.phase_increment
                        if apply_detuning_corrections:
                            if active_move_partner != inst.first_move_component:
                                raise ValueError(
                                    f"""Detuning correction can only be applied if all MOVE operations come in pairs.
                                    Here, there are consecutive MOVE operations acting on
                                    ({component}, {active_move_partner}) and
                                    ({component}, {inst.first_move_component})."""
                                )
                            # dynamic phase change due to detuning
                            first_drive_channel_name = builder.get_drive_channel(active_move_partner)
                            first_drive_channel = builder.channels[first_drive_channel_name]
                            duration = first_drive_channel.duration_to_seconds(accumulated_samples)
                            detuning_phase = 2 * np.pi * (inst.detuning * duration)
                        else:
                            detuning_phase = 0

                        phase_correction[inst].append(accumulated_phase - detuning_phase)
                        active_move_partner = None

                elif active_move_partner is not None:
                    # We are inside a MOVE sandwich and need to accumulate time (to account for the dynamic phase
                    # correction) and fixed phase corrections.
                    if isinstance(inst, VirtualRZ):
                        accumulated_phase += inst.phase_increment
                        accumulated_samples += inst.duration
                    else:
                        # Wait or Block instructions contribute to the time between the MOVE markers only.
                        accumulated_samples += inst.duration

        return phase_correction

    # --- Phase 1: Calculate phase corrections ---
    phase_correction = find_phase_corrections()

    # --- Phase 2: Apply phase corrections to the first MOVE component (needs an actual drive controller) ---
    new_schedule: dict[str, list[Instruction]] = {}

    for qubit in builder.has_drive:
        q_drive_channel_name = builder.get_drive_channel(qubit)
        if q_drive_channel_name not in schedule:
            continue

        instructions: list[Instruction] = []
        active_second_move_component: str | None = None
        q_drive_channel = iter(schedule[q_drive_channel_name])

        # Replace qubit drive channel contents, remove MoveMarker instructions and apply z rotations.
        for inst in q_drive_channel:
            if isinstance(inst, MoveMarker) and qubit == inst.first_move_component:
                if active_second_move_component is None:
                    # --- Start of a MOVE Sandwich ---
                    active_second_move_component = inst.second_move_component
                else:
                    # --- End of a MOVE Sandwich ---
                    if apply_detuning_corrections and active_second_move_component != inst.second_move_component:
                        raise ValueError(
                            f"Qubit {qubit}: interleaved MOVE gates between {active_second_move_component} and "
                            f"{inst.second_move_component}"
                        )
                    active_second_move_component = None

                    # Retrieve the calculated phase correction for this specific MOVE marker and this occurrence.
                    # The corrections are ordered along the drive channel instructions, so we can just pop the first one
                    # off the list.
                    correction = phase_correction[inst].popleft()
                    # Get the VirtualRZ instruction following the ending marker that is part of the second MOVE gate
                    original_rz = consume_move_rz(q_drive_channel, qubit)
                    total_phase_correction = original_rz.phase_increment + correction
                    # replace the MOVE VirtualRZ with one applying the correct phase shift
                    # Normalize the phase increment to (-pi, pi] so that high number of full turns
                    # do not mess up the instruments (280+ full turns seem to cause some problems based
                    # on tests, probably due to IEEE 754 floating point overflows)
                    instructions.append(
                        VirtualRZ(
                            duration=original_rz.duration,
                            phase_increment=normalize_angle(total_phase_correction),
                        )
                    )
            else:
                instructions.append(inst)

        new_schedule[q_drive_channel_name] = instructions

    return Schedule(
        {ch: new_schedule[ch] if ch in new_schedule else list(instructions) for ch, instructions in schedule.items()}
    )


def validate_move_instructions(
    instructions: Iterable[CircuitOperation],
    builder: ScheduleBuilder,
    validate_prx: bool = True,
) -> Iterable[CircuitOperation]:
    """Circuit-level pass to prepare a circuit containing MOVE gates for compilation.

    Validates that circuit conforms to the MOVE gate constraints.

    Args:
        instructions: quantum circuit to validate
        builder: schedule builder, encapsulating information about the station
        validate_prx: whether to validate the circuit for PRX gates between MOVE sandwiches as well
    Returns:
        ``instructions``, unmodified
    Raises:
        ValueError: Circuit does not conform to MOVE constraints.

    """
    # Mapping from central component (second move component) to the qubit whose state was moved to it (first move
    # component).
    central_component_occupations: dict[str, str] = {}
    # Qubits whose states are currently moved to another component
    moved_qubits: set[str] = set()
    chip_topology = builder.chip_topology

    for inst in instructions:
        if inst.name == "move":
            first_move_component, second_move_component = inst.locus
            if not (
                chip_topology.is_qubit(first_move_component)
                and (
                    chip_topology.is_qubit(second_move_component)
                    or chip_topology.is_computational_resonator(second_move_component)
                )
            ):
                raise ValueError(f"MOVE locus must always be (qubit, resonator) or (qubit, qubit), got {inst.locus}")

            if (initial_move_component := central_component_occupations.get(second_move_component)) is None:
                # Beginning MOVE: check that the state of first_move_component hasn't been moved to another central
                # component
                if first_move_component in moved_qubits:
                    raise ValueError(
                        f"Cannot apply MOVE{inst.locus} because the state of {first_move_component} has already "
                        f"been moved to another central component."
                    )
                central_component_occupations[second_move_component] = first_move_component
                moved_qubits.add(first_move_component)
            else:
                # Ending MOVE: need to ensure that the first move component of this MOVE instruction matches the qubit
                # whose state was originally moved to the central component (initial_move_component)
                if initial_move_component != first_move_component:
                    raise ValueError(
                        f"Cannot apply MOVE{inst.locus} because "
                        + f"{second_move_component} already holds the state of '{initial_move_component}'"
                    )
                del central_component_occupations[second_move_component]
                moved_qubits.remove(first_move_component)
        elif moved_qubits:
            # Validate that qubits whose state has been transferred somewhere, are not used during the MOVE sandwich.
            if (inst.name != "barrier") and (validate_prx or inst.name != "prx"):  # noqa: SIM102
                # Barriers are allowed since they're just meta information and
                # not interacting directly with any real channels
                if overlap := set(inst.locus) & moved_qubits:
                    raise ValueError(
                        f"Operation {inst.name} acting on qubits '{overlap}' is forbidden while their states are moved "
                        f"to a central component."
                    )

    # Finally, validate that all MOVEs have been ended before the circuit ends.
    if central_component_occupations:
        raise ValueError(
            "The following central components are still holding qubit states at "
            + f"the end of the circuit: {central_component_occupations}"
        )
    return instructions
