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

The MOVE gate is a population exchange operation between a qubit and a resonator,
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

from collections.abc import Iterable
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

    The *same instance* of this instruction will be inserted into the qubit drive channel and the
    resonator virtual drive channel, right before the beginning MOVE VirtualRZ instructions,
    to link the channels together (otherwise, there would be nothing explicit in the Schedule
    indicating that there is a MOVE gate happening between the qubit and the resonator).

    Another shared instance will be inserted to the aforementioned channels right before the ending
    MOVE VirtualRZ instruction. The VirtualRZ instructions between the markers on the resonator
    channel will be applied to the qubit instead in a post-compilation pass.
    """

    duration: int = 0
    qubit: str
    resonator: str
    detuning: float


class MOVE_CustomWaveforms(FluxPulseGate):
    """Qubit-resonator MOVE gate using flux pulses on both components.

    This class implements the extra phase bookkeeping logic required to make the MOVE
    gates work as intended. Due to the unknown phase in the MOVE gate definition, the MOVEs
    need to be applied in pairs, i.e. the resonator state is always moved back to the qubit
    it came from. Between this pair of MOVE gates you can apply any number of other two-component
    gates (CZs for example) between the resonator and other qubits. This sequence of
    gates acting on the resonator is called a *MOVE sandwich*. At the end of a sandwich we have
    to apply a local phase correction (z rotation) on the state that was moved back to the qubit.

    The :meth:`__call__` method of this class uses the :class:`.MoveMarker` annotation instruction
    to mark the beginning and end of each MOVE sandwich, in order to enable the calculation of the
    angle of the z rotation to be applied on the moved qubit at the end of the sandwich to
    counteract the phase accumulation during the sandwich relative to the computational frame of
    the qubit.
    The phase accumulation has two sources:

    * Phase due to the frequency detuning between the qubit and the resonator,
      proportional to the time duration the MOVE sandwich.

    * Phase due to the virtual z rotations applied on the resonator as
      gates are applied between it and another qubit, which need to be summed up.
      By convention the resonator VirtualRZ angle of the MOVE implementation itself is currently
      always zero (since only the sum of the resonator and qubit z rotation angles matters for MOVE),
      but we also include it in the sum for completeness.

    The phases are calculated and applied on the qubits using :func:`.apply_move_gate_phase_corrections`.
    """

    root_parameters: dict[str, Parameter | Setting | dict] = {
        "duration": Parameter("", "Gate duration", "s"),
        "rz": {
            "*": Parameter("", "Z rotation angle", "rad"),  # wildcard parameter
        },
        "detuning": Parameter("", "Qubit - resonator detuning", "Hz"),
    }
    """Include the frequency difference between qubit and resonator in the gate parameters for phase tracking."""

    def _call(self) -> TimeBox:
        qubit, resonator = self.locus
        qubit_drive_channel = self.builder.get_drive_channel(qubit)
        resonator_drive_channel = self.builder.get_drive_channel(resonator)
        detuning = self.calibration_data["detuning"]

        # special annotated zero-duration Wait instruction appearing in both drive channels
        marker = MoveMarker(qubit=qubit, resonator=resonator, detuning=detuning)
        marker_box = self.to_timebox(
            Schedule(
                {
                    qubit_drive_channel: [marker],
                    resonator_drive_channel: [marker],
                }
            )
        )
        move_box = super()._call()  # box implementing MOVE
        # first the marker, then the MOVE gate
        return TimeBox.composite([marker_box, move_box], label=move_box.label)


class MOVE_CRF_CRF(MOVE_CustomWaveforms, coupler_wave=CosineRiseFall, qubit_wave=CosineRiseFall):
    # type: ignore[call-arg]
    """Qubit-resonator MOVE gate using the CRF waveform for the coupler and the qubit flux pulse."""


class MOVE_SLEPIAN_CRF(MOVE_CustomWaveforms, coupler_wave=Slepian, qubit_wave=CosineRiseFall):
    # type: ignore[call-arg]
    """Qubit-resonator MOVE gate using the Slepian waveform for the coupler flux pulse and the
    CRF waveform for the qubit flux pulse.
    """


class MOVE_TGSS_CRF(MOVE_CustomWaveforms, coupler_wave=TruncatedGaussianSmoothedSquare, qubit_wave=CosineRiseFall):
    # type: ignore[call-arg]
    """Qubit-resonator MOVE gate using the TGSS waveform for the coupler flux pulse and the
    CRF waveform for the qubit flux pulse.
    """


def apply_move_gate_phase_corrections(  # noqa: PLR0915
    schedule: Schedule,
    builder: ScheduleBuilder,
    apply_detuning_corrections: bool = True,
) -> Schedule:
    """Schedule-level pass applying resonator-related phase corrections in MOVE sandwiches to the moved qubit.

    .. note:: Assumes the MOVE gate implementation is based on :class:`.MOVE_CustomWaveforms`.

    Processes all the MOVE sandwiches in ``schedule``, summing up the :class:`.VirtualRZ` instructions
    on the resonator virtual drive channels, adding the phase difference resulting from
    qubit-resonator detuning to the total, and applying it on the qubit at the end of each sandwich.

    Args:
        schedule: instruction schedule to process
        builder: schedule builder that was used to build ``schedule``
        apply_detuning_corrections: if True, also apply detuning phase corrections
    Returns:
        copy of ``schedule`` with the phase corrections applied

    """

    def handle_move_rz(inst: Instruction, component: str) -> float:
        """Handle a RZ from the MOVE gate, acting on the qubit or resonator drive channel.

        Checks that ``inst`` is of the type we expect, and returns its ``phase_increment``.
        """
        if not isinstance(inst, VirtualRZ):
            raise ValueError(f"{type(inst)} following MoveMarker on {component}, should not happen.")
        return inst.phase_increment  # by convention this is currently zero
        # there is no duration-based phase shift for the RZ since it's a part of the MOVE gate
        # and all local phase shifts are already included in the RZs

    def find_phase_corrections() -> dict[MoveMarker, float]:
        """Loop over resonator virtual drive channels to find the phase corrections for
        the MOVE sandwiches.

        Returns:
            mapping from ``MoveMarker`` marking the end of a MOVE sandwich to the total phase
            correction for that sandwich, in radians
            (the same MoveMarker instance appears on both qubit and resonator drive channels)

        """
        phase_correction: dict[MoveMarker, float] = {}

        for resonator in builder.chip_topology.computational_resonators:
            r_drive_channel_name = builder.get_drive_channel(resonator)
            if r_drive_channel_name not in schedule:
                continue

            move_qubit: str | None = None
            rz_phase: float = 0  # accumulated phase correction
            dt_samples: int = 0  # duration of the MOVE sandwich in samples
            r_drive_channel = iter(schedule[r_drive_channel_name])
            for inst in r_drive_channel:
                if not isinstance(inst, (Wait, Block, VirtualRZ, MoveMarker)):
                    raise ValueError(f"{type(inst)} in virtual drive channel of {resonator}, should not happen.")

                if move_qubit:
                    # ongoing MOVE sandwich
                    if isinstance(inst, MoveMarker):
                        # move ends
                        # handle the RZ from the MOVE gate itself
                        rz_phase += handle_move_rz(next(r_drive_channel), resonator)
                        if apply_detuning_corrections:
                            if move_qubit != inst.qubit:
                                raise ValueError(
                                    f"""Resonator {resonator}: interleaved MOVE gates between {move_qubit} and
                                    {inst.qubit}"""
                                )
                            # dynamic phase change due to detuning
                            q_drive_channel_name = builder.get_drive_channel(move_qubit)
                            q_drive_channel = builder.channels[q_drive_channel_name]
                            duration = q_drive_channel.duration_to_seconds(dt_samples)
                            detuning_phase = 2 * np.pi * (inst.detuning * duration)
                        else:
                            detuning_phase = 0

                        phase_correction[inst] = rz_phase - detuning_phase
                        move_qubit = None
                    elif isinstance(inst, VirtualRZ):
                        rz_phase += inst.phase_increment
                        dt_samples += inst.duration
                    else:
                        # Wait or Block
                        dt_samples += inst.duration
                elif isinstance(inst, MoveMarker):
                    # move begins
                    move_qubit = inst.qubit
                    # handle the RZ from the MOVE gate itself
                    rz_phase = handle_move_rz(next(r_drive_channel), resonator)
                    dt_samples = 0
        return phase_correction

    phase_correction = find_phase_corrections()
    new_schedule: dict[str, list[Instruction]] = {}

    # apply the phase corrections on the drive channels of the moved qubits
    for qubit in builder.chip_topology.qubits:
        q_drive_channel_name = builder.get_drive_channel(qubit)
        if q_drive_channel_name not in schedule:
            continue

        # replace qubit drive channel contents, removing MoveMarker instructions and applying z rotations
        instructions: list[Instruction] = []
        move_resonator: str | None = None
        q_drive_channel = iter(schedule[q_drive_channel_name])
        for inst in q_drive_channel:
            if isinstance(inst, MoveMarker):
                if not move_resonator:
                    # move begins
                    move_resonator = inst.resonator
                else:
                    if apply_detuning_corrections and move_resonator != inst.resonator:
                        raise ValueError(
                            f"Qubit {qubit}: interleaved MOVE gates between {move_resonator} and {inst.resonator}"
                        )
                    # move ends
                    move_resonator = None
                    phase = phase_correction[inst]
                    # a VirtualRZ instruction should always follow the ending marker since the MOVE gate itself has one
                    vrz = next(q_drive_channel)
                    phase += handle_move_rz(vrz, qubit)
                    # replace the MOVE VirtualRZ with one applying the correct phase shift
                    # Normalize the phase increment to (-pi, pi] so that high number of full turns
                    # do not mess up the instruments (280+ full turns seem to cause some problems based
                    # on tests, probably due to IEEE 754 floating point overflows)
                    instructions.append(
                        VirtualRZ(
                            duration=vrz.duration,
                            phase_increment=normalize_angle(phase),
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
    # Mapping from resonator to the qubit whose state was moved to it
    resonator_occupations: dict[str, str] = {}
    # Qubits whose states are currently moved to a resonator
    moved_qubits: set[str] = set()
    chip_topology = builder.chip_topology

    for inst in instructions:
        if inst.name == "move":
            qubit, resonator = inst.locus
            if not (chip_topology.is_qubit(qubit) and chip_topology.is_computational_resonator(resonator)):
                raise ValueError(f"Move operation locus must always be (qubit, resonator), got '{inst.locus}'")

            if (resonator_qubit := resonator_occupations.get(resonator)) is None:
                # Beginning MOVE: check that the qubit hasn't been moved to another resonator
                if qubit in moved_qubits:
                    raise ValueError(
                        f"Cannot apply MOVE{inst.locus} because "
                        + "the state of {qubit} is already moved to another resonator"
                    )
                resonator_occupations[resonator] = qubit
                moved_qubits.add(qubit)
            else:
                # Ending MOVE: check that the qubit matches to the qubit that was moved to the resonator
                if resonator_qubit != qubit:
                    raise ValueError(
                        f"Cannot apply MOVE{inst.locus} because "
                        + f"{resonator} already holds the state of '{resonator_qubit}'"
                    )
                del resonator_occupations[resonator]
                moved_qubits.remove(qubit)
        elif moved_qubits:
            # Validate that moved qubits are not used during MOVE operations
            if (inst.name != "barrier") and (validate_prx or inst.name != "prx"):
                # Barriers are allowed since they're just meta information and
                # not interacting directly with any real channels
                if overlap := set(inst.locus) & moved_qubits:
                    raise ValueError(
                        f"Operation {inst.name} to qubits '{overlap}' are forbidden while their states are moved to "
                        + "a resonator"
                    )

    # Finally validate that all moves have been ended before the circuit ends
    if resonator_occupations:
        raise ValueError(
            "The following resonators are still holding qubit states at "
            + f"the end of the circuit: {resonator_occupations}"
        )
    return instructions
