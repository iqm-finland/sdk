#  ********************************************************************************
#
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
"""Utility functions."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from functools import cache
from typing import TYPE_CHECKING

import numpy as np

from iqm.pulse.base_utils import map_waveform_param_types, normalize_angle  # noqa: F401  backw. compatibility for now
from iqm.pulse.gate_implementation import CompositeGate, Locus
from iqm.pulse.locus_mappings import StationProperties
from iqm.pulse.playlist import IQPulse
from iqm.pulse.playlist.waveforms import Samples
from iqm.pulse.timebox import TimeBox

if TYPE_CHECKING:
    from iqm.pulse.circuit_operations import Circuit


TOLERANCE = 1e-10
"""Tolerance for floating point artefacts in waveform computations."""


def phase_transformation(psi_1: float = 0.0, psi_2: float = 0.0) -> tuple[float, float]:
    r"""Implement an (RZ, PRX, RZ) gate sequence by modifying the parameters of the IQ pulse implementing the PRX.

    By commutation rules we have

    .. math::
       RZ(\psi_2) \: PRX(\theta, \phi) \: RZ(\psi_1) = PRX(\theta, \phi+\psi_2) \: RZ(\psi_1 + \psi_2).

    Hence an arbitrary (RZ, PRX, RZ) gate sequence is equivalent to (RZ, PRX) with adjusted angles.

    Use case: with resonant driving, the PRX gate can be implemented using an :class:`IQPulse` instance,
    and the preceding RZ can be handled by decrementing the local oscillator phase beforehand (something
    the IQPulse instruction can also do), which is equivalent to rotating the local computational frame
    around the z axis in the opposite direction of the required quantum state rotation.

    Args:
        psi_1: RZ angle before the PRX (in rad)
        psi_2: RZ angle after the PRX (in rad)

    Returns:
        change to the PRX phase angle (in rad),
        phase increment for the IQ pulse that implements the remaining RZ (in rad)

    """
    return psi_2, -(psi_1 + psi_2)


def modulate_iq(pulse: IQPulse, assert_ranges: bool = False) -> np.ndarray:
    """Sampled baseband waveform of an IQ pulse.

    Note that :attr:`IQPulse.phase_increment` has no effect on the sampled waveform.
    The upconversion oscillator phase incrementation is a separate action performed by the AWG
    that also affects future IQPulses, and thus cannot be represented by an array of waveform samples.
    To replicate the effect of ``pulse`` on an AWG, one should first perform the increment and then
    play the returned samples.

    Args:
        pulse: IQ pulse.
        assert_ranges: If True, check that the resulting waveform is in the range [-1, 1]. Small floating point
            artifacts are tolerated and clipped to the range.

    Returns:
        The waveform of ``pulse`` as an array of complex-valued samples.

    """
    # TODO: Could be an IQPulse method
    wave = pulse.wave_i.sample() * pulse.scale_i + 1j * pulse.wave_q.sample() * pulse.scale_q
    # starting times of the samples, in units of inverse sample rate
    wave_sampletimes = np.arange(len(wave))
    wave *= np.exp(2j * np.pi * pulse.modulation_frequency * wave_sampletimes + 1j * pulse.phase)

    if not assert_ranges:
        return wave

    max_norm = np.max(np.abs(wave))
    if max_norm > 1.0 + TOLERANCE:
        raise ValueError(
            f"Modulated IQ pulse waveform not in range [-1, 1]. Reduce "
            f"scale_i and scale_q. Max norm: {max_norm}. Amplitudes: "
            f"{pulse.scale_i=}, {pulse.scale_q=}."
        )

    return np.clip(wave.real, -1, 1) + 1j * np.clip(wave.imag, -1, 1)


def fuse_iq_pulses(iq_pulses: Iterable[IQPulse]) -> IQPulse:
    """Fuse multiple IQPulses into one by concatenating the sampled waveforms.

    Works by flushing :attr:`IQPulse.phase_increment` s to the front, updating the :attr:`IQPulse.phase` s,
    sampling the pulses, concatenating, normalizing the amplitudes, and putting the result
    into a new IQPulse instruction with a ``phase_increment`` that is a sum of the individual ``phase_increment`` s.

    Additionally, to conserve waveform memory on the AWGs, we normalize the waveform phase by setting
    :attr:`IQPulse.phase` of the fused pulse to the flushed phase of the first pulse.

    Args:
        iq_pulses: IQPulse instructions to fuse.

    Returns:
        Fused IQPulse that behaves indentically to the sequence ``iq_pulses`` on an AWG.

    """
    # flush the phase increments to the start of the pulse sequence
    phases = np.array([i.phase for i in iq_pulses])
    phase_increments = np.array([i.phase_increment for i in iq_pulses])

    # flushed_phases[k] == phases[k] - np.sum(phase_increments[k+1:])
    flushed_phases = phases - np.cumsum(phase_increments[::-1])[::-1] + phase_increments

    # Phase normalization of the samples to save waveform memory: There is an internal degree of freedom
    # in the sampled IQPulse: IQPulse.phase can be represented in the global phase of the samples.
    # Fix this d.o.f. by setting the phase of the fused IQ pulse to the phase of the first constituent IQ pulse.
    fused_phase = flushed_phases[0]
    flushed_phases -= fused_phase
    flushed_iq_pulses = [
        replace(instr, phase=phase, phase_increment=0.0) for instr, phase in zip(iq_pulses, flushed_phases)
    ]
    # sample and concatenate the IQ pulses
    samples = np.hstack([modulate_iq(i) for i in flushed_iq_pulses])

    # normalize the real and imaginary waveform components
    def normalize(samples: np.ndarray) -> tuple[np.ndarray, float]:
        """Normalize real-valued samples to [-1, 1]."""
        scale = np.max(np.abs(samples))
        # avoid division by zero
        if scale > 0:
            return samples / scale, scale
        # samples are all zero, so the component carries no signal and should get zero amplitude budget
        return samples, 0.0

    samples.real, scale_i = normalize(samples.real)
    samples.imag, scale_q = normalize(samples.imag)
    return IQPulse(
        duration=len(samples),
        wave_i=Samples(samples.real),
        wave_q=Samples(samples.imag),
        scale_i=scale_i,
        scale_q=scale_q,
        phase=fused_phase,
        phase_increment=np.sum(phase_increments),
        modulation_frequency=0.0,  # modulate_iq takes care of this
    )


def circuit_to_gate_implementation(circuit: Circuit, circuit_locus: Locus) -> type[CompositeGate]:
    """Wrap a circuit to a single GateImplementation that can then be registered as a gate.

    Returns a composite GateImplementation which, when called, produces a TimeBox with the circuit contents
    scheduled ASAP. ``circuit`` must contain only gates that are registered in IQM Pulse.
    The gate implementation does not need calibration data of its own: it uses the calibration of the member gates.

    Args:
        circuit: Circuit to wrap, typically a small subset of a larger circuit.
        circuit_locus: All the locus component names used in ``circuit``, in the order they
            appear in the gate locus.

    Returns:
        Composite gate implementation class which can be registered to iqm-pulse as
        an implementation of a new gate.

    """

    class CircuitAsComposite(CompositeGate):
        """Dynamically created GateImplementation that wraps a circuit."""

        registered_gates = tuple({instr.name for instr in circuit.instructions})

        def __call__(self):
            # figure out the mapping from template circuit qubits to physical qubits in the locus
            circuit_to_physical = dict(zip(circuit_locus, self.locus))
            boxes = []
            for instr in circuit.instructions:
                locus = tuple(circuit_to_physical[q] for q in instr.locus)
                boxes.append(self.build(instr.name, locus, instr.implementation)(**instr.args))

            return TimeBox.composite(boxes, label=circuit.name)

    return CircuitAsComposite


@cache
def _two_color_qubits(station_properties: StationProperties) -> dict[str, set[str]]:
    """Get bipartite colouring of qubits such that neighbor qubits are in different colours.

    Args:
        station_properties: StationProperties object.

    Returns:
        Dictionary of bipartite colourings of qubits. Keys are the colour "names" (``"A"`` or ``B``) and values
            are the sets of qubits in that colour.

    Raises:
        ValueError: If the bipartite colouring fails, which may happens e.g. if the QPU topology is not fully connected.

    """
    # FIXME: this does not work in all topologies, but should work in CRYSTAL, STAR and CONSTELLATION
    chip_topology = station_properties.qpu_topology
    colours = {"A": {chip_topology.qubits_sorted[0]}, "B": set()}
    other = {"A": "B", "B": "A"}
    # assign each qubit to colour iff it shares no couplers with any qubit in that colour
    unassigned = list(chip_topology.qubits_sorted[1:])
    num_unassigned = len(unassigned)
    while unassigned:
        # iterate until everything is assigned, which should be possible if the QPU topology is fully connected
        for qubit in unassigned.copy():
            assigned = False
            for colour_label, colour in colours.items():
                for colour_qubit in colour:
                    pair = {qubit, colour_qubit}
                    for connected_pair in chip_topology.coupler_to_components.values():
                        # qubit shares a coupler with a qubit in A => it belongs to B (and vice versa)
                        if set(connected_pair) == pair:
                            colours[other[colour_label]].add(qubit)
                            unassigned.remove(qubit)
                            assigned = True
                            break
                    if assigned:
                        break
                if assigned:
                    break
        if num_unassigned == len(unassigned):
            # if this happens, the iteration would not terminate, so we must raise
            raise ValueError("Bipartite colouring failed; perhaps the QPU topology is not fully connected?")
        num_unassigned = len(unassigned)

    if chip_topology.get_connecting_couplers(colours["A"]) or chip_topology.get_connecting_couplers(colours["B"]):
        # if there are couplers connecting the two colours, the colouring is not valid
        raise ValueError("Bipartite colouring failed, the used QPU topology is not supported.")

    return colours


def get_other_colour(station_properties: StationProperties, qubit: str) -> set[str]:
    """Get the set of qubits that are in the other colour than the given qubit.

    Args:
        station_properties: StationProperties object.
        qubit: Qubit to get the other colour from.

    Returns:
        The qubits in the other colour.

    """
    colours = _two_color_qubits(station_properties)
    return colours["A"] if qubit not in colours["A"] else colours["B"]
