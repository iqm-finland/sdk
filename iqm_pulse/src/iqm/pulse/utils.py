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
from typing import get_args, get_origin

import numpy as np

from exa.common.data.parameter import CollectionType, DataType
from iqm.pulse.playlist import IQPulse
from iqm.pulse.playlist.waveforms import Samples


def map_waveform_param_types(type_hint: type) -> tuple[DataType, CollectionType]:
    """Map a python typehint into EXA Parameter's `(DataType, CollectionType)` tuple.

    Args:
        type: python typehint.

    Returns:
        A `(DataType, CollectionType)` tuple

    Raises:
        ValueError: for a non-supported type.

    """
    value_error = ValueError(f"Nonsupported datatype for a waveform parameter: {type_hint}")
    if hasattr(type_hint, "__iter__") and type_hint is not str:
        if type_hint == np.ndarray:
            data_type = DataType.COMPLEX  # due to np.ndarray not being generic we assume complex numbers
            collection_type = CollectionType.NDARRAY
            return (data_type, collection_type)
        if get_origin(type_hint) is list:
            collection_type = CollectionType.LIST
            type_hint = get_args(type_hint)[0]
        else:
            raise value_error
    else:
        collection_type = CollectionType.SCALAR

    if type_hint is float:
        data_type = DataType.FLOAT
    elif type_hint is int:
        data_type = DataType.INT
    elif type_hint is str:
        data_type = DataType.STRING
    elif type_hint is complex:
        data_type = DataType.COMPLEX
    elif type_hint is bool:
        data_type = DataType.BOOLEAN
    else:
        raise value_error
    return (data_type, collection_type)


def normalize_angle(angle: float) -> float:
    """Normalize the given angle to (-pi, pi].

    Args:
        angle: angle to normalize (in radians)

    Returns:
        ``angle`` normalized to (-pi, pi]

    """
    half_turn = np.pi
    full_turn = 2 * half_turn
    return (angle - half_turn) % -full_turn + half_turn


def phase_transformation(psi_1: float = 0.0, psi_2: float = 0.0) -> tuple[float, float]:
    r"""Implement an arbitrary (RZ, PRX, RZ) gate sequence by modifying the parameters of the
    IQ pulse implementing the PRX.

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


def modulate_iq(pulse: IQPulse) -> np.ndarray:
    """Sampled baseband waveform of an IQ pulse.

    Note that :attr:`IQPulse.phase_increment` has no effect on the sampled waveform.
    The upconversion oscillator phase incrementation is a separate action performed by the AWG
    that also affects future IQPulses, and thus cannot be represented by an array of waveform samples.
    To replicate the effect of ``pulse`` on an AWG, one should first perform the increment and then
    play the returned samples.

    Args:
        pulse: IQ pulse.

    Returns:
        The waveform of ``pulse`` as an array of complex-valued samples.

    """
    # TODO: Could be an IQPulse method
    wave = pulse.wave_i.sample() * pulse.scale_i + 1j * pulse.wave_q.sample() * pulse.scale_q
    # starting times of the samples, in units of inverse sample rate
    wave_sampletimes = np.arange(len(wave))
    wave *= np.exp(2j * np.pi * pulse.modulation_frequency * wave_sampletimes + 1j * pulse.phase)
    return wave


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
        return samples, 1.0

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
