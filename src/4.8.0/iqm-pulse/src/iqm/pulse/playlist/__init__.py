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
"""Control pulses and instruction schedules for quantum computers."""

__all__ = [
    "NONSOLID",
    "TOLERANCE",
    "TOLERANCE_DECIMALS",
    "AcquisitionMethod",
    "Block",
    "ChannelDescription",
    "ChannelProperties",
    "Chirp",
    "ChirpImag",
    "ComplexIntegration",
    "ConditionalInstruction",
    "Constant",
    "Cosine",
    "CosineFall",
    "CosineFallFlex",
    "CosineRise",
    "CosineRiseFall",
    "CosineRiseFallDerivative",
    "CosineRiseFlex",
    "FastDragI",
    "FastDragQ",
    "FluxPulse",
    "Gaussian",
    "GaussianDerivative",
    "GaussianSmoothedSquare",
    "HdDragI",
    "HdDragQ",
    "IQChannelConfig",
    "IQPulse",
    "Instruction",
    "ModulatedCosineRiseFall",
    "MultiplexedIQPulse",
    "Nothing",
    "PiecewiseConstant",
    "Playlist",
    "PolynomialCosine",
    "ProbeChannelProperties",
    "ReadoutMetrics",
    "ReadoutTrigger",
    "RealPulse",
    "Samples",
    "Schedule",
    "Segment",
    "Slepian",
    "SuperGaussianRiseFall",
    "ThresholdStateDiscrimination",
    "TimeTrace",
    "TruncatedGaussian",
    "TruncatedGaussianDerivative",
    "TruncatedGaussianSmoothedSquare",
    "VirtualRZ",
    "Wait",
    "Waveform",
    "build_channel_properties",
    "evaluate_hd_drag_i_envelope",
    "evaluate_hd_drag_q_envelope",
    "round_duration_to_granularity_samples",
    "solve_cosine_coefs_for_hd_drag",
    "solve_hd_drag_coefficients_from_suppressed_frequencies",
]

from iqm.models.playlist.playlist import Playlist

from .channel import (
    ChannelDescription,
    ChannelProperties,
    IQChannelConfig,
    ProbeChannelProperties,
    round_duration_to_granularity_samples,
)
from .channel import _build_channel_properties as build_channel_properties
from .fast_drag import FastDragI, FastDragQ
from .hd_drag import (
    HdDragI,
    HdDragQ,
    evaluate_hd_drag_i_envelope,
    evaluate_hd_drag_q_envelope,
    solve_cosine_coefs_for_hd_drag,
    solve_hd_drag_coefficients_from_suppressed_frequencies,
)
from .instructions import (
    NONSOLID,
    AcquisitionMethod,
    Block,
    ComplexIntegration,
    ConditionalInstruction,
    FluxPulse,
    Instruction,
    IQPulse,
    MultiplexedIQPulse,
    Nothing,
    ReadoutMetrics,
    ReadoutTrigger,
    RealPulse,
    ThresholdStateDiscrimination,
    TimeTrace,
    VirtualRZ,
    Wait,
)
from .schedule import TOLERANCE, TOLERANCE_DECIMALS, Schedule, Segment
from .waveforms import (
    Chirp,
    ChirpImag,
    Constant,
    Cosine,
    CosineFall,
    CosineFallFlex,
    CosineRise,
    CosineRiseFall,
    CosineRiseFallDerivative,
    CosineRiseFlex,
    Gaussian,
    GaussianDerivative,
    GaussianSmoothedSquare,
    ModulatedCosineRiseFall,
    PiecewiseConstant,
    PolynomialCosine,
    Samples,
    Slepian,
    SuperGaussianRiseFall,
    TruncatedGaussian,
    TruncatedGaussianDerivative,
    TruncatedGaussianSmoothedSquare,
    Waveform,
)
