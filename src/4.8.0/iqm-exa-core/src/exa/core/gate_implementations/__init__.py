#  ********************************************************************************
#   _____  ____ _
#   / _ \ \/ / _` |  Framework for control
#  |  __/>  < (_| |  and measurement of
#   \___/_/\_\__,_|  superconducting qubits
#
#  Copyright 2019-2026 IQM Finland Oy
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ********************************************************************************
"""Implementations of non-standard gates and waveforms."""

__all__ = [
    "RZ_FluxPulse",
    "X_EF_GateImplementation",
    "RZ_ConstantFluxPulse",
    "ModulatedGaussianSmoothedSquare",
    "ModulatedTGSS",
    "FluxPulse",
    "ConstantFluxPulse",
    "CosineFluxPulse",
    "CRFFluxPulse",
    "PiecewiseConstantFluxPulse",
    "PolynomialCosineFluxPulse",
    "FluxPulseSmoothConstant",
    "CouplerFluxPulseSquare",
    "Coupler_FluxPulse_Square",
    "FluxPulseSmoothConstantBase",
]

from .flux_pulse import (
    ConstantFluxPulse,
    CosineFluxPulse,
    Coupler_FluxPulse_Square,
    CouplerFluxPulseSquare,
    CRFFluxPulse,
    FluxPulse,
    FluxPulseSmoothConstant,
    FluxPulseSmoothConstantBase,
    PiecewiseConstantFluxPulse,
    PolynomialCosineFluxPulse,
)
from .modulated_pulse_gate import ModulatedGaussianSmoothedSquare, ModulatedTGSS
from .rz import RZ_ConstantFluxPulse, RZ_FluxPulse
from .x_ef_gate import X_EF_GateImplementation
