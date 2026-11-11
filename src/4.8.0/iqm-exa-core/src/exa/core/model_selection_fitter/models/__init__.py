#  ********************************************************************************
#    _____  ____ _
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
r"""Contains ``lmfit.model``\s that are compatible with the model selection fitter."""

from .absolute_value_model import AbsoluteValueModel
from .decaying_sine_model import GaussExpAbsModel, GaussExpSineModel
from .dispersion_models.double_qubit_coupler_model import DoubleQubitCouplerDispersionModel
from .dispersion_models.simple_transmon_model import SimpleTransmonDispersionModel
from .dispersion_models.single_qubit_coupler_model import QubitCouplerDispersionModel
from .double_exponential_decay_model import DoubleExponentialDecay
from .double_gaussian_model import DoubleGaussianModel
from .exponential_offset_model import BiexponentialDecayWithOffset, ExponentialDecayWithOffset
from .gaussian_model import GaussianModel
from .linear_model import LinearModel
from .lorentzian_model import LorentzianModel
from .model_utils import BaseModel
from .nyquist_sawtooth_model import NyquistSawtoothModel
from .polynomial_model import FirstOrderPolyModel, SecondOrderPolyModel, ThirdOrderPolyModel
from .ramsey2f_model import Ramsey2fModel
from .ramsey2f_separable_model import (
    RAMSEY_2F_LINEAR_PARAMS,
    RAMSEY_2F_SEPARABLE_PARAMS,
    Ramsey2fSeparableModel,
    ramsey_2f_basis,
    ramsey_2f_separable_response,
)
from .sine_expansion_model import SmallAngleSineExpansion
from .sine_model import SineModel

__all__ = [
    "AbsoluteValueModel",
    "BaseModel",
    "BiexponentialDecayWithOffset",
    "DoubleGaussianModel",
    "GaussExpSineModel",
    "GaussExpAbsModel",
    "GaussianModel",
    "LorentzianModel",
    "NyquistSawtoothModel",
    "Ramsey2fModel",
    "Ramsey2fSeparableModel",
    "RAMSEY_2F_LINEAR_PARAMS",
    "RAMSEY_2F_SEPARABLE_PARAMS",
    "ramsey_2f_basis",
    "ramsey_2f_separable_response",
    "ExponentialDecayWithOffset",
    "SmallAngleSineExpansion",
    "DoubleQubitCouplerDispersionModel",
    "SimpleTransmonDispersionModel",
    "QubitCouplerDispersionModel",
    "LinearModel",
    "LorentzianModel",
    "ThirdOrderPolyModel",
    "SecondOrderPolyModel",
    "FirstOrderPolyModel",
    "DoubleExponentialDecay",
    "SineModel",
]
