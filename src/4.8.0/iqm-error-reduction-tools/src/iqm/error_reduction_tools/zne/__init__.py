# Copyright 2022-2026 IQM
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

"""Zero-noise extrapolation (ZNE): amplify the noise by global folding, then extrapolate to zero noise."""

from iqm.error_reduction_tools.zne.zne_modifiers import (
    ACHIEVED_FOLDS_KEY,
    as_fold_count,
    compact_factor,
    fold_global_pulse,
    noise_factor_for,
)
from iqm.error_reduction_tools.zne.zne_plots import plot_zne_comparison, plot_zne_fit
from iqm.error_reduction_tools.zne.zne_processors import extrapolate_at_zero, fit_curve
from iqm.error_reduction_tools.zne.zne_technique import ZNE, NoiseAmplifiedVariant, ZNEDiagnostics, ZNEEstimation

__all__ = [
    "ACHIEVED_FOLDS_KEY",
    "ZNE",
    "NoiseAmplifiedVariant",
    "ZNEDiagnostics",
    "ZNEEstimation",
    "as_fold_count",
    "compact_factor",
    "extrapolate_at_zero",
    "fit_curve",
    "fold_global_pulse",
    "noise_factor_for",
    "plot_zne_comparison",
    "plot_zne_fit",
]
