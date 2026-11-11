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
"""Principal component analysis for datasets of complex numbers."""

import numpy as np


def principal_component_analysis(
    complex_amplitudes: np.ndarray, poq_param_name: str = "phase_optimal_quad"
) -> tuple[dict[str, float | np.ndarray], np.ndarray]:
    """Use Principal component analysis to compute contrast and the angle of the first principal component
    (Heinsoo2019).

    Can be replaced with sklearn.decomposition.PCA in the future
    (sklearn library version is approx. 2x faster and more versatile).

    Args:
        complex_amplitudes: array of complex numbers to perform principal component analysis on
        poq_param_name: optional custom parameter name to be used for the phase of the first principal component.
            By default, will be ``Parameter("phase_optimal_quad")``.

    Returns:
        - dictionary with parameter name as key and phase optimal quadrature value. This format is needed so it can
            be added to a DataArray
        - resulting real contrast after applying PCA and rotating complex_amplitudes

    """
    scatter = np.array([np.real(complex_amplitudes), np.imag(complex_amplitudes)])
    cov = np.cov(scatter)
    _, eivec = np.linalg.eigh(cov)
    contrast = np.array([np.dot((eivec[:, 1]), row) for row in np.transpose(scatter)])
    phase_optimal_quad = np.angle(eivec[0, 1] + 1j * eivec[1, 1])

    if (contrast < 0).all():
        contrast *= -1
        phase_optimal_quad = (phase_optimal_quad % (2 * np.pi)) - np.pi

    return {poq_param_name: phase_optimal_quad}, contrast
