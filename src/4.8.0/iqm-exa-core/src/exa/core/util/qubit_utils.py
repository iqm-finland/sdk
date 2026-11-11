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

"""Utility functions for qubit characterization, dispersion functions for different qubit and coupler spectra."""

import numpy as np

from exa.common.data import SettingNode


def simple_transmon_ge_frequency_from_settings(
    v: np.ndarray | float, component: str, settings: SettingNode
) -> np.ndarray | float:
    """Get the simplified transmon dispersion directly from settings.

    Args:
        v: The voltages at which the transition frequencies are returned.
        component: name of the component
        settings: full setting tree

    Returns:
        g-e transition frequency at voltage v for the chosen component.

    """
    model_settings = settings.characterization.model[component]

    f_sweet_spot = model_settings.sweet_spot_frequency.value
    v_offs = model_settings.voltage_offset.value
    v_period = model_settings.voltage_period.value
    d = model_settings.squid_asymmetry.value

    return simple_transmon_ge_frequency(v, f_sweet_spot, v_offs, v_period, d)


def simple_transmon_ge_frequency(
    v: np.ndarray | float, f_sweet_spot: float, v_offs: float, v_period: float, d: float = 0
) -> np.ndarray:
    """Simplified model for transmon dispersion. The transmon dispersion can be derived from koch2007 by calculating
    :math:`E_1 - E_0` using equation 2.11 and then substituting :math:`E_j` with equation 2.18.

    To reduce the number of unknown parameters, :math:`E_c` is ignored here and assumed to
    be accounted for by `f_sweet_spot`. This results in a slight inaccuracy in the model
    when `f_sweet_spot` is similar to :math:`E_c`.

    Args:
        v: list of voltages
        f_sweet_spot: Sweet spot frequency of the transmon.
        v_offs: offset voltage of the sweet spot
        v_period: period of transmon spectrum in voltage
        d: asymmetry of the transmon SQUID junction.

    Returns:
        g-e transition frequency at voltage v.

    """
    phi = np.pi * (v - v_offs) / v_period
    return f_sweet_spot * (np.cos(phi) ** 2 + (d * np.sin(phi)) ** 2) ** 0.25
