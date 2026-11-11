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
"""CHAD utils."""

from exa.common.data import SettingNode
from exa.common.qcm_data.chad_model import CHAD


# TODO: Consider typing
#  Should be "ReadoutController" instead of "SettingNode" but it can't be imported from station-control
#  The details of "ReadoutController" are leaking to exa-experiment anyway,
#  so we should be able to type this correctly (import "ReadoutController" interface),
#  or hide the details to station-control.
def get_readout_controllers(chad: CHAD, settings: SettingNode, components: list[str]) -> list[SettingNode] | None:
    """Get readout controllers from settings to given components in CHAD data."""
    probe_lines = chad.get_probe_line_names_for(components)
    return [settings.controllers[probe_line].readout for probe_line in probe_lines]


# TODO: Consider typing
#  Should be "LocalOscillatorController" instead of "SettingNode" but it can't be imported from station-control
#  The details of "LocalOscillatorController" are leaking to exa-experiment anyway,
#  so we should be able to type this correctly (import "LocalOscillatorController" interface),
#  or hide the details to station-control.
def get_local_oscillator_controllers(
    chad: CHAD, settings: SettingNode, components: list[str]
) -> list[SettingNode] | None:
    """Get local oscillator controllers from settings to given components in CHAD data."""
    probe_lines = chad.get_probe_line_names_for(components)
    return [settings.controllers[probe_line].readout.local_oscillator for probe_line in probe_lines]
