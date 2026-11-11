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
"""Utility functions for the circuit generation of Experiment."""

from exa.common.data import SettingNode
from exa.common.qcm_data import ChipTopology
from iqm.pulse.builder import ScheduleBuilder
from iqm.pulse.playlist.channel import ChannelProperties, _build_channel_properties
from iqm.pulse.timebox import TimeBox

from .setting_utils import get_components_without_readout


def get_default_channel_properties(
    settings: SettingNode, chip_topology: ChipTopology
) -> tuple[dict[str, ChannelProperties], dict[str, dict[str, str]]]:
    """Get the channel properties and component to channel mapping for ScheduleBuilder.

    Args:
        settings: The settings tree.
        chip_topology: The chip topology.

    Returns:
        The channel properties and the component to channel mapping.

    """
    # TODO remove this function, channel properties should be obtained from the Compiler.
    drive_controllers = {}
    flux_controllers = {}
    readout_controllers = {}
    for component_name, component_node in settings.controllers.child_nodes:
        if "readout" in component_node.children and component_name in chip_topology.probe_lines:
            readout_controllers[component_name] = component_node.readout
        if "drive" in component_node.children and "awg" in component_node.drive.children:
            drive_controllers[component_name] = component_node.drive.awg
        if "flux" in component_node.children and "awg" in component_node.flux.children:
            flux_controllers[component_name] = component_node.flux.awg
    return _build_channel_properties(chip_topology, drive_controllers, flux_controllers, readout_controllers)


def handle_component_without_readout(
    group: tuple[str, ...],
    builder: ScheduleBuilder,
    circuit: TimeBox,
) -> TimeBox:
    """Handles components without native readout by transferring their state to a helper component.

    If a gate component (first two elements in ``group``) does not have a probe line connection, its state is
    transferred to the helper component (third element in ``group``) via a MOVE operation.

    Args:
        group: A single group of the run components. The first two are gate components
            and the optional third is the helper component.
        builder: ScheduleBuilder
        circuit: The experiment circuit (without the readout part).

    Returns:
        The updated circuit with the MOVE operation added in case it is required and a helper component is specified.

    Raises:
        ValueError: If both gate components do not have a probe line connection.

    """
    components_without_readout = get_components_without_readout(list(group[0:2]), builder.chip_topology)

    if not components_without_readout:
        return circuit

    if len(components_without_readout) > 1:
        raise ValueError(
            f"{len(components_without_readout)} components {components_without_readout} do not have readout. "
            f"This case is not supported."
        )

    component_without_readout = next(iter(components_without_readout))
    if len(group) == 3:
        barrier = builder.get_implementation("barrier", group)()
        move = builder.get_implementation("move", (group[2], component_without_readout))()
        circuit += barrier + move

    return circuit
