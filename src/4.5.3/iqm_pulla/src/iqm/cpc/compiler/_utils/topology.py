# Copyright 2024-2025 IQM
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
from exa.common.data.setting_node import SettingNode
from exa.common.qcm_data.chip_topology import ChipTopology
from iqm.pulse.gate_implementation import (
    PROBE_LINES_LOCUS_MAPPING,
    SINGLE_COMPONENTS_WITH_DRIVE_LOCUS_MAPPING,
    SINGLE_COMPONENTS_WITH_FLUX_AWG_LOCUS_MAPPING,
    SINGLE_COMPONENTS_WITH_READOUT_LOCUS_MAPPING,
)
from iqm.pulse.gates.cz import FLUX_PULSED_QUBITS_2QB_MAPPING

QUBITS_CONNECTED_THROUGH_RESONATOR_MAPPING: str = "qubits_connected_through_resonator_mapping"


def get_component_to_available_channels_mapping(
    chip_topology: ChipTopology,
    station_control_settings: SettingNode,
    operations: dict[str, dict[str, str]],
) -> dict[str, frozenset[str]]:
    """Find the control channels available for each component.

    Args:
        chip_topology: The chip topology.
        station_control_settings: The station control settings (i.e. the controller nodes).
        operations: The component to operations mapping.

    Returns:
        A mapping of channel names to available channels.

    """
    controllers = station_control_settings
    components = chip_topology.qubits_sorted + chip_topology.couplers_sorted + chip_topology.probe_lines_sorted
    mapping: dict[str, frozenset[str]] = {}
    for component in components:
        available_channels = []
        if component in chip_topology.component_to_probe_line:
            available_channels.append("readout")
        if "drive" in operations.get(component, {}):
            available_channels.append("drive")
        if controllers and "flux" in operations.get(component, {}):
            flux_name = operations[component]["flux"]
            if flux_name in controllers.children and "awg" in controllers[flux_name].children:  # type: ignore[union-attr]
                available_channels.append("flux")
        mapping[component] = frozenset(available_channels)
    return mapping


def set_single_component_mapping_in_init(
    chip_topology: ChipTopology, available_channels: dict[str, frozenset[str]]
) -> None:
    """Set readout, drive, and flux channel mappings in a Chip Topology.

    Args:
        chip_topology: The chip topology.
        available_channels: The available channels (see `:func:`get_available_channels`).

    """
    mappings: dict[str, dict] = {}
    for operation in ("drive", "readout", "flux"):
        mappings[operation] = {(c,): (c,) for c, chans in available_channels.items() if operation in chans}
    probe_line_mapping = {(p,): (p,) for p in chip_topology.probe_lines}
    chip_topology.set_locus_mapping(SINGLE_COMPONENTS_WITH_DRIVE_LOCUS_MAPPING, mappings["drive"])
    chip_topology.set_locus_mapping(SINGLE_COMPONENTS_WITH_READOUT_LOCUS_MAPPING, mappings["readout"])
    chip_topology.set_locus_mapping(SINGLE_COMPONENTS_WITH_FLUX_AWG_LOCUS_MAPPING, mappings["flux"])
    chip_topology.set_locus_mapping(PROBE_LINES_LOCUS_MAPPING, probe_line_mapping)  # type: ignore[arg-type]


def set_fast_cz_mapping_in_init(chip_topology: ChipTopology, available_channels: dict[str, frozenset[str]]) -> None:
    """Set Fast CZ mapping in a Chip Topology.

    Args:
        chip_topology: The chip topology.
        available_channels: The available channels (see `:func:`get_available_channels`).

    """
    fast_cz_mapping = {}
    coupler_mapping = {c: list(qs) for c, qs in chip_topology.coupler_to_components.items()}
    for coupler, qubits in coupler_mapping.items():
        for qubit_ind, qubit in enumerate(qubits):
            if qubit in available_channels:
                if "flux" in available_channels[qubit]:
                    mapped_pair = qubits.copy()
                    if qubit_ind > 0:
                        mapped_pair.reverse()
                    fast_cz_mapping[tuple(mapped_pair)] = (coupler,)
    chip_topology.set_locus_mapping(FLUX_PULSED_QUBITS_2QB_MAPPING, fast_cz_mapping)  # type: ignore[arg-type]


def set_qubits_through_resonator_mapping_in_init(chip_topology: ChipTopology) -> None:
    """Create and register the QUBITS_CONNECTED_THROUGH_RESONATOR_MAPPING locus mapping.

    Searches the chip topology looking for pairs of qubits such that they are connected through a sequence
    of coupler-computational resonator-coupler and adds those pairs to the mapping. These mappings are used to
    enable two-qubit gates, and so it does not matter if qubits are connected through more than one resonator.
    If there is at least one, they will be part of the mapping. If there is no common neighbor resonator, they
    will not be part of the mapping. The first resonator is added to the mapping for convenience, but if there are
    more resonators, they can be found from chip_topology anyway.

    Args:
        chip_topology: The chip topology.

    """
    # TODO: When moving locus mappings to gates, make use of the locus mapping, letting the gate implementation
    # pick the resonator correctly.
    mapping = {}
    qubits = chip_topology.qubits
    for first_qubit in qubits:
        for second_qubit in qubits:
            if first_qubit == second_qubit:
                continue
            resonators = chip_topology.get_all_common_resonators([first_qubit, second_qubit])
            if resonators:
                resonator = sorted(resonators)[0]
                first_coupler = chip_topology.get_coupler_for(first_qubit, resonator)
                second_coupler = chip_topology.get_coupler_for(second_qubit, resonator)
                mapping[(first_qubit, second_qubit)] = (first_coupler, resonator, second_coupler)
    chip_topology.set_locus_mapping(QUBITS_CONNECTED_THROUGH_RESONATOR_MAPPING, mapping)  # type: ignore[arg-type]
