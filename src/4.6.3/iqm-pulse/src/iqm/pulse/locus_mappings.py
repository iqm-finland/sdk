# Copyright 2024-2026 IQM
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
"""Locus mapping creation and caching.

A locus mapping defines all the possible loci a particular gate implementation can be applied on,
on a specific station. It is determined from the QPU topology and the available control channels
of the station.

For operations with zero :attr:`~Quantum.arity`, the locus mapping must consist of single-qubit loci,
and a multiqubit locus is valid if all its components are valid single-qubit loci.

The functions in this module build locus mappings for various :class:`.GateImplementation` subclasses,
and are shared by them. Each function caches the mappings it creates.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import cache
from typing import TYPE_CHECKING, TypeAlias

from exa.common.qcm_data.chip_topology import ChipTopology, _key_numeric
from iqm.pulse.playlist.channel import ChannelProperties, ProbeChannelProperties, _build_channel_properties

if TYPE_CHECKING:
    from exa.common.data.setting_node import SettingNode


Locus: TypeAlias = tuple[str, ...]
"""Sequence of QPU component names a quantum operation is acting on. The order may matter."""

LocusMapping: TypeAlias = dict[Locus, tuple[str, ...]]
"""Mapping from the possible loci of a particular GateImplementation subclass
to the QPU components used by the implementation on that locus, for a specific station."""
# TODO do we actually need/use the values of LocusMapping? It's created by the
# GateImplementation which already knows which components it wants to use.


def _key_locus(locus: Locus) -> tuple[int, ...]:
    """Sorting key for loci."""
    return tuple(_key_numeric(component) for component in locus)


@dataclass(frozen=True)
class StationProperties:
    """Static properties of a station."""

    qpu_topology: ChipTopology
    """QPU topology."""
    channels: dict[str, ChannelProperties]
    """Mapping of controller names to the configurations of their channels."""
    component_channels: dict[str, dict[str, str]]
    """Mapping from QPU component name to a mapping of ``('drive', 'flux', 'readout')``
    to the name of the control channel / controller responsible for that function of the component."""
    # NOTE: only contains fast flux controllers

    can_be_read_out: frozenset[str] = field(init=False)
    """QPU components that can be read out via a probeline.
    NOTE: Only probelines have a readout controller!"""
    has_drive: frozenset[str] = field(init=False)
    """QPU components that have a physical drive channel."""
    has_virtual_drive: frozenset[str] = field(init=False)
    """QPU components that have a physical or virtual drive channel, and hence can apply virtual phase corrections."""
    has_fast_flux: frozenset[str] = field(init=False)
    """QPU components that have a fast flux channel."""
    station_id: int = field(init=False)
    """Uniquely identifies the station properties, works as a hash key."""

    def __post_init__(self):
        """Compute additional properties."""
        topo = self.qpu_topology
        # TODO variant, mask_set_name and the rest act here as substitutes for a station identifier
        station_id = (topo.mask_set_name, topo.variant) + tuple(
            (c, tuple(channels.items())) for c, channels in self.component_channels.items()
        )
        super().__setattr__("station_id", hash(station_id))

        can_be_read_out = set()
        has_drive = set()
        has_virtual_drive = set()
        has_fast_flux = set()
        components = (
            topo.qubits_sorted + topo.computational_resonators_sorted + topo.couplers_sorted + topo.probe_lines_sorted
        )
        for c in components:
            if c in topo.component_to_probe_line:
                # connected to probeline => can be read out
                # we assume here that every probeline has a readout controller...
                can_be_read_out.add(c)
            ops = self.component_channels.get(c, {})
            if "drive" in ops:
                has_virtual_drive.add(c)
                if ".awg" in ops["drive"]:
                    has_drive.add(c)
            if "flux" in ops:
                has_fast_flux.add(c)

        super().__setattr__("can_be_read_out", frozenset(can_be_read_out))
        super().__setattr__("has_drive", frozenset(has_drive))
        super().__setattr__("has_virtual_drive", frozenset(has_virtual_drive))
        super().__setattr__("has_fast_flux", frozenset(has_fast_flux))

    def __hash__(self) -> int:
        # Assumes station_id will uniquely identify the station properties, and allows
        # effectively using it as a cache key for the locus mapping functions.
        return self.station_id

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StationProperties):
            return False
        return hash(self) == hash(other)

    @classmethod
    def _construct(
        cls,
        qpu_topology: ChipTopology,
        controllers: SettingNode,
        controller_mapping: dict[str, dict[str, str]],
    ) -> StationProperties:
        """Collect together the static properties of a station.

        Args:
            qpu_topology: The QPU topology.
            controllers: Station Control settings (i.e. the controller nodes).
            controller_mapping: Mapping from QPU component name to mapping of operation name to the
                name of the corresponding controller, from ``experiment.yml``.
                The ``"flux"`` operation can mean either fast or slow flux.

        Returns:
            Static station properties.

        """
        # TODO SW-1108: ``controllers`` and ``controller_mapping`` should be replaced with what the
        # ``GET /station/channel-properties`` endpoint gives,
        # which would make this function mostly unnecessary. Now SC converts channel properties into
        # readonly settings, and then CPC converts them back into channel properties which makes no sense.

        # We have currently several ways of getting channel props that use _build_channel_properties
        # (former get_channel_properties), which derives channel names from SettingNode names, and channel
        # props from their read-only settings plus center_frequency (HACK)
        # 1. This method, uses controller_mapping to filter SC controller nodes.
        # 2. Former get_channel_properties_from_station_settings, uses ChipTopology components
        #    and controller naming convention to filter SC controller nodes.
        # 3. get_default_channel_properties, filters the EXA tree controller branch using ChipTopology.

        # A better way would be to just take whatever controllers SC has, use controller
        # naming convention to deduce components and functions, and
        # handle the center_frequencies HACK separately.
        # Even better would be to remove read-only settings and use the ``GET /station/channel-properties`` endpoint.

        drive_controllers = {}
        fast_flux_controllers = {}
        readout_controllers = {}
        for component, ops in controller_mapping.items():
            for op, controller_name in ops.items():
                controller = controllers.subtrees[controller_name]
                if op == "readout":
                    readout_controllers[component] = controller
                elif op == "drive":
                    if (awg_controller := controller.subtrees.get("awg")) is not None:
                        drive_controllers[component] = awg_controller
                elif op == "flux":
                    if (awg_controller := controller.subtrees.get("awg")) is not None:
                        # flux controller has an awg subcontroller => has fast flux
                        fast_flux_controllers[component] = awg_controller
                elif op != "twpa":
                    raise ValueError(f"Unknown operation '{op}: {controller_name}' for {component}")

        channels, component_channels = _build_channel_properties(
            qpu_topology, drive_controllers, fast_flux_controllers, readout_controllers
        )
        return StationProperties(
            qpu_topology=qpu_topology,
            channels=channels,
            component_channels=component_channels,
        )

    def _hack_readout_channels(self, controllers: SettingNode) -> StationProperties:
        """Mutate :attr:`.ProbeChannelProperties.center_frequency` based on ``controllers``.

        HACK: ``center_frequency`` should not be a channel property since it can be changed
        using controller settings. Now we have to call this method whenever we construct a ScheduleBuilder
        to get the correct center frequencies in :attr:`channels`.

        Args:
            controllers: Controllers branch of the EXA settings tree.

        """
        # TODO SW-1108: remove
        for component, ops in self.component_channels.items():
            for op, controller in ops.items():
                if op == "readout":
                    if node := controllers.subtrees.get(component):
                        if node := node.subtrees.get("readout"):
                            center_freq = node.settings["center_frequency"].value
                            channel = self.channels[controller]
                            if isinstance(channel, ProbeChannelProperties):
                                # mutate a frozen dataclass :(
                                self.channels[controller] = replace(channel, center_frequency=center_freq)  # type: ignore[arg-type]
        return self


@cache
def one_qubit(sp: StationProperties) -> LocusMapping:
    """One-qubit loci."""
    # formerly DEFAULT_1QB_MAPPING
    return {(q,): (q,) for q in sp.qpu_topology.qubits_sorted}


@cache
def one_component(sp: StationProperties) -> LocusMapping:
    """One-component loci."""
    return {(c,): (c,) for c in sp.qpu_topology.all_components}


@cache
def one_component_channel(sp: StationProperties) -> LocusMapping:
    """One-component loci that have at least one channel."""
    return {(c,): (c,) for c in sp.qpu_topology.all_components if c in sp.component_channels}


@cache
def one_component_drive(sp: StationProperties) -> LocusMapping:
    """One-component loci with drive."""
    # formerly SINGLE_COMPONENTS_WITH_DRIVE_LOCUS_MAPPING
    topo = sp.qpu_topology
    components = topo.qubits_sorted + topo.couplers_sorted + topo.computational_resonators_sorted
    return {(c,): (c,) for c in components if c in sp.has_drive}


@cache
def one_component_virtual_drive(sp: StationProperties) -> LocusMapping:
    """One-component loci with virtual drive capability."""
    # formerly SINGLE_COMPONENTS_WITH_DRIVE_LOCUS_MAPPING
    topo = sp.qpu_topology
    components = topo.qubits_sorted + topo.couplers_sorted + topo.computational_resonators_sorted
    return {(c,): (c,) for c in components if c in sp.has_virtual_drive}


@cache
def one_component_flux(sp: StationProperties) -> LocusMapping:
    """One-component loci (qubits and couplers) with fast flux."""
    # formerly FLUX_COMPONENTS_MAPPING
    components = sp.qpu_topology.qubits_sorted + sp.qpu_topology.couplers_sorted
    return {(c,): (c,) for c in components if c in sp.has_fast_flux}


@cache
def one_qubit_flux(sp: StationProperties) -> LocusMapping:
    """One-qubit loci with fast flux."""
    return {(q,): (q,) for q in sp.qpu_topology.qubits_sorted if q in sp.has_fast_flux}


@cache
def one_qubit_readout(sp: StationProperties) -> LocusMapping:
    """One-qubit loci with readout."""
    # formerly SINGLE_COMPONENTS_WITH_READOUT_LOCUS_MAPPING
    return {(q,): (q,) for q in sp.qpu_topology.qubits_sorted if q in sp.can_be_read_out}


@cache
def one_qubit_readout_and_flux(sp: StationProperties) -> LocusMapping:
    """One-qubit loci with readout and fast flux."""
    return {(q,): (q,) for q in sp.qpu_topology.qubits_sorted if q in sp.can_be_read_out and q in sp.has_fast_flux}


@cache
def one_qubit_drive_readout(sp: StationProperties) -> LocusMapping:
    """One-qubit loci with physical drive and readout."""
    return {(q,): (q,) for q in sp.qpu_topology.qubits_sorted if q in sp.has_drive and q in sp.can_be_read_out}


@cache
def probe_lines(sp: StationProperties) -> LocusMapping:
    """One-component loci on probe lines."""
    # formerly PROBE_LINES_LOCUS_MAPPING
    return {(p,): (p,) for p in sp.qpu_topology.probe_lines_sorted}


@cache
def qubit_coupler(sp: StationProperties) -> LocusMapping:
    """Connected (qubit, coupler) pairs."""
    topo = sp.qpu_topology
    return {
        (qubit, coupler): (coupler,) for qubit in topo.qubits_sorted for coupler in topo.component_to_couplers[qubit]
    }


@cache
def two_component_coupler_flux(sp: StationProperties) -> LocusMapping:
    """Two-component loci (qubits + resonators), connected through a coupler with fast flux, symmetric."""
    # formerly DEFAULT_2QB_MAPPING
    topo = sp.qpu_topology
    computational_components = frozenset().union(topo.qubits, topo.computational_resonators)
    return {
        # comps is already sorted
        comps: (coupler,)
        for coupler, comps in topo.coupler_to_components.items()
        if len(comps) == 2 and set(comps) <= computational_components and coupler in sp.has_fast_flux
    }


@cache
def two_qubit_drive_coupler_flux(sp: StationProperties) -> LocusMapping:
    """Two-qubit loci, connected through a coupler with fast flux, both qubits have drive."""
    # provides former AC_STARK_PULSED_QUBITS_2QB_MAPPING
    topo = sp.qpu_topology
    return {
        # comps is already sorted
        comps: (coupler,)
        for coupler, comps in topo.coupler_to_components.items()
        if len(comps) == 2
        and set(comps) <= topo.qubits
        and coupler in sp.has_fast_flux
        and comps[0] in sp.has_drive
        and comps[1] in sp.has_drive
    }


@cache
def two_component_one_flux_coupler_flux(sp: StationProperties) -> LocusMapping:
    """Two-component loci connected through a coupler where the coupler and at least one component has fast flux.

    There must be two locus components connected by a coupler, where the first locus component has fast flux.
    If both locus components have fast flux, we include both locus orders.
    """
    # TODO do we need both locus orders?
    # formerly FLUX_PULSED_QUBITS_2QB_MAPPING
    mapping: LocusMapping = {}
    for coupler, comps in sp.qpu_topology.coupler_to_components.items():
        if len(comps) != 2:
            continue
        if coupler not in sp.has_fast_flux:
            continue
        component_has_flux = [c in sp.has_fast_flux for c in comps]
        if not any(component_has_flux):
            continue
        # one or both locus components have fast flux
        if component_has_flux[0]:
            # comps is already sorted
            mapping[comps] = (coupler,)
        if component_has_flux[1]:
            # reverse order
            mapping[comps[::-1]] = (coupler,)

    # we may have flipped some loci, so restore sorting for nicer presentation
    return dict(sorted(mapping.items(), key=lambda x: _key_locus(x[0])))


@cache
def two_qubit_flux_coupler_flux(sp: StationProperties) -> LocusMapping:
    """Two-qubit loci connected through a coupler, where the coupler and both qubits have fast flux.

    Only returns the sorted loci.
    """
    # formerly TRIPLE_FLUX_PULSED_QUBITS_2QB_MAPPING
    topo = sp.qpu_topology
    return {
        comps: (coupler,)  # comps is already sorted
        for coupler, comps in topo.coupler_to_components.items()
        if len(comps) == 2
        and set(comps) <= topo.qubits
        and set(comps) <= sp.has_fast_flux
        and coupler in sp.has_fast_flux
    }


@cache
def two_qubit_crc(sp: StationProperties) -> LocusMapping:
    """Two-qubit loci connected through coupler-resonator-coupler.

    It does not matter if qubits are connected through more than one resonator.
    The first resonator is added to the mapping for convenience.
    """
    # formerly QUBITS_CONNECTED_THROUGH_RESONATOR_MAPPING
    # TODO: could be more efficient, starting from the resonators
    # TODO: includes both locus orders, is this intentional?
    topo = sp.qpu_topology
    mapping: LocusMapping = {}
    for first_qubit in topo.qubits:
        for second_qubit in topo.qubits:
            if first_qubit == second_qubit:
                continue
            resonators = topo.get_all_common_resonators([first_qubit, second_qubit])
            if resonators:
                resonator = sorted(resonators)[0]
                first_coupler = topo.get_coupler_for(first_qubit, resonator)
                second_coupler = topo.get_coupler_for(second_qubit, resonator)
                mapping[(first_qubit, second_qubit)] = (first_coupler, resonator, second_coupler)
    return mapping


@cache
def single_qubit_or_resonator_without_probe(sp: StationProperties) -> LocusMapping:
    """Defined for components that have no probe line and that are qubits or resonators."""
    topo = sp.qpu_topology
    return {
        (c,): (c,)
        for c in topo.qubits_sorted + topo.computational_resonators_sorted
        if c not in topo.component_to_probe_line
    }
