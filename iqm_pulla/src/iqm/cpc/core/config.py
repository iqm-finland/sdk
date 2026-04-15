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
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
import importlib
import os
import sys
from typing import Any, TypeAlias, no_type_check
import uuid

from exa.common.control.sweep.option import SweepOptions
from exa.common.control.sweep.option.option_converter import convert_to_options
from exa.common.data.setting_node import SettingNode
from exa.common.qcm_data.chip_topology import ChipTopology
from iqm.pulse.builder import build_quantum_ops
from iqm.pulse.gate_implementation import GateImplementation
from iqm.pulse.gates import register_implementation, register_operation
from iqm.pulse.quantum_ops import QuantumOp
from iqm.station_control.client.station_control import StationControlClient

Components: TypeAlias = list[str] | list[tuple[str, ...]] | list[list[tuple[str, ...]]]
"""Same as :class:`ComponentGrouping` but using a normal :class:`list`."""

MALFORMED_COMPONENTS_ERROR = ValueError(
    "Provided components are malformed. The components should be of "
    "the form `list[str]` or `list[tuple[str, ...]]` or `list[list[tuple[str, ...]]]`."
)


class ComponentGroupingMode(Enum):
    """Types of structures for the components given as run arguments."""

    LIST = "list"
    GROUP = "group"
    COLOUR_GROUP = "colour_group"


class ComponentGrouping(list):
    """Utility class for handling the ``run`` argument ``components``.

    ComponentGrouping is a list of one of the following types:

    1. ``str`` (QPU component names)
    2. ``tuple[str, ...]`` (component groups)
    3. ``list[tuple[str, ...]]`` (group of parallelizable units, each such group can be executed in parallel).

    All the normal list operations and comprehensions work with ComponentGrouping. In addition,
    the class provides utility methods such as flattening the structure and checking whether components or groups
    are found within it. The attribute :attr:`grouping_mode` can be accessed to find out what kind of contents
    (one of the three above) the list contains.
    """

    def __init__(self, components: Components | None, validate: bool = True):
        super().__init__(components or [])
        if validate:
            self._validate_components()
        self.grouping_mode = self._get_grouping_mode()

    def to_list(self) -> Components:
        """Returns the contents as a normal list."""
        return list(self)

    @no_type_check
    def to_json_serializable(self) -> list[str] | list[list[str]] | list[list[list[str]]]:
        """Returns the contents as a JSON serializable format, using lists instead of tuples."""
        component_groups = self.to_list()
        match self.grouping_mode:
            case ComponentGroupingMode.GROUP:
                component_groups = [list(group) for group in component_groups]
            case ComponentGroupingMode.COLOUR_GROUP:
                component_groups = [[list(group) for group in colour_group] for colour_group in component_groups]
        return component_groups

    def copy(self):  # noqa: ANN201
        """Copies the contents into a new instance."""
        return ComponentGrouping(self.to_list(), False)

    def flatten(self) -> list[str]:
        """Flattens the contents into a list of unique strings.

        Any duplicate entries are removed such that only the first occurrence is kept. Otherwise the
        order is preserved.

        Returns:
            Unique component names as a flat list.

        """
        if self.grouping_mode is None or self.grouping_mode == ComponentGroupingMode.LIST:
            return list(dict.fromkeys(self).keys())
        flattened = []

        for elem in self:
            for sub_elem in elem:
                if self.grouping_mode == ComponentGroupingMode.GROUP and sub_elem not in flattened:
                    flattened.append(sub_elem)
                elif self.grouping_mode == ComponentGroupingMode.COLOUR_GROUP:
                    for component in sub_elem:
                        if component not in flattened:
                            flattened.append(component)
        return flattened

    def contains(self, elem: str | tuple[str, ...] | list[tuple[str, ...]]) -> bool:
        """Checks if a component, a component group or a colour group is contained.

        The behaviour depends on the `self.grouping_mode` and on the type of the searched element.

        1. If `self.grouping_mode == ComponentGroupingMode.LIST`:
        - Search for a string returns True if the string is contained within.
        - Search for a group returns True if all the elements of the group are contained within.
        - Search for a colour group returns True if all the groups in it are contained within (similarly as above).

        2. If `self.grouping_mode == ComponentGroupingMode.GROUP`:
        - Search for a string returns True if the string is contained within any of the groups.
        - Search for a group returns True if the group is contained within.
        - Search for a colour group returns True if all the groups in it are contained within.

        3. If `self.grouping_mode == ComponentGroupingMode.COLOUR_GROUP`:
        - Search for a string returns True if the string is contained within any of the groups in the colour groups.
        - Search for a group returns True if the group is contained within any of the colour groups.
        - Search for a colour group returns True if the colour group is contained within.

        Returns:
            Whether the element was found or not.

        """
        if not self:
            return False
        if self.grouping_mode == ComponentGroupingMode.LIST:
            return self._contains_for_list(elem)
        if self.grouping_mode == ComponentGroupingMode.GROUP:
            return self._contains_for_group(elem)
        return self._contains_for_colour_group(elem)

    def _contains_for_list(self, elem: str | tuple[str, ...] | list[tuple[str, ...]]) -> bool:
        """Is an element contained in a list."""
        if isinstance(elem, tuple):
            return set(elem).issubset(set(self))
        if isinstance(elem, list):
            set_data = set(self)
            for group in elem:
                if not set(group).issubset(set_data):
                    return False
            return True
        return elem in self

    def _contains_for_group(self, elem: str | tuple[str, ...] | list[tuple[str, ...]]) -> bool:
        """Is an element contained in a grouped list."""
        if isinstance(elem, str):
            for group in self:
                if elem in group:
                    return True
            return False
        if isinstance(elem, list):
            for group in elem:
                if group not in self:
                    return False
            return True
        return elem in self

    def _contains_for_colour_group(self, elem: str | tuple[str, ...] | list[tuple[str, ...]]) -> bool:
        """Is an element contained in a colour grouped list."""
        if isinstance(elem, str):
            for color in self:
                for group in color:
                    if elem in group:
                        return True
            return False
        if isinstance(elem, tuple):
            for color in self:
                if elem in color:
                    return True
            return False
        return elem in self

    def limit_by_components(self, component_list: list[str] | list[tuple[str, ...]] | None) -> ComponentGrouping:
        """Limits the component groups by a list of component names or component groups while preserving the order.

        If `self.grouping_mode == ComponentGroupingMode.LIST`, and limiting by flat lists,
        will return a new `ComponentGrouping` such that all components not belonging to the given `component_list`
        are removed. Lists cannot be limited by groups, an error is thrown instead.

        If `self.grouping_mode == ComponentGroupingMode.GROUP` or
        `self.grouping_mode == ComponentGroupingMode.COLOUR_GROUP`, and limiting with flat lists,
        will return a new `ComponentGrouping` such that all groups that are not subsets of the `component_list`
        are removed. If limiting by groups, will return only the groups that belong to the limiting groups.
        Completely emptied colour groups will be removed.

        Returns:
            The limited component grouping.

        Raises:
            ValueError: When trying to limit flat lists with groups.

        """
        if component_list is None:
            return ComponentGrouping(self, False)
        if not component_list:
            empty: list[str] = []
            grouping = ComponentGrouping(empty, False)
            grouping.grouping_mode = self.grouping_mode
            return grouping

        if isinstance(component_list[0], str):
            return self._limit_by_list(component_list)

        return self._limit_by_groups(component_list)

    def _limit_by_list(self, component_list: list[str] | list[tuple[str, ...]]) -> ComponentGrouping:
        """Limits the component groups by a list of component names while preserving the order."""
        component_set = set(component_list)
        limited_components = []
        for element in self:
            if self.grouping_mode == ComponentGroupingMode.LIST and element in component_set:
                limited_components.append(element)
            elif self.grouping_mode == ComponentGroupingMode.GROUP and set(element).issubset(component_set):
                limited_components.append(element)
            elif self.grouping_mode == ComponentGroupingMode.COLOUR_GROUP:
                limited_element = [group for group in element if set(group).issubset(component_set)]
                if len(limited_element) > 0:
                    limited_components.append(limited_element)
        return ComponentGrouping(limited_components, False)

    def _limit_by_groups(self, component_list: list[str] | list[tuple[str, ...]]) -> ComponentGrouping:
        """Limits the component groups by a list of component groups while preserving the order."""
        if self.grouping_mode == ComponentGroupingMode.LIST:
            raise ValueError("Cannot limit a flat list of components by a list of component groups.")
        component_set = set(component_list)
        limited_components = []
        for element in self:
            if self.grouping_mode == ComponentGroupingMode.GROUP and element in component_set:
                limited_components.append(element)
            elif self.grouping_mode == ComponentGroupingMode.COLOUR_GROUP:
                limited_element = [group for group in element if group in component_set]
                if len(limited_element) > 0:
                    limited_components.append(limited_element)

        return ComponentGrouping(limited_components, False)

    def _get_grouping_mode(self) -> ComponentGroupingMode | None:
        if not self:
            return None
        if isinstance(self[0], str):
            return ComponentGroupingMode.LIST
        if isinstance(self[0], tuple):
            return ComponentGroupingMode.GROUP
        return ComponentGroupingMode.COLOUR_GROUP

    def _validate_components(self) -> None:
        if len(self) == 0:
            return
        if isinstance(self[0], str):
            self._validate_flat(self)
        elif isinstance(self[0], tuple):
            self._validate_groups(self)
        elif isinstance(self[0], list):
            self._validate_colour_groups(self)
        else:
            raise MALFORMED_COMPONENTS_ERROR

    @staticmethod
    def _validate_flat(flat_components):  # noqa: ANN001
        if sum(isinstance(c, str) for c in flat_components) != len(flat_components):
            raise MALFORMED_COMPONENTS_ERROR

    def _validate_groups(self, groups):  # noqa: ANN001, ANN202
        for group in groups:
            if not isinstance(group, tuple):
                raise MALFORMED_COMPONENTS_ERROR
            self._validate_flat(group)

    def _validate_colour_groups(self, colour_groups):  # noqa: ANN001, ANN202
        for colour in colour_groups:
            if not isinstance(colour, list):
                raise MALFORMED_COMPONENTS_ERROR
            self._validate_groups(colour)


class SweepDefaultsConfiguration:
    """Default sweep configuration.

    See :func:`~exa.common.control.sweep.option.option_converter.convert_to_options` for the rest of the options.

    Args:
        config: Sweep configuration tree.

    """

    def __init__(self, config: dict[str, Any]):
        self.name: str = config["name"]
        """Sweep name."""
        self.type: str = config["type"]
        """Sweep type."""
        self.parameter: str = config["parameter"]
        """Name of the parameter being swept."""
        self.options: SweepOptions = convert_to_options(config)
        """Rest of the sweep options."""


class ExperimentDefaultsConfiguration:
    """Configuration for a specific :class:`.Experiment` subclass.

    Args:
        config: Configuration tree for the experiment.

    """

    def __init__(self, config: dict[str, Any]):
        self.controllers: dict[str, Any] = config.get("controllers", {})
        """Default controller settings."""
        self.sweeps: list[SweepDefaultsConfiguration] = [
            SweepDefaultsConfiguration(config) for config in config.get("sweeps", [])
        ]
        """Default sweeps."""


@dataclass
class ComponentGroup:
    """Collection of chip components, and the station control Controllers meant to control them."""

    member_names: list[str]
    """Names of the components that belong to this group."""
    controllers: dict[str, str]
    """Mapping from control operation to the name of the controller."""

    @staticmethod
    def from_dict(config: dict[str, Any]) -> ComponentGroup:
        """Construct an instance from a dictionary."""
        controllers = config.copy()
        member_names = controllers.pop("member_names", [])
        return ComponentGroup(member_names=member_names, controllers=controllers)


def set_calibration_data_for_gate_implementation(
    gates: dict[str, Any], gate_name: str, impl_name: str, calib_data: dict[tuple[str, ...], Any]
) -> None:
    """Add gate and/or implementation name and the corresponding calibration data to `gates`"""
    if gate_name not in gates:
        gates[gate_name] = {}
    if impl_name not in gates[gate_name]:
        gates[gate_name][impl_name] = calib_data
    else:
        for locus, locus_data in calib_data.items():
            gates[gate_name][impl_name][locus] = locus_data


def set_gate_implementation_as_default(
    operations: dict[str, QuantumOp],
    gate_name: str,
    impl_name: str,
) -> None:
    """Sets the given implementation as the default implementation for the gate"""
    if gate_name not in operations:
        raise ValueError(f"Gate named {gate_name} is not found in the defined quantum operations.")
    if impl_name not in operations[gate_name].implementations:
        raise ValueError(f"Implementation named {impl_name} is not found for gate named {gate_name}.")
    operations[gate_name].set_default_implementation(impl_name)


def _is_custom_implementation(name: str) -> bool:
    """True iff ``name`` refers to a custom GateImplementation class to be imported from a file."""
    # Custom gate implementations are of the form `my/module.py::MyClass`.
    return "." in name


def _import_from_file(name: str) -> type[GateImplementation]:
    """Attempt to import a GateImplementation class from a source file."""
    filename, separator, class_name = name.rpartition("::")
    if separator:
        filename = os.path.expanduser(os.path.normpath(filename))
        spec = importlib.util.spec_from_file_location("module", filename)
        if spec:
            module = importlib.util.module_from_spec(spec)
            sys.modules["module"] = module
            if spec.loader:
                spec.loader.exec_module(module)

                return getattr(module, class_name)

    raise ValueError(
        f"The custom provided implementation {name} can't be imported. Use a path of the form `/path/to/file.py::Class`"
    )


def get_conventional_control_mapping(
    chip_topology: ChipTopology, station_control_settings: SettingNode | None = None
) -> dict[str, dict[str, str]]:
    """Add conventional controller mapping if ``controller_mapping`` is ``"qpu_convention"``.

    If the configuration is set to follow the standard QPU controller naming convention,
    extend the :attr:`.controllers` and :attr:`.component_groups` based on the convention.
    Components given explicitly have priority.

    The convention is that components map their functions to controller names with the following pattern:

    For qubits:

    * ``drive: <qubit_name>__drive``
    * ``flux: <qubit_name>__flux``

    For couplers:

    * ``flux: <coupler_name>__flux``
    * ``readout: <coupler_name>__readout`` (if connected)

    For probelines:

    * ``readout: <probeline_name>__readout``
    * ``twpa: <probeline_name>__twpa``

    Args:
        chip_topology: The chip topology for the QPU.
        station_control_settings: The station control settings. If provided, will validate the control mappings against
            them such that only the controllers that exist in the settings are included.

    Returns:
         The controller mapping, a dictionary from QPU component names to their operations and associated controller
            names.

    """

    def _filter_with_sc_settings(component_mapping: dict[str, str]) -> dict[str, str]:
        if station_control_settings is None:
            return component_mapping
        return {
            operation: controller_name
            for operation, controller_name in component_mapping.items()
            if station_control_settings.find_by_name(controller_name)
        }

    controller_mapping = {}
    for qubit in chip_topology.qubits_sorted:
        controller_mapping[qubit] = _filter_with_sc_settings(
            {
                "drive": f"{qubit}__drive",
                "flux": f"{qubit}__flux",
            }
        )
    for coupler in chip_topology.couplers_sorted:
        controller_mapping[coupler] = _filter_with_sc_settings(
            {
                "flux": f"{coupler}__flux",
            }
        )
    for probeline in chip_topology.probe_lines_sorted:
        controller_mapping[probeline] = _filter_with_sc_settings(
            {
                "readout": f"{probeline}__readout",
                "twpa": f"{probeline}__twpa",
            }
        )
    for resonator in chip_topology.computational_resonators_sorted:
        controller_mapping[resonator] = {}
    return controller_mapping


class ExperimentConfiguration:
    """General configuration for :class:`.Experiment` objects.

    * Top-level keys ``'user'``, ``'dut_label'``, ``'settings'``, ``'components'``, and ``'component_groups'``
        are reserved for the attributes below, and are optional.
    * Any other top-level keys generate :class:`ExperimentDefaultsConfiguration` instances for specific
        :class:`.Experiment` classes.

    Args:
        config: Top-level experiment configuration tree.

    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.id = uuid.uuid4()
        """Unique identifier for this ExperimentConfiguration."""
        self.user: str = config.get("user", "")
        """Username."""
        self.dut_label: str = config.get("dut_label", "")
        """DUT label."""
        self.components: dict[str, Any] = config.get("components", {})
        """Mapping from chip component names to the station control controllers they use."""
        self.component_groups: dict[str, ComponentGroup] = {
            key: ComponentGroup.from_dict(value) for key, value in config.get("component_groups", {}).items()
        }
        """Mapping from component group names to their definitions."""
        self.controllers: dict[str, Any] = config.get("controllers", {})
        """Controller settings."""

        gate_definitions = config.get("gate_definitions", {})

        for op_name, definition in gate_definitions.items():
            # bit of a hack, here we only resolve custom gate implementations, exposed class names are handled later
            new_implementations: dict[str, str | type[GateImplementation]] = {}
            for impl_name, impl_class_name in definition.pop("implementations", {}).items():
                if _is_custom_implementation(impl_class_name):
                    # Custom gate implementations are of the form `my/module.py::MyClass`.
                    new_implementations[impl_name] = _import_from_file(impl_class_name)
                else:
                    new_implementations[impl_name] = impl_class_name
            definition["implementations"] = new_implementations

        self.gate_definitions = build_quantum_ops(gate_definitions)
        """Defined quantum operations"""

        gates_used = config.get("gates_used")
        # limit gate definitions by gates_used
        if gates_used is not None:
            self.gate_definitions = {k: v for k, v in self.gate_definitions.items() if k in gates_used}

        self.gates: dict[str, Any] = config.get("gates", {}).copy()
        """Gate implementation calibration data."""
        self._controller_mapping = config.get("controller_mapping")

        self.characterization: dict[str, Any] = config.get("characterization", {})
        """Experiment characterization data."""
        self.stages: dict[str, Any] = config.get("stages", {})
        """The stage settings data."""

        experiments = set(config.keys()).difference(
            {
                "user",
                "dut_label",
                "controller_mapping",
                "components",
                "component_groups",
                "gate_definitions",
                "gates_used",
                "gates",
                "characterization",
                "controllers",
                "stages",
            }
        )
        for experiment_name in experiments:
            setattr(self, experiment_name, ExperimentDefaultsConfiguration(config[experiment_name]))

    def add_conventional_controller_mapping(self, station_control: StationControlClient) -> None:
        """Add conventional controller mapping if ``controller_mapping`` is ``"qpu_convention"``.

        If the configuration is set to follow the standard QPU controller naming convention,
        extend the :attr:`.controllers` and :attr:`.component_groups` based on the convention.
        Components given explicitly have priority.

        The convention is that components map their functions to controller names with the following pattern:

        For qubits:

        * ``drive: <qubit_name>__drive``
        * ``flux: <qubit_name>__flux``

        For couplers:

        * ``flux: <coupler_name>__flux``
        * ``readout: <coupler_name>__readout`` (if connected)

        For probelines:

        * ``readout: <probeline_name>__readout``
        * ``twpa: <probeline_name>__twpa``

        Args:
            station_control: Station instance.

        """
        if self._controller_mapping != "qpu_convention":
            return

        chip_design_record = station_control.get_chip_design_record(self.dut_label)
        chip_topology = ChipTopology.from_chip_design_record(chip_design_record)
        controller_mapping = get_conventional_control_mapping(chip_topology)
        self.components = controller_mapping | self.components

    def validate_controllers(self, station_control: StationControlClient) -> None:
        """Validate controller maps against controller names in `station_control`.

        Args:
            station_control: Station instance.

        Raises:
            ValueError: if the station does not have a controller required by the naming convention.

        """
        if self._controller_mapping != "qpu_convention":
            return
        required = {controller_name for mapping in self.components.values() for controller_name in mapping.values()}
        required |= {
            controller_name
            for group in self.component_groups.values()
            for controller_name in group.controllers.values()
        }
        missing = required.difference(station_control.get_settings().children)
        if missing:
            raise ValueError(
                f"These controllers are required because controller_mapping = 'qpu_convention', "
                f"but they do not exist in the station: {missing}."
            )

    @property
    def component_names(self) -> list[str]:
        """Property to return list of component names in experiment configuration."""
        return list(self.components)

    def filter_components(
        self, *filter_groups: str | Callable[..., bool] | list[str | Callable[..., bool]]
    ) -> list[str] | tuple[list[str], ...]:
        """Helper method for filtering components based on what operations they have defined.

        The user can define several groups of operation requirements. Requirement groups can contain
        requirements that are either of the type `str` or any function of the type
        `Callable[str, bool]`. If a `str` requirement is given, it will be interpreted as a check whether
        this particular operation name is defined for a given component.

        For example, the group `[lambda c: c != 'readout, 'flux']` resolves to the components
        that do not have a `readout` operation but have a `flux` operation.

        Requirement groups that contain just one requirement (of either type) can be given as a plain value instead
        of wrapping it into a list.

        Args:
            *filter_groups: requirement groups.

        Returns:
            - Tuple of filtered components corresponding to the provided filter_groups.

        """

        def check_operations(condition: str | Callable, operations: list[str]) -> bool:
            """Check that a condition is met by all operations in a list.

            If the condition is a string, checks that this string is included in the list.
            """
            if callable(condition):
                return all(condition(operation) for operation in operations)
            return condition in operations

        if len(filter_groups) == 0:
            raise ValueError("No filtering conditions were provided.")

        # Need to use [[] for _ in range(N)] to avoid the list containing the same list object N times
        # https://stackoverflow.com/questions/33990673/how-to-create-a-list-of-empty-lists
        filtered_components: list[list[str]] = [[] for _ in range(len(filter_groups))]
        for component in self.components.keys():
            for index, group in enumerate(filter_groups):
                if isinstance(group, str) or callable(group):
                    group = [group]  # noqa: PLW2901
                if all(check_operations(cond, self.components[component]) for cond in group):
                    filtered_components[index].append(component)
        if len(filtered_components) == 1:
            return filtered_components[0]
        return tuple(filtered_components)

    def register_gate_implementation(  # noqa: ANN201
        self,
        gate_name: str,
        impl_name: str,
        impl_class: type[GateImplementation],
        *,
        set_as_default: bool = False,
        quantum_op: QuantumOp | None = None,
    ):
        """Register a new gate implementation for this session. See :func:`.register_implementation`."""
        if quantum_op is not None:
            register_operation(self.gate_definitions, quantum_op)

        register_implementation(
            self.gate_definitions,
            gate_name,
            impl_name,
            impl_class,
            set_as_default=set_as_default,
        )

    def set_calibration_data_for_gate_implementation(
        self, gate_name: str, impl_name: str, calib_data: dict[tuple[str, ...], Any]
    ) -> None:
        """Set calibration data for this session."""
        set_calibration_data_for_gate_implementation(self.gates, gate_name, impl_name, calib_data)

    def set_gate_implementation_as_default(self, gate_name: str, impl_name: str) -> None:
        """Set this gate implementation as default for this session."""
        set_gate_implementation_as_default(self.gate_definitions, gate_name, impl_name)
