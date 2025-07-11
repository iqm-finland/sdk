# Copyright 2024 IQM
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

"""Pydantic models for CHAD."""

from collections.abc import Collection
from functools import cached_property
import re
from typing import Any

from pydantic import Field, field_validator

from exa.common.qcm_data.immutable_base_model import ImmutableBaseModel


def _natural_sort_key(name: str) -> tuple[int | str | Any, ...]:
    return tuple(int(item) if item.isdigit() else item.lower() for item in re.split(r"(\d+)", name))


class Component(ImmutableBaseModel):
    name: str
    connections: tuple[str, ...] = ()

    def __lt__(self, other):
        return _natural_sort_key(self.name) < _natural_sort_key(other.name)

    @field_validator("connections", mode="before")
    @classmethod
    def sort(cls, connections: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(sorted(connections, key=_natural_sort_key))


class Qubit(Component):
    pass


class Coupler(Component):
    pass


class ProbeLine(Component):
    pass


class Launcher(Component):
    pin: str
    function: str


class ComputationalResonator(Component):
    pass


class Components(ImmutableBaseModel):
    # Alias list of components to plurals rather than singulars to follow normal naming conventions
    qubits: tuple[Qubit, ...] = Field((), alias="qubit")
    couplers: tuple[Coupler, ...] = Field((), alias="tunable_coupler")
    probe_lines: tuple[ProbeLine, ...] = Field((), alias="probe_line")
    launchers: tuple[Launcher, ...] = Field((), alias="launcher")
    computational_resonators: tuple[ComputationalResonator, ...] = Field((), alias="computational_resonator")

    @field_validator("*")
    @classmethod
    def sort_components(cls, components: tuple[Component, ...]) -> tuple[Component, ...]:
        return tuple(sorted(components))

    @cached_property
    def all(self) -> dict[str, Component]:
        components: tuple[Qubit | Coupler | ProbeLine | Launcher | ComputationalResonator, ...] = (
            self.qubits + self.couplers + self.probe_lines + self.launchers + self.computational_resonators
        )
        return {component.name: component for component in components}


class CHAD(ImmutableBaseModel):
    mask_set_name: str
    variant: str
    components: Components

    def __init__(self, **kwargs):
        kwargs["components"] = kwargs.pop("content")["components"]
        super().__init__(**kwargs)

    def get_component(self, component_name: str) -> Component:
        """Get component by component name."""
        return self.components.all[component_name]

    @cached_property
    def qubit_names(self) -> list[str]:
        """Names of all the qubits declared in CHAD data."""
        return self._get_component_names(self.components.qubits)

    @cached_property
    def coupler_names(self) -> list[str]:
        """Names of all the couplers declared in CHAD data."""
        return self._get_component_names(self.components.couplers)

    @cached_property
    def probe_line_names(self) -> list[str]:
        """Names of all the probe lines declared in CHAD data."""
        return self._get_component_names(self.components.probe_lines)

    @cached_property
    def computational_resonator_names(self) -> list[str]:
        """Names of all the computational resonators declared in CHAD data."""
        return self._get_component_names(self.components.computational_resonators)

    # TODO: Consider more generic "filter_components(components, component_type)" approach
    def filter_qubit_components(self, component_names: Collection[str]) -> list[str]:
        """Filter qubit components from the input components."""
        self._validate_input(component_names)
        return [component_name for component_name in component_names if component_name in self.qubit_names]

    def get_probe_line_names_for(self, component_names: Collection[str]) -> list[str]:
        """Get probe lines for given qubits in CHAD data."""
        self._validate_input(component_names)
        qubits = (qubit for qubit in self.components.qubits if qubit.name in component_names)

        probe_line_names: set[str] = set()
        for qubit in qubits:
            for probe_line in self.components.probe_lines:
                if probe_line.name in qubit.connections:
                    probe_line_names.add(probe_line.name)

        return list(probe_line_names)

    def group_components_per_default_operations(
        self,
        component_names: Collection[str],
    ) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        """Groups given qubits/couplers based on their defined default operations.

        The default operations that can be deducted from a CHAD are `readout`, `drive`, and `flux`.

        Args:
            component_names: The component names to which to do the grouping.
              Other components present in the CHAD will not be included in the returned data.

        Returns:
            - Tuple of qubits and couplers mapped to their connected default operations.
                The data is in the form of a dict with the keys being `readout`, `drive`, and `flux`,
                and the values the list of component names having that particular operation.

        """
        self._validate_input(component_names)

        def format_output(_types_and_operations: dict) -> tuple[dict, dict]:
            _qubits = {key: list(value) for key, value in _types_and_operations["qubit"].items()}
            _couplers = {key: list(value) for key, value in _types_and_operations["coupler"].items()}
            return _qubits, _couplers

        types_and_operations: dict[str, dict[str, set[str]]] = {
            "qubit": {"readout": set(), "drive": set(), "flux": set()},
            "coupler": {"readout": set(), "drive": set(), "flux": set()},
        }

        launchers = {launcher.name: launcher for launcher in self.components.launchers}

        component: Qubit | Coupler
        for component in [*self.components.qubits, *self.components.couplers]:  # type:ignore[assignment]
            component_type = "qubit" if isinstance(component, Qubit) else "coupler"
            if component.name in component_names:
                for connection in component.connections:
                    if connection in self.probe_line_names:
                        types_and_operations[component_type]["readout"].add(component.name)
                    elif connection in launchers and launchers[connection].function == "drive":
                        types_and_operations[component_type]["drive"].add(component.name)
                    elif connection in launchers and launchers[connection].function == "flux":
                        types_and_operations[component_type]["flux"].add(component.name)
        return format_output(types_and_operations)

    def get_coupler_mapping_for(self, component_names: Collection[str]) -> dict[str, list[str]]:
        """Get the coupler-component mapping for the couplers that connects to at least two components
             in the given qubits.

        Args:
            component_names: The qubit names. May contain any number of qubits.

        Returns:
            Coupler names mapped to the components they connect.

        Raises:
            - ValueError: If the provided qubit name list contains duplicates.

        """
        self._validate_input(component_names)

        component_names = list(component_names) + self.computational_resonator_names
        component_mapping: dict[str, list[str]] = {}
        for coupler in self.components.couplers:
            connections = [
                component_name for component_name in component_names if component_name in coupler.connections
            ]
            if len(connections) > 1:
                component_mapping[coupler.name] = connections
        return component_mapping

    def get_probe_line_mapping_for(self, component_names: Collection[str]) -> dict[str, list[str]]:
        """Get the probe line-component mapping.

        Args:
            component_names: The qubit names. May contain any number of qubits.

        """
        self._validate_input(component_names)

        component_mapping: dict[str, list[str]] = {}
        for probe_line in self.components.probe_lines:
            connections = [
                component_name for component_name in component_names if component_name in probe_line.connections
            ]
            if connections:
                component_mapping[probe_line.name] = connections
        return component_mapping

    def get_common_coupler_for(self, first_component: str, second_component: str) -> str:
        """Convenience method for getting the name of a coupler connecting a pair of components.

        Args:
            first_component: The name of the first component.
            second_component: The name of the second component.
                The order of qubits does not matter, i.e. the `first_qubit` and `second_qubit`
                arguments are interchangeable.

        Returns:
            - The name of the coupler that connects the inputted components.

        Raises:
            - ValueError: If there were no couplers or more than one coupler connecting the component pair (the latter
                should not be possible in a realistic chip).

        """
        coupler_mapping = self.get_coupler_mapping_for([first_component, second_component])
        common_couplers = [
            coupler
            for coupler, components in coupler_mapping.items()
            if all(component in components for component in [first_component, second_component])
        ]
        if len(common_couplers) != 1:
            raise ValueError(
                f"No common coupler was found for {first_component} and {second_component} "
                f"or there were multiple couplers. "
                f"Check your Chip Architecture Definition or the inputted component names.\n"
                f"Found common couplers: {common_couplers}"
            )
        return common_couplers[0]

    @staticmethod
    def _get_component_names(components: tuple[Component, ...]) -> list[str]:
        return [component.name for component in components]

    def _validate_input(self, component_names: Collection[str]) -> None:
        if len(set(component_names)) < len(component_names):
            raise ValueError(f"The provided component names {component_names} contain duplicates.")
        chad_component_names = self.components.all.keys()
        for component_name in component_names:
            if component_name not in chad_component_names:
                raise ValueError(f"The provided component name '{component_name}' doesn't exist in the CHAD.")
        # TODO: Validate only certain type of components are given
