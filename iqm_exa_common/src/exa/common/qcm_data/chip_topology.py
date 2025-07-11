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

# mypy: ignore-errors

"""Chip topology class for parsing CHAD and other QPU related data into human-usable form."""

from __future__ import annotations

from collections.abc import Collection, Iterable
import itertools
import re
from typing import TypeVar

from exa.common.qcm_data.chad_model import CHAD

Locus = tuple[str, ...] | frozenset[str]

DEFAULT_1QB_MAPPING = "qubits"
DEFAULT_2QB_MAPPING = "connected_qubits"

ComponentMap = TypeVar("ComponentMap", bound=dict[str, Collection[str]])


def _get_numeric_id(name: str) -> int:
    """Sorting key for component names."""
    match = re.search(r"(\d+)", name)
    return int(match.group(1)) if match else 0


def _get_coupler_numeric_ids(name: str) -> tuple[int, ...]:
    """Sorting key for coupler names.

    Supports e.g. the following patterns: "TC-1-2", "TC1"
    """
    return tuple(int(m) for m in re.findall(r"\d+", name))


def sort_components(components: Iterable[str]) -> list[str]:
    """Sort the given components in a human-readable way."""
    return sorted(components, key=lambda x: (re.sub(r"[^a-zA-Z]", "", x), -_get_numeric_id(x), x), reverse=True)


def sort_couplers(couplers: Iterable[str]) -> list[str]:
    """Sort the given couplers in a human-readable way."""
    return sorted(couplers, key=_get_coupler_numeric_ids)


class ChipTopology:
    """Topology information for a chip (typically a QPU).

    Can represent the information found in a CHAD, as well as locus mappings for gates.

    Args:
        qubits: names of the qubits.
        computational_resonators: names of the computational resonators.
        couplers: mapping from coupler name to names of chip components it connects to.
        probe_lines: mapping from probe line name to names of chip components it connects to.
        variant: identifier of the QPU design variant.

    """

    def __init__(
        self,
        qubits: Iterable[str],
        computational_resonators: Iterable[str],
        couplers: dict[str, Iterable[str]],
        probe_lines: dict[str, Iterable[str]],
        variant: str = "",
    ):
        self.variant = variant
        self.qubits = frozenset(qubits)
        self.qubits_sorted = tuple(sort_components(self.qubits))

        self.computational_resonators = frozenset(computational_resonators)
        """Computational resonators on the chip, in any order."""
        self.computational_resonators_sorted = tuple(sort_components(self.computational_resonators))
        """Computational resonators on the chip, sorted."""

        data_components = self.qubits | self.computational_resonators
        if diff := set(itertools.chain.from_iterable(couplers.values())) - data_components:
            raise ValueError(f"Couplers connect to unknown components: {diff}")
        if diff := set(itertools.chain.from_iterable(probe_lines.values())) - (data_components | frozenset(couplers)):
            raise ValueError(f"Probe lines connect to unknown components: {diff}")

        self.couplers: frozenset[str] = frozenset(couplers)
        """Tunable couplers on the chip, in any order."""
        self.couplers_sorted: tuple[str, ...] = tuple(sort_couplers(self.couplers))
        """Tunable couplers on the chip, sorted by numerical IDs."""

        self.probe_lines: frozenset[str] = frozenset(probe_lines)
        """Probe lines on the chip, in any order."""
        self.probe_lines_sorted: tuple[str, ...] = tuple(sorted(self.probe_lines))
        """Probe lines on the chip, sorted."""

        self.all_components = data_components | self.couplers | self.probe_lines
        """All components on the chip."""

        self.coupler_to_components: dict[str, tuple[str, ...]] = {
            coupler: tuple(sort_components(components)) for coupler, components in couplers.items()
        }
        """Map from each coupler to all other components it connects to. The values are sorted."""
        component_to_couplers: dict = {}
        for coupler, components in couplers.items():
            for c in components:
                component_to_couplers.setdefault(c, set()).add(coupler)
        self.component_to_couplers: dict[str, frozenset] = {
            c: frozenset(couplers) for c, couplers in component_to_couplers.items()
        }
        """Map from each component to all couplers connected to it."""
        self.probe_line_to_components: dict[str, tuple[str, ...]] = {
            pl: tuple(sort_components(components)) for pl, components in probe_lines.items()
        }
        """Map from each probe line to all components it connects to."""
        # NOTE: assumes just one pl per component
        self.component_to_probe_line = {c: pl for pl, components in probe_lines.items() for c in components}
        """Map from each component to the probeline connected to it.
        Max 1 connection per component is assumed.
        Components without connection to a probe line don't appear.
        """

        self._locus_mappings: dict[str, dict[Locus, tuple[str, ...]]] = {
            DEFAULT_1QB_MAPPING: {(qubit,): (qubit,) for qubit in self.qubits_sorted},
            DEFAULT_2QB_MAPPING: {
                frozenset(comps): (coupler,)
                for coupler, comps in self.coupler_to_components.items()
                if len(comps) == 2
                and set(comps).issubset(frozenset().union(*[self.qubits, self.computational_resonators]))
            },
        }

    @classmethod
    def from_chip_design_record(cls, record: dict) -> ChipTopology:
        """Construct a ChipTopology instance from a raw Chip design record.

        Args:
            record: Record as returned by Station control.

        Returns:
            Corresponding chip topology

        """
        return cls.from_chad(CHAD(**record))

    @classmethod
    def from_chad(cls, chad: CHAD) -> ChipTopology:
        """Construct a ChipTopology instance from a CHAD. Use :meth:`from_chip_design_record` if possible.

        Args:
            chad: parsed CHAD model
        Returns:
            corresponding chip topology

        """
        qubits = chad.qubit_names
        computational_resonators = chad.computational_resonator_names
        data_components = frozenset(qubits + computational_resonators)
        return cls(
            qubits=qubits,
            computational_resonators=computational_resonators,
            couplers={coupler.name: data_components & set(coupler.connections) for coupler in chad.components.couplers},
            probe_lines={
                pl.name: (data_components | frozenset(chad.coupler_names)) & set(pl.connections)
                for pl in chad.components.probe_lines
            },
            variant=chad.variant,
        )

    def get_neighbor_couplers(self, components: Iterable[str]) -> set[str]:
        """Couplers that connect to at least one of the given chip components.

        Args:
            components: some chip components, typically qubits and computational resonators
        Returns:
            couplers that connect to at least one of ``components``

        """
        couplers: set[str] = set()
        for component in components:
            if (coupler := self.component_to_couplers.get(component)) is not None:
                couplers |= coupler
        return couplers

    def get_connecting_couplers(self, components: Collection[str]) -> set[str]:
        """Couplers that only connect to the given chip components, and connect at least two of them.

        Equivalent to returning the edges in the ``components``-induced
        subgraph of the coupling topology.

        Args:
            components: some chip components, typically qubits and computational resonators
        Returns:
            couplers that connect to only members of ``components``, and to at least two of them

        """
        connecting_couplers = set()
        for coupler in self.get_neighbor_couplers(components):
            connections = self.coupler_to_components[coupler]
            if all(q in components for q in connections) and len(connections) >= 2:
                connecting_couplers.add(coupler)
        return connecting_couplers

    def get_coupler_for(self, component_1: str, component_2: str) -> str:
        """Common coupler for the given chip components (e.g. qubit or computational resonator).

        Args:
            component_1: first component
            component_2: second component
        Returns:
            the common coupler
        Raises:
            ValueError: the given components have zero or more than one connecting coupler

        """
        connecting_couplers = self.get_connecting_couplers((component_1, component_2))
        if (n_couplers := len(connecting_couplers)) != 1:
            raise ValueError(f"Components {component_1} and {component_2} have {n_couplers} connecting couplers.")
        return next(iter(connecting_couplers))

    def get_neighbor_locus_components(self, components: Collection[str]) -> set[str]:
        """Chip components that are connected to the given components by a coupler, but not included in them.

        Args:
            components: some chip components, typically qubits and computational resonators
        Returns:
            components that are connected to ``components`` by a coupler, but not included in them

        """
        neighbor_components = set()
        for coupler in self.get_neighbor_couplers(components):
            neighbor_components |= set(self.coupler_to_components[coupler])
        return neighbor_components - set(components)

    def get_connected_probe_lines(self, components: Collection[str]) -> set[str]:
        """Get probelines that are connected to any of the given components."""
        return {self.component_to_probe_line[c] for c in components if c in self.component_to_probe_line}

    def get_connected_coupler_map(self, components: Collection[str]) -> dict[str, tuple[str, ...]]:
        """Returns a `ComponentMap`, including only the couplers between components that both are in the given subset.

        Args:
            components: Collection of coupled components to restrict the returned couplers.

        Returns:
            A `ComponentMap`, a dict mapping coupler names to the names of the coupled components.

        """
        return {
            key: values
            for key, values in self.coupler_to_components.items()
            if (key in self.get_connecting_couplers(components))
        }

    @staticmethod
    def limit_values(dct: ComponentMap, limit_to: Collection[str]) -> dict[str, Collection[str]]:
        """Prunes the given dictionary (e.g. a coupler-to-qubits map) to a subset of values.

        Used to prune e.g. :attr:`coupler_to_components` to a subset of relevant elements.

        Args:
            dct: Dictionary of collections of values.
            limit_to: Components to limit the output to.

        Returns:
            The input dictionary, but only with key-value pairs where the value intersects with `limit_to`.

        """
        return {key: values for key, values in dct.items() if any(v in limit_to for v in values)}

    def is_qubit(self, component: str) -> bool:
        """True iff the given component is a qubit."""
        return component in self.qubits

    def is_coupler(self, component: str) -> bool:
        """True iff the given component is a coupler."""
        return component in self.couplers

    def is_probe_line(self, component: str) -> bool:
        """True iff the given component is a probe line."""
        return component in self.probe_lines

    def is_computational_resonator(self, component: str) -> bool:
        """True iff the given component is a computational resonator."""
        return component in self.computational_resonators

    def set_locus_mapping(self, name: str, mapping: dict[Locus, tuple[str, ...]]) -> None:
        """Add a custom mapping from a gate locus to a set of components required for the gate operation.

        The mapping is of the form {<locus>: <components mapped to locus>}, where a locus can be mapped to one or more
        components. The locus itself can be a frozenset (denoting a symmetric gate) or a tuple (non-symmetric gate).

        Some examples:
        - ``DEFAULT_2QB_MAPPING`` (added in :meth:`__init__`) maps pairs of qubits to their common coupler symmetrically.
        - Fast flux CZ-gate maps pairs of qubits to their couplers non-symmetrically (first locus qubit can perform flux pulses).
        - A two-qubit gate implementation that includes playing pulses on neighboring components in addition to the
          connecting coupler.

        Args:
            name: The name for the gate & implementation this locus mapping represents (typically in the format
                ``"<gate name>.<implementation name>"``).
            mapping: The locus mapping to be added.

        """  # noqa: E501
        self._validate_locus_mapping(mapping)
        self._locus_mappings[name] = mapping

    def _validate_locus_mapping(self, mapping: dict[Locus, tuple[str, ...]]) -> None:
        """Validate that the components given in mapping are found in self and the mapping is correctly formed."""
        for locus, mapped in mapping.items():
            if not isinstance(locus, tuple) and not isinstance(locus, frozenset):
                raise ValueError("Mapped loci need to be tuples or frozen sets of component names")
            for mapped_component in mapped:
                if mapped_component not in self.all_components:
                    raise ValueError(f"Mapped component {mapped_component} is not found in this ChipTopology.")
            for locus_component in locus:
                if locus_component not in self.all_components:
                    raise ValueError(f"Locus component {locus_component} is not found in this ChipTopology.")

    def map_locus(self, locus: Locus, name: str | None = None) -> str | tuple[str, ...] | None:
        """Returns the mapped components for the given locus and the given gate.

        If the locus or the gate is not found from the locus mappings of self, returns None.

        Args:
            locus: The locus to map.
            name: The name for the gate & implementation with which to map the locus (typically in the format
                ``"<gate name>.<implementation name>"``).

        Returns:
            The components mapped to the given locus or `None` if locus is not found in the given mapping.

        """
        if not name:
            if len(locus) == 1:
                name = DEFAULT_1QB_MAPPING
            elif len(locus) == 2:
                name = DEFAULT_2QB_MAPPING
        if name not in self._locus_mappings:
            return None
        return self._locus_mappings[name].get(locus, None)

    def map_to_locus(self, mapped: str | tuple[str], name: str) -> Locus | None:
        """Returns the locus that is mapped to the given components.

        Args:
            mapped: The mapped components.
            name: The name for the gate & implementation with which to map the locus (typically in the format
                ``"<gate name>.<implementation name>"``).

        Returns:
            The locus mapped to the given components or `None` if the components are not mapped to any locus.

        """
        if name not in self._locus_mappings:
            return None
        for locus, mapped_locus in self._locus_mappings[name].items():
            if mapped_locus == mapped:
                return locus
        return None

    def get_loci(self, name: str, default_mapping_dimension: int | None = None) -> list[Locus]:
        """Gives all the loci of a given gate.

        If no mapping with the given the name nor a default mapping with the given dimensionality is found,
        returns an empty list.

        Args:
            name: The name for the gate & implementation with which to map the locus (typically in the format
                ``"<gate name>.<implementation name>"``).
            default_mapping_dimension: If provided, will return the loci of the default mapping of the given
                dimensionality in case no mapping for ``name`` can be found.

        Returns:
            The loci associated with the given gate.

        """
        if name not in self._locus_mappings:
            if default_mapping_dimension == 1:
                name = DEFAULT_1QB_MAPPING
            elif default_mapping_dimension == 2:
                name = DEFAULT_2QB_MAPPING
        return list(self._locus_mappings.get(name, {}))

    def get_common_computational_resonator(self, first_qubit: str, second_qubit: str) -> str:
        """Convenience method for getting the name of a computational resonator which is connected to both specified
        qubit components via tunable couplers.

        Args:
             first_qubit: The name of the first qubit.
             second_qubit: The name of the second qubit.
         The order of qubits does not matter, i.e. the `first_qubit` and `second_qubit` arguments are interchangeable.

        Returns:
             - The name of the computational resonator that is connected to both inputted qubits via tunable couplers.

        Raises:
             - ValueError: If no computational resonator was found that is connected to both qubits via tunable
             couplers.

        """
        neighbor_components = list(self.get_neighbor_locus_components([first_qubit, second_qubit]))  # noqa: F841

        resonators = [
            r
            for r in self.get_neighbor_locus_components([first_qubit, second_qubit])
            if r in self.computational_resonators
        ]
        common_resonators = [
            r
            for r in resonators
            if len(self.get_connecting_couplers([first_qubit, r])) == 1
            and len(self.get_connecting_couplers([second_qubit, r])) == 1
        ]

        if len(common_resonators) == 0:  # if no computational resonator is connected to both qubits
            raise ValueError(
                f"No computational resonator was found, that is connected to both qubits {first_qubit} and "
                f"{second_qubit} via tunable couplers."
            )
        if (
            len(common_resonators) == 1
        ):  # if only one computational resonator is connected to both qubits via tunable couplers
            computational_resonator = common_resonators[0]
        else:
            computational_resonator = sorted(common_resonators)[0]
            print(
                f"Warning: There was no unique computational resonator found, which connects to both qubits"
                f"{first_qubit} and {second_qubit} via tunable couplers. Use first one found: "
                f"{computational_resonator}."
            )
        return computational_resonator

    def get_all_common_resonators(
        self,
        qubits: list[str],
    ) -> set[str]:
        """Computational resonators connected to all the given qubits via a coupler.

        Args:
            qubits: Qubit names.

        Returns:
            Names of the computational resonators neighboring all of ``qubits`` (can be an empty set).

        """
        if not qubits:
            return set()
        common_resonator_set = set.intersection(*(self.get_neighbor_locus_components([qubit]) for qubit in qubits))
        # ensure the resonator set contains only resonators
        return common_resonator_set.intersection(self.computational_resonators)
