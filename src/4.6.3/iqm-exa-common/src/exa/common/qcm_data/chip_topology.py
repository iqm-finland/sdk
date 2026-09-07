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
"""ChipTopology class for parsing CHAD and other QPU related data into human-usable form."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
import itertools
import re

from exa.common.errors.iqm_error import NotFoundError, ValidationError
from exa.common.qcm_data.chad_model import CHAD


def _key_numeric(name: str) -> int:
    """Sorting key for component names."""
    match = re.search(r"(\d+)", name)
    return int(match.group(1)) if match else 0


def _key_numeric_tuple(name: str) -> tuple[int, ...]:
    """Sorting key for coupler names.

    Supports e.g. the following patterns: "TC-1-2", "TC1"
    """
    return tuple(int(m) for m in re.findall(r"\d+", name))


def _key_alphanumeric(name: str) -> tuple[str, int, str]:
    """Sorting key for alphanumeric sorting of components."""
    return re.sub(r"[^a-zA-Z]", "", name), _key_numeric(name), name


def sort_components(components: Iterable[str]) -> list[str]:
    """Sort the given components in a human-readable way."""
    # sorts strings in reverse order, but numbers in normal order
    return sorted(components, key=lambda x: (re.sub(r"[^a-zA-Z]", "", x), -_key_numeric(x), x), reverse=True)


def sort_couplers(couplers: Iterable[str]) -> list[str]:
    """Sort the given couplers in a human-readable way."""
    return sorted(couplers, key=_key_numeric_tuple)


def _reverse_dict(dct: dict[str, Iterable[str]]) -> dict[str, set[str]]:
    """Reverses the given many-to-many mapping.

    Returned dict is in no particular order.
    """
    reverse: dict[str, set[str]] = {}
    for key, values in dct.items():
        for value in values:
            reverse.setdefault(value, set()).add(key)
    return reverse


@dataclass(init=False, frozen=True, kw_only=True)
class ChipTopology:
    """Topology information for a chip (typically a QPU).

    Represents the information found in a CHAD/CHEDDAR.

    Args:
        qubits: Names of the qubits.
        computational_resonators: Names of the computational resonators.
        couplers: Mapping from coupler name to names of chip components it connects to.
        probe_lines: Mapping from probe line name to names of chip components it connects to.
        variant: Identifier of the QPU design variant.
        mask_set_name: Identifier of the QPU mask set name.

    """

    mask_set_name: str
    """Identifier of the chip mask set name."""
    variant: str
    """Identifier of the chip design variant."""

    qubits: frozenset[str]
    """Names of the qubits of the chip."""
    qubits_sorted: tuple[str, ...]
    """Names of the qubits sorted in canonical order."""
    computational_resonators: frozenset[str]
    """Names of the computational resonators of the chip."""
    computational_resonators_sorted: tuple[str, ...]
    """Names of the computational resonators sorted in canonical order."""
    couplers: frozenset[str]
    """Names of the tunable couplers of the chip."""
    couplers_sorted: tuple[str, ...]
    """Names of the tunable couplers sorted in canonical order."""
    probe_lines: frozenset[str]
    """Names of the probe lines of the chip."""
    probe_lines_sorted: tuple[str, ...]
    """Names of the probe lines sorted in canonical order."""
    all_components: frozenset[str]
    """All the components on the chip."""

    coupler_to_components: dict[str, tuple[str, ...]]
    """Map from each coupler to all other components it connects to. The values are sorted."""
    component_to_couplers: dict[str, frozenset[str]]
    """Map from each component to all couplers connected to it."""
    probe_line_to_components: dict[str, tuple[str, ...]]
    """Map from each probe line to all components it connects to."""
    component_to_probe_line: dict[str, str]
    """Map from each component to the probe line connected to it.
    Max 1 probe line per component is assumed.
    Components without connection to a probe line don't appear here.
    """

    def __init__(
        self,
        qubits: Iterable[str],
        computational_resonators: Iterable[str],
        couplers: dict[str, Iterable[str]],
        probe_lines: dict[str, Iterable[str]],
        *,
        variant: str = "",
        mask_set_name: str = "",
    ):
        # start by sorting the dicts based on the keys
        couplers = dict(sorted(couplers.items(), key=lambda x: _key_numeric_tuple(x[0])))
        probe_lines = dict(sorted(probe_lines.items(), key=lambda x: _key_alphanumeric(x[0])))

        # frozen dataclass with custom __init__ means we have to use __setattr__
        super().__setattr__("mask_set_name", mask_set_name)
        super().__setattr__("variant", variant)

        super().__setattr__("qubits", frozenset(qubits))
        super().__setattr__("qubits_sorted", tuple(sort_components(self.qubits)))
        super().__setattr__("computational_resonators", frozenset(computational_resonators))
        super().__setattr__("computational_resonators_sorted", tuple(sort_components(self.computational_resonators)))

        data_components = self.qubits | self.computational_resonators
        if diff := set(itertools.chain.from_iterable(couplers.values())) - data_components:
            raise ValidationError(f"Couplers connect to unknown components: {diff}")
        if diff := set(itertools.chain.from_iterable(probe_lines.values())) - (data_components | frozenset(couplers)):
            raise ValidationError(f"Probe lines connect to unknown components: {diff}")

        super().__setattr__("couplers", frozenset(couplers))
        super().__setattr__("couplers_sorted", tuple(couplers))  # already sorted
        super().__setattr__("probe_lines", frozenset(probe_lines))
        super().__setattr__("probe_lines_sorted", tuple(probe_lines))  # already sorted
        super().__setattr__("all_components", frozenset(data_components | self.couplers | self.probe_lines))

        def sort_values(dct: dict[str, Iterable[str]]) -> dict[str, tuple[str, ...]]:
            """Keep key order, sort the values for each key."""
            return {key: tuple(sort_components(values)) for key, values in dct.items()}

        super().__setattr__("coupler_to_components", sort_values(couplers))
        component_to_couplers = _reverse_dict(couplers)
        super().__setattr__(
            # keys in no particular order
            "component_to_couplers",
            {c: frozenset(couplers) for c, couplers in component_to_couplers.items()},
        )

        super().__setattr__("probe_line_to_components", sort_values(probe_lines))
        component_to_pls = _reverse_dict(probe_lines)
        # NOTE: we handle only one pl per component
        for c, pls in component_to_pls.items():
            if len(pls) != 1:
                raise ValidationError(f"Component {c} is connected to more than one probe line: {pls}")
        super().__setattr__(
            # keys in no particular order
            "component_to_probe_line",
            {c: pl for pl, components in probe_lines.items() for c in components},
        )

    @classmethod
    def from_chip_design_record(cls, chip_design_record: dict) -> ChipTopology:
        """Construct a ChipTopology instance from a raw Chip design record.

        Args:
            chip_design_record: Chip design record as returned by Station Control.

        Returns:
            Corresponding chip topology.

        """
        return cls.from_chad(CHAD(**chip_design_record))

    @classmethod
    def from_chad(cls, chad: CHAD) -> ChipTopology:
        """Construct a ChipTopology instance from a CHAD. Use :meth:`from_chip_design_record` if possible.

        Args:
            chad: Parsed CHAD model.

        Returns:
            Corresponding chip topology.

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
            mask_set_name=chad.mask_set_name,
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
            ValidationError: the given components have zero or more than one connecting coupler

        """
        connecting_couplers = self.get_connecting_couplers((component_1, component_2))
        if (n_couplers := len(connecting_couplers)) != 1:
            raise ValidationError(f"Components {component_1} and {component_2} have {n_couplers} connecting couplers.")
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
        """Mapping from couplers that only connect to ``components``, to the components they connect to.

        Additionally, the returned couplers must connect to at least two of ``components``.

        Args:
            components: Components for which to find shared couplers.

        Returns:
            Mapping from coupler names to the names of ``components`` they connect to.

        """
        return {coupler: self.coupler_to_components[coupler] for coupler in self.get_connecting_couplers(components)}

    @staticmethod
    def limit_values(dct: Mapping[str, Collection[str]], limit_to: Collection[str]) -> dict[str, Collection[str]]:
        """Prunes the given mapping (e.g. a coupler_to_components map) to a subset.

        Used to prune e.g. :attr:`coupler_to_components` to a subset of relevant elements.

        Args:
            dct: Mapping to prune.
            limit_to: At least one of these components must appear in the mapping values.

        Returns:
            The input mapping, but only with key-value pairs where the value intersects with ``limit_to``.

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
            raise NotFoundError(
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
