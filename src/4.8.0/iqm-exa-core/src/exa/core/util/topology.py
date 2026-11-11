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
"""Topology and layout functions."""

from typing import Any

import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher

from exa.common.qcm_data import ChipLayout, ChipTopology

from .calibration_utils import Coloring, Matching


def get_connectivity_graph(
    chip_topology: ChipTopology, chip_layout: ChipLayout | None = None, limit_to_components: set[str] | None = None
) -> nx.Graph:
    """Get QPU connectivity graph from the chip topology, restricted to certain components.

    Construct a networkx graph of the qubits and computational resonators, connected if a coupler exists between
    them. Each node has a "pos" attribute which corresponds to the approximate and rescaled position of the qubit
    on the chip.

    Args:
        chip_topology: Topology of the chip.
        chip_layout: Layout of the chip (physical locations).
        limit_to_components: Only include these components. Might result in disconnected graphs.

    """
    chip_layout = chip_layout or ChipLayout({}, {}, {})
    chip_graph = nx.Graph()  # type: ignore[var-annotated]
    computational_components = chip_topology.qubits_sorted + chip_topology.computational_resonators_sorted
    coordinates = chip_layout.get_all_qubit_coordinates()
    node_tuples = [
        (comp, {"pos": coordinates[comp]}) if comp in coordinates else comp for comp in computational_components
    ]
    chip_graph.add_nodes_from(node_tuples)
    chip_graph.add_edges_from(list(chip_topology.coupler_to_components.values()))  # type: ignore[arg-type]

    if limit_to_components is not None:
        to_remove = [component for component in computational_components if component not in limit_to_components]
        chip_graph.remove_nodes_from(to_remove)

    return chip_graph


def get_chip_graph_matcher(
    layout_graph: nx.Graph, chip_topology: ChipTopology, components: set[str] | None = None
) -> GraphMatcher:
    """Get a networkx GraphMatcher between the specific experiment graph and the used QPU connectivity graph.

    Args:
        layout_graph: The layout graph corresponding to the specific experiment.
        chip_topology: Topology of the chip context.
        components: List of components. If None, uses all the components.

    Returns:
        GraphMatcher object.

    """
    chip_graph = get_connectivity_graph(chip_topology, limit_to_components=components)
    matcher = GraphMatcher(chip_graph, layout_graph)
    return matcher


def get_naive_morphisms(
    layout_graph: nx.Graph,
    chip_topology: ChipTopology,
    components: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Get a list of non-overlapping layouts compatible with the given experiment layout graph.

    Iteratively finds the first match between the needed layout and the QPU connectivity and removes the used qubits.
    Non-optimal, the general problem (finding a perfect morphisms in a k-regular hypergraph) is NP-complete.

    Args:
        layout_graph: The layout graph corresponding to the specific experiment.
        chip_topology: Topology of the chip context.
        components: List of components. If None, uses all the components.

    Returns:
        List of layout morphisms with physical components.

    """
    morphisms = []
    layouts_remain = True

    remaining_components = set(components or list(chip_topology.qubits) + list(chip_topology.computational_resonators))

    while layouts_remain:
        matcher = get_chip_graph_matcher(layout_graph, chip_topology, remaining_components)
        if len(layout_graph.nodes) > len(remaining_components):
            layouts_remain = False
        elif (mapping := next(matcher.subgraph_isomorphisms_iter(), None)) or (
            mapping := next(matcher.subgraph_monomorphisms_iter(), None)
        ):
            morphisms.append(mapping)
            remaining_components = remaining_components - set(mapping.keys())
        else:
            layouts_remain = False

    return morphisms


def get_matching_from_morphisms(morphisms: list[dict[str, Any]], layout_graph: nx.Graph) -> Matching:
    """Convert morphisms into matching.

    Relies on the order of the nodes in the graph being fixed.

    For each mapping, create a tuple of physical components in the order of nodes in the layout graph they correspond
    to.

    Args:
        morphisms: list of mappings of physical qubits to graph labels.
        layout_graph: graph representing the circuit connectivity.

    Returns:
        Matching of hyperedges.

    """
    matching = []
    for hyperedge in morphisms:
        reverse_hyperedge = {label: qubit for qubit, label in hyperedge.items()}

        group = tuple(reverse_hyperedge[label] for label in layout_graph.nodes)
        matching.append(group)

    return matching


def is_valid_homomorphism(
    rev_mapping: dict[str, str],
    layout_graph: nx.Graph,
    chip_graph: nx.Graph,
) -> bool:
    """Check if the mapping corresponds to a valid injective homomorphism between two graphs.

    Args:
        rev_mapping: Mapping of logical qubit labels to physical qubits
        layout_graph: Logical qubit graph representing the experiment.
        chip_graph: Graph representing the chip topology.

    Returns:
        True if the mapping given is an injective homomorphism from the logical qubit graph to the chip graph.

    """
    if set(rev_mapping.keys()) != set(layout_graph.nodes):
        return False

    # check injectivity - each physical qubit appears once
    if len(set(rev_mapping.values())) != len(rev_mapping.values()):
        return False

    return all(chip_graph.has_edge(rev_mapping[u], rev_mapping[v]) for u, v in layout_graph.edges())


def validate_components(components: list[tuple[str, ...]], layout_graph: nx.Graph, chip_topology: ChipTopology) -> None:
    """Validate each experiment specification to be possible using the layout graph.

    For each tuple of computational elements, check whether they correspond to a subset of qubits on the chip
    connected in a correct way and are given in the correct order.

    Args:
        components: A single layer of parallel components.
        layout_graph: The layout graph corresponding to the specific experiment.
        chip_topology: Topology of the chip context.

    Raises:
        ValueError if anything is invalid.

    """
    invalid = set()
    chip_graph = get_connectivity_graph(chip_topology)

    for group in components:
        if len(group) != len(layout_graph.nodes):
            invalid.add(group)

        expected = {role: qubit for qubit, role in zip(group, list(layout_graph.nodes))}
        if not is_valid_homomorphism(expected, layout_graph, chip_graph):
            invalid.add(group)
    if invalid:
        raise ValueError(f"One or more component lists are invalid: {invalid}")


def make_bitsets(all_subsets: list[set[str]]) -> list[int]:
    """Convert a list of subsets into list of bitsets."""
    elems = {}
    idx = 0
    bitsets = []
    for subset in all_subsets:
        for element in subset:
            if element not in elems:
                elems[element] = idx
                idx += 1
    for subset in all_subsets:
        mask = 0
        for element in subset:
            mask |= 1 << elems[element]
        bitsets.append(mask)
    return bitsets


def get_naive_layer_from_subsets(all_subsets: list[set[str]]) -> tuple[list[int], int]:
    """Get a single naive layer of a coloring.

    Going in order, find a set of subsets which does not overlap on any element.

    Args:
        all_subsets: List of all the subsets.

    Returns:
        List of indices included and the total number of included elements.

    """
    bitsets = make_bitsets(all_subsets)
    mask = 0
    combi = []
    n = len(bitsets)

    for i in range(n):
        m = bitsets[i].bit_count()
        new_mask = mask | bitsets[i]
        if new_mask.bit_count() == mask.bit_count() + m:
            mask = new_mask
            combi.append(i)

    return combi, mask.bit_count()


def get_naive_coloring(components: list[tuple[str, ...]]) -> Coloring:
    """Get a naive coloring out of given component lists.

    Going in order, find a set of tuples of qubits which does not overlap on any qubit. Remove these tuples, and repeat,
    obtaining effectively an edge coloring of a hypergraph. Solving this problem optimally is NP-complete.

    Returns:
        Coloring for parallel experiments, compatible with the COLORED_GROUPS mode.

    """
    remaining_components = components.copy()
    coloring = []
    while remaining_components:
        all_subsets = [set(hyperedge) for hyperedge in remaining_components]
        indices, _ = get_naive_layer_from_subsets(all_subsets)
        indices_sorted = sorted(indices, reverse=True)
        coloring.append([remaining_components[idx] for idx in indices_sorted])
        for idx in indices_sorted:
            del remaining_components[idx]

    return coloring
