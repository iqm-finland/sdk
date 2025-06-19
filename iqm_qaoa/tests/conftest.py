# Copyright (c) 2024-2025 IQM Quantum Computers
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted (subject to the
# limitations in the disclaimer below) provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this list of conditions and the following
#   disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
#   disclaimer in the documentation and/or other materials provided with the distribution.
# * Neither the name of IQM Quantum Computers nor the names of its contributors may be used to endorse or promote
#   products derived from this software without specific prior written permission.
#
# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY
# THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""
Common fixtures go here.
"""

# This pylint warning is disabled because "If a fixture is used in the same module in which it is defined,
# the function name of the fixture will be shadowed by the function arg that requests the fixture."
# Reference:
# https://stackoverflow.com/questions/46089480/pytest-fixtures-redefining-name-from-outer-scope-pylint
# https://docs.pytest.org/en/stable/reference/reference.html#pytest-fixture

# pylint: disable=redefined-outer-name

from collections.abc import Callable
import random
from uuid import UUID

from dimod import BinaryQuadraticModel
from dimod.generators import uniform
from iqm.applications.maxcut import MaxCutInstance, maxcut_generator
from iqm.applications.mis import MISInstance
from iqm.applications.sk import SherringtonKirkpatrick, sk_generator
from iqm.iqm_client import DynamicQuantumArchitecture, GateImplementationInfo, GateInfo
from iqm.qaoa.backends import SamplerBackend
from iqm.qaoa.generic_qaoa import QAOA
from iqm.qaoa.transpiler.quantum_hardware import QPU, Grid2DQPU, HardEdge
from iqm.qaoa.transpiler.routing import Layer
from iqm.qiskit_iqm.fake_backends import IQMFakeApollo
from iqm.qiskit_iqm.iqm_backend import IQMBackendBase
import networkx as nx
import pytest
from qiskit.providers import Options


@pytest.fixture
def small_mis_instance() -> MISInstance:
    """An instance of the MIS problem, using a fixed-seed Erdős–Rényi graph."""
    return MISInstance(nx.erdos_renyi_graph(6, 0.5, seed=1337))


@pytest.fixture
def sparse_mis_instance() -> MISInstance:
    """An instance of the MIS problem, using a fixed-seed 3-regular graph."""
    return MISInstance(nx.random_regular_graph(n=14, d=3, seed=1337))


@pytest.fixture
def small_maxcut_instance() -> MaxCutInstance:
    """An instance of the max-cut problem, using a fixed-seed Erdős–Rényi graph."""
    return next(maxcut_generator(n=6, n_instances=1, seed=1337))


@pytest.fixture
def sparse_maxcut_instance() -> MaxCutInstance:
    """An instance of the max-cut problem, using a fixed-seed 3-regular graph."""
    return next(maxcut_generator(n=14, n_instances=1, seed=1337, graph_family="regular", d=3))


@pytest.fixture
def small_sk_instance() -> SherringtonKirkpatrick:
    """An instance of SK problem, using a fixed-seed Gaussian-distribured interactions."""
    return next(sk_generator(n=6, n_instances=1, distribution="gaussian", seed=1337))


@pytest.fixture
def tolerance() -> float:
    """Tolerance to be used in equality comparisons."""
    return 1e-5


@pytest.fixture
def special_g() -> nx.Graph:
    """Special graph to test MIS algorithms."""
    g = nx.Graph()
    g.add_edges_from(
        [
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 4),
            (1, 5),
            (1, 6),
            (2, 4),
            (2, 5),
            (2, 6),
            (3, 4),
            (3, 5),
            (3, 6),
            (4, 5),
            (4, 6),
            (5, 6),
        ]
    )
    return g


@pytest.fixture
def edge_cases_graph_generator() -> Callable[[int], tuple[nx.Graph, ...]]:
    """Fixture that returns a function for generating edge case graphs of a given size."""

    def graph_generator(n: int) -> tuple[nx.Graph, ...]:
        """
        Function that generates 4 edge-case graphs of a given size.

        Args:
            n: The number of nodes of all the graphs to be generated.

        Returns:
            complete_graph: A complete graph (all nodes connected with each other).
            cycle_graph: A cycle graph (looks like a circle).
            linear_graph: A linear graph (looks like a line, obtained by deleting one edge from cycle_graph).
            disconnected_graph: A disconnected graph (no edges, just vertices).
        """
        complete_graph = nx.complete_graph(n)
        cycle_graph = nx.Graph()
        cycle_graph.add_edges_from([(i, i + 1) for i in range(n - 1)] + [(n - 1, 0)])
        linear_graph = nx.Graph()
        linear_graph.add_edges_from([(i, i + 1) for i in range(n - 1)])
        disconnected_graph = nx.Graph()
        disconnected_graph.add_nodes_from(list(range(n)))
        return complete_graph, cycle_graph, linear_graph, disconnected_graph

    return graph_generator


@pytest.fixture
def alpha() -> float:
    """
    The lower bound on the approximation ratio of the ``goemans_williamson`` algorithm.
    """
    return 0.878


@pytest.fixture
def bqms() -> list[BinaryQuadraticModel]:
    """
    A list of quasi-random 4-regular BQMs on 36 variables.
    """
    return [uniform(nx.random_regular_graph(4, 36, seed=1337), "SPIN", low=0.5, high=1.0) for _ in range(10)]


@pytest.fixture(scope="session")
def apollo_backend() -> IQMFakeApollo:
    """
    Apollo backend instantiated here, so that it doesn't need to be done repeatedly.
    """
    backend = IQMFakeApollo()
    return backend


class MockBackend(IQMBackendBase):
    def __init__(self, architecture: DynamicQuantumArchitecture, **kwargs):
        super().__init__(architecture, **kwargs)

    @classmethod
    def _default_options(cls) -> Options:
        return Options(shots=1024)

    @property
    def max_circuits(self) -> int | None:
        return None

    def run(self, run_input, **options):
        raise NotImplementedError


@pytest.fixture(scope="session")
def sirius_mock_backend() -> MockBackend:
    """
    Sirius mock backend defined here to be shared among many tests.
    """
    architecture = DynamicQuantumArchitecture(
        calibration_set_id=UUID("26c5e70f-bea0-43af-bd37-6212ec7d04cb"),
        qubits=[f"QB{i + 1}" for i in range(24)],
        computational_resonators=["COMPR1"],
        gates={
            gate: GateInfo(
                implementations={default_impl: GateImplementationInfo(loci=loci)},
                default_implementation=default_impl,
                override_default_implementation={},
            )
            for gate, default_impl, loci in [
                ("prx", "drag_gaussian", tuple((f"QB{i + 1}",) for i in range(24))),
                ("measure", "constant", tuple((f"QB{i + 1}",) for i in range(24))),
                ("cz", "tgss", tuple((f"QB{i + 1}", "COMPR1") for i in range(24))),
                ("move", "tgss_crf", tuple((f"QB{i + 1}", "COMPR1") for i in range(24))),
            ]
        },
    )
    backend = MockBackend(architecture)
    return backend


@pytest.fixture
def square_qpu() -> QPU:
    """
    A 6-by-6 grid QPU.
    """
    return Grid2DQPU(6, 6)


@pytest.fixture
def layer(square_qpu: QPU) -> Layer:
    """
    Returns a layer from the QPU.
    """
    return Layer(square_qpu)


@pytest.fixture
def matching_gates(square_qpu: QPU) -> set[HardEdge]:
    """
    Returns a set of matching gates generated on the QPU graph.
    """
    g = square_qpu.hardware_graph
    matching = nx.maximal_matching(g)
    matching_gates = set()
    for gate in matching:
        matching_gates.add(HardEdge(gate))
    return matching_gates


@pytest.fixture
def is_layer_valid() -> Callable[[Layer], bool]:
    """
    Fixture for a function that tests whether the layer ``input_layer`` is valid.
    """

    def validity_check(input_layer: Layer) -> bool:
        for hard_qb in input_layer.gates.nodes():
            # The number of qubits around ``hard_qb`` which are involved in gates with ``hard_qb``
            involved_in_gates = 0
            for neighbor in input_layer.gates.neighbors(hard_qb):
                if input_layer.gates[neighbor][hard_qb]["swap"] or input_layer.gates[neighbor][hard_qb]["int"]:
                    involved_in_gates += 1
                    # Only one neighbor of ``hard_qb`` can be involved in gates with it.
                    if involved_in_gates > 1:
                        return False
        return True

    return validity_check


@pytest.fixture
def graphs_for_ec() -> list[nx.Graph]:
    """
    Contains the graphs used for testing edge coloring.
    """
    complete_graph = nx.complete_graph(20)
    circle_graph = nx.cycle_graph(20)
    complete_bipartite = nx.complete_bipartite_graph(8, 12)
    star_graph = nx.star_graph(19)  # 19 is the number of outer nodes, so ``nx.star_graph(19)`` has 20 nodes in total.
    grid_graph = nx.grid_2d_graph(4, 5)
    random_graph = nx.erdos_renyi_graph(20, 0.4)
    disconnected_graph = nx.disjoint_union(nx.erdos_renyi_graph(8, 0.4), nx.erdos_renyi_graph(12, 0.5))

    return [complete_graph, circle_graph, complete_bipartite, star_graph, grid_graph, random_graph, disconnected_graph]


@pytest.fixture
def custom_rigged_sampler() -> SamplerBackend:
    """Returns a sampler object that always 'samples' the same bitstring."""

    # pylint: disable=too-few-public-methods
    class RiggedSampler(SamplerBackend):
        """Locally defined rigged sampler, designed to solve the MIS problem on ``special_g`` defined above."""

        def sample(self, qaoa_object: QAOA, shots: int) -> dict[str, int]:
            return {"0111000": shots}

    return RiggedSampler()


@pytest.fixture
def samples_dict(sparse_mis_instance: MISInstance) -> dict[str, int]:
    """
    Returns a quasi-random dictionary of samples.
    """
    n = sparse_mis_instance.dim
    shots = 10000
    unique_bitstrings = 200
    random.seed(1337)

    bp = sorted(random.sample(range(1, shots), unique_bitstrings - 1))  # bp = breakpoints
    vals = [bp[0]] + [bp[i] - bp[i - 1] for i in range(1, unique_bitstrings - 1)] + [shots - bp[-1]]

    set_of_unique_bitstrings: set[str] = set()
    while len(set_of_unique_bitstrings) < unique_bitstrings:
        bit_str_to_add = "".join(random.choice("01") for _ in range(n))
        set_of_unique_bitstrings.add(bit_str_to_add)
    list_bit_strings = list(set_of_unique_bitstrings)

    return dict(zip(list_bit_strings, vals))


@pytest.fixture(scope="session")
def qpu_with_hole() -> QPU:
    """
    QPU with the following topology:

    +---+---+---+---+---+---+
    |   |   |   |   |   |   |
    +---+---+---+---+---+---+
    |   |   |   |   |   |   |
    +---+---+---+---+---+---+
    |   |   |       |   |   |
    +---+---+       +---+---+
    |   |   |       |   |   |
    +---+---+---+---+---+---+
    |   |   |   |   |   |   |
    +---+---+---+---+---+---+
    |   |   |   |   |   |   |
    +---+---+---+---+---+---+
    """
    graph_with_hole = nx.grid_2d_graph(7, 7)
    graph_with_hole.remove_node((3, 3))

    mapping = {coords: ints for ints, coords in enumerate(graph_with_hole.nodes())}
    inverse_mapping = {i: c for c, i in mapping.items()}
    graph_with_hole = nx.relabel_nodes(graph_with_hole, mapping)
    return QPU(graph_with_hole, inverse_mapping)
