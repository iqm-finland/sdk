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
"""Common fixtures go here."""

from collections.abc import Callable
from uuid import UUID

from dimod import BinaryQuadraticModel
from dimod.generators import uniform
from iqm.applications.maxcut import MaxCutInstance, maxcut_generator
from iqm.applications.mis import MISInstance
from iqm.applications.qubo import QUBOInstance
from iqm.applications.sk import SherringtonKirkpatrick, sk_generator
from iqm.iqm_client import DynamicQuantumArchitecture, GateImplementationInfo, GateInfo
from iqm.qaoa.backends import EstimatorBackend, SamplerBackend
from iqm.qaoa.generic_qaoa import QAOA
from iqm.qaoa.qubo_qaoa import QUBOQAOA
from iqm.qaoa.transpiler.quantum_hardware import QPU, Grid2DQPU, HardEdge, LogQubit
from iqm.qaoa.transpiler.routing import Layer
from iqm.qaoa.tree_calculation.generate_basis import get_z_basis_m, get_z_basis_m_t
from iqm.qaoa.tree_calculation.tree_calculation import get_exp_vals
from iqm.qiskit_iqm.fake_backends.fake_apollo import IQMFakeApollo
from iqm.qiskit_iqm.iqm_backend import IQMBackendBase
import networkx as nx
import numpy as np
from numpy.typing import NDArray
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
        """Function that generates 4 edge-case graphs of a given size.

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
    """The lower bound on the approximation ratio of the ``goemans_williamson`` algorithm."""
    return 0.878


@pytest.fixture
def bqms() -> list[BinaryQuadraticModel]:
    """A list of quasi-random 4-regular BQMs on 36 variables."""
    return [uniform(nx.random_regular_graph(4, 36, seed=1337), "SPIN", low=0.5, high=1.0) for _ in range(10)]


@pytest.fixture(scope="session")
def apollo_backend() -> IQMFakeApollo:
    """Apollo backend instantiated here, so that it doesn't need to be done repeatedly."""
    backend = IQMFakeApollo()
    return backend


class MockBackend(IQMBackendBase):
    """Mock backend class created to define a Sirius mock backend fixture."""

    def __init__(self, architecture: DynamicQuantumArchitecture, **kwargs):
        super().__init__(architecture, **kwargs)

    @classmethod
    def _default_options(cls) -> Options:
        return Options(shots=1024)

    @property
    def max_circuits(self) -> int | None:  # noqa: D102
        return None

    def run(self, run_input, **options):  # noqa: D102, ANN001, ANN201
        raise NotImplementedError


@pytest.fixture(scope="session")
def sirius_mock_backend() -> MockBackend:
    """Sirius mock backend defined here to be shared among many tests."""
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
    """A 6-by-6 grid QPU."""
    return Grid2DQPU(6, 6)


@pytest.fixture
def layer(square_qpu: QPU) -> Layer:
    """Returns a layer from the QPU."""
    return Layer(square_qpu)


@pytest.fixture
def matching_gates(square_qpu: QPU) -> set[HardEdge]:
    """Returns a set of matching gates generated on the QPU graph."""
    g = square_qpu.hardware_graph
    matching = nx.maximal_matching(g)
    matching_gates = set()
    for gate in matching:
        matching_gates.add(HardEdge(gate))
    return matching_gates


@pytest.fixture
def is_layer_valid() -> Callable[[Layer], bool]:
    """Fixture for a function that tests whether the layer ``input_layer`` is valid."""

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
    """Contains the graphs used for testing edge coloring."""
    complete_graph = nx.complete_graph(20)
    circle_graph = nx.cycle_graph(20)
    complete_bipartite = nx.complete_bipartite_graph(8, 12)
    star_graph = nx.star_graph(19)  # 19 is the number of outer nodes, so ``nx.star_graph(19)`` has 20 nodes in total.
    grid_graph = nx.grid_2d_graph(4, 5)
    random_graph = nx.erdos_renyi_graph(20, 0.4)
    disconnected_graph = nx.disjoint_union(nx.erdos_renyi_graph(8, 0.4), nx.erdos_renyi_graph(12, 0.5))

    return [complete_graph, circle_graph, complete_bipartite, star_graph, grid_graph, random_graph, disconnected_graph]


@pytest.fixture
def custom_rigged_sampler() -> Callable[[dict[str, float]], SamplerBackend]:
    """Returns a sampler object that 'samples' deterministically."""

    class RiggedSampler(SamplerBackend):
        """Locally defined rigged sampler, designed to give exactly a fixed output distribution.

        The sampler takes a 'probability distribution' of bitstrings on input and outputs a dictionary of samples
        exactly corresponding to this distribution.

        Args:
            bitstring_distribution: A distribution of bitstrings that the :class:`RiggedSampler` should 'sample' from.
                It's a dictionary whose keys are the bitstrings and values their 'probabilities'. The output from
                :meth:`sample` will be a dictionary whose values are exactly the values of `bitstring_distribution`
                multiplied by `shots` (rounded to an integer, if necessary).

        Raises:
            ValueError: If the input distribution isn't a proper probability distribution (values don't sum up to 1).

        """

        def __init__(self, bitstring_distribution: dict[str, float]) -> None:
            if sum(bitstring_distribution.values()) != 1:
                raise ValueError(f"The values in the input distribution {bitstring_distribution} don't sum up to 1.")
            self.bitstring_distribution = bitstring_distribution

        def sample(self, qaoa_object: QAOA, shots: int) -> dict[str, int]:
            # Step 1: Multiply each float by ``shots``
            prop_counts = {bit_str: prob * shots for bit_str, prob in self.bitstring_distribution.items()}

            # Step 2: Take floor of each value
            floored_counts = {bit_str: int(count) for bit_str, count in prop_counts.items()}

            # Step 3: Compute how many units are missing
            remainder = shots - sum(floored_counts.values())

            # Step 4: Compute fractional parts
            missing_parts = {bit_str: prop_counts[bit_str] - floored_counts[bit_str] for bit_str in prop_counts}

            # Step 5: Sort keys by largest fractional parts
            sorted_bit_strs = sorted(missing_parts, key=lambda k: missing_parts[k], reverse=True)

            # Step 6: Add 1 to top 'remainder' keys
            for k in sorted_bit_strs[:remainder]:
                floored_counts[k] += 1

            return floored_counts

    def _factory(bitstrings_dist: dict[str, float]) -> RiggedSampler:
        return RiggedSampler(bitstrings_dist)

    return _factory


@pytest.fixture
def samples_dict(sparse_mis_instance: MISInstance) -> dict[str, int]:
    """Return a quasi-random dictionary of samples using NumPy RNG."""
    n = sparse_mis_instance.dim
    shots = 10000
    unique_bitstrings = 200

    rng = np.random.default_rng(seed=1337)

    # Generate breakpoints
    bp = np.sort(rng.choice(np.arange(1, shots), size=unique_bitstrings - 1, replace=False))
    vals = [bp[0]] + [int(bp[i] - bp[i - 1]) for i in range(1, unique_bitstrings - 1)] + [int(shots - bp[-1])]

    # Generate unique bitstrings
    set_of_unique_bitstrings: set[str] = set()
    while len(set_of_unique_bitstrings) < unique_bitstrings:
        bits = rng.integers(0, 2, size=n)  # array of 0/1 ints
        bit_str_to_add = "".join(bits.astype(str))
        set_of_unique_bitstrings.add(bit_str_to_add)

    list_bit_strings = list(set_of_unique_bitstrings)

    return dict(zip(list_bit_strings, vals, strict=True))


@pytest.fixture(scope="session")
def qpu_with_hole() -> QPU:
    """QPU with a specific topology (7x7 square grid with a hole in the middle).

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


@pytest.fixture
def basis_6() -> NDArray[np.int8]:
    """Generates a "basis" of +1's and -1's for `m` equal to 6, to be used for testing `generate_basis.py`.

    The "basis" is a 2D array of dimensions :math:`(4^m, 2m)`, containing all possible :math:`2m` combinations of +1's
    and -1's.
    """
    m = 6
    return get_z_basis_m(m)


@pytest.fixture
def make_fun_to_min() -> Callable[[int, int, float], Callable[[NDArray[np.float64]], np.float64]]:
    """Returns the function to generate the function to minimize using ``scipy`` (given ``p``, ``d`` and ``h``)."""

    def _make_fun_to_min(p: int, d: int, h: float) -> Callable[[NDArray[np.float64]], np.float64]:
        basis_list = [get_z_basis_m(m) for m in range(0, p + 1)]
        basis_list_t = [get_z_basis_m_t(basis) for basis in basis_list]

        def fun_to_min(angles: NDArray[np.float64]) -> np.float64:
            n = len(angles) // 2
            # To account for the fact that we scale ``gamma`` angles by ``np.sqrt(d-1)`` and not `np.sqrt(d)`
            gammas = angles[:n] * np.sqrt((d - 1) / d)
            betas = angles[n:]
            big_gamma = np.concatenate([gammas, [0], -gammas[::-1]])
            big_beta = np.concatenate([betas, [0], -betas[::-1]])
            z_and_zz = get_exp_vals(p, d - 1, h, big_gamma, big_beta, basis_list_t)
            return (h * z_and_zz[0] + z_and_zz[1] * d / 2).real

        return fun_to_min

    return _make_fun_to_min


@pytest.fixture
def graph_with_uncommon_node_labels() -> nx.Graph:
    """Creates an arbitrary networkx graph whose nodes are all different kinds of objects.

    In networkx, anything hashable except for `None` can be a graph node. So here we use ``string``, ``int``, ``float``
    and ``tuple[int, int]``.
    """
    g = nx.Graph()
    g.add_edges_from([("A", 25), ("B", (10, 3)), ("A", 1.4), ("A", "B"), ((10, 3), 25), ("B", 1.4)])
    return g


@pytest.fixture
def dummy_qaoa() -> QUBOQAOA:
    """Dummy QUBOQAOA instance (contains trivial cost function)."""
    dummy_qubo_instance = QUBOInstance(np.array([[0]]))
    return QUBOQAOA(dummy_qubo_instance, num_layers=1)


class SignatureRichEstimator(EstimatorBackend):
    """A class for a dummy estimator to be used during testing.

    The point here is to overwrite the `estimate` method by one which shares a keyword argument with
    :func:`scipy.optimize.minimize` to use in a test that check this case.
    """

    def estimate(self, qaoa_object: QUBOQAOA, method: str = "Arbitrary") -> float:
        """Dummy `estimate` with an extra kwarg added to test kwarg overlap with :func:`scipy.optimize.minimize`."""
        return 1337.0

    def estimate_correlations_z(
        self, qaoa_object: QUBOQAOA, target_qubits: set[LogQubit] | list[set[LogQubit]]
    ) -> float:
        """Dummy `estimate_correlations_z` because the method of the parent class is abstract."""
        return 13.37


@pytest.fixture
def dummy_estimator() -> SignatureRichEstimator:
    """Create an instance of :class:`SignatureRichEstimator`."""
    return SignatureRichEstimator()
