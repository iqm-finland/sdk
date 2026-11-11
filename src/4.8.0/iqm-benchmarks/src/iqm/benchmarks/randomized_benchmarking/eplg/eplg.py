"""Error Per Layered Gate (EPLG)."""

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from iqm.benchmarks import (
    Benchmark,
    BenchmarkAnalysisResult,
    BenchmarkCircuit,
    BenchmarkObservation,
    BenchmarkObservationIdentifier,
    BenchmarkRunResult,
)
from iqm.benchmarks.benchmark_definition import BENCHMARK_TIMESTAMP_FORMAT, BenchmarkConfigurationBase
from iqm.benchmarks.logging_config import qcvv_logger
from iqm.benchmarks.randomized_benchmarking.direct_rb.direct_rb import (
    DirectRandomizedBenchmarking,
    DirectRBConfiguration,
    direct_rb_analysis,
)
from iqm.benchmarks.randomized_benchmarking.eplg.eplg_edges import (
    DEFAULT_SEARCH_BUDGET,
    Edge,
    EdgeSelection,
    orient_to_coupling_map,
    rank_eplg_edges,
)
from iqm.benchmarks.utils import split_forest_into_layers
from iqm.benchmarks.utils_plots import (
    GraphPositions,
    chip_figure,
    draw_broken_couplers,
    get_static_connectivity,
    rx_to_nx_graph,
)
from iqm.qiskit_iqm.iqm_backend import IQMBackendBase
from matplotlib.colors import BoundaryNorm, Colormap, ListedColormap, to_rgba
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from qiskit.transpiler import CouplingMap
from uncertainties import ufloat
import xarray as xr


def plot_layered_fidelities_graph(  # noqa: PLR0913
    fidelities: dict[str, dict[str, float]],
    backend_coupling_map: CouplingMap,
    qubit_names: dict[int, str],
    num_qubits_static: int,
    timestamp: str,
    station: str | None = None,
    eplg_estimate: dict[str, float] | None = None,
    static_connectivity: Sequence[Sequence[str]] | None = None,
    disjoint_layers: Sequence[Sequence[Edge]] | None = None,
    color_by_layer: bool = False,
) -> tuple[str, Figure]:
    """Plots the layered fidelity for each corresponding pair of qubits in a graph layout of the given backend.

    The pairs are colored by fidelity and labelled with their disjoint layer, or with ``color_by_layer`` colored by
    layer and labelled with their fidelity.

    Args:
        fidelities:
            A dictionary (str qubit keys) of dictionaries (keys "value"/"uncertainty") of fidelities (float) to plot.
        backend_coupling_map: The CouplingMap instance.
        qubit_names: A dictionary of qubit names corresponding to qubit indices.
        num_qubits_static: The number of qubits in the backend according to the static quantum architecture.
        timestamp: The timestamp of the corresponding experiment.
        station: The name of the station to use for the graph layout.
        eplg_estimate: A dictionary with the EPLG estimate value and its uncertainty.
        static_connectivity: The connectivity of the static quantum architecture; if given, broken couplers are drawn
            in black and qubits not in the backend in gray.
        disjoint_layers: The disjoint layers of pairs the fidelities were measured in.
        color_by_layer: Whether to color the pairs by layer instead of by fidelity; needs ``disjoint_layers``.

    Returns:
        The figure label and the layered fidelities plot figure.

    """
    kind = "layered_fidelities_by_layer_graph" if color_by_layer else "layered_fidelities_graph"
    fig_name = f"{kind}_{station}_{timestamp}" if station is not None else f"{kind}_{timestamp}"

    fidelity_edges = {
        tuple(int(num) for num in key.strip("()[]").replace("...", "").split(", ")): 100 * value["value"]
        for key, value in fidelities.items()
    }
    fidelity_values = list(fidelity_edges.values())
    layer_of_edge = {tuple(pair): k for k, layer in enumerate(disjoint_layers or []) for pair in layer}
    num_layers = len(disjoint_layers or [])

    cmap: Colormap
    norm: plt.Normalize
    if color_by_layer:
        cmap = ListedColormap(plt.colormaps["tab10"](range(num_layers)))
        norm = BoundaryNorm(np.arange(num_layers + 1) - 0.5, num_layers)
        edge_values: Mapping[tuple[int, ...], float] = layer_of_edge
        edge_labels = {edge: f"{value:.2f}" for edge, value in fidelity_edges.items()}
    else:
        cmap = plt.colormaps["winter"]
        norm = plt.Normalize(vmin=min(fidelity_values), vmax=max(fidelity_values))
        edge_values = fidelity_edges
        edge_labels = {edge: f"L{k}" for edge, k in layer_of_edge.items()}

    qubit_positions_static = GraphPositions.get_positions(
        graph=backend_coupling_map.graph.to_undirected(multigraph=False),
        num_qubits=num_qubits_static,
    )
    static_indices = {k: int(v[2:]) - 1 for k, v in qubit_names.items()}
    qubit_positions = {k: qubit_positions_static[v] for k, v in static_indices.items()}

    fig, ax = chip_figure(qubit_positions_static)
    if static_connectivity is not None:
        draw_broken_couplers(ax, static_connectivity, backend_coupling_map, qubit_names, qubit_positions_static)

    edge_colors = []
    for edge in backend_coupling_map:
        if edge in edge_values:
            edge_colors.append(cmap(norm(edge_values[edge])))
        elif (edge[1], edge[0]) in edge_values:
            edge_colors.append(cmap(norm(edge_values[(edge[1], edge[0])])))
        else:
            edge_colors.append(to_rgba("lightgray"))

    nodes = list({v for edge in backend_coupling_map for v in edge})
    active_nodes = list({v for edge in fidelity_edges for v in edge})
    node_colors = ["lightgray" if v not in active_nodes else "k" for v in nodes]

    nx.draw_networkx(
        rx_to_nx_graph(backend_coupling_map),
        pos=qubit_positions,
        nodelist=nodes,
        edgelist=list(backend_coupling_map),
        labels={x: qubit_names[x] for x in nodes},
        font_size=4.5,
        width=4.0,
        edge_color=edge_colors,
        node_color=node_colors,
        node_size=200,
        font_color="w",
        ax=ax,
    )

    # Label each pair at its midpoint
    for (a, b), label in edge_labels.items():
        (x1, y1), (x2, y2) = qubit_positions[a], qubit_positions[b]
        ax.annotate(
            label,
            xy=((x1 + x2) / 2, (y1 + y2) / 2),
            ha="center",
            va="center",
            fontsize=4.5,
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.8},
            zorder=5,
        )

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    if color_by_layer:
        cbar = fig.colorbar(sm, ax=ax, shrink=0.5, label="Disjoint layer")
        cbar.set_ticks(range(num_layers), labels=[f"L{k}" for k in range(num_layers)])
    else:
        cbar = fig.colorbar(sm, ax=ax, shrink=0.5, label="Layered Fidelity (%)", format="%.2f")
        cbar.set_ticks(tuple(np.linspace(min(fidelity_values), max(fidelity_values), 5, endpoint=True)))

    station_string = "IQM Backend" if station is None else station.capitalize()

    eplg_string = (
        f"EPLG estimate: {eplg_estimate['value']:.2e} +/- {eplg_estimate['uncertainty']:.2e}\n" if eplg_estimate else ""
    )
    if color_by_layer:
        heading = "Layered fidelities (%) for qubit pairs"
        layers_string = f"Colors: disjoint layer of each pair ({num_layers} layers)\n"
    else:
        heading = "Layered fidelities for qubit pairs"
        layers_string = f"Labels: disjoint layer of each pair ({num_layers} layers)\n" if num_layers else ""
    plt.title(f"{heading} in {station_string}\n{eplg_string}{layers_string}{timestamp}")
    plt.gca().invert_yaxis()
    plt.close()

    return fig_name, fig


def eplg_analysis(run: BenchmarkRunResult) -> BenchmarkAnalysisResult:
    """EPLG analysis function.

    Args:
        run: The result of the benchmark run.

    Returns:
        AnalysisResult corresponding to EPLG.

    """
    result_direct_rb = direct_rb_analysis(run)

    dataset = result_direct_rb.dataset.copy(deep=True)
    observations = result_direct_rb.observations
    plots: dict[str, Figure] = {}

    timestamp = dataset.attrs["execution_timestamp"]

    backend_name = dataset.attrs["backend_name"]
    backend_coupling_map = dataset.attrs["backend_coupling_map"]

    num_qubits = dataset.attrs["num_qubits"]
    edges = dataset.attrs["edges"]
    disjoint_layers = dataset.attrs["disjoint_layers"]
    qubit_names = dataset.attrs["qubit_names"]

    # The EPLG exponent is the number of 2Q gates (e.g. N-1 for a chain)
    n_2q_gates = len(edges)

    fidelities = {}
    fid_product = ufloat(1, 0)
    for obs in observations:
        # Only fidelities enter the product, should direct RB ever report other observations
        if obs.name != "average_gate_fidelity":
            continue
        # The layer fidelity product includes every measured group (2Q gates and idle qubits)
        fid_product *= ufloat(obs.value, obs.uncertainty)
        # Prepare list of 2q fidelities for plot
        if obs.identifier.qubit_indices is not None and len(obs.identifier.qubit_indices) == 2:  # noqa: PLR2004
            fidelities[str(obs.identifier.qubit_indices)] = {
                "value": obs.value,
                "uncertainty": obs.uncertainty,
            }

    lf = fid_product
    eplg = 1 - lf ** (1 / n_2q_gates)

    observations.append(
        BenchmarkObservation(
            name="layer_fidelity",
            identifier=BenchmarkObservationIdentifier(custom_identifier=f"(n_qubits={num_qubits})"),
            value=lf.nominal_value,
            uncertainty=lf.std_dev,
        )
    )

    observations.append(
        BenchmarkObservation(
            name="eplg",
            identifier=BenchmarkObservationIdentifier(custom_identifier=f"(n_qubits={num_qubits})"),
            value=eplg.nominal_value,
            uncertainty=eplg.std_dev,
        )
    )

    # Plot the layered fidelities graph, colored by fidelity and then by layer
    for color_by_layer in (False, True):
        fig_name, fig = plot_layered_fidelities_graph(
            fidelities=fidelities,
            backend_coupling_map=backend_coupling_map,
            qubit_names=qubit_names,
            num_qubits_static=dataset.attrs["num_qubits_static"],
            timestamp=timestamp,
            station=backend_name,
            eplg_estimate={"value": eplg.nominal_value, "uncertainty": eplg.std_dev},
            static_connectivity=dataset.attrs.get("static_connectivity"),
            disjoint_layers=disjoint_layers,
            color_by_layer=color_by_layer,
        )
        plots[fig_name] = fig

    plots.update(result_direct_rb.plots)

    return BenchmarkAnalysisResult(dataset=dataset, observations=observations, plots=plots)


class EPLGBenchmark(Benchmark):
    """EPLG estimates the layer fidelity of native 2Q gate layers."""

    analysis_function = staticmethod(eplg_analysis)

    @classmethod
    def name(cls) -> str:
        """Returns the name of the benchmark."""
        return "eplg"

    def __init__(self, backend_arg: IQMBackendBase | str, configuration: "EPLGConfiguration"):
        """Construct the EPLG class.

        Args:
            backend_arg: The backend to use for the benchmark,
                either as a backend instance or a backend name string.
            configuration: The configuration settings for the EPLG benchmark.

        """
        super().__init__(backend_arg, configuration)
        # EXPERIMENT
        self.session_timestamp = datetime.now(UTC).strftime(BENCHMARK_TIMESTAMP_FORMAT)
        self.execution_timestamp = ""

        # Initialize the variable to contain the circuits for each layout
        self.untranspiled_circuits = BenchmarkCircuit("untranspiled_circuits")
        self.transpiled_circuits = BenchmarkCircuit("transpiled_circuits")

        self.drb_depths = configuration.drb_depths
        self.drb_circuit_samples = configuration.drb_circuit_samples

        self.custom_qubits_array = configuration.custom_qubits_array
        self.excluded_qubits = configuration.excluded_qubits

        self.edge_selection = configuration.edge_selection
        self.num_qubits = configuration.num_qubits
        self.num_disjoint_layers = configuration.num_disjoint_layers
        self.edge_fidelities = configuration.edge_fidelities
        self.search_budget = configuration.search_budget
        self.edge_selection_rank = configuration.edge_selection_rank

    def add_all_meta_to_dataset(self, dataset: xr.Dataset) -> None:
        """Adds all configuration metadata and circuits to the dataset variable.

        Args:
            dataset: The xarray dataset

        """
        dataset.attrs["session_timestamp"] = self.session_timestamp
        dataset.attrs["execution_timestamp"] = self.execution_timestamp
        dataset.attrs["backend_name"] = self.backend_name
        dataset.attrs["backend_coupling_map"] = self.backend.coupling_map
        dataset.attrs["backend_num_qubits"] = self.backend.num_qubits
        dataset.attrs["num_qubits_static"] = self.num_qubits_static
        dataset.attrs["static_connectivity"] = get_static_connectivity(self.backend)

        for key, value in self.configuration:
            if key == "benchmark":  # Avoid saving the class object
                dataset.attrs[key] = value.name
            else:
                dataset.attrs[key] = value
        # Defined outside configuration - if any

    def _qubit_index(self, qubit: int | str) -> int:
        """The backend index of a qubit given by index or name."""
        index = self.backend.qubit_name_to_index(qubit) if isinstance(qubit, str) else int(qubit)
        if index is None or not 0 <= index < self.backend.num_qubits:
            raise ValueError(f"Qubit {qubit} is not a qubit of the backend.")
        return index

    def validate_custom_qubits_array(self) -> tuple[list[Edge], list[list[Edge]] | None]:
        """Validates the custom qubits array input.

        Returns:
            The qubit pairs, and the layers if the input is given as explicit layers of pairs.

        Raises:
            ValueError: If an item is not a pair of two different qubits, a pair is repeated, or two pairs of the same
                layer share a qubit.

        """
        custom: list[Any] = list(self.custom_qubits_array or [])
        is_layered = bool(custom) and not isinstance(custom[0][0], int | np.integer)
        layers = [[tuple(pair) for pair in layer] for layer in custom] if is_layered else [[tuple(p) for p in custom]]
        pairs = [pair for layer in layers for pair in layer]
        pair_length = 2
        if not pairs or any(len(pair) != pair_length or pair[0] == pair[1] for pair in pairs):
            raise ValueError(
                "The custom qubits array must be a Sequence of pairs of different qubits, or of layers of such pairs."
            )
        if len({frozenset(pair) for pair in pairs}) != len(pairs):
            raise ValueError("The custom qubits array must have unique qubit pairs.")
        for layer in layers:
            layer_qubits = [q for pair in layer for q in pair]
            if is_layered and len(set(layer_qubits)) != len(layer_qubits):
                raise ValueError(f"The pairs of a layer must be disjoint, but layer {layer} uses a qubit twice.")
        return pairs, (layers if is_layered else None)

    def _custom_edges_and_layers(self, excluded: Sequence[int]) -> tuple[list[Edge], list[list[Edge]]]:
        """The edges and layers given by ``custom_qubits_array``, oriented as in the coupling map."""
        pairs, layers = self.validate_custom_qubits_array()
        couplers = {frozenset(edge) for edge in self.backend.coupling_map.get_edges()}
        not_couplers = [pair for pair in pairs if frozenset(pair) not in couplers]
        if not_couplers:
            raise ValueError(f"The custom qubits array pairs {not_couplers} are not couplers of the backend.")
        on_excluded = [pair for pair in pairs if set(pair) & set(excluded)]
        if on_excluded:
            raise ValueError(f"The custom qubits array pairs {on_excluded} use excluded qubits.")

        edges = orient_to_coupling_map(self.backend, pairs)
        if layers is not None:
            all_disjoint = [orient_to_coupling_map(self.backend, layer) for layer in layers]
        else:
            all_disjoint = split_forest_into_layers(edges, self.num_disjoint_layers)
        qcvv_logger.info(f"Using custom_qubits_array: {len(edges)} pairs in {len(all_disjoint)} disjoint layers.")
        return edges, all_disjoint

    def _selected_edges_and_layers(
        self, excluded: Sequence[int], dataset: xr.Dataset
    ) -> tuple[list[Edge], list[list[Edge]]]:
        """The edges chosen by ``edge_selection`` and their layers; stores the ranking in the dataset."""
        num_qubits = self.num_qubits or self.backend.num_qubits - len(excluded)
        edge_fidelities = None
        if self.edge_fidelities is not None:
            edge_fidelities = {
                (self._qubit_index(a), self._qubit_index(b)): fidelity
                for (a, b), fidelity in self.edge_fidelities.items()
            }
        qcvv_logger.info(f"Ranking {num_qubits}-qubit {self.edge_selection} selections by predicted layer fidelity")
        ranked = rank_eplg_edges(
            self.backend,
            num_qubits,
            self.edge_selection,
            excluded_qubits=excluded,
            num_layers=self.num_disjoint_layers,
            edge_fidelities=edge_fidelities,
            num_ranked=max(5, self.edge_selection_rank + 1),
            search_budget=self.search_budget,
        )
        for rank, candidate in enumerate(ranked):
            qcvv_logger.info(
                f"Rank {rank}: {candidate.label} in {candidate.num_layers} layers, cost {candidate.cost:.4f}, "
                f"edges {candidate.edges}"
            )
        if self.edge_selection_rank >= len(ranked):
            raise ValueError(
                f"edge_selection_rank={self.edge_selection_rank} was requested, but only {len(ranked)} selections "
                f"were found."
            )
        chosen = ranked[self.edge_selection_rank]
        qcvv_logger.info(f"Running rank {self.edge_selection_rank}, scored with {chosen.edge_fidelity_source}")
        dataset.attrs["edge_fidelity_source"] = chosen.edge_fidelity_source
        dataset.attrs["ranked_selections"] = [
            {"rank": rank, "label": c.label, "num_layers": c.num_layers, "cost": c.cost, "edges": c.edges}
            for rank, c in enumerate(ranked)
        ]
        return chosen.edges, split_forest_into_layers(chosen.edges, chosen.num_layers)

    def execute(self, backend: IQMBackendBase) -> xr.Dataset:
        """Execute the EPLG Benchmark."""
        self.execution_timestamp = datetime.now(UTC).strftime(BENCHMARK_TIMESTAMP_FORMAT)

        dataset_eplg = xr.Dataset()

        excluded = sorted({self._qubit_index(q) for q in self.excluded_qubits or []})
        if self.custom_qubits_array is not None:
            edges, all_disjoint = self._custom_edges_and_layers(excluded)
        else:
            edges, all_disjoint = self._selected_edges_and_layers(excluded, dataset_eplg)

        backend_qubits = list(range(backend.num_qubits))
        dataset_eplg.attrs["qubit_names"] = {qubit: self.backend.index_to_qubit_name(qubit) for qubit in backend_qubits}

        self.add_all_meta_to_dataset(dataset_eplg)
        # After the configuration, whose num_qubits may be None: the number of qubits that were benchmarked
        dataset_eplg.attrs["num_qubits"] = len({q for edge in edges for q in edge})

        # Add idle qubits as single-qubit DRB groups alongside the 2Q-gate groups.
        chain_qubits = {q for edge in edges for q in edge}
        drb_qubits_array = [
            [list(pair) for pair in layer] + [[q] for q in sorted(chain_qubits - {q for pair in layer for q in pair})]
            for layer in all_disjoint
        ]

        # Execute parallel DRB in all disjoint layers
        drb_config = DirectRBConfiguration(
            qubits_array=drb_qubits_array,
            is_eplg=True,
            depths=self.drb_depths,
            num_circuit_samples=self.drb_circuit_samples,
            shots=self.shots,
            max_gates_per_batch=self.max_gates_per_batch,
            max_circuits_per_batch=self.configuration.max_circuits_per_batch,
            use_dd=self.configuration.use_dd,
            dd_strategy=self.configuration.dd_strategy,
            active_reset_cycles=self.configuration.active_reset_cycles,
        )

        benchmarks_direct_rb = DirectRandomizedBenchmarking(backend, drb_config)
        run_direct_rb = benchmarks_direct_rb.run()
        dataset = run_direct_rb.dataset
        self.circuits = benchmarks_direct_rb.circuits
        dataset_eplg.attrs.update({"disjoint_layers": all_disjoint, "edges": edges})
        dataset.attrs.update(dataset_eplg.attrs)

        return dataset


class EPLGConfiguration(BenchmarkConfigurationBase):
    """EPLG Configuration.

    Attributes:
        drb_depths: Layer depths to consider for the parallel DRB.
        drb_circuit_samples: Number of circuit samples to consider for the parallel DRB.
        custom_qubits_array: Qubit pairs to benchmark, as a ``Sequence`` of pairs (``[(0, 1), (1, 2)]``), split into
            disjoint layers automatically, or as explicit layers of disjoint pairs (``[[(0, 1), (2, 3)], [(1, 2)]]``).
            Default is None: the pairs are selected by ``edge_selection``.
        excluded_qubits: Qubits, by index or name, that the edge selection must avoid. Default is None.
        edge_selection: How to select the pairs if ``custom_qubits_array`` is not given: ``"tree"`` (default) for the
            best chain or tree needing up to 4 layers, ``"chain"`` for the best chain.
        num_qubits: Number of qubits to select pairs for. Default is None: all qubits that are not excluded.
        num_disjoint_layers: Number of disjoint layers to split the pairs into. Default is None: the fewest possible.
        edge_fidelities: Fidelities to select the pairs with, keyed by qubit pair (indices or names); only these pairs
            can be selected. Default is None: the calibrated 2Q Clifford fidelities, or the CZ fidelities.
        search_budget: Maximum number of steps of the chain search. Default is 10**6.
        edge_selection_rank: Which of the ranked selections to run, 0 being the best. Default is 0.

    """

    benchmark: type[Benchmark] = EPLGBenchmark
    drb_depths: Sequence[int]
    drb_circuit_samples: int
    custom_qubits_array: Sequence[tuple[int, int]] | Sequence[Sequence[tuple[int, int]]] | None = None
    excluded_qubits: Sequence[int | str] | None = None
    edge_selection: EdgeSelection = "tree"
    num_qubits: int | None = None
    num_disjoint_layers: int | None = None
    edge_fidelities: Mapping[tuple[int, int] | tuple[str, str], float] | None = None
    search_budget: int = DEFAULT_SEARCH_BUDGET
    edge_selection_rank: int = 0
