"""Error Per Layered Gate (EPLG)."""

from collections.abc import Sequence
from datetime import datetime, timezone
from typing import cast

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
from iqm.benchmarks.utils import split_into_disjoint_pairs
from iqm.benchmarks.utils_plots import (
    GraphPositions,
    draw_graph_edges,
    evaluate_hamiltonian_paths,
    rx_to_nx_graph,
)
from iqm.qiskit_iqm.iqm_backend import IQMBackendBase
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from qiskit.transpiler import CouplingMap
from uncertainties import ufloat
import xarray as xr


def plot_layered_fidelities_graph(
    fidelities: dict[str, dict[str, float]],
    backend_coupling_map: CouplingMap,
    qubit_names: dict[int, str],
    timestamp: str,
    station: str | None = None,
    eplg_estimate: dict[str, float] | None = None,
) -> tuple[str, Figure]:
    """Plots the layered fidelity for each corresponding pair of qubits in a graph layout of the given backend.

    Args:
        fidelities:
            A dictionary (str qubit keys) of dictionaries (keys "value"/"uncertainty") of fidelities (float) to plot.
        backend_coupling_map: The CouplingMap instance.
        qubit_names: A dictionary of qubit names corresponding to qubit indices.
        timestamp: The timestamp of the corresponding experiment.
        station: The name of the station to use for the graph layout.
        eplg_estimate: A dictionary with the EPLG estimate value and its uncertainty.

    Returns:
        The figure label and the layered fidelities plot figure.

    """
    num_qubits = len(qubit_names.keys())
    fig_name = (
        f"layered_fidelities_graph_{station}_{timestamp}"
        if station is not None
        else f"layered_fidelities_graph_{timestamp}"
    )
    # Sort the fidelities by value
    sorted_fidelities = dict(sorted(fidelities.items(), key=lambda item: item[1]["value"]))

    qubit_pairs = [
        tuple(int(num) for num in x.replace("(", "").replace(")", "").replace("...", "").split(", "))
        for x in sorted_fidelities.keys()
    ]
    fidelity_values = [100 * a["value"] for a in sorted_fidelities.values()]

    fidelity_edges = dict(zip(qubit_pairs, fidelity_values, strict=True))

    cmap = plt.colormaps["winter"]

    fig = plt.figure()
    ax = plt.axes()

    qubit_positions = GraphPositions.get_positions(
        station=station,
        graph=backend_coupling_map.graph.to_undirected(multigraph=False),
        num_qubits=num_qubits,
    )

    # Normalize fidelity values to the range [0, 1] for color mapping
    norm = plt.Normalize(vmin=min(fidelity_values), vmax=max(fidelity_values))
    edge_colors = []
    for edge in backend_coupling_map:
        if edge in fidelity_edges:
            edge_colors.append(cmap(norm(fidelity_edges[edge])))
        elif (edge[1], edge[0]) in fidelity_edges:
            edge_colors.append(cmap(norm(fidelity_edges[(edge[1], edge[0])])))
        else:
            edge_colors.append(to_rgba("lightgray"))

    nodes = list({v for edge in backend_coupling_map for v in edge})
    active_nodes = list({v for edge in qubit_pairs for v in edge})
    node_colors = ["lightgray" if v not in active_nodes else "k" for v in nodes]

    nx.draw_networkx(
        rx_to_nx_graph(backend_coupling_map),
        pos=qubit_positions,
        nodelist=nodes,
        edgelist=list(backend_coupling_map),
        labels={x: qubit_names[x] for x in nodes},
        font_size=6.5,
        width=4.0,
        edge_color=edge_colors,
        node_color=node_colors,
        font_color="w",
        ax=ax,
    )

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, label="Layered Fidelity (%)", format="%.2f")
    cbar.set_ticks(tuple(np.linspace(min(fidelity_values), max(fidelity_values), 5, endpoint=True)))

    station_string = "IQM Backend" if station is None else station.capitalize()

    eplg_string = (
        f"EPLG estimate: {eplg_estimate['value']:.2e} +/- {eplg_estimate['uncertainty']:.2e}\n" if eplg_estimate else ""
    )
    plt.title(f"Layered fidelities for qubit pairs in {station_string}\n{eplg_string}{timestamp}")
    plt.gca().invert_yaxis()
    plt.close()

    return fig_name, fig


def eplg_analysis(run: BenchmarkRunResult) -> BenchmarkAnalysisResult:
    """EPLG analysis function.

    Args:
        run: The result of the benchmark run.

    Returns:
        AnalysisResult corresponding to DRB.

    """
    result_direct_rb = direct_rb_analysis(run)

    dataset = result_direct_rb.dataset.copy(deep=True)
    observations = result_direct_rb.observations
    plots: dict[str, Figure] = {}

    timestamp = dataset.attrs["execution_timestamp"]

    backend_name = dataset.attrs["backend_name"]
    backend_coupling_map = dataset.attrs["backend_coupling_map"]
    backend_num_qubits = dataset.attrs["backend_num_qubits"]

    num_edges = len(observations)
    num_qubits = dataset.attrs["num_qubits"]
    edges = dataset.attrs["edges"]
    disjoint_layers = dataset.attrs["disjoint_layers"]
    qubit_names = dataset.attrs["qubit_names"]

    fidelities = {}
    fid_product = ufloat(1, 0)
    for obs in observations:
        fid_product *= ufloat(obs.value, obs.uncertainty)
        fidelities[str(obs.identifier.qubit_indices)] = {
            "value": obs.value,
            "uncertainty": obs.uncertainty,
        }

    lf = fid_product
    eplg = 1 - lf ** (1 / num_edges)

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

    # Plot the edges graph
    fig_name, fig = draw_graph_edges(
        backend_coupling_map,
        backend_num_qubits=backend_num_qubits,
        edge_list=edges,
        timestamp=timestamp,
        station=backend_name,
        disjoint_layers=disjoint_layers,
        qubit_names=qubit_names,
        is_eplg=True,
    )
    plots[fig_name] = fig

    # Plot the layered fidelities graph
    fig_name, fig = plot_layered_fidelities_graph(
        fidelities=fidelities,
        backend_coupling_map=backend_coupling_map,
        qubit_names=qubit_names,
        timestamp=timestamp,
        station=backend_name,
        eplg_estimate={"value": eplg.nominal_value, "uncertainty": eplg.std_dev},
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
        self.session_timestamp = datetime.now(timezone.utc).strftime(BENCHMARK_TIMESTAMP_FORMAT)
        self.execution_timestamp = ""

        # Initialize the variable to contain the circuits for each layout
        self.untranspiled_circuits = BenchmarkCircuit("untranspiled_circuits")
        self.transpiled_circuits = BenchmarkCircuit("transpiled_circuits")

        self.drb_depths = configuration.drb_depths
        self.drb_circuit_samples = configuration.drb_circuit_samples

        self.custom_qubits_array = configuration.custom_qubits_array

        self.chain_length = configuration.chain_length
        self.chain_path_samples = configuration.chain_path_samples
        self.num_disjoint_layers = configuration.num_disjoint_layers
        self.max_hamiltonian_path_tries = configuration.max_hamiltonian_path_tries

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

        for key, value in self.configuration:
            if key == "benchmark":  # Avoid saving the class object
                dataset.attrs[key] = value.name
            else:
                dataset.attrs[key] = value
        # Defined outside configuration - if any

    def validate_custom_qubits_array(self) -> None:
        """Validates the custom qubits array input ."""
        if self.custom_qubits_array is not None:
            # Validate that the custom qubits array is a list of pairs
            pair_length = 2
            if not all(isinstance(pair, tuple) and len(pair) == pair_length for pair in self.custom_qubits_array):
                raise ValueError("The custom qubits array must be a Sequence of tuples.")
            # Validate that the custom qubits array has no repeated qubits
            if len({tuple(sorted(x)) for x in self.custom_qubits_array}) != len(self.custom_qubits_array):
                raise ValueError("The custom qubits array must have unique qubit pairs.")

    def validate_random_chain_inputs(self) -> None:
        """Validates inputs for chain sampling.

        Raises:
            ValueError: If the chain inputs are beyond general or EPLG criteria.

        """
        # Check chain length
        if self.chain_length is None:
            qcvv_logger.warning("chain_length input was None: will assign backend.num_qubits!")
            self.chain_length = self.backend.num_qubits
        elif self.chain_length > self.backend.num_qubits:
            raise ValueError("The chain length cannot exceed the number of qubits in the backend.")

        # Check path samples
        if self.chain_path_samples is None:
            self.chain_path_samples = 20
        elif self.chain_path_samples < 1:
            raise ValueError("The number of chain path samples must be a positive integer.")

        if self.num_disjoint_layers is None:
            self.num_disjoint_layers = 2
        elif self.num_disjoint_layers < 1:
            raise ValueError("The number of disjoint layers must be a positive integer.")

        if self.max_hamiltonian_path_tries is None:
            self.max_hamiltonian_path_tries = 10
        elif self.max_hamiltonian_path_tries < 1:
            raise ValueError("The maximum number of Hamiltonian path tries must be a positive integer.")

    def execute(self, backend: IQMBackendBase) -> xr.Dataset:
        """Execute the EPLG Benchmark."""
        self.execution_timestamp = datetime.now(timezone.utc).strftime(BENCHMARK_TIMESTAMP_FORMAT)

        dataset_eplg = xr.Dataset()

        if self.custom_qubits_array is not None:
            self.validate_custom_qubits_array()
            edges = self.custom_qubits_array
            num_qubits = len(list({x for y in edges for x in y}))
            all_disjoint = split_into_disjoint_pairs(self.custom_qubits_array)
            self.num_disjoint_layers = len(all_disjoint)
            qcvv_logger.info(
                f"Using specified custom_qubits_array: will split into {self.num_disjoint_layers} disjoint layers."
            )

        else:
            self.validate_random_chain_inputs()
            num_qubits = cast(int, self.chain_length)
            qcvv_logger.info("Generating linear chain path")
            if (
                self.chain_length is not None
                and self.chain_path_samples is not None
                and self.max_hamiltonian_path_tries is not None
            ):
                h_path_costs = evaluate_hamiltonian_paths(
                    self.chain_length,
                    self.chain_path_samples,
                    self.backend,
                    self.max_hamiltonian_path_tries,
                )
            qcvv_logger.info("Extracting the path that maximizes total 2Q calibration fidelity")
            max_cost_path = h_path_costs[max(h_path_costs.keys())]

            all_disjoint = [
                max_cost_path[i :: self.num_disjoint_layers] for i in range(cast(int, self.num_disjoint_layers))
            ]
            edges = max_cost_path

        dataset_eplg.attrs["num_qubits"] = num_qubits
        backend_qubits = list(range(backend.num_qubits))
        dataset_eplg.attrs["qubit_names"] = {qubit: self.backend.index_to_qubit_name(qubit) for qubit in backend_qubits}

        self.add_all_meta_to_dataset(dataset_eplg)

        # Execute parallel DRB in all disjoint layers
        drb_config = DirectRBConfiguration(
            qubits_array=all_disjoint,
            is_eplg=True,
            depths=self.drb_depths,
            num_circuit_samples=self.drb_circuit_samples,
            shots=self.shots,
            max_gates_per_batch=self.max_gates_per_batch,
            max_circuits_per_batch=self.configuration.max_circuits_per_batch,
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
        custom_qubits_array: Custom qubits array to consider; this corresponds to a ``Sequence`` of tuple pairs of
            qubits. If not specified, will proceed to generate linear chains at random, selecting the one with the
            highest total 2Q gate fidelity.
        chain_length: Length of a linear chain of 2Q gates to consider, corresponding to the number of qubits, if
            ``custom_qubits_array`` not specified. Default is None: assigns the number of qubits in the backend minus
            one.
        chain_path_samples: Number of chain path samples to consider, if ``custom_qubits_array`` not specified. Default
            is None: assigns 20 path samples (arbitrary).
        num_disjoint_layers: Number of disjoint layers to consider. Default is None: assigns 2 disjoint layers
            (arbitrary).
        max_hamiltonian_path_tries: Maximum number of tries to find a Hamiltonian path. Default is None: assigns 10
            tries (arbitrary).

    """

    benchmark: type[Benchmark] = EPLGBenchmark
    drb_depths: Sequence[int]
    drb_circuit_samples: int
    custom_qubits_array: Sequence[tuple[int, int]] | None = None
    chain_length: int | None = None
    chain_path_samples: int | None = None
    num_disjoint_layers: int | None = None
    max_hamiltonian_path_tries: int | None = None
