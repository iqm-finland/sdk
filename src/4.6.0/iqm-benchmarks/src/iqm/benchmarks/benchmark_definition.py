# Copyright 2024 IQM Benchmarks developers
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

"""Implementation of the base class for defining a benchmark."""

from abc import ABC, abstractmethod
from collections.abc import Callable
import copy
from copy import deepcopy
from dataclasses import dataclass, field
import functools
from typing import Any

from iqm.benchmarks.circuit_containers import BenchmarkCircuit, Circuits
from iqm.benchmarks.logging_config import qcvv_logger
from iqm.benchmarks.utils import PhysicalLayout, RoutingMethod, get_iqm_backend, process_backend, timeit
from iqm.iqm_client.models import CircuitCompilationOptions, DDMode, DDStrategy
from iqm.qiskit_iqm.iqm_backend import IQMBackendBase
from iqm.qiskit_iqm.iqm_provider import IQMBackend, IQMFacadeBackend
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from pydantic import BaseModel
import xarray as xr

BACKEND_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S.%f%z"
BENCHMARK_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S%z"


@dataclass
class BenchmarkObservationIdentifier:
    """Identifier for observations for ease of use.

    Attributes:
        qubit_indices: list containing the indices of the qubits the observation was executed on.
        custom_identifier: optional user-defined string identifier.

    """

    qubit_indices: list[int] | None = None
    custom_identifier: str | None = None

    @property
    def string_identifier(self) -> str:
        """String version of the qubit indices for ease of use, or custom identifier if given.

        Returns:
            A string of the qubit indices

        """
        if self.custom_identifier is not None:
            identifier = self.custom_identifier
        elif self.qubit_indices:
            identifier = str(self.qubit_indices)
        else:
            raise ValueError("Either qubit_indices or custom_identifier must be provided.")
        return identifier


@dataclass
class BenchmarkObservation:
    """Dataclass to store the main results of a single run of a Benchmark.

    Attributes:
        name: name of the observation
        value: value of the observation
        identifier: identifier, which should be a string of the qubit layout
        uncertainty: uncertainty of the observation

    """

    name: str
    value: Any
    identifier: BenchmarkObservationIdentifier
    uncertainty: Any | None = None


@dataclass
class BenchmarkRunResult:
    """A dataclass that stores the results of a single run of a Benchmark.

    RunResult should contain enough information that the Benchmark can be analyzed based on those
    results.
    """

    dataset: xr.Dataset
    circuits: Circuits


@dataclass
class BenchmarkAnalysisResult:
    """A dataclass storing the results of the analysis.

    The result consists of a dataset, plots, and observations. Plots are defined as a dictionary
    that maps a plot name to a figure. Observations are key-value pairs of data that contain the
    main results of the benchmark.
    """

    dataset: xr.Dataset
    circuits: Circuits | None = field(default=None)
    plots: dict[str, Figure] = field(default_factory=lambda: ({}))
    observations: list[BenchmarkObservation] = field(default_factory=list)

    def plot(self, plot_name: str) -> Figure:
        """Plots the given figure.

        Args:
            plot_name: Name of the figure to be plotted.

        """
        return self.plots[plot_name]

    def plot_all(self) -> None:
        """Plots all the figures defined for the analysis."""
        for fig in self.plots.values():
            show_figure(fig)
            plt.show()

    @classmethod
    def from_run_result(cls, run: BenchmarkRunResult) -> "BenchmarkAnalysisResult":
        """Creates a new ``AnalysisResult`` from a ``RunResult``.

        Args:
            run: A run for which analysis result is created.

        """
        return cls(dataset=run.dataset, circuits=run.circuits)


def default_analysis_function(result: BenchmarkRunResult) -> BenchmarkAnalysisResult:
    """The default analysis that only pass the result through."""
    return BenchmarkAnalysisResult.from_run_result(result)


def merge_datasets_dac(datasets: list[xr.Dataset]) -> xr.Dataset:
    """Merges a list of datasets recursively to minimize dataset sizes during merge.

    Args:
        datasets: A list of xarray datasets

    Returns:
        A list containing a single merged dataset

    """
    if len(datasets) == 1:
        return datasets[0]
    datasets_new = []
    for i in range(0, len(datasets), 2):
        if i == (len(datasets) - 1):
            datasets_new.append(datasets[i])
        else:
            datasets_new.append(xr.merge(datasets[i : i + 2]))
    return merge_datasets_dac(datasets_new)


@timeit
def add_counts_to_dataset(counts: list[dict[str, int]], identifier: str, dataset: xr.Dataset) -> xr.Dataset:
    """Adds the counts from a cortex job result to the given dataset.

    If counts with the same identifier are already present in the old dataset, then both counts are added together.

    Args:
        counts: A list of dictionaries with counts of bitstrings.
        identifier: A string to identify the current data, for instance the qubit layout.
        dataset: Dataset to add results to.

    Returns:
        A merged dataset where the new counts are added the input dataset

    """
    qcvv_logger.info("Adding counts to dataset")
    if not isinstance(counts, list):
        counts = [counts]
    datasets = []
    for ii, _ in enumerate(counts):
        ds_temp = xr.Dataset()
        counts_dict = dict(counts[ii])
        counts_array = xr.DataArray(
            list(counts_dict.values()),
            {f"{identifier}_state_{ii}": list(counts_dict.keys())},
        )
        if f"{identifier}_counts_{ii}" in dataset.variables:
            counts_array = counts_array + dataset[f"{identifier}_counts_{ii}"]
        counts_array = counts_array.sortby(counts_array[f"{identifier}_state_{ii}"])
        ds_temp.update({f"{identifier}_counts_{ii}": counts_array})
        datasets.append(ds_temp)
    dataset_new = merge_datasets_dac(datasets)
    dataset_merged = dataset.merge(dataset_new, compat="override")
    return dataset_merged


def show_figure(fig: Figure) -> None:
    """Shows a closed figure.

    Args:
        fig: Figure to show.

    """
    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    if new_manager is not None:
        fig_canvas = new_manager.canvas
        if fig_canvas is not None:
            fig_canvas = new_manager.canvas
            fig_canvas.figure = fig
            fig.set_canvas(fig_canvas)


class Benchmark(ABC):
    """A base class for running cortex-based Benchmark experiments.

    In order to write a new benchmark, it is recommended to derive a class
    from this baseclass. A new benchmark is defined by deriving from this base
    class and implementing the execute method. Additionally, a custom analysis
    for the benchmark can be implemented by giving the pointer to the analysis
    method in ``analysis_function`` field. The given analysis_function should
    accept ``AnalysisResult`` as its input and return the final result.
    """

    analysis_function: Callable[[BenchmarkRunResult], BenchmarkAnalysisResult] = staticmethod(default_analysis_function)
    default_options: dict[str, Any] | None = None
    options: dict[str, Any] | None = None

    def __init__(
        self,
        backend: str | IQMBackendBase,
        configuration: "BenchmarkConfigurationBase",
        **kwargs: dict[str, Any],
    ) -> None:
        self.configuration = configuration
        self.serializable_configuration = deepcopy(self.configuration)
        self.serializable_configuration.benchmark = self.name()

        self.circuits = Circuits()

        if isinstance(backend, str):
            self.backend = get_iqm_backend(backend)
        else:
            self.backend = backend
        self.backend, self.backend_name = process_backend(backend)

        self.shots = self.configuration.shots
        self.max_gates_per_batch = self.configuration.max_gates_per_batch
        self.max_circuits_per_batch = self.configuration.max_circuits_per_batch

        self.routing_method = self.configuration.routing_method
        self.physical_layout = self.configuration.physical_layout

        self.transpiled_circuits: BenchmarkCircuit
        self.untranspiled_circuits: BenchmarkCircuit

        self.options = copy.copy(self.default_options) if self.default_options else {}
        self.options.update(kwargs)
        self.runs: list[BenchmarkRunResult] = []

        # Circuit compilation options
        if self.configuration.use_dd:
            if self.name in [
                "compressive_GST",
                "mirror_rb",
                "interleaved_clifford_rb",
                "clifford_rb",
            ]:
                qcvv_logger.warning(
                    "Beware that activating dynamical decoupling will change fidelities, "
                    "error models and their interpretation."
                )
            self.circuit_compilation_options = CircuitCompilationOptions(
                dd_mode=DDMode.ENABLED,
                dd_strategy=self.configuration.dd_strategy,
                active_reset_cycles=self.configuration.active_reset_cycles,
            )

        else:
            self.circuit_compilation_options = CircuitCompilationOptions(
                dd_mode=DDMode.DISABLED, active_reset_cycles=self.configuration.active_reset_cycles
            )

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """Return the name of the benchmark."""

    @abstractmethod
    def execute(self, backend: IQMBackend | IQMFacadeBackend) -> xr.Dataset:
        """Executes the benchmark.

        This method should be overridden by deriving classes.

        Args:
            backend: Qiskit backend used to execute benchmarks.

        Returns:
            An xarray dataset which contains to results of the benchmark execution.
            The dataset should contain all the information necessary for analysing
            the benchmark results.

        """

    def run(self) -> BenchmarkRunResult:
        """Runs the benchmark using the given backend.

        Returns:
            RunResult: The result of the benchmark run.

        """
        backend_for_execute = copy.copy(self.backend)
        backend_for_execute.run = functools.partial(self.backend.run)
        dataset = self.execute(backend_for_execute)
        run = BenchmarkRunResult(dataset, self.circuits)
        self.runs.append(run)
        return run

    def analyze(self, run_index: int = -1) -> BenchmarkAnalysisResult:
        """The default analysis for the benchmark.

        Internally uses the function defined by the attribute ``analysis_function``
        to perform the analysis. This function makes a shallow copy of the dataset produced by
        run. Therefore, it is recommended to not make changes to the data of the dataset but
        just the structure of the array.

        Args:
            run_index: Index for the run to analyze.

        Returns:
            An analysis result constructed from the run and updated by the analysis method defined by
            the ``analysis_function`` field.

        """
        run = self.runs[run_index]
        updated_result = self.analysis_function(run)
        return updated_result


class BenchmarkConfigurationBase(BaseModel):
    r"""Benchmark configuration base.

    Attributes:
        benchmark: Benchmark configuration.
        shots: Number of shots to use in circuit execution. The default for all benchmarks is 2**8.
        max_gates_per_batch: Maximum number of gates per circuit batch. The default for all benchmarks is None.
        max_circuits_per_batch: Maximum number of circuits per batch. The default for all benchmarks is None.
        routing_method: Qiskit routing method to use in transpilation. The default for all benchmarks is "sabre".
        physical_layout: Whether physical layout is constrained during transpilation to selected physical qubits.
            "fixed": physical layout is constrained during transpilation to the selected initial physical qubits.
            "batching": physical layout is allowed to use any other physical qubits, and circuits are batched according
            to final measured qubits. The default for all benchmarks is "fixed".
        use_dd: Boolean flag determining whether to enable dynamical decoupling during circuit execution. The default
            for all benchmarks is False.
        dd_strategy: Dynamical decoupling strategy.
        active_reset_cycles: Number of active reset cycles to apply after each circuit execution.
            If None, passive reset is used.

    """

    benchmark: type[Benchmark] | str
    shots: int = 2**8
    max_gates_per_batch: int | None = None
    max_circuits_per_batch: int | None = None
    routing_method: RoutingMethod = RoutingMethod.SABRE
    physical_layout: PhysicalLayout = PhysicalLayout.FIXED
    use_dd: bool | None = False
    dd_strategy: DDStrategy | None = None
    active_reset_cycles: int | None = None
