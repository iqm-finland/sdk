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

"""Compressive gate set tomography.

The full benchmark executes the following steps:
    1) Generation of circuits to measure
    2) Running of said circuits on the backend
    3) Conversion of the ideal gate set into numpy arrays to be processed by mGST
    4) Optional generation of an initialization for mGST
    5) Running mGST to get first gate set estimates
    6) Gauge optimization to the target gates using pyGSTi's gauge optimizer
    7) Optional rerun of mGST on bootstrap data to produce error bars
    8) Generation of error measures and result tables
    9) Generation of matrix plots for the reconstructed gate set
"""

from datetime import datetime, timezone
from typing import Any, cast

from iqm.benchmarks.benchmark_definition import (
    BENCHMARK_TIMESTAMP_FORMAT,
    Benchmark,
    BenchmarkConfigurationBase,
    BenchmarkObservationIdentifier,
    add_counts_to_dataset,
)
from iqm.benchmarks.circuit_containers import BenchmarkCircuit, CircuitGroup, Circuits
from iqm.benchmarks.compressive_gst.gst_analysis import mgst_analysis
from iqm.benchmarks.compressive_gst.mgst import additional_fns, qiskit_interface
from iqm.benchmarks.compressive_gst.mgst.qiskit_interface import add_idle_gates, remove_idle_wires
from iqm.benchmarks.logging_config import qcvv_logger
from iqm.benchmarks.utils import (
    get_active_qubits,
    perform_backend_transpilation,
    retrieve_all_counts,
    submit_execute,
    timeit,
)
from iqm.qiskit_iqm import IQMCircuit as QuantumCircuit
from iqm.qiskit_iqm.iqm_provider import IQMBackend, IQMFacadeBackend
import numpy as np
from qiskit.circuit.library import CZGate, RGate
import xarray as xr


class CompressiveGST(Benchmark):
    """SPAM-robust characterization of a set of quantum gates."""

    analysis_function = staticmethod(mgst_analysis)

    @classmethod
    def name(cls) -> str:
        """Returns the name of the benchmark."""
        return "compressive_gst"

    def __init__(self, backend: IQMBackend | IQMFacadeBackend | str, configuration: "GSTConfiguration"):
        """Construct the compressive_gst class.

        Args:
            backend: the backend to execute the benchmark on
            configuration: the configuration of the benchmark

        """
        super().__init__(backend, configuration)

        self.qubit_layouts = parse_layouts(configuration.qubit_layouts)
        self.num_qubits = len(self.qubit_layouts[0])
        self.pdim = 2**self.num_qubits
        self.num_povm = self.pdim
        self.verbose_level = configuration.verbose_level

        self.gate_set, self.gate_labels, self.num_gates = parse_gate_set(
            configuration, self.num_qubits, self.qubit_layouts
        )
        self.gate_context = configuration.gate_context
        self.validate_gate_context()

        if configuration.opt_method not in ["GD", "SFN", "auto"]:
            raise ValueError("Invalid optimization method, valid options are: GD, SFN, auto")
        if configuration.opt_method == "auto":
            self.opt_method = "GD"  # Currently the fastest method in all cases
        else:
            self.opt_method = configuration.opt_method

        if configuration.max_iterations == "auto":
            if self.opt_method == "SFN":
                self.max_iterations = [100, 100]
            if self.opt_method == "GD":
                self.max_iterations = [250, 250]
        elif isinstance(configuration.max_iterations, list):
            self.max_iterations = configuration.max_iterations
        elif isinstance(configuration.max_iterations, int):
            self.max_iterations = [
                configuration.max_iterations,
                configuration.max_iterations,
            ]
        if configuration.batch_size == "auto":
            self.batch_size = 30 * self.pdim
        else:
            self.batch_size = configuration.batch_size

        # Circuit format used by mGST
        self.J = np.empty((cast(GSTConfiguration, self.configuration).num_circuits, self.num_povm))

    def validate_gate_context(self) -> None:
        """Validate that the gate context is properly configured.

        Checks that:
        1. If gate_context is a list, it has the same length as gate_set
        2. The qubits used in gate_context don't overlap with qubits in the layouts

        Raises:
            ValueError: If gate_context is invalid or if qubits in gate_context overlap with qubits in layouts

        """
        if self.gate_context is not None:
            if isinstance(self.gate_context, list):
                if len(self.gate_context) != len(self.gate_set):
                    raise ValueError("If gate_context is a list, it must have the same length as gate_set.")
                # Check that context circuits don't overlap with qubit layouts for GST circuits
                context_qubits = [q for qc in self.gate_context for q in get_active_qubits(qc)]
            else:
                context_qubits = get_active_qubits(self.gate_context)
            layout_qubits = [q for layout in self.qubit_layouts for q in layout]
            if any(q in layout_qubits for q in context_qubits):
                raise ValueError(
                    f"Gate context qubits {set(context_qubits)} "
                    f"must not overlap with qubits in layouts {set(layout_qubits)}."
                )

    @timeit
    def generate_meas_circuits(self) -> tuple[BenchmarkCircuit, BenchmarkCircuit]:
        """Generate random circuits from the gate set.

        The random circuits are distributed among different depths ranging from L_MIN
        to L_MAX, both are configurable and stored in self.configuration.seq_len_list.
        No transpilation other than mapping to the desired qubits is performed,
        as the gates need to be executed axactly as described for GST to give
        meaningful results

        Returns:
            transpiled_circuits: The transpiled circuits
            untranspiled_circuits: The untranspiled circuits
            circuit_gen_transp_time: The time it took to generate and transpile the circuits

        """
        transpiled_circuits = BenchmarkCircuit(name="transpiled_circuits")
        untranspiled_circuits = BenchmarkCircuit(name="untranspiled_circuits")
        config = cast(GSTConfiguration, self.configuration)

        # Calculate number of short and long circuits
        n_short = int(np.ceil(config.num_circuits / 2))
        n_long = int(np.floor(config.num_circuits / 2))
        l_min, l_cut, l_max = config.seq_len_list

        # Generate random circuits
        J = additional_fns.random_seq_design(self.num_gates, l_min, l_cut, l_max, n_short, n_long)  # noqa: N806

        # Reduce circuits to exclude negative placeholder values
        gate_circuits = [list(seq[seq >= 0]) for seq in J]

        # Sequences in GST are applied to the state from right to left
        self.J = np.array(J)[:, ::-1]  # noqa: N806

        unmapped_qubits = list(range(self.num_qubits))
        raw_qc_list = qiskit_interface.get_qiskit_circuits(
            gate_circuits, self.gate_set, self.num_qubits, unmapped_qubits
        )

        if "move" in self.backend.operation_names:
            qcvv_logger.warning(
                "Transpilation on star-architectures currently allows move gates to transit barriers, "
                "leading to context-dependent gates which GST can not accurately resolve."
            )

        # Perform transpilation to backend
        qcvv_logger.info(f"Will transpile all {config.num_circuits} circuits according to fixed physical layout")

        if "move" in self.backend.operation_names:
            backend_qubits = np.arange(1, self.backend.num_qubits)
            qubit_layouts = [[q - 1 for q in layout] for layout in self.qubit_layouts]
        else:
            backend_qubits = np.arange(self.backend.num_qubits)
            qubit_layouts = self.qubit_layouts

        if config.parallel_execution:
            all_qubits = [qubit for layout in qubit_layouts for qubit in layout]
            if len(all_qubits) != len(set(all_qubits)):
                raise ValueError(
                    "Qubit layouts can't overlap when parallel_execution is enabled, "
                    "please choose non-overlapping layouts."
                )
            # For each gate sequence, create a circuit with gate context and GST sequence on all qubits in the layout
            composed_qc_list = qiskit_interface.get_composed_qiskit_circuits(
                gate_circuits,
                self.gate_set,
                self.backend.num_qubits,
                qubit_layouts,
                gate_context=self.gate_context,
                parallel=True,
            )
            transpiled_qc_list, _ = perform_backend_transpilation(
                composed_qc_list,
                self.backend,
                qubits=backend_qubits,
                coupling_map=self.backend.coupling_map,
                qiskit_optim_level=0,
                optimize_sqg=True,
                drop_final_rz=False,
            )
            for qubits in self.qubit_layouts:
                # Saving raw and transpiled circuits in a consistent format with other benchmarks
                transpiled_circuits.circuit_groups.append(CircuitGroup(name=str(qubits), circuits=transpiled_qc_list))
                untranspiled_circuits.circuit_groups.append(CircuitGroup(name=str(qubits), circuits=raw_qc_list))
        else:
            # for parralel = False, a unique list of circuits is generated for each qubit layout
            composed_qc_list = qiskit_interface.get_composed_qiskit_circuits(
                gate_circuits,
                self.gate_set,
                self.backend.num_qubits,
                qubit_layouts,
                gate_context=self.gate_context,
                parallel=False,
            )
            for idx, qubits in enumerate(self.qubit_layouts):
                transpiled_qc_list, _ = perform_backend_transpilation(
                    composed_qc_list[idx],
                    self.backend,
                    backend_qubits,
                    coupling_map=self.backend.coupling_map,
                    qiskit_optim_level=0,
                    optimize_sqg=False,
                    drop_final_rz=False,
                )
                # Saving raw and transpiled circuits in a consistent format with other benchmarks
                transpiled_circuits.circuit_groups.append(CircuitGroup(name=str(qubits), circuits=transpiled_qc_list))
                untranspiled_circuits.circuit_groups.append(CircuitGroup(name=str(qubits), circuits=raw_qc_list))

        return transpiled_circuits, untranspiled_circuits

    def add_configuration_to_dataset(self, dataset: xr.Dataset) -> None:  # CHECK
        """Creates an xarray.Dataset and adds the circuits and configuration metadata to it.

        Args:
            self: Source class
            dataset: Dataset to which the configuration is added

        Returns:
            dataset: Dataset to be used for further data storage

        """
        # Adding configuration entries and class variables, prioritizing the latter in case of conflicts
        avoided_keys = ["runs", "configuration", "circuits"]
        for key, value in (self.serializable_configuration.__dict__ | self.__dict__).items():
            if key == "backend":
                dataset.attrs[key] = value.name
            elif key not in avoided_keys:
                dataset.attrs[key] = value
            else:
                pass
        dataset.attrs["gauge_weights"] = [1.0] * self.num_gates + [0.1]

    def execute(self, backend: IQMBackend | IQMFacadeBackend) -> xr.Dataset:
        """The main GST execution routine."""
        self.execution_timestamp = datetime.now(timezone.utc).strftime(BENCHMARK_TIMESTAMP_FORMAT)
        total_submit: float = 0
        total_retrieve: float = 0
        dataset = xr.Dataset()
        config = cast(GSTConfiguration, self.configuration)
        qcvv_logger.info(f"Generating {config.num_circuits} random GST circuits")

        self.circuits = Circuits()
        # Generate circuits
        (transpiled_circuits, untranspiled_circuits), _ = self.generate_meas_circuits()

        # Submit all
        if config.parallel_execution:
            transpiled_circuit_dict = {
                tuple(range(self.backend.num_qubits)): transpiled_circuits[str(self.qubit_layouts[0])].circuits
            }
            all_jobs_parallel, time_submit = submit_execute(
                transpiled_circuit_dict,
                backend,
                config.shots,
                max_gates_per_batch=config.max_gates_per_batch,
                max_circuits_per_batch=config.max_circuits_per_batch,
                circuit_compilation_options=self.circuit_compilation_options,
            )
            total_submit += time_submit
            # Retrieve
            qcvv_logger.info("Now executing the corresponding circuit batch")
            counts, time_retrieve = retrieve_all_counts(all_jobs_parallel)
            total_retrieve += time_retrieve
            dataset, _ = add_counts_to_dataset(counts, "parallel_results", dataset)
        else:
            all_jobs: dict = {}
            for qubit_layout in self.qubit_layouts:
                transpiled_circuit_dict = {tuple(qubit_layout): transpiled_circuits[str(qubit_layout)].circuits}
                all_jobs[str(qubit_layout)], time_submit = submit_execute(
                    transpiled_circuit_dict,
                    backend,
                    config.shots,
                    max_gates_per_batch=config.max_gates_per_batch,
                    max_circuits_per_batch=config.max_circuits_per_batch,
                    circuit_compilation_options=self.circuit_compilation_options,
                )
                total_submit += time_submit
            # Retrieve all
            qcvv_logger.info("Now executing the corresponding circuit batch")
            for qubit_layout in self.qubit_layouts:
                counts, time_retrieve = retrieve_all_counts(all_jobs[str(qubit_layout)])
                total_retrieve += time_retrieve
                dataset, _ = add_counts_to_dataset(counts, str(qubit_layout), dataset)

        self.circuits.benchmark_circuits = [transpiled_circuits, untranspiled_circuits]
        self.add_configuration_to_dataset(dataset)
        dataset.attrs["total_submit_time"] = total_submit
        dataset.attrs["total_retrieve_time"] = total_retrieve
        qcvv_logger.info("Run completed")
        return dataset


class GSTConfiguration(BenchmarkConfigurationBase):
    """Compressive GST configuration base.

    Attributes:
        benchmark: GHZBenchmark
        qubit_layouts: A sequence (e.g., Tuple or List) of sequences of
            physical qubit layouts, as specified by integer labels, where the benchmark is meant to be run.
        gate_set: The gate set in given either as a list of qiskit quantum
            cirucuits, or as one of the predefined gate sets "1QXYI", "2QXYCZ", "2QXYCZ_extended", "3QXYCZ".
            A gate set should be tomographically complete, meaning from the specified gates and the vacuum state, one
            should be able to prepare states that form a frame for the state space. A practical sufficient condition
            is that the gate set is able the generate all combinations of local Pauli eigenstates.
        num_circuits: How many random circuits are generated from the gate set. Guidelines on choosing this value:
            - At least 50 circuits for a single qubit gate set with 3 gates.
            - At least 400 circuits for a two qubit gate set with 6 gates.
            - At least 2000 circuits for a three qubit gate set with 9 gates.
            The number of random circuits needed is expected to grow linearly in the number of gates and exponentially
            in the number of qubits.
        rank: The Kraus rank of the reconstruction. Choose rank=1 for coherent error analysis and rank<=dim**2
            generally.
        shots: The number of measurement shots per circuit.
        gate_labels: Names of the gates in the gate set. Used for plots and result tables.
        seq_len_list: Three numbers controling the depth of the random sequences. The first is the minimal
            depth, the last is the maximal depth and the middle number specifies a cutoff depth below which all possible
            sequences are selected.
        from_init: Whether the target gate set is used as an initialization to the mGST algorithm.
        max_inits: If from_init = False, random initial points are tried and this parameter limits the amount of
            retries.
        opt_method: Which optimization method is used, can be either of "GD" or "SFN", for gradient descent or
            saddle free Newton, respectively.
        max_iterations: How many iterations to run the optimization algorithm for. Accepted
            values are either "auto" or a list of two integers. The first specifies the number of iterations for the
            optimization on batches or circuit data, while the second specifies the number of iterations on the full
            data for all circuits.
        convergence_criteria: Two parameters which determine when the optimization algorithm
            terminates. The first is a multiplier which specifies how close the cost function should get to a threshold
            for the given shot noise in order for the optimization to be considered "successful". Values in [2,10] are
            usually sensible. The second parameter sets the relative change in cost function between
            two consecutive iterations, below which the algorithm terminates.
        batch_size: The number of circuits per batch in the optimization. This hyperparamters is
            automatically set and determines the convergence behaviour. A smaller batch size reduces runtime but can
            lead to erratic jumps in parameter space and lack of convergence.
        bootstrap_samples: The number of times the optimization algorithm is repeated on fake data to estimate
            the uncertainty via bootstrapping.
        verbose_level: The level of verbosity of the output. 0 is minimal, 1 gives optimization updates,
            2 outputs optimization plots
        parallel_execution: Whether to run the circuits for all layouts in parallel on the backend.

    """

    benchmark: type[Benchmark] = CompressiveGST
    qubit_layouts: list[int] | list[list[int]]
    gate_set: str | list[Any]
    gate_context: Any | list[Any] | None = None
    num_circuits: int
    rank: int
    shots: int = 2**10
    gate_labels: list | None = None
    seq_len_list: list[int] = [1, 8, 14]
    from_init: bool = True
    max_inits: int = 20
    opt_method: str = "auto"
    max_iterations: str | list[int] | int = "auto"
    convergence_criteria: str | list[float] = [4, 1e-5]
    batch_size: str | int = "auto"
    bootstrap_samples: int = 0
    verbose_level: int = 1
    parallel_execution: bool = False
    fixed_elements: list[str] | None = None


def parse_layouts(qubit_layouts: list[int] | list[list[int]]) -> list[list[int]]:
    """Checks for correct setting of qubit_layouts in the configuration and return a correct type.

    Args:
        qubit_layouts:
            The qubit_layouts on the backend where the gates are defined on

    Returns:
        A properly typed qubit_layout if no Error was raised

    """
    if all(isinstance(qubits, int) for qubits in qubit_layouts):
        return [qubit_layouts]  # type: ignore
    if all(isinstance(qubits, list) for qubits in qubit_layouts):
        if all(len(x) == len(qubit_layouts[0]) for x in qubit_layouts):  # type: ignore
            return qubit_layouts  # type: ignore
        raise ValueError(
            "Qubit layouts are of different size, currently only specifying "
            "qubit layouts with the same number of qubits is supported."
        )
    raise ValueError(
        "The qubit_layouts parameters needs to be one of List[int] for a single layout"
        " or List[List[int]] for multiple layouts."
    )


def parse_gate_set(
    configuration: GSTConfiguration, num_qubits: int, qubit_layouts: list[list[int]]
) -> tuple[list[QuantumCircuit], dict[str, dict[int, str]], int]:
    """Handles different gate set inputs and produces a valid gate set.

    Args:
        configuration:
            Configuration class containing variables
        num_qubits:
            The number of qubits on which the gate set is defined
        qubit_layouts:
            A List of physical qubit layouts, as specified by integer labels, where the benchmark is meant to be run.

    Returns:
        gate_set:
            A list of gates defined as quantum circuit objects
        gate_labels_dict:
            The names of gates, i.e. "Rx_pi_2" for a pi/2 rotation around the x-axis.
        num_gates:
            The number of gates in the gate set

    """
    if isinstance(configuration.gate_set, str) and configuration.gate_set not in [
        "1QXYI",
        "2QXYICZ",
        "2QXYCZ_extended",
        "3QXYCZ",
    ]:
        raise ValueError(
            "No gate set of the specified name is implemented, please choose among "
            "1QXYI, 2QXYICZ, 2QXYCZ_extended, 3QXYCZ."
        )
    if configuration.gate_set in ["1QXYI", "2QXYICZ", "2QXYCZ_extended", "3QXYCZ"]:
        gate_set, gate_label_dict, num_gates = create_predefined_gate_set(
            configuration.gate_set, num_qubits, qubit_layouts
        )
        return gate_set, gate_label_dict, num_gates

    if isinstance(configuration.gate_set, list):
        gate_label_dict = {}
        gate_set = configuration.gate_set
        num_gates = len(gate_set)
        if configuration.gate_labels is None:
            gate_labels = {i: "Gate %i" % i for i in range(num_gates)}
        else:
            if len(configuration.gate_labels) != num_gates:
                raise ValueError(
                    "The number of gate labels (%i) does not match the number of gates (%i)"
                    % (len(configuration.gate_labels), num_gates)
                )
            gate_labels = dict(enumerate(configuration.gate_labels))
        for qubit_layout in qubit_layouts:
            gate_label_dict.update({BenchmarkObservationIdentifier(qubit_layout).string_identifier: gate_labels})
        return gate_set, gate_label_dict, num_gates

    raise ValueError(
        "Invalid gate set, choose among 1QXYI, 2QXYICZ, 2QXYCZ_extended,"
        " 3QXYCZ or provide a list of Qiskti circuits to define the gates."
    )


def create_predefined_gate_set(
    gate_set: str | list[Any], num_qubits: int, qubit_layouts: list[list[int]]
) -> tuple[list[QuantumCircuit], dict[str, dict[int, str]], int]:
    """Create a list of quantum circuits corresponding to a predefined gate set.

    The circuits are assigned to the specified qubit_layouts on the backend only during transipilation,
    so the qubit labels at this stage may not represent the actual qubit labels on the backend.

    Args:
        gate_set:
            The name of the gate set
        num_qubits:
            The number of qubits on which the gate set is defined
        qubit_layouts:
            A List of physical qubit layouts, each of which is a list of qubit labels, where the benchmark is
            meant to be run.

    Returns:
        gates:
            The gate set as a list of circuits
        gate_labels_dict:
            The names of gates, i.e. "Rx_pi_2" for a pi/2 rotation around the x-axis.
        num_gates:
            The number of gates in the gate set

    """
    unmapped_qubits = list(range(num_qubits))
    # Define an idle gate using a time delay comparable to single qubit gate duration
    idle = QuantumCircuit(num_qubits, 0)
    idle.delay(32e-9, unit="s")

    # Define the gate set
    if gate_set == "1QXYI":
        gate_list = [idle, RGate(0.5 * np.pi, 0), RGate(0.5 * np.pi, np.pi / 2)]
        gates = [QuantumCircuit(num_qubits, 0) for _ in range(len(gate_list))]
        gate_qubits = [[0], [0], [0]]
        for i, gate in enumerate(gate_list):
            gates[i].append(gate, gate_qubits[i])
        gate_labels = ["Idle", "Rx_pi_2", "Ry_pi_2"]
    elif gate_set == "2QXYICZ":
        gate_qubits = [[0], [0], [1], [0], [1], [0, 1]]
        gates = [QuantumCircuit(num_qubits, 0) for _ in range(6)]
        gates[0].append(idle, [0, 1])
        gates[1].append(RGate(0.5 * np.pi, 0), [0])
        gates[2].append(RGate(0.5 * np.pi, 0), [1])
        gates[3].append(RGate(0.5 * np.pi, np.pi / 2), [0])
        gates[4].append(RGate(0.5 * np.pi, np.pi / 2), [1])
        gates[5].append(CZGate(), [0, 1])
        gate_labels = ["Idle", "Rx_pi_2", "Rx_pi_2", "Ry_pi_2", "Ry_pi_2", "cz"]
    elif gate_set == "2QXYCZ_extended":
        gate_qubits = [[0], [1], [0], [1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]
        gates = [QuantumCircuit(num_qubits, 0) for _ in range(9)]
        gates[0].append(RGate(0.5 * np.pi, 0), [0])
        gates[1].append(RGate(0.5 * np.pi, 0), [1])
        gates[2].append(RGate(0.5 * np.pi, np.pi / 2), [0])
        gates[3].append(RGate(0.5 * np.pi, np.pi / 2), [1])
        gates[4].append(RGate(0.5 * np.pi, 0), [0])
        gates[4].append(RGate(0.5 * np.pi, 0), [1])
        gates[5].append(RGate(0.5 * np.pi, 0), [0])
        gates[5].append(RGate(0.5 * np.pi, np.pi / 2), [1])
        gates[6].append(RGate(0.5 * np.pi, np.pi / 2), [0])
        gates[6].append(RGate(0.5 * np.pi, 0), [1])
        gates[7].append(RGate(0.5 * np.pi, np.pi / 2), [0])
        gates[7].append(RGate(0.5 * np.pi, np.pi / 2), [1])
        gates[8].append(CZGate(), [[0], [1]])
        gate_labels = [
            "Rx_pi_2",
            "Rx_pi_2",
            "Ry_pi_2",
            "Ry_pi_2",
            "Rx_pi_2-Rx_pi_2",
            "Rx_pi_2-Ry_pi_2",
            "Ry_pi_2-Rx_pi_2",
            "Ry_pi_2-Ry_pi_2",
            "CZ",
        ]
    elif gate_set == "3QXYCZ":
        gate_list = [
            RGate(0.5 * np.pi, 0),
            RGate(0.5 * np.pi, 0),
            RGate(0.5 * np.pi, 0),
            RGate(0.5 * np.pi, np.pi / 2),
            RGate(0.5 * np.pi, np.pi / 2),
            RGate(0.5 * np.pi, np.pi / 2),
            CZGate(),
            CZGate(),
        ]
        gates = [QuantumCircuit(num_qubits, 0) for _ in range(len(gate_list))]
        gate_qubits = [[0], [1], [2], [0], [1], [2], [0, 1], [0, 2]]
        for i, gate in enumerate(gate_list):
            gates[i].append(gate, gate_qubits[i])
        gate_labels = [
            "Rx_pi_2",
            "Rx_pi_2",
            "Rx_pi_2",
            "Ry_pi_2",
            "Ry_pi_2",
            "Ry_pi_2",
            "cz",
            "cz",
        ]
    else:
        raise ValueError(
            "Invalid gate set, choose among 1QXYI, 2QXYCZ, 2QXYCZ_extended,"
            " 3QXYCZ or provide a list of Qiskti circuits to define the gates."
        )

    gates = add_idle_gates(gates, unmapped_qubits, gate_qubits)
    gates = [remove_idle_wires(qc) for qc in gates]

    gate_label_dict = {}
    for qubit_layout in qubit_layouts:
        layout_label_dict = dict(enumerate(gate_labels))
        iqm_qubits = [f"QB{q + 1}" for q in qubit_layout]
        gate_qubits_iqm = [(iqm_qubits[q] for q in qubits) for qubits in gate_qubits]
        for key, value in layout_label_dict.items():
            layout_label_dict[key] = value + ":" + "__".join(gate_qubits_iqm[key])
        gate_label_dict.update({BenchmarkObservationIdentifier(qubit_layout).string_identifier: layout_label_dict})
    return gates, gate_label_dict, len(gates)
