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

"""GHZ state benchmark."""

from collections.abc import Sequence
from datetime import UTC, datetime
from itertools import chain
from typing import Any, cast

from iqm.benchmarks.benchmark_definition import (
    BENCHMARK_TIMESTAMP_FORMAT,
    Benchmark,
    BenchmarkAnalysisResult,
    BenchmarkConfigurationBase,
    BenchmarkObservation,
    BenchmarkObservationIdentifier,
    BenchmarkRunResult,
    add_counts_to_dataset,
)
from iqm.benchmarks.circuit_containers import BenchmarkCircuit, CircuitGroup, Circuits
from iqm.benchmarks.entanglement.ghz_optimal import (
    CZ_GATE_DURATION,
    TREE_OBJECTIVES,
    TreeSearchOptions,
    generate_ghz_tree_optimal,
)
from iqm.benchmarks.logging_config import qcvv_logger
from iqm.benchmarks.readout_mitigation import apply_readout_error_mitigation
from iqm.benchmarks.utils import (
    PhysicalLayout,
    extract_fidelities_unified,
    perform_backend_transpilation,
    reduce_to_active_qubits,
    retrieve_all_counts,
    set_coupling_map,
    submit_execute,
    timeit,
    xrvariable_to_counts,
)
from iqm.qiskit_iqm import IQMCircuit as QuantumCircuit, transpile_to_IQM
from iqm.qiskit_iqm.iqm_provider import IQMBackend, IQMFacadeBackend
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from pydantic import field_validator
from qiskit import ClassicalRegister, QuantumRegister
from qiskit.quantum_info import random_clifford
from qiskit.transpiler import CouplingMap
from qiskit_aer import Aer
from scipy.spatial.distance import hamming
import xarray as xr

STATE_GENERATION_ROUTINES = ("tree", "naive", "log", "star", "star_optimal")


def fidelity_ghz_randomized_measurements(
    dataset: xr.Dataset,
    qubit_layout: list[int],
    ideal_probabilities: list[dict[str, int]],
    num_qubits: int,
    num_rms: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Estimates GHZ state fidelity through cross-correlations of RMs.

    Implementation of Eq. (34) in https://arxiv.org/abs/1812.02624

    Args:
        dataset: An xarray dataset containing the measurement data
        qubit_layout: The subset of system-qubits used in the protocol
        ideal_probabilities: List of ideal probability distributions for each randomized measurement
        num_qubits: Number of qubits
        num_rms: Number of random circuits executed for the given qubit layout to estimate the fidelity

    Returns:
        * The fidelities
        * The uncertainties for the fidelities

    """
    idx = BenchmarkObservationIdentifier(qubit_layout).string_identifier
    # List for each RM contribution to the fidelity
    fid_rm = []

    # Loop through RMs and add each contribution
    for u in range(num_rms):
        # Probability estimates for noisy measurements
        probabilities_sample = {}
        c_keys = dataset[f"{idx}_state_{u}"].data  # measurements[u].keys()
        num_shots_noisy = sum(dataset[f"{idx}_counts_{u}"].data)
        for k, key in enumerate(c_keys):
            probabilities_sample[key] = dataset[f"{idx}_counts_{u}"].data[k] / num_shots_noisy
        # Keys for corresponding ideal probabilities
        c_id_keys = ideal_probabilities[u].keys()

        p_sum = []
        for sb in c_id_keys:
            for sa in c_keys:
                exponent = hamming(np.array(list(sa)), np.array(list(sb))) * num_qubits
                p_sum.append(np.power(-2, -exponent) * probabilities_sample[sa] * ideal_probabilities[u][sb])
        fid_rm.append((2**num_qubits) * sum(p_sum))
    values = {"fidelity": np.mean(fid_rm)}
    uncertainties = {"fidelity": np.std(fid_rm) / np.sqrt(num_rms)}

    if dataset.attrs["rem"]:
        fid_rm_rem = []
        for u in range(num_rms):
            # Probability estimates for noisy measurements
            probabilities_sample = {}
            c_keys = dataset[f"{idx}_rem_state_{u}"].data  # measurements[u].keys()
            num_shots_noisy = sum(dataset[f"{idx}_rem_counts_{u}"].data)
            for k, key in enumerate(c_keys):
                probabilities_sample[key] = dataset[f"{idx}_rem_counts_{u}"].data[k] / num_shots_noisy
            # Keys for corresponding ideal probabilities
            c_id_keys = ideal_probabilities[u].keys()

            p_sum = []
            for sb in c_id_keys:
                for sa in c_keys:
                    exponent = hamming(np.array(list(sa)), np.array(list(sb))) * num_qubits
                    p_sum.append(np.power(-2, -exponent) * probabilities_sample[sa] * ideal_probabilities[u][sb])
            fid_rm_rem.append((2**num_qubits) * sum(p_sum))
        values = values | {"fidelity_rem": np.mean(fid_rm_rem)}
        uncertainties = uncertainties | {"fidelity_rem": np.std(fid_rm_rem) / np.sqrt(num_rms)}
    return values, uncertainties


def fidelity_ghz_coherences(dataset: xr.Dataset, qubit_layout: list[int], num_circuits: int) -> list[Any]:
    """Estimates the GHZ state fidelity based on the multiple quantum coherences method based on [Mooney, 2021].

    Args:
        dataset:
            An xarray dataset containing the measurement data
        qubit_layout:
            The subset of system-qubits used in the protocol
        num_circuits:
            The number of circuits executed for the given qubit layout to estimate the fidelity

    Returns:
        The ghz fidelity or, if rem=True, fidelity and readout error mitigated fidelity

    """
    num_qubits = len(qubit_layout)
    phases = [np.pi * i / (num_qubits + 1) for i in range(2 * num_qubits + 2)]
    idx = BenchmarkObservationIdentifier(qubit_layout).string_identifier

    # Computing the phase acquired by the |11...1> component for each interval
    complex_coefficients = np.exp(1j * num_qubits * np.array(phases))

    # Loading the counts from the dataset
    counts = xrvariable_to_counts(dataset, f"{idx}", num_circuits)
    all_zero_probability_list = []  # An ordered list for storing the probabilities of returning to the |00..0> state
    for count in counts[1:]:
        normalization = np.sum(list(count.values()))
        probability = count["0" * num_qubits] / normalization if "0" * num_qubits in count else 0
        all_zero_probability_list.append(probability)

    # Extracting coherence parameter i_n using the fourier transform
    i_n = np.abs(np.dot(complex_coefficients, np.array(all_zero_probability_list))) / (len(phases))

    # Extracting the probabilities of the 00...0 and 11...1 bit strings
    probs_direct = {label: count / np.sum(list(counts[0].values())) for label, count in counts[0].items()}

    # Computing GHZ state fidelity from i_n and the probabilities according to the method in [Mooney, 2021]
    p0 = probs_direct.get("0" * num_qubits, 0)
    p1 = probs_direct.get("1" * num_qubits, 0)
    fidelity = (p0 + p1) / 2 + np.sqrt(i_n)

    # Same procedure for error mitigated data
    if dataset.attrs["rem"]:
        probs_mit = xrvariable_to_counts(dataset, f"{idx}_rem", num_circuits)
        all_zero_probability_list_mit = []
        for prob in probs_mit[1:]:
            probability = prob.get("0" * num_qubits, 0)
            all_zero_probability_list_mit.append(probability)
        i_n_mit = np.abs(np.dot(complex_coefficients, np.array(all_zero_probability_list_mit))) / (len(phases))
        probs_direct_mit = dict(probs_mit[0].items())
        p0_mit = probs_direct_mit.get("0" * num_qubits, 0)
        p1_mit = probs_direct_mit.get("1" * num_qubits, 0)
        fidelity_mit = (p0_mit + p1_mit) / 2 + np.sqrt(i_n_mit)
        return [fidelity, fidelity_mit]
    return [fidelity]


def fidelity_analysis(run: BenchmarkRunResult) -> BenchmarkAnalysisResult:
    """Analyze counts and compute the state fidelity.

    Args:
        run: The RunResult object containing a dataset with counts and benchmark parameters.

    Returns:
        An object containing the dataset, plots, and observations.

    """
    dataset = run.dataset
    routine = dataset.attrs["fidelity_routine"]
    qubit_layouts = dataset.attrs["custom_qubits_array"]
    backend_topology = dataset.attrs["backend_topology"]
    backend_num_qubits = dataset.attrs["backend_num_qubits"]

    observation_list: list[BenchmarkObservation] = []
    for qubit_layout in qubit_layouts:
        match routine:
            case "randomized_measurements":
                ideal_simulator = Aer.get_backend("statevector_simulator")
                ideal_probabilities = []
                idx = BenchmarkObservationIdentifier(qubit_layout).string_identifier
                untranspiled = run.circuits["untranspiled_circuits"]
                if untranspiled is None:
                    raise RuntimeError("untranspiled_circuits not found in circuits")
                circuit_group = untranspiled[f"{idx}_rm_circuits"]
                if circuit_group is None:
                    raise RuntimeError(f"Circuit group {idx}_rm_circuits not found")
                all_circuits = circuit_group.circuits
                for qc in all_circuits:
                    qc_copy = qc.copy()
                    qc_copy.remove_final_measurements()
                    deflated_qc = reduce_to_active_qubits(qc_copy, backend_topology, backend_num_qubits)
                    ideal_probabilities.append(
                        dict(sorted(ideal_simulator.run(deflated_qc).result().get_counts().items()))
                    )
                values, uncertainties = fidelity_ghz_randomized_measurements(
                    dataset,
                    qubit_layout,
                    ideal_probabilities,
                    len(qubit_layout),
                    len(all_circuits),
                )
                observation_list.extend(
                    [
                        BenchmarkObservation(
                            name=key,
                            identifier=BenchmarkObservationIdentifier(qubit_layout),
                            value=value,
                            uncertainty=uncertainties[key],
                        )
                        for key, value in values.items()
                    ]
                )
            case "coherences":
                transpiled = run.circuits["transpiled_circuits"]
                if transpiled is None:
                    raise RuntimeError("transpiled_circuits not found in circuits")
                circuit_group = transpiled[f"{qubit_layout}_native_ghz"]
                if circuit_group is None:
                    raise RuntimeError(f"Circuit group {qubit_layout}_native_ghz not found")
                num_circuits = len(circuit_group.circuits)
                fidelity = fidelity_ghz_coherences(dataset, qubit_layout, num_circuits)
                observation_list.extend(
                    [
                        BenchmarkObservation(
                            name="fidelity",
                            identifier=BenchmarkObservationIdentifier(qubit_layout),
                            value=fidelity[0],
                        )
                    ]
                )
                if len(fidelity) > 1:
                    observation_list.append(
                        BenchmarkObservation(
                            name="fidelity_rem",
                            identifier=BenchmarkObservationIdentifier(qubit_layout),
                            value=fidelity[1],
                        )
                    )
    plots = {"All layout fidelities": plot_fidelities(observation_list, dataset, qubit_layouts)}
    return BenchmarkAnalysisResult(dataset=dataset, observations=observation_list, plots=plots)


def generate_ghz_linear(num_qubits: int) -> QuantumCircuit:
    """Generates a GHZ state by applying a Hadamard and a series of CX gates in a linear fashion.

    The construction is symmetrized to halve the circuit depth.

    Args:
        num_qubits: The number of qubits of the GHZ state.

    Returns:
        A quantum circuit generating a GHZ state on a given number of qubits.

    """
    s = int(num_qubits / 2)
    quantum_register = QuantumRegister(num_qubits)
    qc = QuantumCircuit(quantum_register, name="GHZ_linear")
    qc.h(s)

    for m in range(s, 0, -1):
        qc.cx(m, m - 1)
        if num_qubits % 2 == 0 and m == s:
            continue
        qc.cx(num_qubits - m - 1, num_qubits - m)
    qc.measure_all()
    return qc


def generate_ghz_log_cruz(num_qubits: int) -> QuantumCircuit:
    """Generates a GHZ state in log-depth according to https://arxiv.org/abs/1807.05572.

    Args:
        num_qubits: The number of qubits of the GHZ state

    Returns:
        A quantum circuit generating a GHZ state on a given number of qubits.

    """
    quantum_register = QuantumRegister(num_qubits)
    qc = QuantumCircuit(quantum_register, name="GHZ_log_Cruz")
    qc.h(0)

    for m in range(num_qubits):
        for k in range(2**m):
            if ((2**m) + k) >= num_qubits:
                break
            qc.cx(k, 2**m + k)
    qc.measure_all()
    return qc


def generate_ghz_star_optimal(
    qubit_layout: list[int], backend: IQMBackend | IQMFacadeBackend, inv: bool = False
) -> QuantumCircuit:
    """Generates the circuit for creating a GHZ state by maximizing the number of CZ gates between a pair of MOVE gates.

    Args:
        qubit_layout:
            The layout of qubits for the GHZ state.
        backend:
            The backend to be used for the quantum circuit.
        inv:
            Whether to generate the inverse circuit.

    Returns:
        A quantum circuit generating a GHZ state on a given number of qubits.

    """
    num_qubits = len(qubit_layout)

    # Initialize quantum and classical registers
    comp_r = QuantumRegister(1, "comp_r")  # Computational resonator
    q = QuantumRegister(backend.num_qubits, "q")  # Qubits
    c = ClassicalRegister(num_qubits, "c")
    qc = QuantumCircuit(comp_r, q, c, name="GHZ_star_optimal")

    cal_data = extract_fidelities_unified(backend)
    # Determine the best move qubit
    double_move_fidelities = {
        cast(tuple[int, int], k)[0]: v for k, v in list(cal_data[-1]["double_move_gate_fidelity"].items())[::2]
    }
    move_dict = {q: double_move_fidelities[q + 1] for q in qubit_layout}  ## +1 to match qubit indexing in cal data
    best_move = max(move_dict, key=lambda q: move_dict[q])

    t2_time = cal_data[-1]["t2_time"]
    t2_dict = {qubit: t2_time[qubit + 1] for qubit in qubit_layout}
    cz_order = dict(sorted(t2_dict.items(), key=lambda item: item[1], reverse=True))
    qubits_to_measure = list(cz_order.keys())
    cz_order.pop(best_move)

    # Construct the quantum circuit
    qc.r(np.pi / 2, 3 * np.pi / 2, best_move)
    qc.barrier(best_move)
    for qubit in cz_order:
        qc.barrier(qubit)
        qc.r(np.pi / 2, 3 * np.pi / 2, qubit)
        qc.cz(best_move, qubit)
        qc.r(np.pi / 2, 5 * np.pi / 2, qubit)
        qc.barrier(qubit)
    qc.barrier()
    qc.measure(sorted(qubits_to_measure), list(range(num_qubits)))

    qcvv_logger.info(
        f"Best MOVE qubit selected: {best_move} with double MOVE fidelity {move_dict[best_move]:.4f} "
        f"and T2 time {t2_dict[best_move]:.2f} us."
    )

    if inv:
        q = QuantumRegister(backend.num_qubits, "q")  # Qubits
        c = ClassicalRegister(num_qubits, "c")
        qc = QuantumCircuit(q, c, name="GHZ_star_optimal_inv")
        for qubit in reversed(cz_order.keys()):
            qc.barrier(qubit)
            qc.r(np.pi / 2, 5 * np.pi / 2, qubit)
            qc.cz(best_move, qubit)
            qc.r(np.pi / 2, 3 * np.pi / 2, qubit)
            qc.barrier(qubit)
        qc.r(np.pi / 2, 3 * np.pi / 2, best_move)

    return qc


def generate_ghz_star(num_qubits: int) -> QuantumCircuit:
    """Generates the circuit for creating a GHZ state by maximizing the number of CZ gates between a pair of MOVE gates.

    Args:
        num_qubits: The number of qubits of the GHZ state

    Returns:
        A quantum circuit generating a GHZ state on a given number of qubits.

    """
    quantum_register = QuantumRegister(num_qubits)
    qc = QuantumCircuit(quantum_register, name="GHZ_star")
    qc.h(0)
    for i in range(num_qubits - 1):
        qc.cx(0, i + 1)
    qc.measure_all()
    return qc


def generate_ghz_log_mooney(num_qubits: int) -> QuantumCircuit:
    """Generates a GHZ state in log-depth according to https://arxiv.org/abs/2101.08946.

    Args:
        num_qubits: The number of qubits of the GHZ state

    Returns:
        A quantum circuit generating a GHZ state on a given number of qubits.

    """
    quantum_register = QuantumRegister(num_qubits)
    qc = QuantumCircuit(quantum_register, name="GHZ_log_Mooney")
    qc.h(0)

    aux_n = int(np.ceil(np.log2(num_qubits)))
    for m in range(aux_n, 0, -1):
        for k in range(0, num_qubits, 2**m):
            if k + 2 ** (m - 1) >= num_qubits:
                continue
            qc.cx(k, k + 2 ** (m - 1))
    qc.measure_all()
    return qc


def plot_fidelities(
    observations: list[BenchmarkObservation],
    dataset: xr.Dataset,
    qubit_layouts: list[list[int]],
) -> Figure:
    """Plots all the fidelities stored in the observations into a single plot of fidelity vs. number of qubits.

    Args:
        observations: A list of Observations, each assumed to be a fidelity
        dataset: The experiment dataset containing results and metadata
        qubit_layouts: The list of qubit layouts as given by the user.
            This is used to name the layouts in order for identification in the plot.

    Returns:
        The figure object with the fidelity plot.

    """
    timestamp = dataset.attrs["execution_timestamp"]
    backend_name = dataset.attrs["backend_name"]

    fig, ax = plt.subplots()
    layout_short = {str(qubit_layout): f" L{i}" for i, qubit_layout in enumerate(qubit_layouts)}
    recorded_labels = []
    x_positions = []
    cmap = plt.colormaps["winter"]
    for _i, obs in enumerate(observations):
        label = "With REM" if "rem" in obs.name else "Unmitigated"
        if label in recorded_labels:
            label = "_nolegend_"
        else:
            recorded_labels.append(label)
        identifier = obs.identifier.string_identifier
        x = len(identifier.strip("[]").replace('"', "").replace(" ", "").split(","))
        y = obs.value
        ax.errorbar(
            x,
            y,
            yerr=obs.uncertainty,
            capsize=4,
            color=cmap(0.85) if "rem" in obs.name else cmap(0.15),
            label=label,
            fmt="o",
            alpha=1,
            mec="black",
            markersize=5,
        )
        x_positions.append(x)
        ax.annotate(layout_short[identifier], (x, y))

    ax.set_xticks(x_positions, labels=[str(x) for x in x_positions])
    ax.grid()

    ax.axhline(0.5, linestyle="--", color="red", label="GME threshold")
    ax.set_ylim((0, 1))
    ax.set_title(f"GHZ fidelities of all qubit layouts\nbackend: {backend_name} --- {timestamp}")
    ax.set_xlabel("Number of qubits")
    ax.set_ylabel("Fidelity")
    ax.legend(framealpha=0.5, fontsize=8)
    plt.gcf().set_dpi(250)
    plt.close()
    return fig


class GHZBenchmark(Benchmark):
    """The GHZ Benchmark estimates the quality of generated Greenberger–Horne–Zeilinger states."""

    analysis_function = staticmethod(fidelity_analysis)

    @classmethod
    def name(cls) -> str:
        """Returns the name of the benchmark."""
        return "ghz"

    def __init__(self, backend: IQMBackend | IQMFacadeBackend | str, configuration: "GHZConfiguration"):
        """Construct the GHZBenchmark class.

        Args:
            backend: the backend to execute the benchmark on
            configuration: the configuration of the benchmark

        """
        super().__init__(backend, configuration)

        self.state_generation_routine = configuration.state_generation_routine
        if configuration.custom_qubits_array:
            self.custom_qubits_array = configuration.custom_qubits_array
        else:
            self.custom_qubits_array = [list(set(chain(*self.backend.coupling_map)))]
        self.qubit_counts = [len(layout) for layout in self.custom_qubits_array]

        self.qiskit_optim_level = configuration.qiskit_optim_level
        self.optimize_sqg = configuration.optimize_sqg
        self.fidelity_routine = configuration.fidelity_routine
        self.num_rms = configuration.num_rms
        self.rem = configuration.rem
        self.mit_shots = configuration.mit_shots
        self.timestamp = datetime.now(UTC).strftime(BENCHMARK_TIMESTAMP_FORMAT)
        self.execution_timestamp = ""

        self.tree_options = TreeSearchOptions(
            objective=configuration.tree_objective,
            idle_penalty=configuration.tree_idle_penalty,
            depth_penalty=configuration.tree_depth_penalty,
            cz_duration=configuration.tree_cz_duration,
        )
        self.tree_rank = configuration.tree_rank
        # Physical qubits in tree order, per layout, so every circuit of a layout is transpiled onto
        # the same mapping. See _transpilation_target.
        self.tree_orderings: dict[str, list[int]] = {}

    def _transpilation_target(self, qubit_layout: list[int]) -> tuple[Sequence[int], CouplingMap]:
        """Qubit order and coupling map to transpile this layout's circuits against.

        The GHZ circuit, the coherence circuits and the randomized-measurement circuits are
        transpiled by separate calls, so they have to agree on where the logical qubits go. For
        "tree" the tree search has already decided that, and the order it chose is what makes the
        tree edges land on real couplings without a layout search.

        Args:
            qubit_layout: The subset of system-qubits used in the protocol, indexed from 0.

        Returns:
            The qubits to target, and the coupling map to transpile to.

        """
        idx = BenchmarkObservationIdentifier(qubit_layout).string_identifier
        order = self.tree_orderings.get(idx)
        if order is not None:
            return order, set_coupling_map(order, self.backend, PhysicalLayout.FIXED)
        return qubit_layout, set_coupling_map(qubit_layout, self.backend, PhysicalLayout.FIXED)

    def _generate_tree_ghz(
        self, qubit_layout: list[int], qubit_count: int
    ) -> tuple[QuantumCircuit, list[QuantumCircuit]]:
        """Build and transpile the GHZ circuit of the "tree" routine.

        Args:
            qubit_layout: The subset of system-qubits used in the protocol, indexed from 0.
            qubit_count: The number of qubits the GHZ state should span.

        Returns:
            The untranspiled circuit and the transpiled circuits.

        """
        # has_resonators() rather than "move" in operation_names: the fake star backends do not list
        # move among their operations, but they do route through a resonator.
        if self.backend.has_resonators():
            qcvv_logger.warning(
                "The current backend is a star architecture, for which the tree search is a suboptimal "
                "state generation routine. Consider state_generation_routine='star_optimal'."
            )
        _qubit_mapping, metric_dict = extract_fidelities_unified(self.backend)
        ghz, tree_order, _candidate = generate_ghz_tree_optimal(
            qubit_layout,
            self.backend,
            metric_dict,
            qubit_count,
            self.tree_options,
            self.tree_rank,
        )
        self.tree_orderings[BenchmarkObservationIdentifier(qubit_layout).string_identifier] = list(tree_order)
        transpile_qubits, transpile_coupling_map = self._transpilation_target(qubit_layout)
        # The tree was built on the coupling map reduced to tree_order, so logical qubit i belongs on
        # reduced node i: the identity is the layout, and no search is needed.
        ghz_native_transpiled, _ = perform_backend_transpilation(
            [ghz],
            self.backend,
            transpile_qubits,
            transpile_coupling_map,
            qiskit_optim_level=self.qiskit_optim_level,
            optimize_sqg=self.optimize_sqg,
            initial_layout=list(range(ghz.num_qubits)),
        )
        return ghz, ghz_native_transpiled

    def generate_native_ghz(self, qubit_layout: list[int], qubit_count: int, routine: str) -> CircuitGroup:
        """Generate a circuit preparing a GHZ state.

        The GHZ state is generated according to a given routine and transpiled to the native gate set and topology.

        Args:
            qubit_layout: The subset of system-qubits used in the protocol, indexed from 0
            qubit_count:
                The number of qubits for which a GHZ state should be created. This values should be smaller or equal to
                the number of qubits in qubit_layout
            routine:
                The routine to generate the GHZ circuit

        Returns:
            QuantumCircuit implementing GHZ native state

        """
        circuit_group = CircuitGroup(name=f"{qubit_layout}_native_ghz")
        fixed_coupling_map = set_coupling_map(qubit_layout, self.backend, PhysicalLayout.FIXED)
        ghz_native_transpiled: list[QuantumCircuit]

        if routine == "naive":
            ghz: QuantumCircuit = generate_ghz_linear(qubit_count)
            circuit_group.add_circuit(ghz)
            ghz_native_transpiled, _ = perform_backend_transpilation(
                [ghz],
                self.backend,
                qubit_layout,
                fixed_coupling_map,
                qiskit_optim_level=self.qiskit_optim_level,
                optimize_sqg=self.optimize_sqg,
            )
            final_ghz = ghz_native_transpiled
        elif routine == "tree":
            ghz, final_ghz = self._generate_tree_ghz(qubit_layout, qubit_count)
            circuit_group.add_circuit(ghz)
        elif routine == "star":
            ghz = generate_ghz_star(qubit_count)
            circuit_group.add_circuit(ghz)
            ghz_native_transpiled, _ = perform_backend_transpilation(
                [ghz],
                self.backend,
                qubit_layout,
                fixed_coupling_map,
                qiskit_optim_level=self.qiskit_optim_level,
                optimize_sqg=self.optimize_sqg,
            )
            final_ghz = ghz_native_transpiled
        elif routine == "star_optimal":
            ghz = generate_ghz_star_optimal(qubit_layout, self.backend)
            circuit_group.add_circuit(ghz)
            ghz_native_transpiled = transpile_to_IQM(
                ghz,
                self.backend,
                optimize_single_qubits=self.optimize_sqg,
                optimization_level=self.qiskit_optim_level,
            )
            final_ghz = ghz_native_transpiled
        else:
            ghz_log = [
                generate_ghz_log_cruz(qubit_count),
                generate_ghz_log_mooney(qubit_count),
            ]
            ghz_native_transpiled, _ = perform_backend_transpilation(
                ghz_log,
                self.backend,
                qubit_layout,
                fixed_coupling_map,
                qiskit_optim_level=self.qiskit_optim_level,
                optimize_sqg=self.optimize_sqg,
            )
            # Use either the circuit with min depth after transpilation or min #2q gates
            if ghz_native_transpiled[0].depth() == ghz_native_transpiled[1].depth():
                index_min_2q = np.argmin([c.count_ops()["cz"] for c in ghz_native_transpiled])
                final_ghz = ghz_native_transpiled[index_min_2q]
                circuit_group.add_circuit(ghz_log[index_min_2q])
            else:
                index_min_depth = np.argmin([c.depth() for c in ghz_native_transpiled])
                final_ghz = ghz_native_transpiled[index_min_depth]
                circuit_group.add_circuit(ghz_log[index_min_depth])
        untranspiled = self.circuits["untranspiled_circuits"]
        if untranspiled is None:
            raise RuntimeError("untranspiled_circuits not found in circuits")
        untranspiled.circuit_groups.append(circuit_group)
        return CircuitGroup(name=f"{qubit_layout}_native_ghz", circuits=[final_ghz[0]])

    def generate_coherence_meas_circuits(self, qubit_layout: list[int], qubit_count: int) -> list[QuantumCircuit]:
        """Takes a given GHZ circuit and outputs circuits needed to measure fidelity via mult. q. coherences method.

        Args:
            qubit_layout:
                The subset of system-qubits used in the protocol, indexed from 0
            qubit_count:
                The number of qubits for which a GHZ state should be created. This values should be smaller or equal to
                the number of qubits in qubit_layout
        Returns:
            A list of transpiled quantum circuits to be measured

        """
        idx = BenchmarkObservationIdentifier(qubit_layout).string_identifier
        untranspiled = self.circuits["untranspiled_circuits"]
        if untranspiled is None:
            raise RuntimeError("untranspiled_circuits not found in circuits")
        circuit_group = untranspiled[f"{qubit_layout}_native_ghz"]
        if circuit_group is None:
            raise RuntimeError(f"Circuit group {qubit_layout}_native_ghz not found")
        qc_list = circuit_group.circuits

        qc = qc_list[0].copy()
        qc.remove_final_measurements()
        phases = [np.pi * i / (qubit_count + 1) for i in range(2 * qubit_count + 2)]
        if self.state_generation_routine == "star_optimal":
            qc_inv = generate_ghz_star_optimal(qubit_layout, self.backend, inv=True)
            for phase in phases:
                qc_phase = qc.copy()
                qc_phase.barrier()
                for _, qubit in enumerate(qubit_layout):
                    qc_phase.p(phase, qubit)
                qc_phase.barrier()
                qc_phase.compose(qc_inv, inplace=True)
                qc_phase.measure(qubit_layout, list(range(qubit_count)))
                qc_list.append(qc_phase)
        else:
            qc_inv = qc.inverse()
            for phase in phases:
                qc_phase = qc.copy()
                qc_phase.barrier()
                for qubit, _ in enumerate(qubit_layout):
                    qc_phase.p(phase, qubit)
                qc_phase.barrier()
                qc_phase.compose(qc_inv, inplace=True)
                qc_phase.measure_active()
                qc_list.append(qc_phase)

        # These circuits are built from the GHZ circuit, so they carry its logical-qubit labelling
        # and have to be placed the same way it was; transpiling them against the raw qubit_layout
        # would put them on a different physical mapping and make the coherence fidelity meaningless.
        transpile_qubits, transpile_coupling_map = self._transpilation_target(qubit_layout)

        if self.state_generation_routine == "star_optimal":
            qc_list_transpiled = [
                transpile_to_IQM(
                    ghz,
                    self.backend,
                    optimize_single_qubits=self.optimize_sqg,
                    optimization_level=self.qiskit_optim_level,
                )
                for ghz in qc_list
            ]

        else:
            qc_list_transpiled, _ = perform_backend_transpilation(
                qc_list,
                self.backend,
                transpile_qubits,
                transpile_coupling_map,
                qiskit_optim_level=self.qiskit_optim_level,
                optimize_sqg=self.optimize_sqg,
            )
        circuit_group = CircuitGroup(name=idx, circuits=qc_list)
        untranspiled = self.circuits["untranspiled_circuits"]
        if untranspiled is None:
            raise RuntimeError("untranspiled_circuits not found in circuits")
        untranspiled.circuit_groups.append(circuit_group)
        return qc_list_transpiled

    @timeit
    def append_rms(
        self,
        num_rms: int,
        qubit_layout: list[int],
    ) -> list[QuantumCircuit]:
        """Appends 1Q Clifford gates sampled uniformly at random to all qubits in the given circuit.

        Args:
            num_rms: How many randomized measurement circuits are generated
            qubit_layout: The subset of system-qubits used in the protocol, indexed from 0
        Returns:
            List of the original circuit with 1Q Clifford gates appended to it

        """
        idx = BenchmarkObservationIdentifier(qubit_layout).string_identifier
        # As in generate_coherence_meas_circuits: these are copies of the GHZ circuit, so they must
        # be placed exactly as it was.
        transpile_qubits, transpile_coupling_map = self._transpilation_target(qubit_layout)
        untranspiled = self.circuits["untranspiled_circuits"]
        if untranspiled is None:
            raise RuntimeError("untranspiled_circuits not found in circuits")

        circuit_group = untranspiled[f"{qubit_layout}_native_ghz"]
        if circuit_group is None:
            raise RuntimeError(f"Circuit group {qubit_layout}_native_ghz not found")

        circuit = circuit_group.circuits[0]
        rm_circuits: list[QuantumCircuit] = []
        for _ in range(num_rms):
            rm_circ = circuit.copy()
            # It shouldn't matter if measurement bits get scrambled
            rm_circ.remove_final_measurements()
            rm_circ.barrier()

            active_qubits = set()
            data = rm_circ.data
            for instruction in data:
                for qubit in instruction[1]:
                    active_qubits.add(rm_circ.find_bit(qubit)[0])
            for q in active_qubits:
                if self.backend is not None:
                    rand_clifford = random_clifford(1).to_circuit()
                else:
                    rand_clifford = random_clifford(1).to_instruction()
                rm_circ.compose(rand_clifford, qubits=[q], inplace=True)

            rm_circ.measure_active()
            rm_circuits.append(rm_circ)

        rm_circuits_transpiled, _ = perform_backend_transpilation(
            rm_circuits,
            self.backend,
            transpile_qubits,
            transpile_coupling_map,
            qiskit_optim_level=self.qiskit_optim_level,
            optimize_sqg=self.optimize_sqg,
        )
        untranspiled_rm_group = CircuitGroup(circuits=rm_circuits, name=f"{idx}_rm_circuits")
        untranspiled = cast(BenchmarkCircuit, self.circuits["untranspiled_circuits"])
        untranspiled.circuit_groups.append(untranspiled_rm_group)

        return rm_circuits_transpiled

    def generate_readout_circuit(self, qubit_layout: list[int], qubit_count: int) -> CircuitGroup:
        """A wrapper for the creation of different circuits to estimate the fidelity.

        Args:
            qubit_layout:
                The subset of system-qubits used in the protocol, indexed from 0
            qubit_count:
                The number of qubits for which a GHZ state should be created. This values should be smaller or equal to
                the number of qubits in qubit_layout

        Returns:
            A list of transpiled quantum circuits to be measured

        """
        # Generate the list of circuits

        qcvv_logger.info(f"Now generating a {len(qubit_layout)}-qubit GHZ state on qubits {qubit_layout}")
        transpiled_ghz_group: CircuitGroup = self.generate_native_ghz(
            qubit_layout, qubit_count, self.state_generation_routine
        )
        match self.fidelity_routine:
            case "randomized_measurements":
                all_circuits_list, _ = self.append_rms(cast(int, self.num_rms), qubit_layout)
                transpiled_ghz_group.circuits = all_circuits_list
            case "coherences":
                all_circuits_list = self.generate_coherence_meas_circuits(qubit_layout, qubit_count)
                transpiled_ghz_group.circuits = all_circuits_list
        transpiled = cast(BenchmarkCircuit, self.circuits["transpiled_circuits"])
        transpiled.circuit_groups.append(transpiled_ghz_group)
        return transpiled_ghz_group

    def add_configuration_to_dataset(self, dataset: xr.Dataset) -> None:  # CHECK
        """Creates a xarray.Dataset and adds the circuits and configuration metadata to it.

        Args:
            dataset: The dataset containing the benchmark information
        Returns:
            Dataset to be used for further data storage

        """
        for key, value in self.configuration:
            if key == "benchmark":  # Avoid saving the class object
                dataset.attrs[key] = value.name
            else:
                dataset.attrs[key] = value
        dataset.attrs["backend_name"] = self.backend_name
        dataset.attrs["backend_topology"] = "star" if "move" in self.backend.operation_names else "crystal"
        dataset.attrs["backend_num_qubits"] = self.backend.num_qubits
        dataset.attrs["execution_timestamp"] = self.execution_timestamp
        dataset.attrs["fidelity_routine"] = self.fidelity_routine

    def execute(self, backend: IQMBackend | IQMFacadeBackend) -> xr.Dataset:
        """Executes the benchmark."""
        self.execution_timestamp = datetime.now(UTC).strftime(BENCHMARK_TIMESTAMP_FORMAT)
        total_submit: float = 0
        total_retrieve: float = 0
        aux_custom_qubits_array = cast(list[list[int]], self.custom_qubits_array).copy()
        dataset = xr.Dataset()

        # Submit all
        all_jobs: dict = {}

        self.circuits = Circuits()
        self.circuits.benchmark_circuits.append(BenchmarkCircuit(name="transpiled_circuits"))
        self.circuits.benchmark_circuits.append(BenchmarkCircuit(name="untranspiled_circuits"))
        for qubit_layout in aux_custom_qubits_array:
            identifier = BenchmarkObservationIdentifier(qubit_layout)
            idx = identifier.string_identifier
            qubit_count = len(qubit_layout)
            circuit_group: CircuitGroup = self.generate_readout_circuit(qubit_layout, qubit_count)
            transpiled_circuit_dict = {tuple(qubit_layout): circuit_group.circuits}
            all_jobs[idx], time_submit = submit_execute(
                transpiled_circuit_dict,
                backend,
                self.shots,
                max_gates_per_batch=self.max_gates_per_batch,
                max_circuits_per_batch=self.configuration.max_circuits_per_batch,
                circuit_compilation_options=self.circuit_compilation_options,
            )
            total_submit += time_submit

        # Retrieve all
        for qubit_layout in aux_custom_qubits_array:
            identifier = BenchmarkObservationIdentifier(qubit_layout)
            idx = identifier.string_identifier
            qubit_count = len(qubit_layout)
            counts, time_retrieve = retrieve_all_counts(all_jobs[idx])
            total_retrieve += time_retrieve
            dataset, _ = add_counts_to_dataset(counts, idx, dataset)
            if self.rem:
                qcvv_logger.info("Applying readout error mitigation")
                transpiled_circuits = cast(BenchmarkCircuit, self.circuits["transpiled_circuits"])
                circuit_group = cast(CircuitGroup, transpiled_circuits[f"{idx}_native_ghz"])
                rem_results, _ = apply_readout_error_mitigation(
                    backend,
                    circuit_group.circuits,
                    counts,
                    self.mit_shots,
                    circuit_compilation_options=self.circuit_compilation_options,
                )
                rem_results_dist = [counts_mit.nearest_probability_distribution() for counts_mit in rem_results]
                dataset, _ = add_counts_to_dataset(rem_results_dist, f"{idx}_rem", dataset)

        self.add_configuration_to_dataset(dataset)
        dataset.attrs["total_submit_time"] = total_submit
        dataset.attrs["total_retrieve_time"] = total_retrieve
        return dataset


class GHZConfiguration(BenchmarkConfigurationBase):
    """GHZ state configuration.

    Attributes:
        benchmark: GHZBenchmark
        state_generation_routine: The routine to construct circuits generating a GHZ state. Must be one
            of STATE_GENERATION_ROUTINES; an unknown value is rejected rather than silently treated as
            "log". Possible values:

            - "tree" (default): Optimized GHZ state generation circuit that takes the qubit coupling,
              the CZ and single-qubit gate fidelities, and the time each qubit spends idling into
              account. The qubits, the spanning tree and the gate order are chosen together, and what
              the search minimizes is set by tree_objective below.
              See iqm.benchmarks.entanglement.ghz_optimal.
            - "log": Optimized circuit with parallel application of CX gates such that the number of
              CX gates scales logarithmically in the system size. This implementation currently does
              not take connectivity on the backend into account.
            - "naive": Applies the naive textbook circuit with #CX gates scaling linearly in the
              system size.
            - "star", "star_optimal": For star (resonator) topologies.

        custom_qubits_array: A sequence (e.g., Tuple or List) of sequences of physical qubit layouts,
            as specified by integer labels, where the benchmark is meant to be run. If None, takes all
            qubits specified in the backend coupling map.
        qiskit_optim_level: The optimization level used for transpilation to backend architecture.
        optimize_sqg: Whether consecutive single qubit gates are optimized for reduced gate count via
            iqm.qiskit_iqm.iqm_transpilation.optimize_single_qubit_gates
        fidelity_routine: The method with which the fidelity is estimated. Possible values:

            - "coherences": The multiple quantum coherences method as in [Mooney, 2021]
            - "randomized_measurements": Fidelity estimation via randomized measurements outlined in
              https://arxiv.org/abs/1812.02624

        num_rms: The number of randomized measurements used if the respective fidelity routine is chosen
        rem: Boolean flag determining if readout error mitigation is used
        mit_shots: Total number of shots for readout error mitigation
        tree_objective: What the "tree" search minimizes, one of
            iqm.benchmarks.entanglement.ghz_optimal.TREE_OBJECTIVES. The default,
            "estimated_fidelity", trades gate fidelity against the time qubits spend idling. Set
            "min_error" to minimize gate error alone, which is what this routine did before the
            idle term existed and is the setting to use to reproduce older results.
        tree_idle_penalty: Weight of the idle-decoherence term of the "estimated_fidelity" objective;
            0.0 disables it, which makes the objective equivalent to "min_error". Values above 1.0
            buy depth more aggressively.
        tree_depth_penalty: Cost of one CZ layer for the "depth_penalty" objective, in units of
            -log(fidelity).
        tree_cz_duration: Duration assumed for one CZ layer, in seconds. Calibration data carries no
            gate durations, so this is an assumption rather than a measurement.
        tree_rank: Which of the ranked "tree" candidates to run, 0 being the best-scoring one. The
            ranking is logged, so a later run can ask for a different root.

    """

    benchmark: type[Benchmark] = GHZBenchmark
    state_generation_routine: str = "tree"
    custom_qubits_array: Sequence[Sequence[int]] | None = None
    shots: int = 2**10
    qiskit_optim_level: int = 3
    optimize_sqg: bool = True
    fidelity_routine: str = "coherences"
    num_rms: int | None = 100
    rem: bool = True
    mit_shots: int = 1_000
    tree_objective: str = "estimated_fidelity"
    tree_idle_penalty: float = 1.0
    tree_depth_penalty: float = 0.05
    tree_cz_duration: float = CZ_GATE_DURATION
    tree_rank: int = 0

    @field_validator("state_generation_routine")
    @classmethod
    def _check_state_generation_routine(cls, value: str) -> str:
        """Reject a routine name that would otherwise silently build the log-depth circuits."""
        if value not in STATE_GENERATION_ROUTINES:
            raise ValueError(
                f"Unknown state_generation_routine {value!r}; expected one of {STATE_GENERATION_ROUTINES}."
            )
        return value

    @field_validator("tree_objective")
    @classmethod
    def _check_tree_objective(cls, value: str) -> str:
        """Reject a tree objective the search does not implement."""
        if value not in TREE_OBJECTIVES:
            raise ValueError(f"Unknown tree_objective {value!r}; expected one of {TREE_OBJECTIVES}.")
        return value
