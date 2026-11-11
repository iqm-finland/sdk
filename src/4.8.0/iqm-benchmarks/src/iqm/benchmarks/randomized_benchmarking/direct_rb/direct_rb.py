"""Direct Randomized Benchmarking."""

# ruff: noqa: PLR0912, PLR0913
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
import secrets
from typing import Any, cast

from iqm.benchmarks import (
    Benchmark,
    BenchmarkAnalysisResult,
    BenchmarkCircuit,
    BenchmarkRunResult,
    CircuitGroup,
    Circuits,
)
from iqm.benchmarks.benchmark_definition import (
    BENCHMARK_TIMESTAMP_FORMAT,
    BenchmarkConfigurationBase,
    BenchmarkObservation,
    BenchmarkObservationIdentifier,
    add_counts_to_dataset,
)
from iqm.benchmarks.logging_config import qcvv_logger
from iqm.benchmarks.randomized_benchmarking.randomized_benchmarking_common import (
    compute_inverse_clifford,
    edge_grab,
    exponential_rb,
    fit_decay_lmfit,
    get_survival_probabilities,
    import_native_gate_cliffords,
    lmfit_minimizer,
    plot_rb_decay,
    relabel_qubits_array_from_zero,
    survival_probabilities_parallel,
)
from iqm.benchmarks.utils import (
    RoutingMethod,
    get_iqm_backend,
    retrieve_all_counts,
    retrieve_all_job_metadata,
    submit_execute,
    timeit,
    xrvariable_to_counts,
)
from iqm.iqm_client.models import CircuitCompilationOptions
from iqm.qiskit_iqm import IQMCircuit as QuantumCircuit, transpile_to_IQM
from iqm.qiskit_iqm.iqm_backend import IQMBackendBase
from iqm.qiskit_iqm.iqm_transpilation import IQMOptimizeSingleQubitGates
import numpy as np
from qiskit import ClassicalRegister, transpile
from qiskit.quantum_info import Clifford, random_clifford
from qiskit.transpiler import PassManager
import xarray as xr


@timeit
def generate_drb_circuits(
    qubits: Sequence[int],
    depth: int,
    circ_samples: int,
    backend_arg: IQMBackendBase | str,
    density_2q_gates: float = 0.25,
    two_qubit_gate_ensemble: dict[str, float] | None = None,
    clifford_sqg_probability: float = 1.0,
    sqg_gate_ensemble: dict[str, float] | None = None,
    qiskit_optim_level: int = 1,
    routing_method: RoutingMethod = RoutingMethod.BASIC,
) -> dict[str, list[QuantumCircuit]]:
    """Generates lists of samples of Direct RB circuits.

    The structure is: Stabilizer preparation - Layers of canonical randomly sampled gates - Stabilizer measurement

    Args:
        qubits: Qubits of the backend.
        depth: Depth (number of canonical layers) of the circuit.
        circ_samples: Number of circuit samples to generate.
        backend_arg: Backend.
        density_2q_gates: Expected density of 2Q gates.
        two_qubit_gate_ensemble: Dictionary with keys being str specifying 2Q gates, and values being corresponding
            probabilities. Default is None.
        clifford_sqg_probability: Probability with which to uniformly sample Clifford 1Q gates. Default is 1.0.
        sqg_gate_ensemble: Dictionary with keys being str specifying 1Q gates, and values being corresponding
            probabilities. Default is None.
        qiskit_optim_level: Qiskit transpiler optimization level. Default is 1.
        routing_method: Qiskit transpiler routing method. Default is "basic".

    Returns:
        Dictionary with keys "transpiled", "untranspiled" and values as lists of respective DRB circuits.

    """
    num_qubits = len(qubits)

    # Retrieve backend
    if isinstance(backend_arg, str):
        retrieved_backend = get_iqm_backend(backend_arg)
    else:
        if not isinstance(backend_arg, IQMBackendBase):
            raise TypeError("backend_arg must be of type IQMBackendBase or str")
        retrieved_backend = backend_arg

    # Check if backend includes MOVE gates and set coupling map
    if retrieved_backend.has_resonators():
        # All-to-all coupling map on the active qubits
        effective_coupling_map = [[x, y] for x in qubits for y in qubits if x != y]
    else:
        effective_coupling_map = retrieved_backend.coupling_map

    # Initialize the list of circuits
    all_circuits = {}
    drb_circuits_untranspiled: list[QuantumCircuit] = []
    drb_circuits_transpiled: list[QuantumCircuit] = []

    # simulator = AerSimulator(method=simulation_method)

    for _ in range(circ_samples):
        # Sample Clifford for stabilizer preparation
        clifford_layer = random_clifford(num_qubits)
        # NB: The DRB paper contains a more elaborated stabilizer compilation algorithm.
        # Not having it WILL be an issue here for larger num qubits !
        # Intended usage, however, is solely for 2-qubit DRB subroutines.

        # Sample the layers using edge grab sampler - different samplers may be conditionally chosen here in the future
        cycle_layers = edge_grab(
            qubits,
            depth,
            backend_arg,
            density_2q_gates,
            two_qubit_gate_ensemble,
            clifford_sqg_probability,
            sqg_gate_ensemble,
        )

        # Initialize the quantum circuit object
        circ = QuantumCircuit(num_qubits)

        # Add the edge Clifford
        circ.compose(
            clifford_layer.to_instruction(),
            qubits=list(range(num_qubits)),
            inplace=True,
        )
        circ.barrier()

        # Add the cycle layers
        for k in range(depth):
            circ.compose(cycle_layers[k], inplace=True)
            circ.barrier()

        # Add the inverse Clifford
        circ.compose(
            Clifford(circ.to_instruction().inverse()).to_instruction(),
            qubits=list(range(num_qubits)),
            inplace=True,
        )
        # Similarly, here the DRB paper contains a stabilizer measurement, determined in a more elaborated way.
        # The stabilizer measurement should effectively render the circuit to a Pauli gate (here always the identity).
        # Would need to modify this for larger num qubits !
        # Here, for 2-qubit DRB subroutines, it *should* suffice (in principle) to compile the inverse.

        # Add measurements to untranspiled - after!
        # THIS LINE IS ONLY NEEDED IF STABILIZER MEASUREMENT IS NOT TAKEN TO IDENTITY
        # circ_untranspiled = transpile(Clifford(circ.copy()).to_circuit(), simulator)
        circ_untranspiled = circ.copy()
        circ_untranspiled.measure_all()

        # Add measurements to transpiled - before!
        circ.measure_all()

        if retrieved_backend.has_resonators():
            circ_transpiled = transpile_to_IQM(
                circ,
                backend=retrieved_backend,
                coupling_map=effective_coupling_map,
                optimization_level=qiskit_optim_level,
                initial_layout=qubits,
                routing_method=routing_method.value,
            )
        else:
            circ_transpiled = transpile(
                circ,
                backend=retrieved_backend,
                coupling_map=effective_coupling_map,
                optimization_level=qiskit_optim_level,
                initial_layout=qubits,
                routing_method=routing_method.value,
            )

        drb_circuits_untranspiled.append(circ_untranspiled)
        drb_circuits_transpiled.append(circ_transpiled)

    # Store the circuits
    all_circuits.update(
        {
            "untranspiled": drb_circuits_untranspiled,
            "transpiled": drb_circuits_transpiled,
        }
    )

    return all_circuits


@timeit
def generate_fixed_depth_parallel_drb_circuits(
    qubits_array: Sequence[Sequence[int]],
    depth: int,
    num_circuit_samples: int,
    backend_arg: str | IQMBackendBase,
    assigned_density_2q_gates: dict[str, float],
    assigned_two_qubit_gate_ensembles: dict[str, dict[str, float]],
    assigned_clifford_sqg_probabilities: dict[str, float],
    assigned_sqg_gate_ensembles: dict[str, dict[str, float]],
    cliffords_1q: dict[str, QuantumCircuit],
    cliffords_2q: dict[str, QuantumCircuit],
    qiskit_optim_level: int = 1,
    routing_method: RoutingMethod = RoutingMethod.BASIC,
    is_eplg: bool = False,
) -> dict[str, list[QuantumCircuit]]:
    """Generates DRB circuits in parallel on multiple qubit layouts.

        The circuits follow a layered pattern with barriers, taylored to measure EPLG (arXiv:2311.05933),
        with layers of random Cliffords interleaved among sampled layers of 2Q gates and sequence inversion.

    Args:
        qubits_array: The array of physical qubit layouts on which to generate
            parallel DRB circuits.
        depth: The depth (number of canonical DRB layers) of the circuits.
        num_circuit_samples: The number of DRB circuits to generate.
        backend_arg: The backend on which to generate the circuits.
        assigned_density_2q_gates: The expected densities of 2-qubit gates in the
            final circuits per qubit layout.
        assigned_two_qubit_gate_ensembles: The two-qubit gate ensembles to use in the
            random DRB circuits per qubit layout.
        assigned_clifford_sqg_probabilities: Probability with which to uniformly sample
            Clifford 1Q gates per qubit layout.
        assigned_sqg_gate_ensembles: A dictionary with keys being str specifying 1Q gates,
            and values being corresponding probabilities per qubit layout.
        cliffords_1q: dictionary of 1-qubit Cliffords in terms of IQM-native r gates.
        cliffords_2q: dictionary of 2-qubit Cliffords in terms of IQM-native r and CZ gates.
        qiskit_optim_level: Qiskit transpiler optimization level.
        routing_method: Qiskit transpiler routing method.
        is_eplg: Whether the circuits belong to an EPLG experiment.
                        * If True a single layer is generated.

    Returns:
        A dictionary of untranspiled and transpiled lists of
            parallel (simultaneous) DRB circuits.

    """
    backend = get_iqm_backend(backend_arg) if isinstance(backend_arg, str) else backend_arg

    # Check if backend includes MOVE gates and set coupling map
    flat_qubits_array = [x for y in qubits_array for x in y]
    if backend.has_resonators():
        # All-to-all coupling map on the active qubits
        effective_coupling_map = [[x, y] for x in flat_qubits_array for y in flat_qubits_array if x != y]
        is_circuit_native = False
    else:
        effective_coupling_map = backend.coupling_map
        is_circuit_native = True

    # Identify total amount of qubits
    qubit_counts = [len(x) for x in qubits_array]

    # Shuffle qubits_array: we don't want unnecessary qubit registers
    shuffled_qubits_array = relabel_qubits_array_from_zero(cast(list[list[int]], qubits_array))
    # The total amount of qubits the circuits will have
    n_qubits = sum(qubit_counts)

    # Get the keys of the Clifford dictionaries
    clifford_1q_keys = list(cliffords_1q.keys())
    # clifford_2q_keys = list(cliffords_2q.keys())

    # Generate the circuit samples
    # Initialize the list of circuits
    all_circuits = {}
    drb_circuits_untranspiled: list[QuantumCircuit] = []
    drb_circuits_transpiled: list[QuantumCircuit] = []

    cycle_layers = {}

    # Generate the layer if EPLG:
    # this will be repeated in all samples (and all depths)! So can be done outside the loop over circuit samples.
    if is_eplg:
        for q_idx, q in enumerate(shuffled_qubits_array):
            original_qubits = str(qubits_array[q_idx])
            if any(x != "CZGate" for x in assigned_two_qubit_gate_ensembles[original_qubits]):
                is_circuit_native = False
            cycle_layers[str(q)] = edge_grab(
                qubits_array[q_idx],
                depth,
                backend_arg,
                assigned_density_2q_gates[original_qubits],
                assigned_two_qubit_gate_ensembles[original_qubits],
                assigned_clifford_sqg_probabilities[original_qubits],
                assigned_sqg_gate_ensembles[original_qubits],
            )

    for _ in range(num_circuit_samples):
        # Initialize the quantum circuit object
        circ = QuantumCircuit(n_qubits)

        # Generate small circuits to track inverses
        local_circs = {str(q): QuantumCircuit(len(q)) for q in shuffled_qubits_array}

        # Sample the layers if EPLG is False.
        if not is_eplg:
            for q_idx, q in enumerate(shuffled_qubits_array):
                original_qubits = str(qubits_array[q_idx])
                if any(x != "CZGate" for x in assigned_two_qubit_gate_ensembles[original_qubits]):
                    is_circuit_native = False
                cycle_layers[str(q)] = edge_grab(
                    qubits_array[q_idx],
                    depth,
                    backend_arg,
                    assigned_density_2q_gates[original_qubits],
                    assigned_two_qubit_gate_ensembles[original_qubits],
                    assigned_clifford_sqg_probabilities[original_qubits],
                    assigned_sqg_gate_ensembles[original_qubits],
                )

        # Add the cycle layers
        for k in range(depth):
            # Add the edge Clifford
            # The DRB paper here contains a general stabilizer preparation.
            # We will stick to 1Q Clifford gates for now.
            for q in shuffled_qubits_array:
                for idx, i in enumerate(q):
                    rand_key = secrets.choice(clifford_1q_keys)
                    rand_clif_1q = cast(dict, cliffords_1q)[rand_key]
                    # rand_clif = random_clifford(1)
                    circ.compose(rand_clif_1q, qubits=[i], inplace=True)
                    local_circs[str(q)].compose(rand_clif_1q, qubits=[idx], inplace=True)
            circ.barrier()

            for q in shuffled_qubits_array:
                circ.compose(cycle_layers[str(q)][k], qubits=q, inplace=True)
                local_circs[str(q)].compose(cycle_layers[str(q)][k], inplace=True)
            circ.barrier()

        # Add the inverse Clifford
        for q in shuffled_qubits_array:
            clifford_dict = cliffords_1q if len(q) == 1 else cliffords_2q
            circ.compose(
                compute_inverse_clifford(local_circs[str(q)], clifford_dict),
                qubits=q,
                inplace=True,
            )
        circ.barrier()
        for q_idx, q in enumerate(shuffled_qubits_array):
            original_qubits = str(qubits_array[q_idx])
            local_register = ClassicalRegister(len(q), original_qubits)
            circ.add_register(local_register)
            circ.measure(q, local_register)
        # Similarly, here the DRB paper contains a stabilizer measurement, determined in a more elaborated way.
        # The stabilizer measurement should effectively render the circuit to a Pauli gate (here always the identity).
        # Would need to modify this for larger num qubits !
        # Here, for 2-qubit DRB subroutines, it *should* suffice (in principle) to compile the inverse.

        circ_untranspiled = circ.copy()

        if is_circuit_native:  # Simply compose into a larger circuit
            circ_transpiled = QuantumCircuit(backend.num_qubits)
            circ_transpiled.compose(circ_untranspiled, qubits=flat_qubits_array, inplace=True)
            if is_eplg:
                # Merge each qubit's single-qubit gates between two barriers into one PRX pulse, as qiskit.transpile
                # does; otherwise a native Clifford takes up to 3 pulses per layer.
                circ_transpiled = PassManager(IQMOptimizeSingleQubitGates(drop_final_rz=True)).run(circ_transpiled)
        elif backend.has_resonators():
            circ_transpiled = transpile_to_IQM(
                circ,
                backend=backend,
                coupling_map=effective_coupling_map,
                optimization_level=qiskit_optim_level,
                initial_layout=flat_qubits_array,
                routing_method=routing_method.value,
            )
        else:
            circ_transpiled = transpile(
                circ,
                backend=backend,
                coupling_map=effective_coupling_map,
                optimization_level=qiskit_optim_level,
                initial_layout=flat_qubits_array,
                routing_method=routing_method.value,
            )

        drb_circuits_untranspiled.append(circ_untranspiled)
        drb_circuits_transpiled.append(circ_transpiled)

    # Store the circuits
    all_circuits.update(
        {
            "untranspiled": drb_circuits_untranspiled,
            "transpiled": drb_circuits_transpiled,
        }
    )
    return all_circuits


def direct_rb_analysis(run: BenchmarkRunResult) -> BenchmarkAnalysisResult:
    """Direct RB analysis function.

    Args:
        run: The result of the benchmark run.

    Returns:
        AnalysisResult corresponding to DRB.

    """
    dataset = run.dataset.copy(deep=True)
    observations: list[BenchmarkObservation] = []
    obs_dict = {}
    plots = {}

    is_parallel_execution = dataset.attrs["parallel_execution"]
    all_qubits_array = dataset.attrs["qubits_array"]
    depths = dataset.attrs["depths"]

    num_circuit_samples = dataset.attrs["num_circuit_samples"]

    density_2q_gates = dataset.attrs["densities_2q_gates"]
    two_qubit_gate_ensemble = dataset.attrs["two_qubit_gate_ensembles"]

    is_eplg = dataset.attrs["is_eplg"]

    all_noisy_counts: dict[str, dict[int, list[dict[str, int]]]] = {}

    if isinstance(all_qubits_array[0][0], int):
        wrapped_all_qubits_array = [all_qubits_array]
        flat_all_qubits_array_reshaped = [x for y in wrapped_all_qubits_array for x in y]
    else:
        wrapped_all_qubits_array = all_qubits_array
        flat_all_qubits_array_reshaped = [x for y in all_qubits_array for x in y]

    # Disambiguated qubit layout appearing in several groups (e.g. idle qubits in different EPLG layers).
    layout_counts = Counter(str(q) for q in flat_all_qubits_array_reshaped)

    for q_array_idx, qubits_array in enumerate(wrapped_all_qubits_array):
        # Polarizations are stored per group, so that repeated layouts across groups do not overwrite each other
        polarizations: dict[str, dict[int, list[float]]] = {str(q): {} for q in qubits_array}
        if is_parallel_execution:
            qcvv_logger.info(f"Post-processing parallel Direct RB on qubits {qubits_array}.")
            all_noisy_counts[str(qubits_array)] = {}
            for depth in depths:
                identifier = f"qubits_{str(qubits_array)}_depth_{str(depth)}"
                all_noisy_counts[str(qubits_array)][depth] = xrvariable_to_counts(
                    dataset, identifier, num_circuit_samples
                )

                qcvv_logger.info(f"Depth {depth}")

                # Retrieve the marginalized survival probabilities
                all_survival_probabilities = survival_probabilities_parallel(
                    qubits_array,
                    all_noisy_counts[str(qubits_array)][depth],
                    separate_registers=True,
                )

                # The marginalized survival probabilities will be arranged by qubit layouts
                for qubits_str in all_survival_probabilities:
                    polarizations[qubits_str][depth] = all_survival_probabilities[qubits_str]
                # Remaining analysis is the same regardless of whether execution was in parallel or sequential
        else:  # sequential
            qcvv_logger.info(f"Post-processing sequential Direct RB for qubits {qubits_array}")
            for q in qubits_array:
                all_noisy_counts[str(q)] = {}
                num_qubits = len(q)
                polarizations[str(q)] = {}
                for depth in depths:
                    identifier = f"qubits_{str(q)}_depth_{str(depth)}"
                    all_noisy_counts[str(q)][depth] = xrvariable_to_counts(dataset, identifier, num_circuit_samples)

                    qcvv_logger.info(f"Qubits {q} and depth {depth}")
                    polarizations[str(q)][depth] = get_survival_probabilities(
                        num_qubits, all_noisy_counts[str(q)][depth]
                    )
                    # Remaining analysis is the same regardless of whether execution was in parallel or sequential

        # All remaining (fitting & plotting) is done per qubit layout
        for qubits_idx, qubits in enumerate(qubits_array):
            # Fit decays
            list_of_polarizations = list(polarizations[str(qubits)].values())
            fit_data, fit_parameters = fit_decay_lmfit(exponential_rb, qubits, list_of_polarizations, "drb")
            rb_fit_results = lmfit_minimizer(fit_parameters, fit_data, depths, exponential_rb)

            average_polarizations = {d: np.mean(polarizations[str(qubits)][d]) for d in depths}
            stddevs_from_mean = {
                d: np.std(polarizations[str(qubits)][d]) / np.sqrt(num_circuit_samples) for d in depths
            }
            popt = {
                "amplitude": rb_fit_results.params["amplitude_1"],
                "offset": rb_fit_results.params["offset_1"],
                "decay_rate": rb_fit_results.params["p_drb"],
            }
            fidelity = rb_fit_results.params["fidelity_drb"]

            processed_results = {
                "average_gate_fidelity": {
                    "value": fidelity.value,
                    "uncertainty": fidelity.stderr,
                },
            }

            dataset.attrs[q_array_idx].update(
                {
                    qubits_idx: {
                        "decay_rate": {
                            "value": popt["decay_rate"].value,
                            "uncertainty": popt["decay_rate"].stderr,
                        },
                        "fit_amplitude": {
                            "value": popt["amplitude"].value,
                            "uncertainty": popt["amplitude"].stderr,
                        },
                        "fit_offset": {
                            "value": popt["offset"].value,
                            "uncertainty": popt["offset"].stderr,
                        },
                        "polarizations": polarizations[str(qubits)],
                        "average_polarization_nominal_values": average_polarizations,
                        "average_polarization_stderr": stddevs_from_mean,
                        "fitting_method": str(rb_fit_results.method),
                        "num_function_evals": int(rb_fit_results.nfev),
                        "data_points": int(rb_fit_results.ndata),
                        "num_variables": int(rb_fit_results.nvarys),
                        "chi_square": float(rb_fit_results.chisqr),
                        "reduced_chi_square": float(rb_fit_results.redchi),
                        "Akaike_info_crit": float(rb_fit_results.aic),
                        "Bayesian_info_crit": float(rb_fit_results.bic),
                    }
                }
            )

            obs_dict.update({qubits_idx: processed_results})
            is_repeated_layout = layout_counts[str(qubits)] > 1
            observations.extend(
                [
                    BenchmarkObservation(
                        name=key,
                        identifier=BenchmarkObservationIdentifier(
                            qubit_indices=qubits,
                            custom_identifier=f"{qubits}_group_{q_array_idx}" if is_repeated_layout else None,
                        ),
                        value=values["value"],
                        uncertainty=values["uncertainty"],
                    )
                    for key, values in processed_results.items()
                ]
            )

            # Generate individual decay plots
            fig_name, fig = plot_rb_decay(
                identifier="drb",
                qubits_array=[qubits],
                dataset=dataset,
                observations=obs_dict,
                mrb_2q_density=density_2q_gates,  # Misnomer coming from MRB - ignore
                mrb_2q_ensemble=two_qubit_gate_ensemble,
                is_eplg=is_eplg,
                group_idx=q_array_idx if len(wrapped_all_qubits_array) > 1 else None,
            )
            if is_repeated_layout:
                fig_name = f"{fig_name}_group_{q_array_idx}"
            plots[fig_name] = fig

    return BenchmarkAnalysisResult(dataset=dataset, observations=observations, plots=plots)


def submit_in_batches(
    circuits_by_id: dict[str, tuple[Sequence[int], list[QuantumCircuit]]],
    backend: IQMBackendBase,
    shots: int,
    max_circuits_per_batch: int | None = None,
    max_gates_per_batch: int | None = None,
    circuit_compilation_options: CircuitCompilationOptions | None = None,
) -> tuple[dict[str, list[dict[str, int]]], dict[str, dict[str, Any]], float, float]:
    """Submit the circuits of all identifiers together, filling each job up to the batch limits.

    Circuits keep their order, so a job can hold circuits of several depths and qubit groups. A circuit with more
    gates than ``max_gates_per_batch`` is sent in a job of its own.

    Args:
        circuits_by_id: The measured qubits and the circuits of each identifier.
        backend: The backend to run on.
        shots: The number of shots per circuit.
        max_circuits_per_batch: The maximum number of circuits per job; ``None`` for no limit.
        max_gates_per_batch: The maximum number of gates per job; ``None`` for no limit.
        circuit_compilation_options: Compilation options passed to every job.

    Returns:
        The counts and the job metadata of each identifier, and the total submission and retrieval times.

    Raises:
        ValueError: If a job returns a different number of counts than it has circuits.

    """
    batches: list[list[tuple[str, QuantumCircuit]]] = [[]]
    num_gates = 0
    for identifier, (_, circuits) in circuits_by_id.items():
        for circuit in circuits:
            gates = sum(circuit.count_ops().values()) if max_gates_per_batch else 0
            too_many_circuits = max_circuits_per_batch is not None and len(batches[-1]) >= max_circuits_per_batch
            too_many_gates = max_gates_per_batch is not None and num_gates + gates > max_gates_per_batch
            if batches[-1] and (too_many_circuits or too_many_gates):
                batches.append([])
                num_gates = 0
            batches[-1].append((identifier, circuit))
            num_gates += gates
    batches = [batch for batch in batches if batch]

    total_submit = 0.0
    submitted_jobs = []
    for batch in batches:
        qubits = tuple(sorted({q for identifier, _ in batch for q in circuits_by_id[identifier][0]}))
        jobs, time_submit = submit_execute(
            {qubits: [circuit for _, circuit in batch]},
            backend,
            shots,
            circuit_compilation_options=circuit_compilation_options,
        )
        submitted_jobs.append(jobs)
        total_submit += time_submit
    qcvv_logger.info(f"Submitted {sum(len(batch) for batch in batches)} circuits in {len(batches)} jobs")

    counts: dict[str, list[dict[str, int]]] = {identifier: [] for identifier in circuits_by_id}
    job_metadata: dict[str, dict[str, Any]] = {identifier: {} for identifier in circuits_by_id}
    total_retrieve = 0.0
    for job_index, (batch, jobs) in enumerate(zip(batches, submitted_jobs, strict=True)):
        batch_counts, time_retrieve = retrieve_all_counts(jobs)
        total_retrieve += time_retrieve
        if len(batch_counts) != len(batch):
            raise ValueError(f"Job {job_index + 1} returned {len(batch_counts)} counts for {len(batch)} circuits.")
        metadata = retrieve_all_job_metadata(jobs)["batch_job_1"]
        for (identifier, _), circuit_counts in zip(batch, batch_counts, strict=True):
            counts[identifier].append(circuit_counts)
            job_metadata[identifier][f"batch_job_{job_index + 1}"] = metadata
    return counts, job_metadata, total_submit, total_retrieve


class DirectRandomizedBenchmarking(Benchmark):
    """Direct RB estimates the fidelity of layers of canonical gates."""

    analysis_function = staticmethod(direct_rb_analysis)

    @classmethod
    def name(cls) -> str:
        """Returns the name of the benchmark."""
        return "direct_rb"

    def __init__(self, backend_arg: IQMBackendBase | str, configuration: "DirectRBConfiguration"):
        """Construct the DirectRandomizedBenchmarking class.

        Args:
            backend_arg: The backend on which to define the benchmark,
                can be either a string or an IQMBackendBase object.
            configuration: The DirectRBConfiguration object containing the benchmark configuration parameters.

        """
        super().__init__(backend_arg, configuration)

        # EXPERIMENT
        self.qubits_array = configuration.qubits_array
        self.is_eplg = configuration.is_eplg

        # Override if EPLG is True but parallel_execution was set to False
        if self.is_eplg and not configuration.parallel_execution:
            configuration.parallel_execution = True

        self.parallel_execution = configuration.parallel_execution
        self.depths = configuration.depths
        self.num_circuit_samples = configuration.num_circuit_samples

        self.two_qubit_gate_ensembles = configuration.two_qubit_gate_ensembles
        self.densities_2q_gates = configuration.densities_2q_gates
        self.clifford_sqg_probabilities = configuration.clifford_sqg_probabilities
        self.sqg_gate_ensembles = configuration.sqg_gate_ensembles

        self.qiskit_optim_level = configuration.qiskit_optim_level

        self.session_timestamp = datetime.now(UTC).strftime(BENCHMARK_TIMESTAMP_FORMAT)
        self.execution_timestamp = ""

        # Initialize the variable to contain the circuits for each layout
        self.untranspiled_circuits = BenchmarkCircuit("untranspiled_circuits")
        self.transpiled_circuits = BenchmarkCircuit("transpiled_circuits")

    def add_all_meta_to_dataset(self, dataset: xr.Dataset) -> None:
        """Adds all configuration metadata and circuits to the dataset variable.

        Args:
            dataset: The xarray dataset

        """
        dataset.attrs["session_timestamp"] = self.session_timestamp
        dataset.attrs["execution_timestamp"] = self.execution_timestamp
        dataset.attrs["backend_name"] = self.backend_name

        for key, value in self.configuration:
            if key == "benchmark":  # Avoid saving the class object
                dataset.attrs[key] = value.name
            else:
                dataset.attrs[key] = value
        # Defined outside configuration - if any
        dataset.attrs["two_qubit_gate_ensembles"] = self.assigned_two_qubit_gate_ensembles
        dataset.attrs["densities_2q_gates"] = self.assigned_density_2q_gates
        dataset.attrs["clifford_sqg_probabilities"] = self.assigned_clifford_sqg_probabilities
        dataset.attrs["sqg_gate_ensembles"] = self.assigned_sqg_gate_ensembles

    def assign_inputs_to_qubits(
        self,
    ) -> tuple[
        Sequence[int],
        dict[str, dict[str, float]],
        dict[str, float],
        dict[str, float],
        dict[str, dict[str, float] | None],
    ]:
        """Assigns all DRB inputs to input qubit layouts.

        This method processes optional configuration parameters and assigns them to each qubit layout.
        If a parameter is not provided or has insufficient values for all qubit layouts, default values
        are assigned. The method handles different behaviors for EPLG (Edge-Grab Pauli Layers) mode.

        Returns:
            A tuple containing:
                - assigned_drb_depths: Sequence of DRB circuit depths.
                - assigned_two_qubit_gate_ensembles: Dictionary mapping qubit layout strings to
                  dictionaries of two-qubit gate names and their sampling probabilities.
                - assigned_density_2q_gates: Dictionary mapping qubit layout strings to the density
                  of two-qubit gates (probability of sampling 2Q gates per qubit per layer).
                - assigned_clifford_sqg_probabilities: Dictionary mapping qubit layout strings to
                  the probability of sampling Clifford single-qubit gates.
                - assigned_sqg_gate_ensembles: Dictionary mapping qubit layout strings to
                  dictionaries of single-qubit gate names and their sampling probabilities, or None
                  to indicate Clifford-only sampling.

        """
        # Depths - can be modified as in MRB to be qubit layout-dependent
        assigned_drb_depths = self.depths

        if isinstance(self.qubits_array[0][0], int):
            wrapped_qubits_array: Sequence[Sequence[int]] | Sequence[Sequence[Sequence[int]]] = self.qubits_array
            flat_all_qubits = [x for y in wrapped_qubits_array for x in y]
        else:
            flat_all_qubits = [x for y in self.qubits_array for x in y]

        # 2Q gate ensemble
        if self.two_qubit_gate_ensembles is None:
            # Assign native 2Qg with probability 1.0 - this is also default for EPLG
            assigned_two_qubit_gate_ensembles = {str(q): {"CZGate": 1.0} for q in flat_all_qubits}
        elif len(self.two_qubit_gate_ensembles) != len(flat_all_qubits):
            if len(self.two_qubit_gate_ensembles) != 1:
                qcvv_logger.warning(
                    f"The amount of 2Q gate ensembles ({len(self.two_qubit_gate_ensembles)}) is not the same "
                    f"as the total amount of qubit layout configurations ({len(flat_all_qubits)}):\n\t"
                    f"Will assign to all the first "
                    f"configuration: {self.two_qubit_gate_ensembles[0]} !"
                )
            assigned_two_qubit_gate_ensembles = {str(q): self.two_qubit_gate_ensembles[0] for q in flat_all_qubits}
        else:
            assigned_two_qubit_gate_ensembles = {
                str(q): self.two_qubit_gate_ensembles[q_idx] for q_idx, q in enumerate(flat_all_qubits)
            }

        # Density 2Q gates
        if self.densities_2q_gates is None and self.is_eplg:
            # For EPLG, with density 2Qg of 0.5, the edge_grab will sample 2Qg with probability 1.0
            assigned_density_2q_gates = {str(q): 0.5 for q in flat_all_qubits}
        elif self.densities_2q_gates is None:
            assigned_density_2q_gates = {str(q): 0.25 for q in flat_all_qubits}
        elif len(self.densities_2q_gates) != len(flat_all_qubits):
            if len(self.densities_2q_gates) != 1:
                qcvv_logger.warning(
                    f"The amount of 2Q gate densities ({len(self.densities_2q_gates)}) is not the same "
                    f"as the amount of all qubit layout configurations ({len(flat_all_qubits)}):\n\t"
                    f"Will assign to all the first "
                    f"configuration: {self.densities_2q_gates[0]} !"
                )
            assigned_density_2q_gates = {str(q): self.densities_2q_gates[0] for q in flat_all_qubits}
        else:
            assigned_density_2q_gates = {
                str(q): self.densities_2q_gates[q_idx] for q_idx, q in enumerate(flat_all_qubits)
            }

        # clifford_sqg_probabilities
        if self.clifford_sqg_probabilities is None and self.is_eplg:
            assigned_clifford_sqg_probabilities = {str(q): 0.0 for q in flat_all_qubits}
        elif self.clifford_sqg_probabilities is None:
            assigned_clifford_sqg_probabilities = {str(q): 1.0 for q in flat_all_qubits}
        elif len(self.clifford_sqg_probabilities) != len(flat_all_qubits):
            if len(self.clifford_sqg_probabilities) != 1:
                qcvv_logger.warning(
                    f"The amount of Clifford 1Q gate sampling probabilities "
                    f"({len(self.clifford_sqg_probabilities)}) is not the same "
                    f"as the amount of all qubit layout configurations ({len(flat_all_qubits)}):"
                    f"\n\tWill assign to all the first "
                    f"configuration: {self.clifford_sqg_probabilities[0]} !"
                )
            assigned_clifford_sqg_probabilities = {str(q): self.clifford_sqg_probabilities[0] for q in flat_all_qubits}
        else:
            assigned_clifford_sqg_probabilities = {
                str(q): self.clifford_sqg_probabilities[q_idx] for q_idx, q in enumerate(flat_all_qubits)
            }

        # sqg_gate_ensembles
        assigned_sqg_gate_ensembles: dict[str, dict[str, float] | None]
        if self.sqg_gate_ensembles is not None:
            if len(self.sqg_gate_ensembles) != len(flat_all_qubits):
                if len(self.sqg_gate_ensembles) != 1:
                    qcvv_logger.warning(
                        f"The amount of 1Q gate ensembles ({len(self.sqg_gate_ensembles)}) is not the same "
                        f"as the amount of all qubit layout configurations ({len(flat_all_qubits)}):\n"
                        f"\tWill assign to all the first configuration: {self.sqg_gate_ensembles[0]} !"
                    )
                assigned_sqg_gate_ensembles = {str(q): self.sqg_gate_ensembles[0] for q in flat_all_qubits}
            else:
                assigned_sqg_gate_ensembles = {
                    str(q): self.sqg_gate_ensembles[q_idx] for q_idx, q in enumerate(flat_all_qubits)
                }
        elif self.sqg_gate_ensembles is None and self.is_eplg:  # No Cliffords and no 1Q gates in Cycle Layers
            assigned_sqg_gate_ensembles = {str(q): {"IGate": 1.0} for q in flat_all_qubits}
        elif self.sqg_gate_ensembles is None and assigned_clifford_sqg_probabilities == {
            str(q): 1.0 for q in flat_all_qubits
        }:
            assigned_sqg_gate_ensembles = {str(q): None for q in flat_all_qubits}
            # None (together with condition of clifford sqg probabilities 1) implies that the edge grab algorithm
            # will only sample 1Q Clifford gates as 1Q gates when forming Cycle Layers
        else:
            # In this case, assign the rest to be either Cliffords or HGate with complementary probabilities
            # Choice of HGate is arbitrary, could be any other (Clifford, unless looking for some danger) 1Q gate
            assigned_sqg_gate_ensembles = {
                str(q): {"HGate": 1.0 - assigned_clifford_sqg_probabilities[str(q)]} for q in flat_all_qubits
            }

        # Reset the configuration values to store in dataset
        self.assigned_two_qubit_gate_ensembles = assigned_two_qubit_gate_ensembles
        self.assigned_density_2q_gates = assigned_density_2q_gates
        self.assigned_clifford_sqg_probabilities = assigned_clifford_sqg_probabilities
        self.assigned_sqg_gate_ensembles = assigned_sqg_gate_ensembles

        return (
            assigned_drb_depths,
            assigned_two_qubit_gate_ensembles,
            assigned_density_2q_gates,
            assigned_clifford_sqg_probabilities,
            assigned_sqg_gate_ensembles,
        )

    def execute(self, backend: IQMBackendBase) -> xr.Dataset:  # noqa: PLR0915
        """Executes the Direct Randomized Benchmarking benchmark.

        Args:
            backend: The IQM backend to execute the benchmark on

        Returns:
            Dataset containing benchmark results and metadata

        """
        self.execution_timestamp = datetime.now(UTC).strftime(BENCHMARK_TIMESTAMP_FORMAT)

        dataset = xr.Dataset()

        (
            assigned_drb_depths,
            assigned_two_qubit_gate_ensembles,
            assigned_density_2q_gates,
            assigned_clifford_sqg_probabilities,
            assigned_sqg_gate_ensembles,
        ) = self.assign_inputs_to_qubits()

        self.add_all_meta_to_dataset(dataset)

        clifford_1q_dict = import_native_gate_cliffords("1q")
        clifford_2q_dict = import_native_gate_cliffords("2q")

        # Circuits of all qubit layouts and depths, submitted together at the end to fill the jobs
        circuits_by_id: dict[str, tuple[Sequence[int], list[QuantumCircuit]]] = {}
        submitted: list[tuple[Any, int]] = []
        time_circuit_generation: dict[str, float] = {}

        # Auxiliary dict from str(qubits) to indices
        qubit_idx: dict[str, Any] = {}

        # Main execution
        if isinstance(self.qubits_array[0][0], int):
            # If the qubits_array is a single qubit layout, wrap it in a list
            # (so that the loop below proceeds at the right level)
            wrapped_qubits_array = [self.qubits_array]
        else:
            wrapped_qubits_array = cast(
                list[Sequence[Sequence[int]] | Sequence[Sequence[Sequence[int]]]],
                self.qubits_array,
            )

        for qubits_seq_idx, loop_qubits_sequence in enumerate(wrapped_qubits_array):
            if self.parallel_execution:
                # Take the whole loop_qubits_sequence and do DRB in parallel on each loop_qubits_sequence element
                parallel_drb_circuits = {}
                qcvv_logger.info(
                    f"Executing parallel Direct RB on qubits {loop_qubits_sequence} "
                    f"(group {qubits_seq_idx + 1}/{len(wrapped_qubits_array)})."
                    f" Will generate and submit all {self.num_circuit_samples} DRB circuits"
                    f" for each depth {self.depths}"
                )

                time_circuit_generation[str(loop_qubits_sequence)] = 0
                # Generate and submit all circuits
                for depth in self.depths:
                    qcvv_logger.info(f"Depth {depth}")
                    parallel_drb_circuits[depth], elapsed_time = generate_fixed_depth_parallel_drb_circuits(
                        qubits_array=loop_qubits_sequence,
                        depth=depth,
                        num_circuit_samples=self.num_circuit_samples,
                        backend_arg=backend,
                        assigned_density_2q_gates=assigned_density_2q_gates,
                        assigned_two_qubit_gate_ensembles=assigned_two_qubit_gate_ensembles,
                        assigned_clifford_sqg_probabilities=assigned_clifford_sqg_probabilities,
                        assigned_sqg_gate_ensembles=assigned_sqg_gate_ensembles,
                        cliffords_1q=clifford_1q_dict,
                        cliffords_2q=clifford_2q_dict,
                        qiskit_optim_level=self.qiskit_optim_level,
                        routing_method=self.routing_method,
                        is_eplg=self.is_eplg,
                    )
                    time_circuit_generation[str(loop_qubits_sequence)] += elapsed_time

                    flat_qubits_array = cast(list[int], [x for y in loop_qubits_sequence for x in y])
                    circuits_by_id[f"qubits_{str(loop_qubits_sequence)}_depth_{str(depth)}"] = (
                        flat_qubits_array,
                        cast(list[QuantumCircuit], parallel_drb_circuits[depth]["transpiled"]),
                    )
                    submitted.append((loop_qubits_sequence, depth))

                    self.untranspiled_circuits.circuit_groups.append(
                        CircuitGroup(
                            name=f"{str(loop_qubits_sequence)}_depth_{depth}",
                            circuits=parallel_drb_circuits[depth]["untranspiled"],
                        )
                    )
                    self.transpiled_circuits.circuit_groups.append(
                        CircuitGroup(
                            name=f"{str(loop_qubits_sequence)}_depth_{depth}",
                            circuits=parallel_drb_circuits[depth]["transpiled"],
                        )
                    )
                qubit_idx.update({str(loop_qubits_sequence): qubits_seq_idx})
                dataset.attrs[f"parallel_all_{qubits_seq_idx}"] = {"qubits": loop_qubits_sequence}
                dataset.attrs.update(
                    {qubits_seq_idx: {q_idx: {"qubits": q} for q_idx, q in enumerate(loop_qubits_sequence)}}
                )
            else:  # if sequential
                for qubits_idx, qubits in enumerate(loop_qubits_sequence):
                    qubit_idx[str(qubits)] = qubits_idx

                    qcvv_logger.info(
                        f"Executing DRB on qubits {qubits}."
                        f" Will generate and submit all {self.num_circuit_samples} DRB circuits"
                        f" for depths {assigned_drb_depths}"
                    )
                    drb_circuits = {}
                    drb_transpiled_circuits_lists: dict[int, list[QuantumCircuit]] = {}
                    drb_untranspiled_circuits_lists: dict[int, list[QuantumCircuit]] = {}
                    time_circuit_generation[str(qubits)] = 0
                    for depth in assigned_drb_depths:
                        qcvv_logger.info(f"Depth {depth} - Generating all circuits")
                        drb_circuits[depth], elapsed_time = generate_drb_circuits(
                            qubits,
                            depth=depth,
                            circ_samples=self.num_circuit_samples,
                            backend_arg=backend,
                            density_2q_gates=assigned_density_2q_gates[str(qubits)],
                            two_qubit_gate_ensemble=assigned_two_qubit_gate_ensembles[str(qubits)],
                            clifford_sqg_probability=assigned_clifford_sqg_probabilities[str(qubits)],
                            sqg_gate_ensemble=assigned_sqg_gate_ensembles[str(qubits)],
                            qiskit_optim_level=self.qiskit_optim_level,
                            routing_method=self.routing_method,
                        )
                        time_circuit_generation[str(qubits)] += elapsed_time

                        # Generated circuits at fixed depth are (dict) indexed by Pauli sample number, turn into List
                        drb_transpiled_circuits_lists[depth] = drb_circuits[depth]["transpiled"]
                        drb_untranspiled_circuits_lists[depth] = drb_circuits[depth]["untranspiled"]

                        circuits_by_id[f"qubits_{str(qubits)}_depth_{str(depth)}"] = (
                            cast(Sequence[int], qubits),
                            drb_transpiled_circuits_lists[depth],
                        )
                        submitted.append((qubits, depth))

                        self.untranspiled_circuits.circuit_groups.append(
                            CircuitGroup(
                                name=f"{str(qubits)}_depth_{depth}",
                                circuits=drb_untranspiled_circuits_lists[depth],
                            )
                        )
                        self.transpiled_circuits.circuit_groups.append(
                            CircuitGroup(
                                name=f"{str(qubits)}_depth_{depth}",
                                circuits=drb_transpiled_circuits_lists[depth],
                            )
                        )

                    dataset.attrs[f"{qubits_seq_idx}_{qubits_idx}"] = {"qubits": qubits}

        counts, job_metadata, total_submit, total_retrieve = submit_in_batches(
            circuits_by_id,
            backend,
            self.shots,
            max_circuits_per_batch=self.configuration.max_circuits_per_batch,
            max_gates_per_batch=self.max_gates_per_batch,
            circuit_compilation_options=self.circuit_compilation_options,
        )

        for qubits, depth in submitted:
            identifier = f"qubits_{str(qubits)}_depth_{str(depth)}"
            dataset.attrs[qubit_idx[str(qubits)]].update(
                {
                    f"depth_{str(depth)}": {
                        "time_circuit_generation": time_circuit_generation[str(qubits)],
                        "time_submit": total_submit,
                        "time_retrieve": total_retrieve,
                        "all_job_metadata": job_metadata[identifier],
                    },
                }
            )
            qcvv_logger.info(f"Adding counts of qubits {qubits} and depth {depth} run to the dataset")
            dataset, _ = add_counts_to_dataset(counts[identifier], identifier, dataset)

        self.circuits = Circuits([self.transpiled_circuits, self.untranspiled_circuits])
        dataset.attrs["total_submit_time"] = total_submit
        dataset.attrs["total_retrieve_time"] = total_retrieve
        qcvv_logger.info("DRB experiment execution concluded!")

        return dataset


class DirectRBConfiguration(BenchmarkConfigurationBase):
    """Direct RB configuration.

    Attributes:
        benchmark: DirectRandomizedBenchmarking.
        qubits_array:
            The array of physical qubits in which to execute DRB.
            * It can be specified as a Sequence (e.g. list or tuple) of qubit-index registers, e.g., [[0, 1], [2, 3]],
            or as Sequences of such Sequences, e.g., [[[0, 1], [2, 3]], [[0, 2], [1, 3]]].
            In the second case, each Sequence[Sequence[int]] will execute sequentially, i.e.,
            execution will be done for [[0, 1], [2, 3]] first, then for [[0, 2], [1, 3]],
            each either in parallel or sequence, according to the (bool) value of parallel_execution.
        is_eplg: Whether the DRB experiment is executed as a EPLG subroutine.
            * If True:
            - default parallel_execution below is override to True.
            - default two_qubit_gate_ensembles is {"CZGate": 1.0}.
            - default densities_2q_gates is 0.5 (probability of sampling 2Q gates is 1).
            - default clifford_sqg_probabilities is 0.0.
            - default sqg_gate_ensembles is {"IGate": 1.0}.
            * Default is False.
        parallel_execution: Whether DRB is executed in parallel for all qubit layouts in qubits_array.
            * If is_eplg is False, it executes parallel DRB with MRB gate ensemble and density defaults.
        depths: The list of layer depths in which to execute DRB for all qubit layouts in qubits_array.
        num_circuit_samples: The number of random-layer DRB circuits to generate.
        shots: The number of measurement shots to execute per circuit.
        qiskit_optim_level: The Qiskit-level of optimization to use in transpilation.
        routing_method: The routing method to use in transpilation.
        two_qubit_gate_ensembles: The two-qubit gate ensembles to use in the
            random DRB circuits.
            * Keys correspond to str names of qiskit circuit library gates, e.g., "CZGate" or "CXGate".
            * Values correspond to the probability for the respective gate to be sampled.
            * Each Dict[str,float] corresponds to each qubit layout in qubits_array.
            * If len(two_qubit_gate_ensembles) != len(qubits_array), the first Dict is assigned by default.
            * Default is None, which assigns {str(q): {"CZGate": 1.0} for q in qubits_array}.
        densities_2q_gates: The expected densities of 2-qubit gates in the final circuits
            per qubit layout.
            * If len(densities_2q_gates) != len(qubits_array), the first density value is assigned by default.
            * Default is None, which assigns 0.25 to all qubit layouts.
        clifford_sqg_probabilities: Probability with which to uniformly sample
            Clifford 1Q gates per qubit layout.
            * Default is None, which assigns 1.0 to all qubit layouts.
        sqg_gate_ensembles: A dictionary with keys being str specifying 1Q gates,
            and values being corresponding probabilities.
            * If len(sqg_gate_ensembles) != len(qubits_array), the first ensemble is assigned by default.
            * Default is None, which leaves only uniform sampling of 1Q Clifford gates.

    """

    benchmark: type[Benchmark] = DirectRandomizedBenchmarking
    qubits_array: Sequence[Sequence[int]] | Sequence[Sequence[Sequence[int]]]
    is_eplg: bool = False
    parallel_execution: bool = False
    depths: Sequence[int]
    num_circuit_samples: int
    qiskit_optim_level: int = 1
    two_qubit_gate_ensembles: Sequence[dict[str, float]] | None = None
    densities_2q_gates: Sequence[float] | None = None
    clifford_sqg_probabilities: Sequence[float] | None = None
    sqg_gate_ensembles: Sequence[dict[str, float]] | None = None
    routing_method: RoutingMethod = RoutingMethod.SABRE
