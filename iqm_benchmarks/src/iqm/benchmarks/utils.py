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

"""General utility functions."""

from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from enum import Enum, StrEnum, auto
from functools import wraps
import itertools
from math import floor
import random
import re
import secrets
import time
from typing import Any, cast
import warnings

from iqm.benchmarks.logging_config import qcvv_logger
from iqm.iqm_client import IQMClient
from iqm.iqm_client.models import CircuitCompilationOptions
from iqm.qiskit_iqm import IQMCircuit as QuantumCircuit, IQMFakeDeneb, optimize_single_qubit_gates, transpile_to_IQM
from iqm.qiskit_iqm.fake_backends.fake_adonis import IQMFakeAdonis
from iqm.qiskit_iqm.fake_backends.fake_apollo import IQMFakeApollo
from iqm.qiskit_iqm.iqm_backend import IQMBackendBase
from iqm.qiskit_iqm.iqm_job import IQMJob
from iqm.qiskit_iqm.iqm_provider import IQMProvider
from more_itertools import chunked
from mthree.utils import final_measurement_mapping  # type: ignore
import networkx as nx
import numpy as np
from numpy.random import Generator
from qiskit import ClassicalRegister, transpile
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import Pauli
from qiskit.transpiler import CouplingMap
import xarray as xr


class RoutingMethod(StrEnum):
    """Qiskit routing method options to use in transpilation."""

    BASIC = auto()
    LOOKAHEAD = auto()
    NONE = auto()
    SABRE = auto()
    STOCHASTIC = auto()


class PhysicalLayout(StrEnum):
    """Physical layout type."""

    FIXED = auto()
    BATCHING = auto()


def timeit(f: Callable) -> Callable:
    """Decorator for calculating the amount of time a function takes to execute.

    The decorated function returns a tuple where the first item is the original output,
    and the second item the execution time in seconds.
    Additionally, it will log the elapsed execution time on the DEBUG log level if
    it exceeds 1 second.

    Args:
        f: The function whose execution we wish to time.

    Returns:
        Decorated function ``f``.

    """

    @wraps(f)
    def wrap(*args: Any, **kw: Any) -> tuple[Any, float]:
        ts = time.monotonic()
        result = f(*args, **kw)
        te = time.monotonic()
        elapsed = te - ts
        minute_cutoff = 60.0
        if elapsed < 1.0:
            pass
        elif elapsed <= minute_cutoff:
            qcvv_logger.debug(f'\t"{f.__name__}" took {elapsed:.2f} sec')
        else:
            qcvv_logger.debug(f'\t"{f.__name__}" took {elapsed / 60.0:.2f} min')
        return result, elapsed

    return wrap


def process_backend(backend: str | IQMBackendBase) -> tuple[IQMBackendBase, str]:
    """Process the backend input to ensure it is an IQMBackendBase instance and extract the station name.

    Args:
        backend: The backend to process, either as a string name or an IQMBackendBase

    Returns:
        A tuple containing the processed backend and the station name.

    """
    if isinstance(backend, str):
        backend = get_iqm_backend(backend)
    if hasattr(backend, "error_profile") and backend.error_profile is not None:
        station_name = backend.error_profile.name
    elif hasattr(backend, "client"):
        station_name = backend.client._iqm_server_client._quantum_computer
    else:
        station_name = "Unknown Station"
    return backend, station_name


def bootstrap_counts(
    original_counts: dict[str, int],
    num_bootstrap_samples: int = 100,
    rgen: Generator | None = None,
    include_original_counts: bool = False,
) -> list[dict[str, int]]:
    """Returns num_bootstrap_samples resampled copies of the original_counts.

    Args:
        original_counts: The original counts to bootstrap from.
        num_bootstrap_samples: The number of bootstrapping samples to generate.
        rgen: Random number generator to use. If None, use Numpy's :func:`default_rng`.
        include_original_counts:
            Whether to include the original counts in the returned bootstrapped count samples.

    Returns:
        Bootstrapped counts.

    """
    if rgen is None:
        rgen = np.random.default_rng()

    keys = list(original_counts.keys())
    values = list(original_counts.values())
    tot_shots = int(sum(values))

    # Pre-calculate cumulative sum and create bins
    cumulative_sum = np.cumsum(values)
    bins = np.insert(cumulative_sum, 0, 0)

    if include_original_counts:
        bs_counts_fast = [original_counts]
    else:
        bs_counts_fast = []

    for _ in range(num_bootstrap_samples):
        # Generate random integers
        random_integers = rgen.integers(low=0, high=tot_shots, size=tot_shots)
        # Bin the random integers
        binned_integers = np.digitize(random_integers, bins) - 1
        # Count occurrences in each bin
        occurrences = np.bincount(binned_integers, minlength=len(keys))
        # Create dictionary mapping keys to occurrence counts
        bs_counts_fast.append(dict(zip(keys, occurrences, strict=True)))

    return bs_counts_fast


@timeit
def count_2q_layers(circuit_list: list[QuantumCircuit]) -> list[int]:
    """Calculate the number of layers of parallel 2-qubit gates in a list of circuits.

    Args:
        circuit_list: the list of quantum circuits to analyze.

    Returns:
        The number of layers of parallel 2-qubit gates in the list of circuits.

    """
    all_number_2q_layers = []
    for circuit in circuit_list:
        dag = circuit_to_dag(circuit)
        layers = list(dag.layers())  # Call the method and convert the result to a list
        parallel_2q_layers = 0

        for layer in layers:
            two_qubit_gates_in_layer = [
                node
                for node in layer["graph"].op_nodes()  # Use op_nodes to get only operation nodes
                if node.op.num_qubits == 2  # noqa: PLR2004
            ]
            if two_qubit_gates_in_layer:
                parallel_2q_layers += 1
        all_number_2q_layers.append(parallel_2q_layers)

    return all_number_2q_layers


def count_native_gates(
    backend_arg: str | IQMBackendBase, transpiled_qc_list: list[QuantumCircuit]
) -> dict[str, dict[str, float]]:
    """Count the number of IQM native gates of each quantum circuit in a list.

    Args:
        backend_arg: The backend, either specified as str or as IQMBackendBase.
        transpiled_qc_list: a list of quantum circuits transpiled to ['r','cz','barrier','measure'] gate set.

    Returns:
        Dictionary with
             - outermost keys being native operations.
             - values being Dict[str, float] with mean and standard deviation values of native operation counts.

    """
    if isinstance(backend_arg, str):
        backend = get_iqm_backend(backend_arg)
    else:
        backend = backend_arg

    native_operations = backend.operation_names

    if backend.has_resonators():
        native_operations.append("move")
    # Some backends may not include "barrier" in the operation_names attribute
    if "barrier" not in native_operations:
        native_operations.append("barrier")

    num_native_operations: dict[str, list[int]] = {x: [0] for x in native_operations}
    avg_native_operations: dict[str, dict[str, float]] = {x: {} for x in native_operations}

    for q in transpiled_qc_list:
        for k in q.count_ops().keys():
            if k not in native_operations:
                raise ValueError(f"Count # of gates: '{k}' is not in the backend's native gate set")
        for op in native_operations:
            if op in q.count_ops().keys():
                num_native_operations[op].append(q.count_ops()[op])

    avg_native_operations.update(
        {
            x: {
                "Mean": np.mean(num_native_operations[x]),
                "Std": np.std(num_native_operations[x]),
            }
            for x in native_operations
        }
    )

    return avg_native_operations


@timeit
def generate_state_tomography_circuits(
    qc: QuantumCircuit,
    active_qubits: Sequence[int],
    measure_other: Sequence[int] | None = None,
    measure_other_name: str | None = None,
    native: bool = True,
) -> dict[str, QuantumCircuit]:
    """Generate all quantum circuits required for a quantum state tomography experiment.

    Args:
        qc: The quantum circuit.
        active_qubits: The qubits to perform tomograhy on.
        measure_other: Whether to measure other qubits in the qc QuantumCircuit.
        measure_other_name: Name of the classical register to assign measure_other.
        native: Whether circuits are prepared using IQM-native gates.

    Returns:
        A dictionary with keys being Pauli (measurement) strings and values the respective circuit.
        * Pauli strings are ordered for qubit labels in increasing order, e.g.,
        "XY" for active_qubits 4, 1 corresponds to "X" measurement on qubit 1 and "Y" measurement on qubit 4.

    """
    num_qubits = len(active_qubits)

    # Organize all Pauli measurements as circuits
    aux_circ = QuantumCircuit(1)
    sqg_pauli_strings = ("Z", "X", "Y")
    pauli_measurements = {p: aux_circ.copy() for p in sqg_pauli_strings}

    # Avoid transpilation, generate either directly in native basis or in H, S
    if native:
        # Z measurement
        pauli_measurements["Z"].r(0, 0, 0)
        # X measurement
        pauli_measurements["X"].r(np.pi / 2, np.pi / 2, 0)
        pauli_measurements["X"].r(np.pi, 0, 0)
        # Y measurement
        pauli_measurements["Y"].r(-np.pi / 2, 0, 0)
        pauli_measurements["Y"].r(np.pi, np.pi / 4, 0)
    else:
        # Z measurement
        pauli_measurements["Z"].id(0)
        # X measurement
        pauli_measurements["X"].h(0)
        # Y measurement
        pauli_measurements["Y"].sdg(0)
        pauli_measurements["Y"].h(0)

    all_pauli_labels = ["".join(x) for x in itertools.product(sqg_pauli_strings, repeat=num_qubits)]
    all_circuits = {p_n: qc.copy() for p_n in all_pauli_labels}
    for p_n in all_pauli_labels:
        all_circuits[p_n].barrier()
        for q_idx, q_active in enumerate(sorted(active_qubits)):
            all_circuits[p_n].compose(pauli_measurements[p_n[q_idx]], qubits=q_active, inplace=True)

        all_circuits[p_n].barrier()

        register_tomo = ClassicalRegister(len(active_qubits), "tomo_qubits")
        all_circuits[p_n].add_register(register_tomo)
        all_circuits[p_n].measure(active_qubits, register_tomo)

        if measure_other is not None:
            if measure_other_name is None:
                measure_other_name = "non_tomo_qubits"
            register_neighbors = ClassicalRegister(len(measure_other), measure_other_name)
            all_circuits[p_n].add_register(register_neighbors)
            all_circuits[p_n].measure(measure_other, register_neighbors)

    return all_circuits


def get_active_qubits(qc: QuantumCircuit) -> list[int]:
    """Extract active qubits from a quantum circuit.

    Args:
        qc: The quantum circuit to extract active qubits from.

    Returns:
        A list of active qubits.

    """
    active_qubits = set()
    for instruction in qc.data:
        for qubit in instruction.qubits:
            active_qubits.add(qc.find_bit(qubit).index)
    return list(active_qubits)


def _remap_metrics_dict(
    metrics_dict: dict[str, dict[int | tuple[int, int], float]], qubit_mapping: dict[int, int]
) -> dict[str, dict[int | tuple[int, int], float]]:
    """Remap qubit indices in metrics dictionary according to qubit mapping.

    Args:
        metrics_dict: Dictionary of metrics with qubit indices as keys
        qubit_mapping: Mapping from original qubit indices to new indices

    Returns:
        Remapped metrics dictionary with updated qubit indices

    """
    remapped_metrics_dict: dict[str, dict[int | tuple[int, int], float]] = {}
    for metric_key, metric_values in metrics_dict.items():
        remapped_metrics_dict[metric_key] = {}
        for key, value in metric_values.items():
            if isinstance(key, tuple):
                # Two-qubit metric
                remapped_metrics_dict[metric_key][(qubit_mapping[key[0]], qubit_mapping[key[1]])] = value
            else:
                # Single-qubit metric
                remapped_metrics_dict[metric_key][qubit_mapping[key]] = value
    return remapped_metrics_dict


def _relaxation_fidelity(duration: float, t1: float, t2: float) -> float:
    """Compute the approx. gate fidelity from noise channel with T1 and T2 errors.

    Args:
        duration: Gate or idle time in (us)
        t1: T1 time in (us)
        t2: T2 time in (us)

    Returns:
        Fidelity of the idling gate.

    """
    rate1 = 1 / t1
    rate2 = 1 / t2
    p_decay = np.exp(-duration * rate1)
    p_dephasing = np.exp(-duration * rate2)
    relaxation_fid = (3 + p_decay + 2 * p_dephasing) / 6

    return relaxation_fid


def error_profile_to_fidelities(
    backend: IQMBackendBase,
) -> tuple[dict[Any, int], dict[str, dict[int | tuple[int, int], float]]]:
    """Convert IQMFakeBackend error profile to fidelity metrics.

    Args:
        backend: The fake backend with error_profile attribute.

    Returns:
        Dictionary of metrics with the same structure as extract_fidelities_unified.

    """
    if not hasattr(backend, "error_profile") or backend.error_profile is None:
        raise ValueError("Backend does not have an error_profile attribute")

    error_profile = backend.error_profile

    cz_fidelity: dict[int | tuple[int, int], float] = {}
    single_qubit_fidelity: dict[int | tuple[int, int], float] = {}
    readout_fidelity: dict[int | tuple[int, int], float] = {}
    t1: dict[int | tuple[int, int], float] = {}
    t2: dict[int | tuple[int, int], float] = {}
    move_fidelity: dict[int | tuple[int, int], float] = {}

    qubits = [q for q in error_profile.t1s.keys() if q.startswith("QB")]

    # Extract T1 and T2 times
    t1_times = error_profile.t1s if hasattr(error_profile, "t1s") else {}
    t2_times = error_profile.t2s if hasattr(error_profile, "t2s") else {}

    # Store T1 and T2 values
    for qubit_name in qubits:
        qubit_index = backend._qb_to_idx[qubit_name]
        if qubit_name in t1_times:
            t1[qubit_index] = t1_times[qubit_name] / 10**3  # Convert to microseconds
        if qubit_name in t2_times:
            t2[qubit_index] = t2_times[qubit_name] / 10**3  # Convert to microseconds

    # Readout fidelities
    for qubit_name, readout_error in error_profile.readout_errors.items():
        qubit_index = backend._qb_to_idx[qubit_name]
        readout_fidelity[qubit_index] = 1 - (readout_error["0"] + readout_error["1"]) / 2

    # Single-qubit gate fidelities
    if (
        hasattr(error_profile, "single_qubit_gate_depolarizing_error_parameters")
        and "prx" in error_profile.single_qubit_gate_depolarizing_error_parameters
    ):
        duration = error_profile.single_qubit_gate_durations.get("prx", 0) / 10**3  # Convert to microseconds
        qubit_params = error_profile.single_qubit_gate_depolarizing_error_parameters["prx"]
        for qubit_name, depol_error in qubit_params.items():
            qubit_index = backend._qb_to_idx[qubit_name]
            relaxation_fidelity = _relaxation_fidelity(
                duration, t1_times.get(qubit_name, float("inf")), t2_times.get(qubit_name, float("inf"))
            )
            single_qubit_fidelity[qubit_index] = (1 - depol_error / 3) * relaxation_fidelity
    else:
        for qubit_name in qubits:
            qubit_index = backend._qb_to_idx[qubit_name]
            single_qubit_fidelity[qubit_index] = 1
        warnings.warn(
            "PRX error not found in simulator backend error profile. "
            "Single qubit gate fidelity set to 1 for all qubits.",
            stacklevel=2,
        )

    # Two-qubit gate fidelities
    if (
        hasattr(error_profile, "two_qubit_gate_depolarizing_error_parameters")
        and "cz" in error_profile.two_qubit_gate_depolarizing_error_parameters
    ):
        duration = error_profile.two_qubit_gate_durations.get("cz", 0) / 10**3  # Convert to microseconds
        qubit_params = error_profile.two_qubit_gate_depolarizing_error_parameters["cz"]
        for qubit_names, depol_error in qubit_params.items():
            qubit_indices = (backend._qb_to_idx[qubit_names[0]], backend._qb_to_idx[qubit_names[1]])
            relaxation_fidelity_1 = _relaxation_fidelity(
                duration, t1_times.get(qubit_names[0], float("inf")), t2_times.get(qubit_names[0], float("inf"))
            )
            relaxation_fidelity_2 = _relaxation_fidelity(
                duration, t1_times.get(qubit_names[1], float("inf")), t2_times.get(qubit_names[1], float("inf"))
            )
            cz_fidelity[qubit_indices] = (1 - depol_error * 3 / 5) * relaxation_fidelity_1 * relaxation_fidelity_2
    else:
        for qubit_indices in list(set(backend.coupling_map)):
            cz_fidelity[qubit_indices] = 1
        warnings.warn(
            "CZ error not found in simulator backend error profile. Two-qubit gate fidelity set to 1 for all qubits.",
            stacklevel=2,
        )

    # MOVE gate fidelities for star topologies
    if (
        hasattr(error_profile, "two_qubit_gate_depolarizing_error_parameters")
        and "move" in error_profile.two_qubit_gate_depolarizing_error_parameters
    ):
        duration = error_profile.two_qubit_gate_durations.get("move", 0) / 10**3  # Convert to microseconds
        qubit_params = error_profile.two_qubit_gate_depolarizing_error_parameters["move"]
        for qubit_names, depol_error in qubit_params.items():
            qubit_indices = (backend._qb_to_idx[qubit_names[0]], backend._qb_to_idx[qubit_names[1]])
            relaxation_fidelity_1 = _relaxation_fidelity(
                duration, t1_times.get(qubit_names[0], float("inf")), t2_times.get(qubit_names[0], float("inf"))
            )
            relaxation_fidelity_2 = _relaxation_fidelity(
                duration, t1_times.get(qubit_names[1], float("inf")), t2_times.get(qubit_names[1], float("inf"))
            )
            move_fidelity[qubit_indices] = (1 - depol_error * 3 / 5) * relaxation_fidelity_1 * relaxation_fidelity_2

    metrics_dict: dict[str, dict[int | tuple[int, int], float]] = {
        "cz_gate_fidelity": cz_fidelity,
        "fidelity_1qb_gates_averaged": single_qubit_fidelity,
        "single_shot_readout_fidelity": readout_fidelity,
        "t1_time": t1,
        "t2_time": t2,
        "double_move_gate_fidelity": move_fidelity,
    }

    # Enumerate all calibrated qubits starting from 0
    list_couplings = list(cz_fidelity.keys())
    calibrated_qubits = set(np.array(list_couplings).reshape(-1))
    qubit_mapping = {qubit: idx for idx, qubit in enumerate(calibrated_qubits)}
    # Apply the qubit mapping to metrics_dict
    metrics_dict = _remap_metrics_dict(metrics_dict, qubit_mapping)

    return qubit_mapping, metrics_dict


class ObservationType(Enum):
    """Enumeration representing relevant keys to fetch for each operation in the observations."""

    CZ = "cz"
    CLIFFORD = ("cz", "clifford")
    SQG = "prx"
    READOUT = ("measure_fidelity", ".fidelity")
    READOUT_QNDNESS = ("measure", "qndness", ".fidelity")
    DOUBLE_MOVE = "move"
    T1 = ("t1", "QB")
    T2 = ("t2", "QB")


def extract_fidelities_unified(
    backend: IQMBackendBase,
) -> tuple[dict[Any, int], dict[str, dict[int | tuple[int, int], float]]]:
    # ruff: noqa: PLR0915
    """Returns couplings and CZ-fidelities from calibration data URL for external station API.

    Args:
        backend: The backend to extract the fidelities for.

    Returns:
        A mapping from physical qubits to enumerated qubits starting from 0, for all qubits
            appearing in the calibration data.
        Dictionary of all metrics (returned only if all_metrics=True)
            Format: {metric_name: {qubit: value}} for single qubit metrics
            Format: {metric_name: {(qubit_1, qubit_2): value}} for two qubit metrics

    """
    # Create dictionaries to map key names to their corresponding metrics
    cz_fidelity: dict[int | tuple[int, int], float] = {}
    single_qubit_fidelity: dict[int | tuple[int, int], float] = {}
    readout_fidelity: dict[int | tuple[int, int], float] = {}
    t1: dict[int | tuple[int, int], float] = {}
    t2: dict[int | tuple[int, int], float] = {}
    move_fidelity: dict[int | tuple[int, int], float] = {}
    clifford_fidelity: dict[int | tuple[int, int], float] = {}

    # Check if this is a fake backend with error_profile
    if hasattr(backend, "error_profile") and backend.error_profile is not None:
        return error_profile_to_fidelities(backend)

    # Real backend - use the client to get quality metrics
    iqm_server_url = backend.client._iqm_server_client.root_url
    quantum_computer = backend.client._iqm_server_client._quantum_computer
    calibration_metrics = (
        IQMClient(iqm_server_url, quantum_computer=quantum_computer).get_quality_metric_set().observations
    )

    gates_info: dict[str, dict[str, Any]] = {}
    for gate in backend.architecture.gates.keys():
        gates_info[gate] = {
            x: backend.architecture.gates[gate].implementations[x].loci
            for x in backend.architecture.gates[gate].implementations.keys()
        }

        # Iterate over the calibration metrics
    for metrics in calibration_metrics:
        dut_field = metrics.dut_field
        value = metrics.value

        if not isinstance(value, (int, float)):
            raise TypeError("Calibration metrics value should be float.")

        if all(obs in dut_field for obs in ObservationType.READOUT.value):
            qubit_index = int(dut_field.split("QB")[1].split(".")[0])
            readout_fidelity[qubit_index] = value
        elif ObservationType.SQG.value in dut_field and any(
            x in dut_field for x in gates_info[ObservationType.SQG.value]
        ):
            qubit_index = int(dut_field.split("QB")[1].split(".")[0])
            single_qubit_fidelity[qubit_index] = value
        elif all(obs in dut_field for obs in ObservationType.T1.value):
            qubit_index = int(dut_field.split("QB")[1].split(".")[0])
            t1[qubit_index] = value * 10**6
        elif all(obs in dut_field for obs in ObservationType.T2.value):
            qubit_index = int(dut_field.split("QB")[1].split(".")[0])
            t2[qubit_index] = value * 10**6
        elif ObservationType.DOUBLE_MOVE.value in dut_field and any(
            x in dut_field for x in gates_info[ObservationType.DOUBLE_MOVE.value]
        ):
            qb_matches = re.findall(r"QB\d+", dut_field)
            qbx = int(qb_matches[0].split("QB")[1])
            move_fidelity[(qbx, 0)] = value
            move_fidelity[(0, qbx)] = value
        elif (
            ObservationType.CZ.value in dut_field
            and any(x in dut_field for x in gates_info[ObservationType.CZ.value])
            and backend.has_resonators()
        ):
            qb_matches = re.findall(r"QB\d+", dut_field)
            qbx = int(qb_matches[0].split("QB")[1])
            cz_fidelity[(qbx, 0)] = value
            cz_fidelity[(0, qbx)] = value
        elif ObservationType.CZ.value in dut_field and any(
            x in dut_field for x in gates_info[ObservationType.CZ.value]
        ):
            qb_matches = re.findall(r"QB\d+", dut_field)
            qbx, qby = int(qb_matches[0].split("QB")[1]), int(qb_matches[1].split("QB")[1])
            cz_fidelity[(qbx, qby)] = value
            cz_fidelity[(qby, qbx)] = value
        elif all(obs in dut_field for obs in ObservationType.CLIFFORD.value) and backend.has_resonators():
            qb_matches = re.findall(r"QB\d+", dut_field)
            qbx = qb_matches[0]
            qb_matches = re.findall(r"QB\d+", dut_field)
            qbx = int(qb_matches[0].split("QB")[1])
            clifford_fidelity[(qbx, 0)] = value
            clifford_fidelity[(0, qbx)] = value
        elif all(obs in dut_field for obs in ObservationType.CLIFFORD.value) and not backend.has_resonators():
            qb_matches = re.findall(r"QB\d+", dut_field)
            qbx, qby = int(qb_matches[0].split("QB")[1]), int(qb_matches[1].split("QB")[1])
            clifford_fidelity[(qbx, qby)] = value
            clifford_fidelity[(qby, qbx)] = value

    metrics_dict: dict[str, dict[int | tuple[int, int], float]] = {
        "cz_gate_fidelity": cz_fidelity,
        "clifford_fidelity": clifford_fidelity,
        "fidelity_1qb_gates_averaged": single_qubit_fidelity,
        "single_shot_readout_fidelity": readout_fidelity,
        "t1_time": t1,
        "t2_time": t2,
        "double_move_gate_fidelity": move_fidelity,
    }
    # Enumerate all calibrated qubits starting from 0
    list_couplings = list(cz_fidelity.keys())
    calibrated_qubits = set(np.array(list_couplings).reshape(-1))
    qubit_mapping = {qubit: idx for idx, qubit in enumerate(calibrated_qubits)}
    # Apply the qubit mapping to metrics_dict
    metrics_dict = _remap_metrics_dict(metrics_dict, qubit_mapping)

    return qubit_mapping, metrics_dict


def get_iqm_backend(backend_label: str) -> IQMBackendBase:
    """Get the IQM backend object from a backend name (str).

    Args:
        backend_label: The name of the IQM backend.

    Returns:
        IQMBackendBase object.

    """
    iqm_server_url = "https://resonance.meetiqm.com"

    # Known systems
    if backend_label.lower() in ["garnet", "emerald", "sirius"]:
        provider = IQMProvider(iqm_server_url, quantum_computer=backend_label.lower())
        backend_object = provider.get_backend()

    # FakeAdonis
    elif backend_label.lower() in ("iqmfakeadonis", "fakeadonis"):
        backend_object = IQMFakeAdonis()
    # FakeApollo
    elif backend_label.lower() in ("iqmfakeapollo", "fakeapollo"):
        backend_object = IQMFakeApollo()
    # FakeDeneb
    elif backend_label.lower() in ("iqmfakedeneb", "fakedeneb"):
        backend_object = IQMFakeDeneb()
    else:
        raise ValueError(
            f"Backend {backend_label} not supported. Try 'garnet', 'emerald', 'deneb', 'sirius', "
            f"'fakeadonis', 'fakeapollo', 'fakedeneb'."
        )

    return backend_object


def get_measurement_mapping(circuit: QuantumCircuit) -> dict[int, int]:
    """Extracts the final measurement mapping (qubits to bits) of a quantum circuit.

    Args:
        circuit :
            The quantum circuit to extract the measurement mapping from.

    Returns:
            A dictionary where keys are qubits and values are classical bits.

    """
    mapping = {}
    for instruction, qargs, cargs in circuit.data:
        if instruction.name == "measure":
            qubit = circuit.find_bit(qargs[0]).registers[0][1]
            cbit = circuit.find_bit(cargs[0]).registers[0][1]
            mapping[qubit] = cbit
    return mapping


def get_neighbors_of_edges(edges: Sequence[Sequence[int]], graph: Sequence[Sequence[int]]) -> set[int]:
    """Given a Sequence of edges and a graph, return all neighboring nodes of the edges.

    Args:
        edges: A sequence of pairs of integers, representing edges of a graph.
        graph: The input graph specified as a sequence of edges (Sequence[int]).

    Returns:
        List of all neighboring nodes of the input edges.

    """
    neighboring_nodes = set()
    nodes_in_edges = set()

    for u, v in edges:
        nodes_in_edges.add(u)
        nodes_in_edges.add(v)

    for x, y in graph:
        if x in nodes_in_edges:
            neighboring_nodes.add(y)
        if y in nodes_in_edges:
            neighboring_nodes.add(x)
    neighboring_nodes -= nodes_in_edges

    return neighboring_nodes


def get_pauli_expectation(counts: dict[str, int], pauli_label: str) -> float:
    """Gets an estimate of a Pauli expectation value for a given set of counts and a Pauli measurement label.

    Args:
        counts: A dictionary of counts.
            * NB: keys are assumed to have a single bitstring, i.e., coming from a single classical register.
        pauli_label: A Pauli measurement label, specified as a string of I, X, Y, Z characters.

    Raises:
        ValueError: If Pauli labels are not specified in terms of I, X, Y, Z characters.

    Returns:
        The estimate of the Pauli expectation value.

    """
    num_qubits = len(list(counts.keys())[0])
    sqg_pauli_strings = ("I", "Z", "X", "Y")
    all_pauli_labels = ["".join(x) for x in itertools.product(sqg_pauli_strings, repeat=num_qubits)]

    if pauli_label not in all_pauli_labels:
        raise ValueError("pauli_label must be specified as a string made up of characters 'I', 'X', 'Y', or 'Z'.")

    expect = 0
    if "I" not in pauli_label:
        for b, count_b in counts.items():
            if b.count("1") % 2 == 0:
                expect += count_b
            else:
                expect -= count_b
        return expect / sum(counts.values())

    non_id_indices = [idx for idx, P in enumerate(pauli_label) if P != "I"]
    for b, count_b in counts.items():
        b_z_parity = [1 if b[i] == "1" else 0 for i in non_id_indices]
        if sum(b_z_parity) % 2 == 0:
            expect += count_b
        else:
            expect -= count_b
    return expect / sum(counts.values())


def get_tomography_matrix(pauli_expectations: dict[str, float]) -> np.ndarray:
    """Reconstructs a density matrix from given Pauli expectations.

    Args:
        pauli_expectations: A dictionary of Pauli expectations, with keys being Pauli strings.

    Raises:
        ValueError: If not all 4**n Pauli expectations are specified.

    Returns:
        A tomographically reconstructed density matrix.

    """
    num_qubits = len(list(pauli_expectations.keys())[0])
    sqg_pauli_strings = ("I", "Z", "X", "Y")
    all_pauli_labels = ["".join(x) for x in itertools.product(sqg_pauli_strings, repeat=num_qubits)]
    if set(pauli_expectations.keys()) != set(all_pauli_labels):
        raise ValueError(
            f"Pauli expectations are incomplete ({len(list(pauli_expectations.keys()))} "
            f"out of {len(all_pauli_labels)} expectations)"
        )

    rho = np.zeros([2**num_qubits, 2**num_qubits], dtype=complex)
    for pauli_string, pauli_expectation in pauli_expectations.items():
        rho += 2 ** (-num_qubits) * pauli_expectation * Pauli(pauli_string).to_matrix()
    return rho


def marginal_distribution(prob_dist_or_counts: dict[str, float | int], indices: Iterable[int]) -> dict[str, float]:
    """Compute the marginal distribution over specified bits (indices).

    Args:
        prob_dist_or_counts: A dictionary with keys being bitstrings, and values are either probabilities or counts.
        indices: A list of bit indices to marginalize over.

    Returns:
        A dictionary representing the marginal distribution over the specified bits.

    """
    marginal_dist: dict[str, float] = defaultdict(float)

    for bitstring, prob in prob_dist_or_counts.items():
        # Extract the bits at the specified indices and form the marginalized bitstring
        marginalized_bitstring = "".join(bitstring[i] for i in sorted(indices))
        # Sum up probabilities for each marginalized bitstring
        marginal_dist[marginalized_bitstring] += prob

    return dict(marginal_dist)


def median_with_uncertainty(observations: Sequence[float]) -> dict[str, float]:
    """Computes the median of a Sequence of float observations and returns value and propagated uncertainty.

    Reference: https://mathworld.wolfram.com/StatisticalMedian.html

    Args:
        observations: a Sequence of floating-point numbers.

    Returns:
        A dictionary with keys "value" and "uncertainty" for the median of the input Sequence.

    """
    median = np.median(observations)
    n_observations = len(observations)
    error_from_mean = np.std(observations) / np.sqrt(n_observations)
    median_uncertainty = error_from_mean * np.sqrt(np.pi * n_observations / (2 * (n_observations - 1)))

    return {"value": float(median), "uncertainty": float(median_uncertainty)}


@timeit
def perform_backend_transpilation(  # noqa: PLR0913
    qc_list: list[QuantumCircuit],
    backend: IQMBackendBase,
    qubits: Sequence[int],
    coupling_map: list[list[int]],
    basis_gates: Sequence[str] = ("r", "cz"),
    qiskit_optim_level: int = 1,
    optimize_sqg: bool = False,
    drop_final_rz: bool = True,
    routing_method: RoutingMethod | None = RoutingMethod.SABRE,
    approximation_degree: float = 1.0,
) -> list[QuantumCircuit]:
    """Transpile a list of circuits to backend specifications.

    Args:
        qc_list: The original (untranspiled) list of quantum circuits.
        backend: The backend to execute the benchmark on.
        qubits: The qubits to target in the transpilation.
        coupling_map: The target coupling map to transpile to.
        basis_gates: The basis gates.
        qiskit_optim_level: Qiskit "optimization_level" value.
        optimize_sqg: Whether SQG optimization is performed taking into account virtual Z.
        drop_final_rz: Whether the SQG optimizer drops a final RZ gate.
        routing_method: The routing method employed by Qiskit's transpilation pass.
        approximation_degree: How much the transpiler is allowed to approximate the original circuit in terms of
            lower gate counts and depth. 1.0 means no deviation allowed, 0.0 means maximum deviation allowed.

    Returns:
        A list of transpiled quantum circuits.

    Raises:
        ValueError: if Star topology and label 0 is in qubit layout.

    """

    # Helper function considering whether optimize_sqg is done,
    # and whether the coupling map is reduced (whether final physical layout must be fixed onto an auxiliary QC)
    def transpile_and_optimize(qc: QuantumCircuit, aux_qc: QuantumCircuit | None = None) -> QuantumCircuit:
        if backend.has_resonators():
            coupling_map_red = (
                backend.coupling_map.reduce(qubits[: qc.num_qubits]) if aux_qc is not None else coupling_map
            )
            transpiled = transpile_to_IQM(
                qc,
                backend=backend,
                optimize_single_qubits=optimize_sqg,
                remove_final_rzs=drop_final_rz,
                coupling_map=coupling_map_red,
                approximation_degree=approximation_degree,
            )
        else:
            transpiled = transpile(
                qc,
                basis_gates=basis_gates,
                coupling_map=coupling_map,
                optimization_level=qiskit_optim_level,
                initial_layout=qubits if aux_qc is None else None,
                routing_method=routing_method.value if isinstance(routing_method, RoutingMethod) else routing_method,
                approximation_degree=approximation_degree,
            )
            if aux_qc is not None:
                transpiled = aux_qc.compose(transpiled, qubits=qubits, clbits=list(range(qc.num_clbits)))
            if optimize_sqg:
                transpiled = optimize_single_qubit_gates(transpiled, drop_final_rz=drop_final_rz)
        return transpiled

    qcvv_logger.info(
        f"Transpiling for backend {backend.name} with optimization level {qiskit_optim_level}, "
        f"{routing_method} routing method{' including SQG optimization' if qiskit_optim_level > 0 else ''} all circuits"
    )

    if coupling_map == backend.coupling_map:
        transpiled_qc_list = [transpile_and_optimize(qc) for qc in qc_list]
    else:  # The coupling map will be reduced if the physical layout is to be fixed
        if backend.has_resonators():
            aux_qc_list = [QuantumCircuit(backend.num_qubits, q.num_clbits) for q in qc_list]
        else:
            aux_qc_list = [QuantumCircuit(backend.num_qubits, q.num_clbits) for q in qc_list]
        transpiled_qc_list = [transpile_and_optimize(qc, aux_qc=aux_qc_list[idx]) for idx, qc in enumerate(qc_list)]

    return transpiled_qc_list


def random_hamiltonian_path(graph: nx.Graph, n_vertices: int) -> list[tuple[int, int]]:
    """Generates a random Hamiltonian path with n_vertices from a given NetworkX graph.

    Args:
        graph: The input graph.
        n_vertices: The desired number of vertices in the Hamiltonian path.

    Returns:
        A list of edges (tuples of nodes) representing the Hamiltonian path, or an empty list if not possible.

    """
    if n_vertices > len(graph):
        raise ValueError(
            f"The number of vertices in the Hamiltonian path ({n_vertices}) cannot be greater than "
            f"the number of nodes in the graph ({len(graph)})"
        )

    nodes = list(graph.nodes)
    random.shuffle(nodes)  # Shuffle nodes to introduce randomness

    for start in nodes:
        path = [start]
        visited = set(path)
        edges = []

        while len(path) < n_vertices:
            neighbors = [n for n in graph.neighbors(path[-1]) if n not in visited]

            if not neighbors:
                break  # Dead end, stop trying this path

            next_node = secrets.choice(neighbors)
            edges.append((int(path[-1]), int(next_node)))
            path.append(next_node)
            visited.add(next_node)

        if len(path) == n_vertices:
            return edges  # Successfully found a Hamiltonian path of length n_vertices

    return []  # No valid path found


def reduce_to_active_qubits(
    circuit: QuantumCircuit,
    backend_topology: str | None = None,
    backend_num_qubits: int | None = None,
) -> QuantumCircuit:
    """Reduces a quantum circuit to only its active qubits.

    Args:
        backend_topology: The backend topology to execute the benchmark on.
        circuit: The original quantum circuit.
        backend_num_qubits: The number of qubits in the backend.

    Returns:
        A new quantum circuit containing only active qubits.

    """
    # Identify active qubits
    active_qubits: list | set = set()
    for instruction in circuit.data:
        for qubit in instruction.qubits:
            cast(set, active_qubits).add(circuit.find_bit(qubit).index)
    if backend_topology == "star" and backend_num_qubits not in active_qubits:
        # For star systems, the resonator must always be there, regardless of whether it MOVE gates on it or not
        cast(set, active_qubits).add(backend_num_qubits)

    # Create a mapping from old qubits to new qubits
    active_qubits = list(set(active_qubits))
    qubit_map = {old_idx: new_idx for new_idx, old_idx in enumerate(active_qubits)}

    # Create a new quantum circuit with the reduced number of qubits
    reduced_circuit = QuantumCircuit(len(active_qubits))

    # Add classical registers if they exist
    if circuit.num_clbits > 0:
        creg = ClassicalRegister(circuit.num_clbits)
        reduced_circuit.add_register(creg)

    # Copy operations to the new circuit, remapping qubits and classical bits
    for instruction in circuit.data:
        new_qubits = [reduced_circuit.qubits[qubit_map[circuit.find_bit(qubit).index]] for qubit in instruction.qubits]
        new_clbits = [reduced_circuit.clbits[circuit.find_bit(clbit).index] for clbit in instruction.clbits]
        reduced_circuit.append(instruction.operation, new_qubits, new_clbits)

    return reduced_circuit


def remove_directed_duplicates_to_list(cp_map: CouplingMap) -> list[list[int]]:
    """Remove duplicate edges from a coupling map and returns as a list of edges (as a list of pairs of vertices).

    Args:
        cp_map: A list of pairs of integers, representing a coupling map.

    Returns:
        The edges of the coupling map.

    """
    sorted_cp = [sorted(x) for x in list(cp_map)]
    return [list(x) for x in set(map(tuple, sorted_cp))]


@timeit
def retrieve_all_counts(iqm_jobs: list[IQMJob], identifier: str | None = None) -> list[dict[str, int]]:
    """Retrieve the counts from a list of IQMJob objects.

    Args:
        iqm_jobs: The list of IQMJob objects.
        identifier: a string identifying the job.

    Returns:
        The counts of all the IQMJob objects.

    """
    if identifier is None:
        qcvv_logger.info("Retrieving all counts")
    else:
        qcvv_logger.info(f"Retrieving all counts for {identifier}")
    final_counts = []
    for j in iqm_jobs:
        counts = j.result().get_counts()
        if isinstance(counts, list):
            final_counts.extend(counts)
        elif isinstance(counts, dict):
            final_counts.append(counts)

    return final_counts


def retrieve_all_job_metadata(
    iqm_jobs: list[IQMJob],
) -> dict[str, dict[str, Any]]:
    """Retrieve the metadata from a list of Job objects.

    Args:
        iqm_jobs: The list of IQMJob objects.

    Returns:
        Relevant metadata of all the IQMJob objects.

    """
    all_meta = {}
    for index, j in enumerate(iqm_jobs):
        all_attributes_j = dir(j)
        if not hasattr(j, "_iqm_job"):
            shots = j.metadata["shots"] if "shots" in j.metadata.keys() else None
            timestamps = j.metadata["timestamps"] if "timestamps" in j.metadata.keys() else None
        else:
            job_parameters = j._iqm_job._parameters
            if job_parameters is not None:
                shots = job_parameters.shots if "shots" in job_parameters.__dict__.keys() else None
            else:
                raise ValueError("Job parameters return None, cannot retrieve shots information.")
            timestamps = {}
            for entry in j._iqm_job.data.timeline:
                timestamps.update({entry.status: entry.timestamp})
        all_meta.update(
            {
                "batch_job_" + str(index + 1): {
                    "job_id": j.job_id() if "job_id" in all_attributes_j else None,
                    "backend": (j.backend().name if "backend" in all_attributes_j else None),
                    "status": (j.status().value if "status" in all_attributes_j else None),
                    "circuits_in_batch": (
                        len(cast(list, j.circuit_metadata)) if "circuit_metadata" in all_attributes_j else None
                    ),
                    "shots": shots,
                    "timestamps": timestamps,
                }
            }
        )

    return all_meta


def set_coupling_map(
    qubits: Sequence[int],
    backend: IQMBackendBase,
    physical_layout: PhysicalLayout = PhysicalLayout.FIXED,
) -> CouplingMap:
    """Set a coupling map according to the specified physical layout.

    Args:
        qubits: A list of physical qubits to consider.
        backend: A backend from IQM.
        physical_layout: A physical layout type to consider. "fixed" sets a coupling map restricted to the input
            qubits; the results will be constrained to measure those qubits. "batching" sets the coupling map of the
            backend; the results in a benchmark will be "batched" according to final layouts.

    Returns:
        A coupling map according to the specified physical layout.

    Raises:
        ValueError: If the physical layout is not "fixed" or "batching".

    """
    if physical_layout == PhysicalLayout.FIXED:
        return backend.coupling_map.reduce(mapping=qubits)
    if physical_layout == PhysicalLayout.BATCHING:
        return backend.coupling_map
    raise ValueError('physical_layout must either be "fixed" or "batching"')


def split_sequence_in_chunks(sequence_in: Sequence[Any], split_size: int) -> list[Sequence[Any]]:
    """Split a given Sequence into chunks of a given split size, return as a List of Sequences.

    Args:
        sequence_in: The input list.
        split_size: The split size.

    Returns:
        A List of Sequences.

    """
    if split_size > len(sequence_in):
        raise ValueError("The split size should be smaller or equal than the list length")
    if len(sequence_in) % split_size != 0 and (split_size != 1 and split_size != len(sequence_in)):
        qcvv_logger.debug(
            f"Since len(input_list) = {len(sequence_in)} and split_size = {split_size}, "
            f"the input list will be split into chunks of uneven size!"
        )
        warnings.warn(
            f"Since len(input_list) = {len(sequence_in)} and split_size = {split_size}, "
            f"the input list will be split into chunks of uneven size!",
            stacklevel=2,
        )

    return [sequence_in[i : i + split_size] for i in range(0, len(sequence_in), split_size)]


def split_into_disjoint_pairs(
    pairs: Sequence[tuple[int, int]],
) -> list[list[tuple[int, int]]]:
    """Split a Sequence of pairs of integers into a List of a minimal number of Lists of disjoint pairs.

    Example: input [(0,3), (2,3), (3,8), (8,13), (13,17), (17,18)] gives
    output [[(0, 3), (8, 13), (17, 18)], [(2, 3), (13, 17)], [(3, 8)]].

    Args:
        pairs: The input list of pairs of integers.

    Returns:
        A List of Lists of disjoint pairs.

    """
    result: list[list[tuple[int, int]]] = []

    for pair in pairs:
        added = False
        for group in result:
            if not any(elem in pair for p in group for elem in p):
                group.append(pair)
                added = True
                break
        if not added:
            result.append([pair])

    return result


@timeit
def sort_batches_by_final_layout(
    transpiled_circuit_list: list[QuantumCircuit],
) -> tuple[dict[tuple, list[QuantumCircuit]], dict[tuple, list[int]]]:
    """Sort batches of circuits according to the final measurement mapping in their corresponding backend.

    Args:
        transpiled_circuit_list: the list of circuits transpiled to a given backend.

    Returns:
        sorted_circuits: dictionary, keys: final measured qubits, values: corresponding circuits.
        sorted_indices: dictionary, keys: final measured qubits, values: corresponding circuit indices.

    """
    qcvv_logger.info("Now getting the final measurement maps of all circuits")
    all_measurement_maps = [tuple(final_measurement_mapping(qc).values()) for qc in transpiled_circuit_list]
    unique_measurement_maps = {tuple(sorted(x)) for x in all_measurement_maps}
    sorted_circuits: dict[tuple, list[QuantumCircuit]] = {u: [] for u in unique_measurement_maps}
    sorted_indices: dict[tuple, list[int]] = {i: [] for i in unique_measurement_maps}
    for index, qc in enumerate(transpiled_circuit_list):
        final_measurement = all_measurement_maps[index]
        final_measurement = tuple(sorted(final_measurement))
        sorted_circuits[final_measurement].append(qc)
        sorted_indices[final_measurement].append(index)

    if len(sorted_circuits) == 1:
        qcvv_logger.info("The routing method generated a single batch of circuits to be measured")
    else:
        qcvv_logger.info(f"The routing method generated {len(sorted_circuits)} batches of circuits to be measured")

    return sorted_circuits, sorted_indices


@timeit
def submit_execute(
    sorted_transpiled_qc_list: dict[tuple[int] | str, list[QuantumCircuit]],
    backend: IQMBackendBase,
    shots: int,
    max_gates_per_batch: int | None = None,
    max_circuits_per_batch: int | None = None,
    circuit_compilation_options: CircuitCompilationOptions | None = None,
) -> list[IQMJob]:
    """Submit function to execute lists of quantum circuits on the specified backend.

        Organized as a dictionary with keys being identifiers of a batch (normally qubits)
        and values corresponding lists of quantum circuits. The result is returned as a single list of IQMJob objects.

    Args:
        sorted_transpiled_qc_list:
            A dictionary of lists of quantum circuits to be executed.
            * The keys (Tuple[int] | str) should correspond to final measured qubits.
            * The values (List[QuantumCircuit]) should be the corresponding list (batch) of quantum circuits.
        backend: the backend to execute the circuits on.
        shots: the number of shots per circuit.
        max_gates_per_batch: the maximum number of gates per batch sent to the backend,
            used to make manageable batches.
        max_circuits_per_batch: the maximum number of circuits per batch sent to the backend,
            used to make manageable batches.
        circuit_compilation_options: Ability to pass a compilation options object,
            enabling execution with dynamical decoupling, among other options - see qiskit-iqm documentation.

    Returns:
        A list of IQMJob objects corresponding to the submitted circuits.

    """
    final_jobs = []
    for k in sorted(
        sorted_transpiled_qc_list.keys(),
        key=lambda x: len(sorted_transpiled_qc_list[x]),
        reverse=True,
    ):
        # sorted is so batches are looped from larger to smaller
        qcvv_logger.info(
            f"Submitting batch with {len(sorted_transpiled_qc_list[k])} circuits corresponding to qubits {list(k)}"
        )
        # Divide into batches according to maximum gate count per batch
        if max_gates_per_batch is None and max_circuits_per_batch is None:
            jobs = backend.run(
                sorted_transpiled_qc_list[k], shots=shots, circuit_compilation_options=circuit_compilation_options
            )
            final_jobs.append(jobs)

        else:
            if max_gates_per_batch is None and max_circuits_per_batch is not None:
                restriction = "max_circuits_per_batch"
                batching_size = max_circuits_per_batch

            elif max_circuits_per_batch is None and max_gates_per_batch is not None:
                restriction = "max_gates_per_batch"
                # Calculate average gate count per quantum circuit
                avg_gates_per_qc = sum(sum(qc.count_ops().values()) for qc in sorted_transpiled_qc_list[k]) / len(
                    sorted_transpiled_qc_list[k]
                )
                batching_size = max(1, floor(max_gates_per_batch / avg_gates_per_qc))

            else:  # Both are not None - select the one rendering the smallest batches.
                # Calculate average gate count per quantum circuit
                avg_gates_per_qc = sum(sum(qc.count_ops().values()) for qc in sorted_transpiled_qc_list[k]) / len(
                    sorted_transpiled_qc_list[k]
                )
                qcvv_logger.warning(
                    "Both max_gates_per_batch and max_circuits_per_batch are not None. "
                    "Selecting the one giving the smallest batches."
                )
                batching_size = min(
                    cast(int, max_circuits_per_batch),
                    max(1, floor(cast(int, max_gates_per_batch) / avg_gates_per_qc)),
                )
                if batching_size == max_circuits_per_batch:
                    restriction = "max_circuits_per_batch"
                else:
                    restriction = "max_gates_per_batch"

            final_batch_jobs = []
            for index, qc_batch in enumerate(chunked(sorted_transpiled_qc_list[k], batching_size)):
                qcvv_logger.info(
                    f"{restriction} restriction: submitting subbatch #{index + 1} "
                    f"with {len(qc_batch)} circuits corresponding to qubits {list(k)}"
                )
                batch_jobs = backend.run(
                    qc_batch,
                    shots=shots,
                    circuit_compilation_options=circuit_compilation_options,
                )
                final_batch_jobs.append(batch_jobs)
            final_jobs.extend(final_batch_jobs)

    return final_jobs


def xrvariable_to_counts(dataset: xr.Dataset, identifier: str, counts_range: int) -> list[dict[str, int]]:
    """Retrieve counts from xarray dataset.

    Args:
        dataset: the dataset to extract counts from.
        identifier: the identifier for the dataset counts.
        counts_range: the range of counts to extract (e.g., the amount of circuits that were executed).

    Returns:
        A list of counts dictionaries from the dataset.

    """
    return [
        dict(
            zip(
                list(dataset[f"{identifier}_state_{u}"].data),
                dataset[f"{identifier}_counts_{u}"].data,
                strict=True,
            )
        )
        for u in range(counts_range)
    ]
