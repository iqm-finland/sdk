# Copyright 2022-2026 IQM
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

"""Utility functions for generating calibration circuits for readout error characterization."""

from collections.abc import Sequence
from dataclasses import dataclass
from time import sleep
from typing import Any
import warnings

from iqm.error_reduction_tools.twirling.twirling_processors import sum_counts
from iqm.error_reduction_tools.utils.topology_utils import operational_qubits_from_qc
import numpy as np

from iqm.pulla.pulla import Pulla, PullaJob
from iqm.pulla.utils_qiskit import sweep_job_to_qiskit
from iqm.pulse import Circuit, CircuitOperation

from .data_processing import CalibrationData


@dataclass
class CalibrationJobInfo:
    """Metadata about a submitted calibration job."""

    num_circuits: int
    """Total number of calibration circuits generated."""
    shots_per_circuit: int
    """Number of shots allocated to each circuit."""
    prep_strings: list[str]
    """List of preparation strings corresponding to each circuit."""
    qubits: list[str]
    """List of qubits that were characterized."""
    context: Any
    """Execution context returned from the compiler."""


def generate_strings(
    number_of_qubits: int,
    number_of_circuits: int,
    rgen: np.random.Generator,
    symmetrize: bool = True,
) -> list[str]:
    """Generate random preparation strings of 'I' and 'X' characters.

    Each string encodes a preparation state for a multi-qubit system:
    'I' = prepare ``|0>``, 'X' = prepare ``|1>`` (via X gate).

    Args:
        number_of_qubits: Length of each preparation string (number of qubits).
        number_of_circuits: Total number of preparation strings to generate.
        rgen: NumPy random number generator instance.
        symmetrize: If ``True``, generates pairs of complementary strings (I↔X)
            to balance ``|0>`` and ``|1>`` preparations.

    Returns:
        List of preparation strings, each of length ``number_of_qubits``.

    """
    strings: list[str] = []
    num_pairs = number_of_circuits // 2

    for _ in range(num_pairs):
        base_string = "".join(rgen.choice(["I", "X"], size=number_of_qubits))
        strings.append(base_string)
        if symmetrize:
            # Add the bit-flipped version
            inverted_string = "".join(["I" if bit == "X" else "X" for bit in base_string])
            strings.append(inverted_string)
        else:
            # Add another random string if not symmetrizing
            strings.append("".join(rgen.choice(["I", "X"], size=number_of_qubits)))

    # Handle odd number of circuits
    if number_of_circuits % 2 != 0:
        strings.append("".join(rgen.choice(["I", "X"], size=number_of_qubits)))

    return strings


def create_calibration_circuits(
    qubits: list[str],
    number_of_circuits: int,
    symmetrize: bool = True,
    seed: int | None = None,
    equatorial_randomization: bool = False,
) -> tuple[list[Circuit], list[str]]:
    """Generate readout error characterization (REC) circuits and preparation strings.

    Creates circuits that prepare qubits in computational basis states (``|0>`` or ``|1>``)
    using I (identity) or X gates, then measure all qubits. Optionally applies
    equatorial randomization to X gates for improved error characterization.

    Args:
        qubits: IQM qubit names to characterize (e.g., ['QB1', 'QB2']).
        number_of_circuits: Number of calibration circuits to generate.
        symmetrize: If ``True``, generates complementary circuit pairs (I↔X) to
            balance ``|0>`` and ``|1>`` state preparations across all qubits.
        seed: Random seed for reproducible circuit generation.
        equatorial_randomization: If ``True``, randomize X gate phase angles (phi)
            around the Bloch sphere equator. This distributes state preparation over
            the XY plane.

    Returns:
        - Circuits ready for compilation.
        - Preparation strings (same length as circuits), where each
          character is 'I' (prepare ``|0>``) or 'X' (prepare ``|1>``).

    Raises:
        ValueError: If qubits list is empty or ``number_of_circuits`` < 1.

    Example:
        >>> circuits, preps = create_calibration_circuits(['QB1', 'QB2'], 4, symmetrize=True)
        >>> len(circuits), len(preps)
        (4, 4)

    """
    num_qubits: int = len(qubits)

    if num_qubits == 0:
        raise ValueError("The qubit list is empty. Please provide a valid list of qubits.")

    if number_of_circuits < 1:
        raise ValueError("The number of circuits must be at least 1.")

    rgen = np.random.default_rng(seed)

    prep_strings: list[str] = generate_strings(num_qubits, number_of_circuits, rgen, symmetrize)

    circuits: list[Circuit] = []

    for i, prep_str in enumerate(prep_strings):
        # Define measurement instructions
        measure_instructions: list[CircuitOperation] = [
            CircuitOperation(name="measure", locus=(qb_name,), args={"key": f"m_{num_qubits}_0_{q}"})
            for q, qb_name in enumerate(qubits)
        ]
        # Defaults to X
        equatorial_angles = np.zeros(shape=num_qubits, dtype=float)
        if equatorial_randomization:
            equatorial_angles = rgen.uniform(0, 2 * np.pi, size=num_qubits)

        # Create a circuit for each string
        instruction_list: list[CircuitOperation] = []
        for q, op in enumerate(prep_str):
            if op == "X":
                instruction_list.append(
                    CircuitOperation(
                        name="prx",
                        locus=(qubits[q],),
                        args={"angle": np.pi, "phase": equatorial_angles[q]},
                    )
                )

        # Combine preparation instructions and measurements
        full_instructions = tuple(instruction_list + measure_instructions)
        circuits.append(Circuit(name=f"calibration_circuit_{i}", instructions=full_instructions))

    return circuits, prep_strings


def run_calibration_circuits(
    client: Pulla,
    qubits: Sequence[str] | None = None,
    number_of_circuits: int = 50,
    shots: int = 50000,
    symmetrize: bool = True,
    seed: int | None = None,
    equatorial_randomization: bool = True,
) -> tuple[PullaJob, CalibrationJobInfo]:
    """Execute readout error characterization circuits on a quantum computer and collect measurement results.

    Generates random basis state preparation circuits and runs them on the specified QC.

    Args:
        client: Client for connecting to an IQM quantum computer.
        qubits: Use ``None`` to characterize every *operational* QPU qubit
            (qubits with a calibrated ``measure`` gate), or a list of specific qubit names
            (e.g., ``["QB1", "QB2"]``). Explicitly requested qubits that are not operational
            are dropped with a warning.
        number_of_circuits: Number of distinct calibration circuits to run.
        shots: Total measurement shots distributed across all circuits.
            Each circuit receives ``shots // number_of_circuits`` shots.
        symmetrize: If ``True``, generates complementary preparation pairs (I↔X).
        seed: Random seed for reproducible preparation string generation.
        equatorial_randomization: If ``True``, randomizes X gate phase angles.

    Returns:
        Submitted job, metadata about the job.

    Raises:
        ValueError: If ``shots`` < ``number_of_circuits`` (no shots per circuit), or if no
            operational qubits remain to characterize.


    """
    # Resolve the operational qubits (those with a calibrated measure gate).
    operational = operational_qubits_from_qc(client)
    if operational is None:
        # DQA unavailable or empty. Fall back to whatever qubits the chip topology reports
        # so characterization can still proceed, and let the user know we could not confirm
        # they are actually operational.
        try:
            chip_topo = client.get_chip_topology()
            operational = list(chip_topo.qubits_sorted)
        except Exception as exc:
            raise ValueError(
                "Could not determine operational qubits from the dynamic quantum architecture, "
                "and chip topology fallback also failed. The server may be unavailable or misconfigured."
            ) from exc
        warnings.warn(
            "Could not determine operational qubits from the dynamic quantum architecture; "
            f"falling back to chip topology qubits {operational}. "
            "Some of these qubits may not have a calibrated measure gate.",
            stacklevel=2,
        )

    qubit_list: list[str]
    if qubits is None:
        qubit_list = operational
    else:
        operational_set = set(operational)
        requested = list(qubits)
        qubit_list = [qubit for qubit in requested if qubit in operational_set]
        dropped = [qubit for qubit in requested if qubit not in operational_set]
        if dropped:
            warnings.warn(
                f"Dropping non-operational qubit(s) {dropped} from readout error characterization; "
                "they have no calibrated measure gate and cannot be characterized or mitigated.",
                stacklevel=2,
            )

    if not qubit_list:
        raise ValueError(
            "No operational qubits to characterize. The requested qubits are not operational, "
            "or the QPU reports no operational qubits."
        )

    compiler = client.get_standard_compiler()
    circuits, prep_strings = create_calibration_circuits(
        qubit_list,
        number_of_circuits,
        symmetrize,
        seed,
        equatorial_randomization,
    )

    shots_per_circuit = shots // number_of_circuits
    if shots_per_circuit == 0:
        raise ValueError(
            f"Total shots ({shots}) is less than the number of circuits "
            f"({number_of_circuits}). Increase shots or decrease number_of_circuits."
        )

    # playlist, context = compiler.compile(circuits)
    # settings, context = compiler.build_settings(context, shots=shots_per_circuit)
    #
    # job = client.submit_playlist(playlist=playlist, settings=settings, context=context)

    job_definition, context = compiler.compile(circuits=circuits)
    job = client.submit_playlist(job_definition, context=context)

    job_info = CalibrationJobInfo(
        num_circuits=number_of_circuits,
        shots_per_circuit=shots_per_circuit,
        prep_strings=prep_strings,
        qubits=qubit_list,
        context=context,
    )

    return job, job_info


def retrieve_calibration_results(
    job: PullaJob,
    job_info: CalibrationJobInfo,
) -> CalibrationData:
    """Retrieve and organize measurement results from the calibration circuits.

    Groups raw measurement counts by their preparation string. The raw counts are preserved
    so that :func:`~iqm.error_reduction_tools.readout_characterization.data_processing.compute_error_probabilities`
    can compare them directly against the preparation string to compute P(0|1) and P(1|0).

    Args:
        job: Job tracker for the calibration job.
        job_info: Metadata about the submitted calibration job.

    Returns:
        Calibration results. Multiple circuits with the same preparation string have merged counts.

    Raises:
        RuntimeError: If the job did not complete after multiple wait attempts.

    """
    num_circuits = job_info.num_circuits
    shots_per_circuit = job_info.shots_per_circuit
    prep_strings = job_info.prep_strings
    qubit_list = job_info.qubits

    max_retries = 10
    retry_delay_s = 30.0
    for attempt in range(max_retries):
        job.wait_for_completion()
        try:
            qiskit_results = sweep_job_to_qiskit(job, shots=shots_per_circuit)
            break
        except ValueError as exc:
            if "WAITING" in str(exc) and attempt < max_retries - 1:
                sleep(retry_delay_s)
            else:
                raise
    else:
        raise RuntimeError("Job did not complete after multiple wait attempts.")

    # Group counts by preparation string and merge duplicates
    counts_per_prep: dict[str, list[dict[str, float]]] = {}
    for i in range(num_circuits):
        counts = qiskit_results.get_counts(i)
        prep_str = prep_strings[i]
        counts_per_prep.setdefault(prep_str, []).append(counts)

    # Merge counts for each preparation string
    counts_by_prep = {prep_str: sum_counts(counts_list) for prep_str, counts_list in counts_per_prep.items()}

    output_data: CalibrationData = {
        "counts_by_prep": counts_by_prep,
        "measured_qubits": qubit_list,
    }

    return output_data
