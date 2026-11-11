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
"""M3 modification for readout mitigation at IQM QPU's."""

from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
import logging
from math import ceil
import os
import threading
from typing import Any, cast
import warnings

from iqm.benchmarks.logging_config import qcvv_logger, temporary_log_level
from iqm.benchmarks.pulse_utils import compile_pulse, prepare_pulse_compiler, retrieve_pulse, submit_pulse
from iqm.benchmarks.utils import get_iqm_backend, timeit
from iqm.iqm_client import CircuitCompilationOptions
from iqm.qiskit_iqm import IQMCircuit as QuantumCircuit
from iqm.qiskit_iqm.iqm_provider import IQMBackend, IQMFacadeBackend
import mthree  # type: ignore[import-untyped]
from mthree.circuits import (  # type: ignore[import-untyped]
    _marg_meas_states,
    _tensor_meas_states,
    balanced_cal_circuits,
)
from mthree.classes import QuasiCollection  # type: ignore[import-untyped]
from mthree.exceptions import M3Error  # type: ignore[import-untyped]
from mthree.generators import HadamardGenerator  # type: ignore[import-untyped]
from mthree.mitigation import _job_thread  # type: ignore[import-untyped]
from mthree.utils import final_measurement_mapping  # type: ignore[import-untyped]
import numpy as np
from qiskit import transpile
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

try:  # Qiskit 2.*
    from qiskit.providers import Backend, BackendV2

    old_qiskit = False
except ImportError:  # Qiskit 1.*
    from qiskit.providers import BackendV1 as Backend, BackendV2  # noqa: F401

    old_qiskit = True


# The code here is close to the original M3 code licenced under Apache 2 as well (https://github.com/Qiskit/qiskit-addon-mthree/blob/main/LICENSE.txt).
class M3IQM(mthree.M3Mitigation):
    """M3 readout mitigation class modified to work with IQM devices."""

    def __init__(self, system: IQMBackend | IQMFacadeBackend | None = None, iter_threshold: int = 4096):
        # Call parent init with None to avoid system_info call
        super().__init__(None, iter_threshold)
        # Now set our custom system_info
        self.system_info = self._custom_system_info(system) if system else {}
        self.system = system
        self.num_qubits = self.system_info["num_qubits"] if system else 0

    @property
    def effective_cal_shots(self) -> int:
        """Number of shots behind each single-qubit calibration probability P(0|0) and P(1|1)."""
        if self.cal_method == "balanced":
            return self._balanced_shots // 2
        return cast(int, self.cal_shots)

    def apply_correction(self, *args: Any, **kwargs: Any) -> Any:
        """Applies the M3 correction (see `mthree.M3Mitigation.apply_correction`) with a single OpenMP thread.

        This avoids oversubscription of the thread pool and subsequent process freeze.
        """
        with threadpool_limits(limits=1, user_api="openmp"):
            return super().apply_correction(*args, **kwargs)

    def _custom_system_info(self, system: IQMBackend | IQMFacadeBackend) -> dict[str, Any]:
        """Custom system_info implementation for IQM systems."""
        return {
            "num_qubits": system.num_qubits,
            "max_shots": 10000,
            "max_circuits": 100,
            "simulator": any("Fake" in key for key in system.__dict__),
            "inoperable_qubits": [],
            "name": getattr(system, "name", "iqm_system"),
        }

    def cals_from_system(
        self,
        qubits: list[dict[int, int]] | dict[int, int] | list[int] | None = None,
        shots: int | None = None,
        method: str | None = None,
        initial_reset: bool = False,
        cals_file: str | None = None,
        async_cal: bool = False,
        cal_id: str | None = None,
        circuit_compilation_options: CircuitCompilationOptions | None = None,
    ) -> None:
        """Grab calibration data from system.

        Args:
            qubits: Qubits over which to correct calibration data.
            shots: Number of shots per circuit. min(1e4, max_shots).
            method: Type of calibration, 'balanced' (default for hardware),
                         'independent' (default for simulators), or 'marginal'.
            initial_reset: Use resets at beginning of calibration circuits.
            cals_file: Output path to write JSON calibration data to.
            async_cal: Do calibration async in a separate thread.
            cal_id: Calibration set ID to use when submitting calibration circuits.
            circuit_compilation_options: Optional compilation options (e.g. active reset) to pass to ``backend.run()``.

        Raises:
            M3Error: Called while a calibration currently in progress.

        """
        if self._thread:
            raise M3Error("Calibration currently in progress.")
        if qubits is None:
            qubits = list(range(self.num_qubits))
            # Remove faulty qubits if any
            if any(self.system_info["inoperable_qubits"]):
                qubits = list(
                    filter(
                        lambda item: item not in self.system_info["inoperable_qubits"],
                        list(range(self.num_qubits)),
                    )
                )
                warnings.warn(
                    "Backend reporting inoperable qubits. Skipping calibrations for: %s",
                    self.system_info["inoperable_qubits"],
                    stacklevel=2,
                )
        else:
            qubits = list(qubits) if not isinstance(qubits, list) else qubits

        if method is None:
            method = "balanced"
        self.cal_method = method
        self.cals_file = cals_file
        self.cal_timestamp = None
        self.cal_shots: int | None = shots
        self.single_qubit_cals: list[np.ndarray | None] | None = None
        self._grab_additional_cals(
            qubits,
            shots=shots,
            method=method,
            initial_reset=initial_reset,
            async_cal=async_cal,
            cal_id=cal_id,
            circuit_compilation_options=circuit_compilation_options,
        )

    def _prepare_calibration_circuits(
        self,
        qubits: list,
        num_cal_qubits: int,
        method: str,
        initial_reset: bool,
    ) -> tuple[Any | list[Any], list[Any], int]:
        """Prepare REM calibration circuits based on the specified method.

        Args:
            qubits: List of qubit indices to calibrate.
            num_cal_qubits: Number of qubits to calibrate.
            method: Calibration method ('marginal', 'balanced', or 'independent').
            initial_reset: Whether to include initial reset operations in circuits.

        Returns:
            A tuple containing:
                - List of calibration circuits to execute.
                - List of calibration strings (currently unused, returns empty list).
                - The shots used for calibration circuits.

        """
        cal_strings: list[str] = []
        cal_shots_value: int = self.cal_shots if self.cal_shots is not None else 1000
        shots = cal_shots_value

        if method == "marginal":
            trans_qcs = _marg_meas_states(qubits, self.num_qubits, initial_reset=initial_reset)
        elif method == "balanced":
            generator = HadamardGenerator(num_cal_qubits)
            trans_qcs = balanced_cal_circuits(generator, qubits, self.num_qubits, initial_reset=initial_reset)
            shots = 2 * cal_shots_value // generator.length
            if 2 * cal_shots_value / generator.length != shots:
                shots += 1
            self._balanced_shots = shots * generator.length
        else:  # Independent
            trans_qcs = []
            for qubit in qubits:
                trans_qcs.extend(_tensor_meas_states(qubit, self.num_qubits, initial_reset=initial_reset))

        return trans_qcs, cal_strings, shots

    def _submit_calibration_jobs(
        self,
        trans_qcs: list,
        shots: int,
        cal_id: str | None,
        circuit_compilation_options: CircuitCompilationOptions | None = None,
    ) -> list:
        """Submit calibration circuits as jobs to the backend.

        Args:
            trans_qcs: List of transpiled calibration circuits to execute.
            shots: Number of shots per circuit.
            cal_id: Optional calibration set ID for job submission.
            circuit_compilation_options: Optional compilation options (e.g. active reset) to pass to ``backend.run()``.

        Returns:
            List of submitted jobs.

        Raises:
            M3Error: If system is not set or backend type is unknown.

        """
        num_circs = len(trans_qcs)

        if self.system is None:
            raise M3Error("System is not set")

        if old_qiskit:  # old Backend API
            max_circuits = getattr(self.system.configuration(), "max_experiments", 300)
        elif isinstance(self.system, BackendV2):
            max_circuits = self.system.max_circuits
            if max_circuits is None:
                max_circuits = 300
        else:
            raise M3Error("Unknown backend type")

        num_jobs = ceil(num_circs / max_circuits)
        circ_slice = ceil(num_circs / num_jobs)
        circs_list = [trans_qcs[kk * circ_slice : (kk + 1) * circ_slice] for kk in range(num_jobs - 1)] + [
            trans_qcs[(num_jobs - 1) * circ_slice :]
        ]

        jobs = []
        use_timeslot = os.getenv("USE_TIMESLOT") not in (None, "False", "false")
        for circs in circs_list:
            transpiled_circuit = transpile(circs, self.system, optimization_level=0)
            if cal_id is None:
                _job = self.system.run(
                    transpiled_circuit,
                    shots=shots,
                    circuit_compilation_options=circuit_compilation_options,
                    use_timeslot=use_timeslot,
                )
            else:
                _job = self.system.run(
                    transpiled_circuit,
                    shots=shots,
                    calibration_set_id=cal_id,
                    circuit_compilation_options=circuit_compilation_options,
                    use_timeslot=use_timeslot,
                )
            jobs.append(_job)
            qcvv_logger.info(f"REM: {len(circs)} calibration circuits to be executed!")

        return jobs

    def _grab_additional_cals(
        self,
        qubits: list[dict[int, int]] | dict[int, int] | list[int],
        shots: int | None = None,
        method: str = "balanced",
        initial_reset: bool = False,
        async_cal: bool = False,
        cal_id: str | None = None,
        circuit_compilation_options: CircuitCompilationOptions | None = None,
    ) -> None:
        """Grab missing calibration data from backend.

        Args:
            qubits: List of qubit indices to calibrate.
            shots: Number of shots per calibration circuit. If None, uses max_shots defined in system or 10000.
            method: Calibration method - 'independent', 'balanced', or 'marginal'.
            initial_reset: Whether to include initial reset operations in calibration circuits.
            async_cal: If True, run calibration in a separate thread asynchronously.
            cal_id: Optional calibration set ID for job submission.
            circuit_compilation_options: Optional compilation options (e.g. active reset) to pass to ``backend.run()``.

        Raises:
            M3Error: If method is invalid or attempting to calibrate inoperable qubits.

        """
        if self.single_qubit_cals is None:
            self.single_qubit_cals = [None] * self.num_qubits

        if self.cal_shots is None:
            if shots is None:
                shots = min(self.system_info["max_shots"], 10000)
            self.cal_shots = shots

        if method not in ["independent", "balanced", "marginal"]:
            raise M3Error(
                f"Invalid calibration method: {method}. Valid methods are 'independent', 'balanced', or 'marginal'."
            )

        # Process qubits argument
        if isinstance(qubits, dict):
            qubits = list(set(qubits.values()))
        elif isinstance(qubits, list) and qubits and isinstance(qubits[0], dict):
            _qubits = []
            for item in qubits:
                _qubits.extend(list(set(cast(dict, item).values())))
            qubits = list(set(_qubits))

        # Check for inoperable qubits
        inoperable_overlap = list(set(qubits) & set(self.system_info["inoperable_qubits"]))
        if any(inoperable_overlap):
            raise M3Error(f"Attempting to calibrate inoperable qubits: {inoperable_overlap}")

        num_cal_qubits = len(qubits)

        # Prepare calibration circuits
        trans_qcs, cal_strings, cal_shots = self._prepare_calibration_circuits(
            qubits, num_cal_qubits, method, initial_reset
        )

        # Submit jobs
        jobs = self._submit_calibration_jobs(trans_qcs, cal_shots, cal_id, circuit_compilation_options)

        # Execute job and cal building
        self._job_error = None
        if async_cal:
            thread = threading.Thread(
                target=_job_thread,
                args=(jobs, self, qubits, num_cal_qubits, HadamardGenerator(num_cal_qubits)),
            )
            self._thread = thread
            self._thread.start()
        else:
            _job_thread(jobs, self, qubits, num_cal_qubits, HadamardGenerator(num_cal_qubits))


@contextmanager
def _quiet_mthree() -> Iterator[None]:
    """Temporarily silences the INFO messages that mthree logs for every calibration and correction."""
    mthree_logger = logging.getLogger("mthree")
    level = mthree_logger.level
    mthree_logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        mthree_logger.setLevel(level)


def measure_readout(
    backend: IQMBackend | IQMFacadeBackend,
    qubit_maps: list[dict[int, int]],
    mit_shots: int,
    circuit_compilation_options: CircuitCompilationOptions | None = None,
) -> dict[str, Any]:
    """Measures M3 readout calibration data once for the union of all qubits measured in the given circuits.

    Args:
        backend: The backend to measure.
        qubit_maps: Classical bit to physical qubit maps of the circuits (see `mthree.utils.final_measurement_mapping`).
        mit_shots: Number of shots per qubit and prepared state.
        circuit_compilation_options: Optional compilation options (e.g. active reset) to pass to ``backend.run()``.

    Returns:
        A serializable dictionary with the 2x2 calibration matrix of each physical qubit ("cals", keyed by the qubit
        index as a string), the number of shots behind each calibration probability ("shots") and the "timestamp".

    """
    mit = M3IQM(backend)
    with _quiet_mthree():
        mit.cals_from_system(qubit_maps, shots=mit_shots, circuit_compilation_options=circuit_compilation_options)
    qubits = sorted({qubit for qubit_map in qubit_maps for qubit in qubit_map.values()})
    return {
        "cals": {str(q): np.asarray(mit.single_qubit_cals[q], dtype=float).tolist() for q in qubits},  # type: ignore[index]
        "shots": mit.effective_cal_shots,
        "timestamp": mit.cal_timestamp,
    }


def mitigator_from_cals(cals: dict[str, Any]) -> M3IQM:
    """Builds an M3 mitigator offline from stored calibration matrices (as returned by `measure_readout`)."""
    matrices: list[np.ndarray | None] = [None] * (max(int(q) for q in cals) + 1)
    for qubit, matrix in cals.items():
        matrices[int(qubit)] = np.asarray(matrix, dtype=np.float32)
    mit = M3IQM()
    mit.cals_from_matrices(matrices)
    return mit


def mitigate_counts(mit: M3IQM, counts: list[dict[str, int]], qubit_maps: list[dict[int, int]]) -> list[Any]:
    """Applies the M3 correction to each counts dictionary and projects onto the nearest probability distribution."""
    # One thread limit around the whole batch (see M3IQM.apply_correction), so that the projection is covered too
    with _quiet_mthree(), threadpool_limits(limits=1, user_api="openmp"):
        return [
            mit.apply_correction(c, q).nearest_probability_distribution()
            for c, q in zip(counts, qubit_maps, strict=True)
        ]


class _PulseCalibrationResult:
    """Counts of a pulse-level calibration job, exposed like a Qiskit result for mthree."""

    def __init__(self, counts: list[dict[str, int]]):
        self._counts = counts

    def get_counts(self) -> list[dict[str, int]]:
        """Returns the counts of every calibration circuit."""
        return self._counts


class _PulseCalibrationJob:
    """Finished pulse-level calibration job, exposed like a Qiskit job for mthree's ``_job_thread``."""

    def __init__(self, counts: list[dict[str, int]]):
        self._result = _PulseCalibrationResult(counts)

    def result(self) -> _PulseCalibrationResult:
        """Returns the result of the job."""
        return self._result


class M3IQMPulse(M3IQM):
    """M3 readout mitigation whose calibration circuits are executed at pulse level.

    The calibration circuits are compiled with the same settings as the pulse-level benchmark circuits, so that
    both run with the same reset conditions (e.g. a shortened reset wait or active reset). The calibration then
    measures the assignment errors of the experiment that is being mitigated.

    Args:
        system: The IQM backend. Must expose an ``IQMClient``.
        pulse_settings: Compiler settings-tree values keyed by dotted path, as used for the benchmark circuits.
        active_reset_cycles: Number of active reset cycles. ``None`` keeps the compiler default.
        use_timeslot: Whether the calibration job is submitted into a reserved timeslot.
        timeout_secs: Maximum time to wait for the calibration job. ``0.0`` waits indefinitely.
        iter_threshold: Number of bitstrings above which iterative solvers are used, as in ``M3IQM``.

    """

    def __init__(  # noqa: PLR0913
        self,
        system: IQMBackend,
        pulse_settings: Mapping[str, Any] | None = None,
        active_reset_cycles: int | None = None,
        use_timeslot: bool = False,
        timeout_secs: float = 0.0,
        iter_threshold: int = 4096,
    ):
        super().__init__(system, iter_threshold)
        self.pulse_settings = dict(pulse_settings or {})
        self.active_reset_cycles = active_reset_cycles
        self.use_timeslot = use_timeslot
        self.timeout_secs = timeout_secs
        self._cal_qubits: list[int] = []

    def _prepare_calibration_circuits(
        self,
        qubits: list,
        num_cal_qubits: int,
        method: str,
        initial_reset: bool,
    ) -> tuple[Any | list[Any], list[Any], int]:
        """Prepare REM calibration circuits and keep the calibrated qubits for the pulse compiler."""
        self._cal_qubits = list(qubits)
        return super()._prepare_calibration_circuits(qubits, num_cal_qubits, method, initial_reset)

    def _submit_calibration_jobs(
        self,
        trans_qcs: list,
        shots: int,
        cal_id: str | None,  # noqa: ARG002 - the calibration set is the one the backend compiles with
        circuit_compilation_options: CircuitCompilationOptions | None = None,  # noqa: ARG002 - set by pulse_settings
    ) -> list:
        """Compile the calibration circuits to pulse level, execute them and wait for the results.

        Args:
            trans_qcs: List of calibration circuits to execute.
            shots: Number of shots per circuit.
            cal_id: Not used; the backend's calibration set is used, as for the benchmark circuits.
            circuit_compilation_options: Not used; reset and other compiler options come from ``pulse_settings``
                and ``active_reset_cycles``.

        Returns:
            A list with one finished job, whose result holds the counts of every calibration circuit.

        Raises:
            M3Error: If system is not set.

        """
        if self.system is None:
            raise M3Error("System is not set")

        circuits = transpile(trans_qcs, self.system, optimization_level=0)
        compiler, settings_tree = prepare_pulse_compiler(
            self.system, self._cal_qubits, circuits, shots, self.pulse_settings, self.active_reset_cycles
        )
        run_definition, context, _ = compile_pulse(self.system, compiler, settings_tree, circuits)
        job_tracker, _ = submit_pulse(self.system, run_definition, context, use_timeslot=self.use_timeslot)
        qcvv_logger.info(f"REM: {len(circuits)} calibration circuits to be executed at pulse level!")
        counts, _, _ = retrieve_pulse(job_tracker, shots, timeout_secs=self.timeout_secs)

        return [_PulseCalibrationJob(counts)]


def readout_error_m3(counts: dict[str, float], mit: M3IQM, qubits: Iterable) -> dict[str, float]:
    """Counts processor using M3IQM for readout error mitigation.

    The qubits argument can be either a dictionary coming from `mthree.utils.final_measurement_mapping`
    or an array like the initial layout coming from [backend.qubit_name_to_index(name)]

    returns a dictionary of quasiprobabilties.

    NOTE: we could also pass a list of input counts and then this would return a list of quasiprobabilities.
    This would not work out of the box for us since we need the annotations of either Dict or List (not Union).
    """
    return mit.apply_correction(counts, qubits)


@timeit
def apply_readout_error_mitigation(
    backend_arg: str | IQMBackend | IQMFacadeBackend,
    transpiled_circuits: list[QuantumCircuit],
    counts: list[dict[str, int]],
    mit_shots: int = 1000,
    circuit_compilation_options: CircuitCompilationOptions | None = None,
    mit: M3IQM | None = None,
) -> list[tuple[Any, Any]] | list[tuple[QuasiCollection, list]] | list[QuasiCollection]:
    """Application of readout error mitigation to a list of counts.

    Args:
        backend_arg: The backend to calibrate an M3 mitigator against.
        transpiled_circuits: The list of transpiled quantum circuits.
        counts: The measurement counts corresponding to the transpiled circuits.
        mit_shots: Number of shots per readout error characterization circuit.
        circuit_compilation_options: Optional compilation options (e.g. active reset) to pass to ``backend.run()``.
        mit: The mitigator to calibrate, e.g. an ``M3IQMPulse``. Defaults to an ``M3IQM`` of the backend.

    Returns:
         A list of dictionaries with REM-corrected quasiprobabilities for each outcome.

    """
    # M3IQM uses mthree.mitigation, which for some reason displays way too many INFO messages
    with temporary_log_level(logging.getLogger("mthree"), level=logging.WARNING):
        backend = get_iqm_backend(backend_arg) if isinstance(backend_arg, str) else backend_arg

        # Initialize with the given system and get calibration data
        qubits_rem = [final_measurement_mapping(c) for c in transpiled_circuits]

    mit = M3IQM(backend)
    mit.cals_from_system(qubits_rem, shots=mit_shots, circuit_compilation_options=circuit_compilation_options)
    # Apply the REM correction to the given measured counts
    rem_quasidistro = [mit.apply_correction(c, q) for c, q in zip(counts, qubits_rem, strict=True)]
    logging.getLogger().setLevel(logging.INFO)

    return rem_quasidistro
