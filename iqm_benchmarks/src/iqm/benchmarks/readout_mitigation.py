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

from collections.abc import Iterable
import logging
from math import ceil
import threading
from typing import Any, cast
import warnings

from iqm.benchmarks.logging_config import qcvv_logger
from iqm.benchmarks.utils import get_iqm_backend, timeit
from iqm.qiskit_iqm import IQMCircuit as QuantumCircuit
from iqm.qiskit_iqm.iqm_provider import IQMBackend, IQMFacadeBackend
import mthree  # type: ignore
from mthree.circuits import (  # type: ignore
    _marg_meas_states,
    _tensor_meas_states,
    balanced_cal_circuits,
)
from mthree.classes import QuasiCollection  # type: ignore
from mthree.exceptions import M3Error  # type: ignore
from mthree.generators import HadamardGenerator  # type: ignore
from mthree.mitigation import _job_thread  # type: ignore
from mthree.utils import final_measurement_mapping  # type: ignore
import numpy as np
from qiskit import transpile

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

    def _custom_system_info(self, system: IQMBackend | IQMFacadeBackend) -> dict[str, Any]:
        """Custom system_info implementation for IQM systems."""
        return {
            "num_qubits": system.num_qubits,
            "max_shots": 10000,
            "max_circuits": 100,
            "simulator": any("Fake" in key for key in system.__dict__.keys()),
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
    ) -> list:
        """Submit calibration circuits as jobs to the backend.

        Args:
            trans_qcs: List of transpiled calibration circuits to execute.
            shots: Number of shots per circuit.
            cal_id: Optional calibration set ID for job submission.

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
        for circs in circs_list:
            transpiled_circuit = transpile(circs, self.system, optimization_level=0)
            if cal_id is None:
                _job = self.system.run(transpiled_circuit, shots=shots)
            else:
                _job = self.system.run(
                    transpiled_circuit,
                    shots=shots,
                    calibration_set_id=cal_id,
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
    ) -> None:
        """Grab missing calibration data from backend.

        Args:
            qubits: List of qubit indices to calibrate.
            shots: Number of shots per calibration circuit. If None, uses max_shots defined in system or 10000.
            method: Calibration method - 'independent', 'balanced', or 'marginal'.
            initial_reset: Whether to include initial reset operations in calibration circuits.
            async_cal: If True, run calibration in a separate thread asynchronously.
            cal_id: Optional calibration set ID for job submission.

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
        jobs = self._submit_calibration_jobs(trans_qcs, cal_shots, cal_id)

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
) -> list[tuple[Any, Any]] | list[tuple[QuasiCollection, list]] | list[QuasiCollection]:
    """Application of readout error mitigation to a list of counts.

    Args:
        backend_arg: The backend to calibrate an M3 mitigator against.
        transpiled_circuits: The list of transpiled quantum circuits.
        counts: The measurement counts corresponding to the transpiled circuits.
        mit_shots: Number of shots per readout error characterization circuit.

    Returns:
         A list of dictionaries with REM-corrected quasiprobabilities for each outcome.

    """
    # M3IQM uses mthree.mitigation, which for some reason displays way too many INFO messages
    logging.getLogger().setLevel(logging.WARN)
    if isinstance(backend_arg, str):
        backend = get_iqm_backend(backend_arg)
    else:
        backend = backend_arg

    # Initialize with the given system and get calibration data
    qubits_rem = [final_measurement_mapping(c) for c in transpiled_circuits]

    mit = M3IQM(backend)
    mit.cals_from_system(qubits_rem, shots=mit_shots)
    # Apply the REM correction to the given measured counts
    rem_quasidistro = [mit.apply_correction(c, q) for c, q in zip(counts, qubits_rem, strict=True)]
    logging.getLogger().setLevel(logging.INFO)

    return rem_quasidistro
