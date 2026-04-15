# Copyright 2024-2025 IQM
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
from typing import Any
from uuid import UUID

from iqm.iqm_server_client.models import JobData, JobStatus
from mockito import when
import numpy as np
import pytest
from qiskit import QuantumCircuit

from iqm.pulla.pulla import Playlist, SettingNode, SweepJob
from iqm.pulla.utils_qiskit import qiskit_circuits_to_pulla, qiskit_to_pulla
from iqm.pulse import CircuitOperation

pytestmark = pytest.mark.usefixtures("unstub")


def test_qiskit_circuits_to_pulla(qiskit_backend_spark):
    """Test that Qiskit circuits with nonnative gates can be converted to the Pulla format."""

    nonnative_gate = QuantumCircuit(3, name="nonnative").to_gate()

    qc = QuantumCircuit(5, 5)
    qc.x(0)
    qc.rx(0.1, 0)
    qc.r(0.1, 0.2, 1)
    qc.barrier(0, 1, 2, 3)
    qc.cz(0, 2)
    qc.append(nonnative_gate, [1, 2, 4])
    qc.measure_all()

    qubit_index_to_name = qiskit_backend_spark._idx_to_qb

    with pytest.raises(ValueError, match="is not natively supported"):
        qiskit_circuits_to_pulla(qc, qubit_index_to_name)

    circuits = qiskit_circuits_to_pulla(qc, qubit_index_to_name, {"nonnative"})

    assert len(circuits) == 1
    c = circuits[0]
    assert c.name == qc.name
    assert len(c.instructions) == len(qc)
    assert c.instructions[5].name == "nonnative"


def test_qiskit_to_pulla(pulla_on_spark, qiskit_backend_spark):
    """Test that Qiskit circuits can be converted to the Pulla format."""
    T = 2 * np.pi
    qc = QuantumCircuit(3, 3)
    qc.x(0)
    qc.rx(0.1 * T, 0)
    qc.r(0.1 * T, 0.2 * T, 1)
    qc.barrier(0, 1)
    qc.cz(1, 2)
    qc.measure_all()
    converted, _ = qiskit_to_pulla(pulla_on_spark, qiskit_backend_spark, qc)

    assert converted[0].instructions == (
        CircuitOperation("prx", ("QB1",), args={"angle": np.pi, "phase": 0.0}),
        CircuitOperation("prx", ("QB1",), args={"angle": 0.2 * np.pi, "phase": 0.0}),
        CircuitOperation("prx", ("QB2",), args={"angle": 0.2 * np.pi, "phase": 0.4 * np.pi}),
        CircuitOperation("barrier", ("QB1", "QB2"), args={}),
        CircuitOperation("cz", ("QB2", "QB3"), args={}),
        CircuitOperation("barrier", ("QB1", "QB2", "QB3"), args={}),
        CircuitOperation("measure", ("QB1",), args={"key": "meas_3_1_0"}),
        CircuitOperation("measure", ("QB2",), args={"key": "meas_3_1_1"}),
        CircuitOperation("measure", ("QB3",), args={"key": "meas_3_1_2"}),
    )


def test_pulla_backend_consistency(pulla_on_spark, pulla_backend_spark):
    """
    Test that Pulla backend's compilation environment is equivalent to the corresponding standard compiler.
    """
    standard_compiler = pulla_on_spark.get_standard_compiler()
    assert standard_compiler._calibration_set_values == pulla_backend_spark.compiler._calibration_set_values
    assert standard_compiler.options == pulla_backend_spark.compiler.options

    # we can't compare stages directly because get_standard_stages() returns a copy, so let's just compare names
    for couple in list(zip(standard_compiler.stages, pulla_backend_spark.compiler.stages)):
        assert couple[0].name == couple[1].name


def test_pulla_backend_compilation(pulla_backend_spark, monkeypatch):
    """Test that Pulla backend compiles a simple circuit."""
    T = 2 * np.pi
    qc = QuantumCircuit(3, 3)
    qc.x(0)
    qc.rx(0.1 * T, 0)
    qc.r(0.1 * T, 0.2 * T, 1)
    qc.barrier(0, 1)
    qc.cz(1, 2)
    qc.measure_all()

    def mocked_submit_playlist(*args, **kwargs):
        raise RuntimeError("Compilation succeeded but execution failed")

    monkeypatch.setattr(
        pulla_backend_spark.pulla,
        "submit_playlist",
        mocked_submit_playlist,
    )

    with pytest.raises(RuntimeError, match="Compilation succeeded but execution failed"):
        pulla_backend_spark.run(qc, shots=1)


def test_pulla_backend_run_result(pulla_backend_spark, monkeypatch):
    """
    Test that Pulla backend.run() returns a job with correct data.
    """

    qc = QuantumCircuit(3, 3)
    qc.measure_all()

    job_id = UUID("a892faba-f1b7-4d13-a9b4-f3fb3b32e2e3")

    def mocked_submit_playlist(
        playlist: Playlist,
        settings: SettingNode,
        *,
        context: dict[str, Any],
        use_timeslot: bool = False,
    ):
        return SweepJob(
            data=JobData(
                id=job_id,
                status=JobStatus.COMPLETED,
            ),
            _pulla=pulla_backend_spark.pulla,
            _context=context,
            _result=[{"meas_3_0_0": [[0], [0], [0]], "meas_3_0_1": [[0], [1], [0]], "meas_3_0_2": [[0], [1], [1]]}],
        )

    monkeypatch.setattr(
        pulla_backend_spark.pulla,
        "submit_playlist",
        mocked_submit_playlist,
    )
    when(SweepJob).wait_for_completion(...).thenReturn(JobStatus.COMPLETED)

    job = pulla_backend_spark.run(qc, shots=3)
    results = job.result().results[0]

    assert job.job_id() == str(job_id)
    assert results.data.memory == ["000", "110", "100"]
