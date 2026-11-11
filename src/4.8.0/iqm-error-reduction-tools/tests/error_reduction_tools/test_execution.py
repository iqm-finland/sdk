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

from uuid import UUID

from iqm.error_reduction_tools import execution
from iqm.error_reduction_tools.execution import ResonanceExecutor, SubmittedJob
from iqm.iqm_client import RunJobTracker
from mockito import ANY, mock, unstub, verify, when
import pytest

from .builders import circuit

JOB_ID = UUID("12345678-1234-5678-1234-567812345678")


class TestResonanceExecutor:
    def teardown_method(self):
        unstub()

    def _client(self, tracker):
        compiler = mock()
        settings = mock()
        when(compiler).get_settings(...).thenReturn(settings)
        when(settings).set_shots(...).thenReturn(None)
        when(compiler).compile(...).thenReturn(("definition", {"shots": 0}))
        client = mock()
        when(client).get_standard_compiler().thenReturn(compiler)
        when(client).submit_playlist(...).thenReturn(tracker)
        return client, settings

    def test_submit_compiles_with_the_requested_shots(self):
        """Test that submit() compiles with the requested shots and returns the job."""
        tracker = mock({"job_id": JOB_ID}, spec=RunJobTracker)
        client, settings = self._client(tracker)
        executor = ResonanceExecutor(client)
        job = executor.submit([circuit([], ["QB1"])] * 3, 123)
        assert job == SubmittedJob(job_id=str(JOB_ID), num_circuits=3, shots=123, compiler_context=None)
        verify(settings).set_shots(123)

    def test_compilation_options_apply_to_circuit_jobs_only(self):
        """Test that compilation options are used for circuit jobs and not for calibration jobs."""
        options = {"DDStrategy": "placeholder"}
        client, _ = self._client(mock({"job_id": JOB_ID}, spec=RunJobTracker))
        executor = ResonanceExecutor(client, options)
        executor.submit([circuit([], ["QB1"])], 10)
        executor.submit([circuit([], ["QB1"])], 10, calibration=True)
        compiler = client.get_standard_compiler()
        verify(compiler, times=1).compile(circuits=ANY, settings=ANY, context=options)
        verify(compiler, times=1).compile(circuits=ANY, settings=ANY, context=None)

    def test_collect_strips_register_spaces_and_keeps_order(self):
        """Test that collect() removes spaces from bitstrings and keeps the circuit order."""
        tracker = mock({"job_id": JOB_ID}, spec=RunJobTracker)
        client, _ = self._client(tracker)
        results = mock()
        when(results).get_counts(0).thenReturn({"0 1": 5})
        when(results).get_counts(1).thenReturn({"1 1": 7})
        when(execution).wait_for_results(tracker, 10).thenReturn(results)
        executor = ResonanceExecutor(client)
        job = executor.submit([circuit([], ["QB1", "QB2"])] * 2, 10)
        assert executor.collect(job) == [{"01": 5}, {"11": 7}]

    def test_restored_job_without_compiler_context_is_rejected(self):
        """Test that a restored job without a compiler context is rejected."""
        client = mock()
        when(client).get_job(JOB_ID).thenReturn(mock(spec=RunJobTracker))
        executor = ResonanceExecutor(client)
        with pytest.raises(ValueError, match="no compiler context"):
            executor.collect(SubmittedJob(job_id=str(JOB_ID), num_circuits=1, shots=1))

    def test_restored_job_that_is_not_a_playlist_job_is_rejected(self):
        """Test that a restored job that is not a playlist job is rejected."""
        client = mock()
        when(client).get_job(JOB_ID).thenReturn(mock())
        with pytest.raises(TypeError, match="not a playlist job"):
            ResonanceExecutor(client).collect(SubmittedJob(job_id=str(JOB_ID), num_circuits=1, shots=1))
