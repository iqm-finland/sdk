# Copyright 2025 IQM
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
import pytest

from iqm.station_control.interface.models import JobExecutorStatus


def test_statuses_can_be_compared():
    assert JobExecutorStatus.READY > JobExecutorStatus.PENDING_EXECUTION
    assert JobExecutorStatus.RECEIVED < JobExecutorStatus.COMPILATION_ENDED
    assert JobExecutorStatus.FAILED == JobExecutorStatus.FAILED
    assert JobExecutorStatus.FAILED >= JobExecutorStatus.EXECUTION_ENDED
    assert JobExecutorStatus.EXECUTION_STARTED <= JobExecutorStatus.EXECUTION_ENDED
    assert JobExecutorStatus.COMPILATION_STARTED >= JobExecutorStatus.COMPILATION_STARTED
    assert JobExecutorStatus.VALIDATION_STARTED <= JobExecutorStatus.VALIDATION_STARTED


def test_statuses_can_be_compared_to_strings():
    assert str(JobExecutorStatus.READY) > JobExecutorStatus.PENDING_EXECUTION
    assert str(JobExecutorStatus.RECEIVED) < JobExecutorStatus.COMPILATION_ENDED
    assert str(JobExecutorStatus.FAILED) == JobExecutorStatus.FAILED
    assert str(JobExecutorStatus.FAILED) >= JobExecutorStatus.EXECUTION_ENDED
    assert str(JobExecutorStatus.EXECUTION_STARTED) <= JobExecutorStatus.EXECUTION_ENDED
    assert str(JobExecutorStatus.COMPILATION_STARTED) >= JobExecutorStatus.COMPILATION_STARTED
    assert str(JobExecutorStatus.VALIDATION_STARTED) <= JobExecutorStatus.VALIDATION_STARTED


def test_job_executor_status_unknown_status_raises_error():
    with pytest.raises(ValueError):
        _ = JobExecutorStatus("foo")
