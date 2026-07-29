# Copyright 2021-2023 IQM client developers
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
"""Tests for the IQM client models."""

from iqm.iqm_client.models import CircuitJobParameters
import pytest

from iqm.station_control.interface.models import CircuitBatch, DDMode, HeraldingMode


@pytest.fixture
def shots():
    return 1024


@pytest.fixture
def circuits_batch():
    return CircuitBatch()


@pytest.fixture
def heralding_mode():
    return HeraldingMode.ZEROS


@pytest.fixture
def dd_mode():
    return DDMode.ENABLED


@pytest.mark.parametrize(
    "metadata_factory",
    [
        lambda shots, heralding_mode, dd_mode: CircuitJobParameters(
            shots=shots, heralding_mode=heralding_mode, dd_mode=dd_mode
        )
    ],
)
def test_circuit_job_parameters(metadata_factory, shots, heralding_mode, dd_mode):
    """Tests different modes of CircuitJobParameters class initialization."""
    metadata = metadata_factory(shots, heralding_mode, dd_mode)
    assert metadata.shots == shots
    assert metadata.heralding_mode == heralding_mode
    assert metadata.dd_mode == dd_mode
    assert metadata.dd_strategy is None
