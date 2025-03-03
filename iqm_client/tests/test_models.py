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
import json
from uuid import UUID

import pytest

from iqm.iqm_client.models import (
    CircuitBatch,
    DDMode,
    DynamicQuantumArchitecture,
    GateImplementationInfo,
    GateInfo,
    HeraldingMode,
    JobParameters,
    Metadata,
    RunRequest,
)


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
    'metadata_factory',
    [
        # V1 and RESONANCE_V1
        lambda shots, circuits_batch, heralding_mode, dd_mode: Metadata(
            request=RunRequest(
                circuits=circuits_batch,
                shots=shots,
                heralding_mode=heralding_mode,
                dd_mode=dd_mode,
            )
        ),
        # V2
        lambda shots, circuits_batch, heralding_mode, dd_mode: Metadata(
            parameters=JobParameters(shots=shots, heralding_mode=heralding_mode, dd_mode=dd_mode),
            circuits_batch=circuits_batch,
        ),
    ],
)
def test_metadata(metadata_factory, shots, circuits_batch, heralding_mode, dd_mode):
    """Tests different modes of Metadata class initialization."""
    metadata = metadata_factory(shots, circuits_batch, heralding_mode, dd_mode)
    assert metadata.shots == shots
    assert metadata.circuits == circuits_batch
    assert metadata.heralding_mode == heralding_mode
    assert metadata.dd_mode == dd_mode
    assert metadata.dd_strategy is None


def test_gate_info_loci():
    gate_info = GateInfo(
        implementations={
            'tgss': GateImplementationInfo(loci=(('QB1', 'QB2'), ('QB2', 'QB3'), ('QB10', 'QB2'), ('QB3', 'COMPR2'))),
            'crf': GateImplementationInfo(loci=(('QB3', 'QB1'), ('QB2', 'QB3'), ('QB2', 'COMPR1'), ('QB3', 'COMPR1'))),
        },
        default_implementation='tgss',
        override_default_implementation={},
    )
    assert gate_info.loci == (
        ('QB1', 'QB2'),
        ('QB2', 'COMPR1'),
        ('QB2', 'QB3'),
        ('QB3', 'COMPR1'),
        ('QB3', 'COMPR2'),
        ('QB3', 'QB1'),
        ('QB10', 'QB2'),
    )


def test_dqa_components():
    dqa = DynamicQuantumArchitecture(
        calibration_set_id=UUID('59478539-dcef-4b2e-80c8-122d7ec3fc89'),
        qubits=['QB3', 'QB1', 'QB11', 'QB20', 'QB2', 'QB10'],
        computational_resonators=['COMPR2', 'COMPR1'],
        gates={},
    )
    assert dqa.components == ('COMPR1', 'COMPR2', 'QB1', 'QB2', 'QB3', 'QB10', 'QB11', 'QB20')


def test_dqa_deserialization():
    dqa = DynamicQuantumArchitecture(
        calibration_set_id=UUID('59478539-dcef-4b2e-80c8-122d7ec3fc89'),
        qubits=['QB1', 'QB2'],
        computational_resonators=['COMP_R'],
        gates={
            'cz': GateInfo(
                implementations={
                    'tgss': GateImplementationInfo(
                        loci=(('QB1', 'QB2'), ('QB1', 'COMP_R')),
                    ),
                    'crf': GateImplementationInfo(loci=(('QB2', 'COMPR'),)),
                },
                default_implementation='tgss',
                override_default_implementation={('QB2', 'COMPR'): 'crf'},
            ),
        },
    )

    dqa_json = dqa.model_dump_json()
    dqa_reconstructed = DynamicQuantumArchitecture(**json.loads(dqa_json))

    assert dqa_reconstructed == dqa
