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

import json
from uuid import UUID

from iqm.station_control.interface.models import DynamicQuantumArchitecture, GateImplementationInfo, GateInfo


def test_dqa_components():
    dqa = DynamicQuantumArchitecture(
        calibration_set_id=UUID("59478539-dcef-4b2e-80c8-122d7ec3fc89"),
        qubits=["QB3", "QB1", "QB11", "QB20", "QB2", "QB10"],
        computational_resonators=["COMPR2", "COMPR1"],
        gates={},
    )
    assert dqa.components == ("COMPR1", "COMPR2", "QB1", "QB2", "QB3", "QB10", "QB11", "QB20")


def test_dqa_deserialization():
    dqa = DynamicQuantumArchitecture(
        calibration_set_id=UUID("59478539-dcef-4b2e-80c8-122d7ec3fc89"),
        qubits=["QB1", "QB2"],
        computational_resonators=["CR"],
        gates={
            "cz": GateInfo(
                implementations={
                    "tgss": GateImplementationInfo(
                        loci=(("QB1", "QB2"), ("QB1", "CR"), ("QB2", "CR")),
                    ),
                    "crf": GateImplementationInfo(loci=(("QB2", "CR"),)),
                },
                default_implementation="tgss",
                override_default_implementation={("QB2", "CR"): "crf"},
            ),
        },
    )

    dqa_json = dqa.model_dump_json()
    dqa_reconstructed = DynamicQuantumArchitecture(**json.loads(dqa_json))

    assert dqa_reconstructed == dqa


def test_gate_info_loci():
    gate_info = GateInfo(
        implementations={
            "tgss": GateImplementationInfo(loci=(("QB1", "QB2"), ("QB2", "QB3"), ("QB10", "QB2"), ("QB3", "COMPR2"))),
            "crf": GateImplementationInfo(loci=(("QB3", "QB1"), ("QB2", "QB3"), ("QB2", "COMPR1"), ("QB3", "COMPR1"))),
        },
        default_implementation="tgss",
        override_default_implementation={},
    )
    assert gate_info.loci == (
        ("QB1", "QB2"),
        ("QB2", "COMPR1"),
        ("QB2", "QB3"),
        ("QB3", "COMPR1"),
        ("QB3", "COMPR2"),
        ("QB3", "QB1"),
        ("QB10", "QB2"),
    )
