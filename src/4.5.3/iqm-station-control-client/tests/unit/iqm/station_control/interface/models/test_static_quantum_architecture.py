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

import pytest

from iqm.station_control.interface.models import StaticQuantumArchitecture


def test_sqa_deserialization():
    sqa = StaticQuantumArchitecture(
        dut_label="M138_W0_A22_Z99",
        qubits=["QB1", "QB2"],
        computational_resonators=["CR1"],
        connectivity=[("QB1", "QB2"), ("QB1", "CR1")],
    )

    dqa_json = sqa.model_dump_json()
    dqa_reconstructed = StaticQuantumArchitecture(**json.loads(dqa_json))

    assert dqa_reconstructed == sqa


def test_dut_label_optional_and_deprecated():
    with pytest.warns(
        DeprecationWarning, match="Missing 'dut_label'. This field will become REQUIRED in a future release."
    ):
        sqa = StaticQuantumArchitecture(
            qubits=["QB1", "QB2"],
            computational_resonators=["CR1"],
            connectivity=[("QB1", "QB2"), ("QB1", "CR1")],
        )

    assert sqa.dut_label is None
