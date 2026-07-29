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
import pytest

from iqm.cpc.core.config import ExperimentConfiguration

COUPLER_1_2 = "TC-1-2"
DUT_LABEL = "M138_W36_A22_N05"
QUBIT_1 = "QB1"
QUBIT_2 = "QB2"


def configuration_with_components():
    components = {
        QUBIT_1: {"readout": "QB1__readout", "flux": "flux_1"},
        QUBIT_2: {"readout": "QB2__readout", "flux": "flux_2", "drive": "drive_2"},
        COUPLER_1_2: {"flux": "flux_coupler_1_2"},
    }
    return ExperimentConfiguration({"dut_label": DUT_LABEL, "components": components})


def test_one_input_str_argument():
    has_readout = configuration_with_components().filter_components("readout")
    assert has_readout == [QUBIT_1, QUBIT_2]


def test_one_input_callable_argument():
    no_drive = configuration_with_components().filter_components(lambda c: c != "drive")
    assert no_drive == [QUBIT_1, COUPLER_1_2]


def test_one_input_list_argument():
    no_readout_but_flux = configuration_with_components().filter_components([lambda c: c != "readout", "flux"])
    assert no_readout_but_flux == [COUPLER_1_2]


def test_several_input_arguments():
    group1, group2, group3 = configuration_with_components().filter_components(
        "readout", [lambda c: "u" in c], ["readout", "flux", "drive"]
    )
    assert group1 == [QUBIT_1, QUBIT_2]
    assert group2 == [QUBIT_1, COUPLER_1_2]
    assert group3 == [QUBIT_2]


def test_empty_list_argument_returns_all_components():
    all_components = configuration_with_components().filter_components([])
    assert all_components == [QUBIT_1, QUBIT_2, COUPLER_1_2]


def test_no_arguments_raises_value_error():
    with pytest.raises(ValueError, match="No filtering conditions"):
        configuration_with_components().filter_components()
