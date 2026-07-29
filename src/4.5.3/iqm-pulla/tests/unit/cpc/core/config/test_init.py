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
from exa.common.control.sweep.option import StartStopOptions
from iqm.cpc.core.config import ComponentGroup, ExperimentConfiguration

PROBE_LINE_1 = "PL_RO-1"
DUT_LABEL = "M138_W36_A22_N05"
QUBIT_1 = "QB1"


def test_defaults():
    conf = ExperimentConfiguration({})

    assert conf.user == ""
    assert conf.dut_label == ""
    assert conf.components == {}
    assert conf.controllers == {}


def test_basic_info_is_parsed():
    settings = {"setting_1": 3}
    components = {QUBIT_1: {"readout": "QB1__readout"}}
    basic = {
        "user": "Jon Doe",
        "dut_label": "M138_W36_A22_N05",
        "controllers": settings,
        "components": components,
    }
    conf = ExperimentConfiguration(basic)

    assert conf.user == "Jon Doe"
    assert conf.dut_label == "M138_W36_A22_N05"
    assert conf.components == components
    assert conf.controllers == settings


def test_experiment_settings_are_parsed():
    settings = {"setting_1": 3}
    data = {
        "dut_label": DUT_LABEL,
        "some_experiment": {"controllers": settings},
    }
    conf = ExperimentConfiguration(data)
    assert conf.some_experiment.controllers == settings


def test_sweep_data_is_parsed():
    data = {
        "dut_label": DUT_LABEL,
        "some_experiment": {
            "sweeps": [
                {
                    "name": "sweep_name",
                    "parameter": "sweep_par_1",
                    "type": "linear",
                    "start": 6.375e9,
                    "stop": 6.425e9,
                    "count": 21,
                },
            ]
        },
    }
    conf = ExperimentConfiguration(data)

    assert conf.some_experiment.sweeps[0].parameter == "sweep_par_1"
    assert conf.some_experiment.sweeps[0].options == StartStopOptions(6.375e9, 6.425e9, count=21)


def test_component_groups_are_parsed():
    conf = ExperimentConfiguration(
        {
            "dut_label": DUT_LABEL,
            "component_groups": {PROBE_LINE_1: {"member_names": [QUBIT_1], "readout": "QB1__readout"}},
        }
    )

    assert conf.component_groups[PROBE_LINE_1] == ComponentGroup([QUBIT_1], {"readout": "QB1__readout"})
