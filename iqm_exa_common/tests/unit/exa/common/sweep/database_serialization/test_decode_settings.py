#  ********************************************************************************
#    _____  ____ _
#   / _ \ \/ / _` |  Framework for control
#  |  __/>  < (_| |  and measurement of
#   \___/_/\_\__,_|  superconducting qubits
#
#  Copyright (c) 2019-2022 IQM Finland Oy.
#  All rights reserved. Confidential and proprietary.
#
#  Distribution or reproduction of any information contained herein
#  is prohibited without IQM Finland Oy’s prior written permission.
#  ********************************************************************************
import json

from exa.common.data.parameter import Parameter, Setting
from exa.common.data.setting_node import SettingNode
from exa.common.sweep.database_serialization import decode_settings


def test_decode_from_string_is_correct():
    settings = SettingNode("root", qubit_1=SettingNode("qubit_1"), frequency=Setting(Parameter("frequency"), 2))
    settings_dict = {
        "name": "root",
        "settings": {
            "frequency": {
                "parameter": {
                    "name": "frequency",
                    "label": "frequency",
                    "unit": "",
                    "data_type": 1,
                    "collection_type": 0,
                },
                "value": 2,
            }
        },
        "subtrees": {"qubit_1": {"name": "qubit_1", "settings": {}, "subtrees": {}}},
    }
    result = decode_settings(json.dumps(settings_dict))
    assert result == settings
