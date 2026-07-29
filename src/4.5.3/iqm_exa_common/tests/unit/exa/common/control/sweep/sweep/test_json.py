#  ********************************************************************************
#  Copyright (c) 2019-2024 IQM Finland Oy.
#  All rights reserved. Confidential and proprietary.
#
#  Distribution or reproduction of any information contained herein
#  is prohibited without IQM Finland Oy’s prior written permission.
#  ********************************************************************************
import json

from exa.common.data.parameter import DataType, Parameter, Sweep


def test_json():
    parameter = Parameter(name="test")
    sweep = Sweep(parameter=parameter, data=[1, 2])
    scalar_parameter = Parameter(name="test")
    fixed_sweep = Sweep(parameter=scalar_parameter, data=[1, 2])
    assert sweep.model_dump_json() == fixed_sweep.model_dump_json()


def test_json_with_complex_number():
    parameter = Parameter(name="test", label="test", unit="", data_type=DataType.COMPLEX)
    sweep = Sweep(parameter=parameter, data=[0.5 + 0.6j, 0.4 + 5j])
    encoded_data = [
        {"__complex__": "true", "real": 0.5, "imag": 0.6},
        {"__complex__": "true", "real": 0.4, "imag": 5.0},
    ]
    sweep_json = sweep.model_dump_json()

    assert json.loads(sweep_json)["data"] == encoded_data
    sweep = Sweep(**json.loads(sweep_json))
    assert sweep.data == [0.5 + 0.6j, 0.4 + 5j]
