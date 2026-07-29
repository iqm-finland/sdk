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
from exa.common.sweep.database_serialization import decode_return_parameters


def test_decode_parallel_sweep_encoded(return_parameters5):
    back = decode_return_parameters(return_parameters5[1])
    assert back == {parameter.name: value for parameter, value in return_parameters5[0].items()}


def test_decode_nd_sweep_encoded(return_parameters6):
    back = decode_return_parameters(return_parameters6[1])
    assert back == {parameter.name: value for parameter, value in return_parameters6[0].items()}


def test_decode_none_encoded(return_parameters7):
    back = decode_return_parameters(return_parameters7[1])
    assert back == {parameter.name: value for parameter, value in return_parameters7[0].items()}
