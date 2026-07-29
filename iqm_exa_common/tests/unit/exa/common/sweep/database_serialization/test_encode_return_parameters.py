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


import random
import uuid

from exa.common.data.parameter import Parameter
from exa.common.sweep.database_serialization import decode_return_parameters, encode_return_parameters


def test_encoded_parallel_sweep_can_be_loaded_back(return_parameters5):
    json_str = encode_return_parameters(return_parameters5[0])
    back = decode_return_parameters(json_str)
    assert back == {parameter.name: value for parameter, value in return_parameters5[0].items()}


def test_decode_is_inverse_of_encode_for_parallel_sweeps(return_parameters5):
    json_str = encode_return_parameters(return_parameters5[0])
    back = decode_return_parameters(json_str)
    assert back == {parameter.name: value for parameter, value in return_parameters5[0].items()}


def test_decode_is_inverse_of_encode_for_nd_sweeps(return_parameters6):
    json_str = encode_return_parameters(return_parameters6[0])
    back = decode_return_parameters(json_str)
    assert back == {parameter.name: value for parameter, value in return_parameters6[0].items()}


def test_decode_is_inverse_of_encode_for_none(return_parameters7):
    json_str = encode_return_parameters(return_parameters7[0])
    back = decode_return_parameters(json_str)
    assert back == {parameter.name: value for parameter, value in return_parameters7[0].items()}


def test_decode_is_inverse_of_encode_for_empty_list(return_parameters8):
    json_str = encode_return_parameters(return_parameters8[0])
    back = decode_return_parameters(json_str)
    assert back == {parameter.name: value for parameter, value in return_parameters8[0].items()}


def test_insertion_order_of_dict_is_preserved_during_encode_decode_round():
    # Previously, with the old raw result table, determining the values for an individual return parameter was
    # dependent on the insertion order of the dictionary return_parameters.
    # Currently, spot results are persisted individually for each return parameter, and the order does not matter.
    # The order is persisted anyway in our implementation, because since Python 3.7, dict remembers its insertion order,
    # and we encode the dict to a JSON array as the outer type.

    # a bunch or return parameters, in a python structure that nails their
    # order no matter what dict/json one uses
    ordered_return_parameters = [(Parameter(name=str(uuid.uuid4()), unit="m"), random.random()) for _ in range(1000)]

    return_parameters = {return_parameter: value for return_parameter, value in ordered_return_parameters}

    json_str = encode_return_parameters(return_parameters)
    return_parameters_back = decode_return_parameters(json_str)

    ordered_return_parameters_back = [
        (return_parameter, value) for return_parameter, value in return_parameters_back.items()
    ]

    assert ordered_return_parameters_back == [(parameter.name, value) for parameter, value in ordered_return_parameters]
