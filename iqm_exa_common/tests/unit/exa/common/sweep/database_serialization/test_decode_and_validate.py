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
import pytest

from exa.common.data.parameter import Sweep
from exa.common.sweep.database_serialization import decode_and_validate_sweeps


@pytest.fixture
def encoded_sweep():
    return (
        '{"parameter": {"name": "param1", "label": "param1",'
        + ' "unit": "", "data_type": 1, "collection_type": 0}, "data": [1.0, 2.0]}'
    )


def test_list_of_sweeps_and_tuples_is_decoded(encoded_sweep):
    sweeps_json = "["
    sweeps_json += encoded_sweep
    sweeps_json += ", "
    sweeps_json += '{"__tuple__": "true", "data": ['
    sweeps_json += encoded_sweep
    sweeps_json += "]}]"
    nested_structure = decode_and_validate_sweeps(sweeps_json)
    assert isinstance(nested_structure[0], Sweep)
    assert isinstance(nested_structure[1], tuple)
    assert isinstance(nested_structure[1][0], Sweep)


def test_outer_type_must_be_list():
    with pytest.raises(ValueError, match=r"^Outer type is not list.*"):
        decode_and_validate_sweeps("1")


def test_list_element_type_must_be_tuple_or_sweep():
    with pytest.raises(ValueError, match=r"^list must contain either Sweep or Tuple.*"):
        decode_and_validate_sweeps("[1]")


def test_list_of_tuples_must_contain_sweeps():
    with pytest.raises(ValueError, match=r"^list of Sweeps must contain tuples*"):
        decode_and_validate_sweeps('[{"__tuple__": "true", "data": [1]}]')
