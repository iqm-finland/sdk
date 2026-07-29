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
from exa.common.data.parameter import Parameter, Sweep
from exa.common.sweep.util import convert_sweeps_to_list_of_tuples


def test_sweep_item_is_converted_to_tuple_item():
    sweep = Sweep(parameter=Parameter("param1"), data=[1, 2])
    converted_sweeps = convert_sweeps_to_list_of_tuples([sweep])

    assert converted_sweeps == [(sweep,)]


def test_tuple_item_is_preserved():
    sweep = Sweep(parameter=Parameter("param1"), data=[1, 2])
    converted_sweeps = convert_sweeps_to_list_of_tuples([(sweep,)])

    assert converted_sweeps == [(sweep,)]


def test_tuple_item_of_multiple_sweeps_is_preserved():
    sweep1 = Sweep(parameter=Parameter("param1"), data=[1, 2])
    sweep2 = Sweep(parameter=Parameter("param2"), data=[1, 2])
    converted_sweeps = convert_sweeps_to_list_of_tuples([(sweep1, sweep2)])

    # Not only the original structure, but the identity of the sweep objects are preserved
    assert converted_sweeps == [(sweep1, sweep2)]


def test_mixed_list_case():
    sweep1 = Sweep(parameter=Parameter("param1"), data=[1, 2])
    sweep2 = Sweep(parameter=Parameter("param2"), data=[1, 2])
    sweep3 = Sweep(parameter=Parameter("param3"), data=[1, 2])
    converted_sweeps = convert_sweeps_to_list_of_tuples([(sweep1, sweep2), sweep3])

    # The input sweeps have been re-packaged, but the identity of the sweep objects are preserved
    assert converted_sweeps == [(sweep1, sweep2), (sweep3,)]
