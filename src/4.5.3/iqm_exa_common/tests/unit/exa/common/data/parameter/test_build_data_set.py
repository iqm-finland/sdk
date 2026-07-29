#  ********************************************************************************
#  Copyright (c) 2019-2024 IQM Finland Oy.
#  All rights reserved. Confidential and proprietary.
#
#  Distribution or reproduction of any information contained herein
#  is prohibited without IQM Finland Oy’s prior written permission.
#  ********************************************************************************
import numpy as np

from exa.common.control.sweep.option.start_stop_options import StartStopOptions
from exa.common.data.parameter import CollectionType, DataType, Parameter, Sweep


def test_with_one_sweep() -> None:
    # Using variable order and result shape which would be produced by sweeper
    sweep_parameter = Parameter("test", "Test", "", DataType.INT)
    sweep = Sweep(parameter=sweep_parameter, data=StartStopOptions(1, 10, 10).data)
    variables = [(sweep.parameter, sweep.data)]
    result_parameter = Parameter("result", "Result", "", DataType.COMPLEX, CollectionType.NDARRAY)
    results = []
    for x in sweep.data:
        results.append(np.asarray([x + 1j]))
    result = (result_parameter, np.concatenate(results).tolist())
    ds = Parameter.build_data_set(variables, result, {})
    assert ds.data_vars["result"][5] == 6 + 1j


def test_with_two_sweeps() -> None:
    # Using variable order and result shape which would be produced by sweeper
    sweep_parameter_1 = Parameter("test1", "Test1", "", DataType.INT)
    sweep1 = Sweep(parameter=sweep_parameter_1, data=StartStopOptions(1, 10, 10).data)
    sweep_parameter_2 = Parameter("test2", "Test2", "", DataType.INT)
    sweep2 = Sweep(parameter=sweep_parameter_2, data=StartStopOptions(1, 3, 3).data)
    variables = [(sweep1.parameter, sweep1.data), (sweep2.parameter, sweep2.data)]
    result_parameter = Parameter("result", "Result", "", DataType.COMPLEX, CollectionType.NDARRAY)
    results = []
    for x1 in sweep1.data:
        for x2 in sweep2.data:
            results.append(np.asarray([x1 + x2 * 1j]))
    result = (result_parameter, np.concatenate(results).tolist())
    ds = Parameter.build_data_set(variables, result, {})
    assert ds.data_vars["result"][5][2] == 9 + 2j


def test_with_three_sweeps() -> None:
    # Using variable order and result shape which would be produced by sweeper
    sweep_parameter_1 = Parameter("test1", "Test1", "", DataType.INT)
    sweep1 = Sweep(parameter=sweep_parameter_1, data=StartStopOptions(1, 10, 10).data)
    sweep_parameter_2 = Parameter("test2", "Test2", "", DataType.INT)
    sweep2 = Sweep(parameter=sweep_parameter_2, data=StartStopOptions(1, 3, 3).data)
    sweep_parameter_3 = Parameter("test3", "Test3", "", DataType.INT)
    sweep3 = Sweep(parameter=sweep_parameter_3, data=StartStopOptions(1, 100, 100).data)
    variables = [(sweep1.parameter, sweep1.data), (sweep2.parameter, sweep2.data), (sweep3.parameter, sweep3.data)]
    result_parameter = Parameter("result", "Result", "", DataType.COMPLEX, CollectionType.NDARRAY)
    results = []
    for x1 in sweep1.data:
        for x2 in sweep2.data:
            for x3 in sweep3.data:
                results.append(np.asarray([x1 + (x2 * x3) * 1j]))
    result = (result_parameter, np.concatenate(results).tolist())
    ds = Parameter.build_data_set(variables, result, {})
    assert ds.data_vars["result"][5][2][56] == 6 + 18j
    assert ds.data_vars["result"].data.shape == (10, 3, 100)
    assert "test1" in ds.coords
    assert "test2" in ds.coords
    assert "test3" in ds.coords


def test_with_mixed_sweeps() -> None:
    # Using variable order and result shape which would be produced by sweeper
    sweep_parameter_1 = Parameter("test1", "Test1", "", DataType.INT)
    sweep1 = Sweep(parameter=sweep_parameter_1, data=StartStopOptions(1, 10, 10).data)
    sweep_parameter_2 = Parameter("test2", "Test2", "", DataType.INT)
    sweep2 = Sweep(parameter=sweep_parameter_2, data=StartStopOptions(1, 3, 3).data)
    sweep_parameter_3 = Parameter("test3", "Test3", "", DataType.INT)
    sweep3 = Sweep(parameter=sweep_parameter_3, data=StartStopOptions(1, 100, 100).data)
    variables = [(sweep1.parameter, sweep1.data), (sweep3.parameter, sweep3.data), (sweep2.parameter, sweep2.data)]
    result_parameter = Parameter("result", "Result", "", DataType.COMPLEX, CollectionType.NDARRAY)
    results = []
    for x1 in sweep1.data:
        for x3 in sweep3.data:
            results.append(np.asarray([x1 + (1 * x3) * 1j, x1 + (2 * x3) * 1j, x1 + (3 * x3) * 1j]))
    result = (result_parameter, np.concatenate(results).tolist())
    ds = Parameter.build_data_set(variables, result, {})
    assert ds.data_vars["result"][5][56][2] == 9 + 56j


def test_with_extra_variables() -> None:
    # Building the dataset without full knowledge about the coordinates
    sweep_parameter_1 = Parameter("test1", "Test1", "", DataType.INT)
    sweep1 = Sweep(parameter=sweep_parameter_1, data=StartStopOptions(1, 10, 10).data)
    sweep_parameter_2 = Parameter("test2", "Test2", "", DataType.INT)
    sweep2 = Sweep(parameter=sweep_parameter_2, data=StartStopOptions(1, 3, 3).data)
    sweep_parameter_3 = Parameter("test3", "Test3", "", DataType.INT)
    sweep3 = Sweep(parameter=sweep_parameter_3, data=StartStopOptions(1, 100, 100).data)
    result_parameter = Parameter("result", "Result", "", DataType.COMPLEX, CollectionType.NDARRAY)
    results = []
    for x1 in sweep1.data:
        for x2 in sweep2.data:
            for x3 in sweep3.data:
                results.append(np.asarray([x1 + (x2 * x3) * 1j]))
    result = (result_parameter, np.concatenate(results).tolist())

    variables = [(sweep1.parameter, sweep1.data), (sweep2.parameter, sweep2.data)]
    extra_variables = [("hardware_index", 100)]
    ds = Parameter.build_data_set(variables, result, {}, extra_variables)

    assert ds.data_vars["result"].data.shape == (10, 3, 100)
    assert ds.data_vars["result"][5][2][56] == 6 + 18j
    assert "test1" in ds.coords
    assert "test2" in ds.coords
    assert "test3" not in ds.coords
    assert "hardware_index" not in ds.coords
    assert "hardware_index" in ds.dims
