#  ********************************************************************************
#  Copyright (c) 2019-2020 IQM Finland Oy.
#  All rights reserved. Confidential and proprietary.
#
#  Distribution or reproduction of any information contained herein
#  is prohibited without IQM Finland Oy’s prior written permission.
#  ********************************************************************************
from pytest import raises

from exa.common.control.sweep import option as opt
from exa.common.control.sweep.option.option_converter import convert_to_options


def test_invalid_config() -> None:
    with raises(ValueError):
        config = {"foo": "foo", "bar": "bar"}
        convert_to_options(config)


def test_start_stop_priority() -> None:
    config = {"start": 10, "stop": 20, "start_exp": 5, "stop_exp": 10}
    assert isinstance(convert_to_options(config), opt.StartStopOptions)


def test_start_stop_base_priority() -> None:
    config = {"center": 5, "span": 10, "start_exp": 10, "stop_exp": 20}
    assert isinstance(convert_to_options(config), opt.StartStopBaseOptions)


def test_center_span_priority() -> None:
    config = {"center": 5, "span": 10, "center_exp": 10, "span_exp": 20}
    assert isinstance(convert_to_options(config), opt.CenterSpanOptions)


def test_center_span_base_priority() -> None:
    config = {"center_exp": 10, "span_exp": 20, "fixed": [1, 2, 3]}
    assert isinstance(convert_to_options(config), opt.CenterSpanBaseOptions)


def test_start_stop() -> None:
    config = {
        "name": "frequency",
        "type": "linear",
        "parameter": "readout::sweeper::frequency",
        "start": 10,
        "stop": 20,
        "count": 5,
    }
    options = convert_to_options(config)
    assert options == opt.StartStopOptions(10, 20, count=5)


def test_start_stop_str_options() -> None:
    config = {"start": "1 + 2j", "stop": "15 + 2j", "count": "10"}
    options = convert_to_options(config)
    assert options == opt.StartStopOptions(1 + 2j, 15 + 2j, count=10)


def test_start_stop_with_upper_case_and_spaces() -> None:
    config = {"Start": " 1 + 2j", "STOP": "15 +   2j", "count": "10 "}
    options = convert_to_options(config)
    assert options == opt.StartStopOptions(1 + 2j, 15 + 2j, count=10)


def test_start_stop_with_step() -> None:
    config = {"start": 10, "stop": 20, "step": 3}
    options = convert_to_options(config)
    assert options == opt.StartStopOptions(10, 20, step=3)


def test_center_span() -> None:
    config = {
        "name": "frequency",
        "type": "linear",
        "parameter": "readout::sweeper::frequency",
        "center": 5,
        "span": 10,
        "count": 5,
    }
    options = convert_to_options(config)
    assert options == opt.CenterSpanOptions(5, 10, count=5)


def test_center_span_str_options() -> None:
    config = {"center": "1 + 2j", "span": "15 + 2j", "count": "10"}
    options = convert_to_options(config)
    assert options == opt.CenterSpanOptions(1 + 2j, 15 + 2j, count=10)


def test_center_span_with_upper_case_and_spaces() -> None:
    config = {"Center": " 1 + 2j", "SPAN": "15 +   2j", "count": "10 "}
    options = convert_to_options(config)
    assert options == opt.CenterSpanOptions(1 + 2j, 15 + 2j, count=10)


def test_center_span_with_step_and_asc() -> None:
    config = {"center": 5, "span": 10, "step": 3, "asc": False}
    options = convert_to_options(config)
    assert options == opt.CenterSpanOptions(5, 10, step=3, asc=False)


def test_start_stop_base() -> None:
    config = {
        "name": "power",
        "type": "exponential",
        "parameter": "readout::amplitude",
        "start_exp": -2.0,
        "stop_exp": 2.0,
        "count": 10,
    }
    options = convert_to_options(config)
    assert options == opt.StartStopBaseOptions(-2.0, 2.0, count=10)


def test_start_stop_base_str_options() -> None:
    config = {"start_exp": "1", "stop_exp": "15", "count": "10"}
    options = convert_to_options(config)
    assert options == opt.StartStopBaseOptions(1, 15, count=10)


def test_start_stop_base_with_upper_case_and_spaces() -> None:
    config = {"Start_exp": "1 ", "STOP_EXP": "  15", "count": "10 "}
    options = convert_to_options(config)
    assert options == opt.StartStopBaseOptions(1, 15, count=10)


def test_start_stop_base_with_base() -> None:
    config = {"start_exp": -2.0, "stop_exp": 2.0, "count": 10, "base": 3}
    options = convert_to_options(config)
    assert options == opt.StartStopBaseOptions(-2.0, 2.0, count=10, base=3)


def test_center_span_base() -> None:
    config = {
        "name": "power",
        "type": "exponential",
        "parameter": "readout::amplitude",
        "center_exp": 2.0,
        "span_exp": 10.0,
        "count": 10,
    }
    options = convert_to_options(config)
    assert options == opt.CenterSpanBaseOptions(2.0, 10.0, count=10)


def test_center_span_base_str_options() -> None:
    config = {"center_exp": "1 ", "span_exp": "15", "count": "10"}
    options = convert_to_options(config)
    assert options == opt.CenterSpanBaseOptions(1, 15, count=10)


def test_center_span_base_with_upper_case_and_spaces() -> None:
    config = {"Center_exp": "1 ", "SPAN_EXP": "  15", "count": "10 "}
    options = convert_to_options(config)
    assert options == opt.CenterSpanBaseOptions(1, 15, count=10)


def test_center_span_base_with_base_and_asc() -> None:
    config = {"center_exp": 2.0, "span_exp": 10.0, "count": 10, "base": 4, "asc": False}
    options = convert_to_options(config)
    assert options == opt.CenterSpanBaseOptions(2.0, 10.0, count=10, base=4, asc=False)


def test_fixed() -> None:
    config = {"name": "power", "type": "exponential", "parameter": "readout::amplitude", "fixed": [1, 2]}
    options = convert_to_options(config)
    assert options == opt.FixedOptions([1, 2])


def test_fixed_str() -> None:
    config = {"name": "power", "type": "exponential", "parameter": "readout::amplitude", "fixed": "[1, 2]"}
    options = convert_to_options(config)
    assert options == opt.FixedOptions([1, 2])


def test_fixed_str_with_upper_case_and_spaces() -> None:
    config = {"name": "power", "type": "exponential", "parameter": "readout::amplitude", "FIxed": "[1  , 2, 3+  4j]"}
    options = convert_to_options(config)
    assert options == opt.FixedOptions([1, 2, 3 + 4j])
