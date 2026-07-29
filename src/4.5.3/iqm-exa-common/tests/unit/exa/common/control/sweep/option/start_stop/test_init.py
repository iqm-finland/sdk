#  ********************************************************************************
#  Copyright (c) 2019-2020 IQM Finland Oy.
#  All rights reserved. Confidential and proprietary.
#
#  Distribution or reproduction of any information contained herein
#  is prohibited without IQM Finland Oy’s prior written permission.
#  ********************************************************************************
from pytest import raises

from exa.common.control.sweep.option.constants import DEFAULT_COUNT
from exa.common.control.sweep.option.start_stop_options import StartStopOptions


def test_use_count() -> None:
    options = StartStopOptions(0, 10, count=5, step=3)
    assert options.step is None


def test_default_count() -> None:
    options = StartStopOptions(0, 10)
    assert options.count == DEFAULT_COUNT


def test_too_big_step() -> None:
    with raises(ValueError):
        StartStopOptions(0, 2, step=4)


def test_zero_step() -> None:
    with raises(ValueError):
        StartStopOptions(0, 6, step=0)


def test_negative_count() -> None:
    with raises(ValueError):
        StartStopOptions(0, 10, count=-1)
