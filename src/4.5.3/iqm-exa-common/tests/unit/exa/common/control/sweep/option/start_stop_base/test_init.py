#  ********************************************************************************
#  Copyright (c) 2019-2020 IQM Finland Oy.
#  All rights reserved. Confidential and proprietary.
#
#  Distribution or reproduction of any information contained herein
#  is prohibited without IQM Finland Oy’s prior written permission.
#  ********************************************************************************
from exa.common.control.sweep.option.constants import DEFAULT_BASE, DEFAULT_COUNT
from exa.common.control.sweep.option.start_stop_base_options import StartStopBaseOptions


def test_default_count() -> None:
    options = StartStopBaseOptions(start=1, stop=5, base=3)
    assert options.base == 3
    assert options.count == DEFAULT_COUNT


def test_default_base() -> None:
    options = StartStopBaseOptions(start=1, stop=5, count=3)
    assert options.count == 3
    assert options.base == DEFAULT_BASE
