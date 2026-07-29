#  ********************************************************************************
#  Copyright (c) 2019-2020 IQM Finland Oy.
#  All rights reserved. Confidential and proprietary.
#
#  Distribution or reproduction of any information contained herein
#  is prohibited without IQM Finland Oy’s prior written permission.
#  ********************************************************************************
from exa.common.control.sweep.option.center_span_base_options import CenterSpanBaseOptions
from exa.common.control.sweep.option.constants import DEFAULT_BASE, DEFAULT_COUNT


def test_default_count() -> None:
    options = CenterSpanBaseOptions(center=2, span=4, base=3)
    assert options.base == 3
    assert options.count == DEFAULT_COUNT


def test_default_base() -> None:
    options = CenterSpanBaseOptions(center=2, span=4, asc=False)
    assert not options.asc
    assert options.base == DEFAULT_BASE


def test_default_asc() -> None:
    options = CenterSpanBaseOptions(center=2, span=4, count=3)
    assert options.count == 3
    assert options.asc
