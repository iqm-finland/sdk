#  ********************************************************************************
#  Copyright (c) 2019-2020 IQM Finland Oy.
#  All rights reserved. Confidential and proprietary.
#
#  Distribution or reproduction of any information contained herein
#  is prohibited without IQM Finland Oy’s prior written permission.
#  ********************************************************************************
from exa.common.control.sweep.option.center_span_options import CenterSpanOptions
from exa.common.control.sweep.option.constants import DEFAULT_COUNT


def test_use_count() -> None:
    options = CenterSpanOptions(center=5001, span=10000, count=10, step=3)
    assert options.step is None


def test_default_count() -> None:
    options = CenterSpanOptions(center=5001, span=10000, asc=False)
    assert not options.asc
    assert options.count == DEFAULT_COUNT


def test_default_asc() -> None:
    options = CenterSpanOptions(center=5001, span=10000, count=10)
    assert options.count == 10
    assert options.asc
