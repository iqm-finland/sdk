#  ********************************************************************************
#  Copyright (c) 2019-2025 IQM Finland Oy.
#  All rights reserved. Confidential and proprietary.
#
#  Distribution or reproduction of any information contained herein
#  is prohibited without IQM Finland Oy’s prior written permission.
#  ********************************************************************************
import pytest

from exa.common.data.parameter import Parameter, Setting


def test_update():
    # test source is also set when updating
    s1 = {"type": "horse_source", "horse": "red"}
    s2 = {"type": "non_horse_source", "horse": "pale"}

    setting = Setting(Parameter("horse"), 1.0, source=s1)
    assert setting.value == 1.0
    assert setting.source == s1

    # test source can be given when updating
    updated = setting.update(2.0, source=s2)
    assert updated.parameter == Parameter("horse")
    assert updated.value == 2.0
    assert updated.source == s2

    # if no source is given, we get a "configured by user" source by default
    updated = setting.update(3.0)
    assert updated.parameter == Parameter("horse")
    assert updated.value == 3.0
    assert updated.source == {"type": "configuration_source", "configurator": "user"}

    # when the value is set to None, source will be None too
    updated = setting.update(None)
    assert updated.parameter == Parameter("horse")
    assert updated.value is None
    assert updated.source is None


def test_update_source_but_no_value():
    # attempting to use a non-None source with a None value gives an error
    setting = Setting(Parameter("horse"), 1.0)
    with pytest.raises(ValueError, match="Setting with no value cannot have a source"):
        _ = setting.update(None, source={"type": "horse_source", "horse": "red"})
