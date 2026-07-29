#  ********************************************************************************
#  Copyright (c) 2019-2024 IQM Finland Oy.
#  All rights reserved. Confidential and proprietary.
#
#  Distribution or reproduction of any information contained herein
#  is prohibited without IQM Finland Oy’s prior written permission.
#  ********************************************************************************
from pydantic import ValidationError
import pytest

from exa.common.data.parameter import Parameter, Sweep


def test_update_parameter_should_raise_error() -> None:
    sweep = Sweep(parameter=Parameter("test_parameter"), data=[])
    with pytest.raises(ValidationError, match="Instance is frozen"):
        sweep.parameter = Parameter("test_parameter_2")


def test_update_data_should_raise_error() -> None:
    sweep = Sweep(parameter=Parameter("test_parameter"), data=[])
    with pytest.raises(ValidationError, match="Instance is frozen"):
        sweep.data = [1, 2]


def test_access_attributes() -> None:
    parameter = Parameter("test_parameter")
    sweep = Sweep(parameter=Parameter("test_parameter"), data=[])
    assert sweep.parameter == parameter
