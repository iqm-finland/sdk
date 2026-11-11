# Copyright 2024-2025 IQM
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Validation rules for observation values."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import TYPE_CHECKING, Literal

import numpy as np

from exa.common.errors import ValidationError

if TYPE_CHECKING:
    from iqm.station_control.interface.models import ObservationDefinition

DUT_FIELD_FRAGMENT: str = "([0-9A-Za-z_][-_A-Za-z0-9]*)"
DUT_FIELD_SUFFIX: str = f"(:{DUT_FIELD_FRAGMENT}=\\S+)"
DUT_FIELD_PATH: str = f"{DUT_FIELD_FRAGMENT}(\\.{DUT_FIELD_FRAGMENT})*{DUT_FIELD_SUFFIX}*"
DUT_FIELD_PATH_PATTERN: re.Pattern = re.compile(DUT_FIELD_PATH)


@dataclass
class DUTFieldValidationRule:
    """Validate that DUT field is in the correct format."""

    def __call__(self, observation: ObservationDefinition) -> None:
        dut_field = observation.dut_field
        if not DUT_FIELD_PATH_PATTERN.fullmatch(dut_field):
            raise ValidationError(f"Invalid DUT field '{dut_field}'")


@dataclass
class RangeValidationRule:
    """Validate value based on provided range (min-max).

    For list/array of values, the validation will be done for each list item separately.
    """

    min: int | float
    max: int | float
    field: Literal["value", "uncertainty"] = "value"

    def __post_init__(self):
        if self.min > self.max:
            raise ValidationError(f"max={self.max} should be bigger than min={self.min}")

    def __call__(self, observation: ObservationDefinition) -> None:
        value = getattr(observation, self.field)
        if isinstance(value, (list, np.ndarray)):
            for item in value:
                self._check_range(item, observation.dut_field)
        else:
            self._check_range(value, observation.dut_field)

    def _check_range(self, value: float, dut_field: str) -> None:
        if not self.min <= value <= self.max:
            raise ValidationError(f"{dut_field}: value={value} not in the range min={self.min}, max={self.max}")
