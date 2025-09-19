# Copyright 2024 IQM
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

"""Base immutable class for sweeps specifications."""

from typing import Any

from exa.common.control.sweep.option import CenterSpanOptions, StartStopOptions
from exa.common.control.sweep.sweep_values import SweepValues
from exa.common.data.base_model import BaseModel
from exa.common.data.parameter import Parameter


class Sweep(BaseModel):
    """Base immutable class for sweeps."""

    parameter: Parameter
    """The Sweep represents changing the values of this Parameter."""

    data: SweepValues
    """List of values for :attr:`parameter`"""

    def model_post_init(self, __context: Any) -> None:
        if not all(self.parameter.validate(value) for value in self.data):
            raise ValueError(f"Invalid range data {self.data} for parameter type {self.parameter.data_type}.")

    @classmethod
    def __from_center_span(cls, parameter: Parameter, options: CenterSpanOptions) -> SweepValues:
        cls._validate_value(parameter, options.center, "center")
        cls._validate_value(parameter, options.span, "span")
        return options.data

    @classmethod
    def __from_start_stop(cls, parameter: Parameter, options: StartStopOptions) -> SweepValues:
        cls._validate_value(parameter, options.start, "start")
        cls._validate_value(parameter, options.stop, "stop")
        cls._validate_value(parameter, options.step, "step")  # type: ignore[arg-type]
        return options.data

    @staticmethod
    def _validate_value(parameter: Parameter, value: complex | str | bool, value_label: str) -> None:
        if not parameter.validate(value):
            raise ValueError(f"Invalid {value_label} value {value} for parameter type {parameter.data_type}.")
