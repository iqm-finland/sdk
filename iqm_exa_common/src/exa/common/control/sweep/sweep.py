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
import warnings

from exa.common.control.sweep.option import CenterSpanOptions, StartStopOptions, SweepOptions
from exa.common.control.sweep.sweep_values import SweepValues
from exa.common.data.base_model import BaseModel
from exa.common.data.parameter import Parameter
from exa.common.errors.exa_error import InvalidSweepOptionsTypeError


class Sweep(BaseModel):
    """Base immutable class for sweeps."""

    parameter: Parameter
    """The Sweep represents changing the values of this Parameter."""

    data: SweepValues
    """List of values for :attr:`parameter`"""

    def __init__(
        self, parameter: Parameter, options: SweepOptions | None = None, *, data: SweepValues | None = None, **kwargs
    ) -> None:
        if options is None and data is None:
            raise ValueError("Either 'options' or 'data' is required.")
        if options is not None and data is not None:
            raise ValueError(
                "Can't use both 'options' and 'data' at the same time, give only either of the parameters."
            )
        if options is not None:
            warnings.warn("'options' attribute is deprecated, use 'data' instead.", DeprecationWarning)

            if not isinstance(options, SweepOptions):
                raise InvalidSweepOptionsTypeError(str(type(options)))

            if isinstance(options, StartStopOptions):
                data = self.__from_start_stop(parameter, options)
            elif isinstance(options, CenterSpanOptions):
                data = self.__from_center_span(parameter, options)
            else:
                data = options.data

        super().__init__(parameter=parameter, data=data, **kwargs)  # type: ignore[call-arg]  # type: ignore[call-arg]

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
    def _validate_value(parameter: Parameter, value: int | float | complex | str | bool, value_label: str):
        if not parameter.validate(value):
            raise ValueError(f"Invalid {value_label} value {value} for parameter type {parameter.data_type}.")
