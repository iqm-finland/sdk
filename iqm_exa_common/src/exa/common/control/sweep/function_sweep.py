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

"""Sweep specification with generated parameter values based on callable object."""

from dataclasses import dataclass

from exa.common.control.sweep.option import FunctionOptions
from exa.common.control.sweep.sweep import Sweep
from exa.common.errors.exa_error import InvalidSweepOptionsTypeError


@dataclass(frozen=True)
class FunctionSweep(Sweep):
    """Generates parameter values based on callable object from `options`."""

    def __post_init__(self):
        if not isinstance(self.options, FunctionOptions):
            raise InvalidSweepOptionsTypeError(str(type(self.options)))

        data = self.options.function()
        if not all(self.parameter.validate(value) for value in data):
            raise ValueError(f"Invalid generated range data {data} for parameter type {self.parameter.data_type}.")
        object.__setattr__(self, "_data", data)
