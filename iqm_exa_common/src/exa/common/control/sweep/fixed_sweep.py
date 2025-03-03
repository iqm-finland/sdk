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

"""Sweep specification with arbitrary values."""

from dataclasses import dataclass

import numpy as np

from exa.common.control.sweep.option import FixedOptions
from exa.common.control.sweep.sweep import Sweep
from exa.common.errors.exa_error import InvalidSweepOptionsTypeError


@dataclass(frozen=True)
class FixedSweep(Sweep):
    """A sweep over arbitrary set of values, given by `options`."""

    def __post_init__(self):
        if not isinstance(self.options, FixedOptions):
            raise InvalidSweepOptionsTypeError(str(type(self.options)))

        if not all(self.parameter.validate(value) for value in self.options.fixed):
            raise ValueError(
                f"Invalid fixed range data {self.options.fixed} for parameter type {self.parameter.data_type.name}."
            )
        data = np.asarray(self.options.fixed).tolist()
        object.__setattr__(self, "_data", data)
