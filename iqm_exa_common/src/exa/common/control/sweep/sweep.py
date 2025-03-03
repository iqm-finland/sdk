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

from dataclasses import dataclass, field
from typing import Any, List, Union

from exa.common.control.sweep.option import SweepOptions
from exa.common.data.parameter import Parameter


@dataclass(frozen=True)
class Sweep:
    """Base immutable class for sweeps."""

    parameter: Parameter
    """The Sweep represents changing the values of this Parameter."""

    options: SweepOptions
    """Range specification where the values are derived from."""

    _data: List[Any] = field(init=False, repr=False)

    @property
    def data(self) -> List[Any]:
        """List of values for :attr:`parameter`"""
        return self._data

    def _validate_value(self, _value: Union[int, float, complex], value_label: str):
        if not self.parameter.validate(_value):
            error = ValueError(f"Invalid {value_label} value {_value} for parameter type {self.parameter.data_type}.")
            raise error
