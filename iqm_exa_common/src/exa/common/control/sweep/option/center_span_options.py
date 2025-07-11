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

"""Range specification to define a range around a center value."""

from dataclasses import dataclass

from exa.common.control.sweep.option.constants import DEFAULT_COUNT
from exa.common.control.sweep.option.start_stop_options import StartStopOptions
from exa.common.control.sweep.option.sweep_options import SweepOptions


@dataclass(frozen=True)
class CenterSpanOptions(SweepOptions):
    """Range generation options.

    Values are generated over the interval with the center `center` and the size of `span`.
    For linear sweep range the number of generated values can be based either on `count` or `step`.
    In case `count` is empty and `step` is not, `step` is used for calculating `count`.
    For exponential sweep range only `count` is used.
    """

    #: Value of interval center.
    center: int | float | complex
    #: Size of the interval.
    span: int | float | complex
    #: Number of values to generate.
    #: If `count` and `step` are empty, the default value of count is
    #: :const:`exa.common.control.sweep.option.constants.DEFAULT_COUNT`.
    count: int | None = None
    #: Size of spacing between values.
    step: int | float | complex | None = None
    #: Order of generated values. Default to ascending
    asc: bool | None = None

    def __post_init__(self):
        if self.count is not None and self.step is not None:
            object.__setattr__(self, "step", None)
        if self.count is None and self.step is None:
            object.__setattr__(self, "count", DEFAULT_COUNT)
        if self.asc is None:
            object.__setattr__(self, "asc", True)

    @property
    def data(self) -> list[int | float | complex]:
        start = self.center - (self.span / 2)
        stop = self.center + (self.span / 2)
        (start, stop) = (start, stop) if self.asc else (stop, start)
        return StartStopOptions(start, stop, count=self.count, step=self.step).data
