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

"""Range specification used with exponential sweeps."""

from dataclasses import dataclass

from exa.common.control.sweep.option.constants import DEFAULT_BASE, DEFAULT_COUNT
from exa.common.control.sweep.option.start_stop_base_options import StartStopBaseOptions
from exa.common.control.sweep.option.sweep_options import SweepOptions


@dataclass(frozen=True)
class CenterSpanBaseOptions(SweepOptions):
    """Range generation options.

    Values are generated over the interval from `base` power start of the range
    with the center `center` and the size of `span` to `base` power end of the range
    with the center `center` and the size of `span`. The number of values = `count`.
    These options are used only for exponential sweep range.
    """

    #: Value of interval center for the power.
    center: int | float
    #: Size of the interval for the power
    span: int | float
    #: Number of values to generate. Default to
    #: :const:`exa.common.control.sweep.option.constants.DEFAULT_COUNT`.
    count: int | None = None
    #: Number, that is raised to the power of the range with the center `center` and the size of `span`.
    # Default to :const:`exa.common.control.sweep.option.constants.DEFAULT_BASE`.
    base: int | float | None = None
    #: Order of generated values. Default to ascending
    asc: bool | None = None

    def __post_init__(self):
        if self.count is None:
            object.__setattr__(self, "count", DEFAULT_COUNT)
        if self.base is None:
            object.__setattr__(self, "base", DEFAULT_BASE)
        if self.asc is None:
            object.__setattr__(self, "asc", True)

    @property
    def data(self) -> list[int | float | complex]:
        start = self.center - (self.span / 2)
        stop = self.center + (self.span / 2)
        (start, stop) = (start, stop) if self.asc else (stop, start)
        return StartStopBaseOptions(start, stop, count=self.count, base=self.base).data  # type:ignore[arg-type,return-value]
