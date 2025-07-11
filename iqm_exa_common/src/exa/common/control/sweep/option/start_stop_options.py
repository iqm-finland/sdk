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

"""Range specification to define a linearly spaced interval."""

from dataclasses import dataclass
import math

import numpy as np

from exa.common.control.sweep.option.constants import DEFAULT_COUNT
from exa.common.control.sweep.option.sweep_options import SweepOptions
from exa.common.control.sweep.sweep_values import SweepValues


@dataclass(frozen=True)
class StartStopOptions(SweepOptions):
    """Range generation options.

    Values are generated over the interval from `start` to `stop`. For linear sweep range the
    number of generated values can be based either on `count` or `step`. In case `count` is empty
    and `step` is not, `step` is used for calculating `count`. For exponential sweep range only
    `count` is used.
    """

    #: Starting value of interval.
    start: int | float | complex
    #: Stopping value of interval.
    stop: int | float | complex
    #: Number of values to generate. Must be non-negative.
    #: If `count` and `step` are empty, the default value of count is
    #: :const:`exa.common.control.sweep.option.constants.DEFAULT_COUNT`.
    count: int | None = None
    #: Size of spacing between values. Must be non-zero.
    #: If both `count` and `step` are not empty, only `count` is used
    step: int | float | complex | None = None

    def __post_init__(self):
        if self.count is not None and self.step is not None:
            object.__setattr__(self, "step", None)
        if self.count is None and self.step is None:
            object.__setattr__(self, "count", DEFAULT_COUNT)
        if self.step is not None and (abs(self.step) > abs(self.stop - self.start) or self.step == 0):
            raise ValueError(
                "Step value specified for range must be less than absolute difference of stop and start values and"
                "must be greater than zero."
            )
        if self.count is not None and self.count <= 0:
            raise ValueError("Count value specified for range must be greater than zero.")

    @property
    def data(self) -> list[int | float | complex]:
        if self.step is not None:
            count = 1 + math.ceil(abs(self.stop - self.start) / float(np.abs(self.step)))
            data = self._generate_by_count(count)
        else:
            data = self._generate_by_count(self.count if self.count is not None else DEFAULT_COUNT)
        return data  # type: ignore[return-value]

    def _generate_by_count(self, count: int) -> SweepValues:
        return np.linspace(self.start, self.stop, count, endpoint=True).tolist()
