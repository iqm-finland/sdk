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

"""Range specification used with ExponentialSweep."""

from dataclasses import dataclass
from typing import Union

from exa.common.control.sweep.option.constants import DEFAULT_BASE, DEFAULT_COUNT
from exa.common.control.sweep.option.sweep_options import SweepOptions


@dataclass(frozen=True)
class StartStopBaseOptions(SweepOptions):
    """Range generation options.

    Values are generated over the interval from `base` power `start` to `base` power `stop`.
    The number of values = `count`. These options are used only for exponential sweep range.
    """

    #: The power for the start of the interval.
    start: Union[int, float, complex]
    #: The power for the end of the interval.
    stop: Union[int, float, complex]
    #: Number of values to generate. Default to
    #: :const:`exa.common.control.sweep.option.constants.DEFAULT_COUNT`.
    count: int = None
    #: Number, that is raised to the power `start` or `stop`. Default to
    #: :const:`exa.common.control.sweep.option.constants.DEFAULT_BASE`.
    base: Union[int, float] = None

    def __post_init__(self):
        if self.count is None:
            object.__setattr__(self, "count", DEFAULT_COUNT)
        if self.base is None:
            object.__setattr__(self, "base", DEFAULT_BASE)
