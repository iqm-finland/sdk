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
import logging
import math

import numpy as np

from exa.common.control.sweep.option.constants import DEFAULT_BASE, DEFAULT_COUNT
from exa.common.control.sweep.option.sweep_options import SweepOptions

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StartStopBaseOptions(SweepOptions):
    """Range generation options.

    Values are generated over the interval from `base` power `start` to `base` power `stop`.
    The number of values = `count`. These options are used only for exponential sweep range.
    """

    #: The power for the start of the interval.
    start: int | float
    #: The power for the end of the interval.
    stop: int | float
    #: Number of values to generate. Default to
    #: :const:`exa.common.control.sweep.option.constants.DEFAULT_COUNT`.
    count: int = DEFAULT_COUNT
    #: Number, that is raised to the power `start` or `stop`. Default to
    #: :const:`exa.common.control.sweep.option.constants.DEFAULT_BASE`.
    base: int = DEFAULT_BASE

    def __post_init__(self):
        if self.start == 0 or self.stop == 0:
            raise ValueError("Exponential range sweep start and stop values must not be zero.")

    @property
    def data(self) -> list[int | float]:
        logger.debug(f"EXPONENTS: ({self.start}, {self.stop}) with base {self.base}")
        start = math.pow(self.base, self.start)
        stop = math.pow(self.base, self.stop)
        return np.geomspace(start, stop, self.count, endpoint=True).tolist()
