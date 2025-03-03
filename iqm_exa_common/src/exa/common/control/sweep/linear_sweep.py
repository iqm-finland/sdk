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

"""Sweep specification with linearly spaced values."""

from dataclasses import dataclass
import math
from typing import Union

import numpy as np

from exa.common.control.sweep.option import CenterSpanOptions, StartStopOptions
from exa.common.control.sweep.sweep import Sweep
from exa.common.errors.exa_error import InvalidSweepOptionsTypeError


@dataclass(frozen=True)
class LinearSweep(Sweep):
    """Generates evenly spaced parameter values based on `options`.

    - If `options` is instance of :class:`.StartStopOptions`, then start and stop options are used for interval
    - If `options` is instance of :class:`.CenterSpanOptions`,
      then the start and stop of the interval are calculated from center and span values

    Raises:
        ValueError: Error is raised if `options` is inconsistent.

    """

    def __post_init__(self):
        if isinstance(self.options, StartStopOptions):
            self.__from_start_stop(self.options)
        elif isinstance(self.options, CenterSpanOptions):
            self.__from_center_span(self.options)
        else:
            raise InvalidSweepOptionsTypeError(str(type(self.options)))

    def __from_start_stop(self, options: StartStopOptions) -> None:
        self._validate_value(options.start, "start")
        self._validate_value(options.stop, "stop")
        self._validate_value(options.step, "step")
        object.__setattr__(self, "_data", self.__generate(options))

    def __from_center_span(self, options: CenterSpanOptions) -> None:
        self._validate_value(options.center, "center")
        self._validate_value(options.span, "span")
        start = options.center - (options.span / 2)
        stop = options.center + (options.span / 2)
        (start, stop) = (start, stop) if options.asc else (stop, start)
        start_stop_options = StartStopOptions(start, stop, count=options.count, step=options.step)
        self.__from_start_stop(start_stop_options)

    def __generate(self, options: StartStopOptions) -> list[Union[int, float, complex]]:
        if options.step is not None:
            count = 1 + math.ceil(abs(options.stop - options.start) / float(np.abs(options.step)))
            data = self.__generate_by_count(options, count)
        else:
            data = self.__generate_by_count(options, options.count)
        return data

    @staticmethod
    def __generate_by_count(options: StartStopOptions, count: int) -> list[Union[int, float, complex]]:
        return np.linspace(options.start, options.stop, count, endpoint=True).tolist()
