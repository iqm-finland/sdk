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

"""Sweep specification with exponentially spaced values."""

from dataclasses import dataclass, field
import logging
import math
from typing import List, Union

import numpy as np

from exa.common.control.sweep.option import CenterSpanBaseOptions, StartStopBaseOptions
from exa.common.control.sweep.sweep import Sweep
from exa.common.errors.exa_error import InvalidSweepOptionsTypeError


@dataclass(frozen=True)
class ExponentialSweep(Sweep):
    """Generates parameter values spaced evenly on a geometric progression based on `options`.

    - If `options` is instance of :class:`.StartStopBaseOptions`,
      the start and stop of the interval are calculated from powers of start and stop.
    - If `options` is instance of :class:`.CenterSpanBaseOptions`,
      the start and stop of the interval are calculated from powers of start and stop,
      which are derived from center and span.

    Raises:
        ValueError: Error is raised if `options` is inconsistent.

    """

    logger: logging.Logger = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "logger", logging.getLogger(__name__ + "." + self.__class__.__name__))
        if isinstance(self.options, StartStopBaseOptions):
            self.__from_start_stop_base(self.options)
        elif isinstance(self.options, CenterSpanBaseOptions):
            self.__from_center_span_base(self.options)
        else:
            raise InvalidSweepOptionsTypeError(str(type(self.options)))

    def __from_start_stop_base(self, options: StartStopBaseOptions) -> None:
        self.logger.debug(f"EXPONENTS: ({options.start}, {options.stop}) with base {options.base}")
        self._validate_value(options.start, "start")
        self._validate_value(options.stop, "stop")
        if options.start == 0 or options.stop == 0:
            raise ValueError("Exponential range sweep start and stop values must not be zero.")
        options = StartStopBaseOptions(
            math.pow(options.base, options.start),
            math.pow(options.base, options.stop),
            count=options.count,
        )
        object.__setattr__(self, "_data", self.__generate(options))

    def __from_center_span_base(self, options: CenterSpanBaseOptions) -> None:
        start = options.center - (options.span / 2)
        stop = options.center + (options.span / 2)
        self.logger.debug("EXPONENTS: ({}, {}) with base {}".format(start, stop, options.base))
        (start, stop) = (start, stop) if options.asc else (stop, start)
        start_stop_base_options = StartStopBaseOptions(
            start,
            stop,
            count=options.count,
            base=options.base,
        )
        self.__from_start_stop_base(start_stop_base_options)

    @staticmethod
    def __generate(options: StartStopBaseOptions) -> List[Union[int, float, complex]]:
        return np.geomspace(options.start, options.stop, options.count, endpoint=True).tolist()
