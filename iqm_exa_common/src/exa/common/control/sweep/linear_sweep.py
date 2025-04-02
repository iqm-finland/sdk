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

from typing import Any

from typing_extensions import deprecated

from exa.common.control.sweep.option import CenterSpanOptions, StartStopOptions
from exa.common.control.sweep.sweep import Sweep
from exa.common.data.parameter import Parameter
from exa.common.errors.exa_error import InvalidSweepOptionsTypeError
from exa.common.helpers.deprecation import format_deprecated


@deprecated(format_deprecated(old="`LinearSweep`", new="`Sweep`", since="28.3.2025"))
class LinearSweep(Sweep):
    """Generates evenly spaced parameter values based on `options`.

    - If `options` is instance of :class:`.StartStopOptions`, then start and stop options are used for interval
    - If `options` is instance of :class:`.CenterSpanOptions`,
      then the start and stop of the interval are calculated from center and span values

    Raises:
        ValueError: Error is raised if `options` is inconsistent.

    """

    def __init__(
        self,
        parameter: Parameter,
        options: StartStopOptions | CenterSpanOptions | None = None,
        *,
        data: list[Any] | None = None,
        **kwargs,
    ) -> None:
        if options and not isinstance(options, StartStopOptions | CenterSpanOptions):
            raise InvalidSweepOptionsTypeError(str(type(options)))
        super().__init__(parameter, options, data=data, **kwargs)
