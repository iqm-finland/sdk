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

from typing import Any

from typing_extensions import deprecated

from exa.common.control.sweep.option import CenterSpanBaseOptions, StartStopBaseOptions
from exa.common.control.sweep.sweep import Sweep
from exa.common.data.parameter import Parameter
from exa.common.errors.exa_error import InvalidSweepOptionsTypeError
from exa.common.helpers.deprecation import format_deprecated


@deprecated(format_deprecated(old="`ExponentialSweep`", new="`Sweep`", since="28.3.2025"))
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

    def __init__(
        self,
        parameter: Parameter,
        options: StartStopBaseOptions | CenterSpanBaseOptions | None = None,
        *,
        data: list[Any] | None = None,
        **kwargs,
    ) -> None:
        if options and not isinstance(options, StartStopBaseOptions | CenterSpanBaseOptions):
            raise InvalidSweepOptionsTypeError(str(type(options)))
        super().__init__(parameter, options, data=data, **kwargs)
