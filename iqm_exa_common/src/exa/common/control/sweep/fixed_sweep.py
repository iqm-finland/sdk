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

"""Sweep specification with arbitrary values."""

from typing import Any

from typing_extensions import deprecated

from exa.common.control.sweep.option import FixedOptions
from exa.common.control.sweep.sweep import Sweep
from exa.common.data.parameter import Parameter
from exa.common.errors.exa_error import InvalidSweepOptionsTypeError
from exa.common.helpers.deprecation import format_deprecated


@deprecated(format_deprecated(old="`FixedSweep`", new="`Sweep`", since="28.3.2025"))
class FixedSweep(Sweep):
    """A sweep over arbitrary set of values, given by `options`."""

    def __init__(
        self, parameter: Parameter, options: FixedOptions | None = None, *, data: list[Any] | None = None, **kwargs
    ) -> None:
        if options and not isinstance(options, FixedOptions):
            raise InvalidSweepOptionsTypeError(str(type(options)))
        super().__init__(parameter, options, data=data, **kwargs)
