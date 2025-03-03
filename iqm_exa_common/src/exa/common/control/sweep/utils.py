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

from typing import Type

from exa.common.control.sweep.exponential_sweep import ExponentialSweep
from exa.common.control.sweep.fixed_sweep import FixedSweep
from exa.common.control.sweep.function_sweep import FunctionSweep
from exa.common.control.sweep.linear_sweep import LinearSweep
from exa.common.control.sweep.option import (
    CenterSpanBaseOptions,
    CenterSpanOptions,
    FixedOptions,
    FunctionOptions,
    StartStopBaseOptions,
    StartStopOptions,
)
from exa.common.control.sweep.option.sweep_options import SweepOptions
from exa.common.control.sweep.sweep import Sweep

SWEEP_CLASS_OPTIONS_MAPPING = {
    FixedOptions.__name__: FixedSweep,
    FunctionOptions.__name__: FunctionSweep,
    StartStopOptions.__name__: LinearSweep,
    CenterSpanOptions.__name__: LinearSweep,
    StartStopBaseOptions.__name__: ExponentialSweep,
    CenterSpanBaseOptions.__name__: ExponentialSweep,
}


def get_sweep_class_from_options(options: SweepOptions) -> Type[Sweep]:
    return SWEEP_CLASS_OPTIONS_MAPPING[type(options).__name__]
