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

"""Range specification for arbitrary set of values."""

from dataclasses import dataclass

import numpy as np

from exa.common.control.sweep.option.sweep_options import SweepOptions
from exa.common.control.sweep.sweep_values import SweepValues


@dataclass(frozen=True)
class FixedOptions(SweepOptions):
    """Range fixed options."""

    #: List of values.
    fixed: SweepValues

    @property
    def data(self) -> SweepValues:
        return np.asarray(self.fixed).tolist()
