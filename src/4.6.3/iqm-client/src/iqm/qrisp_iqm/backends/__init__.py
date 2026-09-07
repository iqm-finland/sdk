# Copyright 2026 IQM
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

"""Backend infrastructure for running Qrisp circuits on IQM hardware.

Provides the central :class:`IQMBackend` class (gate- and pulse-level submission),
as well as job handles for tracking execution:

* :class:`IQMCircuitJob` — gate-level circuit submissions via IQM Client.
* :class:`IQMPulseJob` — pulse-level playlist submissions via Pulla.
"""

from __future__ import annotations

from .backend import IQMBackend
from .iqm_circuit_job import IQMCircuitJob
from .iqm_pulse_job import IQMPulseJob

__all__ = [
    "IQMBackend",
    "IQMCircuitJob",
    "IQMPulseJob",
]
