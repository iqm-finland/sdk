# Copyright 2025 IQM
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
"""Type hint aliases used in the station control interface."""

from typing import Literal
from uuid import UUID

import numpy as np

# Allow using string UUIDs in API calls directly for convenience.
# StrUUID works if UUIDs will be serialized to strings by the client anyway,
# and then deserialized back to UUID on the server side.
StrUUID = str | UUID

DutType = Literal["chip", "twpa"]
GetObservationsMode = Literal["all_latest", "tags_and", "tags_or", "sequence"]
ObservationSetType = Literal["calibration-set", "characterization-set", "generic-set", "quality-metric-set"]
SoftwareVersionSet = dict[str, str]
Statuses = list[tuple[str, int, int]]
SweepResults = dict[str, list[np.ndarray]]
