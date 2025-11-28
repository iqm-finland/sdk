# Copyright 2024-2025 IQM
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

"""Common data types and exceptions for the IQM Pulla interface."""

from typing import TypeAlias

from exa.common.data.value import ObservationValue


class CHADRetrievalException(Exception):
    """Exception for CHAD retrieval failures."""


class SettingsRetrievalException(Exception):
    """Exception for Station Control settings retrieval failures."""


class ChipLabelRetrievalException(Exception):
    """Exception for chip label retrieval failures."""


CalibrationSetValues: TypeAlias = dict[str, ObservationValue]
"""Map from observation name to its value."""


ACQUISITION_LABEL_KEY = "m{idx}"
ACQUISITION_LABEL = "{qubit}__{key}"
MEASUREMENT_MODE_KEY = "__MEASUREMENT_MODE"
HERALDING_KEY = "__HERALD"
RESTRICTED_MEASUREMENT_KEYS = [MEASUREMENT_MODE_KEY, HERALDING_KEY]

# NOTE the buffer duration needs to match all instrument granularities!
# Integer multiples of 80 ns work with 1.8 GHz, 2.0 GHz and 2.4 GHz sample rates and 16 sample granularity,
# which should cover all instruments currently in use. In s.
_BUFFER_GRANULARITY = 80e-9
BUFFER_AFTER_MEASUREMENT_PROBE = 4 * _BUFFER_GRANULARITY
"""Buffer that allows the readout resonator and qubit state to stabilize after a probe pulse, in s.
TODO: not needed after EXA-2089 is done."""
