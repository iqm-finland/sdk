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


class ChipLabelRetrievalException(Exception):
    """Exception for chip label retrieval failures."""


CalibrationSetValues: TypeAlias = dict[str, ObservationValue]
"""Map from observation name to its value."""


ACQUISITION_LABEL_KEY = "m{idx}"
ACQUISITION_LABEL = "{qubit}__{key}"
HERALDING_KEY = "__HERALD"
