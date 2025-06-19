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
"""DUT related interface models."""

from iqm.station_control.interface.models.type_aliases import DutType
from iqm.station_control.interface.pydantic_base import PydanticBase


class DutData(PydanticBase):
    """Represents a Device Under Test, or DUT, for short."""

    label: str
    """DUT label of the device."""
    dut_type: DutType
    """String indicating the DUT type of the device
    Can be either 'chip' or 'twpa'."""


class DutFieldData(PydanticBase):
    """A DUT field or path and its unit."""

    path: str
    """DUT field or path."""
    unit: str
    """SI unit of the value. Empty string means the value is dimensionless."""
