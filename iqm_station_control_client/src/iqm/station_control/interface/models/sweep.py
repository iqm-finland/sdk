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
"""Sweep related station control interface models."""

from dataclasses import dataclass
from datetime import datetime
import uuid

from iqm.models.playlist import Playlist

from exa.common.data.setting_node import SettingNode
from exa.common.sweep.util import NdSweep
from iqm.station_control.interface.models.jobs import JobExecutorStatus


@dataclass(kw_only=True)
class SweepBase:
    """Abstract base class of the sweep definition and data."""

    sweep_id: uuid.UUID
    """Unique identifier of the sweep."""
    dut_label: str
    """DUT label of the device being used."""
    settings: SettingNode
    """A tree representation of the initial settings to set before the sweep."""
    sweeps: NdSweep
    """Sweeps that define the swept parameters, i.e. a list of parallel sweeps,
    where the data values of all sweeps in the tuple are interleaved, and updated simultaneously during the sweep."""
    return_parameters: list[str]
    """Parameters that will be queried from devices and saved for each spot (variable-tuple)
    of the N-dimensional sweep. Each item must correspond to a setting name in `settings`."""


@dataclass(kw_only=True)
class SweepDefinition(SweepBase):
    """The content of the sweep object when creating it."""

    playlist: Playlist | None
    """A :class:`~iqm.models.playlist.Playlist` that should be uploaded to the controllers."""


@dataclass(kw_only=True)
class SweepData(SweepBase):
    """The content of the sweep stored in the database."""

    created_timestamp: datetime
    """Time when the object was created in the database."""
    modified_timestamp: datetime
    """Time when the object was last modified in the database."""
    begin_timestamp: datetime | None
    """Time when the sweep began in the station control."""
    end_timestamp: datetime | None
    """Time when the sweep ended in the station control."""
    job_status: JobExecutorStatus
    """Status of sweep execution."""
