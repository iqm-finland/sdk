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

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
import math
import uuid

from iqm.models.playlist import Playlist

from exa.common.data.parameter import SettingValue
from exa.common.data.setting_node import SettingNode
from exa.common.errors.iqm_error import NotFoundError, ValidationError
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

    @cached_property
    def qpu_runtime(self) -> float:
        return _qpu_runtime(self)


@dataclass(kw_only=True)
class SweepDefinitionLite:
    """Lightweight version of SweepDefinition.

    Can be used when the full tree representation of the settings is not needed.
    """

    sweep_id: uuid.UUID
    """Unique identifier of the sweep."""
    dut_label: str
    """DUT label of the device being used."""
    settings: dict[str, SettingValue]
    """Mapping of setting names to their values for initial settings to set before the sweep.
    Must only contain non-None, non-readonly controller settings."""
    sweeps: NdSweep
    """Sweeps that define the swept parameters, i.e. a list of parallel sweeps,
    where the data values of all sweeps in the tuple are interleaved, and updated simultaneously during the sweep."""
    return_parameters: list[str]
    """Parameters that will be queried from devices and saved for each spot (variable-tuple)
    of the N-dimensional sweep. Each item must correspond to a setting name in `settings`."""
    playlist: Playlist | None
    """A :class:`~iqm.models.playlist.Playlist` that should be uploaded to the controllers."""

    @cached_property
    def qpu_runtime(self) -> float:
        return _qpu_runtime(self)


def _qpu_runtime(sweep: SweepDefinition | SweepDefinitionLite) -> float:
    """Rough estimate of the sweep's QPU runtime in seconds."""
    if isinstance(sweep, SweepDefinition):
        end_delay_setting = sweep.settings.find_by_name("options.end_delay")
        playlist_repeats_setting = sweep.settings.find_by_name("options.playlist_repeats")
        if end_delay_setting is None:
            raise NotFoundError("settings.options.end_delay is missing.")
        if playlist_repeats_setting is None:
            raise NotFoundError("settings.options.playlist_repeats is missing.")
        end_delay_setting_value = end_delay_setting.value
        playlist_repeats_setting_value = playlist_repeats_setting.value
    else:
        end_delay_setting_value = sweep.settings.get("options.end_delay")
        playlist_repeats_setting_value = sweep.settings.get("options.playlist_repeats")

    if end_delay_setting_value is None:
        raise NotFoundError("settings.options.end_delay is missing.")
    if playlist_repeats_setting_value is None:
        raise NotFoundError("settings.options.playlist_repeats is missing.")
    repetitions = playlist_repeats_setting_value
    end_delay = end_delay_setting_value
    if not isinstance(repetitions, int):
        raise ValidationError(f"settings.options.playlist_repeats is a {type(repetitions)}, must be a int.")
    if not isinstance(end_delay, float):
        raise ValidationError(f"settings.options.end_delay is a {type(end_delay)}, must be a float.")

    playlist = sweep.playlist
    if playlist is None:
        return 0  # degenerate sweep, just sets instrument settings

    playlist_duration_seconds = end_delay * len(playlist.segments) * repetitions
    for segment in playlist.segments:
        # Take some channel in the segment, and assume all channels have the same duration.
        channel_name, instructions = next(iter(segment.instructions.items()))
        channel = playlist.channel_descriptions[channel_name]

        samples = sum(
            channel.instruction_table[idx].duration_samples * count for idx, count in Counter(instructions).items()
        )
        playlist_duration_seconds += repetitions * samples / channel.channel_config.sampling_rate

    n_spots = math.prod(len(sweep_dim[0].data) for sweep_dim in sweep.sweeps)

    qpu_runtime = playlist_duration_seconds * n_spots

    return qpu_runtime


@dataclass(kw_only=True)
class SweepData(SweepBase):
    """The content of the sweep stored in the database.

    The raw data for each spot in the sweep is saved as NumPy arrays,
    and the complete data for the whole sweep is saved as an ``xarray.Dataset``
    which has ``SweepBase.sweeps`` as coordinates and
    ``SweepBase.return_parameters`` data as ``xarray.DataArray`` s.
    """

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
