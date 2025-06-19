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
"""Run related station control interface models."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any
import uuid

from exa.common.sweep.util import NdSweep
from iqm.station_control.interface.models.sweep import SweepData, SweepDefinition


@dataclass(kw_only=True)
class RunBase:
    """Abstract base class of run data."""

    run_id: uuid.UUID
    """Unique identifier of the run."""
    username: str
    """User who defined the run."""
    experiment_name: str
    """Identifier of the Experiment (:attr:`.Experiment.name`)."""
    experiment_label: str
    """Freeform label of the Experiment. As opposed to `experiment_name`, no core logic relies on this value."""
    options: dict[str, Any] | None
    """Experiment-specific options or toggles that generated the run."""
    software_version_set_id: int | None
    """Unique identifier of the software version set of the current Python runtime."""


@dataclass(kw_only=True)
class RunConfigurationBase:
    """Abstract base class of the run configuration data."""

    additional_run_properties: dict[str, Any] | None
    """A free-form dictionary of data, used to store information that does not fall into other categories."""
    hard_sweeps: dict[str, NdSweep] | None
    """Maps :attr:`.SweepBase.return_parameters` to "hardware sweep specification" which specifies
    how the data measured at each spot should be interpreted and shaped.
    The hard sweep specification is in the same format as :attr:`.SweepBase.sweeps`,
    which means that the returned data can be interpreted as an N-dimensional sweep inside the spot.
    An empty list is interpreted such that the return parameter is a scalar.
    The hard sweep specification can also be `None`,
    in which case the shape will be whatever the instrument returns."""
    components: list[str]
    """Components that participate in this run."""
    default_data_parameters: list[str]
    """The subset of :attr:`.SweepBase.return_parameters` that were added by default, not by the user.
    Used to select which data to analyze and plot."""
    default_sweep_parameters: list[str]
    """The subset of :attr:`.SweepBase.sweeps` parameters were added by default, not by the user.
    Used to select which data to analyze and plot."""


@dataclass(kw_only=True)
class RunDefinition(RunBase, RunConfigurationBase):
    """The content of the run object when creating it."""

    sweep_definition: SweepDefinition
    """The content of the associated sweep stored in the database."""


@dataclass(kw_only=True)
class RunWithTimestamps(RunBase):
    """Abstract base class of run data including timestamps."""

    created_timestamp: datetime
    """Time when the object was created in the database."""
    modified_timestamp: datetime
    """Time when the object was last modified in the database."""
    begin_timestamp: datetime
    """Time when the run began in the station control."""
    end_timestamp: datetime | None
    """Time when the run ended in the station control."""


@dataclass(kw_only=True)
class RunLite(RunWithTimestamps):
    """The data of the run stored in the database, excluding run configuration data."""

    sweep_id: uuid.UUID | None
    """Unique identifier of the associated sweep."""


@dataclass(kw_only=True)
class RunData(RunWithTimestamps, RunConfigurationBase):
    """The content of the run and its configuration stored in the database."""

    sweep_data: SweepData
    """The content of the associated sweep stored in the database."""
