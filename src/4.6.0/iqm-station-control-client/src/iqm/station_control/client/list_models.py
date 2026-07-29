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
"""Station control client list types for different models.

These are used mainly for easy serialization and deserialization of list of objects.
"""

from typing import Generic, TypeVar

from pydantic import ConfigDict, RootModel

from iqm.station_control.interface.list_with_meta import Meta
from iqm.station_control.interface.models import (
    DutData,
    DutFieldData,
    ObservationData,
    ObservationDefinition,
    ObservationLite,
    ObservationSetData,
    ObservationUpdate,
    RunLite,
    SequenceMetadataData,
    StaticQuantumArchitecture,
    TimelineEntry,
)
from iqm.station_control.interface.pydantic_base import PydanticBase

T = TypeVar("T")


class ListWithMetaResponse(PydanticBase, Generic[T]):
    """Class used for list endpoints to envelope the items to a dict with additional metadata.

    This should be used only in REST API communication (JSON). For Python users,
    this model should be deserialized to :class:`iqm.pulse.ListWithMeta`:,
    which behaves like a standard list.
    """

    items: list[T]
    meta: Meta | None = None


class ListModel(RootModel):
    """A Pydantic `BaseModel` for a container model of a list of objects."""

    def __iter__(self):
        return iter(self.root)

    def __getitem__(self, item):  # noqa: ANN001
        return self.root[item]

    def __len__(self) -> int:
        return len(self.root)

    def __str__(self) -> str:
        return str(self.root)

    model_config = ConfigDict(
        ser_json_inf_nan="constants",  # Will serialize Infinity and NaN values as Infinity and NaN
    )


class DutList(ListModel):  # noqa: D101
    root: list[DutData]


class DutFieldDataList(ListModel):  # noqa: D101
    root: list[DutFieldData]


class ObservationDataList(ListModel):  # noqa: D101
    root: list[ObservationData]


class ObservationDefinitionList(ListModel):  # noqa: D101
    root: list[ObservationDefinition]


class ObservationLiteList(ListModel):  # noqa: D101
    root: list[ObservationLite]


class ObservationUpdateList(ListModel):  # noqa: D101
    root: list[ObservationUpdate]


class ObservationSetDataList(ListModel):  # noqa: D101
    root: list[ObservationSetData]


class RunLiteList(ListModel):  # noqa: D101
    root: list[RunLite]


class SequenceMetadataDataList(ListModel):  # noqa: D101
    root: list[SequenceMetadataData]


class StaticQuantumArchitectureList(ListModel):  # noqa: D101
    root: list[StaticQuantumArchitecture]


class TimelineEntryList(ListModel):  # noqa: D101
    root: list[TimelineEntry]
