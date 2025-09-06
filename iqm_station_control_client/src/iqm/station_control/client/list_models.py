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
)
from iqm.station_control.interface.pydantic_base import PydanticBase

T = TypeVar("T")


class ResponseWithMeta(PydanticBase, Generic[T]):
    """Class used for query endpoints to return metadata in addition to the returned items."""

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


class DutList(ListModel):
    root: list[DutData]


class DutFieldDataList(ListModel):
    root: list[DutFieldData]


class ObservationDataList(ListModel):
    root: list[ObservationData]


class ObservationDefinitionList(ListModel):
    root: list[ObservationDefinition]


class ObservationLiteList(ListModel):
    root: list[ObservationLite]


class ObservationUpdateList(ListModel):
    root: list[ObservationUpdate]


class ObservationSetDataList(ListModel):
    root: list[ObservationSetData]


class SequenceMetadataDataList(ListModel):
    root: list[SequenceMetadataData]


class RunLiteList(ListModel):
    root: list[RunLite]
