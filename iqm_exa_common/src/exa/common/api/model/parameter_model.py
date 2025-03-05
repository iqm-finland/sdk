# Copyright 2024 IQM
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

"""Pydantic model for Parameter."""

from __future__ import annotations

from typing import Optional, Tuple

from pydantic import BaseModel, ConfigDict

from exa.common.data.parameter import CollectionType, DataType, Parameter


class ParameterModel(BaseModel):
    """Pydantic parameter model"""  # noqa: D200

    name: str
    parent_name: Optional[str] = None
    label: Optional[str] = None
    parent_label: Optional[str] = None
    unit: Optional[str] = None
    data_type: Optional[int | Tuple[int, ...]] = None
    collection_type: Optional[int] = None
    element_indices: Optional[int | list[int]] = None

    def __hash__(self):
        return hash(
            (
                self.name,
                self.parent_name,
                self.label,
                self.unit,
                self.data_type,
                self.collection_type,
                self.element_indices,
            )
        )

    def __eq__(self, other: ParameterModel):
        return isinstance(other, ParameterModel) and (
            (
                self.name,
                self.parent_name,
                self.label,
                self.parent_label,
                self.unit,
                self.data_type,
                self.collection_type,
                self.element_indices,
            )
            == (
                other.name,
                other.parent_name,
                other.label,
                other.parent_label,
                other.unit,
                other.data_type,
                other.collection_type,
                other.element_indices,
            )
        )

    model_config = ConfigDict(extra="allow")

    def decode(self) -> Parameter:
        """Creates instance of :class:`exa.common.data.parameter.Parameter` out of pydantic `ParameterModel`"""
        if self.data_type is not None:
            if isinstance(self.data_type, tuple):
                dt = tuple(DataType(dt) for dt in self.data_type)
            else:
                dt = DataType(self.data_type)
        else:
            dt = None
        ct = CollectionType(self.collection_type) if self.collection_type is not None else None
        if self.element_indices is None:
            return Parameter(self.name, self.label, self.unit, dt, ct)
        return Parameter(self.parent_name, self.parent_label, self.unit, dt, ct, self.element_indices)

    @classmethod
    def encode(cls, parameter: Parameter) -> ParameterModel:
        """Creates pydantic `ParameterModel` out of :class:`exa.common.data.parameter.Parameter` instance"""
        if parameter.data_type is not None:
            if isinstance(parameter.data_type, tuple):
                dt = tuple(dt.value for dt in parameter.data_type)
            else:
                dt = parameter.data_type.value
        else:
            dt = None
        ct = parameter.collection_type.value if parameter.collection_type is not None else None
        return ParameterModel(
            name=parameter.name,
            parent_name=parameter.parent_name,
            label=parameter.label,
            parent_label=parameter.parent_label,
            unit=parameter.unit,
            data_type=dt,
            collection_type=ct,
            element_indices=parameter.element_indices,
        )
