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

"""Pydantic model for Setting."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, field_serializer

from exa.common.api.model.parameter_model import ParameterModel
from exa.common.data.parameter import Setting
from exa.common.helpers import json_helper
from exa.common.helpers.numpy_helper import coerce_numpy_type_to_native


class SettingModel(BaseModel):
    """Pydantic setting model"""  # noqa: D200

    parameter: ParameterModel
    value: Any = None

    def __hash__(self):
        return hash((self.parameter, self.value))

    def __eq__(self, other: SettingModel):
        return isinstance(other, SettingModel) and ((self.parameter, self.value) == (other.parameter, other.value))

    model_config = ConfigDict(extra="allow")

    @field_serializer("value")
    def serialize_value(self, value: Any, _info):
        encoder = json_helper.get_json_encoder().get(type(value))
        if encoder:
            return encoder(value)
        return value

    def decode(self) -> Setting:
        """Creates instance of :class:`exa.common.data.parameter.Setting` out of pydantic `SettingModel`"""
        value = json_helper.decode_json(self.value)
        parameter = ParameterModel.decode(self.parameter)
        return Setting(parameter=parameter, value=value)

    @classmethod
    def encode(cls, setting: Setting) -> SettingModel:
        """Creates pydantic :class:`SettingModel` out of :class:`~exa.common.data.parameter.Setting` instance.

        Converts numpy scalars to native types to avoid problems with JSON encoding.
        """
        value = coerce_numpy_type_to_native(setting.value)
        return SettingModel(parameter=ParameterModel.encode(setting.parameter), value=value)
