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

"""Pydantic model for Sweep."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, field_serializer

from exa.common.api.model.parameter_model import ParameterModel
from exa.common.control.sweep.fixed_sweep import FixedSweep
from exa.common.control.sweep.option.fixed_options import FixedOptions
from exa.common.control.sweep.sweep import Sweep
from exa.common.helpers import json_helper


class SweepModel(BaseModel):
    """Pydantic sweep model."""

    parameter: ParameterModel
    data: list[Any]

    def __hash__(self):
        return hash((self.parameter, self.data))

    def __eq__(self, other: SweepModel):
        return isinstance(other, SweepModel) and ((self.parameter, self.data) == (other.parameter, other.data))

    model_config = ConfigDict(extra="allow")

    @field_serializer("data")
    def serialize_data(self, data: Any, _info):
        for index, value in enumerate(data):
            encoder = json_helper.get_json_encoder().get(type(value))
            if encoder:
                data[index] = encoder(value)
        return data

    def decode(self) -> FixedSweep:
        """Creates instance of :class:`exa.common.control.sweep.fixed_sweep.FixedSweep` out of pydantic `SweepModel`."""
        parameter = ParameterModel.decode(self.parameter)
        data = [json_helper.decode_json(d) for d in self.data]
        options = FixedOptions(fixed=data)
        # TODO: FixedSweep is used as a temporary solution, since it is possible to send list of values with FixedSweep.
        return FixedSweep(parameter=parameter, options=options)

    @classmethod
    def encode(cls, sweep: Sweep) -> SweepModel:
        """Creates pydantic `SweepModel` out of :class:`exa.common.control.sweep.sweep.Sweep` instance."""
        return SweepModel(parameter=ParameterModel.encode(sweep.parameter), data=sweep.data)
