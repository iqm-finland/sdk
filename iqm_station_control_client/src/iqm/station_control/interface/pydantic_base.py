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
"""Pydantic related models and types."""

import base64
from typing import Annotated, Any

import numpy as np
from pydantic import BaseModel, ConfigDict, PlainSerializer, PlainValidator, WithJsonSchema
from pydantic_core import core_schema


class PydanticBase(BaseModel):
    """Pydantic base model to change the behaviour of pydantic globally.
    Note that setting model_config in child classes will merge the configs rather than override this one.
    https://docs.pydantic.dev/latest/concepts/config/#change-behaviour-globally
    """

    model_config = ConfigDict(
        extra="ignore",  # Ignore any extra attributes
        validate_assignment=True,  # Validate the data when the model is changed
        validate_default=True,  # Validate default values during validation
        ser_json_inf_nan="constants",  # Will serialize Infinity and NaN values as Infinity and NaN.
    )


def validate_value(value: Any) -> Any:
    """Validate (i.e. deserialize) JSON serializable value to Python type, to support complex and ndarray types."""
    if isinstance(value, dict):
        if "__complex__" in value:
            value = complex(value["real"], value["imag"])
        elif "__ndarray__" in value:
            data = base64.b64decode(value["data"])
            value = np.frombuffer(data, value["dtype"]).reshape(value["shape"])
    return value


def serialize_value(value: Any) -> Any:
    """Serialize value type to JSON serializable type, to support complex and ndarray types."""
    if isinstance(value, complex):
        value = {"__complex__": "true", "real": value.real, "imag": value.imag}
    elif isinstance(value, np.ndarray):
        # ensure array buffer is contiguous and in C order
        value = np.require(value, requirements=["A", "C"])
        data = base64.b64encode(value.data)
        value = {"__ndarray__": "true", "data": data, "dtype": str(value.dtype), "shape": value.shape}
    return value


# TODO: We might want to rename these to ObservationValue and ObservationUncertainty, respectively.
Value = Annotated[
    bool | str | int | float | complex | np.ndarray,
    PlainValidator(validate_value),
    PlainSerializer(serialize_value),
    WithJsonSchema(core_schema.any_schema()),
]


Uncertainty = Annotated[
    int | float | complex | np.ndarray,
    PlainValidator(validate_value),
    PlainSerializer(serialize_value),
    WithJsonSchema(core_schema.any_schema()),
]
