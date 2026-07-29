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

"""Pydantic compatible annotated class for sweep values."""

from typing import Annotated, Any

import numpy as np
from pydantic import PlainSerializer, PlainValidator, WithJsonSchema
from pydantic_core import core_schema


def validate_sweep_values(sweep_values: Any) -> Any:
    """Validate (i.e. deserialize) JSON serializable sweep values to Python type, to support complex types."""
    if sweep_values is None:
        return None
    if isinstance(sweep_values, np.ndarray):
        sweep_values = sweep_values.tolist()
    for index, value in enumerate(sweep_values):
        if isinstance(value, dict) and "__complex__" in value:
            sweep_values[index] = complex(value["real"], value["imag"])
    return sweep_values


def serialize_sweep_values(sweep_values: Any) -> Any:
    """Serialize sweep values type to JSON serializable type, to support complex types."""
    if sweep_values is None:
        return None
    if isinstance(sweep_values, list):
        # This is kind of a hack to clean up Numpy values from a standard list,
        # which can happen is the user converts ndarray to a list using list(array) instead of array.tolist().
        sweep_values = np.asarray(sweep_values).tolist()
    if isinstance(sweep_values, np.ndarray):
        sweep_values = sweep_values.tolist()
    for index, value in enumerate(sweep_values):
        if isinstance(value, complex):
            sweep_values[index] = {"__complex__": "true", "real": value.real, "imag": value.imag}
    return sweep_values


SweepValues = Annotated[
    list[Any] | np.ndarray,
    PlainValidator(validate_sweep_values),
    PlainSerializer(serialize_sweep_values),
    WithJsonSchema(core_schema.any_schema()),
]
