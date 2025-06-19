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
"""Pydantic related models and types."""

from pydantic import BaseModel, ConfigDict


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
