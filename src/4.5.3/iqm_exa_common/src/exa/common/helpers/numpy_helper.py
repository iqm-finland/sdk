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

"""Coercion of numpy types."""

from typing import Any

import numpy as np


def coerce_numpy_type_to_native(value: Any) -> Any:
    """Convert numpy types to underlying native types, and Windows-specific int32 arrays to int64s."""
    if isinstance(value, np.ndarray) and value.dtype == np.int32:
        return value.astype(dtype=np.int64, casting="same_kind")

    if isinstance(value, (np.number, np.bool_)):
        return value.item()  # return the closest native type.

    return value
