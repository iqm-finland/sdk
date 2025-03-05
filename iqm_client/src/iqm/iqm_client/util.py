# Copyright 2021-2023 IQM client developers
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
"""
Helpful utilities that can be used together with IQMClient.
"""
from json import JSONEncoder, dumps, loads
from typing import Any, TypeVar

import numpy as np

T = TypeVar('T')


class IQMJSONEncoder(JSONEncoder):
    """JSONEncoder that that adds support for some non-JSON datatypes"""

    def default(self, o: Any):
        if isinstance(o, np.ndarray):
            return o.tolist()
        return JSONEncoder.default(self, o)


def to_json_dict(obj: dict[str, Any]) -> dict:
    """Convert a dict to JSON serializable dict

    Args:
        obj: dict to convert

    Returns:
        dict containing converted data

    Raises:
        ValueError if the original dict contains unsupported datatypes"""
    try:
        return loads(dumps(obj, allow_nan=False, cls=IQMJSONEncoder))
    except (ValueError, TypeError) as e:
        raise ValueError('Object contains values that are not JSON serializable') from e
