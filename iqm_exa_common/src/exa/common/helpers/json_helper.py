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

import base64
from collections.abc import Callable
from typing import Any

import numpy as np


def get_json_encoder() -> dict[Any, Callable[..., dict[str, Any]]]:
    """Custom JSON encoder for complex number, ndarray or tuple

    Can be used in situation when serialization of JSON can be customised
    (json_encoders config in pydantic models, custom_encoder in jsonable_encoder
    of fastapi, etc.)
    """

    def _encode_complex(obj: complex) -> dict[str, Any]:
        return {"__complex__": "true", "real": obj.real, "imag": obj.imag}

    def _encode_ndarray(obj: np.ndarray) -> dict[str, Any]:
        data_b64 = base64.b64encode(obj.data)
        return {"__ndarray__": "true", "data": data_b64, "dtype": str(obj.dtype), "shape": obj.shape}

    def _encode_tuple(obj: tuple) -> dict[str, Any]:
        return {"__tuple__": "true", "data": obj}

    return {complex: _encode_complex, np.ndarray: _encode_ndarray, tuple: _encode_tuple}


def decode_json(obj: Any) -> Any:
    """Custom json decoder for object, in case it is represented as
    complex number, ndarray or tuple.
    Otherwise decoder won't be applied

    Here is an example of encoded/decoded json with complex number and ndarray:

    .. doctest::

        >>> import base64
        >>> import numpy as np
        >>> from exa.common.helpers.json_helper import decode_json
        >>> complex_encoded = {'__complex__': True, 'real': 3, 'imag': 4}
        >>> decode_json(complex_encoded)
        (3+4j)

        >>> data_b64 = base64.b64encode(np.arange(4).reshape(2, 2))
        >>> ndarray_encoded = {'__ndarray__': True, 'data': data_b64.decode('utf-8'), 'dtype': 'int64', 'shape':[2, 2]}
        >>> decode_json(ndarray_encoded)
        array([[0, 1],
               [2, 3]])
    """
    if isinstance(obj, dict):
        if "__complex__" in obj:
            return complex(obj["real"], obj["imag"])
        elif "__ndarray__" in obj:
            data = base64.b64decode(obj["data"])
            return np.frombuffer(data, obj["dtype"]).reshape(obj["shape"])
        elif "__tuple__" in obj:
            return tuple(obj["data"])
    return obj
