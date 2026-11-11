# Copyright 2022-2026 IQM
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

"""JSON encoding of dataclasses, circuits and arrays that restores them with their exact types."""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import importlib
from typing import Any

import numpy as np

_ALLOWED_MODULES = ("iqm.error_reduction_tools.", "iqm.pulse.")
"""Module prefixes whose dataclasses may be restored. Nothing else is imported while decoding."""


def encode(value: Any) -> Any:
    """Encode a value as JSON-compatible data that :func:`decode` restores exactly.

    Supports ``None``, booleans, numbers, strings, lists, tuples, mappings with string keys, numpy arrays and scalars,
    and dataclasses defined in this package or in ``iqm.pulse``. Fields with ``init=False`` are not stored, since the
    dataclass derives them.

    Args:
        value: The value to encode.

    Returns:
        JSON-compatible data.

    Raises:
        TypeError: If the value, or anything inside it, is not supported.

    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"__complex__": [value.real, value.imag]}
    if isinstance(value, np.ndarray):
        return {"__ndarray__": encode(value.tolist()), "dtype": str(value.dtype)}
    if isinstance(value, tuple):
        return {"__tuple__": [encode(item) for item in value]}
    if isinstance(value, list):
        return [encode(item) for item in value]
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError(f"Only mappings with string keys can be encoded, got keys {list(value)}.")
        return {"__dict__": {key: encode(item) for key, item in value.items()}}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        cls = type(value)
        return {
            "__dataclass__": f"{cls.__module__}:{cls.__qualname__}",
            "fields": {f.name: encode(getattr(value, f.name)) for f in dataclasses.fields(value) if f.init},
        }
    raise TypeError(f"Cannot encode a value of type {type(value).__name__}.")


def decode(data: Any) -> Any:
    """Restore a value written by :func:`encode`.

    Args:
        data: JSON-compatible data.

    Returns:
        The value.

    Raises:
        ValueError: If a dataclass comes from a module outside this package and ``iqm.pulse``.

    """
    if isinstance(data, list):
        return [decode(item) for item in data]
    if not isinstance(data, dict):
        return data
    if "__complex__" in data:
        real, imag = data["__complex__"]
        return complex(real, imag)
    if "__ndarray__" in data:
        return np.asarray(decode(data["__ndarray__"]), dtype=data["dtype"])
    if "__tuple__" in data:
        return tuple(decode(item) for item in data["__tuple__"])
    if "__dict__" in data:
        return {key: decode(item) for key, item in data["__dict__"].items()}
    if "__dataclass__" in data:
        cls = _dataclass(data["__dataclass__"])
        return cls(**{name: decode(item) for name, item in data["fields"].items()})
    raise ValueError(f"Unrecognized encoded value with keys {sorted(data)}.")


def _dataclass(path: str) -> type:
    """Import a dataclass by ``module:qualname``, from the allowed modules only."""
    module_name, _, qualname = path.partition(":")
    if not module_name.startswith(_ALLOWED_MODULES):
        raise ValueError(f"Refusing to restore {path!r}: only classes from {_ALLOWED_MODULES} are restored.")
    target: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        target = getattr(target, part)
    if not (isinstance(target, type) and dataclasses.is_dataclass(target)):
        raise ValueError(f"{path!r} is not a dataclass.")
    return target
