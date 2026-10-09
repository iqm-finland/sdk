#  ********************************************************************************
#
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
"""Base utility functions with no dependencies on other iqm.pulse modules."""

from __future__ import annotations

from collections.abc import Sized
import copy
from typing import Any, get_args, get_origin

import numpy as np

from exa.common.data.parameter import CollectionType, DataType


def merge_dicts(
    A: dict[str, Any], B: dict[str, Any], path: tuple[str, ...] = (), merge_nones: bool = True
) -> dict[str, Any]:
    """Merge two dictionaries recursively, leaving the originals unchanged.

    Args:
        A: Dictionary.
        B: Another dictionary.
        path: Current path into the nested dictionaries, for error reporting.
        merge_nones: Whether to also merge ``None`` and empty ``Sized`` values from B to A.

    Returns:
        Copy of A, with the contents of B merged in (and taking precedence) recursively.

    """

    def is_not_empty(val: Any) -> bool:
        if val is None:
            return False
        return not (isinstance(val, Sized) and len(val) == 0)

    # A and B must be left intact, make a shallow copy
    A = copy.copy(A)
    for key, vb in B.items():
        if (va := A.get(key)) is not None:
            new_path = (*path, key)
            if isinstance(va, dict):
                if isinstance(vb, dict):
                    # replace the dict with the shallow copy returned by merge_dicts
                    A[key] = merge_dicts(va, vb, new_path, merge_nones=merge_nones)
                    continue
                raise ValueError(f"Merging dict with scalar: {'.'.join(new_path)}")
            if isinstance(vb, dict):
                raise ValueError(f"Merging scalar with dict: {'.'.join(new_path)}")
        # scalar overrides scalar, or a new key is inserted
        if merge_nones or is_not_empty(vb):
            A[key] = vb
    return A  # return the shallow copy


def _dicts_differ(a: Any, b: Any) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return True
        return any(_dicts_differ(a[key], b[key]) for key in a)
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return not np.array_equal(a, b)
    return a != b


def map_waveform_param_types(type_hint: type) -> tuple[DataType, CollectionType]:
    """Map a Python typehint into EXA Parameter's ``(DataType, CollectionType)`` tuple.

    Args:
        type_hint: Python typehint.

    Returns:
        Corresponding EXA :class:`.Parameter` type.

    Raises:
        ValueError: for a non-supported type.

    """
    value_error = ValueError(f"Nonsupported datatype for a waveform parameter: {type_hint}")
    if hasattr(type_hint, "__iter__") and type_hint is not str:
        if type_hint == np.ndarray:
            data_type = DataType.COMPLEX  # due to np.ndarray not being generic we assume complex numbers
            collection_type = CollectionType.NDARRAY
            return (data_type, collection_type)
        if get_origin(type_hint) is list:
            collection_type = CollectionType.LIST
            type_hint = get_args(type_hint)[0]
        else:
            raise value_error
    else:
        collection_type = CollectionType.SCALAR

    if type_hint is float:
        data_type = DataType.FLOAT
    elif type_hint is int:
        data_type = DataType.INT
    elif type_hint is str:
        data_type = DataType.STRING
    elif type_hint is complex:
        data_type = DataType.COMPLEX
    elif type_hint is bool:
        data_type = DataType.BOOLEAN
    else:
        raise value_error
    return data_type, collection_type


def normalize_angle(angle: float) -> float:
    """Normalize the given angle to (-pi, pi].

    Args:
        angle: angle to normalize (in radians)

    Returns:
        ``angle`` normalized to (-pi, pi]

    """
    half_turn = np.pi
    full_turn = 2 * half_turn
    return (angle - half_turn) % -full_turn + half_turn
