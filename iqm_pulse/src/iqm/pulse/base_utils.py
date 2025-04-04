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
from typing import Any

import numpy as np


def merge_dicts(A: dict, B: dict, path=(), merge_nones: bool = True) -> dict:
    """Merge two dictionaries recursively, leaving the originals unchanged.

    Args:
        A: dictionary
        B: another dictionary
        merge_nones: whether to also merge ``None`` and empty ``Sized`` values from B to A.

    Returns:
        copy of A, with the contents of B merged in (and taking precedence) recursively

    """

    def is_not_empty(val: Any) -> bool:
        if val is None:
            return False
        if isinstance(val, Sized) and len(val) == 0:
            return False
        return True

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
