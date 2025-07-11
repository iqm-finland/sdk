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

"""Convert Python Sequences to protos and back."""

from collections.abc import Sequence

from iqm.data_definitions.common.v1.data_types_pb2 import Sequence as dpb_Sequence
import numpy as np


def pack(values: Sequence) -> dpb_Sequence:
    """Packs a sequence of native Python types into protobuf format.

    Args:
        values: Sequence to convert.

    Returns:
        A protobuf instance that encapsulates `values`.

    Raises:
        ValueError in case of unsupported value.

    """
    target = dpb_Sequence()
    if not values:
        return target
    dtype = type(values[0])
    if dtype is complex or np.issubdtype(dtype, np.complexfloating):  # dtype is complex check for contingency
        target_field = target.complex128_array
        target_field.real.MergeFrom(np.real(values))
        target_field.imag.MergeFrom(np.imag(values))
        return target
    if dtype is bool:
        target_field = target.bool_array
    elif dtype is int or np.issubdtype(dtype, np.integer):
        target_field = target.int64_array
    elif dtype is float or np.issubdtype(dtype, np.floating):
        target_field = target.float64_array
    elif dtype is str:
        target_field = target.string_array
    else:
        raise TypeError(f"Unsupported numpy array type {dtype} for a sequence.")

    target_field.items.MergeFrom(values)
    return target


def unpack(source: dpb_Sequence) -> list:
    """Unpacks protobuf to list. Reverse operation of :func:`.pack`.

    Args:
        source: A protobuf instance that encapsulates some data.

    Returns:
        Unpacked data.

    Raises:
        ValueError or google.protobuf.message.DecodeError in case of invalid buffer

    """
    kind = source.WhichOneof("kind")
    if kind is None:
        return []

    if kind == "complex128_array":
        return list(
            complex(real, imag) for real, imag in zip(source.complex128_array.real, source.complex128_array.imag)
        )
    return list(getattr(source, kind).items)
