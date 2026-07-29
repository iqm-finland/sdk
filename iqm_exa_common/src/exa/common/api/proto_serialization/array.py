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

"""Convert numpy arrays to protos and back."""

from iqm.data_definitions.common.v1.data_types_pb2 import Array as dpb_Array
import numpy as np


def pack(array: np.ndarray) -> dpb_Array:
    """Packs a numeric numpy array into protobuf format.

    Args:
        array: Numpy array to convert.

    Returns:
        A protobuf instance that encapsulates `array`.

    """
    target = dpb_Array()
    target.shape.MergeFrom(array.shape)
    if not array.size:  # MergeFrom throws with 0-sized iterables
        return target

    dtype_type = array.dtype.type
    if dtype_type == np.complex128:
        target.complex128_array.real.MergeFrom(map(np.ndarray.item, np.nditer(array.real, order="C")))  # type:ignore
        target.complex128_array.imag.MergeFrom(map(np.ndarray.item, np.nditer(array.imag, order="C")))  # type:ignore
        return target

    if dtype_type == np.bool_:
        target_field = target.bool_array
    elif dtype_type == np.int64:
        target_field = target.int64_array
    elif dtype_type == np.int32:
        target_field = target.int64_array
    elif dtype_type == np.float64:
        target_field = target.float64_array
    else:
        raise TypeError(f"Unsupported numpy array type {dtype_type} for an array.")

    target_field.items.MergeFrom(map(np.ndarray.item, np.nditer(array, order="C")))  # type:ignore
    return target


def unpack(source: dpb_Array) -> np.ndarray:
    """Unpacks protobuf to array. Reverse operation of :func:`.pack`.

    Args:
        source: A protobuf instance that encapsulates some data.

    Returns:
        Unpacked data.

    Raises:
        ValueError or google.protobuf.message.DecodeError in case of invalid buffer

    """
    kind = source.WhichOneof("kind")
    if kind is None:
        return np.array([])

    shape = tuple(source.shape)
    if kind == "complex128_array":
        size = np.prod(shape).item()
        array = np.empty((size,), dtype=complex)
        array.real = np.fromiter(source.complex128_array.real, dtype=np.float64, count=size)
        array.imag = np.fromiter(source.complex128_array.imag, dtype=np.float64, count=size)
    elif kind == "bool_array":
        array = np.fromiter(source.bool_array.items, np.bool_)
    elif kind == "int64_array":
        array = np.fromiter(source.int64_array.items, np.int64)
    elif kind == "float64_array":
        array = np.fromiter(source.float64_array.items, np.float64)
    else:
        raise TypeError(f"Cannot unpack value in field {kind}. Field name not recognized.")

    return array.reshape(*shape)
