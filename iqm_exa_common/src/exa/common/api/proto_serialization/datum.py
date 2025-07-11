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

"""Convert native Python types and numpy arrays to protos and back."""

from collections.abc import Sequence

from iqm.data_definitions.common.v1.data_types_pb2 import Complex128 as dpb_Complex128
from iqm.data_definitions.common.v1.data_types_pb2 import Datum as dpb_Datum
import numpy as np

from exa.common.api.proto_serialization import array, sequence


def pack(value: None | bool | str | int | float | complex | np.ndarray | Sequence) -> dpb_Datum:
    """Packs a string, numerical value, or an array thereof into protobuf format.

    Supported data types are:
    - str
    - bool
    - int, float and complex
    - Sequences of above. Note that the type of Sequence (list, tuple...) is lost in the conversion.
    - numeric numpy arrays
    - None

    Args:
        value: The piece of data to convert.

    Returns:
        A protobuf instance that encapsulates `value`.

    Raises:
        TypeError in case of unsupported type.

    """
    if isinstance(value, np.number):
        raise TypeError(
            f"Encoding of numpy type '{type(value)}' is not supported. Cast the value into a native type first."
        )
    if value is None:
        return dpb_Datum(null_value=True)
    if isinstance(value, bool):
        return dpb_Datum(bool_value=value)
    if isinstance(value, str):
        return dpb_Datum(string_value=value)
    if isinstance(value, int):
        return dpb_Datum(sint64_value=value)
    if isinstance(value, float):
        return dpb_Datum(float64_value=value)
    if isinstance(value, complex):
        return dpb_Datum(complex128_value=dpb_Complex128(real=value.real, imag=value.imag))
    if isinstance(value, np.ndarray):
        return dpb_Datum(array=array.pack(value))
    if isinstance(value, Sequence):
        return dpb_Datum(sequence=sequence.pack(value))
    raise TypeError(f"Encoding of type '{type(value)}' is not supported.")


def unpack(source: dpb_Datum) -> None | str | bool | int | float | complex | np.ndarray | list:
    """Unpacks a protobuf into a native Python type or a numpy array. Reverse operation of :func:`.pack`.

    Args:
        source: A protobuf instance that encapsulates some data.

    Returns:
        Unpacked data.

    Raises:
        TypeError or google.protobuf.message.DecodeError in case of invalid buffer

    """
    field_name = source.WhichOneof("kind")
    if field_name == "null_value":
        return None
    if field_name == "bool_value":
        return bool(source.bool_value)
    if field_name == "complex128_value":
        return _unpack_complex128_value(source.complex128_value)
    if field_name in ("string_value", "sint64_value", "float64_value"):
        return getattr(source, field_name)
    if field_name is None:
        # If there is no value, the only possibility should be empty list. Everything else should serialize into
        # something concrete, at least null_value.
        return []
    if field_name == "array":
        return array.unpack(source.array)
    if field_name == "sequence":
        return sequence.unpack(source.sequence)
    raise TypeError(f"Unrecognized datatype field {field_name}")


def serialize(value: None | bool | str | int | float | complex | np.ndarray | Sequence) -> bytes:
    """Serialize a piece of data into a bitstring.

    Args:
        value: same as in :func:`.pack`.

    Returns:
          Bitstring that encodes `value`.

    """
    return pack(value).SerializeToString()


def deserialize(source: bytes) -> None | str | bool | int | float | complex | np.ndarray | list:
    """Deserialize a bitstring into a native Python type or a numpy array. Reverse operation of :func:`.serialize`.

    Args:
        source: Bitstring that encodes some data.

    Returns:
          Deserialized data.

    """
    proto = dpb_Datum()
    proto.ParseFromString(source)
    return unpack(proto)


def _pack_complex128(value: np.complex128 | complex, target: dpb_Complex128 | None = None) -> dpb_Complex128:
    """Packs a numpy complex128 to the respective protobuf type."""
    target = target or dpb_Complex128()
    target.real = value.real
    target.imag = value.imag
    return target


def _unpack_complex128_value(complex128_value: dpb_Complex128) -> complex:
    """Unpack a protobuf to a native complex number."""
    return complex(complex128_value.real, complex128_value.imag)
