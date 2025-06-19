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
"""Serializers and deserializers for struct related models.

We use custom Struct model instead of standard Struct from protobuf,
since standard Struct doesn't support integers but instead casts them always to floats.
"""

import iqm.data_definitions.common.v1.struct_pb2 as pb


def serialize_struct(data: dict) -> pb.Struct:
    """Serialize a free-form dict into a Struct protobuf representation."""
    proto = pb.Struct()
    for key, value in data.items():
        proto.fields[key].CopyFrom(_serialize_value(value))
    return proto


def deserialize_struct(proto: pb.Struct) -> dict:
    """Deserialize a Struct protobuf representation into a free-form dict."""
    return {key: _deserialize_value(value) for key, value in proto.fields.items()}


def _serialize_value(value: None | float | str | bool | dict | list | int) -> pb.Value:
    """Serialize a value into a Value protobuf representation."""
    if value is None:
        return pb.Value(null_value=True)
    if isinstance(value, float):
        return pb.Value(number_value=value)
    if isinstance(value, str):
        return pb.Value(string_value=value)
    if isinstance(value, bool):
        return pb.Value(bool_value=value)
    if isinstance(value, dict):
        return pb.Value(struct_value=serialize_struct(value))
    if isinstance(value, list):
        return pb.Value(list_value=_serialize_list(value))
    if isinstance(value, int):
        return pb.Value(integer_value=value)
    raise TypeError(f"Serializing of type '{type(value)}' is not supported.")


def _deserialize_value(proto: pb.Value) -> None | float | str | bool | dict | list | int:
    """Deserialize a Value protobuf representation into a value."""
    which = proto.WhichOneof("kind")
    if which == "null_value":
        return None
    if which == "number_value":
        return proto.number_value
    if which == "string_value":
        return proto.string_value
    if which == "bool_value":
        return proto.bool_value
    if which == "struct_value":
        return deserialize_struct(proto.struct_value)
    if which == "list_value":
        return _deserialize_list(proto.list_value)
    if which == "integer_value":
        return proto.integer_value
    raise TypeError(f"Unrecognized datatype field {which}")


def _serialize_list(data: list) -> pb.ListValue:
    """Serialize a list into a ListValue protobuf representation."""
    proto = pb.ListValue()
    for value in data:
        proto.values.append(_serialize_value(value))
    return proto


def _deserialize_list(proto: pb.ListValue) -> list:
    """Deserialize a ListValue protobuf representation into a list."""
    return [_deserialize_value(value) for value in proto.values]
