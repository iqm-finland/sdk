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

"""Pack and unpack parameters to protos and back."""

from iqm.data_definitions.common.v1.parameter_pb2 import Parameter as ppb_Parameter

from exa.common.data.parameter import CollectionType, DataType, Parameter

_COLLECTION_TYPES = {
    CollectionType.SCALAR: ppb_Parameter.CollectionType.COLLECTION_TYPE_SCALAR,
    CollectionType.LIST: ppb_Parameter.CollectionType.COLLECTION_TYPE_SEQUENCE,
    CollectionType.NDARRAY: ppb_Parameter.CollectionType.COLLECTION_TYPE_ARRAY,
}
_collection_types_inv = {v: k for k, v in _COLLECTION_TYPES.items()}

_DATA_TYPES = {
    DataType.ANYTHING: ppb_Parameter.DataType.DATA_TYPE_UNSPECIFIED,
    DataType.NUMBER: ppb_Parameter.DataType.DATA_TYPE_FLOAT64,  # Deprecated
    DataType.STRING: ppb_Parameter.DataType.DATA_TYPE_STRING,
    DataType.COMPLEX: ppb_Parameter.DataType.DATA_TYPE_COMPLEX128,
    DataType.BOOLEAN: ppb_Parameter.DataType.DATA_TYPE_BOOL,
    DataType.INT: ppb_Parameter.DataType.DATA_TYPE_INT64,
    DataType.FLOAT: ppb_Parameter.DataType.DATA_TYPE_FLOAT64,
}
_data_types_inv = {v: k for k, v in _DATA_TYPES.items() if k != DataType.NUMBER}


def pack(parameter: Parameter) -> ppb_Parameter:
    """Convert Parameter into protobuf representation."""
    return ppb_Parameter(
        name=parameter.name,
        label=parameter.label,
        unit=parameter.unit,
        data_type=_DATA_TYPES.get(parameter.data_type, ppb_Parameter.DataType.DATA_TYPE_UNSPECIFIED),  # type: ignore[arg-type]
        collection_type=_COLLECTION_TYPES[parameter.collection_type],
        element_indices=parameter.element_indices,
        parent_name=parameter.parent_name,
        parent_label=parameter.parent_label,
    )


def unpack(proto: ppb_Parameter) -> Parameter:
    """Convert protobuf representation into a Parameter."""
    if not proto.element_indices:
        return Parameter(
            name=proto.name,
            label=proto.label,
            unit=proto.unit,
            data_type=_data_types_inv.get(proto.data_type),  # type: ignore[arg-type]
            collection_type=_collection_types_inv.get(proto.collection_type),  # type: ignore[arg-type]
        )
    return Parameter(
        name=proto.parent_name,
        label=proto.parent_label,
        unit=proto.unit,
        data_type=_data_types_inv.get(proto.data_type),  # type: ignore[arg-type]
        collection_type=_collection_types_inv.get(proto.collection_type),  # type: ignore[arg-type]
        element_indices=list(proto.element_indices),
    )
