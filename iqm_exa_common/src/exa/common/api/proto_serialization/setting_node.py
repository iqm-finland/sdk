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

"""Convert SettingNodes to protos and back."""

from iqm.data_definitions.common.v1.setting_pb2 import SettingNode as spb_SettingNode

from exa.common.api.proto_serialization import datum
import exa.common.api.proto_serialization._parameter as param_proto
from exa.common.data.parameter import DataType, Parameter
from exa.common.data.setting_node import Setting, SettingNode
from exa.common.helpers.numpy_helper import coerce_numpy_type_to_native


def _pack_setting(setting: Setting, optimize: bool) -> spb_SettingNode.Setting:
    """Convert a Setting into protobuf representation."""
    value = coerce_numpy_type_to_native(setting.value)
    try:
        packed = datum.pack(value)
    except ValueError as err:
        raise ValueError(
            f"Failed to convert a value to protobuf. Value={setting.value}, Parameter={setting.parameter}."
        ) from err
    if optimize:
        return spb_SettingNode.Setting(parameter_name=setting.name, value=packed)
    return spb_SettingNode.Setting(parameter=param_proto.pack(setting.parameter), value=packed)


def _unpack_setting(proto: spb_SettingNode.Setting) -> Setting:
    """Convert protobuf representation into a Setting."""
    if proto.WhichOneof("parameter_desc") == "parameter":
        parameter = param_proto.unpack(proto.parameter)
    else:
        parameter = Parameter(name=proto.parameter_name, data_type=DataType.ANYTHING)
    try:
        value = datum.unpack(proto.value)
    except Exception as err:
        raise AttributeError(f"Unpacking of {parameter} {proto.value} failed.") from err

    return Setting(parameter, value)  # type: ignore[arg-type]


def pack(node: SettingNode, minimal: bool) -> spb_SettingNode:
    """Convert a SettingNode into protobuf representation.

    Silently coerces some datatypes to be compatible with the proto definition of ``Datum``:
    - Numpy arrays of 32-bit ints are converted to 64-bits (Windows only).
    - Singular numpy types are converted into corresponding native types.

    Args:
        node: SettingNode to pack, recursively.
        minimal: If True, only the :attr:`.Parameter.name` of each Setting is preserved along with the setting value.
            If False, the the whole :attr:`.Setting.parameter` is packed.

    Returns:
        Protobuf instance that represents `node`.

    """
    settings = {key: _pack_setting(item, minimal) for key, item in node.child_settings}
    nodes = {key: pack(item, minimal) for key, item in node.child_nodes}
    return spb_SettingNode(name=node.name, settings=settings, subnodes=nodes)


def unpack(proto: spb_SettingNode) -> SettingNode:
    """Convert protobuf representation into a SettingNode. Reverse operation of :func:`.pack`

    Args:
        proto: Protobuf instance to unpack, recursively.

    Returns:
        Unpacked SettingNode. In case `proto` only contains the parameter names (see ``optimize`` in
        :func:`.pack`), dummy Parameters are generated.

    """
    settings = {key: _unpack_setting(content) for key, content in proto.settings.items()}
    nodes = {key: unpack(content) for key, content in proto.subnodes.items()}
    # Names are currently NEVER aligned with the paths when deserializing. This is safe to do, since currently nothing
    # in the server-side assumes path==name, but if such logic is added this needs to be reconsidered.
    return SettingNode(name=proto.name, **(settings | nodes), align_name=False)  # type: ignore[arg-type]  # type: ignore[arg-type]  # type: ignore[arg-type]
