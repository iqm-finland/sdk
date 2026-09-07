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
"""Serializers and deserializers for setting node related models."""

# FIXME: Re-enable `no-name-in-module` after pylint supports .pyi files: https://github.com/PyCQA/pylint/issues/4987
from iqm.data_definitions.common.v1.setting_pb2 import SettingNode as SettingNodeProto

from exa.common.api import proto_serialization
from exa.common.data.setting_node import SettingNode


def deserialize_setting_node(setting_node_str: bytes) -> SettingNode:
    """Convert binary string into SettingNode."""
    proto = SettingNodeProto()
    proto.ParseFromString(setting_node_str)
    return proto_serialization.setting_node.unpack(proto)
