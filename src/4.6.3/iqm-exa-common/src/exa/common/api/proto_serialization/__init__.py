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

"""Conversions between Protocol buffers in :mod:`iqm.data_structures.common` and Python objects in exa-common.

Each submodule corresponds to a Protobuf definition from :mod:`iqm.data_structures.common`.
Each module has at least 2 functions, ``pack`` and ``unpack``, which pack the Python type into a Protobuf instance or
vice versa.

The packed objects can be nested when constructing new Protobuf objects.
It is intended that the user does the final serialization using the Protobuf service methods.
"""

from iqm.data_definitions.common.v1.setting_pb2 import SettingNode as SettingNodeProto
from iqm.data_definitions.common.v1.sweep_pb2 import CartesianSweep as CartesianSweepProto

import exa.common.api.proto_serialization.array as array
import exa.common.api.proto_serialization.datum as datum
import exa.common.api.proto_serialization.nd_sweep as nd_sweep
import exa.common.api.proto_serialization.sequence as sequence
import exa.common.api.proto_serialization.setting_node as setting_node
