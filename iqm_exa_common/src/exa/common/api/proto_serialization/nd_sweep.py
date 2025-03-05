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

"""Convert NdSweeps to protos and back."""

import iqm.data_definitions.common.v1.sweep_pb2 as spb

from exa.common.api.proto_serialization import sequence
import exa.common.api.proto_serialization._parameter as param_proto
from exa.common.control.sweep.sweep import Sweep
from exa.common.data.parameter import DataType, Parameter
from exa.common.sweep.util import NdSweep


def pack(nd_sweep: NdSweep, minimal: bool = True) -> spb.CartesianSweep:
    """Convert an NdSweep into protobuf representation.

    Note: The protobuf does not make any distinction between different types of Sweeps, so the type information is lost.
    """
    parallels = []
    for parallel in nd_sweep:
        parallel_proto = spb.ParallelSweep()
        parallel_proto.single_parameter_sweeps.MergeFrom((_pack_single_sweep(sweep, minimal) for sweep in parallel))
        parallels.append(parallel_proto)
    proto = spb.CartesianSweep()
    proto.parallel_sweeps.MergeFrom(reversed(parallels))  # In data-definitions, order is outermost loop first
    return proto


def unpack(proto: spb.CartesianSweep) -> NdSweep:
    """Convert protobuf representation into a NdSweep. Reverse operation of :func:`.pack`."""
    nd_sweep = []
    for parallel_proto in reversed(proto.parallel_sweeps):
        parallel = tuple(_unpack_single_sweep(sweep) for sweep in parallel_proto.single_parameter_sweeps)
        nd_sweep.append(parallel)
    return nd_sweep


def _pack_single_sweep(sweep: Sweep, minimal: bool) -> spb.SingleParameterSweep:
    kwargs = {"parameter_name": sweep.parameter.name, "values": sequence.pack(sweep.data)}
    if not minimal:
        kwargs["parameter"] = param_proto.pack(sweep.parameter)
    return spb.SingleParameterSweep(**kwargs)


def _unpack_single_sweep(proto: spb.SingleParameterSweep) -> Sweep:
    sweep_values = sequence.unpack(proto.values)
    if proto.HasField("parameter"):
        parameter = param_proto.unpack(proto.parameter)
    else:
        parameter = Parameter(proto.parameter_name, data_type=DataType.ANYTHING)
    return Sweep(parameter=parameter, data=sweep_values)
