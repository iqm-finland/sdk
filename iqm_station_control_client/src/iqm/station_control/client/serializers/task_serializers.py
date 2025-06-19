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
"""Serializers and deserializers for task related models."""

from typing import Any
import uuid

from iqm.data_definitions.station_control.v1.sweep_request_pb2 import SweepRequest as SweepDefinitionProto

# FIXME: Re-enable `no-name-in-module` after pylint supports .pyi files: https://github.com/PyCQA/pylint/issues/4987
from iqm.data_definitions.station_control.v1.task_service_pb2 import SweepTaskRequest as SweepTaskRequestProto

from iqm.station_control.client.serializers.run_serializers import serialize_run_definition
from iqm.station_control.client.serializers.sweep_serializers import (
    deserialize_sweep_definition,
    serialize_sweep_definition,
)
from iqm.station_control.interface.models import RunDefinition, SweepDefinition


def serialize_run_job_request(run_definition: RunDefinition, queue_name: str) -> bytes:
    """Wrap `run_definition` and `queue_name` into a protobuf message and serialize into a bitstring.

    Args:
        run_definition: The content of the run.
        queue_name: Name of the destination queue.

    Returns:
        :class:`~iqm.data_definitions.station_control.v1.task_service_pb2.SweepTaskRequest`
        encoded into a bitstring.

    """
    payload = serialize_run_definition(run_definition)
    return _serialize_job_request(payload, queue_name, run_definition.sweep_definition.sweep_id)


def serialize_sweep_job_request(sweep_definition: SweepDefinition, queue_name: str) -> bytes:
    """Wrap `sweep_definition` and `queue_name` into a protobuf message and serialize into a bitstring.

    Args:
        sweep_definition: The content of the sweep.
        queue_name: Name of the destination queue.

    Returns:
        :class:`~iqm.data_definitions.station_control.v1.task_service_pb2.SweepTaskRequest`
        encoded into a bitstring.

    """
    payload = serialize_sweep_definition(sweep_definition)
    return _serialize_job_request(payload, queue_name, sweep_definition.sweep_id)


def _serialize_job_request(payload: Any, queue_name: str, sweep_id: uuid.UUID) -> bytes:
    sweep_job_request_proto = SweepTaskRequestProto(queue_name=queue_name, sweep_id=str(sweep_id))
    sweep_job_request_proto.payload.Pack(payload, type_url_prefix="iqm-data-definitions")
    return sweep_job_request_proto.SerializeToString()


def deserialize_sweep_job_request(data: bytes) -> tuple[SweepDefinition, str]:
    """Deserializes `sweep_definition` and `queue_name` from the serialized bitstring.

    Args:
        data: The serialized data

    Returns:
        Deserialized tuple :class:`~iqm.station_control.interface.model.SweepDefinition
        and queue name (string).

    """
    sweep_task_request_proto = SweepTaskRequestProto()
    sweep_task_request_proto.ParseFromString(data)
    sweep_definition_proto = SweepDefinitionProto()
    if sweep_task_request_proto.payload.Unpack(sweep_definition_proto) is False:
        raise ValueError("Can't unpack SweepDefinition from task TaskRequest")

    sweep_definition = deserialize_sweep_definition(sweep_definition_proto)
    queue_name = sweep_task_request_proto.queue_name
    return sweep_definition, queue_name
