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
"""Serializers and deserializers for run related models."""

import uuid

# FIXME: Re-enable `no-name-in-module` after pylint supports .pyi files: https://github.com/PyCQA/pylint/issues/4987
from iqm.data_definitions.station_control.v1.sweep_request_pb2 import SweepRequest as SweepDefinitionProto
from iqm.data_definitions.station_control.v2.run_definition_pb2 import RunDefinition as RunDefinitionProto

from exa.common.api import proto_serialization
from exa.common.sweep.database_serialization import decode_and_validate_sweeps, encode_nd_sweeps
from exa.common.sweep.util import convert_sweeps_to_list_of_tuples
from iqm.station_control.client.serializers.datetime_serializers import deserialize_datetime, serialize_datetime
from iqm.station_control.client.serializers.struct_serializer import deserialize_struct, serialize_struct
from iqm.station_control.client.serializers.sweep_serializers import (
    deserialize_sweep_data,
    deserialize_sweep_definition,
    serialize_sweep_data,
    serialize_sweep_definition,
)
from iqm.station_control.interface.models import RunData, RunDefinition


def serialize_run_definition(run_definition: RunDefinition) -> RunDefinitionProto:
    """Convert RunDefinition into run proto."""
    run_definition_proto = RunDefinitionProto(
        run_id=str(run_definition.run_id),
        username=run_definition.username,
        experiment_name=run_definition.experiment_name,
        experiment_label=run_definition.experiment_label,
        options=serialize_struct(run_definition.options),  # type: ignore[arg-type]
        additional_run_properties=serialize_struct(run_definition.additional_run_properties),  # type: ignore[arg-type]
        software_version_set_id=run_definition.software_version_set_id,
        components=run_definition.components,
        default_data_parameters=run_definition.default_data_parameters,
        default_sweep_parameters=run_definition.default_sweep_parameters,
    )
    run_definition_proto.sweep_definition_payload.Pack(
        serialize_sweep_definition(run_definition.sweep_definition), type_url_prefix="iqm-data-definitions"
    )
    for key, sweep in run_definition.hard_sweeps.items():  # type: ignore[union-attr]
        run_definition_proto.hard_sweeps[key].CopyFrom(proto_serialization.nd_sweep.pack(sweep, minimal=False))
    return run_definition_proto


def deserialize_run_definition(run_definition_proto: RunDefinitionProto) -> RunDefinition:
    """Convert run proto into RunDefinition."""
    if run_definition_proto.sweep_definition.sweep_id:
        # Old, unpacked format. Remove when clients don't fill this field anymore.
        sweep_definition_pb = run_definition_proto.sweep_definition
    else:
        sweep_definition_pb = SweepDefinitionProto()
        run_definition_proto.sweep_definition_payload.Unpack(sweep_definition_pb)

    run_definition = RunDefinition(
        run_id=uuid.UUID(run_definition_proto.run_id),
        username=run_definition_proto.username,
        experiment_name=run_definition_proto.experiment_name,
        experiment_label=run_definition_proto.experiment_label,
        options=deserialize_struct(run_definition_proto.options),
        additional_run_properties=deserialize_struct(run_definition_proto.additional_run_properties),
        software_version_set_id=run_definition_proto.software_version_set_id,
        hard_sweeps={
            key: proto_serialization.nd_sweep.unpack(hard_sweep)
            for key, hard_sweep in run_definition_proto.hard_sweeps.items()
        },
        components=list(run_definition_proto.components),
        default_data_parameters=list(run_definition_proto.default_data_parameters),
        default_sweep_parameters=list(run_definition_proto.default_sweep_parameters),
        sweep_definition=deserialize_sweep_definition(sweep_definition_pb),
    )
    return run_definition


def serialize_run_data(run_data: RunData) -> dict:
    """Convert RunData object to a JSON serializable dictionary."""
    return {
        "run_id": str(run_data.run_id),
        "username": run_data.username,
        "experiment_name": run_data.experiment_name,
        "experiment_label": run_data.experiment_label,
        "options": run_data.options,
        "additional_run_properties": run_data.additional_run_properties,
        "software_version_set_id": run_data.software_version_set_id,
        "hard_sweeps": {key: encode_nd_sweeps(value) for key, value in run_data.hard_sweeps.items()},  # type: ignore[union-attr]
        "components": run_data.components,
        "default_data_parameters": run_data.default_data_parameters,
        "default_sweep_parameters": run_data.default_sweep_parameters,
        "sweep_data": serialize_sweep_data(run_data.sweep_data),
        "created_timestamp": serialize_datetime(run_data.created_timestamp),
        "modified_timestamp": serialize_datetime(run_data.modified_timestamp),
        "begin_timestamp": serialize_datetime(run_data.begin_timestamp),
        "end_timestamp": serialize_datetime(run_data.end_timestamp),
    }


def deserialize_run_data(data: dict) -> RunData:
    """Convert a JSON serializable dictionary to RunData object."""
    return RunData(
        run_id=uuid.UUID(data["run_id"]),
        username=data["username"],
        experiment_name=data["experiment_name"],
        experiment_label=data["experiment_label"],
        options=data["options"],
        additional_run_properties=data["additional_run_properties"],
        software_version_set_id=data["software_version_set_id"],
        hard_sweeps={
            key: convert_sweeps_to_list_of_tuples(decode_and_validate_sweeps(value))
            for key, value in data["hard_sweeps"].items()
        },
        components=data["components"],
        default_data_parameters=data["default_data_parameters"],
        default_sweep_parameters=data["default_sweep_parameters"],
        sweep_data=deserialize_sweep_data(data["sweep_data"]),
        created_timestamp=deserialize_datetime(data["created_timestamp"]),  # type: ignore[arg-type]
        modified_timestamp=deserialize_datetime(data["modified_timestamp"]),  # type: ignore[arg-type]
        begin_timestamp=deserialize_datetime(data["begin_timestamp"]),  # type: ignore[arg-type]
        end_timestamp=deserialize_datetime(data["end_timestamp"]),
    )
