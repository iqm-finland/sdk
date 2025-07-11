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
"""Serializers and deserializers for sweep related models."""

import json
import uuid

# FIXME: Re-enable `no-name-in-module` after pylint supports .pyi files: https://github.com/PyCQA/pylint/issues/4987
from iqm.data_definitions.common.v1.data_types_pb2 import Arrays as ArraysProto
from iqm.data_definitions.station_control.v1.sweep_request_pb2 import SweepRequest as SweepDefinitionProto
from iqm.data_definitions.station_control.v2.task_service_pb2 import SweepResultsResponse as SweepResultsResponseProto

from exa.common.api import proto_serialization
from exa.common.api.proto_serialization import array
from exa.common.data.setting_node import SettingNode
from exa.common.sweep.database_serialization import decode_and_validate_sweeps, encode_nd_sweeps
from iqm.station_control.client.serializers.datetime_serializers import deserialize_datetime, serialize_datetime
from iqm.station_control.client.serializers.playlist_serializers import pack_playlist, unpack_playlist
from iqm.station_control.interface.models import JobExecutorStatus, SweepData, SweepDefinition, SweepResults


def serialize_sweep_definition(sweep_definition: SweepDefinition) -> SweepDefinitionProto:
    """Convert SweepDefinition into sweep proto."""
    sweep_definition_proto = SweepDefinitionProto(
        sweep_id=str(sweep_definition.sweep_id),
        dut_label=sweep_definition.dut_label,
        settings=proto_serialization.setting_node.pack(sweep_definition.settings, minimal=False),
        sweeps=proto_serialization.nd_sweep.pack(sweep_definition.sweeps, minimal=False),
    )
    sweep_definition_proto.return_parameters.extend(sweep_definition.return_parameters)
    if sweep_definition.playlist is not None:
        sweep_definition_proto.playlist.MergeFrom(pack_playlist(sweep_definition.playlist))
    return sweep_definition_proto


def deserialize_sweep_definition(sweep_definition_proto: SweepDefinitionProto) -> SweepDefinition:
    """Convert sweep proto into SweepDefinition."""
    playlist = None

    if sweep_definition_proto.HasField("playlist"):
        playlist = unpack_playlist(sweep_definition_proto.playlist)

    sweep_definition = SweepDefinition(
        sweep_id=uuid.UUID(sweep_definition_proto.sweep_id),
        dut_label=sweep_definition_proto.dut_label,
        settings=proto_serialization.setting_node.unpack(sweep_definition_proto.settings),
        sweeps=proto_serialization.nd_sweep.unpack(sweep_definition_proto.sweeps),
        return_parameters=list(sweep_definition_proto.return_parameters),
        playlist=playlist,
    )
    return sweep_definition


def serialize_sweep_data(sweep_data: SweepData) -> dict:
    """Convert SweepData into JSON serializable dictionary."""
    return {
        "sweep_id": str(sweep_data.sweep_id),
        "dut_label": sweep_data.dut_label,
        "settings": sweep_data.settings.model_dump_json() if sweep_data.settings else None,
        "sweeps": encode_nd_sweeps(sweep_data.sweeps),
        "return_parameters": sweep_data.return_parameters,
        "created_timestamp": serialize_datetime(sweep_data.created_timestamp),
        "modified_timestamp": serialize_datetime(sweep_data.modified_timestamp),
        "begin_timestamp": serialize_datetime(sweep_data.begin_timestamp),
        "end_timestamp": serialize_datetime(sweep_data.end_timestamp),
        "job_status": sweep_data.job_status.value,
    }


def deserialize_sweep_data(data: dict) -> SweepData:
    """Convert JSON serializable dictionary into SweepData."""
    return SweepData(
        sweep_id=uuid.UUID(data["sweep_id"]),
        dut_label=data["dut_label"],
        settings=SettingNode(**json.loads(data["settings"])),
        sweeps=decode_and_validate_sweeps(data["sweeps"]),  # type: ignore[arg-type]
        return_parameters=data["return_parameters"],
        created_timestamp=deserialize_datetime(data["created_timestamp"]),  # type: ignore[arg-type]
        modified_timestamp=deserialize_datetime(data["modified_timestamp"]),  # type: ignore[arg-type]
        begin_timestamp=deserialize_datetime(data["begin_timestamp"]),
        end_timestamp=deserialize_datetime(data["end_timestamp"]),
        job_status=JobExecutorStatus(data["job_status"]),
    )


def serialize_sweep_results(sweep_id: uuid.UUID, sweep_results: SweepResults) -> bytes:
    """Convert SweepResults into binary string."""
    sweep_results_encoded = {}
    for key, results in sweep_results.items():
        list_of_arrays = ArraysProto()
        list_of_arrays.arrays.extend([array.pack(result) for result in results])
        sweep_results_encoded[key] = list_of_arrays
    sweep_results_response = SweepResultsResponseProto(sweep_id=str(sweep_id), results=sweep_results_encoded)
    content = sweep_results_response.SerializeToString()
    return content


def deserialize_sweep_results(sweep_results_str: bytes) -> SweepResults:
    """Convert binary string into SweepResults."""
    sweep_results_response = SweepResultsResponseProto()
    sweep_results_response.ParseFromString(sweep_results_str)
    sweep_results = {}
    for key, list_of_arrays in sweep_results_response.results.items():
        sweep_results[key] = [array.unpack(result) for result in list_of_arrays.arrays]
    return sweep_results
