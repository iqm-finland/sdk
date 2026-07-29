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
import uuid

from iqm.models.playlist import Playlist, Segment
import pytest

from exa.common.control.sweep.option import StartStopOptions
from exa.common.data.parameter import Parameter, Sweep
from exa.common.data.setting_node import SettingNode
from iqm.station_control.client.serializers import (
    deserialize_sweep_definition,
    serialize_sweep_definition,
    serialize_sweep_job_request,
)
from iqm.station_control.client.serializers.task_serializers import deserialize_sweep_job_request
from iqm.station_control.interface.models import SweepDefinition


def test_serializes_minimal():
    sweep_definition = SweepDefinition(
        sweep_id=uuid.uuid4(),
        dut_label="M138_W36_A22_N05",
        settings=SettingNode("root"),
        sweeps=[],
        return_parameters=[],
        playlist=Playlist(channel_descriptions={}, segments=[Segment()]),
    )
    serialize_sweep_job_request(sweep_definition, "some_queue")


@pytest.mark.parametrize("playlist", [Playlist(), None])
def test_serializes_with_playlist(playlist):
    sweep_definition = SweepDefinition(
        sweep_id=uuid.uuid4(),
        dut_label="M138_W36_A22_N05",
        settings=SettingNode("root"),
        sweeps=[
            (
                Sweep(parameter=Parameter("foo"), data=StartStopOptions(1, 2, 3).data),
                Sweep(parameter=Parameter("bar"), data=StartStopOptions(2, 3, 4).data),
            ),
            (Sweep(parameter=Parameter("baz"), data=[1, 2, 3]),),
        ],
        return_parameters=["readme"],
        playlist=playlist,
    )
    serialize_sweep_job_request(sweep_definition, "some_queue")


@pytest.mark.parametrize("playlist", [Playlist(), None])
def test_serialization_deserialization_isomorphism(playlist):
    sweep_definition = SweepDefinition(
        sweep_id=uuid.uuid4(),
        dut_label="M138_W36_A22_N05",
        settings=SettingNode("root"),
        sweeps=[
            (
                Sweep(parameter=Parameter("foo"), data=StartStopOptions(1, 2, 3).data),
                Sweep(parameter=Parameter("bar"), data=StartStopOptions(2, 3, 4).data),
            ),
            (Sweep(parameter=Parameter("baz"), data=[1, 2, 3]),),
        ],
        return_parameters=["readme"],
        playlist=playlist,
    )
    serialized = serialize_sweep_job_request(sweep_definition, "some_queue")
    deserialized, queue_name = deserialize_sweep_job_request(serialized)
    assert deserialized == sweep_definition
    assert queue_name == "some_queue"


@pytest.mark.parametrize("playlist", [Playlist(), None])
def test_deserializes_with_playlist(playlist):
    sweep_definition = SweepDefinition(
        sweep_id=uuid.uuid4(),
        dut_label="M138_W36_A22_N05",
        settings=SettingNode("root"),
        sweeps=[
            (
                Sweep(parameter=Parameter("foo"), data=StartStopOptions(1, 2, 3).data),
                Sweep(parameter=Parameter("bar"), data=StartStopOptions(2, 3, 4).data),
            ),
            (Sweep(parameter=Parameter("baz"), data=[1, 2, 3]),),
        ],
        return_parameters=["readme"],
        playlist=playlist,
    )
    byte_sweep = serialize_sweep_definition(sweep_definition)
    normal = deserialize_sweep_definition(byte_sweep)
    if normal.playlist is not None:
        assert isinstance(normal.playlist, Playlist)
