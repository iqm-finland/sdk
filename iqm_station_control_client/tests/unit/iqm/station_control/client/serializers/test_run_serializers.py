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

from exa.common.control.sweep.option import StartStopOptions
from exa.common.data.parameter import Parameter, Sweep
from exa.common.data.setting_node import SettingNode
from iqm.station_control.client.serializers import (
    deserialize_run_definition,
    serialize_run_definition,
    serialize_run_job_request,
)
from iqm.station_control.interface.models import RunDefinition, SweepDefinition


def test_serializes_optional():
    run_definition = RunDefinition(
        run_id=uuid.uuid4(),
        username="username",
        experiment_name="experiment_name",
        experiment_label="experiment_label",
        components=["QB3", "QB5"],
        default_sweep_parameters=["sweep_parameter"],
        default_data_parameters=["data_parameter"],
        sweep_definition=SweepDefinition(
            sweep_id=uuid.uuid4(),
            dut_label="M138_W36_A22_N05",
            settings=SettingNode("root"),
            sweeps=[],
            return_parameters=[],
            playlist=Playlist(channel_descriptions={}, segments=[Segment()]),
        ),
    )
    serialize_run_job_request(run_definition, "some_queue")


def test_serializes_optional_round_trip():
    run_definition_before = RunDefinition(
        run_id=uuid.uuid4(),
        username="username",
        experiment_name="experiment_name",
        experiment_label="experiment_label",
        components=["QB3", "QB5"],
        default_sweep_parameters=["sweep_parameter"],
        default_data_parameters=["data_parameter"],
        sweep_definition=SweepDefinition(
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
            playlist=Playlist(channel_descriptions={}, segments=[Segment()]),
        ),
    )
    data = serialize_run_definition(run_definition_before)
    run_definition_after = deserialize_run_definition(data)

    assert run_definition_after.options == {}
    assert run_definition_after.additional_run_properties == {}
    assert run_definition_after.software_version_set_id == 0
    assert run_definition_after.hard_sweeps == {}


def test_serializes_minimal():
    run_definition = RunDefinition(
        run_id=uuid.uuid4(),
        username="username",
        experiment_name="experiment_name",
        experiment_label="experiment_label",
        options={"option_1": 1},
        additional_run_properties={"additional_run_property_1": "test"},
        software_version_set_id=1,
        hard_sweeps={"readout_1.result": []},
        components=["QB3", "QB5"],
        default_sweep_parameters=["sweep_parameter"],
        default_data_parameters=["data_parameter"],
        sweep_definition=SweepDefinition(
            sweep_id=uuid.uuid4(),
            dut_label="M138_W36_A22_N05",
            settings=SettingNode("root"),
            sweeps=[],
            return_parameters=[],
            playlist=Playlist(channel_descriptions={}, segments=[Segment()]),
        ),
    )
    serialize_run_job_request(run_definition, "some_queue")


def test_serializes_complex():
    run_definition = RunDefinition(
        run_id=uuid.uuid4(),
        username="username",
        experiment_name="experiment_name",
        experiment_label="experiment_label",
        options={"option_1": 1},
        additional_run_properties={"additional_run_property_1": "test"},
        software_version_set_id=1,
        hard_sweeps={"readout_1.result": []},
        components=["QB3", "QB5"],
        default_sweep_parameters=["sweep_parameter"],
        default_data_parameters=["data_parameter"],
        sweep_definition=SweepDefinition(
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
            playlist=Playlist(channel_descriptions={}, segments=[Segment()]),
        ),
    )
    serialize_run_job_request(run_definition, "some_queue")


def test_serializes_complex_round_trip():
    run_definition_before = RunDefinition(
        run_id=uuid.uuid4(),
        username="username",
        experiment_name="experiment_name",
        experiment_label="experiment_label",
        options={
            "option_1": 1,
            "option_2": 2.0,
        },
        additional_run_properties={
            "additional_run_property_0": None,
            "additional_run_property_1": 1,
            "additional_run_property_2": 2.0,
            "additional_run_property_3": ["a", "b", "c"],
            "additional_run_property_4": [[1, 2], [1], [2.0], [["a", "b"], [1, 1.0]]],
            "additional_run_property_5": {
                "a_result": "a_value",
                "b_result": "b_value",
                "c_result": "c_value",
            },
        },
        software_version_set_id=1,
        hard_sweeps={"readout_1.result": []},
        components=["QB3", "QB5"],
        default_sweep_parameters=["sweep_parameter"],
        default_data_parameters=["data_parameter"],
        sweep_definition=SweepDefinition(
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
            playlist=Playlist(channel_descriptions={}, segments=[Segment()]),
        ),
    )

    # Fix SweepDefinition round-trip serialization
    # assert run_definition_before.sweep_definition == run_definition_after.sweep_definition
    # assert run_definition_before == run_definition_after

    assert isinstance(run_definition_before.options["option_1"], int)
    assert run_definition_before.options["option_1"] == 1
    assert isinstance(run_definition_before.options["option_2"], float)
    assert run_definition_before.options["option_2"] == 2.0
    assert isinstance(run_definition_before.additional_run_properties["additional_run_property_1"], int)
    assert run_definition_before.additional_run_properties["additional_run_property_1"] == 1
    assert isinstance(run_definition_before.additional_run_properties["additional_run_property_2"], float)
    assert run_definition_before.additional_run_properties["additional_run_property_2"] == 2.0

    data = serialize_run_definition(run_definition_before)
    run_definition_after = deserialize_run_definition(data)

    assert isinstance(run_definition_after.options["option_1"], int)
    assert run_definition_after.options["option_1"] == 1
    assert isinstance(run_definition_after.options["option_2"], float)
    assert run_definition_after.options["option_2"] == 2.0
    assert run_definition_after.additional_run_properties["additional_run_property_0"] is None
    assert isinstance(run_definition_after.additional_run_properties["additional_run_property_1"], int)
    assert run_definition_after.additional_run_properties["additional_run_property_1"] == 1
    assert isinstance(run_definition_after.additional_run_properties["additional_run_property_2"], float)
    assert run_definition_after.additional_run_properties["additional_run_property_2"] == 2.0
    assert run_definition_after.additional_run_properties["additional_run_property_3"] == ["a", "b", "c"]
    assert run_definition_after.additional_run_properties["additional_run_property_4"] == [
        [1, 2],
        [1],
        [2.0],
        [["a", "b"], [1, 1.0]],
    ]
    assert run_definition_after.additional_run_properties["additional_run_property_5"] == {
        "a_result": "a_value",
        "b_result": "b_value",
        "c_result": "c_value",
    }
