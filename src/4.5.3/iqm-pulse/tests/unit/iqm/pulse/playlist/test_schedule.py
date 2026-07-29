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
from copy import copy

import pytest

from iqm.pulse.playlist.channel import ChannelProperties
from iqm.pulse.playlist.instructions import RealPulse, Wait
from iqm.pulse.playlist.schedule import Nothing, Schedule, Segment
from iqm.pulse.playlist.waveforms import Gaussian

w = Wait(25)
p1 = RealPulse(25, Gaussian(2, 4), 0.7)
p2 = RealPulse(40, Gaussian(2, 4), 1.0)
p3 = RealPulse(8, Gaussian(2, 4), 1.0)
n = Nothing(duration=15)


def test_nothing():
    """Test Nothing instruction"""
    nothing = Nothing(64)
    assert nothing.validate() is None
    assert nothing.copy(duration=48) == Nothing(48)


def test_segment():
    """Test creating Segment"""
    segment = Segment([w, p1, p2])
    assert segment.duration == 90
    segment = Segment([w, p1, p2], duration=75)
    assert segment.duration == 75
    assert list(reversed(segment)) == [p2, p1, w]
    copied_segment = segment.copy()
    assert segment[2] == p2
    assert len(segment) == 3
    assert copied_segment.pop(2) == p2
    assert len(list(copied_segment)) == 2


def test_schedule_with_no_args():
    """Test creating Schedule with no args"""
    schedule = Schedule()
    assert schedule.duration == 0
    assert len(schedule) == 0
    assert len(list(schedule)) == 0
    assert schedule.validate() is None


def test_schedule_with_only_duration():
    """Test creating Schedule with only duration"""
    schedule = Schedule({}, duration=100)
    assert schedule.duration == 100
    assert schedule.validate() is None


def test_schedule_without_duration():
    """Test creating Schedule without duration"""
    schedule = Schedule({"test_controller": [w, p1, p2]})
    assert schedule.duration == 90
    assert schedule.validate() is None


def test_accessing_schedule_contents():
    """Test accessing Schedule contents"""
    schedule = Schedule({"test_controller": [w, p1, p2]})
    assert isinstance(schedule["test_controller"], Segment)
    schedule["test_controller_2"] = Segment([p2, w])
    for name, segment in schedule.items():
        assert name in ["test_controller", "test_controller_2"]
        assert isinstance(segment, Segment)


def test_copying_schedule():
    """Test copying Schedule"""
    schedule = Schedule({"test_controller": [w, p1, p2]})
    assert isinstance(schedule.copy(), Schedule)
    assert schedule.copy().duration == schedule.duration


def test_adding_channels_and_instructions_to_schedule():
    """Test adding channels and instructions to Schedule"""
    channels = ["test_controller_1", "test_controller_2"]
    schedule = Schedule()
    schedule.add_channels(channels)

    schedule[channels[0]].append(w)
    schedule[channels[1]].append(p2)
    schedule[channels[0]].append(w)
    schedule[channels[1]].append(p1)
    schedule[channels[0]].append(p2)

    assert len(schedule) == 2
    assert len(schedule[channels[0]]) == 3
    assert len(schedule[channels[1]]) == 2


def test_schedule_reversing():
    """Test reversing a Schedule"""
    channels = ["test_controller_1", "test_controller_2"]
    schedule = Schedule({})
    schedule.add_channels(channels)

    schedule.extend(channels[0], [w, p1])
    schedule.extend(channels[1], [w, p2])

    # Test that padding instruction is added to the shorter segment
    reversed_schedule = schedule.reverse()
    assert len(reversed_schedule[channels[0]]) == 3
    assert isinstance(reversed_schedule[channels[0]][0], Nothing)
    assert reversed_schedule[channels[0]][0].duration == 15
    assert isinstance(reversed_schedule[channels[0]][1], RealPulse)
    assert isinstance(reversed_schedule[channels[0]][2], Wait)

    # The longer segment is not padded
    assert len(reversed_schedule[channels[1]]) == 2
    assert isinstance(reversed_schedule[channels[1]][0], RealPulse)
    assert isinstance(reversed_schedule[channels[1]][1], Wait)

    # Test hard box reversing
    reversed_schedule = schedule.reverse_hard_box()
    assert len(reversed_schedule[channels[0]]) == 2
    assert len(reversed_schedule[channels[1]]) == 2
    assert isinstance(reversed_schedule[channels[0]][0], RealPulse)
    assert isinstance(reversed_schedule[channels[0]][1], Wait)
    assert isinstance(reversed_schedule[channels[1]][0], RealPulse)
    assert isinstance(reversed_schedule[channels[1]][1], Wait)


def test_schedule_front_pad():
    """Test front-padding the schedule with a Wait"""
    schedule = Schedule(
        {
            "c1": [p1, p2],
            "c2": [p2],
        }
    )
    original = schedule.copy()
    assert schedule.validate() is None
    assert len(schedule) == 2
    assert schedule.duration == 65

    with pytest.raises(ValueError, match="Target duration 50 is shorter than the current schedule duration 65"):
        schedule.front_pad(50)

    schedule.front_pad(100)
    assert schedule.validate() is None
    assert len(schedule) == 2
    assert schedule.duration == 100
    for ch, seg in schedule.items():
        # each channel as an extra Wait in the beginning
        assert len(seg) == len(original[ch]) + 1
        assert seg[0] == Wait(35)


def test_schedule_front_pad_in_seconds():
    channel_properties = {
        "c1": ChannelProperties(1.0e9, 8, 16),
        "c2": ChannelProperties(2.0e9, 8, 16),
    }
    schedule = Schedule(
        {
            "c1": [p1, p2],  # 25ns + 40ns
            "c2": [p2],  # 20ns
        }
    )
    original = schedule.copy()
    schedule.front_pad_in_seconds(1e-7, channel_properties)
    # ch1 needs 35ns more -> add 4*8ns = 32ns -> in total 97ns
    # ch2 needs 80ns more -> add 20*4ns = 80ns -> in total 100ns
    assert schedule.duration_in_seconds(channel_properties) == 97e-9
    assert len(schedule) == 2
    new_sample_duration_c2 = channel_properties["c2"].duration_to_int_samples(
        channel_properties["c2"].round_duration_to_granularity(
            20e-9 + 100e-9 - original.duration_in_seconds(channel_properties), round_up=False
        )
    )
    assert schedule["c2"].duration == new_sample_duration_c2
    for ch, seg in schedule.items():
        # each channel as an extra Wait in the beginning
        assert len(seg) == len(original[ch]) + 1
        assert isinstance(seg[0], Wait)


def test_pad_to_hard_box():
    schedule = Schedule(
        {
            "c1": [p1, p2],
            "c2": [p2],
        }
    )
    original = copy(schedule)
    schedule.pad_to_hard_box()
    assert schedule["c1"] == original["c1"]
    assert len(schedule["c2"]) == 2
    assert schedule["c2"][0] == original["c2"][0]
    assert schedule["c2"][1] == Wait(25)


def test_pad_to_hard_box_in_seconds():
    channel_properties = {
        "c1": ChannelProperties(1.0e9, 8, 16),
        "c2": ChannelProperties(2.0e9, 8, 16),
    }
    schedule = Schedule(
        {
            "c1": [p1, p2],
            "c2": [p2],
        }
    )
    original = copy(schedule)
    schedule.pad_to_hard_box_in_seconds(channel_properties)
    assert schedule["c1"] == original["c1"]
    assert len(schedule["c2"]) == 2
    assert schedule["c2"][0] == original["c2"][0]
    c2_padding_samples = channel_properties["c2"].duration_to_int_samples(
        channel_properties["c2"].round_duration_to_granularity(
            schedule.duration_in_seconds(channel_properties) - channel_properties["c2"].duration_to_seconds(40),
        )
    )
    assert schedule["c2"][1] == Wait(c2_padding_samples)


def test_pad_to_hard_box_in_second_with_less_than_min_duration():
    channel_properties = {
        "c1": ChannelProperties(1.0e9, 8, 16),
        "c2": ChannelProperties(1.5e9, 8, 16),
    }
    schedule = Schedule(
        {
            "c1": [p1],
            "c2": [p1, p3],
        }
    )
    schedule.pad_to_hard_box_in_seconds(channel_properties)
    # duration differece of c1, c2 is shorter than the min duration of c1, so no padding is added to c1
    assert len(schedule["c1"]) == 1
    assert len(schedule["c2"]) == 2


def test_schedule_cleanup():
    """Test removing unnecessary channels from a Schedule"""
    channels = ["test_controller_1", "test_controller_2"]
    schedule = Schedule()
    schedule.add_channels(channels)
    schedule[channels[0]].append(w)
    assert len(schedule) == 2
    schedule.cleanup()
    assert len(schedule) == 0


def test_schedule_validation(capsys):
    channel = "test_controller"
    schedule = Schedule({channel: []})
    assert schedule.validate() is None

    invalid_duration = -1.5
    schedule[channel].append(Wait(invalid_duration))
    schedule.validate()

    captured = capsys.readouterr()
    assert captured.out == f"('{channel}',) Instruction.duration {invalid_duration} is negative.\n"


def test_schedule_pprint():
    """Test formatting schedule as string"""
    channels = ["test_controller_1", "test_controller_2"]
    schedule = Schedule()
    schedule.add_channels(channels)

    schedule[channels[0]].append(w)
    schedule[channels[1]].append(p2)
    schedule[channels[0]].append(w)
    schedule[channels[1]].append(p1)
    schedule[channels[0]].append(p2)
    schedule[channels[1]].append(n)

    expected = [
        "test_controller_1    3:|         |         R---------------|",
        "test_controller_2    3:R---------------R---------?     |",
    ]
    assert schedule.pprint(2.5).strip().split("\n") == expected
    assert schedule.pprint(9e-3) == "Schedule too long to print (10000 symbols)."
