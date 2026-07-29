#  ********************************************************************************
#
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
from iqm.models.playlist.channel_descriptions import ChannelDescription, IQChannelConfig
from iqm.models.playlist.instructions import Instruction
from iqm.models.playlist.segment import Segment
import pytest

from iqm.pulse.playlist.instructions import RealPulse, Wait
from iqm.pulse.playlist.playlist import Playlist
from iqm.pulse.playlist.visualisation.base import inspect_playlist
from iqm.pulse.playlist.waveforms import Gaussian

w = Wait(25)
p1 = RealPulse(25, Gaussian(2, 4), 0.7)
p2 = RealPulse(40, Gaussian(2, 4), 1.0)


def dummy_playlist(segments_amount=1):
    playlist = Playlist()
    channel_desc = ChannelDescription(IQChannelConfig(2.4e9), "test")
    playlist.add_channel(channel_desc)
    for i in range(segments_amount):
        playlist.segments.append(Segment())
        playlist.segments[i - 1].add_to_segment(channel_desc, Instruction(32, Wait(32)))
    return playlist


def test_visualisation_returns_a_value():
    pl = dummy_playlist()
    html_text = inspect_playlist(pl, [0])
    assert html_text is not None


def test_visualization_is_embedded_into_html_iframe():
    pl = dummy_playlist()
    html_text = inspect_playlist(pl, [0])
    assert html_text.startswith("<iframe")
    assert html_text.endswith("</iframe>")


def test_visualisation_fails_on_bad_segments():
    pl = dummy_playlist()
    html_text = inspect_playlist(pl, [-1])
    assert html_text is not None
    with pytest.raises(IndexError, match="Index '3' not in range of segments 0"):
        html_text = inspect_playlist(pl, [3])
    pl2 = dummy_playlist(segments_amount=2)
    with pytest.raises(IndexError, match="Index '3' not in range of segments 0-1"):
        html_text = inspect_playlist(pl2, [3])
