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

from iqm.models.playlist import ChannelDescription, IQChannelConfig, RealChannelConfig, Segment
from iqm.models.playlist.channel_descriptions import ReadoutChannelConfig
from iqm.models.playlist.instructions import (
    ComplexIntegration,
    ConditionalInstruction,
    Instruction,
    IQPulse,
    MultiplexedIQPulse,
    MultiplexedRealPulse,
    ReadoutTrigger,
    RealPulse,
    ThresholdStateDiscrimination,
    TimeTrace,
    VirtualRZ,
    Wait,
)
from iqm.models.playlist.playlist import Playlist
from iqm.models.playlist.waveforms import (
    Constant,
    CosineRiseFall,
    Gaussian,
    GaussianDerivative,
    GaussianSmoothedSquare,
    Samples,
    TruncatedGaussian,
    TruncatedGaussianDerivative,
    TruncatedGaussianSmoothedSquare,
)
import numpy as np
import pytest

from iqm.station_control.client.serializers.playlist_serializers import pack_playlist, unpack_playlist


def test_serialise_playlist():
    playlist = Playlist()
    wf1 = Gaussian(32, 0.24, 0.24)
    wf2 = GaussianDerivative(32, 0.24, 0.24)
    wf3 = GaussianSmoothedSquare(32, 0.24, 0.24, 0.24)
    wf4 = Samples(np.arange(32, dtype=float))
    wf5 = Constant(32)
    wf6 = TruncatedGaussian(32, 0.1, 0.2)
    wf7 = TruncatedGaussianDerivative(32, 0.1, 0.2)
    wf8 = TruncatedGaussianSmoothedSquare(32, 0.2, 0.1)
    wf9 = CosineRiseFall(32, 0.2, 0.1)
    instructions = [
        Instruction(32, RealPulse(wf1, 0.2)),
        Instruction(32, IQPulse(wf2, wf3, 0.2, 0.2, 0.2, 2)),
        Instruction(32, VirtualRZ(0.2)),
        Instruction(
            32,
            ConditionalInstruction(
                "hello", Instruction(32, RealPulse(wf1, 0.2)), Instruction(32, IQPulse(wf2, wf3, 0.2, 0.2, 0.2))
            ),
        ),
        Instruction(32, Wait()),
        Instruction(32, RealPulse(wf4, 0.2)),
        Instruction(32, RealPulse(wf5, 0.2)),
        Instruction(32, IQPulse(wf6, wf7, 0.2, 0.2, 0.2)),
        Instruction(32, IQPulse(wf8, wf9, 0.2, 0.2, 0.2, 0.1)),
        Instruction(
            64,
            MultiplexedRealPulse(
                ((Instruction(32, RealPulse(wf1, 0.3)), 0), (Instruction(32, RealPulse(wf5, 0.3)), 30))
            ),
        ),
        Instruction(
            64,
            MultiplexedIQPulse(
                (
                    (Instruction(32, IQPulse(wf2, wf3, 0.3, 0.3, 0.3)), 0),
                    (Instruction(32, IQPulse(wf2, wf3, 0.3, 0.3, 0.3)), 30),
                )
            ),
        ),
    ]
    readout_instr = Instruction(
        32,
        ReadoutTrigger(
            Instruction(32, MultiplexedIQPulse(entries=((Instruction(32, IQPulse(wf5, wf5, 0.4, 0.4, 0.0, 0.1)), 0),))),
            acquisitions=(
                TimeTrace("ch3.time_trace", delay_samples=1, duration_samples=32),
                ComplexIntegration("ch3.QB1.result", delay_samples=2, weights=IQPulse(wf5, wf5, 0.2, 0.2, 0.0, 0.0)),
                ComplexIntegration("ch3.QB2.result", delay_samples=2, weights=IQPulse(wf5, wf5, 0.2, 0.2, 0.1, 0.1)),
                ThresholdStateDiscrimination(
                    "ch3.QB2.result",
                    delay_samples=2,
                    weights=IQPulse(wf5, wf5, 0.2, 0.2, 0.1, 0.1),
                    threshold=0.123,
                    feedback_signal_label="some label",
                ),
            ),
        ),
    )

    channel_description1 = ChannelDescription(RealChannelConfig(2.4e9), "ch1")
    channel_description2 = ChannelDescription(IQChannelConfig(2.4e9), "ch2")
    channel_description3 = ChannelDescription(ReadoutChannelConfig(2.0e9), "ch3")
    playlist.add_channel(channel_description1)
    playlist.add_channel(channel_description2)
    playlist.add_channel(channel_description3)

    seg = Segment()
    playlist.segments.append(seg)
    for instr in instructions:
        seg.add_to_segment(channel_description1, instr)
        seg.add_to_segment(channel_description2, instr)
        seg.add_to_segment(channel_description3, instr)
    seg.add_to_segment(channel_description3, readout_instr)
    pb_playlist = pack_playlist(playlist)
    playlist_from_proto = unpack_playlist(pb_playlist)
    assert playlist_from_proto == playlist


def test_serialise_empty_playlist():
    playlist = Playlist()
    pb_playlist = pack_playlist(playlist)
    playlist_from_proto = unpack_playlist(pb_playlist)
    assert playlist == playlist_from_proto


def test_serialise_none_playlist():
    playlist = None
    with pytest.raises(AttributeError):
        pack_playlist(playlist)
