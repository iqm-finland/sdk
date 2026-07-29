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

from iqm.models.channel_properties import AWGProperties, ReadoutProperties
from iqm.models.playlist.instructions import IQPulse, ReadoutTrigger, RealPulse, VirtualRZ, Wait

from iqm.station_control.client.serializers.channel_property_serializer import (
    deserialize_channel_properties,
    serialize_channel_properties,
)


def test_readout_properties():
    properties = {
        "test1": ReadoutProperties(
            sampling_rate=2.4e9,
            instruction_duration_granularity=16,
            instruction_duration_min=16,
            compatible_instructions=(ReadoutTrigger,),
            integration_start_dead_time=2e-16,
            integration_stop_dead_time=4e-16,
        ),
    }
    assert properties == deserialize_channel_properties(serialize_channel_properties(properties))


def test_awg_properties():
    properties = {
        "iq": AWGProperties(
            sampling_rate=2.4e9,
            instruction_duration_granularity=16,
            instruction_duration_min=16,
            compatible_instructions=(IQPulse, Wait, VirtualRZ),
            fast_feedback_sources=["fake_readout"],
            local_oscillator=True,
            mixer_correction=False,
        ),
        "flux": AWGProperties(
            sampling_rate=2.4e9,
            instruction_duration_granularity=16,
            instruction_duration_min=16,
            compatible_instructions=(
                RealPulse,
                Wait,
            ),
            fast_feedback_sources=["fake_readout"],
            local_oscillator=False,
            mixer_correction=False,
        ),
    }
    assert properties == deserialize_channel_properties(serialize_channel_properties(properties))
