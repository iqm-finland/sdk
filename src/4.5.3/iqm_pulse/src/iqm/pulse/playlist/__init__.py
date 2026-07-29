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
"""Control pulses and instruction schedules for quantum computers."""

from iqm.pulse.playlist.instructions import (
    AcquisitionMethod,
    ComplexIntegration,
    ConditionalInstruction,
    FluxPulse,
    Instruction,
    IQPulse,
    MultiplexedIQPulse,
    ReadoutTrigger,
    RealPulse,
    ThresholdStateDiscrimination,
    TimeTrace,
    VirtualRZ,
    Wait,
)
from iqm.pulse.playlist.schedule import Schedule, Segment
