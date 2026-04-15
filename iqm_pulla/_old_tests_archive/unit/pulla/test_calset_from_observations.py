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
from iqm.pulla.utils import calset_from_observations
from iqm.station_control.interface.models import ObservationBase


def test_calset_from_observations():
    observations = [
        ObservationBase(
            dut_field="gates.prx.drag_gaussian.QB1.duration",
            value=1.0e-8,
            unit="s",
        ),
        ObservationBase(
            dut_field="gates.prx.drag_gaussian.QB3.duration",
            value=3.0e-8,
            unit="s",
        ),
        ObservationBase(
            dut_field="gates.cz.tgss.QB1__QB3.coupler.amplitude",
            value=-0.13,
            unit="",
        ),
        ObservationBase(
            dut_field="gates.measure.constant.QB3.frequency",
            value=6.5e9,
            unit="Hz",
        ),
    ]
    calset = calset_from_observations(observations)
    assert len(calset) == len(observations)
    for obs in observations:
        assert calset[obs.dut_field] == obs.value
