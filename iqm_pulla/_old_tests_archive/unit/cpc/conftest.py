# Copyright 2024-2025 IQM
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
from collections.abc import Iterable

import pytest

from iqm.cpc.compiler.compiler import STANDARD_CIRCUIT_EXECUTION_OPTIONS, initialize_schedule_builder
from iqm.cpc.interface.compiler import CircuitExecutionOptions
from iqm.pulla.interface import CalibrationSetValues
from iqm.pulse.builder import ScheduleBuilder
from iqm.pulse.playlist.channel import ChannelProperties, ProbeChannelProperties
from iqm.pulse.playlist.instructions import IQPulse, ReadoutTrigger, RealPulse, VirtualRZ, Wait


@pytest.fixture
def calibration_set_values() -> CalibrationSetValues:
    """Minimal set of fake calibration observations:
        QB1-QB3
         |   |
        QB4-QB2  QB5(separate, same PL_2 as QB2, QB3)

    Matches the CHAD fixture above.
    """
    # fmt: off
    return {
        "controllers.QB1.flux.voltage": 1.0,
        "controllers.QB1.flux.awg.trigger_delay": 55e-9,
        "controllers.QB1.drive.frequency": 3.5e9,
        "controllers.QB1.drive.awg.mixer_correction.dc_bias_i": 0.0,
        "controllers.QB1.drive.awg.mixer_correction.dc_bias_q": 0.0,
        "controllers.QB1.drive.awg.intermediate_frequency": -300e6,  # HDAWG
        "controllers.QB1.drive.awg.trigger_delay": 50e-9,

        "controllers.QB2.flux.voltage": 1.0,
        "controllers.QB2.flux.awg.trigger_delay": 55e-9,
        "controllers.QB2.drive.frequency": 3.6e9,
        "controllers.QB2.drive.mixer_correction.dc_bias_i": 0.0,
        "controllers.QB2.drive.mixer_correction.dc_bias_q": 0.0,
        "controllers.QB2.drive.awg.center_frequency": 4.0e9,  # SHFSG
        "controllers.QB2.drive.awg.trigger_delay": 50e-9,

        "controllers.QB3.flux.voltage": 1.0,
        "controllers.QB3.flux.awg.trigger_delay": 55e-9,
        "controllers.QB3.drive.frequency": 3.7e9,
        "controllers.QB3.drive.awg.mixer_correction.dc_bias_i": 0.0,
        "controllers.QB3.drive.awg.mixer_correction.dc_bias_q": 0.0,
        "controllers.QB3.drive.awg.center_frequency": 4.0e9,
        "controllers.QB3.drive.awg.trigger_delay": 50e-9,

        "controllers.QB4.flux.voltage": 1.0,
        "controllers.QB4.flux.awg.trigger_delay": 55e-9,
        "controllers.QB4.drive.frequency": 3.8e9,
        "controllers.QB4.drive.awg.mixer_correction.dc_bias_i": 0.0,
        "controllers.QB4.drive.awg.mixer_correction.dc_bias_q": 0.0,
        "controllers.QB4.drive.awg.center_frequency": 4.0e9,
        "controllers.QB4.drive.awg.trigger_delay": 50e-9,

        "controllers.QB5.flux.voltage": 1.0,

        "controllers.TC-1-3.flux.voltage": -0.2,
        "controllers.TC-1-3.flux.awg.precompensation.timeconstants": [8.01e-7],
        "controllers.TC-1-3.flux.awg.precompensation.amplitudes": [0.011],
        "controllers.TC-1-3.flux.awg.trigger_delay": 70e-9,

        "controllers.TC-2-3.flux.voltage": -0.2,
        "controllers.TC-2-3.flux.awg.trigger_delay": 70e-9,

        "controllers.TC-1-4.flux.voltage": -0.2,
        "controllers.TC-1-4.flux.awg.trigger_delay": 70e-9,

        "controllers.TC-2-4.flux.voltage": -0.2,
        "controllers.TC-2-4.flux.awg.trigger_delay": 70e-9,

        "controllers.PL_1.readout.center_frequency": 3.8e9,
        "controllers.PL_1.readout.trigger_delay": 35e-9,
        "controllers.PL_1.twpa.status": True,
        "controllers.PL_1.twpa.voltage": 0.0,
        "controllers.PL_1.twpa.frequency": 1.0e9,
        "controllers.PL_1.twpa.power": 10.0,
        "controllers.PL_1.readout.attenuation_out": 15.0,
        "controllers.PL_1.readout.attenuation_in": 20.0,

        "controllers.PL_2.readout.center_frequency": 3.8e9,
        "controllers.PL_2.readout.trigger_delay": 35e-9,
        "controllers.PL_2.twpa.status": True,
        "controllers.PL_2.twpa.voltage": 0.1,
        "controllers.PL_2.twpa.frequency": 1.1e9,
        "controllers.PL_2.twpa.power": 11.0,
        "controllers.PL_2.readout.attenuation_out": 4.0,
        "controllers.PL_2.readout.attenuation_in": 7.0,

        "controllers.options.end_delay": 1e-3,

        "gates.measure.constant.QB1.frequency": 3.5e9,
        "gates.measure.constant.QB1.duration": 1e-6,
        "gates.measure.constant.QB1.amplitude_i": 0.1,
        "gates.measure.constant.QB1.amplitude_q": 0.8,
        "gates.measure.constant.QB1.phase": 0.1,
        "gates.measure.constant.QB1.integration_threshold": 0.1,
        "gates.measure.constant.QB1.integration_length": 1e-6,
        "gates.measure.constant.QB1.acquisition_delay": 1e-7,
        "gates.measure.constant.QB1.acquisition_type": "threshold",

        "gates.measure.constant.QB2.frequency": 3.6e9,
        "gates.measure.constant.QB2.duration": 1e-6,
        "gates.measure.constant.QB2.amplitude_i": 0.2,
        "gates.measure.constant.QB2.amplitude_q": 0.8,
        "gates.measure.constant.QB2.phase": 0.1,
        "gates.measure.constant.QB2.integration_threshold": 0.2,
        "gates.measure.constant.QB2.integration_length": 1e-6,
        "gates.measure.constant.QB2.acquisition_delay": 1e-7,
        "gates.measure.constant.QB2.acquisition_type": "threshold",

        "gates.measure.constant.QB3.frequency": 3.7e9,
        "gates.measure.constant.QB3.duration": 1e-6,
        "gates.measure.constant.QB3.amplitude_i": 0.3,
        "gates.measure.constant.QB3.amplitude_q": 0.8,
        "gates.measure.constant.QB3.phase": 0.1,
        "gates.measure.constant.QB3.integration_threshold": 0.3,
        "gates.measure.constant.QB3.integration_length": 1e-6,
        "gates.measure.constant.QB3.acquisition_delay": 1e-7,

        "gates.measure.constant.QB4.frequency": 3.8e9,
        "gates.measure.constant.QB4.duration": 1e-6,
        "gates.measure.constant.QB4.amplitude_i": 0.4,
        "gates.measure.constant.QB4.amplitude_q": 0.8,
        "gates.measure.constant.QB4.phase": 0.1,
        "gates.measure.constant.QB4.integration_threshold": 0.4,
        "gates.measure.constant.QB4.integration_length": 1e-6,
        "gates.measure.constant.QB4.acquisition_delay": 1e-7,

        "gates.measure.constant.QB5.frequency": 3.9e9,
        "gates.measure.constant.QB5.duration": 1e-6,
        "gates.measure.constant.QB5.amplitude_i": 0.5,
        "gates.measure.constant.QB5.amplitude_q": 0.8,
        "gates.measure.constant.QB5.phase": 0.1,
        "gates.measure.constant.QB5.integration_threshold": 0.5,
        "gates.measure.constant.QB5.integration_length": 1e-6,
        "gates.measure.constant.QB5.acquisition_delay": 1e-7,

        "gates.measure_fidelity.constant.QB1.frequency": 3.5e9,
        "gates.measure_fidelity.constant.QB1.duration": 1e-6,
        "gates.measure_fidelity.constant.QB1.amplitude_i": 0.1,
        "gates.measure_fidelity.constant.QB1.amplitude_q": 0.8,
        "gates.measure_fidelity.constant.QB1.phase": 0.1,
        "gates.measure_fidelity.constant.QB1.integration_threshold": 0.1,
        "gates.measure_fidelity.constant.QB1.integration_length": 1e-6,
        "gates.measure_fidelity.constant.QB1.acquisition_delay": 1e-7,
        "gates.measure_fidelity.constant.QB1.acquisition_type": "threshold",

        "gates.measure_fidelity.constant.QB2.frequency": 3.6e9,
        "gates.measure_fidelity.constant.QB2.duration": 1e-6,
        "gates.measure_fidelity.constant.QB2.amplitude_i": 0.2,
        "gates.measure_fidelity.constant.QB2.amplitude_q": 0.8,
        "gates.measure_fidelity.constant.QB2.phase": 0.1,
        "gates.measure_fidelity.constant.QB2.integration_threshold": 0.2,
        "gates.measure_fidelity.constant.QB2.integration_length": 1e-6,
        "gates.measure_fidelity.constant.QB2.acquisition_delay": 1e-7,
        "gates.measure_fidelity.constant.QB2.acquisition_type": "threshold",

        "gates.measure_fidelity.constant.QB3.frequency": 3.7e9,
        "gates.measure_fidelity.constant.QB3.duration": 1e-6,
        "gates.measure_fidelity.constant.QB3.amplitude_i": 0.3,
        "gates.measure_fidelity.constant.QB3.amplitude_q": 0.8,
        "gates.measure_fidelity.constant.QB3.phase": 0.1,
        "gates.measure_fidelity.constant.QB3.integration_threshold": 0.3,
        "gates.measure_fidelity.constant.QB3.integration_length": 1e-6,
        "gates.measure_fidelity.constant.QB3.acquisition_delay": 1e-7,

        "gates.measure_fidelity.constant.QB4.frequency": 3.8e9,
        "gates.measure_fidelity.constant.QB4.duration": 1e-6,
        "gates.measure_fidelity.constant.QB4.amplitude_i": 0.4,
        "gates.measure_fidelity.constant.QB4.amplitude_q": 0.8,
        "gates.measure_fidelity.constant.QB4.phase": 0.1,
        "gates.measure_fidelity.constant.QB4.integration_threshold": 0.4,
        "gates.measure_fidelity.constant.QB4.integration_length": 1e-6,
        "gates.measure_fidelity.constant.QB4.acquisition_delay": 1e-7,

        "gates.measure_fidelity.constant.QB5.frequency": 3.9e9,
        "gates.measure_fidelity.constant.QB5.duration": 1e-6,
        "gates.measure_fidelity.constant.QB5.amplitude_i": 0.5,
        "gates.measure_fidelity.constant.QB5.amplitude_q": 0.8,
        "gates.measure_fidelity.constant.QB5.phase": 0.1,
        "gates.measure_fidelity.constant.QB5.integration_threshold": 0.5,
        "gates.measure_fidelity.constant.QB5.integration_length": 1e-6,
        "gates.measure_fidelity.constant.QB5.acquisition_delay": 1e-7,

        "gates.prx.drag_gaussian.QB1.duration": 40e-9,
        "gates.prx.drag_gaussian.QB1.full_width": 20e-9,
        "gates.prx.drag_gaussian.QB1.amplitude_i": 0.8,
        "gates.prx.drag_gaussian.QB1.amplitude_q": 0.3,

        "gates.prx.drag_gaussian.QB2.duration": 40e-9,
        "gates.prx.drag_gaussian.QB2.full_width": 20e-9,
        "gates.prx.drag_gaussian.QB2.amplitude_i": 0.6,
        "gates.prx.drag_gaussian.QB2.amplitude_q": 0.1,

        "gates.prx.drag_gaussian.QB3.duration": 40e-9,
        "gates.prx.drag_gaussian.QB3.full_width": 20e-9,
        "gates.prx.drag_gaussian.QB3.amplitude_i": 0.7,
        "gates.prx.drag_gaussian.QB3.amplitude_q": 0.2,

        "gates.prx.drag_gaussian.QB4.duration": 40e-9,
        "gates.prx.drag_gaussian.QB4.full_width": 20e-9,
        "gates.prx.drag_gaussian.QB4.amplitude_i": 0.7,
        "gates.prx.drag_gaussian.QB4.amplitude_q": 0.2,

        "gates.reset_wait.reset_wait.QB1.duration": 299e-6,
        "gates.reset_wait.reset_wait.QB2.duration": 299e-6,
        "gates.reset_wait.reset_wait.QB3.duration": 299e-6,
        "gates.reset_wait.reset_wait.QB4.duration": 299e-6,
        "gates.reset_wait.reset_wait.QB5.duration": 299e-6,

        "gates.cc_prx.prx_composite.QB1.control_delays": [100e-9, 100e-9],
        "gates.cc_prx.prx_composite.QB2.control_delays": [100e-9, 100e-9],
        "gates.cc_prx.prx_composite.QB3.control_delays": [100e-9, 100e-9],
        "gates.cc_prx.prx_composite.QB4.control_delays": [100e-9, 100e-9],
        "gates.cc_prx.prx_composite.QB5.control_delays": [100e-9, 100e-9],

        "gates.cz.tgss.QB1__QB3.duration": 80e-9,
        "gates.cz.tgss.QB1__QB3.coupler.amplitude": 0.7,
        "gates.cz.tgss.QB1__QB3.coupler.full_width": 60e-9,
        "gates.cz.tgss.QB1__QB3.coupler.rise_time": 10e-9,
        "gates.cz.tgss.QB1__QB3.coupler.center_offset": 0,
        "gates.cz.tgss.QB1__QB3.rz.QB1": 0.52,
        "gates.cz.tgss.QB1__QB3.rz.QB3": -0.3,

        "gates.cz.tgss.QB2__QB3.duration": 80e-9,
        "gates.cz.tgss.QB2__QB3.coupler.amplitude": 0.7,
        "gates.cz.tgss.QB2__QB3.coupler.full_width": 60e-9,
        "gates.cz.tgss.QB2__QB3.coupler.rise_time": 10e-9,
        "gates.cz.tgss.QB2__QB3.coupler.center_offset": 0,
        "gates.cz.tgss.QB2__QB3.rz.QB2": 0.52,
        "gates.cz.tgss.QB2__QB3.rz.QB3": -0.3,

        "gates.cz.tgss.QB1__QB4.duration": 80e-9,
        "gates.cz.tgss.QB1__QB4.coupler.amplitude": 0.7,
        "gates.cz.tgss.QB1__QB4.coupler.full_width": 60e-9,
        "gates.cz.tgss.QB1__QB4.coupler.rise_time": 10e-9,
        "gates.cz.tgss.QB1__QB4.coupler.center_offset": 0,
        "gates.cz.tgss.QB1__QB4.rz.QB1": 0.52,
        "gates.cz.tgss.QB1__QB4.rz.QB4": -0.3,

        "gates.cz.tgss.QB2__QB4.duration": 80e-9,
        "gates.cz.tgss.QB2__QB4.coupler.amplitude": 0.7,
        "gates.cz.tgss.QB2__QB4.coupler.full_width": 60e-9,
        "gates.cz.tgss.QB2__QB4.coupler.rise_time": 10e-9,
        "gates.cz.tgss.QB2__QB4.coupler.center_offset": 0,
        "gates.cz.tgss.QB2__QB4.rz.QB2": 0.52,
        "gates.cz.tgss.QB2__QB4.rz.QB4": -0.3
    }


@pytest.fixture
def calibration_set_values_star() -> CalibrationSetValues:
    """Minimal set of fake Star 7 calibration observations:
        QB1  QB2  QB3
         |   |    |
         - COMP_R -
        |   |    |
       QB4 QB5  QB6

    Calibrations:
    prx  = QB1-QB6
    cz   = QB1,QB2,QB4,QB5,QB6
    move = QB1,QB3

    Matches the Star 7 CHAD fixture above.
    """
    return {
        "controllers.QB1.flux.voltage": 1.0,
        "controllers.QB1.flux.awg.precompensation.timeconstants": [8.01e-7, 43e-9],
        "controllers.QB1.flux.awg.precompensation.amplitudes": [0.011, -0.0028],
        "controllers.QB1.flux.awg.trigger_delay": 55e-9,
        "controllers.QB1.drive.frequency": 3.5e9,
        "controllers.QB1.drive.awg.intermediate_frequency": -300e6,  # HDAWG
        "controllers.QB1.drive.awg.trigger_delay": 30e-9,
        "controllers.QB2.flux.voltage": 1.0,
        "controllers.QB2.drive.frequency": 3.6e9,
        "controllers.QB2.drive.awg.center_frequency": 4.0e9,  # SHFSG
        "controllers.QB2.drive.awg.trigger_delay": 30e-9,
        "controllers.QB3.flux.voltage": 1.0,
        "controllers.QB3.drive.frequency": 3.7e9,
        "controllers.QB3.drive.awg.center_frequency": 4.0e9,
        "controllers.QB3.drive.awg.trigger_delay": 30e-9,
        "controllers.QB4.flux.voltage": 1.0,
        "controllers.QB4.drive.frequency": 3.8e9,
        "controllers.QB4.drive.awg.center_frequency": 4.0e9,
        "controllers.QB4.drive.awg.trigger_delay": 30e-9,
        "controllers.QB5.flux.voltage": 1.0,
        "controllers.QB5.drive.frequency": 3.85e9,
        "controllers.QB5.drive.awg.center_frequency": 4.0e9,
        "controllers.QB5.drive.awg.trigger_delay": 30e-9,
        "controllers.QB6.flux.voltage": 1.0,
        "controllers.QB6.drive.frequency": 3.9e9,
        "controllers.QB6.drive.awg.center_frequency": 4.0e9,
        "controllers.QB6.drive.awg.trigger_delay": 30e-9,
        "controllers.QB0.flux.voltage": 0.0,
        "controllers.TC-1.flux.voltage": -0.2,
        "controllers.TC-1.flux.awg.precompensation.timeconstants": [8.01e-7],
        "controllers.TC-1.flux.awg.precompensation.amplitudes": [0.011],
        "controllers.TC-1.flux.awg.trigger_delay": 55e-9,
        "controllers.TC-2.flux.voltage": -0.2,
        "controllers.TC-3.flux.voltage": -0.2,
        "controllers.TC-4.flux.voltage": -0.2,
        "controllers.TC-5.flux.voltage": -0.2,
        "controllers.TC-6.flux.voltage": -0.2,
        "controllers.PL.readout.center_frequency": 3.6e9,
        "controllers.PL.readout.trigger_delay": 35e-9,
        "controllers.PL.twpa.status": True,
        "controllers.PL.twpa.voltage": 0.0,
        "controllers.PL.twpa.frequency": 1.0e9,
        "controllers.PL.twpa.power": 10.0,
        "controllers.PL_2.readout.center_frequency": 3.8e9,
        "controllers.PL_2.readout.trigger_delay": 35e-9,
        "controllers.PL_2.twpa.status": True,
        "controllers.PL_2.twpa.voltage": 0.1,
        "controllers.PL_2.twpa.frequency": 1.1e9,
        "controllers.PL_2.twpa.power": 11.0,
        "controllers.COMP_R.frequency": 4.0e9,
        "controllers.options.end_delay": 1e-3,
        "gates.prx.drag_gaussian.QB1.duration": 40e-9,
        "gates.prx.drag_gaussian.QB1.full_width": 20e-9,
        "gates.prx.drag_gaussian.QB1.amplitude_i": 0.8,
        "gates.prx.drag_gaussian.QB1.amplitude_q": 0.3,
        "gates.prx.drag_gaussian.QB2.duration": 40e-9,
        "gates.prx.drag_gaussian.QB2.full_width": 20e-9,
        "gates.prx.drag_gaussian.QB2.amplitude_i": 0.7,
        "gates.prx.drag_gaussian.QB2.amplitude_q": 0.1,
        "gates.prx.drag_gaussian.QB3.duration": 40e-9,
        "gates.prx.drag_gaussian.QB3.full_width": 20e-9,
        "gates.prx.drag_gaussian.QB3.amplitude_i": 0.6,
        "gates.prx.drag_gaussian.QB3.amplitude_q": 0.2,
        "gates.prx.drag_gaussian.QB4.duration": 40e-9,
        "gates.prx.drag_gaussian.QB4.full_width": 20e-9,
        "gates.prx.drag_gaussian.QB4.amplitude_i": 0.5,
        "gates.prx.drag_gaussian.QB4.amplitude_q": 0.2,
        "gates.prx.drag_gaussian.QB5.duration": 40e-9,
        "gates.prx.drag_gaussian.QB5.full_width": 20e-9,
        "gates.prx.drag_gaussian.QB5.amplitude_i": 0.4,
        "gates.prx.drag_gaussian.QB5.amplitude_q": 0.2,
        "gates.prx.drag_gaussian.QB6.duration": 40e-9,
        "gates.prx.drag_gaussian.QB6.full_width": 20e-9,
        "gates.prx.drag_gaussian.QB6.amplitude_i": 0.3,
        "gates.prx.drag_gaussian.QB6.amplitude_q": 0.2,
        "gates.reset_wait.reset_wait.QB1.duration": 299e-6,
        "gates.reset_wait.reset_wait.QB2.duration": 299e-6,
        "gates.reset_wait.reset_wait.QB3.duration": 299e-6,
        "gates.reset_wait.reset_wait.QB4.duration": 299e-6,
        "gates.reset_wait.reset_wait.QB5.duration": 299e-6,
        "gates.reset_wait.reset_wait.QB6.duration": 299e-6,
        "gates.reset_wait.reset_wait.COMP_R.duration": 299e-6,
        "gates.cz.tgss_crf.QB1__COMP_R.duration": 80e-9,
        "gates.measure.constant.QB1.frequency": 3.5e9,
        "gates.measure.constant.QB1.duration": 1e-6,
        "gates.measure.constant.QB1.amplitude_i": 0.8,
        "gates.measure.constant.QB1.amplitude_q": 0.8,
        "gates.measure.constant.QB1.phase": 0.1,
        "gates.measure.constant.QB1.integration_threshold": 0.1,
        "gates.measure.constant.QB1.integration_length": 1e-6,
        "gates.measure.constant.QB1.acquisition_delay": 1e-7,
        "gates.measure.constant.QB1.acquisition_type": "threshold",
        "gates.measure.constant.QB2.frequency": 3.6e9,
        "gates.measure.constant.QB2.duration": 1e-6,
        "gates.measure.constant.QB2.amplitude_i": 0.8,
        "gates.measure.constant.QB2.amplitude_q": 0.8,
        "gates.measure.constant.QB2.phase": 0.1,
        "gates.measure.constant.QB2.integration_threshold": 0.2,
        "gates.measure.constant.QB2.integration_length": 1e-6,
        "gates.measure.constant.QB2.acquisition_delay": 1e-7,
        "gates.measure.constant.QB2.acquisition_type": "threshold",
        "gates.measure.constant.QB3.frequency": 3.7e9,
        "gates.measure.constant.QB3.duration": 1e-6,
        "gates.measure.constant.QB3.amplitude_i": 0.8,
        "gates.measure.constant.QB3.amplitude_q": 0.8,
        "gates.measure.constant.QB3.phase": 0.1,
        "gates.measure.constant.QB3.integration_threshold": 0.3,
        "gates.measure.constant.QB3.integration_length": 1e-6,
        "gates.measure.constant.QB3.acquisition_delay": 1e-7,
        "gates.measure.constant.QB4.frequency": 3.8e9,
        "gates.measure.constant.QB4.duration": 1e-6,
        "gates.measure.constant.QB4.amplitude_i": 0.8,
        "gates.measure.constant.QB4.amplitude_q": 0.8,
        "gates.measure.constant.QB4.phase": 0.1,
        "gates.measure.constant.QB4.integration_threshold": 0.4,
        "gates.measure.constant.QB4.integration_length": 1e-6,
        "gates.measure.constant.QB4.acquisition_delay": 1e-7,
        "gates.measure_fidelity.constant.QB1.frequency": 3.5e9,
        "gates.measure_fidelity.constant.QB1.duration": 1e-6,
        "gates.measure_fidelity.constant.QB1.amplitude_i": 0.8,
        "gates.measure_fidelity.constant.QB1.amplitude_q": 0.8,
        "gates.measure_fidelity.constant.QB1.phase": 0.1,
        "gates.measure_fidelity.constant.QB1.integration_threshold": 0.1,
        "gates.measure_fidelity.constant.QB1.integration_length": 1e-6,
        "gates.measure_fidelity.constant.QB1.acquisition_delay": 1e-7,
        "gates.measure_fidelity.constant.QB1.acquisition_type": "threshold",
        "gates.measure_fidelity.constant.QB2.frequency": 3.6e9,
        "gates.measure_fidelity.constant.QB2.duration": 1e-6,
        "gates.measure_fidelity.constant.QB2.amplitude_i": 0.8,
        "gates.measure_fidelity.constant.QB2.amplitude_q": 0.8,
        "gates.measure_fidelity.constant.QB2.phase": 0.1,
        "gates.measure_fidelity.constant.QB2.integration_threshold": 0.2,
        "gates.measure_fidelity.constant.QB2.integration_length": 1e-6,
        "gates.measure_fidelity.constant.QB2.acquisition_delay": 1e-7,
        "gates.measure_fidelity.constant.QB2.acquisition_type": "threshold",
        "gates.measure_fidelity.constant.QB3.frequency": 3.7e9,
        "gates.measure_fidelity.constant.QB3.duration": 1e-6,
        "gates.measure_fidelity.constant.QB3.amplitude_i": 0.8,
        "gates.measure_fidelity.constant.QB3.amplitude_q": 0.8,
        "gates.measure_fidelity.constant.QB3.phase": 0.1,
        "gates.measure_fidelity.constant.QB3.integration_threshold": 0.3,
        "gates.measure_fidelity.constant.QB3.integration_length": 1e-6,
        "gates.measure_fidelity.constant.QB3.acquisition_delay": 1e-7,
        "gates.measure_fidelity.constant.QB4.frequency": 3.8e9,
        "gates.measure_fidelity.constant.QB4.duration": 1e-6,
        "gates.measure_fidelity.constant.QB4.amplitude_i": 0.8,
        "gates.measure_fidelity.constant.QB4.amplitude_q": 0.8,
        "gates.measure_fidelity.constant.QB4.phase": 0.1,
        "gates.measure_fidelity.constant.QB4.integration_threshold": 0.4,
        "gates.measure_fidelity.constant.QB4.integration_length": 1e-6,
        "gates.measure_fidelity.constant.QB4.acquisition_delay": 1e-7,
        "gates.cz.tgss_crf.QB1__COMP_R.coupler.amplitude": 0.7,
        "gates.cz.tgss_crf.QB1__COMP_R.coupler.full_width": 60e-9,
        "gates.cz.tgss_crf.QB1__COMP_R.coupler.rise_time": 10e-9,
        "gates.cz.tgss_crf.QB1__COMP_R.coupler.center_offset": 0,
        "gates.cz.tgss_crf.QB1__COMP_R.qubit.amplitude": 0.2,
        "gates.cz.tgss_crf.QB1__COMP_R.qubit.full_width": 70e-9,
        "gates.cz.tgss_crf.QB1__COMP_R.qubit.rise_time": 5e-9,
        "gates.cz.tgss_crf.QB1__COMP_R.qubit.center_offset": 0,
        "gates.cz.tgss_crf.QB1__COMP_R.rz.QB1": 0.112,
        "gates.cz.tgss_crf.QB1__COMP_R.rz.COMP_R": -0.351,
        "gates.cz.tgss_crf.QB2__COMP_R.duration": 80e-9,
        "gates.cz.tgss_crf.QB2__COMP_R.coupler.amplitude": 0.7,
        "gates.cz.tgss_crf.QB2__COMP_R.coupler.full_width": 60e-9,
        "gates.cz.tgss_crf.QB2__COMP_R.coupler.rise_time": 10e-9,
        "gates.cz.tgss_crf.QB2__COMP_R.coupler.center_offset": 0,
        "gates.cz.tgss_crf.QB2__COMP_R.qubit.amplitude": 0.2,
        "gates.cz.tgss_crf.QB2__COMP_R.qubit.full_width": 70e-9,
        "gates.cz.tgss_crf.QB2__COMP_R.qubit.rise_time": 5e-9,
        "gates.cz.tgss_crf.QB2__COMP_R.qubit.center_offset": 0,
        "gates.cz.tgss_crf.QB2__COMP_R.rz.QB2": 0.112,
        "gates.cz.tgss_crf.QB2__COMP_R.rz.COMP_R": -0.351,
        "gates.cz.tgss_crf.QB4__COMP_R.duration": 80e-9,
        "gates.cz.tgss_crf.QB4__COMP_R.coupler.amplitude": 0.7,
        "gates.cz.tgss_crf.QB4__COMP_R.coupler.full_width": 60e-9,
        "gates.cz.tgss_crf.QB4__COMP_R.coupler.rise_time": 10e-9,
        "gates.cz.tgss_crf.QB4__COMP_R.coupler.center_offset": 0,
        "gates.cz.tgss_crf.QB4__COMP_R.qubit.amplitude": 0.2,
        "gates.cz.tgss_crf.QB4__COMP_R.qubit.full_width": 70e-9,
        "gates.cz.tgss_crf.QB4__COMP_R.qubit.rise_time": 5e-9,
        "gates.cz.tgss_crf.QB4__COMP_R.qubit.center_offset": 0,
        "gates.cz.tgss_crf.QB4__COMP_R.rz.QB4": 0.112,
        "gates.cz.tgss_crf.QB4__COMP_R.rz.COMP_R": -0.351,
        "gates.cz.tgss_crf.QB5__COMP_R.duration": 80e-9,
        "gates.cz.tgss_crf.QB5__COMP_R.coupler.amplitude": 0.7,
        "gates.cz.tgss_crf.QB5__COMP_R.coupler.full_width": 60e-9,
        "gates.cz.tgss_crf.QB5__COMP_R.coupler.rise_time": 10e-9,
        "gates.cz.tgss_crf.QB5__COMP_R.coupler.center_offset": 0,
        "gates.cz.tgss_crf.QB5__COMP_R.qubit.amplitude": 0.2,
        "gates.cz.tgss_crf.QB5__COMP_R.qubit.full_width": 70e-9,
        "gates.cz.tgss_crf.QB5__COMP_R.qubit.rise_time": 5e-9,
        "gates.cz.tgss_crf.QB5__COMP_R.qubit.center_offset": 0,
        "gates.cz.tgss_crf.QB5__COMP_R.rz.QB5": 0.112,
        "gates.cz.tgss_crf.QB5__COMP_R.rz.COMP_R": -0.351,
        "gates.cz.tgss_crf.QB6__COMP_R.duration": 80e-9,
        "gates.cz.tgss_crf.QB6__COMP_R.coupler.amplitude": 0.7,
        "gates.cz.tgss_crf.QB6__COMP_R.coupler.full_width": 60e-9,
        "gates.cz.tgss_crf.QB6__COMP_R.coupler.rise_time": 10e-9,
        "gates.cz.tgss_crf.QB6__COMP_R.coupler.center_offset": 0,
        "gates.cz.tgss_crf.QB6__COMP_R.qubit.amplitude": 0.2,
        "gates.cz.tgss_crf.QB6__COMP_R.qubit.full_width": 70e-9,
        "gates.cz.tgss_crf.QB6__COMP_R.qubit.rise_time": 5e-9,
        "gates.cz.tgss_crf.QB6__COMP_R.qubit.center_offset": 0,
        "gates.cz.tgss_crf.QB6__COMP_R.rz.QB6": 0.112,
        "gates.cz.tgss_crf.QB6__COMP_R.rz.COMP_R": -0.351,
        "gates.move.tgss_crf.QB1__COMP_R.duration": 80e-9,
        "gates.move.tgss_crf.QB1__COMP_R.detuning": 3.5e9 - 4.0e9,
        "gates.move.tgss_crf.QB1__COMP_R.coupler.amplitude": 0.6,
        "gates.move.tgss_crf.QB1__COMP_R.coupler.full_width": 60e-9,
        "gates.move.tgss_crf.QB1__COMP_R.coupler.rise_time": 10e-9,
        "gates.move.tgss_crf.QB1__COMP_R.coupler.center_offset": 0,
        "gates.move.tgss_crf.QB1__COMP_R.qubit.amplitude": 0.1,
        "gates.move.tgss_crf.QB1__COMP_R.qubit.full_width": 70e-9,
        "gates.move.tgss_crf.QB1__COMP_R.qubit.rise_time": 5e-9,
        "gates.move.tgss_crf.QB1__COMP_R.qubit.center_offset": 0,
        "gates.move.tgss_crf.QB1__COMP_R.rz.QB1": 0.24,
        "gates.move.tgss_crf.QB1__COMP_R.rz.COMP_R": -0.817,
        "gates.move.tgss_crf.QB3__COMP_R.duration": 80e-9,
        "gates.move.tgss_crf.QB3__COMP_R.detuning": 3.7e9 - 4.0e9,
        "gates.move.tgss_crf.QB3__COMP_R.coupler.amplitude": 0.6,
        "gates.move.tgss_crf.QB3__COMP_R.coupler.full_width": 60e-9,
        "gates.move.tgss_crf.QB3__COMP_R.coupler.rise_time": 10e-9,
        "gates.move.tgss_crf.QB3__COMP_R.coupler.center_offset": 0,
        "gates.move.tgss_crf.QB3__COMP_R.qubit.amplitude": 0.1,
        "gates.move.tgss_crf.QB3__COMP_R.qubit.full_width": 70e-9,
        "gates.move.tgss_crf.QB3__COMP_R.qubit.rise_time": 5e-9,
        "gates.move.tgss_crf.QB3__COMP_R.qubit.center_offset": 0,
        "gates.move.tgss_crf.QB3__COMP_R.rz.QB3": 0.24,
        "gates.move.tgss_crf.QB3__COMP_R.rz.COMP_R": -0.817,
    }


def _build_channel_properties(
    qubits: Iterable[str],
    couplers: Iterable[str],
    computational_resonators: Iterable[str] = (),
    probe_lines: Iterable[str] = (),
    add_feedback_channels: bool = False,
) -> dict[str, ChannelProperties]:
    """Mocks the process of requesting channel information from Station Control."""

    allowed_real = (Wait, RealPulse)
    allowed_iq = (Wait, IQPulse, VirtualRZ)
    HDAWG = ChannelProperties(2.0e9, 16, 32, allowed_real)
    SHFSG = ChannelProperties(2.0e9, 16, 32, allowed_iq, is_iq=True)
    SHFQA = ProbeChannelProperties(2.0e9, 16, 32, (ReadoutTrigger,), is_iq=False, center_frequency=3.6e9)
    VIRTUAL = ChannelProperties(2.0e9, 16, 32, allowed_iq, is_iq=True, is_virtual=True)
    props = {}
    for q in qubits:
        props.update(
            {
                f"{q}__drive.awg": SHFSG,
                f"{q}__flux.awg": HDAWG,
            }
        )
        if add_feedback_channels:
            props.update({f"feedback_to_{q}__drive.awg": VIRTUAL})
    for c in couplers:
        props.update(
            {
                f"{c}__flux.awg": HDAWG,
            }
        )
    for r in computational_resonators:
        props.update(
            {
                f"{r}__drive_virtual": VIRTUAL,
            }
        )
    for pl in probe_lines:
        props.update({f"{pl}__readout": SHFQA})
    return props


def _build_component_channels(
    qubits: list[str],
    couplers: list[str],
    computational_resonators: Iterable[str] = (),
    probe_lines: Iterable[str] = (),
    add_feedback_channels: bool = False,
) -> dict[str, dict[str, str]]:
    """Mocks the process of requesting channel information from Station Control."""
    comp_channels = {}
    for q in qubits:
        comp_channels[q] = {"drive": f"{q}__drive.awg", "flux": f"{q}__flux.awg"}
        if add_feedback_channels:
            for pl in probe_lines:
                comp_channels[q][f"feedback_from_{pl}"] = f"feedback_to_{q}__drive.awg"
    for c in couplers:
        comp_channels[c] = {"flux": f"{c}__flux.awg"}
    for r in computational_resonators:
        comp_channels[r] = {"drive": f"{r}__drive_virtual"}
    for pl in probe_lines:
        comp_channels[pl] = {"readout": f"{pl}__readout"}
        if add_feedback_channels:
            for q in qubits:
                comp_channels[pl][f"feedback_to_{q}__drive.awg"] = f"feedback_to_{q}__drive.awg"
    return comp_channels


@pytest.fixture(scope="module")
def channel_properties(chip_topology) -> dict[str, ChannelProperties]:
    """Mocks the process of requesting channel information from Station Control."""
    return _build_channel_properties(
        qubits=chip_topology.qubits_sorted,
        couplers=chip_topology.couplers_sorted,
        computational_resonators=chip_topology.computational_resonators_sorted,
        probe_lines=chip_topology.probe_lines_sorted,
    )


@pytest.fixture(scope="module")
def channel_properties_feedback(chip_topology) -> dict[str, ChannelProperties]:
    """Mocks the process of requesting channel information from Station Control."""
    return _build_channel_properties(
        qubits=chip_topology.qubits_sorted,
        couplers=chip_topology.couplers_sorted,
        computational_resonators=chip_topology.computational_resonators_sorted,
        probe_lines=chip_topology.probe_lines_sorted,
        add_feedback_channels=True,
    )


@pytest.fixture(scope="module")
def channel_properties_star(chip_topology_star) -> dict[str, ChannelProperties]:
    """Mocks the process of requesting channel information from Station Control."""
    return _build_channel_properties(
        qubits=chip_topology_star.qubits_sorted,
        couplers=chip_topology_star.couplers_sorted,
        computational_resonators=chip_topology_star.computational_resonators_sorted,
        probe_lines=chip_topology_star.probe_lines_sorted,
    )


@pytest.fixture(scope="module")
def component_channels(chip_topology) -> dict[str, dict[str, str]]:
    """Mocks the process of requesting channel information from Station Control."""
    return _build_component_channels(
        qubits=chip_topology.qubits_sorted,
        couplers=chip_topology.couplers_sorted,
        computational_resonators=chip_topology.computational_resonators_sorted,
        probe_lines=chip_topology.probe_lines_sorted,
    )


@pytest.fixture(scope="module")
def component_channels_feedback(chip_topology) -> dict[str, dict[str, str]]:
    """Mocks the process of requesting channel information from Station Control."""
    return _build_component_channels(
        qubits=chip_topology.qubits_sorted,
        couplers=chip_topology.couplers_sorted,
        computational_resonators=chip_topology.computational_resonators_sorted,
        probe_lines=chip_topology.probe_lines_sorted,
        add_feedback_channels=True,
    )


@pytest.fixture(scope="module")
def component_channels_star(chip_topology_star) -> dict[str, dict[str, str]]:
    """Mocks the process of requesting channel information from Station Control."""
    return _build_component_channels(
        qubits=chip_topology_star.qubits_sorted,
        couplers=chip_topology_star.couplers_sorted,
        computational_resonators=chip_topology_star.computational_resonators_sorted,
        probe_lines=chip_topology_star.probe_lines_sorted,
    )


@pytest.fixture
def schedule_builder(calibration_set_values, chip_topology, channel_properties, component_channels) -> ScheduleBuilder:
    """Fully initialized ScheduleBuilder for a partial Crystal 20 chip."""
    return initialize_schedule_builder(
        calibration_set_values,
        chip_topology,
        channel_properties,
        component_channels,
    )


@pytest.fixture
def schedule_builder_star(
    calibration_set_values_star,
    chip_topology_star,
    channel_properties_star,
    component_channels_star,
) -> ScheduleBuilder:
    """Fully initialized ScheduleBuilder for star chip."""
    return initialize_schedule_builder(
        calibration_set_values_star,
        chip_topology_star,
        channel_properties_star,
        component_channels_star,
    )


class DefaultOptionsGenerator:
    @staticmethod
    def from_defaults(  # noqa: PLR0913
        measurement_mode=STANDARD_CIRCUIT_EXECUTION_OPTIONS.measurement_mode,
        heralding_mode=STANDARD_CIRCUIT_EXECUTION_OPTIONS.heralding_mode,
        dd_mode=STANDARD_CIRCUIT_EXECUTION_OPTIONS.dd_mode,
        dd_strategy=None,
        circuit_boundary_mode=STANDARD_CIRCUIT_EXECUTION_OPTIONS.circuit_boundary_mode,
        move_gate_validation=STANDARD_CIRCUIT_EXECUTION_OPTIONS.move_gate_validation,
        move_gate_frame_tracking=STANDARD_CIRCUIT_EXECUTION_OPTIONS.move_gate_frame_tracking,
        active_reset_cycles=STANDARD_CIRCUIT_EXECUTION_OPTIONS.active_reset_cycles,
        convert_terminal_measurements=STANDARD_CIRCUIT_EXECUTION_OPTIONS.convert_terminal_measurements,
    ) -> CircuitExecutionOptions:
        return CircuitExecutionOptions(
            measurement_mode=measurement_mode,
            heralding_mode=heralding_mode,
            dd_mode=dd_mode,
            dd_strategy=dd_strategy,
            circuit_boundary_mode=circuit_boundary_mode,
            move_gate_validation=move_gate_validation,
            move_gate_frame_tracking=move_gate_frame_tracking,
            active_reset_cycles=active_reset_cycles,
            convert_terminal_measurements=convert_terminal_measurements,
        )


@pytest.fixture
def default_options_generator() -> DefaultOptionsGenerator:
    """Generator for CircuitExecutionOptions instances."""
    return DefaultOptionsGenerator()
