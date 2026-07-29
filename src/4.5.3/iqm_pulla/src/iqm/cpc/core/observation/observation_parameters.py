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
"""Physical characteristics and calibration parameters of components."""

from __future__ import annotations

from exa.common.data.parameter import CollectionType, DataType, Parameter

COUPLER_QUBIT_NORMALIZED_COUPLING = Parameter("beta", label="Normalized qubit-coupler coupling")

FLUX_VOLTAGE = Parameter("flux.voltage", "Flux voltage", "V")
"""Bias voltage that determines operating point, qubit flux and resonance frequency."""

DRIVE_FREQUENCY = Parameter("drive.frequency", "Drive frequency", "Hz")
"""Frequency of the g-e transition of the qubit."""

AVERAGE_RESPONSE_PHASE = Parameter("average_response_phase", "Average phase accumulation of response", "rad")
"""Inverse of the phase of average response signal, assuming 0 phase in probe pulse.
The complex signal is rotated in the IQ plane by 2 factors: `SINGLE_SHOT_READOUT_PHASE` which is deliberately added
to the probe pulse, and factors related to the signal travel time in the physical setup (=`AVERAGE_RESPONSE_PHASE`).
This value, together with `AVERAGE_RESPONSE_G` and `AVERAGE_RESPONSE_E`, is used to convert measured complex signals
to probabilities *in software* by rotating the signals to the real in axis. The rotation angle is the difference of
this value and `SINGLE_SHOT_READOUT_PHASE`.
When calibrated correctly, this value equals `SINGLE_SHOT_READOUT_PHASE`, such that the correct rotation is done by
the pulse, and the received complex values do not need to be rotated."""

AVERAGE_RESPONSE_G = Parameter("average_response_g", "Average g-state response", "V")
"""Average response voltage of ground state projected to principal axis (defined by average_response_phase)"""

AVERAGE_RESPONSE_E = Parameter("average_response_e", "Average e-state response", "V")
"""Average response voltage of excited state projected to principal axis (defined by average_response_phase)"""

AVERAGE_RESPONSE_F = Parameter("average_response_f", "Average f-state response", "V", data_type=DataType.COMPLEX)
"""Average response voltage of second excited state, complex."""

SINGLE_SHOT_01_ERROR = Parameter("single_shot_01_error", "Probability of measuring 1 if the 0 state was prepared.")
"""Probability of measuring 1 if the 0 state was prepared."""

SINGLE_SHOT_10_ERROR = Parameter("single_shot_10_error", "Probability of measuring 0 if the 1 state was prepared.")
"""Probability of measuring 0 if the 1 state was prepared."""

EF_PULSE_AMPLITUDE = Parameter("drive.ef_pulse_amplitude", "Drive amplitude for the EF Pulse", "")
"""Amplitude of the pulse for e-f transition."""

DISPERSIVE_SHIFT = Parameter("dispersive_shift", "Dispersive shift", "Hz")
r"""Dispersive shift of the qubit (typically denoted by :math:`\chi`)."""

T1_TIME = Parameter("t1_time", "T1 relaxation time", "s")
""":math:`T_1` relaxation time."""

T1_TIME_DURING_READOUT = Parameter("t1_time_during_readout", "T1 relaxation time during readout", "s")
""":math:`T_1` relaxation time during readout."""

T1_TIME_FITS = Parameter("t1_time_fits", "Fits of T1 time", "s", collection_type=CollectionType.LIST)
"""Measured :math:`T_1` relaxation times.
If uncertainty is not None, it represents the standard deviation of the measurement sampling."""

T2_RAMSEY_TIME = Parameter("t2_time", "T2 ramsey relaxation time", "s")
""":math:`T^*_2` ramsey relaxation time."""

T2_RAMSEY_TIME_FITS = Parameter(
    "t2_time_fits", "Fits of T2 ramsey relaxation time", "s", collection_type=CollectionType.LIST
)
"""Measured :math:`T^*_2` ramsey relaxation timed.
If uncertainty is not None, it represents standard deviation of the measurement sampling."""

T2_RAMSEY_EXP_TIME_FITS = Parameter("t2_exp_time_fits", "Fits of exponential T2 relaxation time", "s")
"""Measured exponential :math:`T^*_2` ramsey relaxation timed.
If uncertainty is not None, it represents standard deviation of the measurement sampling."""

T2_RAMSEY_GAUSS_TIME_FITS = Parameter(
    "t2_gauss_time_fits", "Fits of Gaussian T2 relaxation time", "s", collection_type=CollectionType.LIST
)
"""Measured Gaussian :math:`T^*_2` ramsey relaxation timed.
If uncertainty is not None, it represents standard deviation of the measurement sampling."""

T2_ECHO_TIME = Parameter("t2_echo_time", "T2 echo relaxation time", "s")
""":math:`T_2` echo relaxation time."""

T2_ECHO_TIME_FITS = Parameter(
    "t2_echo_time_fits", "Fits of T2 echo relaxation time", "s", collection_type=CollectionType.LIST
)
"""Measured :math:`T_2` echo relaxation times.
If uncertainty is not None, it represents standard deviation of the measurement sampling"""

T2_ECHO_EXP_TIME_FITS = Parameter(
    "t2_echo_exp_time_fits", "Fits of exponential T2 Echo relaxation time", "s", collection_type=CollectionType.LIST
)
"""Measured exponential :math:`T^*_2` echo ramsey relaxation timed."""

T2_ECHO_GAUSS_TIME_FITS = Parameter(
    "t2_echo_gauss_time_fits", "Fits of Gaussian T2 Echo relaxation time", "s", collection_type=CollectionType.LIST
)
"""Measured Gaussian :math:`T^*_2` echo ramsey relaxation timed."""

SINGLE_SHOT_READOUT_G = Parameter("single_shot_readout_g", "Readout voltage g-state", "V")
"""Best readout voltage of ground state projected to principal axis."""

SINGLE_SHOT_READOUT_E = Parameter("single_shot_readout_e", "Readout voltage e-state", "V")
"""Best readout voltage of excited state projected to principal axis."""

ANHARMONICITY = Parameter("anharmonicity", "Energy level anharmonicity", "Hz")
"""Energy level anharmonicity of the qubit."""

STATE_FIDELITY = Parameter("state_fidelity", "State fidelity", "")
"""State preparation fidelity from state tomography."""

STATE_FIDELITY_ESTIMATE = Parameter("state_fidelity_estimate", "Estimate of State fidelity", "")
"""Estimate of the State preparation fidelity which could be achieved if corrected virtual-Z rotations were used."""

INTEGRATED_DEPHASING = Parameter("integrated_dephasing", "Integrated dephasing", "")
"""Integrated dephasing of target qubit due to measurement of a control qubit."""

DRIVE_OFFSET_CH0 = Parameter("drive.offset_ch0", "Drive offset at ch. 0", "V")
"""DC offset applied at the I port of the drive mixer"""

DRIVE_OFFSET_CH1 = Parameter("drive.offset_ch1", "Drive offset at ch. 1", "V")
"""DC offset applied at the Q port of the drive mixer"""

FLUX_PERIOD = Parameter("flux.period", "Flux Period", "A/fluxquanta")
"""Period of one fluxquanta in Ampere per fluxquanta."""

FLUX_OFFSET = Parameter("flux.offset", "Flux Offset", "fluxquanta")
"""Flux offset in units of fluxquanta. Similar to Flux Voltage."""

QUBIT_FREQUENCY_SWEET_SPOT = Parameter("qubit.frequency.sweet_spot", "Qubit Frequency at Sweet Spot", "Hz")
"""Qubit frequency at it's sweet spot."""

ACTIVE_RESET_FIDELITY = Parameter(
    "active_reset_fidelity", "Total fidelity of the reset in the single shot readout with reset experiment", ""
)
"""Total fidelity of the reset in the single shot readout with reset experiment"""

MINIMAL_FEEDBACK_DELAY = Parameter(
    "minimal_feedback_delay", "Minimal feedback delay for the active reset to perform well", "s"
)
"""Minimal feedback delay for the active reset to perform well"""

COUPLER_IDLING_POINT = Parameter("flux.coupler_idling_point", "Coupler idling point", "V")
"""Coupler flux idling point that minimizes the qubit-to-qubit coupling."""

CIRCUIT_EXECUTION_CONSTANT_OVERHEAD = Parameter(
    "circuit_execution_constant_overhead", "Constant overhead in executing circuits", "s"
)
"""Benchmarked overhead in executing circuits (independent of the number of gates executed)."""

CIRCUIT_EXECUTION_EXCESS_TIME_PER_GATE = Parameter(
    "circuit_execution_excess_time_per_gate", "Excess time per gate in circuit execution", "s"
)
"""Benchmarked extra time per gate in circuit execution that is in addition to the actual gate execution."""

PROBE_LINE_TRIGGER_DELAY = Parameter("readout.trigger_delay", "Probe line trigger delay", "s")
"""Trigger delay of a probe line AWG."""

DRIVE_TRIGGER_DELAY = Parameter("drive.awg.trigger_delay", "Drive awg trigger delay", "s")
"""Trigger delay of a qubit drive AWG."""

FLUX_TRIGGER_DELAY = Parameter("flux.awg.trigger_delay", "Flux awg trigger delay", "s")
"""Trigger delay of a flux pulse AWG."""

COMPONENT_FREQUENCY = Parameter("frequency", label="Component frequency", unit="Hz")
"""Component frequency, used for computational resonators and qubits without drive controller."""

EXCITE_PULSE = Parameter("is_pi_pulse_on", "Pi pulse", "", DataType.BOOLEAN)

EXTREME_FLUX_PULSE_AMPLITUDES = Parameter(
    "extreme_pulse_amplitudes", "Extreme flux pulse amplitudes", collection_type=CollectionType.LIST
)
"""Calibration ranges for the amplitude of a flux pulse for CZ gate.
The first value is an amplitude where no population exchange happens, and the second marks the amplitude where the
second exchange happens.
"""

FLUX_PULSE_SPINE_AMPLITUDES = Parameter(
    "pulse_spine_amplitudes", "Flux pulse spine amplitudes", collection_type=CollectionType.LIST
)

FLUX_PULSE_SPINE_POLYNOMIAL_COEFFICIENTS = Parameter(
    "pulse_spine_polynomial_coefficients",
    "Flux pulse spine polynomial coefficients",
    collection_type=CollectionType.LIST,
)
"""Polynomial coefficients of the 1D curve in amplitude-width plane, used to search for the optimal pulse for the CZ
gate."""

F0G1_PULSE_STARK_SHIFT_POLYNOMIAL_COEFFICIENTS = Parameter(
    "f0g1_pulse_stark_shift_polynomial_coefficients",
    "F0G1 pulse stark shift polynomial coefficients",
    unit="Hz",
    collection_type=CollectionType.LIST,
)
"""Polynomial coefficients of the 1D curve in amplitude-frequency plane, used to search for the optimal IQPulse
modulation frequency to drive the F0G1 transition while taking into the account the AC Stark shift at a
certain amplitude (used for ``lru`` or ``reset`` gates)."""

FLUX_CROSSTALK_MATRIX = Parameter(
    "flux_group.crosstalk.crosstalk_matrix",
    "Normalized Flux crosstalk matrix",
    unit="",
    collection_type=CollectionType.NDARRAY,
)

FLUX_GATE_DETUNING = Parameter("flux_gate_detuning", "Qubit frequency detuning during flux pulse gate", "Hz")

FREQUENCY_TO_FLUX_POLYNOMIAL_COEFFICIENTS = Parameter(
    "frequency_to_flux_polynomial_coefficients",
    "Frequency to flux polynomial coefficients",
    unit="V/Hz",
    collection_type=CollectionType.LIST,
)

SAMPLES = Parameter("samples", "Sample axis from the monitor trace", "", DataType.INT)

SINGLE_SHOT_READOUT_01_ERROR = Parameter(
    "single_shot_01_error", "Probability of measuring 1 if the 0 state was prepared.", ""
)
SINGLE_SHOT_READOUT_10_ERROR = Parameter(
    "single_shot_10_error", "Probability of measuring 0 if the 1 state was prepared.", ""
)
SIGNAL = Parameter("signal", "Signal", "V", DataType.FLOAT)

SNR = Parameter("snr", "Signal to noise ratio", "", DataType.FLOAT)

NOISE = Parameter("noise", "Noise", "V", DataType.FLOAT)

TIME = Parameter("time", "Time axis from the monitor trace", "s", DataType.FLOAT)

TIME_TRACE = Parameter("time_trace", "Time trace", "V", DataType.FLOAT)

WEIGHTS_I = Parameter("readout.integration_weights_I", "I integration weights", "", DataType.FLOAT)

WEIGHTS_Q = Parameter("readout.integration_weights_Q", "Q integration weights", "", DataType.FLOAT)

CPHASE_DERIVATIVE = Parameter("cphase_derivative", "Derivative of CPhase wrt Pulse Amplitude", "rad/V")
"""Parameter for the CPhase derivative observation."""

CPHASE_PI_DISTANCE = Parameter("cphase_pi_distance", "Distance from the pi CPhase angle", "rad")
"""Parameter for the CPhase derivative observation."""

KAPPA_EFF = Parameter("kappa_eff", "effective kappa", "Hz", DataType.FLOAT)
KAPPA_EFF_G = Parameter("kappa_eff_g", "effective kappa for g state", "Hz", DataType.FLOAT)
KAPPA_EFF_E = Parameter("kappa_eff_e", "effective kappa for e state", "Hz", DataType.FLOAT)
"""A state independent and two state dependent effective kappas."""

KAPPA_PURCELL = Parameter("kappa_purcell", "Purcell kappa", "Hz", DataType.FLOAT)
"""Kappa for Purcell resonator."""

J_RR_PR = Parameter("j_rr_pr", "coupling between readout and purcell resonators", "Hz", DataType.FLOAT)
"""Coupling strength between readout and Purcell resonators."""

PURCELL_FREQUENCY = Parameter("purcell_frequency", "Purcell filter frequency", "Hz", DataType.FLOAT)
"""Resonant frequency of the Purcell resonator."""

CRITICAL_PHOTON_NUMBER = Parameter("critical_photon_number", "Critical photon number", "", DataType.FLOAT)
"""Jaynes-Cummings critical photon number."""

QUBIT_RESONATOR_COUPLING = Parameter("qb_resonator_coupling", "Qubit-resonator coupling", "Hz", DataType.FLOAT)
"""Coupling strength between qubit and readout resonator."""

QUANTUM_EFFICIENCY = Parameter("quantum_efficiency", "Quantum efficiency", "", DataType.FLOAT)
"""Quantum efficiency as defined in arXiv:1711.05336. """

PHOTON_NUMBER = Parameter("photon_number", "Number of photons in resonator")
PHOTON_NUMBER_G = Parameter("photon_number_g", "Number of photons in resonator in ground state")
PHOTON_NUMBER_E = Parameter("photon_number_e", "Number of photons in resonator in excited state")
"""Number of photons in readout resonator for state independent and state dependent cases."""

PHOTON_CONVERSION_FACTOR = Parameter(
    "photon_conversion_factor",
    "Linear conversion factor between readout amplitude and photon number.",
    "Hz",
    DataType.FLOAT,
)
PHOTON_CONVERSION_FACTOR_G = Parameter(
    "photon_conversion_factor_g",
    "Linear conversion factor between readout amplitude and photon number for g state",
    "Hz",
    DataType.FLOAT,
)
PHOTON_CONVERSION_FACTOR_E = Parameter(
    "photon_conversion_factor_e",
    "Linear conversion factor between readout amplitude and photon number for e state",
    "Hz",
    DataType.FLOAT,
)
"""Conversion factor between readout DAC amplitude and photon numbers.
Conversion defined as: A^2 * a_dac*2 = n where A is the photon conversion factor, a_dac is the
amplitude in arbitrary units and n is the number of photons."""

SELF_KERR = Parameter("self_kerr", "Self Kerr term", "Hz", DataType.FLOAT)
SELF_KERR_G = Parameter("self_kerr_g", "Self Kerr term for g state", "Hz", DataType.FLOAT)
SELF_KERR_E = Parameter("self_kerr_e", "Self Kerr term for e state", "Hz", DataType.FLOAT)
"""State independent and state dependent self-Kerr terms."""

AC_STARK_SHIFT_G = Parameter("ac_stark_shift_g", "AC Stark shift due to probe tone in ground state")
AC_STARK_SHIFT_E = Parameter("ac_stark_shift_e", "AC Stark shift due to probe tone in excited state")
"""Readout induced AC Stark shifts for two states."""

PHOTON_NUMBER_OFFSET = Parameter("photon_number_offset", "Lorentzian offset of photon numbers")
PHOTON_NUMBER_OFFSET_G = Parameter("photon_number_offset_g", "Lorentzian offset of photon numbers for g state")
PHOTON_NUMBER_OFFSET_E = Parameter("photon_number_offset_e", "Lorentzian offset of photon numbers for e state")
"""Offsets of photon numbers for state independent and state dependent cases."""

READOUT_FREQUENCY = Parameter("frequency", "state independent readout frequency", "Hz", DataType.FLOAT)
READOUT_FREQUENCY_G = Parameter("frequency_g", "g state readout frequency", "Hz", DataType.FLOAT)
READOUT_FREQUENCY_E = Parameter("frequency_e", "e state readout frequency", "Hz", DataType.FLOAT)
"""Resonant frequency of the dressed readout resonator."""

QUBIT_RESONATOR_DETUNING = Parameter("qb_resonator_detuning", "Qubit-resonator detuning", "Hz", DataType.FLOAT)
"""Detuning between the qubit and the resonator frequenies."""

AC_STARK_QUBIT_TO_SWEEP = Parameter("qubit_to_sweep", "AC Stark qubit to sweep", "", DataType.STRING)

MEASURE_GATE_NAME = Parameter("measure_gate_name", "Measure gate (quantum operation) name", "", DataType.STRING)

MAX_RABI_FREQUENCY = Parameter("max_rabi_frequency", "Rabi frequency at unit amplitude", "Hz", DataType.FLOAT)

# whitelisted "characterization" parameters

MEASURE_CHARACTERIZATION_PARAMETERS = [
    AVERAGE_RESPONSE_PHASE,
    AVERAGE_RESPONSE_G,
    AVERAGE_RESPONSE_E,
    AVERAGE_RESPONSE_F,
    SINGLE_SHOT_01_ERROR,
    SINGLE_SHOT_10_ERROR,
]

# TODO: handle the characterization parameter loading differently, this whitelist is just a temp solution
FLUX_PULSE_GATE_CHARACTERIZATION_PARAMETERS = [
    EXTREME_FLUX_PULSE_AMPLITUDES,
    FLUX_PULSE_SPINE_AMPLITUDES,
    FLUX_PULSE_SPINE_POLYNOMIAL_COEFFICIENTS,
    T1_TIME,
    T2_RAMSEY_TIME,
]

# TODO: handle the characterization parameter loading differently, this whitelist is just a temp solution
SINGLE_QUBIT_MODEL_CHARACTERIZATION_PARAMETERS = [
    T1_TIME,
    T1_TIME_FITS,
    T2_RAMSEY_TIME,
    T2_RAMSEY_TIME_FITS,
    T2_ECHO_TIME,
    T2_ECHO_TIME_FITS,
    ANHARMONICITY,
    Parameter("voltage_period", label="Bias voltage period", unit="V"),
    Parameter("voltage_offset", label="Bias voltage offset", unit="V"),
    Parameter("squid_asymmetry", label="SQUID asymmetry"),
    Parameter("josephson_energy", "Josephson energy", unit="Hz"),
    Parameter("charging_energy", "Charging energy", unit="Hz"),
    Parameter("sweet_spot_frequency", "Sweet Spot Frequency", unit="Hz"),
    MAX_RABI_FREQUENCY,
]

SINGLE_QUBIT_READOUT_MODEL_CHARACTERIZATION_PARAMETERS = [
    DISPERSIVE_SHIFT,
    KAPPA_EFF,
    KAPPA_EFF_G,
    KAPPA_EFF_E,
    KAPPA_PURCELL,
    J_RR_PR,
    PURCELL_FREQUENCY,
    CRITICAL_PHOTON_NUMBER,
    QUBIT_RESONATOR_COUPLING,
    T1_TIME_DURING_READOUT,
    PHOTON_CONVERSION_FACTOR_G,
    PHOTON_CONVERSION_FACTOR_E,
    READOUT_FREQUENCY_G,
    READOUT_FREQUENCY_E,
    PHOTON_NUMBER_OFFSET_G,
    PHOTON_NUMBER_OFFSET_E,
    SELF_KERR,
    SELF_KERR_G,
    SELF_KERR_E,
    QUANTUM_EFFICIENCY,
]

SINGLE_COUPLER_MODEL_CHARACTERIZATION_PARAMETERS = [COUPLER_QUBIT_NORMALIZED_COUPLING]

RZ_ANGLE = Parameter("rz_angle", "Z rotation angle", "rad")
"""Parameter for the Z rotation angle. Used for adjusting the relative phase between two components in the MOVE
calibration."""
