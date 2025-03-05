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
"""Mapping calibration observations to station settings.

Calibration observations
------------------------

Observations are physical quantities stored in the EXA database. CPC uses them to represent the
calibrated values of various instrument settings and gate parameters, needed to execute quantum circuits.
The following :class:`iqm.station_control.interface.models.observation.ObservationLite` fields are needed
for this purpose:

* ``dut_field``: identifies the quantity, and possibly the QPU component(s) it is associated with
* ``value``: value of the quantity
* ``unit``: unit of the quantity (currently only base SI units are used, e.g. Hz instead of GHz)

The calibration observations come from two conceptually different sources.

* :class:`ConfigurationSource` observations determine the **base operating point** of the station
  (in principle an arbitrary choice), and form the input of the calibration procedure.
* :class:`AnalysisSource` observations are the output of the calibration procedure.


Mapping to station settings
---------------------------

When EXA saves its settings as observations, it maps the settings tree paths directly to observation
``dut_field`` paths. CPC uses the same direct mapping, with a single minor modification ("options.end_delay"),
to map observation ``dut_field`` paths to Station Control controller settings paths.

Quantum operation parameters are stored in the EXA settings tree under the top-level branch ``gates``.
This data is only used in building the instruction schedules.

* The controller settings paths may change whenever Station Control is updated, since it
  consumes the settings. Hence, this may break old calibration sets.
* New paths may be introduced into the calibration set when the calibration procedure changes,
  or a new gate implementation is introduced.

CPC only consumes observations created by the calibration process, stored explicitly as calibration
sets in the database.


Static station settings
-----------------------

In addition to the station settings obtained from the calibration set, circuit execution also
requires some static settings that typically change only when Station Control is updated, or
the station itself is physically modified. The settings that depend on the makeup of a particular
station, e.g. the types of the control instruments, are set in the ``station.yml`` configuration file
if possible. Examples of such settings are the input and output ranges and powers of various instruments.
Note that these settings can also be considered a part of the definition of the basic operating point.

For some station settings, the default value is already good.

Finally, there are some static station settings that are defined in this module, mostly because if
they were set in ``station.yml`` they might negatively interfere with running EXA experiments on
the station.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from exa.common.data.parameter import CollectionType, DataType, Parameter, Setting
from exa.common.data.setting_node import SettingNode
from iqm.cpc.compiler.errors import CalibrationError
from iqm.pulla.interface import CalibrationSet

if TYPE_CHECKING:
    from exa.common.data.value import ObservationValue

### Naming convention for controllers.

READOUT_CONTROLLER = "{}__readout"
DRIVE_CONTROLLER = "{}__drive"
FLUX_CONTROLLER = "{}__flux"


@dataclass(frozen=True)
class Map:
    """Mapping from a calibration observation path to a corresponding station settings path.

    If :attr:`settings_path_template` is ``None``, it is derived from :attr:`observation_path_template`.

    A Parameter object is included in the mapping to conveniently handle the unit and data type.
    Its name is unused.
    """

    parameter: Parameter
    observation_path_template: str
    settings_path_template: str | None = None
    required: bool = True

    def observation_path(self, component: str) -> str:
        """Observation path for the given component."""
        return self.observation_path_template.format(component)

    def settings_path(self, component: str) -> str:
        """Settings path for the given component."""
        if self.settings_path_template is None:
            # derive from observation_path
            return self.observation_path_template.replace("controllers.", "").replace(".", "__", 1).format(component)
        return self.settings_path_template.format(component)


def find_observation(
    observation_path: str,
    calibration_set: CalibrationSet,
    *,
    required: bool = True,
) -> ObservationValue:
    """Return the value of the given calibration observation, or raise an error.

    Args:
        observation_path: observation we want to find in ``calibration_set``
        calibration_set: mapping of observation paths to observation values
        required: iff ``True`` and the observation cannot be found, raise an error

    Returns:
        value of the observation, or ``None`` if not found

    Raises:
        CalibrationError: ``required`` is ``True`` and the observation cannot be
            found in ``calibration_set``

    """
    obs_value = calibration_set.get(observation_path)
    if obs_value is None and required:
        raise CalibrationError(f"Missing calibration observation: {observation_path}")
    return obs_value


# TODO until station type records (defining the instrumentation for each chip component) are available,
# CPC cannot know which instruments are being used, so we must generate settings for all supported instruments
# if the corresponding observation is in the calset.

_per_qubit = (
    Map(Parameter("", "Flux voltage at parking spot", "V"), "controllers.{}.flux.voltage"),
    Map(
        Parameter("", "Qubit drive frequency, should match the g-e transition frequency at parking spot", "Hz"),
        "controllers.{}.drive.frequency",
    ),
    Map(
        Parameter("", "DC offset at I port of the drive mixer", "V"),
        "controllers.{}.drive.awg.mixer_correction.dc_bias_i",
        required=False,
    ),  # ZI HDAWG
    Map(
        Parameter("", "DC offset at Q port of the drive mixer", "V"),
        "controllers.{}.drive.awg.mixer_correction.dc_bias_q",
        required=False,
    ),  # ZI HDAWG
    # For SHFSG the center frequency has to be a multiple of some fixed granularity.
    # Hence we must set the center frequency instead of the IF here, as the target frequency can be anything.
    # For the HDAWG, setting the IF to a good value is more optimal.
    Map(
        Parameter("", "Intermediate frequency for drive mixer", "Hz"),
        "controllers.{}.drive.awg.intermediate_frequency",
        required=False,
    ),  # ZI HDAWG
    Map(
        Parameter("", "Center frequency for drive mixer", "Hz"),
        "controllers.{}.drive.awg.center_frequency",
        required=False,
    ),  # ZI SHFSG
    Map(Parameter("", "Trigger delay for drive AWG", "s"), "controllers.{}.drive.awg.trigger_delay", required=False),
)

_per_flux_pulsed_qubit = {
    # precompensation settings
    Map(
        Parameter("", "Precompensation timeconstants", "", DataType.FLOAT, CollectionType.LIST),
        "controllers.{}.flux.awg.precompensation.timeconstants",
        required=False,
    ),
    Map(
        Parameter("", "Precompensation amplitudes", "", DataType.FLOAT, CollectionType.LIST),
        "controllers.{}.flux.awg.precompensation.amplitudes",
        required=False,
    ),
    Map(Parameter("", "Trigger delay for flux AWG", "s"), "controllers.{}.flux.awg.trigger_delay", required=False),
}

_per_coupler = (
    Map(Parameter("", "Flux voltage at idling spot", "V"), "controllers.{}.flux.voltage"),
    # precompensation settings
    Map(
        Parameter("", "Precompensation timeconstants", "", DataType.FLOAT, CollectionType.LIST),
        "controllers.{}.flux.awg.precompensation.timeconstants",
        required=False,
    ),
    Map(
        Parameter("", "Precompensation amplitudes", "", DataType.FLOAT, CollectionType.LIST),
        "controllers.{}.flux.awg.precompensation.amplitudes",
        required=False,
    ),
    Map(Parameter("", "Trigger delay for flux AWG", "s"), "controllers.{}.flux.awg.trigger_delay", required=False),
)

_per_probe_line = (
    # TWPA settings.
    Map(Parameter("", "On/Off status of TWPA", "", DataType.BOOLEAN), "controllers.{}.twpa.status", required=False),
    Map(Parameter("", "TWPA bias voltage", "V"), "controllers.{}.twpa.voltage", required=False),
    Map(Parameter("", "TWPA bias voltage_1", "V"), "controllers.{}.twpa.voltage_1", required=False),
    Map(Parameter("", "TWPA bias voltage_2", "V"), "controllers.{}.twpa.voltage_2", required=False),
    Map(Parameter("", "Frequency of the LO driving the TWPA", "Hz"), "controllers.{}.twpa.frequency", required=False),
    Map(Parameter("", "Power of the LO driving the TWPA", "dBm"), "controllers.{}.twpa.power", required=False),
    # Center frequency must be within the AWG IF bandwidth from every <qubit>.readout.frequency
    Map(
        Parameter("", "Probe line center frequency", "Hz"),
        "controllers.{}.readout.center_frequency",
        required=True,
    ),
    Map(
        Parameter("", "Trigger delay for readout instrument", "s"),
        "controllers.{}.readout.trigger_delay",
        required=False,
    ),
)

_per_qpu = (
    Map(
        Parameter("", "Delay from end of sequence to next trigger", "s"),
        "controllers.options.end_delay",
        "options.end_delay",
    ),
)

_per_boundary_qubit = (
    Map(Parameter("", "Flux voltage at parking spot", "V"), "controllers.{}.flux.voltage"),
    # For SHFSG neighboring (paired) cores must have identical center frequencies.
    # If we do not set it here, it is possible that the cal set and the station.yaml have different values.
    Map(
        Parameter("", "Center frequency for drive mixer", "Hz"),
        "controllers.{}.drive.awg.center_frequency",
        required=False,
    ),
)

_per_boundary_coupler = (Map(Parameter("", "Flux voltage at idling spot", "V"), "controllers.{}.flux.voltage"),)


# Mapping from station settings parameter to the value of the setting, for settings
# which need to be set to fixed values for circuit execution.

_per_qubit_static = {
    # turn on the drive instruments
    # TODO Is this required? Isn't a nonempty playlist segment enough?
    Parameter("{}__drive.status", "", "", DataType.BOOLEAN): True,
    Parameter("{}__drive.local_oscillator.status", "", "", DataType.BOOLEAN): True,  # ZI HDAWG
    Parameter("{}__drive.awg.status", "", "", DataType.BOOLEAN): True,
}

_per_flux_pulsed_qubit_static = {
    # turn on the flux pulse instruments
    Parameter("{}__flux.awg.status", "", "", DataType.BOOLEAN): True,
}

_per_coupler_static = {
    # turn on the flux pulse instruments
    Parameter("{}__flux.awg.status", "", "", DataType.BOOLEAN): True,
}

_per_probe_line_static = {
    # turn on the readout instruments
    Parameter("{}__readout.local_oscillator.status", "", "", DataType.BOOLEAN): True,  # ZI UHFQA
}

_per_boundary_qubit_static: dict[Parameter, ObservationValue] = {
    # turning on the slow flux voltage instruments currently does not require a setting
}

_per_boundary_flux_pulsed_qubit_static = {
    # turn on the flux pulse instruments (which also produce DC flux!)
    Parameter("{}__flux.awg.status", "", "", DataType.BOOLEAN): True,
}

_per_boundary_coupler_static = {
    # turn on the flux pulse instruments (which also produce DC flux!)
    Parameter("{}__flux.awg.status", "", "", DataType.BOOLEAN): True,
}


# Settings that are needed to achieve the necessary number of circuit execution shots requested by user. They are
# neither static nor get their value from calibration set.
_per_qpu_repetitions = (
    Parameter("options.averaging_bins", "Average the repeats into this many bins", ""),
    Parameter("options.playlist_repeats", "Number of times to repeat execution of corresponding playlist", ""),
)


def _apply_static_settings(component: str, settings: dict[Parameter, ObservationValue], node: SettingNode) -> None:
    """Add the given static settings to the given settings tree for the given component."""
    for base_param, value in settings.items():
        _create_and_add_setting(base_param.name.format(component), base_param, value, node)


def _create_and_add_setting(setting_path: str, param: Parameter, value: ObservationValue, node: SettingNode) -> None:
    """Add the given parameter, with the given value, to the given settings tree path, starting from node."""
    param = param.copy(name=setting_path)
    node[setting_path] = Setting(param, value)


def build_station_settings(
    *,
    circuit_qubits: Iterable[str],
    circuit_couplers: Iterable[str],
    measured_probe_lines: Iterable[str],
    shots: int,
    calibration_set: CalibrationSet,
    boundary_qubits: Iterable[str],
    boundary_couplers: Iterable[str],
    flux_pulsed_qubits: Collection[str],
) -> SettingNode:
    """Build the station settings for executing a batch of quantum circuits using the given QPU
    elements and calibration data.

    Args:
        circuit_qubits: physical qubit names used in the circuit
        circuit_couplers: coupler names used in the circuit
        measured_probe_lines: probe line names used in the measurements
        shots: number of times to repeat each circuit's execution
        calibration_set: calibration set as a mapping from observation paths to observation values
        boundary_qubits: physical qubits connected to the boundary_couplers but not in circuit_qubits
        boundary_couplers: coupler names of couplers connected to the circuit boundary but not in circuit_couplers
        flux_pulsed_qubits: names of qubits that have flux pulse capability

    Returns:
        station settings tree

    """
    root = SettingNode("root")

    def apply_observations(component: str, observation_maps: Iterable[Map]) -> None:
        """Find all the required observations for the given component, and add
        the corresponding settings to the settings tree.

        Args:
            component: component name the corresponding observation applies to
            observation_maps: templates that map QPU components to observation and station setting paths,
                one per required observation

        """
        for obs_map in observation_maps:
            value = find_observation(obs_map.observation_path(component), calibration_set, required=obs_map.required)
            if value is not None:
                _create_and_add_setting(obs_map.settings_path(component), obs_map.parameter, value, root)

    # For each qubit, coupler and probe line, try to find the required calibration observations.
    # Then add the required static settings for circuit execution.
    for qubit in circuit_qubits:
        apply_observations(qubit, _per_qubit)
        _apply_static_settings(qubit, _per_qubit_static, root)
        if qubit in flux_pulsed_qubits:
            apply_observations(qubit, _per_flux_pulsed_qubit)
            _apply_static_settings(qubit, _per_flux_pulsed_qubit_static, root)

    for coupler in circuit_couplers:
        apply_observations(coupler, _per_coupler)
        _apply_static_settings(coupler, _per_coupler_static, root)

    for pl in measured_probe_lines:
        apply_observations(pl, _per_probe_line)
        _apply_static_settings(pl, _per_probe_line_static, root)

    for qubit in boundary_qubits:
        apply_observations(qubit, _per_boundary_qubit)
        _apply_static_settings(qubit, _per_boundary_qubit_static, root)
        if qubit in flux_pulsed_qubits:
            _apply_static_settings(qubit, _per_boundary_flux_pulsed_qubit_static, root)

    for coupler in boundary_couplers:
        apply_observations(coupler, _per_boundary_coupler)
        _apply_static_settings(coupler, _per_boundary_coupler_static, root)

    # QPU-wide options
    apply_observations("", _per_qpu)
    for par in _per_qpu_repetitions:
        _create_and_add_setting(par.name, par, shots, root)

    return root
