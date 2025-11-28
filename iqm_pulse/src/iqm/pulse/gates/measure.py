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
r"""Projective measurement in the Z basis."""

from __future__ import annotations

from collections.abc import Iterable
from copy import copy, deepcopy
from dataclasses import replace
import functools
from typing import TYPE_CHECKING

import numpy as np

from exa.common.data.parameter import CollectionType, DataType, Parameter, Setting
from iqm.pulse.gate_implementation import (
    PROBE_LINES_LOCUS_MAPPING,
    SINGLE_COMPONENTS_WITH_READOUT_LOCUS_MAPPING,
    CompositeGate,
    CustomIQWaveforms,
    Locus,
    OILCalibrationData,
)
from iqm.pulse.playlist import Schedule, Segment
from iqm.pulse.playlist.channel import ProbeChannelProperties
from iqm.pulse.playlist.instructions import (
    AcquisitionMethod,
    Block,
    ComplexIntegration,
    IQPulse,
    MultiplexedIQPulse,
    ReadoutTrigger,
    ThresholdStateDiscrimination,
    TimeTrace,
)
from iqm.pulse.playlist.waveforms import Constant, Samples
from iqm.pulse.timebox import MultiplexedProbeTimeBox, ProbeTimeBoxes, SchedulingStrategy, TimeBox

if TYPE_CHECKING:  # pragma: no cover
    from iqm.pulse.builder import ScheduleBuilder
    from iqm.pulse.quantum_ops import QuantumOp

DEFAULT_INTEGRATION_KEY = "readout.result"
DEFAULT_TIME_TRACE_KEY = "readout.time_trace"
FEEDBACK_KEY = "feedback"
TIMING_TOLERANCE = 1e-12


class Measure_CustomWaveforms(CustomIQWaveforms):
    """Base class for implementing dispersive measurement operations with custom probe pulse waveforms.

    You may define a measurement implementation that uses the :class:`.Waveform`
    instances ``Something`` and ``SomethingElse`` as the probe pulse waveforms in the
    I and Q channels as follows:
    ``class MyGate(Measure_CustomWaveforms, i_wave=Something, q_wave=SomethingElse)``.

    The ``measure`` operation is factorizable, and its :attr:`arity` is 0, which together mean that it can operate
    on loci of any length, but is calibrated only on single component loci. When the gate is constructed in the
    ``len(locus) > 1`` case (e.g. ``builder.get_implementation('measure', ('QB1', 'QB2', 'QB3'))()``), the resulting
    :class:`.TimeBox` is constructed from the calibrated single-component implementations.

    For each measured component, the readout :class:`.IQPulse` will be modulated with the
    intermediate frequency (IF), computed as the difference between the readout
    frequency of that component and the probe line center frequency, and offset in phase
    by the readout phase of the component.

    The measurement is implemented using a :class:`.ReadoutTrigger` instruction, with a duration set by the
    requirements of the acquisition(s). Note that this is typically different from
    ``gates.measure.constant.{locus}.duration``, which is the probe pulse duration.
    """

    root_parameters = {
        "duration": Parameter("", "Readout pulse duration", "s"),
        "frequency": Parameter("", "Readout pulse frequency", "Hz"),
        "phase": Parameter("", "Readout pulse phase", "rad"),
        "amplitude_i": Parameter("", "Readout channel I amplitude", ""),
        # TODO do we really need these defaults? are they used anywhere?
        "amplitude_q": Setting(Parameter("", "Readout channel Q amplitude", ""), 0.0),
        "integration_length": Parameter("", "Integration length", "s"),
        "integration_weights_I": Setting(
            Parameter("", "Integration weights for channel I", "", collection_type=CollectionType.NDARRAY),
            np.array([]),
        ),
        "integration_weights_Q": Setting(
            Parameter("", "Integration weights for channel Q", "", collection_type=CollectionType.NDARRAY),
            np.array([]),
        ),
        "integration_threshold": Parameter("", "Integration threshold", ""),
        "acquisition_type": Setting(Parameter("", "Acquisition type", "", data_type=DataType.STRING), "threshold"),
        "acquisition_delay": Parameter("", "Acquisition delay", "s"),
    }

    def __init__(
        self, parent: QuantumOp, name: str, locus: Locus, calibration_data: OILCalibrationData, builder: ScheduleBuilder
    ):
        super().__init__(parent, name, locus, calibration_data, builder)

        self._multiplexed_timeboxes: dict[tuple[str, str, bool], TimeBox] = {}
        """Cache for :meth:`probe_timebox`."""
        self._time_traces: dict[tuple[str, float | None, float | None, str], TimeBox] = {}
        """Cache for :meth:`time_trace`."""
        self._neighborhood_components: set[str] = set(self.locus) | set(
            self.builder.chip_topology.component_to_probe_line[q] for q in self.locus
        )

        if len(locus) == 1:
            # prepare the single-component measurement
            probe_line: ProbeChannelProperties = builder.channels[  # type: ignore[assignment]
                builder.get_probe_channel(locus[0])
            ]
            # readout duration is determined by the acquisition, probe pulses are truncated to fit this window
            self._duration = (
                probe_line.duration_to_int_samples(
                    probe_line.round_duration_to_granularity(
                        calibration_data["acquisition_delay"] + calibration_data["integration_length"]
                    )
                )
                + probe_line.integration_stop_dead_time
            )
            self._probe_offset = probe_line.integration_start_dead_time
            # "duration" is only used by the probe pulse
            waveform_params = self.convert_calibration_data(
                calibration_data,
                {k: v for k, v in self.parameters.items() if k not in self.root_parameters},
                probe_line,
            )
            # unconverted cal data that corresponds to a root param (not duration)
            root_params = {k: v for k, v in calibration_data.items() if k in self.root_parameters and k != "duration"}
            # do some conversions TODO are these consistent?
            root_params["integration_length"] = probe_line.duration_to_int_samples(root_params["integration_length"])
            root_params["acquisition_delay"] = round(probe_line.duration_to_samples(root_params["acquisition_delay"]))

            if_freq = (calibration_data["frequency"] - probe_line.center_frequency) / probe_line.sample_rate

            self._probe_instruction, self._acquisition_method = self._build_instructions(
                waveform_params, root_params, if_freq
            )

    def _build_instructions(
        self, waveform_params: OILCalibrationData, root_params: OILCalibrationData, if_freq: float
    ) -> tuple[IQPulse, AcquisitionMethod]:
        """Builds a probe pulse and an acquisition method using the calibration data.

        Subclasses may override this method if needed.
        """
        if self.dependent_waves:
            wave_i = self.wave_i(**waveform_params)
            wave_q = self.wave_q(**waveform_params)
        else:
            wave_i = self.wave_i(**waveform_params["i"])
            wave_q = self.wave_q(**waveform_params["q"])

        probe_pulse = IQPulse(
            duration=waveform_params["n_samples"],
            wave_i=wave_i,
            wave_q=wave_q,
            scale_i=root_params["amplitude_i"],
            scale_q=root_params["amplitude_q"],
            phase=root_params["phase"],
            modulation_frequency=if_freq,
        )

        integration_length = root_params["integration_length"]
        weights_i = root_params.get("integration_weights_I")
        weights_q = root_params.get("integration_weights_Q")
        if weights_i is not None and weights_i.size and weights_q is not None and weights_q.size:
            # TODO: the weights should be in the params, so we should not need to check that
            # make sure everything indeed works like this
            if not integration_length == weights_i.size == weights_q.size:
                raise ValueError(
                    "Integration length does not match with the provided integration weight lengths. "
                    f"For {self.locus}: the integration length is {integration_length} samples, "
                    f" the I weights vector length is {weights_i.size}, and the Q weights vector length"
                    f" is {weights_q.size}."
                )
            weights = IQPulse(
                duration=integration_length,
                wave_i=Samples(weights_i),
                wave_q=Samples(weights_q),
                scale_i=1.0,
                scale_q=1.0,
                phase=0.0,
                modulation_frequency=if_freq,  # TODO: should be fixed to -if_freq in Programmable RO Phase2?
            )
        else:
            const = Constant(integration_length)
            weights = IQPulse(
                duration=integration_length,
                wave_i=const,
                wave_q=const,
                scale_i=1.0,
                scale_q=0.0,
                phase=0.0,
                modulation_frequency=if_freq,  # TODO: should be fixed to -if_freq in Programmable RO Phase2?
            )

        acquisition_type = root_params.get("acquisition_type", self.root_parameters["acquisition_type"].value)  # type: ignore[union-attr]
        acquisition_label = "TO_BE_REPLACED"
        op_and_implementation = f"{self.parent.name}.{self.name}"
        if acquisition_type == "complex":
            acquisition_method = ComplexIntegration(
                label=acquisition_label,
                delay_samples=root_params["acquisition_delay"],
                weights=weights,
                implementation=op_and_implementation,
            )
        elif acquisition_type == "threshold":
            acquisition_method = ThresholdStateDiscrimination(
                label=acquisition_label,
                delay_samples=root_params["acquisition_delay"],
                weights=weights,
                threshold=root_params["integration_threshold"],
                implementation=op_and_implementation,
            )
        else:
            raise ValueError(f"Unknown acquisition type {acquisition_type}")

        return probe_pulse, acquisition_method

    def probe_timebox(self, key: str = "", feedback_key: str = "", do_acquisition: bool = True, **kwargs) -> TimeBox:
        """Returns a "naked" probe timebox that supports convenient multiplexing through
        ``MultiplexedProbeTimeBox.__add__``.

        This method can be used if the user wants to control the multiplexing explicitly. With two
        ``MultiplexedProbeTimeBox``es ``A`` and ``B`` the result ``A + B`` has all the ``ReadoutTrigger`` instructions
        on each probe channel of ``A`` and ``B`` multiplexed together and played simultaneously.

        Args:
            key: The readout results generated on this trigger will be assigned to
                ``f"{qubit}__{key}"``, where ``qubit`` goes over the component names in ``self.locus``. If empty,
                the key `"readout.result"` will be used to maintain backwards compatibility.
            feedback_key: The signals generated by this measure operation are routed using this key for
                fast feedback purposes. See :meth:`__call__`.
            do_acquisition: if False, no acquisitions are added.

        Returns:
            MultiplexedProbeTimeBox containing the ReadoutTrigger instruction.

        """
        args = (key, feedback_key, do_acquisition)
        # additional caching for probe timeboxes due to the fact that both ._call and .time_trace use this
        if args not in self._multiplexed_timeboxes:
            if len(self.locus) == 1:
                label_key = key or DEFAULT_INTEGRATION_KEY
                replacements = {"label": f"{self.locus[0]}__{label_key}"}
                if feedback_key and isinstance(self._acquisition_method, ThresholdStateDiscrimination):
                    # TODO: use the actual ``feedback_key`` when AWGs support multiple feedback labels
                    replacements["feedback_signal_label"] = f"{self.locus[0]}__{FEEDBACK_KEY}"
                acquisitions = (replace(self._acquisition_method, **replacements),) if do_acquisition else ()  # type: ignore[arg-type]
                multiplexed_iq = MultiplexedIQPulse(
                    duration=self._probe_instruction.duration + self._probe_offset,
                    entries=((self._probe_instruction, self._probe_offset),),
                )
                readout_trigger = ReadoutTrigger(
                    duration=self._duration, probe_pulse=multiplexed_iq, acquisitions=acquisitions
                )
                probe_channel = self.builder.get_probe_channel(self.locus[0])
                try:
                    drive_channel = self.builder.get_drive_channel(self.locus[0])
                except KeyError:
                    drive_channel = ""

                if drive_channel:
                    # drive channel must be blocked, to prevent DD insertion while measurement is taking place
                    # unfortunately we must allow for different channel sample rates because of UHFQA
                    channels = self.builder.channels
                    drive_channel_props = channels[drive_channel]
                    rt_duration_in_seconds = channels[probe_channel].duration_to_seconds(readout_trigger.duration)
                    block_duration = drive_channel_props.duration_to_int_samples(
                        drive_channel_props.round_duration_to_granularity(
                            rt_duration_in_seconds, round_up=True, force_min_duration=True
                        )
                    )
                    probe_timebox = MultiplexedProbeTimeBox.from_readout_trigger(
                        readout_trigger=readout_trigger,
                        probe_channel=probe_channel,
                        locus_components=self.locus,
                        label=f"{self.__class__.__name__} on {self.locus}",
                        block_channels=[drive_channel],
                        block_duration=block_duration,
                    )
                else:
                    probe_timebox = MultiplexedProbeTimeBox.from_readout_trigger(
                        readout_trigger=readout_trigger,
                        probe_channel=probe_channel,
                        locus_components=self.locus,
                        label=f"{self.__class__.__name__} on {self.locus}",
                    )
            else:
                probe_timeboxes = [
                    self.sub_implementations[c].probe_timebox(key, feedback_key, do_acquisition)  # type: ignore[attr-defined]
                    for c in self.locus
                ]
                probe_timebox = functools.reduce(lambda x, y: x + y, probe_timeboxes)
            if isinstance(probe_timebox, TimeBox):  # FIXME: not needed once the measure implementations are cleaned up
                probe_timebox.neighborhood_components[0] = copy(self._neighborhood_components)
                if feedback_key:
                    # Block all the virtual channels from the probes involved in self.locus as we cannot know what AWG
                    # might be listening to the sent bits. NOTE: No Waits are added, the channels are just blocked in
                    # scheduling, so the impact to performance is negligible
                    probelines = {self.builder.chip_topology.component_to_probe_line[q] for q in self.locus}
                    for probe in probelines:
                        probe_timebox.neighborhood_components[0].update(
                            set(self.builder.get_virtual_feedback_channels(probe))
                        )
            self._multiplexed_timeboxes[args] = probe_timebox
        return self._multiplexed_timeboxes[args]

    def _call(self, key: str = "", feedback_key: str = "") -> TimeBox:
        """Returns a TimeBox containing the multiplexed simultaneous measurement.

        If ``len(self.locus) == 1``, the TimeBox contains the measurement for just that component, otherwise
        the measurements of components that belong to the same probeline are multiplexed together.

        The returned :class:`.TimeBox` instances behave like any other TimeBox in scheduling and circuit
        generation. With measurement TimeBoxes ``A`` and ``B`` the result ``A + B`` first plays the ``ReadoutTrigger``
        instructions of ``A`` and only then those of ``B`` in each probe channel. If the multiplexing features of
        :class:`.MultiplexedProbeTimeBox` are needed, the method :meth:`probe_timebox` can be used.

        In scheduling, the returned TimeBox blocks the locus components and the probe
        lines they are associated with.

        Args:
            key: Readout results generated on this trigger will be assigned to the acquisition labels
                ``f"{qubit}__{key}"``, where ``qubit`` goes over the component names in ``self.locus``.
                If empty, the key ``"readout.result"`` will be used to maintain backwards compatibility.
            feedback_key: If the readout mode is "threshold", the results generated by this ``measure`` operation
                are routed using the label ``f"{qubit}__{feedback_key}"`` for fast feedback purposes.
                The signals are picked up by :class:`.ConditionalInstruction`s that have the same label.
                The default value ``""`` means the signal is not routed anywhere. TODO: currently the HW does not
                support multiple feedback keys per drive channel, so the actual key used will be ``FEEDBACK_KEY``
                whenever any non-empty key is inputted. When the HW is improved, the actual key the user inputs
                should be passed.

        Returns:
            TimeBox containing the :class:`.ReadoutTrigger` instruction.

        """
        final_box = TimeBox.composite(
            [self.probe_timebox(key=key, feedback_key=feedback_key)], label=f"Readout on {self.locus}"
        )
        final_box.neighborhood_components[0] = final_box.children[0].neighborhood_components[0]
        return final_box

    def _get_probe_timebox_for_time_trace(self, key: str = "", feedback_key: str = "") -> TimeBox:
        """Utility method that can be overridden in subclasses if they have a return type `.probe_pulse`."""
        # FIXME: not needed once we align the return types of all these measure gates
        return self.probe_timebox(key=key, feedback_key=feedback_key)

    def time_trace(
        self,
        key: str = "",
        acquisition_delay: float | None = None,
        acquisition_duration: float | None = None,
        feedback_key: str = "",
    ) -> TimeBox:
        """Returns a multiplexed simultaneous measurement with an additional time trace acquisition.

        The returned ``TimeBox`` is the same as the one returned by :meth:`__call__` except the time trace
        acquisition is appended to the acquisitions of each probe line's ``ReadoutTrigger`` instruction.

        Args:
            key: Readout results generated on this trigger will be used to assigned to
                ``f"{qubit}__{key}"``, where ``qubit`` goes over the component names in ``self.locus``, whereas
                the recorded time traces will be assigned to ``f"{probe_line}__{key}"`` where
                ``probe_line`` goes through all the probe lines associated with ``self.locus``.
                If empty, the key ``"readout.result"`` will be used for integrated results and the key
                ``"readout.time_trace"`` for the recorded time traces.
            acquisition_delay: optionally override the time trace acquisition delay with this value (given in
                seconds). Does not affect the acqusition delays of the integrated measurements.
            acquisition_duration: optionally override the time trace acquisition duration with this value (given in
                seconds). Does not affect the integration lengths of the integrated measurements.
            feedback_key: The signals generated by the integration are routed using this label, prefixed by
                the component. See :meth:`__call__`.

        Returns:
            TimeBox containing the ReadoutTrigger instruction.

        """
        args = (key, acquisition_delay, acquisition_duration, feedback_key)
        # additional caching for time traces since the acquisitions differ from the ones in _call
        if args not in self._time_traces:
            probe_timebox = deepcopy(self._get_probe_timebox_for_time_trace(key, feedback_key))
            for probe_channel, segment in probe_timebox.atom.items():  # type: ignore[union-attr]
                readout_trigger = None
                for inst in segment:
                    if isinstance(inst, ReadoutTrigger):
                        readout_trigger = inst
                        break
                # TODO instead of editing the probe_timebox output contents, we should make the function itself do this
                # so we would not need to blindly search through the channels
                if readout_trigger is None:
                    continue

                probe_line = self.builder.channels[probe_channel]
                probe_name = self.builder._channel_to_component[probe_channel]

                if acquisition_delay is not None:
                    delay_samples = probe_line.duration_to_int_samples(acquisition_delay)
                else:
                    delay_samples = min(acq.delay_samples for acq in readout_trigger.acquisitions)
                if acquisition_duration is not None:
                    duration_samples = probe_line.duration_to_int_samples(acquisition_duration)
                else:
                    duration_samples = max(
                        acq.weights.duration + acq.delay_samples - delay_samples  # type: ignore[attr-defined]
                        for acq in readout_trigger.acquisitions
                    )
                label_key = key or DEFAULT_TIME_TRACE_KEY
                time_trace = TimeTrace(
                    label=f"{probe_name}__{label_key}",
                    delay_samples=delay_samples,
                    duration_samples=duration_samples,
                    implementation=f"{self.parent.name}.{self.name}",
                )
                trigger_with_trace = replace(readout_trigger, acquisitions=readout_trigger.acquisitions + (time_trace,))
                segment._instructions[0] = trigger_with_trace

            final_box = TimeBox.composite([probe_timebox])
            final_box.neighborhood_components[0] = probe_timebox.neighborhood_components[0]
            self._time_traces[args] = final_box
        return self._time_traces[args]

    def duration_in_seconds(self) -> float:
        probe_timebox = self.probe_timebox()
        readout_schedule = probe_timebox.atom
        return readout_schedule.duration_in_seconds(self.builder.channels)  # type: ignore[union-attr]

    @classmethod
    def get_locus_mapping_name(cls, operation_name: str, implementation_name: str) -> str:
        return SINGLE_COMPONENTS_WITH_READOUT_LOCUS_MAPPING


class Measure_Constant(Measure_CustomWaveforms, wave_i=Constant, wave_q=Constant):  # type: ignore[call-arg]
    """Implementation of a single-qubit projective, dispersive measurement in the Z basis.

    Uses a constant probe pulse.
    """


class Measure_Constant_Qnd(Measure_CustomWaveforms, wave_i=Constant, wave_q=Constant):  # type:ignore[call-arg]
    """Implementation of a single-qubit projective, non quantum demolition, dispersive
    measurements in the Z basis.

    Uses a constant probe pulse.
    """


class ProbePulse_CustomWaveforms(CustomIQWaveforms):
    """Base class for implementing a probe line measurement pulse with custom waveforms in the I and Q channels.

    With given :class:`.Waveform` waveform definitions ``Something`` and ``SomethingElse``,
    you may define a measurement implementation that uses them as follows:
    ``class MyGate(ProbePulse_CustomWaveforms, i_wave=Something, q_wave=SomethingElse)``.
    The measurement :class:`.IQPulse` instruction will not be automatically modulated
    by any frequency, so any modulations should be included in the I and Q waveforms themselves.

    Due to device limitations this implementation also has to integrate the readout signal
    (using arbitrary weights), even though it does not make much sense.

    Contrary to the ``Measure_CustomWaveforms`` class, this implementation acts on proble lines directly (i.e. its
    ``locus`` is a single probe line).
    """

    root_parameters = {
        "duration": Parameter("", "Readout pulse duration", "s"),
        "phase": Parameter("", "Readout pulse phase", "rad"),
        "amplitude_i": Parameter("", "Readout channel I amplitude", ""),
        "amplitude_q": Parameter("", "Readout channel Q amplitude", ""),
        "integration_length": Parameter("", "Integration length", "s"),
        "acquisition_delay": Parameter("", "Acquisition delay", "s"),
    }

    def __init__(
        self, parent: QuantumOp, name: str, locus: Locus, calibration_data: OILCalibrationData, builder: ScheduleBuilder
    ):
        super().__init__(parent, name, locus, calibration_data, builder)
        self._probe_line: ProbeChannelProperties = builder.channels[  # type: ignore[assignment]
            builder.component_channels[locus[0]]["readout"]
        ]
        self._duration = (
            self._probe_line.duration_to_int_samples(
                self._probe_line.round_duration_to_granularity(
                    calibration_data["acquisition_delay"] + calibration_data["integration_length"]
                )
            )
            + self._probe_line.integration_stop_dead_time
        )

        waveform_params = self.convert_calibration_data(
            calibration_data,
            {k: v for k, v in self.parameters.items() if k not in self.root_parameters},
            self._probe_line,
        )
        root_params = {k: v for k, v in calibration_data.items() if k in self.root_parameters and k != "duration"}
        probe_instruction, acquisitions = self._build_instructions(waveform_params, root_params)
        self._probe_instruction = probe_instruction
        self._acquisitions = acquisitions

    def _build_instructions(
        self, waveform_params: OILCalibrationData, root_params: OILCalibrationData
    ) -> tuple[IQPulse, tuple[AcquisitionMethod, AcquisitionMethod]]:
        """Builds a probe pulse and acquisition methods using the calibration data.

        Subclasses may override this method if needed.
        """
        if self.dependent_waves:
            wave_i = self.wave_i(**waveform_params)
            wave_q = self.wave_q(**waveform_params)
        else:
            wave_i = self.wave_i(**waveform_params["i"])
            wave_q = self.wave_q(**waveform_params["q"])

        probe_pulse = IQPulse(
            duration=waveform_params["n_samples"],
            wave_i=wave_i,
            wave_q=wave_q,
            scale_i=root_params["amplitude_i"],
            scale_q=root_params["amplitude_q"],
            phase=root_params["phase"],
            modulation_frequency=0,
        )

        integration_length = self._probe_line.duration_to_int_samples(root_params["integration_length"])
        acquisition_delay = round(self._probe_line.duration_to_samples(root_params["acquisition_delay"]))
        time_trace_label = "TO_BE_REPLACED"
        time_trace_acquisition = TimeTrace(
            label=time_trace_label,
            delay_samples=acquisition_delay,
            duration_samples=integration_length,
            implementation=f"{self.parent.name}.{self.name}",
        )

        # TODO: due to device limitations, we need to integrate always, even though it does not make much sense here
        const = Constant(integration_length)
        weights = IQPulse(
            duration=integration_length,
            wave_i=const,
            wave_q=const,
            scale_i=1.0,
            scale_q=0.0,
            phase=0.0,
            modulation_frequency=0,
        )
        integration_label = "dummy__integration"
        integration_acquisition = ComplexIntegration(
            label=integration_label,
            delay_samples=acquisition_delay,
            weights=weights,
            implementation=f"{self.parent.name}.{self.name}",
        )
        return probe_pulse, (integration_acquisition, time_trace_acquisition)

    def _call(self, key: str = "") -> TimeBox:
        """Returns a ``TimeBox`` containing the probe pulse measurement.

        In scheduling, the returned ``TimeBox`` blocks only the probe line (``self.locus[0]``).

        Args:
            key: The time trace results generated on this trigger will be used to assigned to
                ``f"{probe_line}__{key}"``, where ``probe_line`` is the one that handles ``self.locus[0]``. If empty,
                the key `"readout.time_trace"` is used.

        Returns:
            TimeBox containing the ReadoutTrigger instruction.

        """
        probe_channel = self.builder.component_channels[self.locus[0]]["readout"]
        multiplexed_iq = MultiplexedIQPulse(
            duration=self._probe_instruction.duration + self._probe_line.integration_start_dead_time,
            entries=((self._probe_instruction, self._probe_line.integration_start_dead_time),),
        )
        label_key = key or DEFAULT_TIME_TRACE_KEY
        acquisition_label = f"{self.locus[0]}__{label_key}"
        acquisitions = (self._acquisitions[0], replace(self._acquisitions[1], label=acquisition_label))
        readout_trigger = ReadoutTrigger(duration=self._duration, probe_pulse=multiplexed_iq, acquisitions=acquisitions)
        probe_timebox = MultiplexedProbeTimeBox.from_readout_trigger(
            readout_trigger=readout_trigger,
            probe_channel=probe_channel,
            locus_components=(self.locus[0],),
            label=f"{self.__class__.__name__} on {self.locus}",
        )
        final_box = TimeBox.composite([probe_timebox], label=probe_timebox.label)
        final_box.neighborhood_components[0] = {self.locus[0]}
        return final_box

    def duration_in_seconds(self) -> float:
        probe_timebox = self().children[0]  # type: ignore[union-attr]
        probe_channel = self.builder.component_channels[self.locus[0]]["readout"]
        return self.builder.channels[probe_channel].duration_to_seconds(probe_timebox.atom.duration)  # type: ignore[union-attr]

    @classmethod
    def get_locus_mapping_name(cls, operation_name: str, implementation_name: str) -> str:
        return PROBE_LINES_LOCUS_MAPPING


class ProbePulse_CustomWaveforms_noIntegration(CustomIQWaveforms):
    """Base class for implementing a probe line probe pulse with custom waveforms in the I and Q channels without
    any integration.

    Similar to the :class:`ProbePulse_CustomWaveforms` except that signal integration is removed.
    """

    root_parameters = {
        "frequency": Parameter("", "Readout pulse frequency", "Hz"),
        "duration": Parameter("", "Readout pulse duration", "s"),
        "phase": Parameter("", "Readout pulse phase", "rad"),
        "amplitude_i": Parameter("", "Readout channel I amplitude", ""),
        "amplitude_q": Parameter("", "Readout channel Q amplitude", ""),
    }

    def __init__(
        self, parent: QuantumOp, name: str, locus: Locus, calibration_data: OILCalibrationData, builder: ScheduleBuilder
    ):
        super().__init__(parent, name, locus, calibration_data, builder)

        self._multiplexed_timeboxes: dict[tuple[str, str, bool], MultiplexedProbeTimeBox] = {}
        """Cache for :meth:`probe_timebox`."""

        if len(locus) == 1:  # factorizable gates only need calibration on 1-loci
            self._probe_line: ProbeChannelProperties = builder.channels[  # type: ignore[assignment]
                builder.get_probe_channel(locus[0])
            ]
            c_freq = self._probe_line.center_frequency
            if_freq = (calibration_data["frequency"] - c_freq) / self._probe_line.sample_rate
            self._duration = (
                self._probe_line.duration_to_int_samples(
                    self._probe_line.round_duration_to_granularity(calibration_data["duration"])
                )
                + self._probe_line.instruction_duration_granularity
            )

            waveform_params = self.convert_calibration_data(
                calibration_data,
                {k: v for k, v in self.parameters.items() if k not in self.root_parameters or k == "duration"},
                self._probe_line,
            )
            root_params = {k: v for k, v in calibration_data.items() if k in self.root_parameters and k != "duration"}

            probe_instruction = self._build_instructions(waveform_params, root_params, if_freq)
            self._probe_instruction = probe_instruction
            self._prio_calibration: OILCalibrationData | None = None
        else:
            # we need to store the possible cal_data == priority calibration in order to propagate it to the factored
            # single-component measure calls in :meth:`probe_timebox`
            self._prio_calibration = calibration_data or None

    def _build_instructions(
        self, waveform_params: OILCalibrationData, root_params: OILCalibrationData, if_freq: float
    ) -> IQPulse:
        """Builds a probe pulse and an acquisition method using the calibration data.

        Subclasses may override this method if needed.
        """
        if self.dependent_waves:
            wave_i = self.wave_i(**waveform_params)
            wave_q = self.wave_q(**waveform_params)
        else:
            wave_i = self.wave_i(**waveform_params["i"])
            wave_q = self.wave_q(**waveform_params["q"])

        probe_pulse = IQPulse(
            duration=waveform_params["n_samples"],
            wave_i=wave_i,
            wave_q=wave_q,
            scale_i=root_params["amplitude_i"],
            scale_q=root_params["amplitude_q"],
            phase=root_params["phase"],
            modulation_frequency=if_freq,
        )

        return probe_pulse

    def probe_timebox(
        self, key: str = "", feedback_key: str = "", do_acquisition: bool = False
    ) -> MultiplexedProbeTimeBox:
        """Returns a "naked" probe timebox that supports convenient multiplexing through
        ``MultiplexedProbeTimeBox.__add__``.

        This method can be used if the user wants to control the multiplexing explicitly. With two
        ``MultiplexedProbeTimeBox``es ``A`` and ``B`` the result ``A + B`` has all the ``ReadoutTrigger`` instructions
        on each probe channel of ``A`` and ``B`` multiplexed together and played simultaneously.

        Args:
            key: The readout results generated on this trigger will be assigned to
                ``f"{qubit}__{key}"``, where ``qubit`` goes over the component names in ``self.locus``. If empty,
                the key `"readout.result"` will be used to maintain backwards compatibility.
            feedback_key: The signals generated by this measure operation are routed using this key for
                fast feedback purposes. See :meth:`__call__`.
            do_acquisition: if False, no acquisitions are added.

        Returns:
            MultiplexedProbeTimeBox containing the ReadoutTrigger instruction.

        """
        args = (key, feedback_key, do_acquisition)
        # additional caching for probe timeboxes due to the fact that both ._call and .time_trace use this
        if args not in self._multiplexed_timeboxes:
            if len(self.locus) == 1:
                multiplexed_iq = MultiplexedIQPulse(
                    duration=self._probe_instruction.duration,
                    entries=((self._probe_instruction, 0),),
                )
                readout_trigger = ReadoutTrigger(
                    probe_pulse=multiplexed_iq,
                    acquisitions=(),
                    duration=self._duration,
                )
                probe_channel = self.builder.get_probe_channel(self.locus[0])
                try:
                    drive_channel = self.builder.get_drive_channel(self.locus[0])
                except KeyError:
                    drive_channel = ""

                if drive_channel:
                    # drive channel must be blocked, to prevent DD insertion while measurement is taking place
                    # unfortunately we must allow for different channel sample rates because of UHFQA
                    channels = self.builder.channels
                    drive_channel_props = channels[drive_channel]
                    rt_duration_in_seconds = channels[probe_channel].duration_to_seconds(readout_trigger.duration)
                    block_duration = drive_channel_props.duration_to_int_samples(
                        drive_channel_props.round_duration_to_granularity(
                            rt_duration_in_seconds, round_up=True, force_min_duration=True
                        )
                    )
                    probe_timebox = MultiplexedProbeTimeBox.from_readout_trigger(
                        readout_trigger=readout_trigger,
                        probe_channel=probe_channel,
                        locus_components=self.locus,
                        label=f"{self.__class__.__name__} on {self.locus}",
                        block_channels=[drive_channel],
                        block_duration=block_duration,
                    )
                else:
                    probe_timebox = MultiplexedProbeTimeBox.from_readout_trigger(
                        readout_trigger=readout_trigger,
                        probe_channel=probe_channel,
                        locus_components=self.locus,
                        label=f"{self.__class__.__name__} on {self.locus}",
                    )
            else:
                probe_timeboxes = [
                    self.builder.get_implementation(  # type: ignore[attr-defined]
                        self.parent.name, (c,), impl_name=self.name, priority_calibration=self._prio_calibration
                    ).probe_timebox(key, feedback_key, do_acquisition)
                    for c in self.locus
                ]
                probe_timebox = functools.reduce(lambda x, y: x + y, probe_timeboxes)
            probe_timebox.neighborhood_components[0] = set(
                self.locus + (self.builder.chip_topology.component_to_probe_line[self.locus[0]],)
            )

            self._multiplexed_timeboxes[args] = probe_timebox
        return self._multiplexed_timeboxes[args]

    def _call(self, key: str = "", feedback_key: str = "") -> TimeBox:
        """Returns a TimeBox containing the multiplexed simultaneous measurement.

        If ``len(self.locus) == 1``, the TimeBox contains the measurement for just that component, otherwise
        the measurements of components that belong to the same probeline are multiplexed together.

        The returned :class:`.TimeBox` instances behave like any other TimeBox in scheduling and circuit
        generation. With measurement TimeBoxes ``A`` and ``B`` the result ``A + B`` first plays the ``ReadoutTrigger``
        instructions of ``A`` and only then those of ``B`` in each probe channel. If the multiplexing features of
        :class:`.MultiplexedProbeTimeBox` are needed, the method :meth:`probe_timebox` can be used.

        In scheduling, the returned TimeBox blocks the locus components and the probe
        lines they are associated with.

        Args:
            key: Readout results generated on this trigger will be assigned to the acquisition labels
                ``f"{qubit}__{key}"``, where ``qubit`` goes over the component names in ``self.locus``.
                If empty, the key ``"readout.result"`` will be used to maintain backwards compatibility.
            feedback_key: If the readout mode is "threshold", the results generated by this ``measure`` operation
                are routed using the label ``f"{qubit}__{feedback_key}"`` for fast feedback purposes.
                The signals are picked up by :class:`.ConditionalInstruction`s that have the same label.
                The default value ``""`` means the signal is not routed anywhere. TODO: currently the HW does not
                support multiple feedback keys per drive channel, so the actual key used will be ``FEEDBACK_KEY``
                whenever any non-empty key is inputted. When the HW is improved, the actual key the user inputs
                should be passed.

        Returns:
            TimeBox containing the :class:`.ReadoutTrigger` instruction.

        """
        final_box = TimeBox.composite(
            [self.probe_timebox(key=key, feedback_key=feedback_key)], label=f"Readout on {self.locus}"
        )
        final_box.neighborhood_components[0] = final_box.children[0].neighborhood_components[0]
        return final_box

    def duration_in_seconds(self) -> float:
        probe_timebox = self.probe_timebox()
        readout_schedule = probe_timebox.atom
        return readout_schedule.duration_in_seconds(self.builder.channels)  # type: ignore[union-attr]

    @classmethod
    def get_locus_mapping_name(cls, operation_name: str, implementation_name: str) -> str:
        return SINGLE_COMPONENTS_WITH_READOUT_LOCUS_MAPPING


class Probe_Constant(ProbePulse_CustomWaveforms_noIntegration, wave_i=Constant, wave_q=Constant):  # type: ignore[call-arg]
    """Implementation of a single-qubit projective, dispersive measurement in the Z basis.

    Uses a constant probe pulse.
    """


class ShelvedMeasureTimeBox(TimeBox):
    """TimeBox representing a shelved measurement (ReadoutTrigger sandwiched between two PRX_12 operations).

    ShelvedMeasureTimeBox is a composite TimeBox containing two children:
    * first one being the first PRX_12 operation for the locus components of the measure
    * second one being the ReadoutTrigger (MultiplexedProbeTimeBox) that includes the second PRX_12 operation.

    Multiplexing is achieved so that ShelvedMeasureTimeBoxes support ``__add__`` and ``__radd__`` operations with other
    boxes of the same type and MultiplexedProbeTimeBoxes. The multiplexing operation is defined such that the
    initial PRX_12 boxes are added together (in case one of the multiplexed boxes is a MultiplexedProbeTimeBoxes, the
    initial PRX_12 is considered empty), and the probe boxes are multiplexed together via the logic defined in
    ``MultiplexedProbeTimeBoxes.__add__``. This behaviour results in the correct timings of the associated pulses
    after the multiplexing.

    """

    def __post_init__(self):
        if len(self.children) != 2:
            raise ValueError(
                "ShelvedMeasureTimeBox must have exactly two children: the first one corresponding to the "
                "initial prx_12 operations, and the second one to the ReadoutTrigger and the final prx_12"
            )
        if self.children[1].atom is None or not isinstance(self.children[1], MultiplexedProbeTimeBox):
            raise ValueError("The second child must be an atomic MultiplexedProbeTimeBox.")

    @property
    def prx_12_box(self) -> TimeBox:
        return self.children[0]

    @property
    def trigger_box(self) -> TimeBox:
        return self.children[1]

    def __add__(self, other: TimeBox | Iterable[TimeBox]) -> TimeBox:
        """Add the initial PRX_12 boxes together via the ``TimeBox``"""
        if isinstance(other, (ShelvedMeasureTimeBox, MultiplexedProbeTimeBox)):
            if isinstance(other, ShelvedMeasureTimeBox):
                prx_12_box = self.prx_12_box + other.prx_12_box
                trigger_box = self.trigger_box + other.trigger_box
            else:
                prx_12_box = self.prx_12_box
                trigger_box = self.trigger_box + other
            locus_components = self.locus_components.union(other.locus_components)
            multiplexed = ShelvedMeasureTimeBox(
                label=f"Shelved measure on {locus_components}",
                locus_components=locus_components,
                atom=None,
                children=(prx_12_box, trigger_box),
                scheduling=self.scheduling,
                scheduling_algorithm=self.scheduling_algorithm,
            )
            # neighborhood components by the trigger_box
            multiplexed.neighborhood_components[0] = trigger_box.neighborhood_components[0]
            return multiplexed
        return super().__add__(other)

    def __radd__(self, other: TimeBox | Iterable[TimeBox]) -> TimeBox:
        if isinstance(other, MultiplexedProbeTimeBox):
            return self.__add__(other)  # this commutes
        return super().__radd__(other)


class Shelved_Measure_CustomWaveforms(Measure_CustomWaveforms, CompositeGate):
    """Base class for shelved readout.

    Shelved readout applies a ``prx_12(pi)`` gate before and after a standard dispersive readout on each qubit measured.
    The first ``prx_12(pi)`` swaps the amplitudes of the |1> and |2> states, and the second one swaps them back after
    the measurement has (roughly) collapsed the state. If the discriminator of the readout is calibrated such that
    the |0> state is on one side and the |1> and |2> states are on the other, the end result is equivalent to the
    standard readout operation but with the advantage that the population in the |2> state is less susceptible to
    :math:`T_1` decay during the readout than the population in the |1> state.

    """

    root_parameters = Measure_CustomWaveforms.root_parameters | {
        "second_prx_12_offset": Setting(
            Parameter(
                "second_prx_12_offset", "Offset of the second PRX_12 pulse from the end the ReadoutTrigger", unit="s"
            ),
            0.0,
        ),
        "do_prx_12": Setting(
            Parameter(
                "do_prx_12",
                "Whether to do the prx_12 flips in the measure operation",
                unit="",
                data_type=DataType.BOOLEAN,
            ),
            True,
        ),
    }
    registered_gates = ("prx_12",)

    def probe_timebox(self, key: str = "", feedback_key: str = "", do_acquisition: bool = True, **kwargs) -> TimeBox:  # type: ignore[union-attr, override]
        """Returns a "naked" probe timebox that supports convenient multiplexing through
        ``ShelvedMeasureTimeBox.__add__``.

        This method can be used if the user wants to control the multiplexing explicitly. Supports adding together
        boxes of type :class:`.ShelvedMeasureTimeBox` and/or :class:`.MultiplexedProbeTimeBox`. See
        :meth:`.ShelvedMeasureTimeBox.__add__` for more information on the logic.

        Args:
            key: The readout results generated on this trigger will be assigned to
                ``f"{qubit}__{key}"``, where ``qubit`` goes over the component names in ``self.locus``. If empty,
                the key `"readout.result"` will be used to maintain backwards compatibility.
            feedback_key: The signals generated by this measure operation are routed using this key for
                fast feedback purposes. See :meth:`__call__`.
            do_acquisition: if False, no acquisitions are added.

        Returns:
            ShelvedMeasureTimeBox containing the ReadoutTrigger instruction.

        """
        args = (key, feedback_key, do_acquisition)
        if args not in self._multiplexed_timeboxes:
            if len(self.locus) == 1:
                probe_timebox = super().probe_timebox(key, feedback_key, do_acquisition, **kwargs)
                shelved_box = probe_timebox
                prx_12_box = TimeBox.composite(
                    [self.build("prx_12", self.locus)(np.pi)], scheduling=SchedulingStrategy.ALAP
                )
                if self.calibration_data["do_prx_12"]:
                    shelved_box = probe_timebox + prx_12_box  # type: ignore[operator, assignment, override]
                # schedule the shelved box to get an atomic schedule
                shelved_atom = deepcopy(self.builder.resolve_timebox(shelved_box, neighborhood=0))
                offset = self.calibration_data["second_prx_12_offset"]
                if self.calibration_data["do_prx_12"] and abs(offset) > TIMING_TOLERANCE:
                    drive_channel_name = self.builder.get_drive_channel(self.locus[0])
                    drive_channel = self.builder.channels[drive_channel_name]
                    offset_sign = offset / abs(offset)
                    offset_in_samples = offset_sign * drive_channel.duration_to_int_samples(abs(offset))
                    trigger_block = shelved_atom[drive_channel_name][0]
                    block_with_offset = Block(trigger_block.duration + offset_in_samples)
                    shelved_atom[drive_channel_name]._instructions[0] = block_with_offset
                trigger_box = MultiplexedProbeTimeBox(
                    label=f"{self.__class__.__name__} on {self.locus}",
                    locus_components=probe_timebox.locus_components,
                    atom=shelved_atom,
                )
                trigger_box.neighborhood_components[0] = probe_timebox.neighborhood_components[0]
                pre_box = prx_12_box if self.calibration_data["do_prx_12"] else TimeBox.composite([])
                final_box = ShelvedMeasureTimeBox(
                    label=f"Shelved Measure on {self.locus}",
                    locus_components=set(self.locus),
                    atom=None,
                    children=(pre_box, trigger_box),
                )
                final_box.neighborhood_components[0] = probe_timebox.neighborhood_components[0]
            else:
                # NOTE: the super call can be a bit misleading; it is actually calling the `self.probe_timebox` of len 1
                # in this class inside, via the factorizable gate's sub_implementations
                final_box = super().probe_timebox(key, feedback_key)  # type: ignore[assignment]
            self._multiplexed_timeboxes[args] = final_box
        return self._multiplexed_timeboxes[args]

    def _get_probe_timebox_for_time_trace(self, key: str = "", feedback_key: str = "") -> TimeBox:
        """Utility method that can be overridden in subclasses if they have a return type `.probe_pulse`.

        The ``ShelvedMeasureTimeBox`` resulting from :meth:`.probe_timebox` is first scheduled to obtain an atomic
        ``MultiplexedProbeTimeBox`` which is wrapped into a TimeBox.
        """
        # FIXME: not needed once we align the return types of all these measure gates
        probe_timebox = self.probe_timebox(key=key, feedback_key=feedback_key)
        # resolve the box to get an atomic time_box.
        probe_schedule = self.builder.resolve_timebox(probe_timebox, neighborhood=0)
        atomic_probe_box = MultiplexedProbeTimeBox.atomic(
            probe_schedule,
            label=f"Time Trace atomic probe box of {self.__class__.__name__} on {self.locus}",
            locus_components=probe_timebox.locus_components,
        )
        atomic_probe_box.neighborhood_components[0] = probe_timebox.neighborhood_components[0]
        return atomic_probe_box


class Shelved_Measure_Constant(Shelved_Measure_CustomWaveforms, wave_i=Constant, wave_q=Constant):  # type:ignore[call-arg]
    """Implementation of a shelved readout.

    A measure gate implemented as a constant waveform is surrounded by two `prx_12` gates.
    """


class Fast_Measure_CustomWaveforms(Measure_CustomWaveforms):
    """Measure implementation that blocks locus qubits for a shorter duration than the probes.

    The locus qubits are blocked only for the physical probe pulse duration plus (calibratable) extra dead time that
    can be used to take into account e.g. ring down delay of waiting the readout resonator to empty itself. The probe
    channels are still blocked as in ``Measure_CustomWaveforms``, i.e. for the duration of
    ``acquisition_delay + integration_length + integration_dead_time``.
    """

    root_parameters = Measure_CustomWaveforms.root_parameters | {
        "locus_deadtime": Setting(
            Parameter("locus_deadtime", "Locus dead time after the probe pulse", unit="s"),
            0.0,
        ),
    }

    def probe_timebox(  # type: ignore[override]
        self, key: str = "", feedback_key: str = "", do_acquisition: bool = True, **kwargs
    ) -> ProbeTimeBoxes:
        """Otherwise the same as ``Measure_CustomWaveforms.probe_timebox``, but returns two TimeBoxes, the
        actual MultiplexedProbeTimeBox and the rest of the probe-blocking wait time in its own TimeBox. This
        allows the "tetris logic" in scheduling to block the locus qubits for a shorter duration.
        """
        args = (key, feedback_key, do_acquisition)
        if args not in self._multiplexed_timeboxes:
            if len(self.locus) == 1:
                probe = self.builder.chip_topology.component_to_probe_line[self.locus[0]]
                probe_channel = self.builder.get_probe_channel(self.locus[0])
                try:
                    drive_channel = self.builder.get_drive_channel(self.locus[0])
                except KeyError:
                    drive_channel = ""

                combined_probe_timebox = super().probe_timebox(
                    key, feedback_key, do_acquisition=do_acquisition, **kwargs
                )
                readout_trigger = combined_probe_timebox.atom[probe_channel][0]  # type: ignore[index]
                actual_probe_duration = readout_trigger.probe_pulse.duration

                # MultiplexedProbeTimeBox that has the minimum possible duration
                if self.calibration_data["locus_deadtime"] > TIMING_TOLERANCE:
                    deadtime = self.builder.channels[probe_channel].duration_to_int_samples(
                        self.calibration_data["locus_deadtime"]
                    )
                else:
                    deadtime = 0
                # Must be: ReadoutTrigger.duration > ReadoutTrigger.probe_pulse.duration so if they would match, we
                # need to add a minimum offset to make it hold, i.e. the smallest granularity allowed by the probe
                # channel
                probe_granularity = self.builder.channels[probe_channel].instruction_duration_granularity
                offset = max(deadtime, probe_granularity)

                truncated_readout_trigger = replace(readout_trigger, duration=actual_probe_duration + offset)
                physical_probe_box = MultiplexedProbeTimeBox.from_readout_trigger(
                    truncated_readout_trigger,
                    probe_channel,
                    locus_components=self.locus,
                    label=f"Physical probe box of {self.__class__.__name__} on {self.locus}",
                    block_channels=[drive_channel] if drive_channel else [],
                    block_duration=truncated_readout_trigger.duration if drive_channel else 0,
                )
                physical_probe_box.neighborhood_components[0] = combined_probe_timebox.neighborhood_components[0].copy()
                # extra Blocks (integration etc) for the probe channel
                extra_block_duration = max(combined_probe_timebox.atom.duration - truncated_readout_trigger.duration, 0)  # type:ignore[union-attr]
                virtual_extra_wait_box = TimeBox.atomic(
                    Schedule({probe_channel: Segment([Block(extra_block_duration)])}),
                    locus_components=[probe],
                    label=f"Virtual probe extra wait box of {self.__class__.__name__} on {self.locus}",
                )
                virtual_extra_wait_box.neighborhood_components[0] = {probe}
                final_boxes = ProbeTimeBoxes([physical_probe_box, virtual_extra_wait_box])
            else:
                # FIXME: all the return types in the measure impls need to be cleaned up
                final_boxes = super().probe_timebox(key, feedback_key, do_acquisition=do_acquisition, **kwargs)  # type:ignore[assignment]
            if feedback_key:
                probelines = {self.builder.chip_topology.component_to_probe_line[q] for q in self.locus}
                for probe in probelines:
                    final_boxes[1].neighborhood_components[0].update(
                        set(self.builder.get_virtual_feedback_channels(probe))
                    )
            self._multiplexed_timeboxes[args] = final_boxes  # type:ignore[assignment]
            return final_boxes  # type:ignore[return-value]
        return self._multiplexed_timeboxes[args]  # type:ignore[return-value]

    def _call(self, key: str = "", feedback_key: str = "") -> list[TimeBox]:  # type:ignore[override]
        """The same as ``Measure_CustomWaveforms._call``, i.e. wrap the "naked" multiplexable probe_timeboxes
        into a composite TimeBox.
        """
        probe_box, wait_box = self.probe_timebox(key, feedback_key, do_acquisition=True)
        final_box = TimeBox.composite([probe_box], label=f"Readout on {self.locus}")
        final_box.neighborhood_components[0] = final_box.children[0].neighborhood_components[0]
        return [final_box, wait_box]

    def _get_probe_timebox_for_time_trace(self, key: str = "", feedback_key: str = "") -> TimeBox:
        """Get the probe TimeBox for TimeTrace. This is just a single MultiplexedProbeTimeBox which has the full
        probe duration (including integration). The faster logic wrt. qubit blocking is not important in the context of
        TimeTraces.
        """
        # FIXME: not needed once we align the return types of all these measure gates
        # Then we can make this method also shorter for the locus qubits
        probe_boxes = self.probe_timebox(key, feedback_key, do_acquisition=True)
        total_box = TimeBox.composite(self.probe_timebox(key, feedback_key, do_acquisition=True))
        probe_schedule = self.builder.resolve_timebox(total_box, neighborhood=0)
        atomic_probe_box = MultiplexedProbeTimeBox.atomic(
            probe_schedule,
            label=f"Time Trace atomic probe box of {self.__class__.__name__} on {self.locus}",
            locus_components=probe_boxes[0].locus_components,
        )
        atomic_probe_box.neighborhood_components[0] = probe_boxes[0].neighborhood_components[0]
        return atomic_probe_box  # type:ignore[return-value]


class Fast_Measure_Constant(Fast_Measure_CustomWaveforms, wave_i=Constant, wave_q=Constant):  # type:ignore[call-arg]
    """Implementation of a faster measure with constant i and q waveforms.

    Does not block the drive and flux channels of the locus qubits during the integration, but just during the probe
    pulse and extra calibrated dead time after it.
    """
