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

from copy import deepcopy
from dataclasses import replace
import functools
from typing import TYPE_CHECKING

import numpy as np

from exa.common.data.parameter import CollectionType, DataType, Parameter, Setting
from iqm.pulse.gate_implementation import (
    PROBE_LINES_LOCUS_MAPPING,
    SINGLE_COMPONENTS_WITH_READOUT_LOCUS_MAPPING,
    CustomIQWaveforms,
    Locus,
    OILCalibrationData,
)
from iqm.pulse.playlist.channel import ProbeChannelProperties
from iqm.pulse.playlist.instructions import (
    AcquisitionMethod,
    ComplexIntegration,
    IQPulse,
    MultiplexedIQPulse,
    ReadoutTrigger,
    ThresholdStateDiscrimination,
    TimeTrace,
)
from iqm.pulse.playlist.waveforms import Constant, Samples
from iqm.pulse.timebox import MultiplexedProbeTimeBox, TimeBox

if TYPE_CHECKING:  # pragma: no cover
    from iqm.pulse.builder import ScheduleBuilder
    from iqm.pulse.quantum_ops import QuantumOp

DEFAULT_INTEGRATION_KEY = "readout.result"
DEFAULT_TIME_TRACE_KEY = "readout.time_trace"
FEEDBACK_KEY = "feedback"


class Measure_CustomWaveforms(CustomIQWaveforms):
    """Base class for implementing dispersive measurement operations with custom probe pulse waveforms.

    You may define a measurement implementation that uses the :class:`.Waveform`
    instances ``Something`` and ``SomethingElse`` as the probe pulse waveforms in the
    I and Q channels as follows:
    ``class MyGate(Measure_CustomWaveforms, i_wave=Something, q_wave=SomethingElse)``.

    The ``measure`` operation is factorizable, and its :attr:`arity` is 0, which together mean that it can operate
    on loci of any dimensionality, but is calibrated only on single component loci. When the gate is constructed in the
    ``len(locus) > 1``, case (e.g. ``builder.get_implementation('measure', ('QB1', 'QB2', 'QB3'))()``) the resulting
    :class:`.TimeBox` is constructed from the calibrated single-component gates.

    For each measured component, the readout ``IQPulse`` will be modulated with the
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

        self._multiplexed_timeboxes: dict[tuple[str, str, bool], MultiplexedProbeTimeBox] = {}
        """Cache for :meth:`probe_timebox`."""
        self._time_traces: dict[tuple[str, float | None, float | None, str], TimeBox] = {}
        """Cache for :meth:`time_trace`."""

        if len(locus) == 1:  # factorizable gates only need calibration on 1-loci
            self._probe_line: ProbeChannelProperties = self.builder.channels[  # type: ignore[assignment]
                self.builder.get_probe_channel(self.locus[0])
            ]
            c_freq = self._probe_line.center_frequency
            if_freq = (calibration_data["frequency"] - c_freq) / self._probe_line.sample_rate
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
                {k: v for k, v in self.parameters.items() if k not in self.root_parameters or k == "duration"},
                self._probe_line,
            )
            root_params = {k: v for k, v in calibration_data.items() if k in self.root_parameters and k != "duration"}

            probe_instruction, acquisition_method = self._build_instructions(waveform_params, root_params, if_freq)
            self._probe_instruction = probe_instruction
            self._acquisition_method = acquisition_method
            self._prio_calibration: OILCalibrationData | None = None
        else:
            # we need to store the possible cal_data == priority calibration in order to propagate it to the factored
            # single-component measure calls in :meth:`probe_timebox`
            self._prio_calibration = calibration_data or None

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

        integration_length = self._probe_line.duration_to_int_samples(root_params["integration_length"])
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
        acquisition_delay = round(self._probe_line.duration_to_samples(root_params["acquisition_delay"]))
        acquisition_label = "TO_BE_REPLACED"
        if acquisition_type == "complex":
            acquisition_method = ComplexIntegration(
                label=acquisition_label, delay_samples=acquisition_delay, weights=weights
            )
        elif acquisition_type == "threshold":
            acquisition_method = ThresholdStateDiscrimination(
                label=acquisition_label,
                delay_samples=acquisition_delay,
                weights=weights,
                threshold=root_params["integration_threshold"],
            )
        else:
            raise ValueError(f"Unknown acquisition type {acquisition_type}")

        return probe_pulse, acquisition_method

    def probe_timebox(
        self, key: str = "", feedback_key: str = "", do_acquisition: bool = True, **kwargs
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
                label_key = key or DEFAULT_INTEGRATION_KEY
                replacements = {"label": f"{self.locus[0]}__{label_key}"}
                if feedback_key and isinstance(self._acquisition_method, ThresholdStateDiscrimination):
                    # TODO: use the actual ``feedback_key`` when AWGs support multiple feedback labels
                    feedback_bit = f"{self.locus[0]}__{FEEDBACK_KEY}"
                    replacements["feedback_signal_label"] = feedback_bit
                acquisitions = (replace(self._acquisition_method, **replacements),) if do_acquisition else ()  # type: ignore[arg-type]
                multiplexed_iq = MultiplexedIQPulse(
                    duration=self._probe_instruction.duration + self._probe_line.integration_start_dead_time,
                    entries=((self._probe_instruction, self._probe_line.integration_start_dead_time),),
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
                # _skip_override used for child classes build on `Measure_CustomWaveforms` to not call `.probe_timebox`
                # from the parent class, but from this class instead.
                probe_timeboxes = [
                    self.builder.get_implementation(  # type: ignore[attr-defined]
                        self.parent.name, (c,), impl_name=self.name, priority_calibration=self._prio_calibration
                    ).probe_timebox(key, feedback_key, do_acquisition, _skip_override=True)
                    for c in self.locus
                ]
                probe_timebox = functools.reduce(lambda x, y: x + y, probe_timeboxes)
            probe_timebox.neighborhood_components[0] = set(
                self.locus + (self.builder.chip_topology.component_to_probe_line[self.locus[0]],)
            )
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
            probe_timebox = deepcopy(self.probe_timebox(key=key, feedback_key=feedback_key, _skip_override=True))
            for probe_channel, segment in probe_timebox.atom.items():  # type: ignore[union-attr]
                readout_trigger = segment[0]
                # TODO instead of editing the probe_timebox output contents, we should make the function itself do this
                # so we would not need to blindly search through the channels
                if not isinstance(readout_trigger, ReadoutTrigger):
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

    Contrary to the ``Measure_CustomWaveforms`` class, this implementation acts on proble lines directly (i.e. its
    ``locus`` is a single probe line). The measurement ``IQPulse`` instruction will not be automatically modulated
    by any frequency, so any modulations should be included in the I and Q waveforms themselves.
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
        self._probe_line: ProbeChannelProperties = self.builder.channels[  # type: ignore[assignment]
            self.builder.component_channels[self.locus[0]]["readout"]
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
            {k: v for k, v in self.parameters.items() if k not in self.root_parameters or k == "duration"},
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
            label=time_trace_label, delay_samples=acquisition_delay, duration_samples=integration_length
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
            label=integration_label, delay_samples=acquisition_delay, weights=weights
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
            self._probe_line: ProbeChannelProperties = self.builder.channels[  # type: ignore[assignment]
                self.builder.get_probe_channel(self.locus[0])
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


class Shelved_Measure_CustomWaveforms(Measure_CustomWaveforms):
    """Base class for shelved readout.

    Shelved readout applies a ``prx_12(pi)`` gate before and after a standard dispersive readout on each qubit measured.
    The first ``prx_12(pi)`` swaps the amplitudes of the |1> and |2> states, and the second one swaps them back after
    the measurement has (roughtly) collapsed the state. If the discriminator of the readout is calibrated such that
    the |0> state is on one side and the |1> and |2> states are on the other, the end result is equivalent to the
    standard readout operation but with the advantage that the population in the |2> state is less susceptible to
    :math:`T_1` decay during the readout than the population in the |1> state.

    .. note:: Mixed implementation multiplexing is not supported.
    """

    # Copied from `CompositeGate` to refresh caching after any calibration changes (in particular for the `prx_12`
    # calibration)
    def __call__(self, *args, **kwargs):
        default_cache_key = tuple(args) + tuple(kwargs.items())
        try:
            hash(default_cache_key)
            key_is_hashable = True
        except TypeError:
            key_is_hashable = False
        if key_is_hashable:
            if box := self.builder.composite_cache.get(self, default_cache_key):
                return box
        box = self._call(*args, **kwargs)
        if key_is_hashable:
            self.builder.composite_cache.set(self, default_cache_key, box)
        return box

    # `probe_timebox` is needed for making certain experiments work (e.g. `MeasurementQNDness`), since they call this
    # function explicitly. However, the main functionality of this method will not work: Enabling mixed
    # implementation multiplexing. This is because the method has to return time boxes due to the `prx_12` pulses,
    # instead of `MultiplexedProbeTimeBox`
    # TODO: Enable mixed implementation multiplexing for shelved readout
    def probe_timebox(  # type: ignore[override]
        self, key: str = "", feedback_key: str = "", do_acquisition: bool = True, _skip_override: bool = False
    ) -> TimeBox:
        if _skip_override:
            return super().probe_timebox(key, feedback_key, do_acquisition)
        multiplexed_timeboxes = super().probe_timebox(key, feedback_key)
        prx_12_impl = [self.builder.get_implementation("prx_12", [q])(np.pi) for q in self.locus]

        boxes = prx_12_impl + multiplexed_timeboxes + prx_12_impl  # type: ignore[operator]
        return boxes

    def _call(self, key: str = "", feedback_key: str = "") -> TimeBox:  # type: ignore[override]
        shelved_measure_box = TimeBox.composite(
            self.probe_timebox(key=key, feedback_key=feedback_key),  # type: ignore[arg-type]
            label=f"Readout on {self.locus}",
        )
        shelved_measure_box.neighborhood_components[0] = shelved_measure_box.children[
            len(self.locus)
        ].neighborhood_components[0]

        return shelved_measure_box


class Shelved_Measure_Constant(Shelved_Measure_CustomWaveforms, wave_i=Constant, wave_q=Constant):  # type:ignore[call-arg]
    """Implementation of a shelved readout.

    A measure gate implemented as a constant waveform is surrounded by two `prx_12` gates.
    """
