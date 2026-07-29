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

from collections.abc import Sequence
from copy import copy, deepcopy
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from exa.common.data.parameter import CollectionType, DataType, Parameter, Setting
from iqm.pulse.gate_implementation import (
    CompositeGate,
    CustomIQWaveforms,
    GateImplementation,
    Locus,
    OILCalibrationData,
)
from iqm.pulse.locus_mappings import (
    one_qubit_readout,
    one_qubit_readout_and_flux,
    probe_lines,
    single_qubit_or_resonator_without_probe,
)
from iqm.pulse.playlist import FluxPulse, Segment
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
from iqm.pulse.timebox import (
    MeasureBox,
    MeasureBoxes,
    MultiplexedProbeTimeBox,
    SchedulingStrategy,
    TimeBox,
)

if TYPE_CHECKING:  # pragma: no cover
    from iqm.pulse.builder import ScheduleBuilder
    from iqm.pulse.quantum_ops import QuantumOp

DEFAULT_INTEGRATION_KEY = "readout.result"
DEFAULT_TIME_TRACE_KEY = "readout.time_trace"
FEEDBACK_KEY = "feedback"
TIMING_TOLERANCE = 1e-12
BIAS_VOLTAGE_TOLERANCE = 1e-9


def multiplex(sequence_of_boxes: Sequence[MeasureBoxes]) -> MeasureBoxes:
    """Multiplex several :class:``MeasureBoxes`` instances together.

    Iterates through ``sequence_of_boxes`` and reduces them into a single ``MeasureBoxes`` instance via
    :meth:`MeasureBoxes.multiplex`. Implements an as soon as possible (ASAP) algorithm for creating the minimal
    length ``MeasureBoxes`` instance while adhering to multiplexing rules.

    Starting from the ``sequence_of_boxes``, denoted with ``[S0, S1, ..., Sm]``, we try to multiplex ``S0`` and ``S1``
    together, then add ``S2`` and so on. Continuing like this, the multiplexed result is denoted as ``M0``. If some
    member ``Sk`` of  ``sequence_of_boxes`` cannot be multiplexed into ``M0``, we keep it as is and denote it with
    ``M1``. We then try to multiplex the next member of ``sequence_of_boxes``, ``Sk+1``, with ``M0`` and if that is not
    possible, with ``M1``, and if that also fails, we store it as ``M2``. Iterating over ``sequence_of_boxes``, we
    thus obtain a new sequence of ``MeasureBoxes``, ``[M0, M1, ... Mn]``, where each member of the original sequence is
    executed ASAP. In the optimal case, this is just a single ``MeasureBoxes`` instance ``M0`` and in the worst case,
    the exact same sequence as in ``sequence_of_boxes`` is retained. The new sequence is then squeezed into a single
    ``MeasureBoxes`` instance by combining the last auxiliary box of ``Mk`` into the first auxiliary box of ``Mk+1``.

    Args:
         sequence_of_boxes: Sequence of :class:``MeasureBoxes`` instances to be multiplexed together.

    Returns:
        A single :class:``MeasureBoxes`` instance where each member of ``sequence_of_boxes`` is executed ASAP.

    """
    multiplexed_sequence: list[MeasureBoxes] = []
    for boxes in sequence_of_boxes:
        for idx, multiplexed in enumerate(multiplexed_sequence):
            if multiplexed.is_multiplexable_with(boxes):
                multiplexed_sequence[idx] = multiplexed.multiplex(boxes)
                break
        else:
            multiplexed_sequence.append(boxes)
    if len(multiplexed_sequence) == 1:
        # all the member boxes were successfully multiplexed together
        return multiplexed_sequence[0]
    final_measure_boxes: list[MeasureBoxes] = [multiplexed_sequence[0][0]]  # the first Aux box of M0
    measure_components = set(multiplexed_sequence[0].measure_components)
    additional_components = set(multiplexed_sequence[0].additional_components)
    for idx, multiplexed in enumerate(multiplexed_sequence[:-1]):
        # union of measure_components and additional_components
        next_one = multiplexed_sequence[idx + 1]
        measure_components.update(next_one.measure_components)
        additional_components.update(next_one.additional_components)
        # the last Aux box of M{idx} is combined with the first Aux box of M{idx+1} via MeasureBox.multiplex
        final_measure_boxes.extend(list(multiplexed[1:-1]) + [multiplexed[-1].multiplex(next_one[0])] + next_one[1:])
    return MeasureBoxes(final_measure_boxes, measure_components, additional_components)  # type: ignore[arg-type]


class Measure_Factorizable(GateImplementation):
    """Base class for implementing a factorizable measure.

    The ``measure`` operation is factorizable, and its :attr:`arity` is 0, which together mean that it can operate
    on loci of any length, but is calibrated only on single component loci. When the gate is constructed in the
    ``len(locus) > 1`` case (e.g. ``builder.get_implementation('measure', ('QB1', 'QB2', 'QB3'))()`` ), the resulting
    :class:`.MeasureBoxes` is constructed from the calibrated single-component implementations.

    """

    locus_mapping_function = one_qubit_readout

    def __init__(
        self, parent: QuantumOp, name: str, locus: Locus, calibration_data: OILCalibrationData, builder: ScheduleBuilder
    ):
        super().__init__(parent, name, locus, calibration_data, builder)
        self._neighborhood_components: set[str] = set(self.locus)
        # add the probes attached to self.locus into the neighborhood components (if any)
        self._neighborhood_components.update(
            {self.builder.chip_topology.component_to_probe_line.get(q, q) for q in self.locus}
        )
        self._time_traces: dict[tuple[str, float | None, float | None, str], MeasureBoxes] = {}
        """Cache for :meth:`time_trace`."""

    def _call(self, *args, **kwargs) -> MeasureBoxes:
        """Multiplex single component call results (of the type MeasureBoxes) together."""
        # NOTE: self.sub_implementations[c] is typically an instance of any _inherting class_, not this base class
        # itself. The multiplexing code must be able to multiplex locus-specific (1QB locus) default implementations
        # together
        measure_boxes = [
            self.sub_implementations[c](*args, **kwargs)  # type: ignore[attr-defined, union-attr]
            for c in self.locus
        ]
        return multiplex(measure_boxes)  # type: ignore[arg-type]

    def time_trace(
        self,
        key: str = "",
        acquisition_delay: float | None = None,
        acquisition_duration: float | None = None,
        feedback_key: str = "",
    ) -> MeasureBoxes:
        """Returns a multiplexed simultaneous measurement with an additional time trace acquisition.

        The returned ``MeasureBoxes`` are the same as the one returned by :meth:`__call__` except the time trace
        acquisition is appended to the acquisitions of each probe line's ``ReadoutTrigger`` instruction.

        Args:
            key: Readout results generated on this trigger will be used to assigned to
                ``f"{qubit}__{key}"``, where ``qubit`` goes over the component names in ``self.locus``, whereas
                the recorded time traces will be assigned to ``f"{probe_line}__{key}"`` where
                ``probe_line`` goes through all the probe lines associated with ``self.locus``.
                If empty, the key ``"readout.result"`` will be used for integrated results and the key
                ``"readout.time_trace"`` for the recorded time traces.
            acquisition_delay: optionally override the time trace acquisition delay with this value (given in
                seconds). Does not affect the acquisition delays of the integrated measurements.
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
            measure_boxes = deepcopy(self(key, feedback_key))
            for probe_timebox in measure_boxes:  # type: ignore[union-attr]
                if isinstance(probe_timebox, MultiplexedProbeTimeBox):
                    for probe_channel, segment in probe_timebox.atom.items():  # type: ignore[union-attr]
                        for inst in segment:
                            if isinstance(inst, ReadoutTrigger):
                                readout_trigger = inst
                                break
                        else:
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
                        trigger_with_trace = replace(
                            readout_trigger, acquisitions=readout_trigger.acquisitions + (time_trace,)
                        )
                        segment._instructions[0] = trigger_with_trace

                self._time_traces[args] = measure_boxes  # type: ignore[assignment]
        return self._time_traces[args]

    def duration_in_seconds(self) -> float:
        measure_timebox = TimeBox.composite(self())
        readout_schedule = self.builder.resolve_timebox(measure_timebox, neighborhood=0)
        return readout_schedule.duration_in_seconds(self.builder.channels)  # type: ignore[union-attr]


class Measure_CustomWaveforms(CustomIQWaveforms, Measure_Factorizable):
    """Base class for implementing dispersive measurement operations with custom probe pulse waveforms.

    You may define a measurement implementation that uses the :class:`.Waveform`
    instances ``Something`` and ``SomethingElse`` as the probe pulse waveforms in the
    I and Q channels as follows:
    ``class MyGate(Measure_CustomWaveforms, i_wave=Something, q_wave=SomethingElse)``.

    The ``measure`` operation is factorizable, and its :attr:`arity` is 0, which together mean that it can operate
    on loci of any length, but is calibrated only on single component loci.

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

        if len(locus) != 1:
            raise NotImplementedError(
                "Measure_CustomWaveforms cannot be constructed for multi-component loci. "
                "Instead the factorizability feature should be utilized."
            )

        # prepare the single-component measurement
        probe_line: ProbeChannelProperties = builder.channels[  # type: ignore[assignment]
            builder.get_probe_channel(locus[0])
        ]
        # readout duration is determined by the acquisition, probe pulses are truncated to fit this window
        self._probe = probe_line
        self._probe_offset = probe_line.integration_start_dead_time
        # "duration" is only used by the probe pulse
        waveform_params = self.convert_calibration_data(
            calibration_data,
            {k: v for k, v in self.parameters.items() if k not in self.root_parameters},
            probe_line,
        )
        # unconverted cal data that corresponds to a root param (not duration)
        root_params = {k: v for k, v in calibration_data.items() if k in self.root_parameters and k != "duration"}

        self._probe_instruction, self._acquisitions = self._build_instructions(waveform_params, root_params)

    def _build_instructions(
        self, waveform_params: OILCalibrationData, root_params: OILCalibrationData
    ) -> tuple[IQPulse, tuple[AcquisitionMethod, ...]]:
        """Builds a probe pulse and acquisition methods using the calibration data.

        Subclasses may override this method if needed.
        """
        if_freq = (self.calibration_data["frequency"] - self._probe.center_frequency) / self._probe.sample_rate
        # do some conversions TODO are these consistent?
        root_params["integration_length"] = self._probe.duration_to_int_samples(root_params["integration_length"])
        root_params["acquisition_delay"] = round(self._probe.duration_to_samples(root_params["acquisition_delay"]))

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

        return probe_pulse, (acquisition_method,)

    def _get_readout_trigger_durations(self) -> tuple[int, int]:
        """Durations of the ReadoutTrigger instruction and the extra probe block box after it.

        In the base class, the ReadoutTrigger duration contains also the acquisition delay and integration, and
        consequently there is no extra probe block.

        Inheriting classes can override this method if they employ the standard ReadoutTrigger but need to adjust
        its duration or the extra probe block duration.
        """
        trigger_duration = (
            self._probe.duration_to_int_samples(
                self._probe.round_duration_to_granularity(
                    self.calibration_data["acquisition_delay"] + self.calibration_data["integration_length"]
                )
            )
            + self._probe.integration_stop_dead_time
        )
        return trigger_duration, 0

    def _build_readout_trigger(
        self,
        key: str = "",
        feedback_key: str = "",
        do_acquisition: bool = True,
        acquisitions: tuple[AcquisitionMethod, ...] | None = None,
        readout_trigger_duration: int | None = None,
    ) -> ReadoutTrigger:
        """Build a ReadoutTrigger for the measurement.

        Applies the call args ``(key, feedback_key, do_acquisition)`` and creates the :class:`ReadoutTrigger`
        instruction that contains the logic for performing the measurement.

        Inheriting classes can override this method if they cannot employ the standard ReadoutTrigger.

        Args:
            key: The readout results generated on this trigger will be assigned to
                ``f"{qubit}__{key}"``, where ``qubit`` goes over the component names in ``self.locus``. If empty,
                the key `"readout.result"` will be used to maintain backwards compatibility.
            feedback_key: The signals generated by this measure operation are routed using this key for
                fast feedback purposes. See :meth:`__call__`.
            do_acquisition: if False, no acquisitions are added.
            acquisitions: Optionally, override the acquisitions with these acquisition methods.
            readout_trigger_duration: Optionally, override the readout trigger duration with this number of samples.

        Returns:
            The readout trigger instruction.

        """
        label_key = key or DEFAULT_INTEGRATION_KEY
        replacements = {"label": f"{self.locus[0]}__{label_key}"}
        if feedback_key and isinstance(self._acquisitions[0], ThresholdStateDiscrimination):
            # TODO: use the actual ``feedback_key`` when AWGs support multiple feedback labels
            replacements["feedback_signal_label"] = f"{self.locus[0]}__{FEEDBACK_KEY}"
        if acquisitions is None:
            if self._acquisitions:
                acquisitions = (replace(self._acquisitions[0], **replacements),) if do_acquisition else ()  # type: ignore[arg-type]
            else:
                acquisitions = ()
        multiplexed_iq = MultiplexedIQPulse(
            duration=self._probe_instruction.duration + self._probe_offset,
            entries=((self._probe_instruction, self._probe_offset),),
        )
        duration = (
            readout_trigger_duration
            if readout_trigger_duration is not None
            else self._get_readout_trigger_durations()[0]
        )
        return ReadoutTrigger(duration=duration, probe_pulse=multiplexed_iq, acquisitions=acquisitions)

    def _call(self, key: str = "", feedback_key: str = "", do_acquisition: bool = True, **kwargs) -> MeasureBoxes:
        """Build and return a single-qubit measure timeboxes.

        The base class method creates a :class:`MeasureBoxes` that contains a MultiplexedProbeTimeBox, with
        the auxiliary boxes being empty. Inheriting classes can override this method e.g. if they need to add
        some auxiliary boxes, more ReadoutTriggers or some other fundamentally different logic from the base class
        version. If an inheriting class just needs to adjust the ReadoutTrigger/probe extra block durations, it
        is enough to override :meth:`_get_readout_trigger_durations`.
        """
        readout_trigger_duration, probe_extra_wait_duration = self._get_readout_trigger_durations()
        readout_trigger = self._build_readout_trigger(
            key, feedback_key, do_acquisition, readout_trigger_duration=readout_trigger_duration
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
            segments = {drive_channel: Segment([Block(block_duration)])}
        else:
            segments = None
        probe_name = self.builder.chip_topology.component_to_probe_line[self.locus[0]]
        if feedback_key:
            ff_neighborhood_components = set(self.builder.get_virtual_feedback_channels(probe_name))
        else:
            ff_neighborhood_components = set()
        boxes = MeasureBoxes.from_readout_trigger(
            readout_trigger=readout_trigger,
            probe_channel=probe_channel,
            measure_components=self.locus,
            probe_box_segments=segments,
            extra_probe_block_duration=probe_extra_wait_duration,
        )
        boxes[1].neighborhood_components[0] = copy(self._neighborhood_components).union(ff_neighborhood_components)
        boxes[2].neighborhood_components[0] = ff_neighborhood_components.union({probe_name})
        return boxes


class Measure_Constant(Measure_CustomWaveforms, wave_i=Constant, wave_q=Constant):
    """Implementation of a single-qubit projective, dispersive measurement in the Z basis.

    Uses a constant probe pulse.
    """


class Measure_Constant_Qnd(Measure_CustomWaveforms, wave_i=Constant, wave_q=Constant):
    """Implementation of a single-qubit projective, non quantum demolition, dispersive measurements in the Z basis.

    Uses a constant probe pulse.
    """


class ProbePulse_CustomWaveforms(Measure_CustomWaveforms):
    """Base class for implementing a probe line measurement pulse with custom waveforms in the I and Q channels.

    With given :class:`.Waveform` waveform definitions ``Something`` and ``SomethingElse``,
    you may define a measurement implementation that uses them as follows:
    ``class MyGate(ProbePulse_CustomWaveforms, i_wave=Something, q_wave=SomethingElse)``.
    The measurement :class:`.IQPulse` instruction will not be automatically modulated
    by any frequency, so any modulations should be included in the I and Q waveforms themselves.

    Due to device limitations this implementation also has to integrate the readout signal
    (using arbitrary weights), even though it does not make much sense.

    Contrary to the ``Measure_CustomWaveforms`` class, this implementation acts on probe lines directly (i.e. its
    ``locus`` is a single probe line) as it is intended to be used in contexts where well-defined qubits may not have
    been characterized yet (such as readout frequency calibration with a chirp pulse).
    """

    root_parameters = {
        "duration": Parameter("", "Readout pulse duration", "s"),
        "phase": Parameter("", "Readout pulse phase", "rad"),
        "amplitude_i": Parameter("", "Readout channel I amplitude", ""),
        "amplitude_q": Parameter("", "Readout channel Q amplitude", ""),
        "integration_length": Parameter("", "Integration length", "s"),
        "acquisition_delay": Parameter("", "Acquisition delay", "s"),
    }
    locus_mapping_function = probe_lines

    def _build_instructions(
        self, waveform_params: OILCalibrationData, root_params: OILCalibrationData, **kwargs
    ) -> tuple[IQPulse, tuple[AcquisitionMethod, ...]]:
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

        integration_length = self._probe.duration_to_int_samples(root_params["integration_length"])
        acquisition_delay = round(self._probe.duration_to_samples(root_params["acquisition_delay"]))
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

    def _call(self, key: str = "", **kwargs) -> MeasureBoxes:  # type: ignore[override]
        """Returns ``MeasureBoxes`` containing the probe pulse measurement.

        In scheduling, the returned TimeBoxes block only the probe line (``self.locus[0]``).

        Args:
            key: The time trace results generated on this trigger will be used to assigned to
                ``f"{probe_line}__{key}"``, where ``probe_line`` is the one that handles ``self.locus[0]``. If empty,
                the key `"readout.time_trace"` is used.
            kwargs: Ignored.

        Returns:
            TimeBox containing the ReadoutTrigger instruction.

        """
        label_key = key or DEFAULT_TIME_TRACE_KEY
        acquisition_label = f"{self.locus[0]}__{label_key}"
        rt_duration, probe_block_duration = self._get_readout_trigger_durations()
        acquisitions = (self._acquisitions[0], replace(self._acquisitions[1], label=acquisition_label))
        readout_trigger = self._build_readout_trigger(
            key, acquisitions=acquisitions, readout_trigger_duration=rt_duration
        )

        measure_boxes = MeasureBoxes.from_readout_trigger(
            readout_trigger=readout_trigger,
            probe_channel=f"{self.locus[0]}__readout",
            measure_components={self.locus[0]},
            extra_probe_block_duration=probe_block_duration,
        )
        return measure_boxes

    def _multiplex(self, key: str = "", feedback_key: str = "", do_acquisition: bool = True) -> MeasureBoxes:
        raise NotImplementedError(
            "Multiplexing is not supported as ProbePulse is not a factorizable gate implementation."
        )

    def time_trace(self, key: str = "", **kwargs) -> TimeBox:  # type: ignore[override]
        raise NotImplementedError(
            "ProbePulse does not implement time_trace, as the __call__ does the time trace measurement."
        )


class NoAcquisition_Measure_CustomWaveforms(Measure_CustomWaveforms):
    """Base class for ``measure`` with custom probe IQ waveforms without any integration or acquisition.

    Similar to the :class:`Measure_CustomWaveforms` except that signal acquisition is removed. The implementation is
    still factorizable, the locus contains qubits, and multi-component readout is performed via multiplexing.
    """

    root_parameters = {
        "frequency": Parameter("", "Readout pulse frequency", "Hz"),
        "duration": Parameter("", "Readout pulse duration", "s"),
        "phase": Parameter("", "Readout pulse phase", "rad"),
        "amplitude_i": Parameter("", "Readout channel I amplitude", ""),
        "amplitude_q": Parameter("", "Readout channel Q amplitude", ""),
    }

    def _build_instructions(
        self, waveform_params: OILCalibrationData, root_params: OILCalibrationData
    ) -> tuple[IQPulse, tuple[AcquisitionMethod, ...]]:
        """Just the probe instruction, without any acquisitions."""
        if_freq = (self.calibration_data["frequency"] - self._probe.center_frequency) / self._probe.sample_rate
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

        return probe_pulse, tuple()

    def _build_readout_trigger(
        self,
        key: str = "",
        feedback_key: str = "",
        do_acquisition: bool = True,
        acquisitions: tuple[AcquisitionMethod, ...] | None = None,
        readout_trigger_duration: int | None = None,
    ) -> ReadoutTrigger:
        multiplexed_iq = MultiplexedIQPulse(
            duration=self._probe_instruction.duration,
            entries=((self._probe_instruction, 0),),
        )
        return ReadoutTrigger(
            probe_pulse=multiplexed_iq,
            acquisitions=(),
            duration=self._get_readout_trigger_durations()[0],
        )

    def _get_readout_trigger_durations(self) -> tuple[int, int]:
        readout_trigger_duration = (
            self._probe.duration_to_int_samples(
                self._probe.round_duration_to_granularity(self.calibration_data["duration"])
            )
            + self._probe.instruction_duration_min
        )
        return readout_trigger_duration, 0


class NoAcquisition_Measure_Constant(NoAcquisition_Measure_CustomWaveforms, wave_i=Constant, wave_q=Constant):  # type: ignore[call-arg]
    """Implementation of a single-qubit projective, dispersive measurement in the Z basis.

    Uses a constant probe pulse.
    """


class Shelved_Measure_CustomWaveforms(Measure_CustomWaveforms, CompositeGate):
    r"""Base class for shelved readout.

    Shelved readout applies a ``prx_12(pi)`` gate before and after a standard dispersive readout on
    each qubit measured.  The first ``prx_12(pi)`` swaps the amplitudes of the :math:`|1\rangle` and
    :math:`|2\rangle` states, and the second one swaps them back after the measurement has (roughly)
    collapsed the state. If the discriminator of the readout is calibrated such that the
    :math:`|0\rangle` state is on one side and the :math:`|1\rangle` and :math:`|2\rangle` states
    are on the other, the end result is equivalent to the standard readout operation but with the
    advantage that the population in the :math:`|2\rangle` state is less susceptible to :math:`T_1`
    decay during the readout than the population in the :math:`|1\rangle` state.

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

    def _call(self, key: str = "", feedback_key: str = "", do_acquisition: bool = True, **kwargs) -> MeasureBoxes:
        """Return a MeasureBoxes instance implementing a single-component shelved measurement.

        The first ``prx_12`` is set as the first auxiliary box, while the second one is baked into the
        ``MultiplexedProbeTimeBox`` containing the ``ReadoutTrigger`` instruction.

        Args:
            key: The readout results generated on this trigger will be assigned to
                ``f"{qubit}__{key}"``, where ``qubit`` goes over the component names in ``self.locus``. If empty,
                the key `"readout.result"` will be used to maintain backwards compatibility.
            feedback_key: The signals generated by this measure operation are routed using this key for
                fast feedback purposes. See :meth:`__call__`.
            do_acquisition: if False, no acquisitions are added.
            kwargs: Ignored.

        Returns:
            MeasureBoxes instance implementing a single-component shelved measurement.

        """
        readout_trigger = self._build_readout_trigger(key, feedback_key, do_acquisition)
        measure_boxes = MultiplexedProbeTimeBox.from_readout_trigger(
            readout_trigger,
            self.builder.get_probe_channel(self.locus[0]),
            self.locus,
            label=f"MultiplexedProbeTimeBox of {self.__class__.__name__}",
            block_channels=(self.builder.get_drive_channel(self.locus[0]),),
            block_duration=readout_trigger.duration,
        )
        prx_12_box = MeasureBox.composite([self.build("prx_12", self.locus)(np.pi)], scheduling=SchedulingStrategy.ALAP)
        probe_name = self.builder.chip_topology.component_to_probe_line[self.locus[0]]
        prx_12_box.neighborhood_components[0] = {probe_name, self.locus[0]}
        shelved_box = measure_boxes
        if self.calibration_data["do_prx_12"]:
            shelved_box = measure_boxes + prx_12_box  # type: ignore[operator, assignment, override]

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
            locus_components=measure_boxes.locus_components,
            atom=shelved_atom,
        )
        trigger_box.neighborhood_components[0] = self._neighborhood_components
        if feedback_key:
            trigger_box.neighborhood_components[0].update(set(self.builder.get_virtual_feedback_channels(probe_name)))

        pre_box = prx_12_box if self.calibration_data["do_prx_12"] else MeasureBox.composite([])
        return MeasureBoxes(
            [pre_box, trigger_box, MeasureBox.composite([]), MeasureBox.composite([])],
            measure_components=set(self.locus),
        )


class Shelved_Measure_Constant(Shelved_Measure_CustomWaveforms, wave_i=Constant, wave_q=Constant):
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

    def _get_readout_trigger_durations(self) -> tuple[int, int]:
        """Get readout trigger and extra probe block durations for the fast measure.

        The readout trigger duration is the physical probe pulse duration plus a calibratable locus deadtime.
        The extra probe block duration then accounts for the acquisition delay and the integration, containing
        the remainder
        `<acquisition_delay> + <integration_length> + <integration_stop_dead_time> - <readout trigger duration>`.` .
        """
        if self.calibration_data["locus_deadtime"] > TIMING_TOLERANCE:
            deadtime = self._probe.duration_to_int_samples(self.calibration_data["locus_deadtime"])
        else:
            deadtime = 0
        # Must be: ReadoutTrigger.duration > ReadoutTrigger.probe_pulse.duration so if they would match, we
        # need to add a minimum offset to make it hold, i.e. the smallest granularity allowed by the probe
        # channel
        probe_granularity = self._probe.instruction_duration_granularity
        offset = max(deadtime, probe_granularity)
        readout_trigger_duration = self._probe_instruction.duration + self._probe_offset + offset

        # then the extra block duration
        full_duration = super()._get_readout_trigger_durations()[0]
        probe_extra_block_duration = max(full_duration - readout_trigger_duration, 0)
        return readout_trigger_duration, probe_extra_block_duration


class Fast_Measure_Constant(Fast_Measure_CustomWaveforms, wave_i=Constant, wave_q=Constant):
    """Implementation of a faster measure with constant i and q waveforms.

    Does not block the drive and flux channels of the locus qubits during the integration, but just during the probe
    pulse and extra calibrated dead time after it.
    """


class Move_Measure_Composite(CompositeGate, Measure_Factorizable):
    """Measure implementation intended for measuring components that have no probe line.

    If a component has no attached probe line, it can be measured by moving its state to a connected component
    that does have a probe line. This requires that the ``move`` gate is defined between the (non-probeable) locus
    component and the move qubit that has a probe line. The move qubit's name is provided as a calibration data
    parameter to this implementation.

    """

    parameters = {
        "phase": Parameter("", "Readout pulse phase", "rad"),
        "integration_threshold": Parameter("", "Integration threshold", ""),
        "acquisition_type": Setting(Parameter("", "Acquisition type", "", data_type=DataType.STRING), "threshold"),
        "measure_qubit": Parameter(
            "measure_qubit", "Qubit used for the probe pulse", unit="", data_type=DataType.STRING
        ),
    }
    registered_gates = ("measure", "measure_fidelity")
    customizable_gates = tuple()
    # TODO: the customizable gates cannot currently have loci outside the parent gate, but this gate
    # would need move registered to the pair between self.locus[0] and the measure_qubit and measure & measure_fidelity
    # for the measure_qubit

    locus_mapping_function = single_qubit_or_resonator_without_probe

    def _call(
        self,
        key: str = "",
        feedback_key: str = "",
        do_acquisition: bool = True,
    ) -> MeasureBoxes:
        """Single component MeasureBoxes for the move-measure composite gate.

        The first auxiliary box is the ``move`` operation between ``self.locus[0]`` and the `"measure_qubit"`
        defined in the calibration data. In case of mid-circuit-measurements, we must also move the state back
        to ``self.locus[0]`` after the measurement is done.

        Args:
            key: The measurement key.
            feedback_key: The feedback key.
            do_acquisition: Whether to record the measurement data.

        Returns:
            The single-component measure boxes.

        """
        label_key = key or DEFAULT_INTEGRATION_KEY
        measure_qubit = self.calibration_data["measure_qubit"]
        move = self.builder.move((measure_qubit, self.locus[0]))()
        measure_op = self.parent.name
        # override the discrimination & post-processing-related settings in the measure_qubit's calibration
        # TODO: these should eventually be taken from the normal CompositeGate recursive node, but we cannot
        # do that yet due to the post-processing logic breaking, so we use priority calibration for now.
        prio_calibration = {
            "phase": self.calibration_data["phase"],
            "integration_threshold": self.calibration_data["integration_threshold"],
            "acquisition_type": self.calibration_data["acquisition_type"],
        }
        measure_boxes: list[MeasureBox] = list(
            self.build(
                measure_op,
                (measure_qubit,),
                priority_calibration=prio_calibration,
            )(key, feedback_key, do_acquisition)  # type: ignore[arg-type]
        )
        # replace the readout label with that of the actual measured component, "<self.locus[0]>__<key>"
        trigger_box = measure_boxes[1]
        measure_qubit_pl = self.builder.get_probe_channel(measure_qubit)
        readout_trigger, inst_idx = next(
            (inst, idx)
            for idx, inst in enumerate(trigger_box.atom[measure_qubit_pl]._instructions)  # type: ignore[index]
            if isinstance(inst, ReadoutTrigger)
        )
        trigger_box.atom[measure_qubit_pl]._instructions[inst_idx] = replace(  # type: ignore[index]
            readout_trigger,
            acquisitions=(
                replace(
                    readout_trigger.acquisitions[0],
                    label=f"{self.locus[0]}__{label_key}",
                    implementation=f"{self.parent.name}.{self.name}",
                ),
            ),
        )

        pre_box = MeasureBox.composite([move + measure_boxes[0]])
        nbhood = pre_box.neighborhood_components.get(0, set()) | {
            self.locus[0],
            measure_qubit,
            self.builder.chip_topology.component_to_probe_line[measure_qubit],
        }
        pre_box.neighborhood_components[0] = nbhood
        measure_boxes[0] = pre_box
        if self.parent.name == "measure":
            # if mid-circuit-measurement, we must move the measured state back to the locus
            post_box = MeasureBox.composite([measure_boxes[3] + move])
            post_box.neighborhood_components[0] = nbhood
            measure_boxes[3] = post_box
        return MeasureBoxes(measure_boxes, measure_components={self.locus[0], measure_qubit})


class QBBias_Measure_CustomWaveforms(Measure_CustomWaveforms):
    """Base class of a measure with an additional fast flux voltage pulse to bias the measured qubit."""

    locus_mapping_function = one_qubit_readout_and_flux

    root_parameters = Measure_CustomWaveforms.root_parameters | {
        "flux_bias_voltage_qubit": Setting(
            Parameter("Bias voltage applied to qubit", "", unit="V"),
            0.0,
        ),
        "delay": Setting(
            Parameter(
                "Delay",
                "Time between start of readout and bias voltage ramp up",
                unit="s",
            ),
            0.0,
        ),
        "block_neighbors": Setting(
            Parameter(
                "Block neighbors",
                "Whether to block neighboring qubits in multiplexing",
                data_type=DataType.BOOLEAN,
            ),
            True,
        ),
    }

    def _call(
        self,
        key: str = "",
        feedback_key: str = "",
        do_acquisition: bool = True,
        **kwargs,
    ) -> MeasureBoxes:
        """Return a MeasureBoxes instance implementing a measurement with a parallel square voltage bias pulse.

        Blocks the neighboring qubits in the multiplexing in order to prevent unwanted resonances due to the flux
        pulse changing the qubit frequencies.

        Args:
            key: The readout results generated on this trigger will be assigned to
                ``f"{qubit}__{key}"``, where ``qubit`` goes over the component names in ``self.locus``. If empty,
                the key `"readout.result"` will be used to maintain backwards compatibility.
            feedback_key: The signals generated by this measure operation are routed using this key for
                fast feedback purposes. See :meth:`__call__`.
            do_acquisition: if False, no acquisitions are added.
            **kwargs: Additional keyword arguments, for API compatibility. They are ignored here.

        Returns:
            MeasureBoxes instance implementing a single-component biased measurement.

        """
        readout_trigger = self._build_readout_trigger(key, feedback_key, do_acquisition)
        measure_boxes = MultiplexedProbeTimeBox.from_readout_trigger(
            readout_trigger,
            self.builder.get_probe_channel(self.locus[0]),
            self.locus,
            label=f"MultiplexedProbeTimeBox of {self.__class__.__name__}",
            block_channels=(self.builder.get_drive_channel(self.locus[0]),),
            block_duration=readout_trigger.duration,
        )
        biased_atom = deepcopy(measure_boxes.atom)

        ampl = self.calibration_data["flux_bias_voltage_qubit"]
        delay = self.calibration_data["delay"]

        # Construct flux pulse from scratch, just like in FluxPulse
        flux_channel_name = self.builder.get_flux_channel(self.locus[0])
        flux_channel = self.builder.channels[flux_channel_name]
        n_samples_flux = flux_channel.duration_to_int_samples(self.calibration_data["duration"])

        probe_name = self.builder.chip_topology.component_to_probe_line[self.locus[0]]

        flux_pulse = FluxPulse(
            duration=n_samples_flux,
            wave=Constant(n_samples=n_samples_flux),
            scale=ampl,
        )

        if abs(ampl) > BIAS_VOLTAGE_TOLERANCE:
            biased_atom.add_channels([flux_channel_name])  # type: ignore[union-attr]
            if abs(delay) > TIMING_TOLERANCE:
                wait_box = self.builder.wait(self.locus, delay, rounding=True)
                wait_pulse = wait_box.atom[flux_channel_name][0]  # type: ignore[index]
                biased_atom[flux_channel_name] = Segment([wait_pulse, flux_pulse])  # type: ignore[index]
            else:
                biased_atom[flux_channel_name] = Segment([flux_pulse])  # type: ignore[index]

        final_box = MultiplexedProbeTimeBox(
            label=f"{self.__class__.__name__} on {self.locus}",
            locus_components=measure_boxes.locus_components,
            atom=biased_atom,
        )
        final_box.neighborhood_components[0] = self._neighborhood_components
        if feedback_key:
            final_box.neighborhood_components[0].update(set(self.builder.get_virtual_feedback_channels(probe_name)))
        if self.calibration_data["block_neighbors"]:
            nb_qubits = self.builder.chip_topology.get_neighbor_locus_components(self.locus)
        else:
            nb_qubits = set()
        return MeasureBoxes(
            [MeasureBox.composite([]), final_box, MeasureBox.composite([]), MeasureBox.composite([])],
            measure_components=set(self.locus),
            additional_components=nb_qubits,  # block neighboring qubits in multiplexing
        )

    def get_additional_components(self) -> set[str]:
        return self.builder.chip_topology.get_neighbor_locus_components(self.locus)


class QBBias_Measure_Constant(QBBias_Measure_CustomWaveforms, wave_i=Constant, wave_q=Constant):  # type:ignore[call-arg]
    """Implementation of a measure with an additional fast flux voltage pulse to bias the measured qubit.

    Uses a probe pulse with constant i and q waveforms, and a square flux bias pulse.

    """


class QBBias_Probe_Constant(
    QBBias_Measure_CustomWaveforms, NoAcquisition_Measure_CustomWaveforms, wave_i=Constant, wave_q=Constant
):  # type: ignore[call-arg]
    """Implementation of a measure like QBBias_Measure_Constant, but without obtaining data from it.

    Useful for experiments, where a mid-circuit probe is used before the end-circuit readout.

    """

    def get_additional_components(self) -> set[str]:
        return set()
