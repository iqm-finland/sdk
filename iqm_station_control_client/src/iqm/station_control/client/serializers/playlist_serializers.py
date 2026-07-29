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
"""Serializers and deserializers for :class:`~iqm.models.playlist.playlist.Playlist`"""

import iqm.data_definitions.common.v1.playlist_pb2 as pb
from iqm.models.playlist import ChannelDescription, IQChannelConfig, RealChannelConfig, Segment, instructions, waveforms
from iqm.models.playlist.channel_descriptions import ReadoutChannelConfig
from iqm.models.playlist.playlist import Playlist
import numpy as np


def _iq_pulse_to_proto(
    iq_pulse: instructions.IQPulse, reverse_waveform_index: dict[waveforms.CanonicalWaveform, int]
) -> pb.IQPulse:
    return pb.IQPulse(
        waveform_i_ref=reverse_waveform_index[iq_pulse.wave_i],
        waveform_q_ref=reverse_waveform_index[iq_pulse.wave_q],
        scale_i=iq_pulse.scale_i,
        scale_q=iq_pulse.scale_q,
        phase=iq_pulse.phase,
        phase_mod_freq=iq_pulse.modulation_frequency,
        phase_increment=iq_pulse.phase_increment,
    )


def _proto_to_iq_pulse(iq_pulse: pb.IQPulse, wf_table: list[waveforms.CanonicalWaveform]) -> instructions.IQPulse:
    return instructions.IQPulse(
        wave_i=wf_table[iq_pulse.waveform_i_ref],
        wave_q=wf_table[iq_pulse.waveform_q_ref],
        scale_i=iq_pulse.scale_i,
        scale_q=iq_pulse.scale_q,
        phase=iq_pulse.phase,
        modulation_frequency=iq_pulse.phase_mod_freq,
        phase_increment=iq_pulse.phase_increment,
    )


def _instruction_to_proto(
    instruction: instructions.Instruction,
    reverse_instruction_index: dict[instructions.Instruction, int],
    reverse_waveform_index: dict[waveforms.CanonicalWaveform, int],
    reverse_acquisition_index: dict[instructions.AcquisitionMethod, int],
) -> pb.Instruction:
    inst = pb.Instruction()
    inst.duration_samples = instruction.duration_samples
    match instruction.operation:
        case instructions.VirtualRZ():
            inst.virtual_rz.phase_increment = instruction.operation.phase_increment

        case instructions.RealPulse():
            inst.real_pulse.waveform_ref = reverse_waveform_index[instruction.operation.wave]
            inst.real_pulse.scale = instruction.operation.scale

        case instructions.IQPulse():
            inst.iq_pulse.MergeFrom(_iq_pulse_to_proto(instruction.operation, reverse_waveform_index))

        case instructions.MultiplexedRealPulse():
            entries = (
                pb.MultiplexedRealPulse.Entry(offset_samples=offset, instruction_ref=reverse_instruction_index[pulse])
                for pulse, offset in instruction.operation.entries
            )
            inst.multiplexed_real_pulse.entries.extend(entries)

        case instructions.MultiplexedIQPulse():
            entries = (
                pb.MultiplexedIQPulse.Entry(offset_samples=offset, instruction_ref=reverse_instruction_index[pulse])
                for pulse, offset in instruction.operation.entries
            )
            inst.multiplexed_iq_pulse.entries.extend(entries)

        case instructions.ReadoutTrigger():
            inst.readout_trigger.probe_pulse_ref = reverse_instruction_index[instruction.operation.probe_pulse]
            acquisitions = (reverse_acquisition_index[acq] for acq in instruction.operation.acquisitions)
            inst.readout_trigger.acqusitions.extend(acquisitions)

        case instructions.ConditionalInstruction():
            inst.conditional_instruction.condition = instruction.operation.condition
            inst.conditional_instruction.if_true = reverse_instruction_index[instruction.operation.if_true]
            inst.conditional_instruction.if_false = reverse_instruction_index[instruction.operation.if_false]

        case instructions.Wait():
            inst.wait.SetInParent()

    return inst


def _waveform_to_proto(waveform: waveforms.CanonicalWaveform) -> pb.Waveform:
    ret_wf = pb.Waveform()
    ret_wf.n_samples = waveform.n_samples
    match waveform:
        case waveforms.Gaussian():
            ret_wf.gaussian.sigma = waveform.sigma
            ret_wf.gaussian.center_offset = waveform.center_offset

        case waveforms.GaussianDerivative():
            ret_wf.gaussian_derivative.sigma = waveform.sigma
            ret_wf.gaussian_derivative.center_offset = waveform.center_offset

        case waveforms.GaussianSmoothedSquare():
            ret_wf.gaussian_smoothed_square.square_width = waveform.square_width
            ret_wf.gaussian_smoothed_square.gaussian_sigma = waveform.gaussian_sigma
            ret_wf.gaussian_smoothed_square.center_offset = waveform.center_offset

        case waveforms.Samples():
            ret_wf.samples.samples.extend(waveform.samples)

        case waveforms.Constant():
            ret_wf.constant.SetInParent()

        case waveforms.TruncatedGaussian():
            ret_wf.truncated_gaussian.full_width = waveform.full_width
            ret_wf.truncated_gaussian.center_offset = waveform.center_offset

        case waveforms.TruncatedGaussianDerivative():
            ret_wf.truncated_gaussian_derivative.full_width = waveform.full_width
            ret_wf.truncated_gaussian_derivative.center_offset = waveform.center_offset
        case waveforms.TruncatedGaussianSmoothedSquare():
            ret_wf.truncated_gaussian_smoothed_square.full_width = waveform.full_width
            ret_wf.truncated_gaussian_smoothed_square.center_offset = waveform.center_offset
            ret_wf.truncated_gaussian_smoothed_square.rise_time = waveform.rise_time

        case waveforms.CosineRiseFall():
            ret_wf.cosine_rise_fall.full_width = waveform.full_width
            ret_wf.cosine_rise_fall.center_offset = waveform.center_offset
            ret_wf.cosine_rise_fall.rise_time = waveform.rise_time

    return ret_wf


def pack_playlist(playlist: Playlist) -> pb.Playlist:
    """Pack the given playlist into a protobuf format for further serialization.

    Args:
        playlist: playlist to pack

    Returns:
        ``playlist`` in protobuf format

    """
    proto_playlist = pb.Playlist()

    # segments
    for segment in playlist.segments:
        pb_schedule = pb.Schedule()
        for channel_name, instr_refs in segment.instructions.items():
            pb_schedule.channels[channel_name].instruction_refs.extend(instr_refs)
        proto_playlist.schedules.append(pb_schedule)

    # channel descriptions
    for channel_name, channel_description in playlist.channel_descriptions.items():
        proto_playlist.channels[channel_name].controller_name = channel_name

        match channel_description.channel_config:
            case IQChannelConfig(sampling_rate):  # type: ignore[misc]
                proto_playlist.channels[channel_name].channel_config.iq_channel.sample_rate = sampling_rate
            case RealChannelConfig(sampling_rate):  # type: ignore[misc]
                proto_playlist.channels[channel_name].channel_config.real_channel.sample_rate = sampling_rate
            case ReadoutChannelConfig(sampling_rate):  # type: ignore[misc]
                proto_playlist.channels[channel_name].channel_config.ro_channel.sample_rate = sampling_rate

        for instruction in channel_description.instruction_table:
            instr = _instruction_to_proto(
                instruction,
                channel_description._reverse_instruction_index,
                channel_description._reverse_waveform_index,
                channel_description._reverse_acquisition_index,
            )
            proto_playlist.channels[channel_name].instruction_table.append(instr)

        for waveform in channel_description.waveform_table:
            wf_pb = _waveform_to_proto(waveform)
            proto_playlist.channels[channel_name].waveform_table.append(wf_pb)

        for acquisition in channel_description.acquisition_table:
            acq_pb = _aqcuisition_method_to_proto(acquisition, channel_description._reverse_waveform_index)
            proto_playlist.channels[channel_name].acquisition_table.append(acq_pb)

    return proto_playlist


def _aqcuisition_method_to_proto(
    acquisition: instructions.AcquisitionMethod, reverse_waveform_index: dict[waveforms.CanonicalWaveform, int]
) -> pb.AcquisitionMethod:
    proto = pb.AcquisitionMethod(label=acquisition.label, delay_samples=acquisition.delay_samples)
    match acquisition:
        case instructions.TimeTrace(duration_samples=duration_samples):
            proto.timetrace.duration_samples = duration_samples  # type: ignore[has-type]
        case instructions.ThresholdStateDiscrimination(
            weights=weights, threshold=threshold, feedback_signal_label=feedback_signal_label
        ):
            proto.threshold_discrimination.weights.MergeFrom(
                _iq_pulse_to_proto(weights, reverse_waveform_index)  # type: ignore[has-type]
            )
            proto.threshold_discrimination.threshold = threshold  # type: ignore[has-type]
            proto.threshold_discrimination.feedback_signal_label = feedback_signal_label  # type: ignore[has-type]
        case instructions.ComplexIntegration(weights=weights):
            proto.integration.weights.MergeFrom(
                _iq_pulse_to_proto(weights, reverse_waveform_index)  # type: ignore[has-type]
            )
    return proto


def _proto_to_acqusition_method(
    acquisition: pb.AcquisitionMethod, waveform_table: list[waveforms.CanonicalWaveform]
) -> instructions.AcquisitionMethod:
    match acquisition.WhichOneof("acquisition_type"):
        case "timetrace":
            return instructions.TimeTrace(
                acquisition.label, acquisition.delay_samples, acquisition.timetrace.duration_samples
            )
        case "integration":
            return instructions.ComplexIntegration(
                label=acquisition.label,
                delay_samples=acquisition.delay_samples,
                weights=_proto_to_iq_pulse(acquisition.integration.weights, waveform_table),
            )
        case "threshold_discrimination":
            return instructions.ThresholdStateDiscrimination(
                label=acquisition.label,
                delay_samples=acquisition.delay_samples,
                weights=_proto_to_iq_pulse(acquisition.threshold_discrimination.weights, waveform_table),
                threshold=acquisition.threshold_discrimination.threshold,
                feedback_signal_label=acquisition.threshold_discrimination.feedback_signal_label,
            )


def _proto_to_waveform(waveform: pb.Waveform) -> waveforms.CanonicalWaveform:
    wf_type = waveform.WhichOneof("waveform_description")
    match wf_type:
        case "samples":
            return waveforms.Samples(samples=np.array(waveform.samples.samples))
        case "gaussian":
            return waveforms.Gaussian(
                n_samples=waveform.n_samples,
                sigma=waveform.gaussian.sigma,
                center_offset=waveform.gaussian.center_offset,
            )

        case "gaussian_derivative":
            return waveforms.GaussianDerivative(
                n_samples=waveform.n_samples,
                sigma=waveform.gaussian_derivative.sigma,
                center_offset=waveform.gaussian_derivative.center_offset,
            )

        case "constant":
            return waveforms.Constant(n_samples=waveform.n_samples)

        case "gaussian_smoothed_square":
            return waveforms.GaussianSmoothedSquare(
                n_samples=waveform.n_samples,
                square_width=waveform.gaussian_smoothed_square.square_width,
                gaussian_sigma=waveform.gaussian_smoothed_square.gaussian_sigma,
                center_offset=waveform.gaussian_smoothed_square.center_offset,
            )

        case "truncated_gaussian":
            return waveforms.TruncatedGaussian(
                n_samples=waveform.n_samples,
                full_width=waveform.truncated_gaussian.full_width,
                center_offset=waveform.truncated_gaussian.center_offset,
            )

        case "truncated_gaussian_derivative":
            return waveforms.TruncatedGaussianDerivative(
                n_samples=waveform.n_samples,
                full_width=waveform.truncated_gaussian_derivative.full_width,
                center_offset=waveform.truncated_gaussian_derivative.center_offset,
            )

        case "truncated_gaussian_smoothed_square":
            return waveforms.TruncatedGaussianSmoothedSquare(
                n_samples=waveform.n_samples,
                full_width=waveform.truncated_gaussian_smoothed_square.full_width,
                center_offset=waveform.truncated_gaussian_smoothed_square.center_offset,
                rise_time=waveform.truncated_gaussian_smoothed_square.rise_time,
            )

        case "cosine_rise_fall":
            return waveforms.CosineRiseFall(
                n_samples=waveform.n_samples,
                full_width=waveform.cosine_rise_fall.full_width,
                center_offset=waveform.cosine_rise_fall.center_offset,
                rise_time=waveform.cosine_rise_fall.rise_time,
            )


def _proto_to_instruction(instr: pb.Instruction, channel_desc: ChannelDescription) -> instructions.Instruction:
    instr_type = instr.WhichOneof("operation")
    match instr_type:
        case "iq_pulse":
            operation = _proto_to_iq_pulse(instr.iq_pulse, channel_desc.waveform_table)
        case "real_pulse":
            operation = instructions.RealPulse(
                wave=channel_desc.waveform_table[instr.real_pulse.waveform_ref],
                scale=instr.real_pulse.scale,
            )
        case "virtual_rz":
            operation = instructions.VirtualRZ(
                phase_increment=instr.virtual_rz.phase_increment,
            )
        case "conditional_instruction":
            operation = instructions.ConditionalInstruction(
                condition=instr.conditional_instruction.condition,
                if_true=channel_desc.instruction_table[instr.conditional_instruction.if_true],
                if_false=channel_desc.instruction_table[instr.conditional_instruction.if_false],
            )
        case "multiplexed_iq_pulse":
            operation = instructions.MultiplexedIQPulse(
                tuple(
                    (channel_desc.instruction_table[entry.instruction_ref], entry.offset_samples)
                    for entry in instr.multiplexed_iq_pulse.entries
                )
            )
        case "multiplexed_real_pulse":
            operation = instructions.MultiplexedRealPulse(
                tuple(
                    (channel_desc.instruction_table[entry.instruction_ref], entry.offset_samples)
                    for entry in instr.multiplexed_real_pulse.entries
                )
            )
        case "readout_trigger":
            operation = instructions.ReadoutTrigger(
                channel_desc.instruction_table[instr.readout_trigger.probe_pulse_ref],
                tuple(channel_desc.acquisition_table[idx] for idx in instr.readout_trigger.acqusitions),
            )
        case _:
            operation = instructions.Wait()

    return instructions.Instruction(instr.duration_samples, operation)


def unpack_playlist(proto_playlist: pb.Playlist) -> Playlist:
    """Unpack a protobuf representation of a playlist into its runtime representation.

    Args:
        proto_playlist: serialized playlist

    Returns:
        ``proto_playlist`` in runtime representation

    """
    playlist = Playlist()
    for channel in proto_playlist.channels.values():
        match channel.channel_config.WhichOneof("extended"):
            case "iq_channel":
                channel_config = IQChannelConfig(channel.channel_config.iq_channel.sample_rate)
            case "real_channel":
                channel_config = RealChannelConfig(channel.channel_config.real_channel.sample_rate)
            case "ro_channel":
                channel_config = ReadoutChannelConfig(channel.channel_config.ro_channel.sample_rate)
            case _:
                raise RuntimeError(f"Unknown channel type {channel.channel_config}.")
        channel_desc = ChannelDescription(channel_config, channel.controller_name)

        # The waveform and acquisition tables already contain unique entries in order,
        # so we don't need to do any lookups. The reverse lookup index can be filled in directly.
        channel_desc.waveform_table = [_proto_to_waveform(waveform) for waveform in channel.waveform_table]
        channel_desc.acquisition_table = [
            _proto_to_acqusition_method(acq, channel_desc.waveform_table) for acq in channel.acquisition_table
        ]
        channel_desc._reverse_waveform_index = {wf: idx for idx, wf in enumerate(channel_desc.waveform_table)}
        channel_desc._reverse_acquisition_index = {acq: idx for idx, acq in enumerate(channel_desc.acquisition_table)}

        # Instructions can contain references to other instructions, so we must use lookups to get the references
        # right.
        for instruction in channel.instruction_table:
            instr: instructions.Instruction = _proto_to_instruction(instruction, channel_desc)
            channel_desc._lookup_or_insert_instruction(instr)
        playlist.add_channel(channel_desc)

    for segment in proto_playlist.schedules:
        seg_schedule = Segment()
        for controller, instruction_list in segment.channels.items():
            instr_list = list(instruction_list.instruction_refs)
            seg_schedule.instructions[controller] = instr_list
        playlist.segments.append(seg_schedule)

    return playlist
