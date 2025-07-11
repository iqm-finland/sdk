#  ********************************************************************************
#  Copyright (c) 2019-2024 IQM Finland Oy.
#  All rights reserved. Confidential and proprietary.
#
#  Distribution or reproduction of any information contained herein
#  is prohibited without IQM Finland Oy’s prior written permission.
#  ********************************************************************************
"""GateImplementation for correcting flux crosstalk for a given set of flux-pulse TimeBoxes"""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy

from iqm.models.playlist.waveforms import Samples
import numpy as np

from exa.common.data.parameter import CollectionType, DataType, Parameter
from exa.common.qcm_data.chip_topology import ChipTopology
from iqm.pulse.gate_implementation import GateImplementation
from iqm.pulse.playlist.instructions import RealPulse, Wait
from iqm.pulse.playlist.schedule import Segment
from iqm.pulse.timebox import TimeBox

TOLERANCE = 1e-10
"""Tolerance for the maximum absolute value in a waveform for considering it to be uniformly zero."""


def _assert_flux_pulse_equality(pulse1: RealPulse, pulse2: RealPulse, assert_waveform: bool = True) -> None:
    error_msg = "FluxMultiplexer requires all multiplexed flux pulses to have equal durations"
    if assert_waveform:
        error_msg += " and waveforms (the amplitude can differ)"
    error = ValueError(error_msg)
    if pulse1.duration != pulse2.duration:
        raise error
    if assert_waveform and pulse1.wave != pulse2.wave:
        raise error


class FluxMultiplexer_SampleLinear(GateImplementation):
    # pylint: disable=anomalous-backslash-in-string, too-many-branches
    r"""Linear flux pulse multiplexed (user for correcting flux crosstalk).

    The required calibration data is the flux cross-talk correction matrix, where the element :math:`C_ij` represents
    the correction needed for flux component ``i`` arising from the crosstalk caused by flux component ``j``, so that
    after the corrections, the flux pulse played at ``i`` is :math:`f(t) = A_i w_i(t) + \sum_j C_ij A_j w_j(t)`,
    where :math:`A_j`` is the flux pulse amplitude for ``j`` and :math:`w_j(t)` the (normalized) waveform.

    The flux crosstalk correction matrix is given in a sparse form via two calibration parameters (we do not support
    dict- or xarray-valued Parameters yet...). Parameter ``matrix_index`` lists the relevant (non-zero) elements of the
    matrix as a flat ``np.array`` of strings of the form ``<flux component i>__<flux component j>``. Parameter
    ``matrix_elements`` lists the corresponding matrix values :math:`C_ij` (the lengths of these arrays must match).

    TODO: this is for now an experimental R&D implementation, and everything here is subject to change still
    """

    parameters = {
        "matrix_index": Parameter(
            "matrix_index",
            label="Flux crosstalk correction sparse matrix index",
            data_type=DataType.STRING,
            collection_type=CollectionType.LIST,
        ),
        "matrix_elements": Parameter(
            "matrix_elements",
            label="Flux crosstalk correction sparse matrix elements",
            data_type=DataType.FLOAT,
            collection_type=CollectionType.NDARRAY,
        ),
    }

    def __call__(self, to_be_multiplexed: Iterable[TimeBox] | TimeBox) -> TimeBox:  # noqa: PLR0912
        """TimeBox where flux-crosstalk errors have been corrected using the crosstalk matrix for a set of
        flux-pulse TimeBoxes.

        The following limitations apply:

        - All the TimeBoxes must be either pure flux-pulses (real pulse on a flux channel) or TimeBoxes created by a
        CZ or MOVE gate ``__call__`` method (real pulse on a flux channel, VirtualRZs on drive channels).
        - All the flux pulses must have the same duration.
        - Any VirtualRZ angles will not be adjusted after doing the flux pulse corrections, this must for now be done
          outside this gate implementation.
        - The flux crosstalk is now corrected only for flux-capable components. There are ideas how to potentially
        correct them for "slow qubits" as well, but this is not implemented yet.

        Args:
            to_be_multiplexed: TimeBoxes to be corrected for flux-crosstalk.

        Returns:
            Equivalent crosstalk-corrected flux pulses.

        """
        # TODO: reimplement the matrix stuff with e.g. scipy sparse matrices
        # TODO: assert that `to_be_multiplexed` contains only allowed TimeBoxes
        matrix_index = self.calibration_data["matrix_index"]
        matrix_elements = self.calibration_data["matrix_elements"]
        # get all crosstalk flux channels
        crosstalk_components = sorted({c for m_ind in matrix_index for c in m_ind.split("__")})
        crosstalk_channels = [self.builder.get_flux_channel(c) for c in crosstalk_components]
        no_flux = [channel for channel, component in zip(crosstalk_channels, crosstalk_components) if channel is None]
        if no_flux:
            raise ValueError(f"Flux crosstalk matrix components {no_flux} have no flux channels.")
        # schedule the fluxes
        if isinstance(to_be_multiplexed, Iterable):
            to_be_multiplexed = TimeBox.composite(to_be_multiplexed)
        scheduled_fluxes = self.builder.resolve_timebox(to_be_multiplexed, neighborhood=0)
        # find the RealPulse template
        real_pulse_candidate = next(p for _, seg in scheduled_fluxes.items() for p in seg if isinstance(p, RealPulse))
        real_pulse_dict = {
            "duration": real_pulse_candidate.duration,
            "wave": real_pulse_candidate.wave,
        }
        # Fix the waits in the flux channels into "duration"-length tiles.
        # We can do this because a) to_be_multiplexed contains only pure fLux pulses and CZ/MOVE TimeBoxes
        # and b) the durations of all flux pulses must be equal.
        for channel in crosstalk_channels:
            if channel in scheduled_fluxes:
                seg = scheduled_fluxes[channel]
                chopped_seg = []
                for inst in seg:
                    if isinstance(inst, Wait):
                        multiple = int(inst.duration / real_pulse_dict["duration"])
                        chopped_seg.extend(multiple * [Wait(real_pulse_dict["duration"])])
                    else:
                        chopped_seg.append(inst)  # type: ignore[arg-type]
                scheduled_fluxes[channel] = Segment(chopped_seg)
        max_flux_depth = next(len(seg) for ch, seg in scheduled_fluxes.items() if ch in crosstalk_channels)
        # pad the relevant flux channels not present in the schedule at all with correct number of wait tiles
        for channel in crosstalk_channels:
            if channel not in scheduled_fluxes:
                scheduled_fluxes[channel] = Segment(max_flux_depth * [Wait(real_pulse_dict["duration"])])
        # go through the schedule, layer by layer, and multiplex the flux pulses in each
        primary_components = [p.split("__")[0] for p in matrix_index]
        corrected_fluxes = deepcopy(scheduled_fluxes)
        for layer in range(max_flux_depth):
            for component in primary_components:
                flux_channel = self.builder.get_flux_channel(component)
                crosstalk_links = [
                    (i, p.split("__")[1]) for i, p in enumerate(matrix_index) if component == p.split("__")[0]
                ]
                primary_pulse = scheduled_fluxes[flux_channel][layer]
                if isinstance(primary_pulse, RealPulse):
                    _assert_flux_pulse_equality(primary_pulse, real_pulse_candidate, assert_waveform=False)
                multiplexed_samples = (
                    primary_pulse.scale * primary_pulse.wave.sample()
                    if isinstance(primary_pulse, RealPulse)
                    else np.zeros(primary_pulse.duration)
                )
                for matrix_idx, corrected_component in crosstalk_links:
                    corrected_channel = self.builder.get_flux_channel(corrected_component)
                    corrected_pulse = scheduled_fluxes[corrected_channel][layer]
                    if isinstance(corrected_pulse, RealPulse):
                        _assert_flux_pulse_equality(corrected_pulse, real_pulse_candidate, assert_waveform=False)
                        multiplexed_samples += (
                            matrix_elements[matrix_idx] * corrected_pulse.scale * corrected_pulse.wave.sample()
                        )
                if max(abs(multiplexed_samples)) < TOLERANCE:
                    multiplexed_pulse = Wait(real_pulse_dict["duration"])
                else:
                    multiplexed_pulse = RealPulse(  # type: ignore[assignment]
                        scale=1.0, duration=real_pulse_dict["duration"], wave=Samples(multiplexed_samples)
                    )
                corrected_fluxes[flux_channel]._instructions[layer] = multiplexed_pulse
        locus_components = {c for c in primary_components if c in self.builder.chip_topology.qubits}
        return TimeBox.atomic(
            corrected_fluxes,
            locus_components=locus_components,
            label=f"Cross-talk corrected flux pulses on {locus_components}",
        )

    @classmethod
    def get_custom_locus_mapping(
        cls, chip_topology: ChipTopology, component_to_channels: dict[str, Iterable[str]]
    ) -> dict[tuple[str, ...] | frozenset[str], tuple[str, ...]] | None:
        """Locus is "global" (the whole QPU) represented by an empty tuple for now."""
        # pylint: disable=unused-argument
        return {tuple(): tuple()}
