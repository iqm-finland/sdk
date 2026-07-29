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
"""Compiler stages and passes for layering quantum circuits and applying transformations on the layers."""

from collections import defaultdict
from dataclasses import dataclass, field
import uuid

import numpy as np

from iqm.cpc.compiler.compilation_stage import CompilationStage
from iqm.cpc.core.config import ComponentGrouping
from iqm.cpc.interface.circuit_execution import Circuit
from iqm.pulse import CircuitOperation
from iqm.pulse.builder import ScheduleBuilder
from iqm.pulse.playlist import ConditionalInstruction, FluxPulse, Instruction, IQPulse, ReadoutTrigger, Schedule, Wait
from iqm.pulse.timebox import TimeBox

_FLUX_OP_NAMES = ("cz", "move", "flux_pulse")  # TODO: create attr GateImplementation.has_flux and use that instead


@dataclass
class CircuitLayer:
    """Represents CircuitOperations in a Circuit that can be executed consequently (or in parallel).

    Used to group together CircuitOperations that share a specific property.
    """

    layer_type: str
    """Type of the layer, e.g. 'flux' or 'non-flux', shared between its member CircuitOperations."""
    locus_components: set[str]
    """Union of the loci of the CircuitOperations in this layer."""
    instructions: list[CircuitOperation] = field(default_factory=list)
    """Instruction (CircuitOperations) in this layer, ordered as in :class:´.CircuitOperation``. These should correspond
    to a transformation "gate" registered in :class:`.ScheduleBuilder`."""
    transformations: list[str] = field(default_factory=list)
    """Transformation names to be applied for this layer when resolving it."""

    def __post_init__(self):
        self.id = uuid.uuid4()

    def resolve(self, builder: ScheduleBuilder) -> TimeBox:
        """Apply ``self.transformations`` to this layer and resolve it into a TimeBox.

        Args:
            builder: The schedule builder.

        Returns:
            The layer's contents resolved into a TimeBox.
        .

        """
        payload = builder.circuit_to_timebox(self.instructions)
        for transformation in self.transformations:
            payload = builder.get_implementation(transformation, ())(payload)  # type: ignore[assignment]
        return payload


def layer_circuits(
    circuits: list[Circuit],
    components: ComponentGrouping,
    skip_layering: bool = False,
) -> list[list[CircuitLayer]]:
    """Layer circuits into flux and non-flux layers.

    The algorithm maximises the flux-layer depth, by merging any :class:`.CircuitOperation` involving flux pulses
    into the previous flux layer if there's no operations in between that with colliding loci. The layers
    containing flux pulses will have `CircuitLayer.layer_type == "flux"`, and the ones that do not contain flux pulses
    will have `CircuitLayer.layer_type == "non-flux"`.

    .. note::

       If a flux CircuitOperation resolves into :class:`.Schedule` that contains also drive pulses (like ACStark CZ
       gate), this pass will not produce correct flux layering.

    Args:
        circuits: The circuits to layer.
        components: The full set of active locus components for the circuit batch.
        skip_layering: Iff True, group the whole circuit into a single layer with
            `CircuitLayer.layer_type == "full_circuit"`.

    Returns:
        For each circuit, its layering into flux and non-flux layers.

    """
    if skip_layering:
        return [
            [
                CircuitLayer(
                    layer_type="full_circuit",
                    locus_components=set(components.flatten()),
                    instructions=list(c.instructions),
                )
            ]
            for c in circuits
        ]

    def layer_circuit(circuit: Circuit) -> list[CircuitLayer]:
        """Layer a single circuit."""
        layers: list[CircuitLayer] = []
        for inst in circuit.instructions:
            locus = set(inst.locus)
            if inst.name in _FLUX_OP_NAMES:
                for layer_idx, layer in enumerate(reversed(layers)):
                    if layer.layer_type == "flux":
                        layer.locus_components = locus.union(layer.locus_components)
                        layer.instructions.append(inst)
                        break
                    elif layer.locus_components.intersection(locus):
                        layers.insert(
                            len(layers) - layer_idx,
                            CircuitLayer(
                                layer_type="flux",
                                locus_components=set(inst.locus),
                                instructions=[inst],
                            ),
                        )
                        break
                else:
                    layers.insert(
                        0,
                        CircuitLayer(
                            layer_type="flux",
                            locus_components=set(inst.locus),
                            instructions=[inst],
                        ),
                    )
            elif len(layers) > 0 and layers[-1].layer_type == "non-flux":
                layers[-1].locus_components = set(inst.locus).union(layers[-1].locus_components)
                layers[-1].instructions.append(inst)
            else:
                layers.append(
                    CircuitLayer(layer_type="non-flux", locus_components=set(inst.locus), instructions=[inst])
                )
        return layers

    layered_circuits = [layer_circuit(c) for c in circuits]
    return layered_circuits


def set_layer_transformations(
    circuits: list[list[CircuitLayer]],
    flux_layer_transformations: list[str] | None = None,
    non_flux_layer_transformations: list[str] | None = None,
) -> list[list[CircuitLayer]]:
    """Set layer transformations for the layered circuits.

    .. note:: Modifies ``circuits`` in-place.

    Args:
        circuits: The layered circuits.
        flux_layer_transformations: List of transformations applied to flux layers. These should correspond to a
            transformation "gate" registered in the schedule builder.
        non_flux_layer_transformations: List of transformations applied to non-flux layers. These should correspond to a
            transformation "gate" registered in the schedule builder.

    Returns:
        The layered circuits with the applied transformations.

    """
    if not flux_layer_transformations and not non_flux_layer_transformations:
        return circuits

    for circuit in circuits:
        for layer in circuit:
            if layer.layer_type == "flux":
                layer.transformations = flux_layer_transformations  # type: ignore[assignment]
            elif layer.layer_type == "non-flux":
                layer.transformations = non_flux_layer_transformations  # type: ignore[assignment]
    return circuits


def resolve_layers(
    circuits: list[list[CircuitLayer]],
    components: ComponentGrouping,
    builder: ScheduleBuilder,
) -> list[TimeBox]:
    """Resolve the layered circuits into TimeBoxes, while applying the prescribed transformations.

    Args:
        circuits: The layered circuits.
        components: The full set of active locus components for the circuit batch.
        builder: The schedule builder

    Returns:
        TimeBoxes representing the circuits.

    """
    full_circuit_barrier = builder.get_implementation("barrier", tuple(components.flatten()))()
    timeboxes: list[TimeBox] = []
    for circuit in circuits:
        timebox = TimeBox.composite([])
        for layer in circuit:
            timebox += layer.resolve(builder) + full_circuit_barrier
        timeboxes.append(timebox)
    return timeboxes


def _find_first_offset(
    offset_duration: int, flux_pulses: dict[str, list[int]], non_flux_pulses: dict[str, list[int]], type: str | None
) -> tuple[str, int]:
    """Find the timepoint for the first instruction of the given type in the Schedule."""

    def _find_min(pulses: dict[str, list[int]]) -> int:
        min_duration = np.inf
        for channel, indexes in pulses.items():
            for duration in indexes:
                if duration > offset_duration and duration < min_duration:
                    min_duration = duration
        return min_duration  # type: ignore[return-value]

    if type == "flux":
        duration = _find_min(flux_pulses)
        return "flux", duration
    if type == "non-flux":
        duration = _find_min(non_flux_pulses)
        return "non-flux", duration
    flux_duration = _find_min(flux_pulses)
    non_flux_duration = _find_min(non_flux_pulses)
    if flux_duration < non_flux_duration:
        return "flux", flux_duration
    return "non-flux", non_flux_duration


def _extract_layer(
    schedule: Schedule,
    start_duration: int,
    end_duration: int,
) -> Schedule:
    """Extract a layer from the full schedule."""
    layers: dict[str, list[Instruction]] = defaultdict(list)
    for channel, segment in schedule.items():
        duration_counter = 0
        for inst in segment:
            if duration_counter >= start_duration and duration_counter < end_duration:
                layers[channel].append(inst)
            duration_counter += inst.duration
    return Schedule(layers)


def layer_schedules(
    schedules: list[Schedule],
    builder: ScheduleBuilder,
    components: ComponentGrouping,
    flux_layer_transformations: list[str] | None = None,
    non_flux_layer_transformations: list[str] | None = None,
) -> list[Schedule]:
    """Extract flux and non-flux layers from the full schedule and apply the prescribed transformations to them.

    .. note::

        Assumes the Schedules are already layered, will not be enforcing layering.

    Args:
        schedules: The Schedules representing the circuits.
        builder: The schedule builder
        components: The full set of active locus components for the circuit batch.
        flux_layer_transformations: List of transformations applied to flux layers. These should correspond to a
            transformation "gate" registered in the schedule builder.
        non_flux_layer_transformations: List of transformations applied to non-flux layers. These should correspond to a
            transformation "gate" registered in the schedule builder.

    Args:
        schedules: The Schedules with the prescribed transformations applied to their respective layers.

    """
    if not flux_layer_transformations and not non_flux_layer_transformations:
        return schedules

    new_schedules: list[Schedule] = []
    locus_components = tuple(components.flatten())

    for schedule in schedules:
        flux_pulses: dict[str, list[int]] = defaultdict(list)
        non_flux_pulses: dict[str, list[int]] = defaultdict(list)
        for channel, segment in schedule.items():
            duration_counter = 0
            # find the locations of all physical pulses
            for pulse_idx, pulse in enumerate(segment._instructions):
                if "flux" in channel and isinstance(pulse, FluxPulse):
                    flux_pulses[channel].append(duration_counter)
                elif (
                    "drive" in channel
                    and isinstance(pulse, (IQPulse, ConditionalInstruction))
                    or "readout" in channel
                    and isinstance(pulse, ReadoutTrigger)
                ):
                    non_flux_pulses[channel].append(duration_counter)
                duration_counter += pulse.duration
        # find, extract, and transform layers
        pulse_type, offset = _find_first_offset(0, flux_pulses, non_flux_pulses, None)
        pre_wait_box = TimeBox.atomic(
            Schedule({ch: [Wait(offset)] for ch in schedule.channels()}),
            locus_components=locus_components,
            label="pre_wait_layer",
        )
        layer_boxes: list[TimeBox] = [pre_wait_box]
        while offset < np.inf:
            next_pulse_type = "flux" if pulse_type == "non-flux" else "non-flux"
            _, next_offset = _find_first_offset(offset, flux_pulses, non_flux_pulses, next_pulse_type)
            layer_box = TimeBox.atomic(
                _extract_layer(schedule, offset, next_offset), locus_components=locus_components, label=pulse_type
            )
            transformations = flux_layer_transformations if pulse_type == "flux" else non_flux_layer_transformations
            for transformation in transformations or []:
                layer_box = builder.get_implementation(transformation, ())(layer_box)  # type: ignore[assignment]
            layer_boxes.append(layer_box)
            offset = next_offset
            pulse_type = next_pulse_type
        # finally schedule the layers
        new_schedules.append(builder.resolve_timebox(TimeBox.composite(layer_boxes), neighborhood=0))
    return new_schedules


layering_stage = CompilationStage(
    name="circuit_layering_stage",
    info="Group circuit operations into layers.",
)
layering_stage.add_passes(
    layer_circuits,
    set_layer_transformations,
)

layer_resolution_stage = CompilationStage(
    name="layer_resolution_stage",
    info="Resolve the layered circuits into TimeBoxes while applying the prescribed layer transformations.",
)
layer_resolution_stage.add_passes(resolve_layers)

schedule_layering_stage = CompilationStage(
    name="schedule_layering_stage",
    info="Apply transformations on layered instruction schedules.",
)
schedule_layering_stage.add_passes(layer_schedules)
