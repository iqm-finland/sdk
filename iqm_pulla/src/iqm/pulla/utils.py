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

"""Utility functions for IQM Pulla."""

from collections import namedtuple
from collections.abc import Hashable, Iterable, Sequence
from dataclasses import replace
from itertools import chain
from typing import Any

import numpy as np

from exa.common.data.setting_node import SettingNode
from exa.common.qcm_data.chip_topology import ChipTopology
from iqm.cpc.compiler.errors import InsufficientContextError
from iqm.cpc.compiler.station_settings import build_station_settings
from iqm.cpc.interface.compiler import Circuit as CPC_Circuit
from iqm.cpc.interface.compiler import (
    CircuitBoundaryMode,
    CircuitExecutionOptions,
    CircuitMetrics,
    HeraldingMode,
    ReadoutMappingBatch,
)
from iqm.pulla.interface import HERALDING_KEY, CalibrationSet, CircuitMeasurementResultsBatch
from iqm.pulse.builder import CircuitOperation, ScheduleBuilder, build_quantum_ops
from iqm.pulse.gate_implementation import CompositeGate, OpCalibrationDataTree
from iqm.pulse.playlist.channel import ChannelProperties
from iqm.pulse.playlist.instructions import Instruction
from iqm.pulse.playlist.schedule import Schedule, Segment
from iqm.pulse.timebox import TimeBox
from iqm.station_control.interface.models.observation import ObservationBase

LOCUS_SEPARATOR = "__"  # EXA uses this, currently


def circuit_operations_to_cpc(circ_ops: tuple[CircuitOperation], name: str | None = None) -> CPC_Circuit:
    """Convert a list of CircuitOperations to an IQM CPC Circuit.

    Args:
        circ_ops: The CircuitOperations to convert.
        name: Optional name of the circuit.

    Returns:
        The equivalent IQM CPC Circuit.

    """
    if name is None:
        name = f"circuit-{id(circ_ops)}"

    return CPC_Circuit(name=name, instructions=circ_ops)


def iqm_circuit_to_gate_implementation(circuit: CPC_Circuit, qubit_mapping: dict[str, str]) -> type[CompositeGate]:
    """Wrap a circuit to a single GateImplementation that can then be registered as an independent "gate".

    Returns a composite GateImplementation which, when called, produces a TimeBox with the circuit contents
    scheduled ASAP. The input ``circuit`` must contain only gates that are registered in IQM Pulse.
    The gate implementation does not need calibration data of its own: it uses the calibration of the registered gates.

    Args:
        circuit: circuit to wrap, typically a small subset of a larger circuit.
        qubit_mapping: Mapping from logical qubit names to physical qubit names.

    Returns:
        A new class ``CircuitAsComposite`` which can be registered as a new gate implementation.

    """

    class CircuitAsComposite(CompositeGate):
        """Dynamically created GateImplementation that wraps a circuit."""

        # TODO This method does not heed GateImplementation.locus, and will always apply the
        # gate on fixed qubits. It just pastes ``circuit`` contents, with the qubits mapped using
        # the likewise fixed mapping, into a TimeBox.
        registered_gates = list({instr.name for instr in circuit.instructions})

        def __call__(self):
            boxes = []
            for instr in circuit.instructions:
                locus = tuple(qubit_mapping[qubit] for qubit in instr.locus)

                boxes.append(self.build(instr.name, locus, instr.implementation)(**instr.args))

            return TimeBox.composite(boxes, label=circuit.name)

    return CircuitAsComposite


def _get_trigger_indexing_for(readout_mappings: ReadoutMappingBatch) -> tuple[dict[str, int], list[list[str]]]:
    """Information for deciphering the circuit batch RO results returned by Station Control.

    Args:
        readout_mappings: RO mappings for the circuit batch

    Returns:
        mapping from acquisition label to total number of times it appears in the batch (at most once per circuit),
        acquisition labels present in each circuit in the batch

    """
    num_triggers_for_label: dict[str, int] = {}  # how many triggers for each RO acq label
    labels_for_circuit: list[list[str]] = []  # which RO acquisition labels were present in each circuit
    for readout_mapping in readout_mappings:
        circuit_labels: list[str] = []
        for labels in readout_mapping.values():
            circuit_labels.extend(labels)
            for label in labels:
                if label not in num_triggers_for_label:
                    num_triggers_for_label[label] = 1
                else:
                    num_triggers_for_label[label] += 1
        labels_for_circuit.append(circuit_labels)

    return num_triggers_for_label, labels_for_circuit


def _result_idx(label: str, circuit_idx: int, labels_for_circuit: list[list[str]]) -> int:
    """Index of the RO result that corresponds to the given acquisition label in the given circuit."""
    return len([circuit_labels for circuit_labels in labels_for_circuit[:circuit_idx] if label in circuit_labels])


def convert_sweep_spot(
    results: dict[str, np.ndarray], readout_mappings: ReadoutMappingBatch
) -> CircuitMeasurementResultsBatch:
    """Convert the sweep measurement results from Station Control into circuit measurement results.

    Args:
        results: mapping of acquisition labels to 1D arrays of readout results with the length
            ``num_shots * num_triggers_for_label_in_batch``
        readout_mappings: for each circuit in the batch, a mapping of measurement keys to corresponding
            tuples of acquisition labels

    Returns:
        converted measurement results

    """
    num_triggers_for_label, labels_for_circuit = _get_trigger_indexing_for(readout_mappings)
    first_key = next(iter(results))
    num_shots = len(results[first_key]) // num_triggers_for_label[first_key]  # num shots equal for all labels

    results = {
        label: measurements.reshape((num_shots, num_triggers_for_label[label]))
        for label, measurements in results.items()
    }
    return [
        {
            mk: np.stack(
                [results[label][:, _result_idx(label, circuit_idx, labels_for_circuit)] for label in result_labels],
                axis=1,
            ).tolist()
            for mk, result_labels in readout_mapping.items()
        }
        for circuit_idx, readout_mapping in enumerate(readout_mappings)
    ]


def convert_sweep_spot_with_heralding_mode_zero(
    results: dict[str, np.ndarray], readout_mappings: ReadoutMappingBatch
) -> CircuitMeasurementResultsBatch:
    """Like :func:`convert_sweep_spot`, but for results that contain heralding measurements.

    * For each circuit we only keep the shots for which the heralding result is zero for all the
      qubits used in the circuit.

    Args:
        results: Mapping of acquisition labels to 1D arrays of readout results with the length
            ``num_shots * num_triggers_for_label_in_batch``. The herald
            results are found under ``HERALDING_KEY``.
        readout_mappings: For each circuit in the batch, a mapping of measurement keys to corresponding
            tuples of acquisition labels.

    Returns:
        converted, filtered measurement results, with the heralding measurement data removed

    """
    num_triggers_for_label, labels_for_circuit = _get_trigger_indexing_for(readout_mappings)
    first_key = next(iter(results.keys()))
    num_shots = len(results[first_key]) // num_triggers_for_label[first_key]  # num shots equal for all labels
    transformed_results: CircuitMeasurementResultsBatch = []
    results = {
        label: measurements.reshape((num_shots, num_triggers_for_label[label]))
        for label, measurements in results.items()
    }
    for circuit_idx, readout_mapping in enumerate(readout_mappings):
        # only use the wanted data for each circuit
        herald_readout_labels = readout_mapping[HERALDING_KEY]
        # For each circuit, we only keep those shots for which the heralding result is zero for
        # all the qubits used in that circuit.
        herald_results = np.stack(
            [results[label][:, _result_idx(label, circuit_idx, labels_for_circuit)] for label in herald_readout_labels],
            axis=0,
        )
        mask = np.all(herald_results == 0, axis=0)
        if not np.any(mask):
            # TODO this is the best we can do right now, since the current iqm-client transfer format
            # cannot handle returning zero shots
            raise RuntimeError(
                f'Execution of circuit {circuit_idx} in heralding mode "{HeraldingMode.ZEROS}" discarded all the shots.'
            )
        transformed_results.append(
            {
                mk: np.stack(
                    [results[label][mask, _result_idx(label, circuit_idx, labels_for_circuit)] for label in labels],
                    axis=1,
                ).tolist()
                for mk, labels in readout_mapping.items()
                if mk != HERALDING_KEY
            }
        )
    return transformed_results


def extract_readout_controller_result_names(readout_mappings: ReadoutMappingBatch) -> set[str]:
    """Prepare readout controller names for the request."""
    return {item for mapping in readout_mappings for item in chain(*mapping.values())}


def map_sweep_results_to_logical_qubits(
    sweep_results: dict[str, list[np.ndarray]], readout_mappings: ReadoutMappingBatch, heralding_mode: HeraldingMode
) -> CircuitMeasurementResultsBatch:
    """Convert sweep results returned by Station Control to the circuit measurement results the client expects.

    Args:
        sweep_results: mapping of acquisition labels to a list of soft sweep spots, each represented by a 1D
            array of readout results, with ``shots * num_triggers_for_label`` elements.
        readout_mappings: for each circuit in the batch, a mapping of measurement keys to corresponding
            tuples of result parameter names.
        heralding_mode: Heralding mode, either ``ZEROS`` (when doing heralded readout) or ``NONE``.

    Returns:
        converted, filtered measurement results, with the heralding measurement data removed

    """
    # TODO the SC return data format should be rationalized, for example list[dict[str, np.ndarray]]
    # where the list has soft sweep spots, dict keys are result labels, and the array has shape
    # (shots, num_triggers_for_label) where the latter represents the hard sweep/circuit batch.

    # circuit execution uses just one soft sweep spot
    results = {k: v[0] for k, v in sweep_results.items()}
    if heralding_mode == HeraldingMode.NONE:
        return convert_sweep_spot(results, readout_mappings)
    return convert_sweep_spot_with_heralding_mode_zero(results, readout_mappings)


InstructionLocation = namedtuple("InstructionLocation", ["channel_name", "index", "duration"])
"""Return type for :func:`locate_instructions`."""


def locate_instructions(
    schedule: Schedule,
    instruction_type: type[Instruction],
    min_duration: int = 0,
    *,
    channels: Iterable[str] | None = None,
) -> list[InstructionLocation]:
    """Locate specific instructions in a schedule.

    Args:
        schedule: The schedule to search.
        instruction_type: The type of the instruction to search for.
        min_duration: The minimum duration of the instruction to search for (in samples).
        channels: Names of channels in ``schedule`` to search. Iff None, search all the channels.

    Returns:
        For each located instruction, a namedtuple containing the channel name, instruction index, and duration.

    """
    if channels is None:
        channels = schedule.channels()

    result = [
        InstructionLocation(channel_name=channel, index=index, duration=inst.duration)
        for channel in channels
        for index, inst in enumerate(schedule[channel]._instructions)
        if isinstance(inst, instruction_type)
        if inst.duration >= min_duration
    ]
    return result


def print_schedule(schedule: Schedule) -> None:
    """Print all instructions in each segment of a schedule.

    Args:
        schedule: The schedule to print

    """
    for channel in schedule.channels():
        print("\n--------------\n")
        print_channel(schedule, channel)


def print_channel(schedule: Schedule, channel_name: str) -> None:
    """Print all instructions in a channel of a schedule.

    Args:
        schedule: The schedule to search.
        channel_name: The name of the channel to print.

    """
    if channel_name not in schedule:
        raise ValueError(f"No channel named {channel_name} in schedule")
    segment = schedule[channel_name]
    print(f"Channel name: {channel_name}\n")
    for index, inst in enumerate(segment._instructions):
        print(f"[{index}]: {inst.__class__.__name__} ({inst.duration})")


def replace_instruction_in_place(
    schedule: Schedule,
    channel_name: str,
    index: int,
    replacement: Iterable[Instruction],
) -> Schedule:
    """Replace an instruction in a schedule with one or more instructions.

    Args:
        schedule: The schedule to modify.
        channel_name: The name of the channel containing the instruction to replace.
        index: The index of the instruction to replace.
        replacement: Instructions to replace the original instruction with.

    Returns:
        The modified schedule.

    """
    if channel_name not in schedule:
        raise ValueError(f"No channel named {channel_name} in schedule")
    segment = schedule[channel_name]
    if 0 <= index < len(segment):
        replacement_segment = Segment(replacement)
        if segment[index].duration != replacement_segment.duration:
            raise ValueError(
                f"""Replacement duration does not match original segment duration.
Original:    {segment[index].duration}
Replacement: {replacement_segment.duration}"""
            )
        # create a new segment with old values until index + replacement_segment contents + old values after index
        new_segment = Segment([])
        new_segment.extend(segment[0:index])
        new_segment.extend(replacement_segment._instructions)
        new_segment.extend(segment[index + 1 :])
    else:
        raise ValueError(f"Index {index} is not in the segment.")

    schedule[channel_name] = new_segment
    return schedule


def map_qubit_indices(
    circuits: Iterable[CPC_Circuit], context: dict[str, Any]
) -> tuple[list[CPC_Circuit], dict[str, Any]]:
    """Map qubit indices in circuits to the indices in the component mapping."""
    try:
        component_mapping = context["component_mapping"]
    except Exception as exc:
        raise InsufficientContextError(
            f"Missing context data for circuit map_names_and_validate_circuits pass: {exc}"
        ) from exc

    qb_to_index_mapping = {v: k for k, v in component_mapping.items()}
    for circuit in circuits:
        for inst in circuit.instructions:
            inst.locus = tuple(qb_to_index_mapping.get(item, item) for item in inst.locus)

    return list(circuits), context


def get_hash_for(circuit: CPC_Circuit) -> int:
    """Get a unique id hash for a given circuit.

    In the context of this function, two CPC circuits are considered equal if they have:
    1. The same CircuitOperations in the same order.
    2. The loci of those circuit operations are the same in all operation.
    """
    str_repr = ""
    for idx, inst in enumerate(circuit.instructions):
        locus_str = "__".join(inst.locus)
        str_repr += f"{idx}_{inst.name}_{locus_str}"
    return hash(str_repr)


def calset_to_cal_data_tree(calibration_set: CalibrationSet) -> OpCalibrationDataTree:
    """Build an iqm-pulse QuantumOp calibration data tree from a calibration set.

    Splits the dotted observation names that are prefixed with "gates." into the corresponding
    calibration data tree paths.
    """

    def set_path(node: dict[Hashable, Any], path: Sequence[Hashable], value: Any) -> None:
        """Insert ``value`` into the tree ``node``, at the location given by ``path``.

        Modifies ``node``.
        """
        if len(path) == 1:
            node[path[0]] = value
            return
        # recurse into a subnode
        set_path(node.setdefault(path[0], {}), path[1:], value)

    tree: OpCalibrationDataTree = {}
    for key, value in calibration_set.items():
        path = key.split(".")
        if path[0] == "gates":
            if len(path) < 5:
                raise ValueError(f"Calibration observation name '{key}' is malformed.")
            # treat the locus specially
            locus = tuple(path[3].split(LOCUS_SEPARATOR))
            locus = () if locus == ("",) else locus
            # mypy likes this
            set_path(tree.setdefault(path[1], {}).setdefault(path[2], {}).setdefault(locus, {}), path[4:], value)
    return tree


def initialize_schedule_builder(
    calibration_set: CalibrationSet,
    chip_topology: ChipTopology,
    channel_properties: dict[str, ChannelProperties],
    component_channels: dict[str, dict[str, str]],
) -> ScheduleBuilder:
    """Initialize a new schedule builder for the station, validate that it is configured properly.

    Args:
        calibration_set: calibration data for the station the circuits are executed on
        chip_topology: topology of the QPU the circuits are executed on
        channel_properties: properties of control channels on the station
        component_channels: QPU component to function to channel mapping
    Returns:
        schedule builder for the station

    """
    op_table = build_quantum_ops({})

    channel_properties = _update_channel_props_from_calibration(channel_properties, component_channels, calibration_set)

    builder = ScheduleBuilder(
        op_table,
        calset_to_cal_data_tree(calibration_set),
        chip_topology,
        channel_properties,
        component_channels,
    )
    return builder


def _update_channel_props_from_calibration(
    channel_properties: dict[str, ChannelProperties],
    component_channels: dict[str, dict[str, str]],
    calset: CalibrationSet,
):
    """Copy probe line center frequencies from a calset to their readout channel properties.

    Args:
        channel_properties: channel properties to update
        component_channels: mapping from QPU component to its functions/channels that perform them
        calset: calibration data
    Returns:
        updated channel properties

    """
    # TODO get rid of this and find a better way of passing center_frequency to measure gate implementation
    replacements = {}
    for component, channels in component_channels.items():
        if "readout" in channels:
            center_frequency = calset.get(f"controllers.{component}.readout.center_frequency")
            if center_frequency is None:
                center_frequency = calset.get(f"controllers.{component}.readout.local_oscillator.frequency")
            if center_frequency is None:
                raise ValueError(
                    f"No calibration value found for the center frequency or local oscillator frequency of {component}."
                )
            channel_name = channels["readout"]
            replacements[channel_name] = replace(channel_properties[channel_name], center_frequency=center_frequency)

    return channel_properties | replacements


def find_circuit_boundary(
    mode: CircuitBoundaryMode,
    circuit_components: set[str] | frozenset[str],
    circuit_couplers: set[str],
    device: ChipTopology,
) -> tuple[set[str], set[str]]:
    """Determine the boundary of a circuit executed on the QPU.

    See :class:`.CircuitBoundaryMode` for the definitions of the circuit boundaries.

    Args:
        mode: method of determining the circuit border
        circuit_components: all locus components used in the circuit
        circuit_couplers: all couplers used in the circuit to apply gates
        device: describes the QPU topology

    Returns:
        boundary locus components, boundary couplers

    Raises:
        ValueError: unknown ``mode``

    """
    if mode == CircuitBoundaryMode.NEIGHBOUR:
        boundary_components = device.get_neighbor_locus_components(circuit_components)
        boundary_couplers = {
            coupler for coupler in device.get_neighbor_couplers(circuit_components) if coupler not in circuit_couplers
        }
    elif mode == CircuitBoundaryMode.ALL:
        # maybe safer/better: all unused locus components/couplers are considered boundary
        boundary_components = (device.qubits | device.computational_resonators) - circuit_components
        boundary_couplers = device.couplers - circuit_couplers
    else:
        raise ValueError(f"Unknown circuit boundary mode '{str(mode)}'")
    return boundary_components, boundary_couplers


def build_settings(
    shots: int,
    calibration_set: CalibrationSet,
    builder: ScheduleBuilder,
    circuit_metrics: Iterable[CircuitMetrics],
    *,
    options: CircuitExecutionOptions,
) -> SettingNode:
    """Construct the Station Control settings needed for executing a batch of quantum circuits.

    Args:
        shots: number of times to execute/sample each circuit
        calibration_set: calibration data for the station the circuits are executed on
        builder: schedule builder object, encapsulating station properties and gate calibration data
        circuit_metrics: statistics about the circuits to be executed
        options: various discrete options for circuit execution that affect compilation

    Returns:
        Station Control settings

    """
    # NOTE: We prepare just one set of SC settings for the entire circuit batch!
    device = builder.chip_topology
    # coupling topology: mapping coupled locus component pairs to the name of the tunable coupler
    device_coupled_pair_to_coupler: dict[frozenset[str], str] = {
        frozenset(components): coupler
        for coupler, components in device.coupler_to_components.items()
        if len(components) == 2
    }

    # sets of components used by any of the circuits
    circuit_components_set = frozenset.union(*(m.components for m in circuit_metrics))
    circuit_component_pairs_set = frozenset.union(
        *(frozenset(frozenset(pair) for pair in m.component_pairs_with_gates) for m in circuit_metrics)
    )
    # currently we assume all two-component gates require a coupler
    # TODO this information should come from the gate implementation?
    circuit_couplers_set = {device_coupled_pair_to_coupler[pair] for pair in circuit_component_pairs_set}

    # build station settings using the calibration data
    boundary_components, boundary_couplers = find_circuit_boundary(
        options.circuit_boundary_mode,
        circuit_components_set,
        circuit_couplers_set,
        device,
    )
    settings = build_station_settings(
        circuit_qubits=circuit_components_set & device.qubits,
        circuit_couplers=circuit_couplers_set,
        # always turn all probe lines on, since figuring out exactly which lines we need for a batch
        # of circuits is needlessly complicated and usually would yield a small benefit
        measured_probe_lines=device.probe_lines,
        shots=shots,
        calibration_set=calibration_set,
        boundary_qubits=boundary_components & device.qubits,
        boundary_couplers=boundary_couplers,
        flux_pulsed_qubits=[
            component
            for component, functions in builder.component_channels.items()
            if "flux" in functions and component in device.qubits
        ],
    )

    return settings


def calset_from_observations(calset_observations: Iterable[ObservationBase]) -> CalibrationSet:
    """Create a Pulla calibration set from the given observations.

    Args:
        calset_observations: observations that form a calibration set

    Returns:
        calibration set in Pulla format

    """
    return {obs.dut_field: obs.value for obs in calset_observations}
