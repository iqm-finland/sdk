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
"""Standard compilation stages, their constituent compiler passes, and functions for implementing them.

There are 6 standard compilation stages:

1. Circuit-to-circuit.
2. Circuit-to-timebox.
3. Timebox-to-timebox.
4. Timebox-to-schedule.
5. Schedule-to-schedule.
6. Schedule-to-playlist.

Breakdown of compiler passes of each stage:

1. Circuit-to-circuit
*********************

1. Validate the circuit execution options.
2. Map backwards-compatible aliases for quantum operation names into the current name. This is needed until old
   operation names such as ``phased_rx`` and ``measurement`` are no longer supported.
3. Validate the contents of the circuits.
4. Map the logical QPU components to physical QPU components. Provided mapping is used, if any.
   Otherwise, identity mapping is used.
5. Choose implementations for circuit operations based on the calibration set.
6. Derive mapping between station acquisition labels and user's measurement keys. Populates ``readout_mappings`` and
   ``heralded_components`` of context.

2. Circuit-to-timebox
*********************

1. Resolve the circuits to timeboxes using :meth:`ScheduleBuilder.circuit_to_timebox`.

3. Timebox-to-timebox
*********************

1. Merge any MultiplexedProbeTimeBoxes inside each TimeBox using :meth:`TimeBox.composite`.
2. Add the heralding measurement timebox to all circuits if :class:`HeraldingMode` in circuit execution options
   requires it.
3. Add a reset timebox to all circuits.

4. Timebox-to-schedule
**********************

1. Resolve the timeboxes to schedules using :meth:`ScheduleBuilder.resolve_timebox`.

5. Schedule-to-schedule
***********************

1. Apply dynamical decoupling sequences to the schedule if requested.
2. Apply resonator-related phase corrections if MOVE gates are used.
3. Remove non-functional instructions from schedules using :meth:`ScheduleBuilder._finish_schedule`.

6. Schedule-to-playlist
***********************

1. Build the playlist from the schedules using :meth:`ScheduleBuilder.build_playlist`.
"""

from collections import Counter
from collections.abc import Iterable
from copy import copy, deepcopy
from typing import Any

import numpy as np

from iqm.cpc.compiler.compiler import CompilationStage, compiler_pass, pass_function_idempotent
from iqm.cpc.compiler.dd import STANDARD_DD_STRATEGY, insert_dd_sequences, merge_wait_instructions_in_schedule
from iqm.cpc.compiler.errors import (
    CalibrationError,
    CircuitError,
    UnknownHardwareComponentError,
    UnknownLogicalQubitError,
)
from iqm.cpc.interface.compiler import Circuit as Circuit_  # something weird with sphinx
from iqm.cpc.interface.compiler import (
    CircuitExecutionOptions,
    CircuitMetrics,
    DDMode,
    DDStrategy,
    HeraldingMode,
    Locus,
    MeasurementMode,
    MoveGateFrameTrackingMode,
    MoveGateValidationMode,
    ReadoutMapping,
)
from iqm.pulla.interface import (
    ACQUISITION_LABEL,
    ACQUISITION_LABEL_KEY,
    BUFFER_AFTER_MEASUREMENT_PROBE,
    HERALDING_KEY,
    MEASUREMENT_MODE_KEY,
    RESTRICTED_MEASUREMENT_KEYS,
    CalibrationErrors,
)
from iqm.pulse.base_utils import merge_dicts
from iqm.pulse.builder import CircuitOperation, ScheduleBuilder, validate_quantum_circuit
from iqm.pulse.gate_implementation import OpCalibrationDataTree
from iqm.pulse.gates import move
from iqm.pulse.playlist import Schedule
from iqm.pulse.playlist.playlist import Playlist
from iqm.pulse.quantum_ops import QuantumOpTable
from iqm.pulse.timebox import MultiplexedProbeTimeBox, TimeBox


# Compilation functions
def _map_old_operation_names(instructions: Iterable[CircuitOperation]) -> None:
    """Maps backwards-compatible aliases for quantum operation names into the current name.

    Modifies the given instructions.
    """
    mapping = {"phased_rx": "prx", "measurement": "measure"}
    for inst in instructions:
        inst.name = mapping.get(inst.name, inst.name)


def _map_old_operation_arguments(instructions: Iterable[CircuitOperation]) -> None:
    """Maps instruction arguments from various old conventions to the current
    ``iqm-pulse`` convention for backwards compatibility.

    In iqm-pulse,
    * prx arguments are now measured in radians instead of full turns.

    Modifies the given instructions.
    """
    prx_map = {"angle_t": "angle", "phase_t": "phase"}

    for inst in instructions:
        if inst.name in ("prx", "cc_prx"):
            for old, new in prx_map.items():
                if (val := inst.args.pop(old, None)) is not None:
                    inst.args[new] = 2 * np.pi * val


def _map_components_in_instructions(
    component_mapping: dict[str, str] | None,
    instructions: Iterable[CircuitOperation],
    device_components: set[str],
) -> None:
    """Maps the logical component names in a sequence of instructions to the corresponding physical component names.

    Modifies ``instructions`` in place.

    Args:
        component_mapping: Mapping of logical component names to physical component names.
            ``None`` means the identity mapping.
        instructions: instructions to map
        device_components: names of the physical locus components on the device

    """

    def logical_to_physical_component_name(component: str) -> str:
        """Map the logical name of a component to the corresponding physical name.

        Raises user-friendly error messages if the component is not found.

        Args:
            component: logical component name

        Returns:
            corresponding physical component name

        Raises:
            UnknownLogicalQubitError: ``component`` does not map to a physical component
            UnknownHardwareComponentError: the physical component corresponding to ``component`` does not exist

        """
        p_component = component if component_mapping is None else component_mapping.get(component)
        if p_component is None:
            raise UnknownLogicalQubitError(f"Logical qubit '{component}' does not map to a physical qubit.")
        if p_component not in device_components:
            if component_mapping is None:
                raise UnknownHardwareComponentError(f"Unknown physical qubit '{p_component}'.")
            raise UnknownHardwareComponentError(
                f"The logical qubit '{component}' maps to the physical qubit '{p_component}'. "
                f"However, this physical qubit does not appear on the QPU."
            )
        return p_component

    for inst in instructions:
        inst.locus = tuple(map(logical_to_physical_component_name, inst.locus))


def _fix_implementation_and_locus(
    instructions: Iterable[CircuitOperation], builder: ScheduleBuilder, calibration_errors: CalibrationErrors
) -> CircuitMetrics:
    """Try to find a usable implementation and locus order for each of the given instructions.

    .. note:: This function modifies ``instructions`` in place.

    Args:
        instructions: instructions to be analyzed
        builder: encapsulates the known instructions and gate calibration data
        calibration_errors: Found errors for each OIL.

    Returns:
        circuit metrics for ``instructions``

    Raises:
        CircuitError: No valid implementation was found for an instruction.
        CalibrationError: ``instructions`` require an implementation which has faulty calibration data.

    """
    circuit_components: set[str] = set()
    circuit_component_pairs: set[tuple[str, str]] = set()
    circuit_gate_loci: dict[str, dict[str, Counter[Locus]]] = {}
    for inst in instructions:
        op = builder.op_table[inst.name]
        try:
            inst.implementation, locus = builder._find_implementation_and_locus(
                op, inst.implementation, inst.locus, strict_locus=False
            )
        except ValueError as exc:
            raise CircuitError(f"{exc}") from exc
        if (error := calibration_errors.get((inst.name, inst.implementation, locus))) is not None:
            all_errors = "\n- ".join(calibration_errors.values())
            raise CalibrationError(f"Inconsistencies in calibration set:\n- {all_errors}\n\nCause of error:\n{error}")
        inst.locus = locus
        circuit_components.update(locus)
        if op.arity == 2:
            circuit_component_pairs.add(locus)
        circuit_gate_loci.setdefault(inst.name, {}).setdefault(inst.implementation, Counter()).update([locus])

    return CircuitMetrics(
        components=frozenset(circuit_components),
        component_pairs_with_gates=frozenset(circuit_component_pairs),
        gate_loci=circuit_gate_loci,
    )


def _build_readout_mappings(
    circuits: Iterable[Circuit_],
    circuit_metrics: tuple[CircuitMetrics, ...],
    builder: ScheduleBuilder,
    measurement_mode: MeasurementMode,
    heralding_mode: HeraldingMode,
) -> tuple[tuple[ReadoutMapping, ...], tuple[tuple[str, ...], ...]]:
    """Create a readout mapping for each circuit, add required measurement instructions.

    Depending on ``measurement_mode``, we may need to add an additional final measurement instruction
    to each circuit to measure the qubits that the mode requires to be measured, but aren't.
    These extra measured qubits are not added to the readout mapping, because the user is not
    interested in the result.

    Also figures out which qubits should get a heralding measurement for each circuit.

    Args:
        circuits: circuits to modify
        circuit_metrics: circuit metrics (including which qubits are used in each circuit)
        builder: schedule builder object, encapsulating station properties and gate calibration data
        measurement_mode: measurement mode to be used
        heralding_mode: type of heralding used

    Returns:
        readout mappings, heralded components

    """
    readout_mappings = []
    heralded_components = []

    # find out which components have measurement data
    components_that_can_be_measured = frozenset(
        q for impl_loci in builder.calibration["measure"].values() for locus in impl_loci for q in locus
    )

    # Figure out which qubits should always be measured on the QPU in the final measurement.
    if measurement_mode == MeasurementMode.ALL:
        # those that can be measured
        final_components_to_measure = components_that_can_be_measured
    elif measurement_mode == MeasurementMode.CIRCUIT:
        # no extras on top of the measurements actually in each circuit
        final_components_to_measure = frozenset()
    else:
        raise ValueError(f"Unknown measurement mode: {measurement_mode}")

    for idx, c in enumerate(circuits):
        try:
            readout_mapping: ReadoutMapping = {}
            final_measurements = []  # measurement instructions that are final
            eclipsed_qubits: set[str] = set()  # qubits which already have had an instruction on them
            total_measurement_instructions = sum(1 for inst in c.instructions if inst.name == "measure")
            for inst in reversed(c.instructions):
                # add all measure instructions to the readout mapping
                # iterate from the end to figure out final measurements
                if inst.name == "measure":
                    key = inst.args["key"]

                    # Some keys are reserved for internal use
                    if key in RESTRICTED_MEASUREMENT_KEYS:
                        raise CircuitError(
                            f"Measurement keys {RESTRICTED_MEASUREMENT_KEYS} are reserved for internal use."
                        )
                    # In an earlier stage, we have already validated the circuit and know that the keys are unique.
                    # Database has limitations for the length of the measurement keys, so replace user-given
                    # keys with something short and unique across all the measurements in the circuit.
                    mapped_key = ACQUISITION_LABEL_KEY.format(
                        idx=(total_measurement_instructions - len(readout_mapping))
                    )
                    readout_mapping[key] = tuple(
                        ACQUISITION_LABEL.format(qubit=qb, key=mapped_key) for qb in inst.locus
                    )
                    inst.args["key"] = mapped_key

                    if not set(inst.locus) & eclipsed_qubits:
                        # measurement is final iff none of its qubits is eclipsed
                        final_measurements.append(inst)
                eclipsed_qubits.update(inst.locus)

            # Add a single final measurement instruction for qubits that still need to be measured
            # (order does not matter). this measurement instruction is not reflected in circuit_metrics,
            # nor are the results returned by SC. NOTE we assume that iqm-pulse is clever enough to
            # combine the separate final measurement instructions into simultaneous ReadoutTriggers.
            if missing := final_components_to_measure - set(q for inst in final_measurements for q in inst.locus):
                # TODO which implementation to use? Now we leave it to None.
                c.instructions += (CircuitOperation("measure", tuple(missing), {"key": MEASUREMENT_MODE_KEY}),)

            if heralding_mode != HeraldingMode.NONE:
                # heralding must return results for the components used in the circuit, if they can be measured
                # NOTE that no error will be raised here if a qubit used in a circuit has no measurements available
                # and thus cannot be heralded!
                heralded = tuple(circuit_metrics[idx].components & components_that_can_be_measured)
                readout_mapping[HERALDING_KEY] = tuple(
                    ACQUISITION_LABEL.format(qubit=qb, key=HERALDING_KEY) for qb in heralded
                )
            else:
                heralded = ()

            readout_mappings.append(readout_mapping)
            heralded_components.append(heralded)
        except (CircuitError, ValueError) as exc:
            raise CircuitError(f"Circuit {idx}: {exc}") from exc
    return tuple(readout_mappings), tuple(heralded_components)


def _get_op_calibration_errors(calibration: OpCalibrationDataTree, ops: QuantumOpTable) -> CalibrationErrors:
    """Validates quantum operation calibration data against the known quantum operations.

    NOTE: calibration data parameters that have a defined default value are not required to be in the calibration data.

    Args:
        calibration: quantum operation calibration data tree to validate
        ops: known quantum operations and their implementations

    Returns:
        Mapping from op name, implementation name, locus to an error string or a None, if there are no errors for
        that locus.

    """

    def diff_dicts(cal_data: dict[str, Any], impl_parameters: dict[str, Any], path) -> None | str:
        """Compare the calibration"""
        full = set(impl_parameters)
        have = set(cal_data)
        need = set(k for k, v in impl_parameters.items() if not hasattr(v, "value"))
        if need == {"*"}:
            return None
        if diff := have - full:
            return f"Unknown calibration data for {op_name}.{impl_name} at {locus}: {'.'.join(path)} {diff}"
        if diff := need - have:
            return f"Missing calibration data for {op_name}.{impl_name} at {locus}: {'.'.join(path)} {diff}"
        for key, data in cal_data.items():
            required_value = impl_parameters[key]
            new_path = path + [key]
            if isinstance(required_value, dict):
                if isinstance(data, dict):
                    diff_dicts(data, required_value, new_path)
                else:
                    return (
                        f"Calibration data for {op_name}.{impl_name} at {locus}: "
                        f"'{'.'.join(new_path)}' should be a dict."
                    )
            elif isinstance(data, dict):
                return (
                    f"Calibration data for {op_name}.{impl_name} at {locus}: '{'.'.join(new_path)}' should be a scalar."
                )
        return None

    errors = {}

    for op_name, implementations in calibration.items():
        if (op := ops.get(op_name)) is None:
            errors[(op_name, "", ())] = f"Unknown operation '{op_name}'. Known operations: {tuple(ops.keys())}"
            continue

        for impl_name, loci in implementations.items():
            if (impl := op.implementations.get(impl_name)) is None:
                errors[(op_name, impl_name, ())] = (
                    f"Unknown implementation '{impl_name}' for quantum operation '{op_name}'. "
                    f"Known implementations: {tuple(op.implementations.keys())}"
                )
                continue

            default_cal_data = loci.get((), {})
            for locus, cal_data in loci.items():
                if not locus:
                    continue  # default cal data for all loci
                # since OILCalibrationData can have nested dicts, we do a recursive diff
                error = diff_dicts(merge_dicts(default_cal_data, cal_data), impl.parameters, [])
                if error is not None:
                    errors[(op_name, impl_name, locus)] = error
                    continue

                n_components = len(locus)
                arity = op.arity
                if arity == 0:
                    if n_components != 1:
                        errors[(op_name, impl_name, locus)] = (
                            f"{op_name}.{impl_name} at {locus}: for zero-arity operations, "
                            "calibration data must be provided for single-component loci only"
                        )
                elif n_components != arity:
                    errors[(op_name, impl_name, locus)] = (
                        f"{op_name}.{impl_name} at {locus}: locus must have {arity} component(s)"
                    )
    return errors


def merge_multiplexed_timeboxes(circuit_box: TimeBox) -> TimeBox:
    """Merge any MultiplexedProbeTimeBoxes inside a TimeBox representing a circuit.

    This pass optimizes a situation where multiple "measure" gates on disjoint set of loci exist sequentially in the
    circuit.
    Without optimization, each gate would result in a separate trigger event, which results in worse performance.
    For example, with the measurement instructions [M(QB1), M(QB2), M(QB3)], we'd first measure QB1, then QB2, then QB3.
    This optimization merges the measurement timeboxes, so that we'll measure QB1, QB2, and QB3 at the same time
    (if the hardware channel configuration allows it), corresponding to M(QB1, QB2, QB3).

    Goes through the children of `circuit_box`, and places them in the same temporal order.
    Whenever a MultiplexedProbeTimeBox is encountered (i.e. from a measure gate), it is merged with the previous pending
    MultiplexedProbeTimeBox and left pending.
    If any other box type with colliding loci is encountered, first places the pending MultiplexedProbeTimeBox.
    This essentially delays all measurements until the last possible moment.

    Args:
        circuit_box: Timebox representing a circuit, where each child should represent a single gate.

    Returns:
        A new Timebox with the same content, except with some MultiplexedProbeTimeBoxes merged.

    """

    # TODO make this a timebox-level compiler pass, maybe move it to iqm-pulse, make able to discover more
    # mergable cases like for example [M(1), U(1), M(2)] -> [M(1, 2), U(1)]
    # Consider implementing this step completely differently. This seems very fragile.
    def disjoint_boxes(box1: TimeBox, box2: TimeBox) -> bool:
        if not box1.locus_components or not box2.locus_components:
            if box1.neighborhood_components.get(0, set()).intersection(box2.neighborhood_components.get(0, set())):
                return False
        return len(box1.locus_components & box2.locus_components) == 0

    placed_boxes = []
    pending = None
    for gate_box in circuit_box.children:
        if gate_box.children and isinstance(gate_box.children[0], MultiplexedProbeTimeBox):
            if pending:
                if disjoint_boxes(pending, gate_box):
                    # Pending box and new candidate have disjoint loci, merge is possible.
                    pending = pending + gate_box.children[0]
                    continue
                # Pending box collides with the new candidate, so we must place it immediately and continue with
                # the new candidate.
                placed_boxes.append(pending)
            pending = gate_box.children[0]
        else:
            # If the pending box touches the same locus components as this gate, we need to place it now.
            if pending and not disjoint_boxes(pending, gate_box):
                placed_boxes.append(pending)
                pending = None
            placed_boxes.append(gate_box)

    if pending:
        placed_boxes.append(pending)
    return TimeBox.composite(
        placed_boxes,
        label=circuit_box.label,
        scheduling=circuit_box.scheduling,
        scheduling_algorithm=circuit_box.scheduling_algorithm,
    )


# Passes of the standard stages
# CIRCUIT-LEVEL PASSES
@compiler_pass
def validate_execution_options(circuits: Iterable[Circuit_], options: CircuitExecutionOptions):
    """Validate the circuit execution options (only some combinations make sense)."""
    if options.move_gate_frame_tracking == MoveGateFrameTrackingMode.FULL and options.move_gate_validation not in [
        MoveGateValidationMode.STRICT,
        MoveGateValidationMode.ALLOW_PRX,
    ]:
        raise CircuitError("Full MOVE gate frame tracking requires MOVE gate validation to be 'strict' or 'allow_prx'.")

    return circuits


@compiler_pass
def map_old_operations(circuits: Iterable[Circuit_]):
    """Map backwards-compatible aliases for quantum operation names into the current name."""
    for c in circuits:
        _map_old_operation_names(c.instructions)
        _map_old_operation_arguments(c.instructions)
    return circuits


@compiler_pass
def validate_circuits(circuits: Iterable[Circuit_], builder: ScheduleBuilder):
    """Validate the contents of the quantum circuits."""
    for idx, c in enumerate(circuits):
        try:
            validate_quantum_circuit(c.instructions, builder.op_table, require_measurements=True)
        except (CircuitError, ValueError) as exc:
            raise CircuitError(f"Circuit {idx}: {exc}") from exc
    return circuits


@compiler_pass
def map_components(
    circuits: Iterable[Circuit_],
    builder: ScheduleBuilder,
    component_mapping: dict[str, str],
):
    """Map the logical QPU components to physical QPU components using ``component_mapping``."""
    device = builder.chip_topology
    device_components = device.qubits | device.computational_resonators
    for idx, c in enumerate(circuits):
        try:
            _map_components_in_instructions(
                component_mapping,
                c.instructions,
                device_components=device_components,
            )
        except CircuitError as exc:
            raise CircuitError(f"Circuit {idx}: {exc}") from exc

    return circuits


@compiler_pass
def choose_op_implementations(
    circuits: Iterable[Circuit_],
    builder: ScheduleBuilder,
    options: CircuitExecutionOptions,
) -> tuple[list[Circuit_], dict[str, Any]]:
    """Analyze the instructions in the circuits and pick an implementation for each (operation, locus).

    .. note:: This function modifies ``circuits`` in place.
    """
    calibration_errors = _get_op_calibration_errors(builder.calibration, builder.op_table)
    circuit_metrics = []
    for idx, c in enumerate(circuits):
        try:
            # circuit part of the metrics describes the circuit as given in the execution request
            metrics = _fix_implementation_and_locus(c.instructions, builder, calibration_errors)
            circuit_metrics.append(metrics)
            if "move" in metrics.gate_loci and options.move_gate_validation != MoveGateValidationMode.NONE:
                # only run the MOVE passes if we need to
                move.validate_move_instructions(
                    c.instructions,
                    builder,
                    validate_prx=options.move_gate_validation != MoveGateValidationMode.ALLOW_PRX,
                )

        except (CircuitError, ValueError) as exc:
            raise CircuitError(f"Circuit {idx}: {exc}") from exc

    return list(circuits), {"circuit_metrics": tuple(circuit_metrics)}


@compiler_pass
def derive_readout_mappings(
    circuits: Iterable[Circuit_],
    builder: ScheduleBuilder,
    options: CircuitExecutionOptions,
    circuit_metrics: Iterable[CircuitMetrics],
) -> tuple[list[Circuit_], dict[str, Any]]:
    """Derive mapping between station acquisition labels and user's measurement keys."""
    readout_mappings, heralded_components = _build_readout_mappings(
        circuits, tuple(circuit_metrics), builder, options.measurement_mode, options.heralding_mode
    )
    return list(circuits), {"readout_mappings": readout_mappings, "heralded_components": heralded_components}


# CIRCUIT RESOLUTION PASS
@compiler_pass
def resolve_circuits(circuits: Iterable[Circuit_], builder: ScheduleBuilder) -> list[TimeBox]:
    """Resolve the circuits to timeboxes."""
    return [builder.circuit_to_timebox(c.instructions, name=c.name) for c in circuits]


# TIMEBOX-LEVEL PASSES
@compiler_pass
def multiplex_readout(timeboxes: Iterable[TimeBox]):
    """Merge any MultiplexedProbeTimeBoxes inside a TimeBox representing a circuit."""
    return [merge_multiplexed_timeboxes(circuit_box) for circuit_box in timeboxes]


@compiler_pass
def resolve_timeboxes(timeboxes: Iterable[TimeBox], builder: ScheduleBuilder) -> list[Schedule]:
    """Resolve the timeboxes to schedules."""
    schedules = [builder.resolve_timebox(box, neighborhood=1) for box in timeboxes]
    return schedules


@compiler_pass
def prepend_heralding(
    timeboxes: Iterable[TimeBox],
    builder: ScheduleBuilder,
    heralded_components: tuple[tuple[str, ...], ...],
    options: CircuitExecutionOptions,
) -> list[TimeBox]:
    """Add the heralding measurement timebox to all circuits."""
    if options.heralding_mode != HeraldingMode.ZEROS:
        return list(copy(timeboxes))
    return [
        builder.get_implementation("measure", heralded_components[circuit_idx])(key=HERALDING_KEY)
        + builder.wait(heralded_components[circuit_idx], BUFFER_AFTER_MEASUREMENT_PROBE)
        | box
        for circuit_idx, box in enumerate(timeboxes)
    ]


@compiler_pass
def prepend_reset(
    timeboxes: Iterable[TimeBox],
    builder: ScheduleBuilder,
    options: CircuitExecutionOptions,
    circuit_metrics: Iterable[CircuitMetrics],
) -> Iterable[TimeBox]:
    """Add a reset timebox to all circuits."""
    if "reset_wait" not in builder.calibration:  # backwards compatibility
        return timeboxes
    new_boxes: list[TimeBox] = []
    for box, metrics in zip(timeboxes, circuit_metrics):
        reset_components = tuple(metrics.components)
        if options.active_reset_cycles is None:
            reset_box = TimeBox.composite([builder.get_implementation("reset_wait", reset_components)()])
        else:
            reset_box = TimeBox.composite(
                [builder.get_implementation("reset", reset_components)()] * options.active_reset_cycles
            )
        new_boxes.append(reset_box + box)
    return new_boxes


# SCHEDULE-LEVEL PASSES
@compiler_pass
def apply_dd_strategy(
    schedules: Iterable[Schedule],
    builder: ScheduleBuilder,
    options: CircuitExecutionOptions,
) -> list[Schedule]:
    """Insert dynamical decoupling sequences into the schedules, if dynamical decoupling is enabled."""
    if options.dd_mode == DDMode.DISABLED:  # Skip the pass and move on
        return [schedule.copy() for schedule in schedules]

    if options.dd_mode == DDMode.ENABLED:
        if options.dd_strategy is None:
            strategy = STANDARD_DD_STRATEGY
        elif isinstance(options.dd_strategy, DDStrategy):
            strategy = options.dd_strategy
        else:
            raise ValueError("Unsupported dynamical decoupling strategy submitted.")

    else:
        raise ValueError(f"Unsupported dynamical decoupling mode requested ({options.dd_mode}).")

    new_schedules: list[Schedule] = []

    for schedule in schedules:
        if strategy.merge_contiguous_waits:
            new_schedule = merge_wait_instructions_in_schedule(builder, schedule)
        else:
            new_schedule = schedule.copy()

        new_schedule.cleanup()  # remove idling-only channels, where we do not want any DD taking place
        insert_dd_sequences(builder, new_schedule, strategy)
        new_schedules.append(new_schedule)

    return new_schedules


@compiler_pass
def apply_move_gate_phase_corrections(
    schedules: Iterable[Schedule],
    builder: ScheduleBuilder,
    circuit_metrics: Iterable[CircuitMetrics],
    options: CircuitExecutionOptions,
) -> list[Schedule]:
    """Apply calibrated phase corrections if MOVE gates are used."""
    processed_schedules = []
    for schedule, metrics in zip(schedules, circuit_metrics):
        if "move" in metrics.gate_loci and options.move_gate_frame_tracking != MoveGateFrameTrackingMode.NONE:
            # only run the MOVE passes if we need to
            try:
                schedule = move.apply_move_gate_phase_corrections(  # noqa: PLW2901
                    schedule,
                    builder,
                    apply_detuning_corrections=options.move_gate_frame_tracking
                    != MoveGateFrameTrackingMode.NO_DETUNING_CORRECTION,
                )
            except ValueError as exc:
                raise CircuitError(
                    "Unable to determine MOVE gate phase correction, consider using MOVE gate validation or turn "
                    + f"off MOVE gate frame tracking: {exc}"
                ) from exc

        processed_schedules.append(schedule)

    return processed_schedules


@compiler_pass
def clean_schedule(schedules: Iterable[Schedule], builder: ScheduleBuilder) -> list[Schedule]:
    """Remove non-functional instructions from `schedules`."""
    return [builder._finish_schedule(schedule) for schedule in schedules]


@compiler_pass
def build_playlist(schedules: Iterable[Schedule], builder: ScheduleBuilder) -> tuple[Playlist, dict[str, Any]]:
    """Build the playlist from the schedules."""
    playlist = builder.build_playlist(schedules)
    return playlist, {"schedules": schedules}  # save the schedules for building settings, debugging, etc


circuit_stage = CompilationStage(name="circuit")  # Circuit -> ... -> Circuit
circuit_resolution_stage = CompilationStage(name="circuit_resolution")  # Circuit -> TimeBox
timebox_stage = CompilationStage(name="timebox")  # TimeBox -> ... -> TimeBox
timebox_resolution_stage = CompilationStage(name="timebox_resolution")  # TimeBox -> Schedule
dd_stage = CompilationStage(name="dynamical_decoupling")  # Schedule -> Schedule
schedule_stage = CompilationStage(name="schedule")  # Schedule -> ... -> Schedule
final_stage = CompilationStage(name="schedule_resolution")  # Schedule -> Playlist

circuit_stage.add_passes(
    validate_execution_options,
    map_old_operations,
    validate_circuits,
    map_components,
    choose_op_implementations,
    derive_readout_mappings,
)
circuit_resolution_stage.add_passes(resolve_circuits)
timebox_stage.add_passes(
    multiplex_readout,
    prepend_heralding,
    prepend_reset,
)
timebox_resolution_stage.add_passes(resolve_timeboxes)
dd_stage.add_passes(apply_dd_strategy)
schedule_stage.add_passes(
    apply_move_gate_phase_corrections,
    clean_schedule,
)
final_stage.add_passes(build_playlist)

_STANDARD_STAGES = [
    circuit_stage,
    circuit_resolution_stage,
    timebox_stage,
    timebox_resolution_stage,
    dd_stage,
    schedule_stage,
    final_stage,
]


def get_standard_stages(idempotent: bool = True) -> list[CompilationStage]:
    """Get a copy of the standard compilation stages.

    Args:
        idempotent: If True, the passes will be made idempotent.

    Returns:
        list[CompilationStage]: The standard compilation stages.

    """
    stages = deepcopy(_STANDARD_STAGES)
    if idempotent:
        for stage in stages:
            stage.passes = [pass_function_idempotent(f) for f in stage.passes]

    return stages
