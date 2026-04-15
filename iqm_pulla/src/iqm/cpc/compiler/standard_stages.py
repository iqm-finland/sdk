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

1. Map the logical QPU components to physical QPU components. Provided mapping is used, if any.
   Otherwise, identity mapping is used.
2. Choose implementations for circuit operations based on the calibration set.
3. Add additional terminal measurements and/or convert the terminal measurements to the `measure_fidelity` QuantumOp.

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

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence, Set
from dataclasses import replace
import logging
from typing import Any, TypeAlias

from exa.common.data.parameter import Parameter, Setting
from exa.common.data.setting_node import SettingNode
from exa.common.sweep.util import Sweeps, convert_sweeps_to_list_of_tuples
from iqm.cpc.compiler._utils.stages import (
    assert_data_size_safety,
    flatten_subscribed_components,
    get_subscribed_components,
    process_return_parameters,
)
from iqm.cpc.compiler.compilation_stage import DEFAULT_CONTEXT_KEYS
from iqm.cpc.compiler.compiler import (
    CompilationStage,
)
from iqm.cpc.compiler.dd import (
    STANDARD_DD_STRATEGY,
    insert_dd_sequences,
    merge_wait_instructions_in_schedule,
)
from iqm.cpc.compiler.errors import (
    CalibrationError,
    CircuitError,
    UnknownHardwareComponentError,
    UnknownLogicalQubitError,
)
from iqm.cpc.core.config import ComponentGrouping
from iqm.cpc.interface.circuit_execution import Circuit as Circuit_  # something weird with sphinx
from iqm.cpc.interface.circuit_execution import CircuitMetrics, DDStrategy, Locus, MoveGateFrameTrackingMode
from iqm.pulse.base_utils import merge_dicts
from iqm.pulse.builder import CircuitOperation, ScheduleBuilder
from iqm.pulse.gate_implementation import OpCalibrationDataTree
from iqm.pulse.gates import move
from iqm.pulse.gates.measure import ShelvedMeasureTimeBox
from iqm.pulse.playlist import Schedule
from iqm.pulse.playlist.playlist import Playlist
from iqm.pulse.quantum_ops import QuantumOpTable, diff_cal_data
from iqm.pulse.timebox import MultiplexedProbeTimeBox, ProbeBlockBox, ProbeTimeBoxes, SchedulingStrategy, TimeBox
from iqm.station_control.interface.models import RunDefinition, SweepDefinition

_DATA_SIZE_UPPER_LIMIT = 10**8

ACQUISITION_LABEL_KEY = "m{idx}"
ACQUISITION_LABEL = "{qubit}__{key}"
ONLY_PROBE_KEY = "__ONLY_PROBE"
ADDITIONAL_SUBSCRIPTION_KEY = "__ADDITIONAL_SUBSCRIPTION"
HERALDING_KEY = "__HERALD"
RESTRICTED_MEASUREMENT_KEYS = [HERALDING_KEY, ONLY_PROBE_KEY, ADDITIONAL_SUBSCRIPTION_KEY]

# NOTE the buffer duration needs to match all instrument granularities!
# Integer multiples of 80 ns work with 1.8 GHz, 2.0 GHz and 2.4 GHz sample rates and 16 sample granularity,
# which should cover all instruments currently in use. In s.
_BUFFER_GRANULARITY = 80e-9
BUFFER_AFTER_MEASUREMENT_PROBE = 4 * _BUFFER_GRANULARITY
"""Buffer that allows the readout resonator and qubit state to stabilize after a probe pulse, in s.
TODO: not needed after EXA-2089 is done."""

CalibrationErrors: TypeAlias = dict[tuple[str, str, Locus], str]


logger = logging.getLogger(__name__)


def validate_settings(circuits: list[Circuit_], settings: SettingNode) -> list[Circuit_]:  # noqa: ANN201
    """Validate the settings for circuit execution options (only some combinations make sense).

    Raises an error if full MOVE gate tracking is used without move gate validation. Raises a warning if terminal
    measurements would be used with active reset.

    Args:
        circuits: The circuits to compiler.
        settings: The settings tree to validate.

    Returns:
        The circuits as they were.

    Raises:
        CircuitError: if full MOVE gate tracking is used without move gate validation.

    """
    if (
        "validate_circuits" in settings.stages.circuit_stage.subtrees
        and "apply_move_gate_phase_corrections" in settings.stages.schedule_stage.subtrees
    ):
        move_gate_tracking = MoveGateFrameTrackingMode(
            settings.stages.schedule_stage.apply_move_gate_phase_corrections.move_gate_frame_tracking_mode.value
        )
        if (
            move_gate_tracking == MoveGateFrameTrackingMode.FULL
            and not settings.stages.circuit_stage.validate_circuits.move_gate_validation.value
        ):
            raise CircuitError("Full MOVE gate frame tracking requires MOVE gate validation to be turned on.")

    if (
        "subscribe_and_probe" in settings.stages.circuit_stage.subtrees
        and "prepend_reset" in settings.stages.timebox_stage.subtrees
    ):
        if (
            settings.stages.circuit_stage.subscribe_and_probe.convert_terminal_measurements.value
            and settings.stages.timebox_stage.prepend_reset.active_reset_cycles.value
        ):
            settings.stages.circuit_stage.subscribe_and_probe.convert_terminal_measurements = False
            logger.warning(
                "When using active reset, the terminal measurements must also be calibrated to minimize leakage."
                " Thus the terminal measurements cannot be converted to use the fidelity-optimized"
                " `measure_fidelity` operation."
            )

    return circuits


def _map_components_in_instructions(
    instructions: Iterable[CircuitOperation],
    component_mapping: dict[str, str],
    device_components: set[str],
) -> None:
    """Maps the logical component names in a sequence of instructions to the corresponding physical component names.

    Modifies ``instructions`` in place.

    Args:
        instructions: instructions to map.
        component_mapping: Mapping of logical component names to physical component names.
        device_components: names of the physical locus components on the device.

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
        p_component = component_mapping.get(component)
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


def map_components(  # noqa: ANN201
    circuits: Iterable[Circuit_], component_mapping: dict[str, str] | None, context: dict[str, Any]
) -> Iterable[Circuit_]:
    """Maps the logical component names in a sequence of instructions to the corresponding physical component names.

    Modifies ``instructions`` in place. If no mapping is provided, returns the circuits as they are.

    Args:
        circuits: The circuits to compiler.
        component_mapping: Mapping of logical component names to physical component names.
            ``None`` means the identity mapping.
        context: The Compiler context.

    """
    if not component_mapping:
        return circuits

    device = context["builder"].chip_topology
    device_components = device.qubits | device.computational_resonators
    for idx, c in enumerate(circuits):
        try:
            _map_components_in_instructions(
                c.instructions,
                component_mapping,
                device_components=device_components,
            )
        except CircuitError as exc:
            raise CircuitError(f"Circuit {idx}: {exc}") from exc

    return circuits


_READOUT_KEY_MAX_LENGTH = 64


def _get_components_that_can_be_measured(builder: ScheduleBuilder, context: dict[str, Any]) -> set[str]:
    """Get components that are not excluded and can be measured.

    We check for calibration errors for certain very basic operations, and if none are encountered,
    we consider the component measurable. The same function is called multiple times in the default
    compilation loop, so we cache the result in the context.
    """
    if "components_that_can_be_measured" in context:
        return context["components_that_can_be_measured"]
    components = set()
    for component in builder.chip_topology.qubits:
        # FIXME: there should be a better, high-level method for the passes to be aware of the excluded components
        # We should not be needed to check for calibration errors.
        for op in ("measure", "reset_wait"):
            impl = builder.op_table[op].get_default_implementation_for_locus((component,))
            impl_class = builder.op_table[op].implementations[impl]
            impl_cal_data = builder.calibration.get(op, {}).get(impl, {}).get((component,), {})
            error = diff_cal_data(impl_cal_data, impl_class.parameters, impl_class.optional_calibration_keys())
            if error is not None:
                break
        else:
            components.add(component)
    context["components_that_can_be_measured"] = components
    return components


def subscribe_and_probe(  # noqa: PLR0912
    circuits: list[Circuit_],
    settings: SettingNode,
    context: dict[str, Any],
    additionally_subscribed_components: list[str],
    additionally_probed_components: list[str],
    probe_all: bool = True,
    convert_terminal_measurements: bool = True,
) -> list[Circuit_]:
    """Add additional terminal measurements to the circuit and modify measurement instruction arguments.

    The additional measurements can be subscribed to (i.e. they'd return measurement data) or be just probe pulses
    for potentially improving the terminal measurement fidelity in case the measurement calibration is not 100%
    factorizable.

    In addition, the pass hashes all readout keys since there is a data processing limit for the readout key length.
    The keys should then be unmapped in the return data post-processing. The terminal measurements can also be converted
    to the ``measure_fidelity`` operation which is calibrated to maximize the fidelity while not necessarily being
    projective (QNDness is typically not important for the terminal measurement).

    Args:
        circuits: The circuits to compile.
        settings: The settings tree.
        context: The Compiler context.
        additionally_subscribed_components: Additional components to measure in the terminal measurement (besides the
            ones explicitly measured in the circuit itself).
        additionally_probed_components: Additional components to send the probe pulse to besides the
            ones explicitly measured in the circuit itself). The measurement data will not be collected from these
            components.
        probe_all: Whether to send to probe pulse to all components in the terminal measurement (overrides
            ``additionally_probed_components``).
        convert_terminal_measurements: Whether to convert the terminal measurement data to the ``measure_fidelity``
            operation that is calibrated to maximize the fidelity while not necessarily being QND. This option will
            be turned to ``False`` automatically if active reset is used (active reset is not reliable in the presence
            of leakage).

    Returns:
        The circuits with the aforementioned modifications.

    """
    builder = context["builder"]
    mapped_keys: dict[str, str] = {}
    if (
        convert_terminal_measurements
        and "timebox_stage" in settings.stages.subtrees
        and "prepend_reset" in settings.stages.timebox_stage.subtrees
        and settings.stages.timebox_stage.prepend_reset.active_reset_cycles.value is not None
    ):
        convert_terminal_measurements = False
        logger.warning(
            "When using active reset, the terminal measurements must also be calibrated to minimize leakage."
            " Thus the terminal measurements cannot be converted to use the fidelity-optimized"
            " `measure_fidelity` operation."
        )

    components_that_can_be_measured = _get_components_that_can_be_measured(builder, context)
    if "subscribed_groups" in settings.stages.subtrees and settings.stages.subscribed_groups.settings:
        # counter readout overrides additionally_subscribed_components
        grouped_components: set[str] = set()
        for group in settings.stages.subscribed_groups.settings:
            group_components = settings.stages.subscribed_groups[group].value
            grouped_components.update(group_components)
        additionally_subscribed_components = list(grouped_components)

    subscribe_also = frozenset(additionally_subscribed_components)
    probed = frozenset(additionally_probed_components) if not probe_all else components_that_can_be_measured
    final_components_to_measure = subscribe_also.union(probed)

    modified_circuits: list[Circuit_] = []
    for idx, orig_c in enumerate(circuits):
        final_measurements: list = []  # measurement instructions that are final
        eclipsed_qubits: set[str] = set()  # qubits which already have had an instruction on them
        c = replace(orig_c)
        try:
            # total_measurement_instructions = sum(1 for inst in c.instructions if inst.name == "measure")
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
                    # Database has limitations for the length of the measurement keys, so replace user-given
                    # keys with something short and unique across all the measurements in the circuit.
                    if len(key) >= _READOUT_KEY_MAX_LENGTH:
                        mapped_key = str(hash(key))
                        mapped_keys[mapped_key] = key
                        inst.args["key"] = mapped_key

                    if not set(inst.locus) & eclipsed_qubits:
                        # measurement is final iff none of its qubits is eclipsed
                        final_measurements.append(inst)
                eclipsed_qubits.update(inst.locus)

            if convert_terminal_measurements:
                final_measurements_name = "measure_fidelity"
                for measurement in final_measurements:
                    if measurement.name != final_measurements_name:
                        measurement.name = final_measurements_name
                        measurement.implementation = None
            else:
                final_measurements_name = "measure"
            # Add a single final measurement instruction for qubits that still need to be measured
            # (order does not matter). NOTE we assume that iqm-pulse is clever enough to
            # combine the separate final measurement instructions into simultaneous ReadoutTriggers.
            if missing := final_components_to_measure - {q for inst in final_measurements for q in inst.locus}:
                should_be_subscribed = missing.intersection(subscribe_also)
                should_be_probed = missing.intersection(probed) - should_be_subscribed
                if should_be_subscribed:
                    c.instructions += (
                        CircuitOperation(final_measurements_name, tuple(missing), {"key": ADDITIONAL_SUBSCRIPTION_KEY}),
                    )
                if should_be_probed:
                    c.instructions += (
                        CircuitOperation(final_measurements_name, tuple(missing), {"key": ONLY_PROBE_KEY}),
                    )
        except (CircuitError, ValueError) as exc:
            raise CircuitError(f"Circuit {idx}: {exc}") from exc
        modified_circuits.append(c)
    context["mapped_readout_keys"] = mapped_keys
    return modified_circuits


def _get_calibration_set_errors(calibration: OpCalibrationDataTree, ops: QuantumOpTable) -> CalibrationErrors:
    """Validate quantum operation calibration data against the known quantum operations.

    NOTE: calibration data parameters that have a defined default value are not required to be in the calibration data.

    Args:
        calibration: quantum operation calibration data tree to validate
        ops: known quantum operations and their implementations

    Returns:
        Mapping from op name, implementation name, locus to an error string or a None, if there are no errors for
        that locus.

    """
    errors: CalibrationErrors = {}

    def add_error(msg: str) -> None:
        """Add an error message for the current (op, impl, locus) triplet."""
        errors[(op_name, impl_name, locus)] = f"{op_name}.{impl_name} at {locus}: {msg}"

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
                error = diff_cal_data(
                    merge_dicts(default_cal_data, cal_data), impl.parameters, impl.optional_calibration_keys()
                )
                if error is not None:
                    add_error(error)
                    continue

                n_components = len(locus)
                arity = op.arity
                if arity == 0:
                    if n_components != 1:
                        add_error(
                            "for arity-0 operations, calibration data must be provided for single-component loci only"
                        )
                elif n_components != arity:
                    add_error(f"locus must have {arity} component(s)")

    return errors


def _get_circuit_metrics(
    circuit: Circuit_, builder: ScheduleBuilder, calibration_errors: CalibrationErrors
) -> CircuitMetrics:
    """Get circuit metrics for a circuit.

    Args:
        circuit: The circuit to get metrics for.
        builder: encapsulates the known instructions and gate calibration data
        calibration_errors: Found errors for each OIL.

    Returns:
        circuit metrics for ``instructions``

    """
    circuit_components: set[str] = set()
    circuit_component_pairs: set[tuple[str, str]] = set()
    circuit_gate_loci: dict[str, dict[str, Counter[Locus]]] = {}
    for inst in circuit.instructions:
        op = builder.op_table[inst.name]
        try:
            inst.implementation, locus = builder._pick_implementation_and_locus(op, inst.implementation, inst.locus)
        except ValueError as exc:
            raise CircuitError(f"{exc}") from exc
        if (error := calibration_errors.get((inst.name, inst.implementation, locus))) is not None:
            all_errors = "\n- ".join(calibration_errors.values())
            raise CalibrationError(f"Inconsistencies in calibration set:\n- {all_errors}\n\nCause of error:\n{error}")
        inst.locus = locus
        circuit_components.update(locus)
        if op.arity == 2:
            circuit_component_pairs.add(locus)  # type: ignore[arg-type]
        circuit_gate_loci.setdefault(inst.name, {}).setdefault(inst.implementation, Counter()).update([locus])

    return CircuitMetrics(
        components=frozenset(circuit_components),
        component_pairs_with_gates=frozenset(circuit_component_pairs),
        gate_loci=circuit_gate_loci,
    )


def validate_circuits(
    circuits: list[Circuit_],
    builder: ScheduleBuilder,
    context: dict[str, Any],
    move_gate_validation: bool = True,
    validate_prx: bool = True,
    validate_calset: bool = False,
) -> list[Circuit_]:
    """Validate circuits and aggerate metrics data from them.

    Args:
        circuits: The circuits to compile.
        builder: The ScheduleBuilder.
        context: The compiler context.
        move_gate_validation: Whether to do the move gate validation.
        validate_prx: Whether to do the validation.
        validate_calset: Whether to validate the calibration set (if the calibration point used is not from a
            calibration set, this validation might not make sense).

    """
    if validate_calset:
        calibration_errors = _get_calibration_set_errors(builder.calibration, builder.op_table)
    else:
        calibration_errors = {}

    circuit_metrics = []
    components_set = set()  # type:ignore[var-annotated]
    modified_circuits: list[Circuit_] = []
    for idx, c_orig in enumerate(circuits):
        c = replace(c_orig)
        try:
            # circuit part of the metrics describes the circuit as given in the execution request
            metrics = _get_circuit_metrics(c, builder, calibration_errors)
            components_set.update(metrics.components)
            circuit_metrics.append(metrics)
            if "move" in metrics.gate_loci and move_gate_validation:
                # only run the MOVE passes if we need to
                move.validate_move_instructions(
                    c.instructions,
                    builder,
                    validate_prx=validate_prx,
                )

        except (CircuitError, ValueError) as exc:
            raise CircuitError(f"Circuit {idx}: {exc}") from exc
        modified_circuits.append(c)
    if context["components"] is None:
        chip_topology = context["chip_topology"]
        all_components_sorted = chip_topology.qubits_sorted
        context["components"] = ComponentGrouping([c for c in all_components_sorted if c in components_set])
    context["circuit_metrics"] = tuple(circuit_metrics)

    return modified_circuits  # type:ignore[return-value]


# CIRCUIT RESOLUTION PASS
def resolve_circuits(
    circuits: list[Circuit_], builder: ScheduleBuilder, scheduling_strategy: str = "ASAP"
) -> list[TimeBox]:
    """Resolve the circuits to timeboxes.

    Args:
        circuits: The circuit to resolve.
        builder: The schedule builder.
        scheduling: The scheduling strategy to be used in the resolved TimeBoxes (see :class:`.TimeBox`).

    Returns:
        List of TimeBoxes (one TimeBox per circuit).

    """
    scheduling_enum = SchedulingStrategy(scheduling_strategy)
    if isinstance(circuits[0], TimeBox):
        return circuits

    timeboxes = []
    for idx, c in enumerate(circuits):
        try:
            timeboxes.append(builder.circuit_to_timebox(c.instructions, name=c.name, scheduling=scheduling_enum))  # type: ignore[union-attr]
        except ValueError as exc:
            raise CircuitError(f"Circuit {idx}: {exc}") from exc
    return timeboxes


# TIMEBOX-LEVEL PASSES


def _merge_multiplexed_timeboxes(circuit_box: TimeBox) -> TimeBox:
    """Merge any MultiplexedProbeTimeBoxes inside a TimeBox representing a circuit."""

    # TODO make this a timebox-level compiler pass, maybe move it to iqm-pulse, make able to discover more
    # mergable cases like for example [M(1), U(1), M(2)] -> [M(1, 2), U(1)]
    # Consider implementing this step completely differently. This seems very fragile.
    def disjoint_boxes(box1: TimeBox | list[TimeBox], box2: TimeBox | list[TimeBox]) -> bool:
        if isinstance(box1, list):
            box1 = box1[0]
        if isinstance(box2, list):
            box2 = box2[0]
        if not box1.locus_components or not box2.locus_components:
            if box1.neighborhood_components.get(0, set()).intersection(box2.neighborhood_components.get(0, set())):
                return False
        return len(box1.locus_components & box2.locus_components) == 0

    def append_pending(placed_boxes: list[TimeBox], pending: TimeBox | list[TimeBox]) -> None:
        """Append pending measure box or list of boxes (ProbeTimeBoxes) into the already placed boxes."""
        if isinstance(pending, TimeBox):
            placed_boxes.append(pending)
        else:
            placed_boxes.extend(pending)

    placed_boxes: list[TimeBox] = []
    pending = None
    for box_idx, gate_box in enumerate(circuit_box.children):
        if gate_box.children and isinstance(gate_box.children[0], (MultiplexedProbeTimeBox, ShelvedMeasureTimeBox)):
            next_box = circuit_box.children[box_idx + 1] if box_idx < len(circuit_box.children) - 1 else None
            # check if the next box is ProbeBlockBox which means the measure box came from Fast_Measure
            # FIXME: this is hacky, the multiplexing approach needs clean-up once Fast_Measure becomes the default
            to_be_multiplexed = (
                ProbeTimeBoxes([gate_box.children[0], next_box])
                if isinstance(next_box, ProbeBlockBox)
                else gate_box.children[0]
            )
            if pending:
                if disjoint_boxes(pending, gate_box):
                    # Pending box and new candidate have disjoint loci, merge is possible.
                    pending = pending + to_be_multiplexed  # type: ignore[assignment]
                    continue
                # Pending box collides with the new candidate, so we must place it immediately and continue with
                # the new candidate.
                append_pending(placed_boxes, pending)
            pending = to_be_multiplexed
        else:
            if isinstance(gate_box, ProbeBlockBox):
                # ProbeBlockBoxes are always preceded by a measure box, in which case they have already been
                # multiplexed above, so we can ignore them here.
                continue
            if pending and not disjoint_boxes(pending, gate_box):
                # If the pending box touches the same locus components as this gate, we need to place it now.
                append_pending(placed_boxes, pending)
                pending = None
            placed_boxes.append(gate_box)  # type: ignore[arg-type]

    if pending:
        append_pending(placed_boxes, pending)
    return TimeBox.composite(
        placed_boxes,
        label=circuit_box.label,
        scheduling=circuit_box.scheduling,
        scheduling_algorithm=circuit_box.scheduling_algorithm,
    )


def multiplex_readout(timeboxes: list[TimeBox], timebox_input: bool) -> list[TimeBox]:  # noqa: ANN201
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

    This stage pass is skipped if the circuits to be compiled were given in the TimeBox-level, as the logic within does
    not work for deep recursive TimeBoxes.

    Args:
        timeboxes: Timeboxes representing circuits.
        timebox_input: Whether the circuits were inputted already in the TimeBox format.


    Returns:
        New TimeBoxes with the same content, except with some MultiplexedProbeTimeBoxes merged.

    """
    if timebox_input:
        return timeboxes
    return [_merge_multiplexed_timeboxes(circuit_box) for circuit_box in timeboxes]


def resolve_timeboxes(timeboxes: list[TimeBox], builder: ScheduleBuilder, neighborhood: int = 1) -> list[Schedule]:
    """Resolve the timeboxes to schedules.

    Args:
        timeboxes: TimeBoxes representing circuits.
        builder: The ScheduleBuilder.
        neighborhood: The neighborhood for the scheduling (see: :meth:`.ScheduleBuilder.resolve_timebox`).

    Returns:
        The time-resolved schedules (one for each circuit).

    """
    schedules = [builder.resolve_timebox(box, neighborhood=neighborhood) for box in timeboxes]
    return schedules


def _get_active_components(
    circuit_idx: int,
    context: dict[str, Any],
    *,
    filter_set: set[str] | None = None,
) -> Set[str]:
    """Determine QPU components whose controllers will have instructions in the final Schedule.

    Args:
        circuit_idx: Index of the circuit/timebox/schedule in the batch.
        context: Compiler context.
        filter_set: If not None, return only QPU components which also appear in this set.

    Returns:
        Active QPU components of ``circuit_idx``.

    """
    if "circuit_metrics" in context:  # only circuit-level input has this for now
        circuit_metrics = context["circuit_metrics"]
        components = circuit_metrics[circuit_idx].components
    else:
        # union of all the active components in all circuits/timeboxes (we cannot do better ATM)
        components = frozenset(context["components"].flatten())
    if filter_set is None:
        return components
    return filter_set.intersection(components)


def prepend_heralding(
    timeboxes: list[TimeBox],
    builder: ScheduleBuilder,
    context: dict[str, Any],
    add_heralding: bool = False,
) -> list[TimeBox]:
    """Add the heralding measurement TimeBox to all circuits (locus: active components that can be measured).

    Args:
        timeboxes: Timeboxes representing circuits.
        builder: The ScheduleBuilder.
        context: Compiler context.
        add_heralding: Whether to add the heralding measurement timebox to the schedules.

    Returns:
        The timeboxes with the prepended heralding measurement.

    """
    if not add_heralding:
        return timeboxes

    components_that_can_be_measured = _get_components_that_can_be_measured(builder, context)
    heralded_timeboxes = []
    for circuit_idx, box in enumerate(timeboxes):
        # locus order does not matter, since we only care if all herald measurements yield a zero
        herald_locus = tuple(_get_active_components(circuit_idx, context, filter_set=components_that_can_be_measured))
        heralded_timeboxes.append(
            builder.get_implementation("measure", herald_locus)(key=HERALDING_KEY)
            # TODO is this wait already in measure?
            + builder.wait(herald_locus, BUFFER_AFTER_MEASUREMENT_PROBE)
            | box
        )
    return heralded_timeboxes


def prepend_reset(
    timeboxes: list[TimeBox],
    builder: ScheduleBuilder,
    context: dict[str, Any],
    active_reset_cycles: int | None = None,
) -> list[TimeBox]:
    """Add a reset timebox to all circuits for all active components.

    Args:
        timeboxes: TimeBoxes representing circuits.
        builder: The ScheduleBuilder.
        context: The compiler context.
        active_reset_cycles: Number of active reset cycles applied. `None` means no active reset cycles, in which case
            reset is done by relaxation (waiting).

    Returns:
        The timeboxes with the prepended reset.

    """
    if active_reset_cycles is not None and active_reset_cycles <= 0:
        return timeboxes
    if "reset_wait" not in builder.calibration:  # backwards compatibility
        return timeboxes
    new_boxes: list[TimeBox] = []
    components_that_can_be_measured = _get_components_that_can_be_measured(builder, context)
    for circuit_idx, box in enumerate(timeboxes):
        if active_reset_cycles is None:
            reset_components = tuple(_get_active_components(circuit_idx, context))
            reset_box = TimeBox.composite([builder.get_implementation("reset_wait", reset_components)()])
        else:
            reset_components = tuple(
                _get_active_components(circuit_idx, context, filter_set=components_that_can_be_measured)
            )
            reset_box = TimeBox.composite(
                [builder.get_implementation("reset", reset_components)()] * active_reset_cycles
            )
        new_boxes.append(reset_box | box)
    return new_boxes


# SCHEDULE-LEVEL PASSES
def apply_dd_strategy(  # noqa: PLR0913
    schedules: list[Schedule],
    builder: ScheduleBuilder,
    context: dict[str, Any],
    dd_is_disabled: bool = True,
    use_standard_dd_strategy: bool = True,
    DDStrategy_merge_contiguous_waits: bool = True,
    DDStrategy_target_qubits: list[str] | None = None,
    DDStrategy_skip_leading_wait: bool = True,
    DDStrategy_skip_trailing_wait: bool = True,
    DDStrategy_gate_sequences_ratio: list[int] | None = None,
    DDStrategy_gate_sequences_gate_pattern_xy: list[str] | None = None,
    DDStrategy_gate_sequences_align: list[str] | None = None,
) -> list[Schedule]:
    """Insert dynamical decoupling sequences into the schedules, if dynamical decoupling is enabled.

    DDStrategy can also be read from the Compiler context, from under the key `"DDStrategy"`. In this case, the
    strategy provided will override the DDStrategy options given as args to this function.

    Args:
        schedules: Schedules representing the compiled circuits.
        builder: The ScheduleBuilder.
        dd_is_disabled: Set to ``False`` to enable dynamical decoupling.
        use_standard_dd_strategy: Whether to use the standard decoupling strategy (overrides the below arguments).
        DDStrategy_target_qubits: The qubits to which DD is applied (``None`` means apply to every applicable qubit).
        DDStrategy_merge_contiguous_waits: Whether to merge contiguous waits (see :class:`.DDStrategy`).
        DDStrategy_skip_leading_wait: Whether to skip leading waits (see :class:`.DDStrategy`).
        DDStrategy_skip_trailing_wait: Whether to skip trailing waits (see :class:`.DDStrategy`).
        DDStrategy_gate_sequences_ratio: Minimal durations for the Wait to be replaced with the DD sequence
            (in PRX gate durations) in the DD sequence.
        DDStrategy_gate_sequences_gate_pattern_xy: XY Gate patterns in the DD sequence. If you want to provide custom
            PRX angles instead of XY patterns, you must provide the DDStrategy in the Compiler context.
        DDStrategy_gate_sequences_align: Alignments in the DD sequence ("asap", "alap" or "center")

    Returns:
        THe schedules where applicable Waits are replaced with DD sequences.

    """
    if dd_is_disabled:  # Skip the pass and move on
        return schedules

    if "DDStrategy" in context:
        strategy = context["DDStrategy"]
    elif use_standard_dd_strategy:
        strategy = STANDARD_DD_STRATEGY
    else:
        # construct a custom DD strategy
        gate_sequences = list(
            zip(
                DDStrategy_gate_sequences_ratio,  # type:ignore[arg-type]
                DDStrategy_gate_sequences_gate_pattern_xy,  # type:ignore[arg-type]
                DDStrategy_gate_sequences_align,  # type:ignore[arg-type]
            )
        )
        strategy = DDStrategy(
            merge_contiguous_waits=DDStrategy_merge_contiguous_waits,
            target_qubits=DDStrategy_target_qubits,  # type:ignore[arg-type]
            skip_leading_wait=DDStrategy_skip_leading_wait,
            skip_trailing_wait=DDStrategy_skip_trailing_wait,
            gate_sequences=gate_sequences,  # type:ignore[arg-type]
        )

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


def apply_move_gate_phase_corrections(
    schedules: list[Schedule],
    builder: ScheduleBuilder,
    context: dict[str, Any],
    move_gate_frame_tracking_mode: str = "full",
) -> list[Schedule]:
    """Apply calibrated phase corrections if MOVE gates are used."""

    def _circuit_has_moves(circuit_idx: int, context: dict[str, Any]) -> bool:
        if "circuit_metrics" in context:  # circuit level job
            return "move" in context["circuit_metrics"][circuit_idx].gate_loci
        # in TimeBox-level jobs, we have the same components in the whole batch
        for component in context["components"].flatten():
            if component in context["chip_topology"].computational_resonators:
                return True
        return False

    processed_schedules = []
    mgftm_enum = MoveGateFrameTrackingMode(move_gate_frame_tracking_mode)
    for circuit_idx, schedule in enumerate(schedules):
        if _circuit_has_moves(circuit_idx, context) and mgftm_enum != MoveGateFrameTrackingMode.NONE:
            # only run the MOVE passes if we need to
            try:
                schedule = move.apply_move_gate_phase_corrections(  # noqa: PLW2901
                    schedule,
                    builder,
                    apply_detuning_corrections=mgftm_enum != MoveGateFrameTrackingMode.NO_DETUNING_CORRECTION,
                )
            except ValueError as exc:
                raise CircuitError(
                    "Unable to determine MOVE gate phase correction, consider using MOVE gate validation or turn "
                    + f"off MOVE gate frame tracking: {exc}"
                ) from exc

        processed_schedules.append(schedule)

    return processed_schedules


def clean_schedule(schedules: list[Schedule], builder: ScheduleBuilder) -> list[Schedule]:
    """Remove non-functional instructions from `schedules`."""
    return [builder._finish_schedule(schedule) for schedule in schedules]


# FINAL PASSES -- not sweepable


def _complete_circuit_metrics(context: dict[str, Any], schedules: list[Schedule]) -> None:
    """Add schedule durations and minimum execution times to ``context["circuit_metrics"]``."""
    circuit_metrics: Iterable[CircuitMetrics] = context["circuit_metrics"]
    settings = context["settings"]
    # Assumes all channels have the same sampling rate
    channel = next(iter(context["builder"].channels.values()))
    # fill in the schedule durations to the metrics
    for metrics, schedule in zip(circuit_metrics, schedules):
        metrics.schedule_duration = channel.duration_to_seconds(schedule.duration)
        shots = settings.controllers.options.playlist_repeats.value
        end_delay = settings.controllers.options.end_delay.value
        # lower bound on the actual execution time: schedule duration + reset
        # does not include the heralding measurement
        metrics.min_execution_time = shots * (metrics.schedule_duration + end_delay)


def build_playlist_and_merge_contexts(
    schedules: list[Schedule],
    builder: ScheduleBuilder,
    context: dict[str, Any],
    compute_execution_time_metrics: bool = False,
) -> Playlist:
    """Build the playlist from the schedules and merge the contexts for individual sweep spots.

    When merging the context, the active components are the union of the active components in each sweep spot
    (unless explicitly given by the user). The default context keys are not merged (these should not be modified
    by any of the sweep spots), and any other keys in the context will be merged to mapping from the sweep spot id
    to the context entry.

    Args:
        schedules: Schedules to build into a Playlist.
        builder: The ScheduleBuilder.
        context: The Compiler context.
        compute_execution_time_metrics: Whether to compute schedule duration and minimum execution time for circuits.

    Returns:
        The Playlist containing schedules.

    """
    # TODO: refactor this function, it's stupid
    playlist, readout_metrics = builder.build_playlist(schedules)  # type:ignore[arg-type]
    context["readout_metrics"] = readout_metrics
    if "mapped_readout_keys" not in context:
        mapped_readout_keys = {}
        for spot_context in context["spot_contexts"].values():
            if "mapped_readout_keys" in spot_context:
                mapped_readout_keys.update(spot_context["mapped_readout_keys"])
        context["mapped_readout_keys"] = mapped_readout_keys
    if "circuit_metrics" not in context:
        circuit_metrics_tmp: list[tuple[CircuitMetrics]] = []
        for spot_context in context["spot_contexts"].values():
            if "circuit_metrics" not in spot_context:  # we cannot gather circuit_metrics from TimeBox-level circuits
                break  # we can only aggregate circuit metrics if every spot produced them
            circuit_metrics_tmp.append(spot_context["circuit_metrics"])
        else:
            if circuit_metrics_tmp:
                # reorder the metrics to conform to the typical sweep ordering
                context["circuit_metrics"] = tuple(
                    spot[i] for i in range(len(circuit_metrics_tmp[0])) for spot in circuit_metrics_tmp
                )
            if compute_execution_time_metrics:
                _complete_circuit_metrics(context, schedules)
    if context["inputted_components"] is None:
        # if not inputted by the user, the components are given by the union of the components in each sweep spot
        # of the batch
        components = []
        for spot_context in context["spot_contexts"].values():
            components.extend([c for c in spot_context["components"] if c not in components])
        context["components"] = ComponentGrouping(components)
    spot_context_values: dict[str, dict[int, Any]] = defaultdict(dict)
    context_keys_set = set(context.keys())
    for spot_idx, spot_context in enumerate(context["spot_contexts"].values()):
        all_keys = set(spot_context.keys()).union(context_keys_set)
        for context_key in all_keys:
            if context_key not in DEFAULT_CONTEXT_KEYS:
                spot_context_values[context_key][spot_idx] = spot_context[context_key]
    for context_key, values in spot_context_values.items():
        context[context_key] = values
    del context["spot_contexts"]
    return playlist


def _map_readout_label_to_implementation(labels_to_impls: dict[str, Sequence[str]]) -> dict[str, str | None]:
    """Map readout labels to "<op>.<implementation>"."""
    readout_label_to_impl: dict[str, str | None] = {
        label: None if len(impls) > 1 else next(iter(impls)) for label, impls in labels_to_impls.items()
    }

    return readout_label_to_impl


def create_run_definition(
    playlist: Playlist,
    context: dict[str, Any],
    data_size_safety_switch: bool = True,
    force_ragged_data: bool = False,
) -> RunDefinition:
    """Create MQE-style RunDefinition.

    Args:
        playlist: The Playlist to create the RunDefinition for.
        context: The Compiler context.
        data_size_safety_switch: Whether to throw an error with exceedingly large return data sizes (set to ``False``
            if you want to still run the job and are sure the DB and/or the stack can handle it).
        force_ragged_data: Whether to force the return data to the sparse ragged format even if the
            dimensions are representable as a cartesian product.

    Returns:
        The RunDefinition.

    """
    readout_metrics = context["readout_metrics"]
    components = context["components"]
    chip_topology = context["chip_topology"]
    settings = context["settings"]

    # filter out unsubscribed labels/keys from the readout metrics
    readout_metrics.filter_out(
        components=["dummy"],  # probe pulse dummy integration should never be subscribed
        labels=settings.stages.unsubscribed_labels.value,
        keys=[ONLY_PROBE_KEY],
    )

    labels_to_impls = readout_metrics.implementations
    # we currently do not support doing complex-mode discrimination for readout labels with multiple
    # measure implementations.

    readout_label_to_impl: dict[str, str | None] = _map_readout_label_to_implementation(labels_to_impls)

    hard_sweeps = context["hard_sweeps"]
    # store the actual hard sweeps (circuit & pulse sweeps) and their order for post-processing
    # these don't include the "meta" sweep dimensions, such as repetitions or counter index
    actual_hard_sweeps = {s.parameter.name: s.data for st in hard_sweeps for s in st}
    actual_parallel_hard_sweeps = {
        st[0].parameter.name: [s.parameter.name for s in st] for st in hard_sweeps if len(st) > 1
    }
    return_parameters: dict[Parameter | Setting, Sweeps | None] = context.get("return_parameters", {})
    subscribed_components = get_subscribed_components(
        components,
        settings,
        readout_metrics,
    )
    groups_without_label = (
        list(subscribed_components.keys()) if isinstance(subscribed_components, dict) else ["all_components"]
    )

    probe_lines = {
        chip_topology.component_to_probe_line[c]
        for c in components.flatten()
        if c in chip_topology.component_to_probe_line
    }
    qubits = [c for c in components.flatten() if c in chip_topology.qubits]
    couplers = [c for c in components.flatten() if c in chip_topology.couplers]
    computational_resonators = [c for c in components.flatten() if c in chip_topology.computational_resonators]
    readout_components, data_parameters, ragged_labels_counts = process_return_parameters(
        return_parameters=return_parameters,
        readout_metrics=readout_metrics,
        settings=context["settings"],
        subscribed_components=subscribed_components,
        hard_sweeps=hard_sweeps,
        force_ragged_data=force_ragged_data,
        chip_topology=context["chip_topology"],
        probe_lines=probe_lines,
    )
    flattened_readout_components = flatten_subscribed_components(readout_components)
    readout_groups = list(readout_components.values())
    readout_group_names = list(readout_components.keys())
    integration_data_parameters = data_parameters.get("integration_data_parameters", [])
    time_trace_data_parameters = data_parameters.get("time_trace_data_parameters", [])
    hard_sweeps = {parameter.name: hard_sweep for parameter, hard_sweep in return_parameters.items()}  # type: ignore[assignment]
    if data_size_safety_switch:
        assert_data_size_safety(_DATA_SIZE_UPPER_LIMIT, hard_sweeps, context["soft_sweeps"])  # type: ignore[arg-type]

    run_definition = RunDefinition(
        run_id=None,  # type:ignore[arg-type] # set when sending for execution
        username="Pulla User",  # TODO
        experiment_name="Pulla",  # TODO
        experiment_label="pulla",  # TODO
        additional_run_properties={
            "qubits": qubits,
            "computational_resonators": computational_resonators,
            "couplers": couplers,
            "probe_lines": {
                c: pl
                for c, pl in chip_topology.component_to_probe_line.items()
                if c in flattened_readout_components and pl in probe_lines
            },
            "components": components.to_json_serializable(),
            "components_grouping_mode": components.grouping_mode.value,  # type: ignore[union-attr]
            "coupler_mapping": {c: list(qs) for c, qs in chip_topology.coupler_to_components.items()},
            "readout_components": flattened_readout_components,
            "readout_groups": readout_groups,
            "readout_group_names": readout_group_names,
            "readout_groups_without_label": groups_without_label,
            "integration_data_parameters": integration_data_parameters,
            "time_trace_data_parameters": time_trace_data_parameters,
            "circuit_generation_function_name": "Foo",  # TODO
            "target_data_parameters": [],
            "ragged_data_labels": ragged_labels_counts,
            "ragged_hard_sweeps": list(actual_hard_sweeps.keys()) if ragged_labels_counts else [],
            "ragged_hard_sweeps_data": list(actual_hard_sweeps.values()) if ragged_labels_counts else [],
            "ragged_parallel_sweeps": actual_parallel_hard_sweeps if ragged_labels_counts else {},
            "readout_label_to_impl": readout_label_to_impl,
        },
        software_version_set_id=context["software_version_set_id"],
        hard_sweeps=hard_sweeps,
        components=qubits + couplers,
        default_data_parameters=integration_data_parameters + time_trace_data_parameters,
        default_sweep_parameters=[],  # TODO
        sweep_definition=SweepDefinition(
            sweep_id=None,  # type:ignore[arg-type] # set when sending for execution
            dut_label=context["dut_label"],
            settings=settings,
            sweeps=context["soft_sweeps"],  # type: ignore[arg-type]
            return_parameters=[parameter.name for parameter in return_parameters],
            playlist=playlist,
        ),
    )

    # FIXME: Typing of sweep_definition.sweeps is wrong, fix
    #  sweep_definition.sweeps should be always "NdSweep", not even temporarily "Sweeps".
    # Validate sweeps parameter, before sending it over to the station control.
    sweep_definition = run_definition.sweep_definition
    sweep_definition.sweeps = convert_sweeps_to_list_of_tuples(sweep_definition.sweeps)  # type: ignore[arg-type]
    for key, hard_sweep in run_definition.hard_sweeps.items():  # type: ignore[union-attr]
        if hard_sweep is not None:
            run_definition.hard_sweeps[key] = convert_sweeps_to_list_of_tuples(hard_sweep)  # type: ignore[index,arg-type]
    run_definition.sweep_definition = sweep_definition
    return run_definition


circuit_stage = CompilationStage(
    name="circuit_stage",
    info="Perform Circuit-level transformations.",
)  # Circuit -> ... -> Circuit
circuit_stage_experiment = CompilationStage(
    name="circuit_stage",
    info="Perform Circuit-level transformations.",
)  # Circuit -> ... -> Circuit
circuit_resolution_stage = CompilationStage(
    name="circuit_resolution",
    info="Resolve Circuits into TimeBoxes.",
)  # Circuit -> TimeBox
timebox_stage = CompilationStage(
    name="timebox_stage",
    info="Perform TimeBox-level transformations.",
)  # TimeBox -> ... -> TimeBox
timebox_resolution_stage = CompilationStage(
    name="timebox_resolution",
    info="Resolve TimeBoxes into Schedules.",
)  # TimeBox -> Schedule
dd_stage = CompilationStage(
    name="dynamical_decoupling",
    info="Apply dynamical decoupling sequences to idle qubits in the Schedules.",
)  # Schedule -> Schedule
schedule_stage = CompilationStage(
    name="schedule_stage",
    info="Perform Schedule-level transformations.",
)  # Schedule -> ... -> Schedule
schedule_stage_experiment = CompilationStage(
    name="schedule_stage",
    info="Perform Schedule-level transformations.",
)
schedule_resolution_stage = CompilationStage(
    name="schedule_resolution",
    info="Translate Schedules into a hardware-executable Playlist.",
)  # Schedule -> Playlist
job_stage = CompilationStage(
    name="job_creation",
    info="Package the Playlist and metadata into a job description to be submitted to IQM Server.",
)

circuit_stage.add_passes(
    validate_settings,
    map_components,
    subscribe_and_probe,
    validate_circuits,
)
circuit_stage_experiment.add_passes(
    validate_settings,
    map_components,
    validate_circuits,
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
schedule_stage_experiment.add_passes(clean_schedule)
schedule_resolution_stage.add_passes(build_playlist_and_merge_contexts)
job_stage.add_passes(create_run_definition)

_STANDARD_CIRCUIT_STAGES = [
    circuit_stage,
]

_STANDARD_CIRCUIT_STAGES_EXPERIMENT = [
    circuit_stage_experiment,
]

_STANDARD_PULSE_STAGES = [
    circuit_resolution_stage,
    timebox_stage,
    timebox_resolution_stage,
    dd_stage,
    schedule_stage,
]

# FIXME: the move phase tracking and readout multiplexing stages not yet compatible with Experiment
_STANDARD_PULSE_STAGES_EXPERIMENT = [
    circuit_resolution_stage,
    timebox_stage,
    timebox_resolution_stage,
    dd_stage,
    schedule_stage_experiment,
]

_STANDARD_FINAL_STAGES = [
    schedule_resolution_stage,
    job_stage,
]
