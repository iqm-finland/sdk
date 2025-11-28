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
"""Tools for building instruction schedules."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import copy
from dataclasses import dataclass, field, replace
import itertools
import logging
from types import MethodType
from typing import Any

from iqm.models.playlist.channel_descriptions import (
    ChannelDescription,
    IQChannelConfig,
    ReadoutChannelConfig,
    RealChannelConfig,
)
import iqm.models.playlist.instructions as sc_instructions  # TODO make SC use iqm.pulse Instructions?
from iqm.models.playlist.segment import Segment as SC_Schedule  # iqm.models names are temporary
from iqm.models.playlist.waveforms import to_canonical

from exa.common.helpers.yaml_helper import load_yaml
from exa.common.qcm_data.chip_topology import ChipTopology, sort_components
from iqm.pulse.base_utils import _dicts_differ, merge_dicts
from iqm.pulse.gate_implementation import (
    CompositeCache,
    GateImplementation,
    Locus,
    OICalibrationData,
    OILCalibrationData,
    OpCalibrationDataTree,
)
from iqm.pulse.gates import _validate_implementation, get_implementation_class
from iqm.pulse.gates.default_gates import _deprecated_implementations, _deprecated_ops, _quantum_ops_library
from iqm.pulse.playlist.channel import ChannelProperties, ProbeChannelProperties
from iqm.pulse.playlist.instructions import (
    AcquisitionMethod,
    ComplexIntegration,
    ConditionalInstruction,
    FluxPulse,
    Instruction,
    IQPulse,
    MultiplexedIQPulse,
    ReadoutMetrics,
    ReadoutTrigger,
    RealPulse,
    ThresholdStateDiscrimination,
    TimeTrace,
    VirtualRZ,
    Wait,
)
from iqm.pulse.playlist.playlist import Playlist
from iqm.pulse.playlist.schedule import Schedule, Segment
from iqm.pulse.quantum_ops import QuantumOp, QuantumOpTable, validate_locus_calibration, validate_op_calibration
from iqm.pulse.scheduler import (
    NONSOLID,
    Block,
    extend_hard_boundary,
    extend_hard_boundary_in_seconds,
    extend_schedule_new,
)
from iqm.pulse.timebox import SchedulingAlgorithm, SchedulingStrategy, TimeBox

logger = logging.getLogger(__name__)


@dataclass
class CircuitOperation:
    r"""Specific quantum operation applied on a specific part of the QPU, e.g. in a quantum circuit.

    We currently support the following native operations for circuit execution:

    ================ =========== ======================================= ===========
    name             # of qubits args                                    description
    ================ =========== ======================================= ===========
    measure          >= 1        ``key: str``, ``feedback_key: str``     Measurement in the Z basis.
    prx              1           ``angle: float``, ``phase: float``      Phased x-rotation gate.
    cc_prx           1           ``angle: float``, ``phase: float``,
                                 ``feedback_qubit: str``,
                                 ``feedback_key: str``                   Classically controlled PRX gate.
    reset            >= 1                                                Reset the qubit(s) to :math:`|0\rangle`.
    cz               2                                                   Controlled-Z gate.
    move             2                                                   Move a qubit state between a qubit and a
                                                                         computational resonator, as long as
                                                                         at least one of the components is
                                                                         in the :math:`|0\rangle` state.
    barrier          >= 1                                                Execution barrier.
    delay            >= 1        ``duration: float``                     Force a delay between circuit operations.
    ================ =========== ======================================= ===========

    For each CircuitOperation you may also optionally specify :attr:`implementation`,
    which contains the name of an implementation of the operation to use.
    Support for multiple implementations is currently experimental and in normal use the
    field should be omitted, this selects the default implementation for the operation for that locus.

    See the submodules under :mod:`iqm.pulse.gates` for more details about each operation.

    """

    name: str
    """name of the quantum operation"""
    locus: Locus
    """names of the information-bearing QPU components (qubits, computational resonators...) the operation acts on"""
    args: dict[str, Any] = field(default_factory=dict)
    """arguments for the operation"""
    implementation: str | None = None
    """name of the implementation"""

    def validate(self, op_table: QuantumOpTable) -> None:
        """Validate the operation against a table of operation definitions.

        Args:
            op_table: table containing allowed quantum operations

        Raises:
            ValueError: operation is not valid

        """
        op_type = op_table.get(self.name)
        if op_type is None:
            message = ", ".join(op_table)
            raise ValueError(f"Unknown operation '{self.name}'. Supported operations are '{message}'.")
        self._validate_implementation(op_type)
        self._validate_locus(op_type)
        self._validate_args(op_type)

    def _validate_implementation(self, op_type: QuantumOp) -> None:
        if self.implementation is not None:
            if not self.implementation:
                raise ValueError("Implementation of the instruction should be None, or a non-empty string")
            if self.implementation not in op_type.implementations:
                raise ValueError(f"Unknown implementation '{self.implementation}' for quantum operation '{self.name}'.")

    def _validate_locus(self, op_type: QuantumOp) -> None:
        arity = op_type.arity
        if (0 < arity) and (arity != len(self.locus)):
            raise ValueError(
                f"The '{self.name}' operation acts on {arity} qubit(s), but {len(self.locus)} were given: {self.locus}."
            )
        if len(self.locus) != len(set(self.locus)):
            raise ValueError(f"Repeated locus components: {self.locus}.")

    def _validate_args(self, op_type: QuantumOp) -> None:
        # Check argument names
        submitted_arg_names = set(self.args)
        allowed_arg_types = op_type.params | op_type.optional_params
        if not set(op_type.params) <= submitted_arg_names:
            raise ValueError(
                f"The operation '{self.name}' requires "
                f"the argument(s) {tuple(op_type.params)}, "
                f"but {tuple(submitted_arg_names)} were given."
            )

        if not submitted_arg_names <= set(allowed_arg_types):
            allowed_arg_names = tuple(allowed_arg_types)
            message = f"the arguments {allowed_arg_names}" if allowed_arg_names else "no arguments"
            raise ValueError(f"The operation '{self.name}' allows {message}, but {submitted_arg_names} were given.")
        # Check argument types
        for arg_name, arg_value in self.args.items():
            allowed_types = allowed_arg_types[arg_name]
            if not isinstance(arg_value, allowed_types):
                raise ValueError(
                    f"The argument '{arg_name}' should be of one of the following supported types"
                    f" {allowed_types}, but ({type(arg_value)}) was given."
                )


def load_config(path: str) -> tuple[QuantumOpTable, OpCalibrationDataTree]:
    """Load quantum operation definitions and calibration data from a YAML config file.

    Args:
        path: path to a YAML config file

    Returns:
        quantum operation definitions, calibration data tree

    """
    yaml = load_yaml(path)
    if (exp := yaml.get("experiment")) is None:
        raise ValueError('The config YAML is missing the "experiment" section.')

    ops = exp.get("gate_definitions", {})
    calibration = exp.get("gates", {})
    return build_quantum_ops(ops), calibration


def validate_quantum_circuit(
    operations: Iterable[CircuitOperation],
    op_table: QuantumOpTable,
    *,
    require_measurements: bool = False,
) -> None:
    """Validate a sequence of circuit operations constituting a quantum circuit.

    Args:
        operations: quantum circuit to be validated
        op_table: table containing allowed/calibrated quantum operations
        require_measurements: iff True the circuit must include at least one measurement operation

    Raises:
        ValueError: ``operations`` do not constitute a valid quantum circuit

    """
    measurement_keys: set[str] = set()
    measured_qubits: set[str] = set()

    for op in operations:
        op.validate(op_table)

        # extra validation for specific operations
        if op.name == "measure":
            if (key := op.args["key"]) in measurement_keys:
                raise ValueError(f"Measurement key '{key}' is not unique.")

            measurement_keys.add(key)
            for qubit in op.locus:
                measured_qubits.add(qubit)

    if require_measurements and not measured_qubits:
        raise ValueError("Circuit contains no measurements.")


def build_quantum_ops(ops: dict[str, dict[str, Any]]) -> QuantumOpTable:
    """Builds the table of known quantum operations.

    Hardcoded canonical ops table is extended by the ones in ``ops``.
    In case of name collisions, the content of ``ops`` takes priority over the defaults,
    with the following caveats:

    * canonical implementation names cannot be redefined, and
    * for canonical operations you can only change :attr:`implementations` and :attr:`defaults_for_locus`.

    Args:
        ops: Contents of the ``gate_definitions`` section defining
            the quantum operations in the configuration YAML file.
            Implementation names must be mapped to either exposed GateImplementation
            class names, or actual GateImplementation classes.
            NOTE: Modified by the function.

    Returns:
        Mapping from quantum operation names to their definitions.

    Raises:
        ValueError: Requested implementation class is not exposed.
        ValueError: A canonical implementation name is being redefined.
        ValueError: Locus default references an undefined implementation.
        ValueError: Operation attributes don't match defaults or are invalid.

    """
    # build the table of native ops
    op_table = {}
    for op_name, op in _quantum_ops_library.items():
        # filter out deprecated ops, unless requested by the user in ``ops``
        if op_name in _deprecated_ops and op_name not in ops:
            continue

        new_op = copy.deepcopy(op)  # to prevent modifications to the hardcoded table
        if (deprecated_impls := _deprecated_implementations.get(op_name)) is not None:
            # filter out deprecated implementations
            non_deprecated = {
                impl_name: impl for impl_name, impl in op.implementations.items() if impl_name not in deprecated_impls
            }
            new_op = replace(op, implementations=non_deprecated)
        op_table[op_name] = new_op

    # add ops defined by the user
    for op_name, op_definition in ops.items():
        # prepare the implementations
        implementations: dict[str, type[GateImplementation]] = {}
        for impl_name, impl_class_def in op_definition.pop("implementations", {}).items():
            if isinstance(impl_class_def, str):
                # check if the impl class name has been exposed
                impl_class_name = impl_class_def
                if (impl_class := get_implementation_class(impl_class_name)) is None:
                    raise ValueError(
                        f"'{op_name}': Requested implementation class '{impl_class_name}' has not been exposed."
                    )
            elif issubclass(impl_class_def, GateImplementation):
                # bit of a hack: also accept GateImplementation classes directly
                impl_class = impl_class_def
                impl_class_name = impl_class.__name__
            else:
                raise ValueError(f"{op_name}: {impl_class_def} is neither a str or a type[GateImplementation].")

            # check if we are overriding a canonical implementation name for this op
            _validate_implementation(op_name, impl_name, impl_class_name)
            implementations[impl_name] = impl_class

        # validate defaults_for_locus
        defaults_for_locus: dict[Locus, str] = op_definition.pop("defaults_for_locus", {})
        for locus, impl_name in defaults_for_locus.items():
            if impl_name not in implementations:
                raise ValueError(
                    f"'{op_name}': defaults_for_locus[{locus}] implementation '{impl_name}' does not "
                    f"appear in the implementations dict."
                )

        if (old_op := _quantum_ops_library.get(op_name)) is not None:
            # modify a canonical operation
            # only some fields can be modified, and they have been popped out already from op_definition
            if op_definition:
                # TODO this should be an error, but there are so many old experiment.yml files in use
                # that still have the old syntax that being strict about this would be disruptive.
                # Now we just ignore the fields you cannot change.
                logger.warning(
                    f"'{op_name}' is a canonical operation, which means the fields {set(op_definition)} "
                    "provided by the user may not be changed."
                )

            op = replace(old_op, implementations=implementations, defaults_for_locus=defaults_for_locus)
        else:
            # entirely new quantum operation defined by the user
            op = QuantumOp(
                name=op_name,
                implementations=implementations,
                defaults_for_locus=defaults_for_locus,
                **op_definition,
            )
        op_table[op_name] = op

    return op_table


class ScheduleBuilder:
    """Builds instruction schedules out of quantum circuits or individual quantum operations.

    Encapsulates known quantum ops, the calibration information for them, QPU components and their
    topology, and controller properties.

    Args:
        op_table: definitions of known quantum ops
        calibration: calibration data tree for the quantum ops
        chip_topology: Chip topology derived from the CHAD.
        channels: mapping of controller names to the configurations of their channels
        component_channels: Mapping from QPU component name to a mapping of ``('drive', 'flux', 'readout')``
            to the name of the control channel responsible for that function of the component.

    """

    def __init__(
        self,
        op_table: QuantumOpTable,
        calibration: OpCalibrationDataTree,
        chip_topology: ChipTopology,
        channels: dict[str, ChannelProperties],
        component_channels: dict[str, dict[str, str]],
    ):
        # TODO: channels and component_channels should be given in a single Protocol here.
        self.op_table = op_table
        self.calibration = calibration
        self.chip_topology = chip_topology
        self.channels = channels
        self.component_channels = component_channels
        self._cache: dict[str, dict[str, dict[Locus, GateImplementation]]] = {}
        """Cached GateImplementations. The tree has the same structure as OpCalibrationDataTree."""
        self.composite_cache = CompositeCache()
        """Cache for the CompositeGate TimeBoxes. Flushed whenever ANY calibration data is injected into the builder.
        """
        self._channel_to_component: dict[str, str] = {
            c: q
            for q, chans in self.component_channels.items()
            for c in chans.values()
            if self.channels[c].blocks_component
        }
        """``self.component_channels`` mapping inverted cached for scheduling algorithm performance. This mapping is
        used in the scheduling to determine the components to block based on their associated channels. Only
        blocking channels are included in this mapping, non-blocking channels (e.g. certain virtual channels) do not
        block their components, just themselves."""
        self._channel_types = {
            "other": [c for c, prop in self.channels.items() if not isinstance(prop, ProbeChannelProperties)],
            "probe": [c for c, prop in self.channels.items() if isinstance(prop, ProbeChannelProperties)],
        }
        """Cache the probe and non-probe channel names for the scheduling algorithm performance"""
        self._require_scheduling_in_seconds: bool = len(
            {c.sample_rate for c in self.channels.values() if not c.is_virtual}
        ) > 1 or {
            self.channels[c].instruction_duration_granularity
            for c in self._channel_types["other"]
            if not self.channels[c].is_virtual
        } != {
            self.channels[c].instruction_duration_granularity
            for c in self._channel_types["probe"]
            if not self.channels[c].is_virtual
        }
        """Whether to require scheduling of probe instructions in seconds instead of in samples. This can happen for two
        reasons: 1) the probe channel has a different sampling rate to the other channels (e.g. with UHFQA) or
        2) the probe channels have a different instruction granularity to some of the other channels (e.g. with
        mixed stations that have the RO device from a different vendor than some of the AWGs)."""
        self._channel_templates: dict[str, ChannelProperties | None] = {
            "other": next((c for k, c in self.channels.items() if k in self._channel_types["other"]), None),
            "probe": next((c for k, c in self.channels.items() if k in self._channel_types["probe"]), None),
        }
        """Cache representative channel properties for a probe and a non-probe channel for the scheduling algorithm
        performance."""

        for op_name in op_table:
            self._set_gate_implementation_shortcut(op_name)

        self._logger = logging.getLogger(__name__)

    def __getitem__(self, item: str) -> GateImplementation:
        """If ``item`` is an operation name in ``self.op_table``, returns the shortcut method ``self.<op_name>``"""
        if item in self.op_table:
            return getattr(self, item)
        raise ValueError(f"No operation found with the name {item} in ``self.op_table``.")

    def inject_calibration(self, partial_calibration: OpCalibrationDataTree) -> None:
        """Inject new calibration data, changing :attr:`calibration` after initialisation.

        Invalidates the GateImplementation caches for the affected operations/implementations/loci. Also invalidates
        the cache for any factorizable gate implementation, if any of its locus components was affected.

        Args:
            partial_calibration: data to be injected. Must have the same structure as :attr:`calibration` but does not
                have to contain all operations/implementations/loci/values. Only the parts of the data that are
                found will be merged into :attr:`calibration` (including any ``None`` values). :attr:`_cache` will
                be invalidated for the found operations/implementations/loci and only if the new calibration data
                actually differs from the previous.

        """
        # composite gates are always flushed (though we could only flush the ones whose member gate cal is changed!)
        self.composite_cache.flush()
        # merge the calibration changes
        for op, op_data in partial_calibration.items():
            for impl, impl_data in op_data.items():
                for locus, locus_data in impl_data.items():
                    prev_calibration = self.calibration[op][impl][locus]
                    new_calibration = merge_dicts(prev_calibration, locus_data)
                    self.calibration[op][impl][locus] = new_calibration
                    if (
                        op in self._cache
                        and impl in self._cache[op]
                        and locus in self._cache[op][impl]
                        and _dicts_differ(prev_calibration, new_calibration)
                    ):
                        # invalidate only the affected GateImplementations
                        del self._cache[op][impl][locus]
                        if self.op_table[op].factorizable:
                            # factorizable ops only have cal data for single-component loci,
                            # but we also need to flush all loci that include the single-component locus
                            locus_component = locus[0]
                            # dict size cannot change while you iterate over it, hence the list of keys
                            for cached_locus in list(self._cache[op][impl]):
                                if locus_component in set(cached_locus):
                                    del self._cache[op][impl][cached_locus]

    def validate_calibration(self) -> None:
        """Check that the calibration data matches the known quantum operations.

        Raises:
            ValueError: there is something wrong with the calibration data

        """
        validate_op_calibration(self.calibration, self.op_table)

    def get_drive_channel(self, component: str) -> str:
        """Drive channel for the given QPU component.

        Args:
            component: name of a QPU component
        Returns:
            Name of the drive channel for ``component``, if it exists.

        Raises:
            KeyError: if component does not exist or does not have a drive channel

        """
        return self._get_channel_for_component(component, "drive")

    def get_flux_channel(self, component: str) -> str:
        """Flux channel for the given QPU component.

        See :meth:`.get_drive_channel`.
        """
        return self._get_channel_for_component(component, "flux")

    def get_probe_channel(self, component: str) -> str:
        """Probe line channel for the probe line ``component`` belongs to.

        See :meth:`.get_drive_channel`.

        Args:
            component: name of a QPU component (typically qubit) to probe

        """
        probe_line = self.chip_topology.component_to_probe_line.get(component, None)
        if probe_line is None:
            raise KeyError(f'probe line not found for component "{component}"')
        return self._get_channel_for_component(probe_line, "readout")

    def get_virtual_feedback_channels(self, component: str) -> list[str]:
        """All virtual feedback signal channels for the given QPU component.

        A virtual feedback channel between a source and a destination exists if the station configuration allows it.
        `component` can be either the source or the destination of the signal.

        Args:
            component: name of a QPU component
        Returns:
            Names of the virtual channels.

        """
        return [
            channel_name
            for operation, channel_name in self.component_channels[component].items()
            if operation.startswith("feedback")
        ]

    def get_virtual_feedback_channel_for(self, awg_name: str, feedback_qubit: str) -> str:
        """Get virtual feedback channel for feedback to a given AWG from a given probe line.

        Args:
            awg_name: name of the awg node that receives the feedback bit.
            feedback_qubit: which qubit's measurement resulted in the feedback bit

        Returns:
            The virtual feedback channel name.

        Raises:
            ValueError: if the given AWG does not support fast feedback from the given probe line.

        """
        probe_line = self.chip_topology.component_to_probe_line.get(feedback_qubit)
        all_virtual_channels = self.get_virtual_feedback_channels(probe_line)  # type: ignore[arg-type]
        channel = next((c for c in all_virtual_channels if awg_name in c), None)
        if not channel:
            raise ValueError(f"AWG node {awg_name} does not support fast feedback from {probe_line}")
        return channel

    def _get_channel_for_component(self, component: str, operation: str) -> str:
        """Control channel name for the given QPU component and operation.

        Returns:
            name of the channel
        Raises:
            KeyError: if component does not exist or does not have the operation

        """
        if (
            self.component_channels is None
            or component not in self.component_channels
            or operation not in self.component_channels[component]
        ):
            raise KeyError(f'channel not found for component "{component}" and operation "{operation}"')
        return self.component_channels[component][operation]

    def has_calibration(
        self,
        op_name: str,
        impl_name: str,
        locus: Locus,
    ) -> bool:
        """Is there calibration data for the given quantum operation, implementation and locus?

        Args:
            op_name: name of the quantum operation
            impl_name: name of the implementation
            locus: locus of the operation

        Returns:
            True iff requested calibration data was found

        """
        if (op_cal := self.calibration.get(op_name)) is None:
            return False
        if (impl_cal := op_cal.get(impl_name)) is None:
            return False
        if locus not in impl_cal:
            # if a locus node is missing entirely, is not replaced with the defaults
            return False
        return True

    def get_calibration(
        self,
        op_name: str,
        impl_name: str,
        locus: Locus,
    ) -> OILCalibrationData:
        """Calibration data for the given quantum operation, implementation and locus.

        Args:
            op_name: name of the quantum operation
            impl_name: name of the implementation
            locus: locus of the operation

        Returns:
            requested calibration data

        Raises:
            ValueError: requested calibration data was not found

        """
        if (op_cal := self.calibration.get(op_name)) is None:
            raise ValueError(f"No calibration data for op '{op_name}'.")
        if (impl_cal := op_cal.get(impl_name)) is None:
            raise ValueError(f"No calibration data for '{op_name}.{impl_name}'.")
        if (cal_data := impl_cal.get(locus)) is None:
            # if a locus node is missing entirely, is not replaced with the defaults
            raise ValueError(f"No calibration data for '{op_name}.{impl_name}' at {locus}.")

        default_cal_data = impl_cal.get((), {})  # empty locus contains the default cal data for all the loci
        return merge_dicts(default_cal_data, cal_data, merge_nones=False)

    def get_control_channels(
        self,
        locus: Iterable[str],
    ) -> tuple[str, ...]:
        """Control channels that directly affect quantum operations at the given locus.

        Includes the probe, drive and flux channels of the locus QPU components.
        Does not include e.g. any neighboring coupler channels, these will have to be added
        separately in the TimeBox resolution phase.

        Will only return channels that are known to exist, i.e. are found in :attr:`ScheduleBuilder.channels`.

        Args:
            locus: locus on which the operation acts

        Returns:
            names of the control channels that directly affect the operation

        """
        controllers: list[str] = []
        for q in locus:
            if self.component_channels is not None and q in self.component_channels:
                controllers.extend(self.component_channels[q].values())
        return tuple(c for c in controllers if c in self.channels and not self.channels[c].is_virtual)

    def wait(self, locus: Iterable[str], duration: float, *, rounding: bool = False) -> TimeBox:
        """Utility method for applying Block instructions on every channel of the given locus.

        The Block instructions guarantee the locus components to idle for the given duration,
        and cannot e.g. be replaced with e.g. dynamical decoupling sequences.
        They are treated the same as any other TimeBox contents:

        1. Blocks on different channels remain aligned in time during scheduling.
        2. The actual waiting time on a particular channel may thus be >= ``duration``,
           if the other channels have less non-blocking space on either side.

        .. note::
           TODO For now, this method can round ``duration`` to the nearest value allowed by each
           channel if requested. This is for the benefit of EXA sweeping over waiting durations.
           In the future, EXA sweep generation should be responsible for doing the rounding.

        Args:
            locus: locus components that should experience the wait
            duration: how long to wait (in seconds)
            rounding: Iff True, for each channel separately, ``duration`` will be rounded to the
                nearest value allowed by the granularity of that channel. The Waits will start
                simultaneously.

        Returns:
            box containing :class:`.Block` instructions on every control channel of ``locus``

        """
        channels = self.get_control_channels(locus)
        # Zero-duration wait is equivalent to doing nothing on the channels, however
        # it will act as a barrier during scheduling.
        if duration == 0:
            segments = {ch: [Block(0)] for ch in channels}
        else:
            segments = {}
            for channel in channels:
                if rounding:
                    rounded = self.channels[channel].round_duration_to_granularity(duration)
                    int_duration = self.channels[channel].duration_to_int_samples(rounded) if duration > 0 else 0
                else:
                    int_duration = self.channels[channel].duration_to_int_samples(duration)
                segments[channel] = [Block(int_duration)]

        return TimeBox.atomic(
            Schedule(segments),
            locus_components=locus,
            label="Wait",
        )

    def get_implementation(
        self,
        op_name: str,
        locus: Iterable[str],
        impl_name: str | None = None,
        *,
        use_priority_order: bool = False,
        strict_locus: bool = False,
        priority_calibration: OILCalibrationData | OICalibrationData | None = None,
    ) -> GateImplementation:
        """Provide an implementation for a quantum operation at a given locus.

        Args:
            op_name: name of the quantum operation
            locus: locus of the operation
            impl_name: name of the implementation (``None`` means the implementation is chosen automatically
                using the logic described below)
            strict_locus: iff False, for non-symmetric implementations of symmetric ops the locus order may
                be changed if no calibration data is available for the requested locus order
            use_priority_order: Only has an effect if ``impl_name`` is ``None``. Iff ``False``,
                :meth:`QuantumOp.get_default_implementation_for_locus` is used. Otherwise, the first implementation in
                the priority order that has calibration data for ``locus`` is chosen. The priority order is as follows:
                1. The locus-specific priority defined in ``QuantumOp.defaults_for_locus[locus]`` if any.
                2. The global priority order defined in :attr:`QuantumOp.implementations`.
            priority_calibration: Calibration data from which to load the calibration instead of the common calibration
                data in :attr:`calibration`. Any non-None values found in ``priority_calibration``
                will be merged to the common calibration.
                For factorizable QuantumOps this is a mapping from single-qubit loci to their calibration data,
                otherwise just the calibration data for a single locus.
                Note: using ``priority_calibration`` will prevent caching.

        Returns:
            requested implementation

        Raises:
            ValueError: requested implementation could not be provided

        """
        if (op := self.op_table.get(op_name)) is None:
            raise ValueError(f"Unknown quantum operation '{op_name}'.")

        if impl_name is None and not use_priority_order:
            impl_name = op.get_default_implementation_for_locus(locus)
        return self._get_implementation(
            op, impl_name, tuple(locus), strict_locus, priority_calibration=priority_calibration
        )

    def _find_implementation_and_locus(
        self,
        op: QuantumOp,
        impl_name: str | None,
        locus: Locus,
        *,
        strict_locus: bool = False,
    ) -> tuple[str, Locus]:
        """Find an implementation and locus for the given quantum operation instance compatible
        with both the calibration data and the implementation and locus requested by the caller.

        Args:
            op: quantum operation
            impl_name: Name of the requested implementation. ``None`` means use the highest-priority
                implementation for which we have calibration data.
            locus: requested locus of the operation
            strict_locus: Iff False, for non-symmetric implementations of symmetric ops the locus order may
                be changed to an equivalent one if no calibration data is available for the requested locus order.

        Returns:
            chosen implementation name, locus

        Raises:
            ValueError: requested implementation could not be found
            ValueError: requested implementation had no calibration data for this locus
            ValueError: no specific implementation was requested, but no known implementation had
                calibration data for this locus

        """

        def find_locus(
            op: QuantumOp,
            impl_name: str,
            impl_class: type[GateImplementation],
            given_locus: Locus,
            strict_locus: bool = False,
        ) -> Locus | None:
            """Tries to find a valid locus for the given operation and implementation, that has
            calibration data available. If none can be found, returns None.
            """
            if not impl_class.needs_calibration():
                # any locus is ok
                # FIXME This is wrong for compositegates, see SW-1016
                return given_locus
            if op.factorizable and len(given_locus) > 1:
                # check delegated to subimplementations
                return given_locus
            # find out which loci we need to check for cal data
            if op.symmetric:
                # all locus orders are equivalent for perfectly calibrated symmetric ops, locus can be permuted
                if impl_class.symmetric:
                    # Cal data for symmetric implementations uses always a sorted locus order.
                    loci = [tuple(sort_components(given_locus))]
                elif strict_locus:
                    # If the operation is symmetric but implementation is not (e.g. fast flux CZ)
                    # the locus order can be meaningful in practice.
                    # Users must be able to request implementations for any order of the locus, which may have
                    # independent cal data.
                    loci = [given_locus]
                else:
                    # User did not request a strict locus order, pick the first one that has cal data.
                    loci = list(itertools.permutations(given_locus))
            else:
                # For non-symmetric ops the locus is always strict.
                loci = [given_locus]

            for new_locus in loci:
                # use the first locus under which we can find cal data
                if self.has_calibration(op.name, impl_name, new_locus):
                    return new_locus
            return None

        def merge_impl_priorities_for_locus(op: QuantumOp, locus: Locus) -> dict[str, type[GateImplementation]]:
            """Merge locus-specific and global implementation priorities for a given locus. The implementation
            defined for the locus will take precedence over the global ones.
            """
            if not op.defaults_for_locus:
                return op.implementations
            locus_priority = op.get_default_implementation_for_locus(locus)
            priorities: dict[str, type[GateImplementation]] = {locus_priority: op.implementations[locus_priority]}
            for global_priority, priority_class in op.implementations.items():
                if global_priority not in priorities:
                    priorities[global_priority] = priority_class
            return priorities

        if impl_name is None:
            # no specific implementation requested, choose the highest-priority implementation
            # for which there is cal data and take into account locus-specific implementation order
            priorities = merge_impl_priorities_for_locus(op, locus)
            for new_impl_name, impl in priorities.items():
                if (
                    not impl.special_implementation
                    and (new_locus := find_locus(op, new_impl_name, impl, locus, strict_locus)) is not None
                ):
                    # valid implementation and locus found
                    return new_impl_name, new_locus
            # no valid implementation found
            raise ValueError(f"No calibration data for '{op.name}' at {locus}.")

        # use the requested implementation if it exists
        if (requested_impl := op.implementations.get(impl_name)) is None:
            raise ValueError(f"Unknown quantum operation implementation '{op.name}.{impl_name}'.")
        if (cal_locus := find_locus(op, impl_name, requested_impl, locus, strict_locus)) is None:
            raise ValueError(f"No calibration data for '{op.name}.{impl_name}' at {locus}.")
        return impl_name, cal_locus

    def _get_implementation(
        self,
        op: QuantumOp,
        impl_name: str | None,
        locus: Locus,
        strict_locus: bool = False,
        *,
        priority_calibration: OILCalibrationData | OICalibrationData | None = None,
    ) -> GateImplementation:
        """Build a factory class for the given quantum operation, implementation and locus.

        The GateImplementations are built when they are first requested, and cached for later use.

        The attributes :attr:`QuantumOp.factorizable`, :attr:`GateImplementation.needs_calibration` and whether
        the implementation is a :class:`CompositeGate` interact in a nontrivial way, described in the table below.

        .. list-table::
           :header-rows: 1
           :stub-columns: 1

           * - composite / not composite
             - factorizable
             - not factorizable
           * - needs_calibration
             - not in use yet / ``measure.constant``
             - ``cc_prx.prx_composite`` / ``prx.drag_crf``
           * - not needs_calibration
             - ``reset.conditional`` / not meaningful
             - ``rz.prx_composite`` / ``rz.virtual``

        Args:
            op: quantum operation
            impl_name: Name of the implementation. ``None`` means use the highest-priority implementation for
                which we have calibration data.
            locus: locus of the operation
            strict_locus: iff False, for non-symmetric implementations of symmetric ops the locus order may
                be changed if no calibration data is available for the requested locus order
            priority_calibration: Calibration data node from which to load the calibration instead of the common
                calibration data. Only overrides the given parameters. For this to work, ``impl_name`` should be given,
                since ``priority_calibration`` is implementation-specific.
                For factorizable QuantumOps this is a mapping from single-qubit loci to their calibration data,
                otherwise just the calibration data for a single locus.

        Returns:
            requested implementation

        Raises:
            ValueError: requested implementation could not be provided or had no calibration data for this locus

        """
        new_impl_name, new_locus = self._find_implementation_and_locus(op, impl_name, locus, strict_locus=strict_locus)
        # use caching if no priority calibration is used
        if not priority_calibration:
            # use a cached factory if it exists
            op_cache = self._cache.setdefault(op.name, {})
            impl_cache = op_cache.setdefault(new_impl_name, {})
            if factory := impl_cache.get(new_locus):
                return factory

        # find the calibration data
        impl_class = op.implementations[new_impl_name]
        if op.factorizable and len(new_locus) > 1 and impl_class.may_have_calibration():
            # E.g. measure.constant (needs_calibration), reset.conditional (composite, not needs_calibration)
            # Currently there are no factorizable gates that are both composite and need calibration.
            # For factorizable QuantumOps all the calibration data is for single-component loci,
            # so priority_calibration is of the type OICalibrationData.
            # Build (and possibly cache) the required single-component implementations, then
            # use them to construct the full-locus implementation.
            priority_calibration = priority_calibration or {}
            factory = impl_class.construct_factorizable(
                parent=op,
                name=new_impl_name,
                locus=new_locus,
                sub_implementations={
                    c: self._get_implementation(
                        op,
                        new_impl_name,
                        (c,),
                        priority_calibration={(c,): c_cal} if (c_cal := priority_calibration.get((c,))) else None,  # type: ignore
                    )
                    for c in new_locus
                },
                builder=self,
            )
        else:
            if impl_class.may_have_calibration():
                # Either needs_calibration or is CompositeGate.
                # Find the calibration data, which is all found under new_locus.
                if impl_class.needs_calibration():
                    cal_data = self.get_calibration(op.name, new_impl_name, new_locus)
                else:
                    # cal data optional
                    try:
                        cal_data = self.get_calibration(op.name, new_impl_name, new_locus)
                    except ValueError:
                        cal_data = {}

                if priority_calibration:
                    if op.factorizable:
                        # pick out the single-component locus
                        priority_calibration = priority_calibration[new_locus]  # type: ignore[index]
                    cal_data = merge_dicts(cal_data, priority_calibration, merge_nones=False)
                validate_locus_calibration(cal_data, impl_class, op, new_impl_name, new_locus)
            else:
                # no cal data needed, e.g. rz.virtual
                cal_data = {}

            # construct the factory
            factory = impl_class(
                parent=op,
                name=new_impl_name,
                locus=new_locus,
                calibration_data=cal_data,
                builder=self,
            )

        # cache the factory if no priority_calibration was used
        if not priority_calibration:
            impl_cache[new_locus] = factory
        return factory

    def get_implementation_class(self, op_name: str, impl_name: str | None = None) -> type[GateImplementation]:
        """Implementation class for the given operation.

        Args:
            op_name: name of the quantum operation
            impl_name: name of the implementation (``None`` means use the default implementation)

        Returns:
            requested implementation class

        """
        op = self.op_table[op_name]
        if impl_name is None:
            if not op.implementations:
                raise ValueError(f"No implementations found for operation {op_name}.")
            return next(iter(op.implementations.values()))
        return op.implementations[impl_name]

    def validate_quantum_circuit(
        self,
        operations: Iterable[CircuitOperation],
        *,
        require_measurements: bool = False,
    ) -> None:
        """Validate a sequence of circuit operations constituting a quantum circuit.

        Args:
            operations: quantum circuit to be validated
            require_measurements: iff True the circuit must include at least one measurement operation

        Raises:
            ValueError: ``operations`` do not constitute a valid quantum circuit

        """
        validate_quantum_circuit(operations, self.op_table, require_measurements=require_measurements)

    def circuit_to_timebox(
        self,
        circuit: Iterable[CircuitOperation],
        *,
        name: str = "",
        scheduling_algorithm: SchedulingAlgorithm = SchedulingAlgorithm.HARD_BOUNDARY,
        locus_mapping: dict[str, str] | None = None,
    ) -> TimeBox:
        """Convert a quantum circuit to a TimeBox.

        Args:
            circuit: quantum circuit
            name: name of the circuit
            scheduling_algorithm: scheduling algorithm to be used in resolving the TimeBoxes.
            locus_mapping: optional mapping of placeholder component names to the physical component names used
                while resolving the circuit into a TimeBox.

        Returns:
            unresolved TimeBox that implements ``circuit``

        Raises:
            ValueError: failed to convert ``circuit`` to a TimeBox

        """
        # TODO for now do not force validation here
        # self.validate_quantum_circuit(circuit, require_measurements=True)

        boxes = []
        locus_mapping = locus_mapping or {}
        for op in circuit:
            self._logger.debug("Adding %s", op)

            # we have already validated the operations, so we know they can be found in self.op_table
            op_type = self.op_table[op.name]
            # append the operation to the pulse schedule
            # we operate in non-strict mode, i.e. for symmetric gates the caller does not
            # need to know the locus order, any will do
            mapped_locus = tuple(locus_mapping.get(qubit, qubit) for qubit in op.locus) if locus_mapping else op.locus
            factory = self._get_implementation(op_type, op.implementation, mapped_locus)
            boxes.append(factory(**op.args))
        return TimeBox.composite(boxes, label=name, scheduling_algorithm=scheduling_algorithm)

    def timeboxes_to_front_padded_playlist(
        self, boxes: Iterable[TimeBox], *, neighborhood: int = 0
    ) -> tuple[Playlist, ReadoutMetrics]:
        """Temporary helper function, for converting a sequence of TimeBoxes to a Playlist.

        Each individual TimeBox in ``boxes`` is resolved into a Schedule, and then
        each schedules is front-padded with :class:`.Wait` instructions on each channel
        such that the resulting Schedules have equal durations. This is required since
        for now in Station Control the delay before the final measurement is the same for
        all the Schedules in a Playlist, and we do not wish to lose coherence waiting for
        the measurement after each Schedule is done.

        TODO Once Station Control can handle measurements better, this method should be removed,
        and :meth:`timeboxes_to_playlist` be used instead.

        Args:
            boxes: TimeBoxes to include in the playlist
            neighborhood: During scheduling, block neighboring channels of the used components this far. By default,
                blocks only the defined locus components and any other components which have occupied channels.

        Returns:
            playlist that implements ``boxes`` and the readout metrics for that playlist.

        """
        schedules = [self.resolve_timebox(box, neighborhood=neighborhood).cleanup() for box in boxes]
        max_duration_in_seconds = max(
            (
                self.channels[ch].duration_to_seconds(seg.duration)
                for schedule in schedules
                for ch, seg in schedule.items()
            ),
            default=0.0,
        )
        T = max((schedule.duration for schedule in schedules), default=0)
        # We can assume every segment has a probe pulse (readout), but let's check the first one just in case
        if self._require_scheduling_in_seconds and schedules[0].has_content_in(self._channel_types["probe"]):
            schedules = [
                schedule.front_pad_in_seconds(max_duration_in_seconds, self.channels) for schedule in schedules
            ]
        else:
            schedules = [schedule.front_pad(T) for schedule in schedules]
        return self.build_playlist(schedules)

    def timeboxes_to_playlist(
        self,
        boxes: Iterable[TimeBox],
        *,
        neighborhood: int = 1,
    ) -> Playlist:
        """Convert a sequence of TimeBoxes to a Playlist.

        Resolves the boxes, converts them to Schedules, removes unnecessary channels, and then packs
        the Schedules into a Playlist. Assumes all the TimeBoxes refer to the same QPU and its control channels.

        Args:
            boxes: TimeBoxes to include in the playlist
            neighborhood: During scheduling, block neighboring channels of the used components this far.
                The default value ensures that quantum operations work as intended, assuming the station
                is properly calibrated. Higher values may help defend against crosstalk, at the expense
                of a longer instruction schedule and thus more decoherence.

        Returns:
            playlist that implements ``boxes``

        """
        return self.build_playlist([self.timebox_to_schedule(box, neighborhood=neighborhood) for box in boxes])[0]

    def timebox_to_schedule(
        self,
        box: TimeBox,
        *,
        neighborhood: int = 1,
    ) -> Schedule:
        """Convert a TimeBox to a finished instruction schedule, ready for execution.

        Resolves the box, then converts the durations of the instructions in the schedule to samples
        at the channel sample_rate.

        Args:
            box: TimeBox to resolve
            neighborhood: During scheduling, block neighboring channels of the used components this far.
                The default value ensures that quantum operations work as intended, assuming the station
                is properly calibrated. Higher values may help defend against crosstalk, at the expense
                of a longer instruction schedule and thus more decoherence.

        Returns:
            finished schedule that implements ``box``

        """
        schedule = self.resolve_timebox(box, neighborhood=neighborhood)
        return self._finish_schedule(schedule)

    def _finish_schedule(self, schedule: Schedule) -> Schedule:
        """Finishes the instruction schedule.

        * removes channels that only have Waits in them
        * fuses long-distance Rz corrections to the correct drive pulses

        Args:
            schedule: schedule to finish

        Returns:
            finished copy of ``schedule``

        """
        has_long_distance_rzs = False
        for channel in schedule.channels():
            if "flux" in channel:
                for inst in schedule[channel]:
                    if isinstance(inst, FluxPulse) and inst.rzs:
                        has_long_distance_rzs = True
                        break
            if has_long_distance_rzs:
                return self._fuse_long_distance_virtual_rzs(schedule)
        return schedule.cleanup()

    def _fuse_long_distance_virtual_rzs(self, schedule: Schedule) -> Schedule:
        """Fuse long-distance (i.e. out-of-gate-locus) VirtualRZ corrections with the next drive pulse
        happening after the FluxPulse they are correcting.

        """

        def fuse(inst: Instruction, idx: int, rz_angle: float, segment: Segment) -> None:
            """Fuse a VirtualRZ angle to an IQPulse."""
            if isinstance(inst, (VirtualRZ, IQPulse)):
                segment._instructions[idx] = replace(inst, phase_increment=inst.phase_increment + rz_angle)
            else:
                raise ValueError(f"Unknown drive channel instruction {inst}")

        schedule_copy = schedule.copy()
        for channel in [ch for ch in schedule.channels() if "flux" in ch]:
            sample_counter = 0
            for inst in schedule[channel]:
                if isinstance(inst, FluxPulse):
                    rzs = {ch: angle for ch, angle in inst.rzs if ch in schedule}
                    for drive_channel, rz_angle in rzs.items():
                        drive_sample_counter = 0
                        for drive_inst_idx, drive_inst in enumerate(schedule[drive_channel]):
                            if (
                                not isinstance(drive_inst, Wait)
                                and not isinstance(drive_inst, NONSOLID)
                                and drive_sample_counter >= sample_counter
                            ):
                                fuse(
                                    schedule_copy[drive_channel][drive_inst_idx],
                                    drive_inst_idx,
                                    -rz_angle,  # invert the angle as per the agreed convention
                                    schedule_copy[drive_channel],
                                )
                                break
                            drive_sample_counter += drive_inst.duration
                sample_counter += inst.duration
        return schedule_copy.cleanup()

    def resolve_timebox(
        self, box: TimeBox, *, neighborhood: int, compute_neighborhood_hard_boundary: bool = False
    ) -> Schedule:
        """Resolve a TimeBox.

        Resolves recursively each of the children of the box, and then concatenates the resulting
        Schedules into a new one using a specific scheduling strategy and algorithm.

        The supported algorithms are ``HARD_BOUNDARY``, which treats each composite TimeBox as a solid rectangle (the
        longest channel within defines the duration) and ``TETRIS``, which packs the schedule as tightly as possible
        (solid instructions still cannot overlap) regardless of the TimeBox boundaries.

        Modifies ``box`` so that it becomes atomic, if it isn't already.

        Args:
            box: TimeBox to resolve
            neighborhood: During scheduling, block control channels of neighboring QPU components this far
                from the locus. Values higher than 0 may help defend against crosstalk, at the expense
                of a longer instruction schedule and thus more decoherence.
            compute_neighborhood_hard_boundary: Whether to precompute the neighborhood components while resolving
                a composite ``TimeBox`` in the ``HARD_BOUNDARY`` algorithm. Typically one does not want to do this on
                the top layer composite ``TimeBox``, since it would be unused. The algorithm sets this ``True`` on
                lower layers, where it improves the performance as the neighborhood components are needed in scheduling.

        Returns:
            instruction schedule that implements ``box``

        """
        if box.scheduling_algorithm == SchedulingAlgorithm.HARD_BOUNDARY:
            return self._resolve_timebox_hard_boundary(
                box, neighborhood=neighborhood, compute_neighborhood=compute_neighborhood_hard_boundary
            )
        return self._resolve_timebox_tetris(box, neighborhood=neighborhood)

    def _resolve_timebox_hard_boundary(
        self, box: TimeBox, neighborhood: int, compute_neighborhood: bool = False
    ) -> Schedule:
        """Resolves a TimeBox using the ``HARD_BOUNDARY`` algorithm, which treats each composite TimeBox as a solid
        rectangle (the longest channel within defines the duration).
        """
        if box.atom is not None:
            # already resolved
            return box.atom

        self._logger.debug("\nResolving '%s':", box.label)
        schedule = Schedule()
        if compute_neighborhood:
            for dist in range(neighborhood + 1):
                box.neighborhood_components.setdefault(dist, set())

        # If variable sampling rates are detected, we need to start scheduling in seconds.
        # Until then, the scheduling uses samples, and all time durations are integers.
        component_durations: dict[str, int] = {}
        component_durations_seconds: dict[str, float] = {}
        scheduling_in_seconds = False

        child_order = reversed(box.children) if box.scheduling == SchedulingStrategy.ALAP else box.children
        for child in child_order:
            child_schedule = self.resolve_timebox(
                child, neighborhood=neighborhood, compute_neighborhood_hard_boundary=True
            )
            if (
                self._require_scheduling_in_seconds
                and not scheduling_in_seconds
                and child_schedule.has_content_in(self._channel_types["probe"])
            ):
                # convert the locus durations to seconds (no probe channels have been used so far)
                scheduling_in_seconds = True
                if self._channel_templates["other"] is not None:
                    component_durations_seconds = {
                        component: self._channel_templates["other"].duration_to_seconds(duration)
                        for component, duration in component_durations.items()
                    }

            child_components = self._get_neighborhood_hard_boundary(child, 0)
            neighborhood_components = self._get_neighborhood_hard_boundary(child, neighborhood)
            if scheduling_in_seconds:
                extend_hard_boundary_in_seconds(
                    schedule,
                    child_schedule,
                    child_components,
                    neighborhood_components,
                    component_durations_seconds,
                    box.scheduling == SchedulingStrategy.ALAP,
                    self.channels,
                )
            else:
                extend_hard_boundary(
                    schedule,
                    child_schedule,
                    child_components,
                    neighborhood_components,
                    component_durations,
                    box.scheduling == SchedulingStrategy.ALAP,
                )
            if compute_neighborhood:
                box.neighborhood_components[0].update(child_components)
                if neighborhood > 0:
                    box.neighborhood_components[neighborhood].update(neighborhood_components)
            if self._logger.isEnabledFor(logging.DEBUG):
                self._logger.debug("Updated schedule:\n%s", schedule.pprint())

        if scheduling_in_seconds:
            schedule.pad_to_hard_box_in_seconds(self.channels)
        else:
            schedule.pad_to_hard_box()

        if box.scheduling == SchedulingStrategy.ALAP:
            schedule = schedule.reverse_hard_box()

        self._logger.debug("'%s' resolved.", box.label)
        box.atom = schedule
        return schedule

    def _get_neighborhood_hard_boundary(self, box: TimeBox, neighborhood: int) -> set[str]:
        """Computes and caches the blocking neighborhoods for HARD_BOUNDARY algorithm.

        Args:
            box: Atomic TimeBox whose neighborhood to compute.
            neighborhood: Return QPU components this far from the locus.

        Returns:
            QPU components (plus maybe channels?) belonging the the given neighborhood of ``box``.

        """
        # check the cache
        if neighborhood in box.neighborhood_components:
            return box.neighborhood_components[neighborhood]

        # Box may have other channels beside the ones that belong to its locus (e.g. coupler flux for CZ gates).
        # We need to find the associated components in order to block them.
        # Cache the found components in case they are not yet cached.
        # In case the channel is not blocking (i.e. not found in ``channel_to_component``), we just
        # block the channel itself
        components = box.neighborhood_components.get(0)
        if components is None:
            components = box.locus_components.copy()
            for channel in box.atom:  # type: ignore[union-attr]
                components.add(self._channel_to_component.get(channel, channel))
            box.neighborhood_components[0] = components

        # iteratively find and cache the higher neighborhoods
        for nh_index in range(1, neighborhood + 1):
            prev = components
            components = box.neighborhood_components.get(nh_index)
            if components is None:
                if nh_index % 2 == 1:
                    components = prev | self.chip_topology.get_neighbor_couplers(prev)
                else:
                    components = prev | self.chip_topology.get_neighbor_locus_components(prev)
                box.neighborhood_components[nh_index] = components

        return components

    def _resolve_timebox_tetris(
        self,
        box: TimeBox,
        *,
        neighborhood: int,
    ) -> Schedule:
        """Resolves a TimeBox using the ``TETRIS`` algorithm, which packs the schedule as tightly as possible
        (solid instructions still cannot overlap) regardless of the TimeBox boundaries.
        """
        raise NotImplementedError(
            "SchedulingAlgorithm.TETRIS is not supported in this iqm-pulse version. "
            "Updating it to work with integer-duration scheduling will be done in COMP-1281."
        )

        if box.atom is not None:
            # already resolved
            return box.atom

        self._logger.debug("\nResolving '%s':", box.label)
        schedule = Schedule()

        # ALAP scheduling strategy works by reversing both the contents and order of the children before extending,
        # and then reversing the result schedule in the end.
        child_order = reversed(box.children) if box.scheduling == SchedulingStrategy.ALAP else box.children
        for child in child_order:
            child_schedule = self.resolve_timebox(child, neighborhood=neighborhood)
            self._block_neighborhood_tetris(child_schedule, child.locus_components, neighborhood)

            if self._logger.isEnabledFor(logging.DEBUG):  # evaluate pprint only if needed
                self._logger.debug("Adding %s\n%s", child.label, child_schedule.pprint())
            if box.scheduling == SchedulingStrategy.ALAP:
                extend_schedule_new(schedule, child_schedule.reverse(), self.channels)
            else:
                extend_schedule_new(schedule, child_schedule, self.channels)

            if self._logger.isEnabledFor(logging.DEBUG):
                self._logger.debug("Updated schedule:\n%s", schedule.pprint())

        if box.scheduling == SchedulingStrategy.ALAP:
            schedule = schedule.reverse()
        self._logger.debug("'%s' resolved.", box.label)

        box.atom = schedule
        return box.atom

    def _block_neighborhood_tetris(self, schedule: Schedule, locus: set[str], neighborhood: int) -> None:
        """Add additional blocked channels to the schedule, preventing their use during the schedule.

        In the idealized computational model we assume that in a (correctly calibrated) quantum computer
        there is no effective interaction between QPU components (in the computational frame and subspace) while
        a :class:`.Wait` instruction is acting on the flux channel of the coupler connecting those components
        (i.e., the coupler is idling).
        Hence a QPU component experiences no effective evolution if Wait instructions are
        acting on its drive, flux and probe channels, and the flux channels of all its couplers.

        Of course, in reality the QPU will experience at least some decoherence during a Wait, and
        possibly some crosstalk.
        In some applications, e.g. decoherence experiments, it is essential to Wait for a specific time,
        since it's precisely the decoherence that we are interested in.

        However, if we are only interested in applying well-defined local quantum operations on specific
        loci, it is essential to shut down all unwanted dynamics by adding :class:`.Block` instructions on
        control channels of the unused and neighboring channels.
        They act like Waits (and are converted into Waits at the end of the scheduling), but are allowed
        to overlap in time, since we are only interested in blocking those channels for the duration of the
        quantum operation.

        Args:
            schedule: instruction schedule to modify
            locus: information-carrying QPU components ``schedule`` is meant to operate on (does not include couplers)
            neighborhood: How far should we block neighboring QPU components?
                Zero means just the locus qubits, one means neighboring couplers, two means *their* neighboring
                qubits etc.

        Returns:
            ``schedule``, with added :class:`.Block` instructions on all the neighbor channels,
            for the duration of the schedule

        """
        channels: list[str] = []
        if neighborhood >= 0:
            # locus channels
            channels.extend(self.get_control_channels(locus))
        if neighborhood >= 1:
            # channels of couplers neighboring the locus
            couplers = self.chip_topology.get_neighbor_couplers(locus)
            channels.extend(
                self.get_flux_channel(coupler)
                for coupler in couplers
                if coupler in self.component_channels and "flux" in self.component_channels[coupler]
            )
        if neighborhood >= 2:
            # TODO the scheduling could be improved here
            # channels of qubits neigboring the locus
            neighbors = self.chip_topology.get_neighbor_locus_components(locus)
            channels.extend(self.get_control_channels(neighbors))
        # TODO systematically allow even larger neighborhoods?

        # add the blocks
        # TODO VirtualRZ on a drive channel is no reason to block coupler fluxes!
        schedule.add_channels(channels)
        T = schedule.duration
        for ch in channels:
            # if ch already exist in the schedule, nothing happens to it
            if not schedule[ch]:
                schedule.append(ch, Block(T))

    def build_playlist(  # noqa: PLR0915
        self, schedules: Sequence[Schedule], finish_schedules: bool = True
    ) -> tuple[Playlist, ReadoutMetrics]:
        """Build a playlist from a number of instruction schedules.

        This involves compressing the schedules so that no duplicate information
        needs to be transferred to Station Control.

        All virtual channels are dropped at this point.

        Args:
            schedules: finished instruction schedules to include in the playlist
            finish_schedules: whether to finalise the schedules before building the playlist. Should be set ``True``
                unless some process has already finalised them before calling this function.

        Returns:
            playlist containing the schedules and the readout metrics for this playlist.

        Raises:
            ValueError: if the schedules contain channels with non-uniform sampling rates

        """

        # build the channel descriptions
        def _map_channel_config(props: ChannelProperties) -> IQChannelConfig | RealChannelConfig | ReadoutChannelConfig:
            if isinstance(props, ProbeChannelProperties):
                return ReadoutChannelConfig(props.sample_rate)
            if props.is_iq:
                return IQChannelConfig(props.sample_rate)
            return RealChannelConfig(props.sample_rate)

        channel_descriptions = {
            channel_name: ChannelDescription(controller_name=channel_name, channel_config=_map_channel_config(props))
            for channel_name, props in self.channels.items()
            if not props.is_virtual
        }

        pl = Playlist()
        mapped_instructions: dict[str, dict[int | Instruction, Any]] = {}
        readout_metrics = ReadoutMetrics(num_segments=len(schedules))

        def _append_to_schedule(sc_schedule: SC_Schedule, channel_name: str, instr: Instruction) -> None:
            """Append ``instr`` to ``sc_schedule`` into the channel``channel_name``."""
            try:
                # Check if instr can be used as a dictionary key, and use it if possible.
                # 2 dataclasses can have the same hash if their fields are identical. We must
                # distinguish between different Waveform classes which may have identical fields,
                # so we use the instruction itself as a key, so that the class is checked too.
                instr_id = instr  # type: ignore[attr-defined]
                is_mapped = instr_id in mapped_instructions.setdefault(channel_name, {})
            except TypeError:
                instr_id = instr.id  # type: ignore[attr-defined]
                is_mapped = instr_id in mapped_instructions.setdefault(channel_name, {})
            if not is_mapped:
                mapped = _map_instruction(instr)
                idx = pl.channel_descriptions[channel_name].add_instruction(mapped)
                sc_schedule.instructions.setdefault(channel_name, []).append(idx)
                mapped_instructions[channel_name][instr_id] = idx
            else:
                sc_schedule.instructions.setdefault(channel_name, []).append(
                    mapped_instructions[channel_name][instr_id]
                )

                # add the schedules in the playlist

        def _map_instruction(inst: Instruction) -> sc_instructions.Instruction:
            """TODO only necessary until SC has been updated to use the iqm.pulse Instruction class."""
            operation: Any

            def _map_acquisition(acq: AcquisitionMethod) -> sc_instructions.AcquisitionMethod:
                if isinstance(acq, TimeTrace):
                    return sc_instructions.TimeTrace(
                        label=acq.label, delay_samples=acq.delay_samples, duration_samples=acq.duration_samples
                    )
                if isinstance(acq, ThresholdStateDiscrimination):
                    return sc_instructions.ThresholdStateDiscrimination(
                        label=acq.label,
                        delay_samples=acq.delay_samples,
                        weights=_map_instruction(acq.weights).operation,
                        threshold=acq.threshold,
                        feedback_signal_label=acq.feedback_signal_label,
                    )
                if isinstance(acq, ComplexIntegration):
                    return sc_instructions.ComplexIntegration(
                        label=acq.label,
                        delay_samples=acq.delay_samples,
                        weights=_map_instruction(acq.weights).operation,
                    )

                raise ValueError(f"Unknown AcquisitionMethod {acq}")

            if isinstance(inst, Wait):
                operation = sc_instructions.Wait()
            elif isinstance(inst, VirtualRZ):
                operation = sc_instructions.VirtualRZ(inst.phase_increment)
            elif isinstance(inst, RealPulse):
                operation = sc_instructions.RealPulse(
                    wave=to_canonical(inst.wave),
                    scale=inst.scale,
                )
            elif isinstance(inst, IQPulse):
                operation = sc_instructions.IQPulse(
                    wave_i=to_canonical(inst.wave_i),
                    wave_q=to_canonical(inst.wave_q),
                    scale_i=inst.scale_i,
                    scale_q=inst.scale_q,
                    phase=inst.phase,
                    modulation_frequency=inst.modulation_frequency,
                    phase_increment=inst.phase_increment,
                )
            elif isinstance(inst, ConditionalInstruction):
                if len(inst.outcomes) != 2:
                    raise ValueError("ConditionalInstruction requires exactly two outcomes.")
                operation = sc_instructions.ConditionalInstruction(
                    condition=inst.condition,
                    if_true=_map_instruction(inst.outcomes[1]),
                    if_false=_map_instruction(inst.outcomes[0]),
                )
            elif isinstance(inst, MultiplexedIQPulse):
                sc_entries = tuple((_map_instruction(p), d) for p, d in inst.entries)
                operation = sc_instructions.MultiplexedIQPulse(sc_entries)
            elif isinstance(inst, ReadoutTrigger):
                sc_acquisitions = tuple(_map_acquisition(a) for a in inst.acquisitions)
                operation = sc_instructions.ReadoutTrigger(
                    probe_pulse=_map_instruction(inst.probe_pulse),
                    acquisitions=sc_acquisitions,
                )
            else:
                raise ValueError(f"{inst} not supported.")
            return sc_instructions.Instruction(duration_samples=int(inst.duration), operation=operation)

        # NOTE that there is no implicit right-alignment or equal duration for schedules, unlike in old-style playlists!
        for seg_idx, schedule in enumerate(schedules):
            finished_schedule = self._finish_schedule(schedule) if finish_schedules else schedule
            sc_schedule = SC_Schedule()
            for channel_name, segment in finished_schedule.items():
                if (channel := channel_descriptions.get(channel_name)) is None:
                    continue
                pl.add_channel(channel)
                prev_wait = None
                for instruction in segment:
                    if isinstance(instruction, ReadoutTrigger):
                        readout_metrics.extend(instruction, seg_idx)
                    # convert all NONSOLID instructions into Waits
                    if finish_schedules and (isinstance(instruction, NONSOLID) or isinstance(instruction, Wait)):
                        if instruction.duration > 0:
                            if prev_wait:  # if the previous instruction was a Wait, combine durations
                                prev_wait = Wait(prev_wait.duration + instruction.duration)
                            else:
                                prev_wait = Wait(instruction.duration)
                    else:
                        if prev_wait:  # if there's a prev_wait not yet added to schedule, place it before instruction
                            instructions_to_add = [prev_wait, instruction]
                            prev_wait = None
                        else:
                            instructions_to_add = [instruction]
                        for instr in instructions_to_add:
                            _append_to_schedule(sc_schedule, channel_name, instr)
                if prev_wait:
                    _append_to_schedule(sc_schedule, channel_name, prev_wait)
            pl.segments.append(sc_schedule)
        return pl, readout_metrics

    def _set_gate_implementation_shortcut(self, op_name: str) -> None:
        """Create shortcut for `self.get_implementation(<op_name>, ...)` as `self.<op_name>(...)`.

        If there is a name collision with another attribute in ``self``, the shortcut method won't be added and
        a warning is raised.
        """

        def _shortcut_mthd(
            self,  # noqa: ANN001
            locus: Iterable[str],
            impl_name: str | None = None,
            *,
            use_priority_order: bool = False,
            strict_locus: bool = False,
            priority_calibration: OILCalibrationData | None = None,
        ) -> GateImplementation:
            return self.get_implementation(
                op_name,
                locus,
                impl_name=impl_name,
                use_priority_order=use_priority_order,
                strict_locus=strict_locus,
                priority_calibration=priority_calibration,
            )

        if not hasattr(self, op_name):
            setattr(self, op_name, MethodType(_shortcut_mthd, self))
        else:
            warning_msg = (
                f"Shortcut method ``ScheduleBuilder.{op_name}`` for "
                f'``ScheduleBuilder.get_implementation("{op_name}", ...)`` was not added as there is already'
                f"a class attribute ``ScheduleBuilder.{op_name}``."
            )
            self._logger.warning(warning_msg)
